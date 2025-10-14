import os
import glob
import numpy as np
import joblib
import torch
import random
from torch.utils.data import Dataset
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from uniphys.utils.motion_repr_utils import get_repr, cano_seq_smpl_or_smplx,REPR_DIM_DICT_SMPLX, REPR_DIM_DICT_REAL_SMPL

printed_messages = set()

def print_once(message):
    if message not in printed_messages:
        print(message)
        printed_messages.add(message)

def have_overlap(seg1, seg2):
    if seg1[0] > seg2[1] or seg2[0] > seg1[1]:
        return False
    else:
        return True

from uniphys.utils.clip_utils import load_and_freeze_clip, encode_text
clip_model = load_and_freeze_clip(clip_version='ViT-B/32', device="cuda")

class StateActionDataset(Dataset):
    """
    A PyTorch Dataset for loading and processing state-action sequences.

    This dataset handles loading data from pickle files, processing it into
    standardized representations, and providing it in a format suitable for
    training models. It supports various data representations, normalization,
    and text-based filtering and conditioning.
    """

    def __init__(self, cfg, split):
        """
        Initializes the dataset.
        """
        self.cfg = cfg
        self.split = split
        self.data_path_dir = cfg.data_path_list
        self.text_embedding_dict = None

        # --- Data Loading and Path Handling ---
        # Determine if data_path_dir is a file or a directory
        if os.path.isfile(self.data_path_dir):
            self.data_path_list = [self.data_path_dir]
        elif os.path.isdir(self.data_path_dir):
            # Get all .pkl files, excluding those starting with "text"
            self.data_path_list = glob.glob(os.path.join(self.data_path_dir, "*.pkl"))
            self.data_path_list = [
                p for p in self.data_path_list if "text" not in os.path.basename(p)
            ]
        else:
            raise ValueError(f"{self.data_path_dir} is neither a file nor a directory")

        # --- Configuration and Hyperparameters ---
        self.target_fps = 30
        self.text_tolerance = 0 / self.target_fps  # Time tolerance for text-frame alignment
        self.text_strict = 3 / self.target_fps     # Strictness for text-frame alignment
        self.n_frames = (
            cfg.n_frames
            if split == "training"
            else cfg.n_frames
        )
        self.context_length = cfg.context_length
        self.stride = cfg.stride
        self.load_only_succ = cfg.get("load_only_succ", True)  # Load only successful sequences
        self.filter_keyword = cfg.get("filter_keyword", None)  # Keyword to filter sequences by
        self.repr_name_list = cfg.repr_name_list               # List of state representations to use
        self.loaded_keys = {
            # 'action_all', 'pred_dof_pos_all', 'body_pos_all',
            'action_all', 'body_pos_all',
            'root_state_all', 'dof_state_all', 'z_all'
        }
        self.action_keys = cfg.get("action_keys", ["action"])
        self.pre_divide = True  # Divide long sequences into fixed-length clips
        self.norm_action = cfg.norm_action
        self.only_smpl = cfg.only_smpl  # Use only SMPL body model (without hands)
        self.action_rep = cfg.action_rep
        self.key_actions = cfg.load_key_actions  # Keywords for specific actions to load
        self.skip_text = self.cfg.skip_text # Skip loading text annotation

        # --- Data Storage and Normalization Parameters ---
        self.sequences = []
        self.clip_list = {k: [] for k in self.loaded_keys}
        self.clip_list["text_all"] = []
        self.index = []

        # Normalization means and standard deviations
        self.Mean = None #cfg.mean
        self.Std = None #cfg.std
        self.ActionMean = None #cfg.action_mean
        self.ActionStd = None #cfg.action_std
        self.zMean = None
        self.zStd = None

        # Determine representation dimensions based on the body model
        self.repr_dim_dict = (
            REPR_DIM_DICT_REAL_SMPL if self.only_smpl else REPR_DIM_DICT_SMPLX
        )

        # --- Text Data Loading ---
        if not self.skip_text:
            text_embed_file = os.path.join(os.path.dirname(self.data_path_dir), "text_embedding_dict_clip.pkl")
            if os.path.isfile(text_embed_file):
                self.text_embedding_dict = joblib.load(text_embed_file)
            else:
                raise ValueError(f"Text embedding file {text_embed_file} does not exist.")

        # --- Main Data Loading and Processing ---
        self.load_data(self.load_only_succ, self.filter_keyword)

        # Dictionary to store processed representations
        self.repr_list_dict = {
            repr_name: [] for repr_name in self.repr_name_list
        }
        self.repr_list_dict["action"] = []
        self.repr_list_dict["z"] = []
        # self.repr_list_dict["self_obs"] = []
        if not self.skip_text:
            self.repr_list_dict["text"] = []
            self.repr_list_dict["text_embedding"] = []

        self.create_body_repr()

    def create_body_repr(self):
        """
        Processes raw loaded data clips into standardized representations.
        This includes canonicalization, calculating various body representations,
        and preparing them for the final dataset.
        """
        self.n_samples = len(self.clip_list["action_all"])

        for i in tqdm(range(self.n_samples), desc="Processing data clips"):
            source_data_joints_position = self.clip_list["body_pos_all"][i]
            source_data_dof_state = self.clip_list["dof_state_all"][i]
            source_data_root_state = self.clip_list["root_state_all"][i]

            # Canonicalize the sequence to a standard coordinate frame
            cano_joints_position, cano_dof_state, cano_root_state, _ = cano_seq_smpl_or_smplx(
                source_data_joints_position, source_data_dof_state, source_data_root_state
            )

            # Get the desired representations from the canonicalized data
            repr_dict = get_repr(cano_joints_position, cano_dof_state, cano_root_state)

            # Add other data from the clip list
            # repr_dict["self_obs"] = self.clip_list["self_obs_all"][i][:-1]
            repr_dict["z"] = self.clip_list["z_all"][i][:-1] if self.clip_list["z_all"][i] is not None else None

            # Handle text data if not skipped
            if not self.cfg.skip_text:
                # import ipdb; ipdb.set_trace()
                repr_dict["text"] = self.clip_list["text_all"][i]
                if repr_dict["text"] in self.text_embedding_dict:
                    text_embedding = self.text_embedding_dict[repr_dict["text"]]
                else:
                    text_embedding = encode_text(clip_model, [repr_dict["text"]]).squeeze().cpu().numpy()
                    self.text_embedding_dict[repr_dict["text"]] = text_embedding
                
                repr_dict["text_embedding"] = text_embedding

            # Append the representations to the lists
            for repr_name in self.repr_name_list:
                self.repr_list_dict[repr_name].append(repr_dict[repr_name])

            # Separate handling for 'self_obs' and 'action' which may not be in repr_name_list
            # if "self_obs" not in self.repr_name_list:
            #     self.repr_list_dict["self_obs"].append(repr_dict["self_obs"])

            # Process action representation
            if self.action_rep == "3d":
                action = self.clip_list["action_all"][i][:-1]
            elif self.action_rep == "6d":
                # Convert 3D rotation vectors to 6D continuous representation
                action_to_pd_tar_stats = joblib.load("./action_to_pd_tar_stats.pkl")
                scale = action_to_pd_tar_stats["scale"].cpu().numpy()
                offset = action_to_pd_tar_stats["offset"].cpu().numpy()
                action = self.clip_list["action_all"][i][:-1] * scale + offset
                action = action.reshape(len(action), -1, 3)
                action = R.from_rotvec(action.reshape(-1, 3)).as_matrix().reshape(len(action), -1, 3, 3)
                action = action[..., :-1].reshape(len(action), -1, 6).reshape(len(action), -1)
            else:
                raise NotImplementedError(f"Action representation {self.action_rep} is not supported.")
            self.repr_list_dict["action"].append(action)

            # Append z-values (latent space) and text data
            self.repr_list_dict["z"].append(repr_dict["z"])
            if not self.cfg.skip_text:
                self.repr_list_dict["text"].append(repr_dict["text"])
                self.repr_list_dict["text_embedding"].append(repr_dict["text_embedding"])
                
        # --- Convert lists to numpy arrays for efficient access and normalization ---
        for repr_name in self.repr_name_list:
            self.repr_list_dict[repr_name] = np.asarray(self.repr_list_dict[repr_name])
            print(f"Shape of {repr_name}: {self.repr_list_dict[repr_name].shape}")

        # if "self_obs" not in self.repr_name_list:
        #     self.repr_list_dict["self_obs"] = np.asarray(self.repr_list_dict["self_obs"])
        #     print(f"Shape of self_obs: {self.repr_list_dict['self_obs'].shape}")

        self.repr_list_dict["action"] = np.asarray(self.repr_list_dict["action"])
        self.repr_list_dict["z"] = np.asarray(self.repr_list_dict["z"])
        print(f"Shape of action: {self.repr_list_dict['action'].shape}")
        print(f"Shape of z: {self.repr_list_dict['z'].shape}")

        if not self.cfg.skip_text:
            self.repr_list_dict["text"] = np.asarray(self.repr_list_dict["text"], dtype=object)
            self.repr_list_dict["text_embedding"] = np.asarray(self.repr_list_dict["text_embedding"])

        # --- Calculate or load normalization stats ---
        if self.Mean is None:
            self._calculate_normalization_stats()
        else:
            print("Using pre-defined normalization stats.")
            self.Mean = self.Mean
            self.Std = self.Std
            self.ActionMean = self.ActionMean
            self.ActionStd = self.ActionStd
            self.zMean = self.zMean
            self.zStd = self.zStd
            
    def _calculate_normalization_stats(self):
        """Calculates mean and standard deviation for normalization."""
        self.Mean_dict = {}
        self.Std_dict = {}
        for repr_name in self.repr_name_list:
            data = self.repr_list_dict[repr_name].reshape(-1, self.repr_dim_dict[repr_name])
            self.Mean_dict[repr_name] = data.mean(axis=0).astype(np.float32)
            self.Std_dict[repr_name] = data.std(axis=0).astype(np.float32)
                
        self.Mean = np.concatenate(list(self.Mean_dict.values()), axis=-1)
        self.Std = np.concatenate(list(self.Std_dict.values()), axis=-1) + 1e-10   # Avoid invalid zero Std

        # Calculate stats for action and z-latent variables
        self.ActionMean = self.repr_list_dict["action"].reshape(-1, self.repr_list_dict["action"].shape[-1]).mean(axis=0).astype(np.float32)
        self.ActionStd = self.repr_list_dict["action"].reshape(-1, self.repr_list_dict["action"].shape[-1]).std(axis=0).astype(np.float32)
        
        if self.repr_list_dict["z"][0] is not None:
            self.zMean = self.repr_list_dict["z"].reshape(-1, self.repr_list_dict["z"].shape[-1]).mean(axis=0).astype(np.float32)
            self.zStd = self.repr_list_dict["z"].reshape(-1, self.repr_list_dict["z"].shape[-1]).std(axis=0).astype(np.float32)

    def load_data(self, only_succ=False, filter_keyword=None):
        """
        Loads raw data from pickle files.

        Args:
            only_succ (bool): If True, loads only successful sequences.
            filter_keyword (str): A keyword to filter sequences by their success key. Ignored.
        """
        for path in tqdm(self.data_path_list, desc="Loading raw data files"):
            data = joblib.load(path)
            indices = data["succ_idxes"] if only_succ else range(len(data['action_all']))
            
            for succ_i, i in enumerate(indices):
                # Divide long sequences into fixed-length clips
                N = len(data["action_all"][i])
                if N > self.n_frames:
                    frame_labels = None
                    if not self.skip_text and "frame_labels_all" in data:
                        frame_labels = data["frame_labels_all"][i]
                        # Recalculate time scale based on annotations
                        end_t = max([seg["end_t"] for seg in frame_labels if "end_t" in seg], default=0)
                        data_frame = data["action_all"][i].shape[0]
                        t_scale = (int(30 * end_t) / (data_frame + 1)) if data_frame > 0 else 1.0
                        
                        # Adjust for known problematic FPS rates
                        if "KIT" in data["motion_file"][succ_i] or "EKUT" in data["motion_file"][succ_i] or "MPI_mosh" in data["motion_file"][succ_i]:
                            t_scale = 0.9

                    for j in range(0, N - self.n_frames, self.stride):
                        texts = []
                        if not self.skip_text and frame_labels:
                            future_start = (j + 1) / self.target_fps
                            future_end = (j + self.n_frames + 1) / self.target_fps

                            for seg in frame_labels:
                                if "start_t" not in seg:
                                    continue
                                
                                # Check for overlap between clip time and text annotation time
                                if have_overlap(
                                    [seg['start_t'] / t_scale + self.text_strict, seg['end_t'] / t_scale - self.text_strict],
                                    [future_start - self.text_tolerance, future_end + self.text_tolerance]
                                ):
                                    texts.append(seg['proc_label'])

                        if len(texts) > 0:
                            # Randomly choose one text annotation per clip
                            text = random.choice(texts)
                            # Clean up text and filter by keywords
                            text = text.replace("transition to ", "")
                            if "walk back " in text or text == "walk back":
                                text = "walk"
                                
                            if self.key_actions is not None:
                                if not any(word in text for word in self.key_actions):
                                    continue
                            self.clip_list["text_all"].append(text)

                            # Append the corresponding data clip
                            for k in self.loaded_keys:
                                if k in data and data[k][i] is not None:
                                    clip_data = data[k][i][j : self.n_frames + j + 1]
                                    self.clip_list[k].append(clip_data)
                                else:
                                    self.clip_list[k].append(None)
                                    print_once(f"skip {k} as it is not in data")
                        else:
                            # If no text and we need it, we skip this clip
                            if not self.skip_text:
                                continue
                            # If text is not needed, we just add a placeholder and the clip
                            self.clip_list["text_all"].append('')
                            for k in self.loaded_keys:
                                if k in data and data[k][i] is not None:
                                    clip_data = data[k][i][j : self.n_frames + j + 1]
                                    self.clip_list[k].append(clip_data)
                                else:
                                    self.clip_list[k].append(None)
                                    print_once(f"skip {k} as it is not in data")
                else:
                    continue


    def __len__(self):
        """Returns the total number of processed clips in the dataset."""
        return len(self.clip_list["action_all"])
    
    def __getitem__(self, idx):
        """
        Retrieves a single data sample (clip) from the dataset.

        Args:
            idx (int): The index of the clip to retrieve.

        Returns:
            tuple: A tuple containing the combined state-action data and a dictionary
                   of conditioning information.
        """
        # Concatenate various state representations into a single state vector
        repr_dict_clean = {
            repr_name: self.repr_list_dict[repr_name][idx]
            for repr_name in self.repr_name_list
        }
        state = np.concatenate(
            [repr_dict_clean[key] for key in self.repr_name_list], axis=-1
        )
        # Normalize the state vector
        state = ((state - self.Mean) / self.Std).astype(np.float32)

        # Normalize action data
        action = np.concatenate(
            [self.repr_list_dict[k][idx] for k in self.action_keys], axis=-1
        )
        mean = self.ActionMean if self.action_keys == ["action"] else self.zMean
        std = self.ActionStd if self.action_keys == ["action"] else self.zStd
        action = ((action - mean) / std).astype(np.float32)

        # Create the conditioning dictionary
        cond = {}
        # cond['y']['state'] = torch.from_numpy(state[:self.context_length])
        # cond['y']['all_states'] = torch.from_numpy(state[:])
        # cond['y']['action'] = torch.from_numpy(self.repr_list_dict["action"][idx])

        # Add text conditioning if not skipped
        if not self.cfg.skip_text:
            cond['text'] = self.repr_list_dict["text"][idx]
            cond['text_embedding'] = torch.from_numpy(self.repr_list_dict["text_embedding"][idx])
        
        # Combine action and state for the final output
        combined_data = np.concatenate([action, state], axis=-1)
        return torch.from_numpy(combined_data), cond