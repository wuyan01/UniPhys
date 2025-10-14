import joblib

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
from einops import rearrange
from omegaconf import DictConfig
from tqdm import tqdm
from lightning.pytorch.utilities.types import STEP_OUTPUT

from phc.utils.flags import flags
from uniphys.utils.buffer_utils import FixedLengthBuffer
from uniphys.utils.clip_utils import load_and_freeze_clip, encode_text
from uniphys.utils.motion_repr_utils import (
    REPR_LIST_DOF_JOINT,
    REPR_LIST_ROOT_DOF_JOINT,
    get_repr,
    cano_seq_smpl_or_smplx_batch,
)

from .models.diffusion import Diffusion
from .df_base import DiffusionForcingBase


class DiffusionForcingHumanoid(DiffusionForcingBase):
    """
    A video prediction algorithm using Diffusion Forcing.
    """

    def __init__(self, cfg: DictConfig):
        self.n_tokens = cfg.n_frames // cfg.frame_stack  # number of max tokens for the model
        self.action_dim = cfg.action_dim
        self.state_dim = cfg.state_dim
        self.norm_action = cfg.norm_action
        self.observation_dim = self.action_dim + self.state_dim
        assert self.observation_dim == cfg.observation_shape[0]
        self.stabilization_schedule = cfg.diffusion.stabilization_schedule
        self.exec_step = cfg.diffusion.exec_step

        # EMA setup
        self.use_ema = cfg.diffusion.get("use_ema", False)
        if self.use_ema:
            self.ema_decay = cfg.diffusion.ema_decay
            self.ema_model = None

        self.text_embedding_dict = joblib.load("data/babel_tracking_results/text_embedding_dict_clip.pkl")
        super().__init__(cfg)

        ## for saving episodes
        self.pred_dof_pos, self.pred_dof_pos_all = [], []
        self.root_state, self.root_state_all = [], []
        self.dof_state, self.dof_state_all = [], []
        self.action, self.action_all = [], []
        self.zs, self.zs_all = [], []
        self.pred_pos, self.pred_pos_all = [], []
        self.episode_length = []
        self.text = []

        self.hml_tokens = []
        self.db_keys = []
        self.hml_lengths = []

        self._init_hist_buffer()

    def _init_hist_buffer(self):
        self.root_state_buffer = FixedLengthBuffer(self.cfg.n_frames)
        self.dof_state_buffer = FixedLengthBuffer(self.cfg.n_frames)
        self.joint_pos_buffer = FixedLengthBuffer(self.cfg.n_frames)
        self.self_obs_buffer = FixedLengthBuffer(self.cfg.n_frames)
        self.action_buffer = FixedLengthBuffer(self.cfg.n_frames)

    def _clear_hist_buffer(self):
        self.root_state_buffer.clear()
        self.dof_state_buffer.clear()
        self.joint_pos_buffer.clear()
        self.self_obs_buffer.clear()
        self.action_buffer.clear()

    def post_step(self, info, done, save_dir=None):

        if save_dir is not None:
            self.pred_pos.append(info["body_pos"])
            self.pred_dof_pos.append(info["dof_pos"])
            self.root_state.append(info["root_state"])
            self.dof_state.append(info["dof_state"])
            self.action.append(info["action"].clone().detach().cpu().numpy())

            if "z" in info: 
                self.zs.append(info["z"])

        done = done | (self.env.task.progress_buf >= self.max_episode_length) # timeout

        return done

    def _build_model(self):
        self.cfg.data_mean = 0 ## normalization is done in the dataloader
        self.cfg.data_std = 1 ## TODO: regsiter the mean and std after dataset initialization
        self.diffusion_model = Diffusion(
            x_shape=self.x_stacked_shape,
            external_cond_dim=self.external_cond_dim,
            is_causal=self.causal,
            cond_mask_prob=self.cond_mask_prob,
            cfg=self.cfg.diffusion,
            action_dim=self.cfg.action_dim,
            state_dim=self.cfg.state_dim,
        )

        if self.use_ema and self.ema_model is None:
            self.ema_model = AveragedModel(
                    self.diffusion_model,
                    avg_fn=lambda avg_model_param, model_param, num_averaged: self.ema_decay * avg_model_param
                    + (1 - self.ema_decay) * model_param,
                )
            self.ema_model.to(self.device)
        
        self.register_data_mean_std(self.cfg.data_mean, self.cfg.data_std)

        self.clip_model = load_and_freeze_clip(clip_version='ViT-B/32', device="cuda")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.use_ema:
            self.ema_model.update_parameters(self.diffusion_model)

    def load_state_dict(self, state_dict, strict=True):
        if self.use_ema and "ema_state_dict" in state_dict:
            # This is likely from an EMA checkpoint, load the EMA weights
            ema_state_dict = state_dict.pop("ema_state_dict")
            super().load_state_dict(ema_state_dict, strict=False)
        else:
            super().load_state_dict(state_dict, strict=False)

    def on_save_checkpoint(self, checkpoint):
        if self.use_ema:
            # Save the EMA model's state_dict in the checkpoint
            checkpoint["ema_state_dict"] = self.ema_model.module.state_dict()

    def on_train_epoch_end(self, ) -> None:
        if (torch.cuda.current_device() == 0) & self.cfg.play:
            if self.use_ema and self.ema_model is not None:
                self.ema_model.eval()
            else:
                self.diffusion_model.eval()

            current_epoch = self.current_epoch
            if (self.current_epoch + 1) % 100 == 0 or self.current_epoch == 0:
                mean_episode_length, std_episode_length, max_episode_length, min_episode_length, num_played_games = self.interact(save=False)
                
                self.log("mean_episode_length", mean_episode_length, on_step=False, on_epoch=True, sync_dist=True)
                self.log("std_episode_length", std_episode_length, on_step=False, on_epoch=True, sync_dist=True)
                self.log("max_episode_length", max_episode_length, on_step=False, on_epoch=True, sync_dist=True)
                self.log("min_episode_length", min_episode_length, on_step=False, on_epoch=True, sync_dist=True)
                print(f"Epoch[{current_epoch}] -- Mean Episode length [{mean_episode_length:.2f}] -- Std Episode length [{std_episode_length:.2f}] -- Played games [{num_played_games}]")
                
                if self.use_ema and self.ema_model is not None:
                    # self.ema_model.train() # ignore
                    pass
                else:
                    self.diffusion_model.train()


    def _preprocess_batch(self, batch):
        xs = batch[0]
        
        batch_size, n_frames = xs.shape[:2]

        if n_frames % self.frame_stack != 0:
            raise ValueError("Number of frames must be divisible by frame stack size")
        if self.context_frames % self.frame_stack != 0:
            raise ValueError("Number of context frames must be divisible by frame stack size")

        masks = torch.ones(n_frames, batch_size).to(xs.device)
        n_frames = n_frames // self.frame_stack

        xs = self._normalize_x(xs)

        xs = rearrange(xs, "b (t fs) c ... -> t b (fs c) ...", fs=self.frame_stack).contiguous()

        if not self.cfg.skip_text and "text_embedding" in batch[1]:
            conditions = batch[1]["text_embedding"]
        else:
            conditions = None

        return xs, conditions, masks

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        xs, conditions, masks = self._preprocess_batch(batch)

        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            xs_pred, loss = self.diffusion_model(xs, conditions, force_mask=False, noise_levels=self._generate_noise_levels(xs))
        
        if self.cfg.action_keys == ["z"]:
            assert self.action_dim == 32
            z = xs_pred[..., :self.action_dim][1:]

            T, B = z.shape[:2]

            z = z.reshape(T*B, -1)

            z_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, :self.action_dim].reshape(*loss.shape[:2], -1)

            state_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, self.action_dim:].reshape(*loss.shape[:2], -1)
            
            if self.cfg.state_with_root:
                root_trans_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, self.action_dim:self.action_dim+3].reshape(*loss.shape[:2], -1)
                root_rot_6d_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, self.action_dim+3:self.action_dim+9].reshape(*loss.shape[:2], -1)
                root_trans_vel_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, self.action_dim+9:self.action_dim+12].reshape(*loss.shape[:2], -1)
                root_rot_vel_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, self.action_dim+12:self.action_dim+15].reshape(*loss.shape[:2], -1)
                local_pos_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, self.action_dim+15:self.action_dim+15+24*3].reshape(*loss.shape[:2], -1)
                local_pos_vel_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, self.action_dim+15+24*3:self.action_dim+15+24*3+24*3].reshape(*loss.shape[:2], -1)
                dof_pose_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, -23*9:-23*3].reshape(*loss.shape[:2], -1)
                dof_vel_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, -23*3:].reshape(*loss.shape[:2], -1)
            
                root_trans_loss = self.reweight_loss(root_trans_loss, masks)
                root_rot_6d_loss = self.reweight_loss(root_rot_6d_loss, masks)
                root_trans_vel_loss = self.reweight_loss(root_trans_vel_loss, masks)
                root_rot_vel_loss = self.reweight_loss(root_rot_vel_loss, masks)
                local_pos_loss = self.reweight_loss(local_pos_loss, masks)
                local_pos_vel_loss = self.reweight_loss(local_pos_vel_loss, masks)
                dof_pose_loss = self.reweight_loss(dof_pose_loss, masks)
                dof_vel_loss = self.reweight_loss(dof_vel_loss, masks)
            else:
                local_pos_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, self.action_dim:self.action_dim+24*3].reshape(*loss.shape[:2], -1)
                local_pos_vel_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, self.action_dim+24*3:self.action_dim+24*3+24*3].reshape(*loss.shape[:2], -1)
                dof_pose_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, -23*9:-23*3].reshape(*loss.shape[:2], -1)
                dof_vel_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, -23*3:].reshape(*loss.shape[:2], -1)

                local_pos_loss = self.reweight_loss(local_pos_loss, masks)
                local_pos_vel_loss = self.reweight_loss(local_pos_vel_loss, masks)
                dof_pose_loss = self.reweight_loss(dof_pose_loss, masks)
                dof_vel_loss = self.reweight_loss(dof_vel_loss, masks)


            z_loss = self.reweight_loss(z_loss, masks)
            state_loss = self.reweight_loss(state_loss, masks)

            dim_masks = torch.ones((1, 1, loss.shape[-1])).cuda()

            if self.cfg.state_with_root:
                dim_masks[..., self.action_dim:self.action_dim+15] = 1

            loss = self.reweight_loss(loss * dim_masks, masks)

        else:
            assert self.norm_action
            assert self.action_dim == 69
            z_loss = torch.zeros(())
            action_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, :self.action_dim].reshape(*loss.shape[:2], -1)
            state_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, self.action_dim:].reshape(*loss.shape[:2], -1)

            if self.cfg.state_with_root:
                root_trans_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, self.action_dim:self.action_dim+3].reshape(*loss.shape[:2], -1)
                root_rot_6d_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, self.action_dim+3:self.action_dim+9].reshape(*loss.shape[:2], -1)
                root_trans_vel_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, self.action_dim+9:self.action_dim+12].reshape(*loss.shape[:2], -1)
                root_rot_vel_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, self.action_dim+12:self.action_dim+15].reshape(*loss.shape[:2], -1)
                local_pos_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, self.action_dim+15:self.action_dim+15+24*3].reshape(*loss.shape[:2], -1)
                local_pos_vel_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, self.action_dim+15+24*3:self.action_dim+15+24*3+24*3].reshape(*loss.shape[:2], -1)
                dof_pose_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, -23*9:-23*3].reshape(*loss.shape[:2], -1)
                dof_vel_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, -23*3:].reshape(*loss.shape[:2], -1)
            
                root_trans_loss = self.reweight_loss(root_trans_loss, masks)
                root_rot_6d_loss = self.reweight_loss(root_rot_6d_loss, masks)
                root_trans_vel_loss = self.reweight_loss(root_trans_vel_loss, masks)
                root_rot_vel_loss = self.reweight_loss(root_rot_vel_loss, masks)
                local_pos_loss = self.reweight_loss(local_pos_loss, masks)
                local_pos_vel_loss = self.reweight_loss(local_pos_vel_loss, masks)
                dof_pose_loss = self.reweight_loss(dof_pose_loss, masks)
                dof_vel_loss = self.reweight_loss(dof_vel_loss, masks)
            else:
                local_pos_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, self.action_dim:self.action_dim+24*3].reshape(*loss.shape[:2], -1)
                local_pos_vel_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, self.action_dim+24*3:self.action_dim+24*3+24*3].reshape(*loss.shape[:2], -1)
                dof_pose_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, -23*9:-23*3].reshape(*loss.shape[:2], -1)
                dof_vel_loss = loss.reshape(*loss.shape[:2], -1, self.frame_stack)[:, :, -23*3:].reshape(*loss.shape[:2], -1)

                local_pos_loss = self.reweight_loss(local_pos_loss, masks)
                local_pos_vel_loss = self.reweight_loss(local_pos_vel_loss, masks)
                dof_pose_loss = self.reweight_loss(dof_pose_loss, masks)
                dof_vel_loss = self.reweight_loss(dof_vel_loss, masks)
        
            action_loss = self.reweight_loss(action_loss, masks)
            state_loss = self.reweight_loss(state_loss, masks)
            loss = self.reweight_loss(loss, masks)

        # log the loss
        if batch_idx % 20 == 0:
            self.log("training/loss", loss, on_step=True, on_epoch=False, sync_dist=True)
            self.log("training/z_loss", z_loss, on_step=True, on_epoch=False, sync_dist=True)
            self.log("training/state_loss", state_loss, on_step=True, on_epoch=False, sync_dist=True)

            if self.cfg.state_with_root:
                self.log("training/root_trans_loss", root_trans_loss, on_step=True, on_epoch=False, sync_dist=True)
                self.log("training/root_rot_6d_loss", root_rot_6d_loss, on_step=True, on_epoch=False, sync_dist=True)
                self.log("training/root_trans_vel_loss", root_trans_vel_loss, on_step=True, on_epoch=False, sync_dist=True)
                self.log("training/root_rot_vel_loss", root_rot_vel_loss, on_step=True, on_epoch=False, sync_dist=True)
                self.log("training/local_pos_loss", local_pos_loss, on_step=True, on_epoch=False, sync_dist=True)
                self.log("training/local_pos_vel_loss", local_pos_vel_loss, on_step=True, on_epoch=False, sync_dist=True)
                self.log("training/dof_pose_loss", dof_pose_loss, on_step=True, on_epoch=False, sync_dist=True)
                self.log("training/dof_vel_loss", dof_vel_loss, on_step=True, on_epoch=False, sync_dist=True)
            else:
                self.log("training/local_pos_loss", local_pos_loss, on_step=True, on_epoch=False, sync_dist=True)
                self.log("training/local_pos_vel_loss", local_pos_vel_loss, on_step=True, on_epoch=False, sync_dist=True)
                self.log("training/dof_pose_loss", dof_pose_loss, on_step=True, on_epoch=False, sync_dist=True)
                self.log("training/dof_vel_loss", dof_vel_loss, on_step=True, on_epoch=False, sync_dist=True)

        xs = self._unstack_and_unnormalize(xs)
        xs_pred = self._unstack_and_unnormalize(xs_pred)

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

        return output_dict


    def preprocess_obs(self, source_data_joints_position, source_data_dof_state, source_data_root_state, state_key):
        cano_joints_position, cano_dof_state, cano_root_state, cano_transf_matrix = cano_seq_smpl_or_smplx_batch(source_data_joints_position, source_data_dof_state, source_data_root_state)

        states = []
        for i in range(len(cano_joints_position)):
            repr_dict = get_repr(cano_joints_position[i], cano_dof_state[i], cano_root_state[i], return_last=True)

            state_list = []
            state_list = [repr_dict[repr_name] for repr_name in state_key]
           
            state = np.concatenate(state_list, axis=-1)
            state = ((state - self.state_mean) / self.state_std).astype(np.float32)
            states.append(state)

        return torch.from_numpy(np.stack(states)).cuda()


    def policy(self):
        """
        Policy function for generating actions using a diffusion-based model.
        It processes buffered states and actions, builds conditional input sequences,
        and runs iterative denoising with optional classifier-free guidance.
        """
        # Use EMA model for inference if it's enabled.
        if self.use_ema and self.ema_model is not None:
            model_to_use = self.ema_model.module
        else:
            model_to_use = self.diffusion_model


        # -------------------------------------------------------
        # Select representation list depending on root inclusion
        # -------------------------------------------------------
        REPR_LIST = (
            REPR_LIST_ROOT_DOF_JOINT if self.cfg.state_with_root else REPR_LIST_DOF_JOINT
        )

        # -------------------------------------------------------
        # Collect buffered history (limited by context length)
        # -------------------------------------------------------
        context_frames = min(len(self.root_state_buffer.buffer), self.context_frames)

        root_state = self.root_state_buffer.get().clone().cpu().numpy()[-context_frames:]   # (T, B, 13)
        dof_state = self.dof_state_buffer.get().clone().cpu().numpy()[-context_frames:]    # (T, B, 69, 2)
        joint_position = self.joint_pos_buffer.get().clone().cpu().numpy()[-context_frames:]
        hist_actions = self.action_buffer.get().clone().cpu().numpy()[-context_frames:]    # (T, B, 69)
        H, B = root_state.shape[:2]
        self.H = H

        # -------------------------------------------------------
        # Preprocess state (normalize + feature representation)
        # Output shape: (B, T, D_state)
        # -------------------------------------------------------
        state = self.preprocess_obs(
            joint_position.transpose(1, 0, 2, 3),  # (B, T, J, 3)
            dof_state.transpose(1, 0, 2, 3),       # (B, T, J, 2)
            root_state.transpose(1, 0, 2),         # (B, T, D_root)
            REPR_LIST,
        )

        # -------------------------------------------------------
        # Preprocess historical actions (normalize + reshape)
        # -------------------------------------------------------
        if self.cfg.action_keys == ["z"]:
            hist_actions = torch.from_numpy(hist_actions).permute(1, 0, 2).cuda()  # (B, T, d)
        else:
            hist_actions = (
                torch.from_numpy(hist_actions.reshape(*hist_actions.shape[:2], 23, -1))
                .permute(1, 0, 2, 3)
                .cuda()
            )  # (B, T, 23, d_sub)

        # Normalize actions
        hist_actions = (
            (hist_actions.reshape(*hist_actions.shape[:2], -1) - torch.from_numpy(self.a_mean).cuda())
            / torch.from_numpy(self.a_std).cuda()
        )
        hist_actions = hist_actions.reshape(*hist_actions.shape[:2], -1)  # (B, T, d)

        # -------------------------------------------------------
        # Combine state + action into diffusion input
        # -------------------------------------------------------
        xs_pred = torch.cat([hist_actions, state], dim=-1).permute(1, 0, 2).float().cuda()  # (T, B, D)
        xs_pred = rearrange(xs_pred, "(t fs) b c -> t b (fs c)", fs=self.frame_stack).contiguous()

        # Horizon setup
        horizon = self.chunk_size // self.frame_stack
        assert horizon <= self.n_tokens, "horizon exceeds the number of tokens."
        assert self.chunk_size >= self.exec_step, "chunk size smaller than open-loop control step"
        scheduling_matrix = self._generate_scheduling_matrix(horizon)

        # -------------------------------------------------------
        # Sample-based search [search is optional, disable search with self.n_samples=1]
        # -------------------------------------------------------
        B = self.env.task.num_envs * self.n_samples
        chunk = torch.randn((horizon, B, *self.x_stacked_shape)).cuda()
        chunk = torch.clamp(chunk, -self.clip_noise, self.clip_noise)

        if xs_pred is not None:
            # Repeat input across n_samples
            xs_pred = xs_pred.repeat_interleave(self.n_samples, dim=1)
            xs_pred = torch.cat([xs_pred, chunk], 0)
        else:
            xs_pred = torch.randn((self.n_tokens, B, *self.x_stacked_shape)).cuda()
            xs_pred = torch.clamp(xs_pred, -self.clip_noise, self.clip_noise)

        curr_frame = H // self.frame_stack
        start_frame = 0

        # -------------------------------------------------------
        # Diffusion sampling loop (conditional + CFG guidance)
        # -------------------------------------------------------
        xs_pred_cond = xs_pred.clone()

        for m in range(scheduling_matrix.shape[0] - 1):

            # Construct noise schedules
            from_noise_levels = np.concatenate(
                (np.zeros((curr_frame,), dtype=np.int64), scheduling_matrix[m])
            )[:, None].repeat(B, axis=1)
            to_noise_levels = np.concatenate(
                (np.zeros((curr_frame,), dtype=np.int64), scheduling_matrix[m + 1])
            )[:, None].repeat(B, axis=1)

            from_noise_levels = torch.from_numpy(from_noise_levels).cuda()
            to_noise_levels = torch.from_numpy(to_noise_levels).cuda()

            # Unconditional sampling step
            out = model_to_use.sample_step(
                xs_pred_cond[start_frame:],
                # self.text_embedding.expand(B, 512).cuda(),
                self.text_embedding.repeat(self.n_samples, 1).cuda(),
                from_noise_levels[start_frame:],
                to_noise_levels[start_frame:],
                force_mask=True,   # True = unconditional
                guidance_fn=self.guidance_fn,
            )

            # Conditional branch with guidance (if enabled)
            if not self.cfg.skip_text and self.guidance_params > 0:
                out_cond = model_to_use.sample_step(
                    xs_pred_cond[start_frame:],
                    # self.text_embedding.expand(B, 512).cuda(),
                    self.text_embedding.repeat(self.n_samples, 1).cuda(),
                    from_noise_levels[start_frame:],
                    to_noise_levels[start_frame:],
                    force_mask=False,  # False = conditional
                    guidance_fn=self.guidance_fn,
                )
                # Classifier-free guidance
                out = out + self.guidance_params * (out_cond - out)

            xs_pred_cond[start_frame:] = out
            xs_pred = xs_pred_cond.clone()

            # Reshape back for action/state decoding
            xs_pred = rearrange(xs_pred, "t b (fs c) -> (t fs) b c", fs=self.frame_stack).contiguous()

            if self.n_samples > 1:
                xs_pred = rearrange(xs_pred, "t (n b) c -> n t b c", n=self.n_samples).contiguous()

            # -------------------------------------------------------
            # Decode actions and states (de-normalize)
            # -------------------------------------------------------
            action_pred = xs_pred[..., :self.action_dim]
            state_pred = xs_pred[..., self.action_dim:]

            action_pred = action_pred * torch.from_numpy(self.a_std).cuda() + torch.from_numpy(self.a_mean).cuda()
            state_pred = state_pred * torch.from_numpy(self.state_std).cuda() + torch.from_numpy(self.state_mean).cuda()

        return action_pred, state_pred
    

    @torch.no_grad()
    def interact(self, save=False):

        # === Environment setup ===
        self.env.task._termination_distances[:] = 1e6
        self.env.task.termination_mode = "sampling"
        num_envs = self.env.task.num_envs
        self.max_episode_length = self.env.task.max_episode_length

        pbar = tqdm(self.max_episode_length)

        # === Guidance & text setup ===
        self.guidance_fn = None
        self.n_samples = 1 # for interactive mode, set n_samples to 1
        self.guidance_params = self.cfg.guidance_params
        self.clip_model = load_and_freeze_clip(clip_version='ViT-B/32', device="cuda")
        self.text_embedding = None

        # === Buffers ===
        self._clear_hist_buffer()

        # === Action Mean and Std stats ===
        REPR_LIST = (
            REPR_LIST_ROOT_DOF_JOINT if self.cfg.state_with_root else REPR_LIST_DOF_JOINT
        )
        if self.norm_action:
            if self.cfg.action_keys == ["action"]:
                self.a_mean, self.a_std = self.action_mean, self.action_std
            elif self.cfg.action_keys == ["z"]:
                self.a_mean, self.a_std = self.z_mean, self.z_std
            else:
                raise ValueError("Unsupported action_keys")
        else:
            self.a_mean, self.a_std = np.zeros(self.action_dim), np.ones(self.action_dim)
        
        # === Bookkeeping ===
        completed_episode_lengths = []
        self.env.task.completed_episode_lengths = []

        ## === Save dir ===
        if save:
            import os
            save_dir = os.path.join(self.save_dir, "eval", "step_{}_exec_{}_noise_{}_n_{}".format(self.resume_step, self.exec_step, self.diffusion_model.stabilization_level, self.cfg.diffusion.sampling_timesteps))
            if not self.cfg.skip_text and self.guidance_params > 0:
                save_dir += "_g{}".format(self.cfg.guidance_params)
            else:
                save_dir += "_uncond"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        else:
            save_dir = None

        episode_length = np.zeros(num_envs)
        is_done = np.zeros(num_envs, dtype=bool)
        done_indices = []
        obs_dict = self.player.env_reset()

        t = 0
        break_flag = False
        self.env.task.set_char_color(col=[0, 0.5, 0.8], env_ids=torch.arange(self.env.task.num_envs))  ## blue

        
        while t < self.max_episode_length and not np.all(is_done):
            self.env.task.set_char_color(col=[1, 0.5, 0.0], env_ids=torch.arange(self.env.task.num_envs))  ## orange

            """
            Sampling
            """
            with torch.no_grad():
                while len(self.root_state_buffer.buffer) < 2:
                    """
                    Init with zero actions for 2 steps
                    """
                    obs_dict = self.player.env_reset(done_indices)
                    if self.cfg.action_keys == ["z"]:
                        # Action representation: latent space
                        zero_z = torch.zeros((self.env.task.num_envs, self.action_dim)).cuda()
                        zero_z = zero_z * torch.from_numpy(self.a_std).cuda() + torch.from_numpy(self.a_mean).cuda()
                        if self.cfg.action_keys == ["z"]:
                            with torch.no_grad():
                                zero_action, _ = self.player.dec_action(zero_z, obs_dict)
                    else:
                        # Action representation: raw action
                        zero_z = zero_action = torch.zeros((self.env.task.num_envs, self.action_dim)).cuda()

                    obs_dict, r, done, info = self.player.env_step(self.env, zero_action)

                    self.root_state_buffer.add(self.env.task._root_states[self.env.task._humanoid_actor_ids].clone())
                    self.dof_state_buffer.add(self.env.task._dof_state.reshape(num_envs, -1, 2).clone())
                    self.joint_pos_buffer.add(self.env.task._rigid_body_pos.clone())
                    self.action_buffer.add(zero_z.clone())
                
                else:
                    """
                    Get text input
                    """
                    if not self.cfg.skip_text and self.guidance_params > 0:
                        if getattr(self.cfg, "interactive_input_prompt", False):
                            if flags.prompt:
                                text = input("\033[34mEnter the text prompt: \033[0m")
                                flags.prompt = False
                                print("\033[1mPress Q to input new command\033[0m")
                            else:
                                text = text
                        else:
                            text = self.cfg.text_prompt

                        if text not in self.text_embedding_dict:
                            self.text_embedding = encode_text(self.clip_model, [text])
                        else:
                            self.text_embedding = torch.from_numpy(self.text_embedding_dict[text]) # [1, 512]
                    else:
                        text = None
                        self.text_embedding = torch.zeros((1, 512))

                    action_pred, state_pred = self.policy()
                    
                    action_exec = action_pred[self.H:].permute(1, 0, 2).float()

                    for i in range(self.exec_step):

                        self.text.append(text)

                        a = action_exec[:, i]

                        if self.cfg.action_keys == ["z"]:
                            with torch.no_grad():
                                input_dict = {"obs": obs_dict}
                                a_exec, _ = self.player.dec_action(a, input_dict)

                        else:
                            a_exec = a

                        obs_dict, r, done, info = self.player.env_step(self.env, a_exec)
                        done = self.post_step(info, done.clone(), save_dir=save_dir)

                        self.root_state_buffer.add(self.env.task._root_states[self.env.task._humanoid_actor_ids].clone())
                        self.dof_state_buffer.add(self.env.task._dof_state.reshape(num_envs, -1, 2).clone())
                        self.joint_pos_buffer.add(self.env.task._rigid_body_pos.clone())
                        self.action_buffer.add(a.clone())  ## "z" or "action"

                        t += 1
                        episode_length += 1

                        for j in range(num_envs):
                            if done[j] and not is_done[j]:  # If the environment is done
                                completed_episode_lengths.append(episode_length[j])  # Store the length
                                episode_length[j] = 0  # Reset the step count for the new episode
                                is_done[j] = True

                            if is_done[j] and done[j]:  # Reset environment for next episode
                                episode_length[j] = 0  # Reset the step count for the new episode


                        if np.all(is_done):
                            self.player.curr_stpes = 0
                            break_flag = True
                            break

                        pbar.refresh()
                        pbar.set_description(f"Steps {t}/{self.max_episode_length} | Terminated {sum(is_done)}/{num_envs} | Text: {text} | Guidance: {self.guidance_params}")
            
            if break_flag:
                break

        print()

        if len(completed_episode_lengths) > 0:
            mean_episode_length = np.mean(completed_episode_lengths)
            std_episode_length = np.std(completed_episode_lengths)
            max_episode_length = np.max(completed_episode_lengths)
            min_episode_length = np.min(completed_episode_lengths)
            print(f"Mean episode length: {mean_episode_length:.2f}; Std episode length: {std_episode_length:.2f}; Played game: {len(completed_episode_lengths)}")
            print(f"Max episode length: {max_episode_length:.2f}; Min episode length: {min_episode_length:.2f};")
        else:
            print("No episodes completed.")
            mean_episode_length = 0
        
        if save:
            ### save episodes
            save_info = {"body_pos": np.stack(self.pred_pos),   # T, B, ...
                        "dof_pos": np.stack(self.pred_dof_pos),  # T, B, ...
                        "root_state": np.stack(self.root_state),   # T, B, ...
                        "dof_state": np.stack(self.dof_state),    # T, B, ...
                        "action": np.stack(self.action),    # T, B, ...
                        "text": self.text, # T
                        "episode_length": completed_episode_lengths,  # B
                        }
            
            joblib.dump(save_info, os.path.join(save_dir, "saved_episodes.pkl"))

        # clear buffer
        self._clear_hist_buffer()
        self.pred_pos, self.pred_dof_pos, self.root_state, self.dof_state, self.action, self.text = [], [], [], [], [], []

        return mean_episode_length, std_episode_length, max_episode_length, min_episode_length, len(completed_episode_lengths)


    @torch.no_grad()
    def evaluate_t2m_babel(self, save=True):
        # === Load evaluation text commands ===
        if not hasattr(self.cfg, "evaluate_text_file") or self.cfg.evaluate_text_file is None:
            self.cfg.evaluate_text_file = "sample_data/evaluate_text.txt"
        
        try:
            with open(self.cfg.evaluate_text_file, "r", encoding="utf-8") as f:
                test_cmd_list = f.read().splitlines()
        except FileNotFoundError:
            print(f"{self.cfg.evaluate_text_file} not found. Skipping evaluation.")
            return 0
        
        n_rollout_per_text_cmd = 5   # 5 samples per text

        # === Environment setup ===
        self.env.task._termination_distances[:] = 1e6
        self.env.task.termination_mode = "sampling"
        num_envs = self.env.task.num_envs
        self.max_episode_length = self.env.task.max_episode_length

        # === Guidance & text setup ===
        self.guidance_fn = None
        self.guidance_params = self.cfg.guidance_params
        self.n_samples = 1
        self.clip_model = load_and_freeze_clip(clip_version='ViT-B/32', device="cuda")
        self.text_embedding = None

        # === Buffers ===
        self._clear_hist_buffer()

        # === Action Mean and Std stats ===
        REPR_LIST = (
            REPR_LIST_ROOT_DOF_JOINT if self.cfg.state_with_root else REPR_LIST_DOF_JOINT
        )

        if self.norm_action:
            if self.cfg.action_keys == ["action"]:
                self.a_mean, self.a_std = self.action_mean, self.action_std
            elif self.cfg.action_keys == ["z"]:
                self.a_mean, self.a_std = self.z_mean, self.z_std
            else:
                raise ValueError("Unsupported action_keys")
        else:
            self.a_mean, self.a_std = np.zeros(self.action_dim), np.ones(self.action_dim)
        
        # === Bookkeeping ===
        self.env.task.completed_episode_lengths = []
                                               
        ## === Save dir ===
        if save:
            import os
            save_dir = os.path.join(self.save_dir, "evalate_t2m", "step_{}_exec_{}_noise_{}_n_{}".format(self.resume_step, self.exec_step, self.diffusion_model.stabilization_level, self.cfg.diffusion.sampling_timesteps))
            if not self.cfg.skip_text and self.guidance_params > 0:
                save_dir += "_g{}".format(self.cfg.guidance_params)
            else:
                save_dir += "_uncond"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        else:
            save_dir = None

        # === Loop over the text_cmd_list_all ===
        text_cmd_list_all = test_cmd_list * n_rollout_per_text_cmd
        batch_num = (len(text_cmd_list_all) + num_envs - 1) // num_envs

        all_episodes_data = {
            "body_pos": [], "dof_pos": [], "root_state": [], "dof_state": [],
            "action": [], "text": [], "episode_length": []
        }

        pbar = tqdm(total=len(text_cmd_list_all), desc="Evaluating Text-to-Motion")

        for i in range(batch_num):
            start_idx = i * num_envs
            end_idx = min(start_idx + num_envs, len(text_cmd_list_all))
            test_cmd_list_batch = text_cmd_list_all[start_idx:end_idx]
            
            current_batch_size = len(test_cmd_list_batch)
            if current_batch_size < num_envs:
                # Pad the last batch if necessary
                test_cmd_list_batch.extend([text_cmd_list_all[0]] * (num_envs - current_batch_size))

            self.text_embedding = encode_text(self.clip_model, test_cmd_list_batch)

            # Reset buffers and environment for the new batch
            self._clear_hist_buffer()
            episode_length = np.zeros(num_envs, dtype=int)
            is_done = np.zeros(num_envs, dtype=bool)
            obs_dict = self.player.env_reset()
            done_indices = []
            t = 0

            while t < self.max_episode_length and not np.all(is_done):
                self.env.task.set_char_color(col=[1, 0.5, 0.0], env_ids=torch.arange(self.env.task.num_envs))  ## orange

                """
                Sampling
                """
                with torch.no_grad():
                    while len(self.root_state_buffer.buffer) < 2:
                        """
                        Init with zero actions for 2 steps
                        """
                        obs_dict = self.player.env_reset(done_indices)
                        if self.cfg.action_keys == ["z"]:
                            # Action representation: latent space
                            zero_z = torch.zeros((self.env.task.num_envs, self.action_dim)).cuda()
                            zero_z = zero_z * torch.from_numpy(self.a_std).cuda() + torch.from_numpy(self.a_mean).cuda()
                            if self.cfg.action_keys == ["z"]:
                                with torch.no_grad():
                                    zero_action, _ = self.player.dec_action(zero_z, obs_dict)
                        else:
                            # Action representation: raw action
                            zero_z = zero_action = torch.zeros((self.env.task.num_envs, self.action_dim)).cuda()

                        obs_dict, r, done, info = self.player.env_step(self.env, zero_action)

                        self.root_state_buffer.add(self.env.task._root_states[self.env.task._humanoid_actor_ids].clone())
                        self.dof_state_buffer.add(self.env.task._dof_state.reshape(num_envs, -1, 2).clone())
                        self.joint_pos_buffer.add(self.env.task._rigid_body_pos.clone())
                        self.action_buffer.add(zero_z.clone())
                    
                    else:
                        action_pred, state_pred = self.policy()
                        
                        action_exec = action_pred[self.H:].permute(1, 0, 2).float()

                        for i in range(self.exec_step):

                            a = action_exec[:, i]

                            if self.cfg.action_keys == ["z"]:
                                with torch.no_grad():
                                    input_dict = {"obs": obs_dict}
                                    a_exec, _ = self.player.dec_action(a, input_dict)

                            else:
                                a_exec = a

                            obs_dict, r, done, info = self.player.env_step(self.env, a_exec)
                            done = self.post_step(info, done.clone(), save_dir=save_dir)

                            self.root_state_buffer.add(self.env.task._root_states[self.env.task._humanoid_actor_ids].clone())
                            self.dof_state_buffer.add(self.env.task._dof_state.reshape(num_envs, -1, 2).clone())
                            self.joint_pos_buffer.add(self.env.task._rigid_body_pos.clone())
                            self.action_buffer.add(a.clone())  ## "z" or "action"

                            t += 1
                            episode_length[~is_done] += 1

                            for j in range(num_envs):
                                if done[j] and not is_done[j]:  # If the environment is done
                                    is_done[j] = True

                            if np.all(is_done):
                                break
            
            # After episode batch finishes, collect and store data
            all_episodes_data["body_pos"].append(np.stack(self.pred_pos)[:, :current_batch_size])
            all_episodes_data["dof_pos"].append(np.stack(self.pred_dof_pos)[:, :current_batch_size])
            all_episodes_data["root_state"].append(np.stack(self.root_state)[:, :current_batch_size])
            all_episodes_data["dof_state"].append(np.stack(self.dof_state)[:, :current_batch_size])
            all_episodes_data["action"].append(np.stack(self.action)[:, :current_batch_size])
            all_episodes_data["text"].extend(test_cmd_list_batch[:current_batch_size])
            all_episodes_data["episode_length"].extend(episode_length[:current_batch_size])
            
            # Clear per-batch buffers
            self.pred_pos, self.pred_dof_pos, self.root_state, self.dof_state, self.action = [], [], [], [], []
            pbar.update(current_batch_size)

        pbar.close()

        if save:
            # Concatenate data from all batches
            final_data = {
                "body_pos": np.concatenate(all_episodes_data["body_pos"], axis=1),      # T, B ,...
                "dof_pos": np.concatenate(all_episodes_data["dof_pos"], axis=1),        # T, B ,...
                "root_state": np.concatenate(all_episodes_data["root_state"], axis=1),  # T, B ,...
                "dof_state": np.concatenate(all_episodes_data["dof_state"], axis=1),    # T, B ,...
                "action": np.concatenate(all_episodes_data["action"], axis=1),          # T, B ,...
                "text": all_episodes_data["text"],     # B
                "episode_length": all_episodes_data["episode_length"],    # B
            }
            
            save_path = os.path.join(save_dir, "saved_episodes.pkl")
            joblib.dump(final_data, save_path)
            print(f"Saved {len(final_data['text'])} episodes to {save_path}")

