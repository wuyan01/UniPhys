
import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import numpy as np
import os
import yaml
from tqdm import tqdm

from phc.utils import torch_utils
import joblib
import torch
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
import torch.multiprocessing as mp
import gc
from scipy.spatial.transform import Rotation as sRot
import random
from phc.utils.flags import flags
from enum import Enum
USE_CACHE = False
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)
from phc.utils.motion_lib_base import MotionLibBase
from phc.utils.motion_lib_smpl import MotionLibSMPL
# from uhc.utils.torch_ext import to_torch
# from isaacgym.torch_utils import to_torch
import copy

def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)

class FixHeightMode(Enum):
    no_fix = 0
    full_fix = 1
    ankle_fix = 2

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy

    class Patch:

        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy


def local_rotation_to_dof_vel(local_rot0, local_rot1, dt):
    # Assume each joint is 3dof
    diff_quat_data = torch_utils.quat_mul(torch_utils.quat_conjugate(local_rot0), local_rot1)
    diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
    dof_vel = diff_axis * diff_angle.unsqueeze(-1) / dt

    return dof_vel[1:, :].flatten()


def compute_motion_dof_vels(motion):
    num_frames = motion.tensor.shape[0]
    dt = 1.0 / motion.fps
    dof_vels = []

    for f in range(num_frames - 1):
        local_rot0 = motion.local_rotation[f]
        local_rot1 = motion.local_rotation[f + 1]
        frame_dof_vel = local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
        dof_vels.append(frame_dof_vel)

    dof_vels.append(dof_vels[-1])
    dof_vels = torch.stack(dof_vels, dim=0).view(num_frames, -1, 3)

    return dof_vels


class DeviceCache:

    def __init__(self, obj, device):
        self.obj = obj
        self.device = device

        keys = dir(obj)
        num_added = 0
        for k in keys:
            try:
                out = getattr(obj, k)
            except:
                # print("Error for key=", k)
                continue

            if isinstance(out, torch.Tensor):
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1
            elif isinstance(out, np.ndarray):
                out = torch.tensor(out)
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1

        # print("Total added", num_added)

    def __getattr__(self, string):
        out = getattr(self.obj, string)
        return out

class MotionlibMode(Enum):
    file = 1
    directory = 2
    
class MotionLibFromTracking(MotionLibSMPL):

    def __init__(self, motion_lib_cfg):
        self.m_cfg = motion_lib_cfg
        self._device = self.m_cfg.device
        
        self.mesh_parsers = None
        
        self.load_data(self.m_cfg.motion_file,  min_length = self.m_cfg.min_length, im_eval = self.m_cfg.im_eval, object_type=self.m_cfg.object_type, keyword=self.m_cfg.keyword)
        self.setup_constants(fix_height = self.m_cfg.fix_height,  multi_thread = self.m_cfg.multi_thread)

        if flags.real_traj:
            self.track_idx = self._motion_data_load[next(iter(self._motion_data_load))].get("track_idx", [19, 24, 29])
        return

    def load_data(self, motion_file,  min_length=-1, im_eval=False, object_type="all", keyword=None):
        if osp.isfile(motion_file):
            self.mode = MotionlibMode.file
            load_data= joblib.load(motion_file)
        else:
            self.mode = MotionlibMode.directory
            load_data = glob.glob(osp.join(motion_file, "*.pkl"))
        
        ## load successful sequences only
        load_keys = ["action_all", "root_state_all", "dof_state_all", "pred_pos_all"]
        succ_id = load_data["succ_idxes"]
        succ_names = load_data["success_keys"]

        self._motion_data_load = {}

        for i, name in zip(succ_id, succ_names):
            data = {k: load_data[k][i] for k in load_keys}
            self._motion_data_load[name] = data

        # import ipdb; ipdb.set_trace()
        self._motion_data_list = []
        self._motion_data_keys = []

        if self.mode == MotionlibMode.file:
            if min_length != -1:
                data_list = {k: v for k, v in list(self._motion_data_load.items()) if len(v['root_state_all']) >= min_length}
            elif im_eval:
                data_list = {item[0]: item[1] for item in sorted(self._motion_data_load.items(), key=lambda entry: len(entry[1]['root_state_all']), reverse=True)}
                # data_list = self._motion_data
            else:
                data_list = self._motion_data_load

            # import ipdb; ipdb.set_trace()

            for data_key in data_list:
                if "object_name" in data_list[data_key]:
                    if object_type == "all" or object_type == data_list[data_key]["object_name"]:
                        if keyword is not None and keyword in data_key:
                            self._motion_data_list.append(data_list[data_key])
                            self._motion_data_keys.append(data_key)
                        if keyword is None:
                            self._motion_data_list.append(data_list[data_key])
                            self._motion_data_keys.append(data_key)
                else:
                    if keyword is not None and keyword in data_key:
                        self._motion_data_list.append(data_list[data_key])
                        self._motion_data_keys.append(data_key)
                    if keyword is None:
                        # import ipdb; ipdb.set_trace()
                        self._motion_data_list.append(data_list[data_key])
                        self._motion_data_keys.append(data_key)
            # self._motion_data_list = np.array(list(data_list.values()))
            # self._motion_data_keys = np.array(list(data_list.keys()))
            self._motion_data_list = np.array(self._motion_data_list)
            self._motion_data_keys = np.array(self._motion_data_keys)
        else:
            self._motion_data_list = np.array(self._motion_data_load)
            self._motion_data_keys = np.array(self._motion_data_load)
        
        self._num_unique_motions = len(self._motion_data_list)
        if self.mode == MotionlibMode.directory:
            self._motion_data_load = joblib.load(self._motion_data_load[0]) # set self._motion_data_load to a sample of the data 

    def setup_constants(self, fix_height = FixHeightMode.full_fix, multi_thread = True):
        self.fix_height = fix_height
        self.multi_thread = multi_thread
        
        #### Termination history
        self._curr_motion_ids = None
        self._termination_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._success_rate = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_prob = torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions  # For use in sampling batches
        self._sampling_batch_prob = None  # For use in sampling within batches
        
        
    def load_motions(self, skeleton_trees, gender_betas, limb_weights, random_sample=True, start_idx=0, max_len=-1):
        # load motion load the same number of motions as there are skeletons (humanoids)
        if "gts" in self.__dict__:
            del self.gts, self.grs, self.lrs, self.grvs, self.gravs, self.gavs, self.gvs, self.dvs, self.o_gts, self.o_grs, self.o_names, self.t_gts, self.t_grs
            del self._motion_lengths, self._motion_fps, self._motion_dt, self._motion_num_frames, self._motion_bodies, self._motion_aa
            if flags.real_traj:
                del self.q_gts, self.q_grs, self.q_gavs, self.q_gvs

        motions = []
        self._motion_lengths = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_bodies = []
        self._motion_aa = []

        if flags.real_traj:
            self.q_gts, self.q_grs, self.q_gavs, self.q_gvs = [], [], [], []

        self.o_gts, self.o_grs, self.o_names, self.t_gts, self.t_grs = [], [], [], [], []

        total_len = 0.0
        self.num_joints = len(skeleton_trees[0].node_names)
        num_motion_to_load = len(skeleton_trees)

        if random_sample:
            sample_idxes = torch.multinomial(self._sampling_prob, num_samples=num_motion_to_load, replacement=True).to(self._device)
        else:
            sample_idxes = torch.remainder(torch.arange(len(skeleton_trees)) + start_idx, self._num_unique_motions ).to(self._device)


        # import ipdb; ipdb.set_trace()
        self._curr_motion_ids = sample_idxes
        self.one_hot_motions = torch.nn.functional.one_hot(self._curr_motion_ids, num_classes = self._num_unique_motions).to(self._device)  # Testing for obs_v5
        self.curr_motion_keys = self._motion_data_keys[sample_idxes]
        self._sampling_batch_prob = self._sampling_prob[self._curr_motion_ids] / self._sampling_prob[self._curr_motion_ids].sum()

        print("\n****************************** Current motion keys ******************************")
        print("Sampling motion:", sample_idxes[:30])
        if len(self.curr_motion_keys) < 100:
            print(self.curr_motion_keys)
        else:
            print(self.curr_motion_keys[:30], ".....")
        print("*********************************************************************************\n")


        motion_data_list = self._motion_data_list[sample_idxes.cpu().numpy()]
        # import ipdb; ipdb.set_trace()
        mp.set_sharing_strategy('file_descriptor')
        # mp.set_sharing_strategy('file_system')

        manager = mp.Manager()
        queue = manager.Queue()
        num_jobs = min(mp.cpu_count(), 64)
        num_jobs = 1
        # print("number of jobs:", num_jobs)

        if num_jobs <= 8 or not self.multi_thread:
            num_jobs = 1
        if flags.debug:
            num_jobs = 1
        
        res_acc = {}  # using dictionary ensures order of the results.
        jobs = motion_data_list
        # print("len of motions", len(jobs))
        chunk = np.ceil(len(jobs) / num_jobs).astype(int)
        ids = np.arange(len(jobs))

        jobs = [(ids[i:i + chunk], jobs[i:i + chunk], skeleton_trees[i:i + chunk], gender_betas[i:i + chunk],  self.mesh_parsers, self.m_cfg) for i in range(0, len(jobs), chunk)]
        job_args = [jobs[i] for i in range(len(jobs))]
        # print("len of jobs", len(jobs))
        for i in range(1, len(jobs)):
            worker_args = (*job_args[i], queue, i)
            worker = mp.Process(target=self.load_motion_with_skeleton, args=worker_args)
            worker.start()
        res_acc.update(self.load_motion_with_skeleton(*jobs[0], None, 0))
        # print('first done')

        for i in tqdm(range(len(jobs) - 1)):
            res = queue.get()
            # print(i, "done")
            res_acc.update(res)

        for f in tqdm(range(len(res_acc))):
            motion_file_data, curr_motion = res_acc[f]
            if USE_CACHE:
                curr_motion = DeviceCache(curr_motion, self._device)

            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps

            num_frames = curr_motion.tensor.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            
            # print(self.num_joints)
            if "beta" in motion_file_data:
                self._motion_aa.append(motion_file_data['pose_aa'].reshape(-1, self.num_joints * 3))
                self._motion_bodies.append(curr_motion.gender_beta)
            else:
                self._motion_aa.append(np.zeros((num_frames, self.num_joints * 3)))
                self._motion_bodies.append(torch.zeros(17))

            # if "object_names" in motion_file_data:
                # self.o_gts.append(torch.tensor(motion_file_data["object_trans"]))
                # self.o_grs.append(torch.tensor(motion_file_data["object_quat"]))
                # self.o_names.append(motion_file_data["object_name"])

            # if "table_trans" in motion_file_data:
            #     self.t_gts.append(torch.tensor(motion_file_data["table_trans"]))
            #     self.t_grs.append(torch.tensor(motion_file_data["table_quat"]))

            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            self._motion_lengths.append(curr_len)
            
            if flags.real_traj:
                self.q_gts.append(curr_motion.quest_motion['quest_trans'])
                self.q_grs.append(curr_motion.quest_motion['quest_rot'])
                self.q_gavs.append(curr_motion.quest_motion['global_angular_vel'])
                self.q_gvs.append(curr_motion.quest_motion['linear_vel'])
                
            del curr_motion
            
        self._motion_lengths = torch.tensor(self._motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_fps = torch.tensor(self._motion_fps, device=self._device, dtype=torch.float32)
        self._motion_bodies = torch.stack(self._motion_bodies).to(self._device).type(torch.float32)
        self._motion_aa = torch.tensor(np.concatenate(self._motion_aa), device=self._device, dtype=torch.float32)

        self._motion_dt = torch.tensor(self._motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, device=self._device)
        self._motion_limb_weights = torch.tensor(np.array(limb_weights), device=self._device, dtype=torch.float32)
        self._num_motions = len(motions)

        # self.o_gts = torch.stack(self.o_gts).float().to(self._device).type(torch.float32)
        # self.o_grs = torch.stack(self.o_grs).float().to(self._device).type(torch.float32)
        # self.t_gts = torch.stack(self.t_gts).float().to(self._device).type(torch.float32)
        # self.t_grs = torch.stack(self.t_grs).float().to(self._device).type(torch.float32)
        # noise_mask = torch.zeros((1, 52, 3))
        # noise_mask[:, -18:, :-1] = 1  ## arm
        # noise_mask[:, 15:18+15, :-1] = 1

        # noise_mask[:, -15:, :-1] = 1  ## hand
        # noise_mask[:, 18:18+15, :-1] = 1

        # self.gts = torch.cat([m.global_translation + 0.05 * noise_mask * torch.ones_like(m.global_translation) for m in motions], dim=0).float().to(self._device)
        self.gts = torch.cat([m.global_translation for m in motions], dim=0).float().to(self._device)
        self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float().to(self._device)
        self.lrs = torch.cat([m.local_rotation for m in motions], dim=0).float().to(self._device)
        self.grvs = torch.cat([m.global_root_velocity for m in motions], dim=0).float().to(self._device)
        self.gravs = torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gavs = torch.cat([m.global_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gvs = torch.cat([m.global_velocity for m in motions], dim=0).float().to(self._device)
        self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float().to(self._device)

        if "action_all" in motion_file_data:
            self.gta = torch.cat([m.gt_action for m in motions], dim=0).float().to(self._device)


        noise_mask = torch.ones((1, 3))
        noise_mask[:, -1] = 0
        if "object_trans" in motion_file_data:
            self.o_gts = torch.cat([m.object_trans + 0.00 * noise_mask * torch.ones_like(m.object_trans) for m in motions], dim=0).float().to(self._device)
            self.o_grs = torch.cat([m.object_quat for m in motions], dim=0).float().to(self._device)
            self.o_names = [m.object_name for m in motions]
            self.t_gts = torch.cat([m.table_trans for m in motions], dim=0).float().to(self._device)
            self.t_grs = torch.cat([m.table_quat for m in motions], dim=0).float().to(self._device)

        else:
            self.o_names = [None for m in motions]
            self.o_gts = torch.zeros((self.lrs.shape[0], 3)).to(self._device)
            self.o_grs = torch.zeros((self.lrs.shape[0], 4)).to(self._device)
            self.t_gts = torch.zeros((self.lrs.shape[0], 3)).to(self._device)
            self.t_grs = torch.zeros((self.lrs.shape[0], 4)).to(self._device)

        if "object_contact" in motion_file_data:
            self.o_contact = torch.cat([m.object_contact for m in motions], dim=0).to(self._device)
            self.b_contact = torch.cat([m.body_contact for m in motions], dim=0).to(self._device)
        else:
            self.o_contact = torch.zeros((self.lrs.shape[0], 1)).to(self._device)
            self.b_contact = torch.zeros((self.lrs.shape[0], 52)).to(self._device)
            # self.o_contact = torch.zeros_like(self.gts).to(self._device)  ## bug
            # self.b_contact = torch.zeros_like(self.gts).to(self._device)  ## bug

        if flags.real_traj:
            self.q_gts = torch.cat(self.q_gts, dim=0).float().to(self._device)
            self.q_grs = torch.cat(self.q_grs, dim=0).float().to(self._device)
            self.q_gavs = torch.cat(self.q_gavs, dim=0).float().to(self._device)
            self.q_gvs = torch.cat(self.q_gvs, dim=0).float().to(self._device)

        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        self.motion_ids = torch.arange(len(motions), dtype=torch.long, device=self._device)
        motion = motions[0]
        self.num_bodies = motion.num_joints

        num_motions = self.num_motions()
        total_len = self.get_total_length()
        print(f"Loaded {num_motions:d} motions with a total length of {total_len:.3f}s and {self.gts.shape[0]} frames.")
        return motions

    @staticmethod
    def load_motion_with_skeleton(ids, motion_data_list, skeleton_trees, shape_params, mesh_parsers, config, queue, pid):
        # ZL: loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
        max_len = config.max_length
        fix_height = config.fix_height
        np.random.seed(np.random.randint(5000)* pid)
        res = {}
        assert (len(ids) == len(motion_data_list))
        for f in range(len(motion_data_list)):
            curr_id = ids[f]  # id for this datasample
            curr_file = motion_data_list[f]
            if not isinstance(curr_file, dict) and osp.isfile(curr_file):
                key = motion_data_list[f].split("/")[-1].split(".")[0]
                curr_file = joblib.load(curr_file)[key]
            curr_gender_beta = shape_params[f]

            seq_len = curr_file['root_state_all'].shape[0]
            if max_len == -1 or seq_len < max_len:
                start, end = 0, seq_len
            else:
                start = random.randint(0, seq_len - max_len)
                end = start + max_len

            # ## for debug
            # start, end = 0, 32

            trans = curr_file['root_state_all'][start:end, :3]
            root_rot_quat = curr_file['root_state_all'][start:end, 3:7]
            root_rot = sRot.from_quat(root_rot_quat).as_rotvec()[:, None]

            dof_pos = curr_file["dof_state_all"][start:end, :, 0].reshape(end-start, -1, 3)

            
            pose_aa = np.concatenate([root_rot, dof_pos], axis=1)
            pose_quat = sRot.from_rotvec(pose_aa.reshape(-1, 3)).as_quat().reshape(end-start, -1, 4)

            trans = to_torch(trans)
            pose_aa = to_torch(pose_aa)
            pose_quat = to_torch(pose_quat)

            # pose_aa = to_torch(curr_file['pose_aa'][start:end])

            # pose_quat_global = curr_file['pose_quat_global'][start:end]

            B, J, N = pose_quat.shape

            # ##### ZL: randomize the heading ######
            # if (not flags.im_eval) and (not flags.test):
            #     # if True:
            #     random_rot = np.zeros(3)
            #     random_rot[2] = np.pi * (2 * np.random.random() - 1.0)
            #     random_heading_rot = sRot.from_euler("xyz", random_rot)
            #     pose_aa[:, :3] = torch.tensor((random_heading_rot * sRot.from_rotvec(pose_aa[:, :3])).as_rotvec())
            #     pose_quat_global = (random_heading_rot * sRot.from_quat(pose_quat_global.reshape(-1, 4))).as_quat().reshape(B, J, N)
            #     trans = torch.matmul(trans.float(), torch.from_numpy(random_heading_rot.as_matrix().T).float())
            # ##### ZL: randomize the heading ######

            # if not mesh_parsers is None:
            #     trans, trans_fix = MotionLibSMPLX.fix_trans_height(pose_aa, trans, curr_gender_beta, mesh_parsers, fix_height_mode = fix_height)
            # else:
            #     trans_fix = 0

            trans_fix = 0

            # pose_quat_global = to_torch(pose_quat_global)

            # ### add noise to the rotation
            # noise_mask = torch.zeros((pose_quat_global.shape[0], 52))
            # # noise_mask[:, -18:, :-1] = 1  ## arm
            # # noise_mask[:, 15:18+15, :-1] = 1
            # noise_mask[:, -15:] = 1  ## hand
            # noise_mask[:, 18:18+15] = 1
            # pose_quat_global = add_rotation_noise_batch(pose_quat_global, noise_level=0.5, mask=noise_mask)
            # pose_quat_global = to_torch(pose_quat_global)


            # import ipdb; ipdb.set_trace()

            sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_trees[f], pose_quat, trans, is_local=True)

            curr_motion = SkeletonMotion.from_skeleton_state(sk_state, curr_file.get("fps", 30))
            curr_dof_vels = compute_motion_dof_vels(curr_motion)
            
            # if flags.real_traj:
            #     quest_sensor_data = to_torch(curr_file['quest_sensor_data'])
            #     quest_trans = quest_sensor_data[..., :3]
            #     quest_rot = quest_sensor_data[..., 3:]
                
            #     quest_trans[..., -1] -= trans_fix # Fix trans
                
            #     global_angular_vel = SkeletonMotion._compute_angular_velocity(quest_rot, time_delta=1 / curr_file['fps'])
            #     linear_vel = SkeletonMotion._compute_velocity(quest_trans, time_delta=1 / curr_file['fps'])
            #     quest_motion = {"global_angular_vel": global_angular_vel, "linear_vel": linear_vel, "quest_trans": quest_trans, "quest_rot": quest_rot}
            #     curr_motion.quest_motion = quest_motion

            curr_motion.dof_vels = curr_dof_vels
            curr_motion.gender_beta = curr_gender_beta

            if "action_all" in curr_file:
                curr_motion.gt_action = to_torch(curr_file["action_all"])

            # import ipdb; ipdb.set_trace()

            # if "object_trans" in curr_file.keys():
            #     curr_motion.object_trans = to_torch(object_trans)
            #     curr_motion.object_quat = to_torch(object_quat)
            #     curr_motion.object_name = curr_file["object_name"]
            # if "table_trans" in curr_file.keys():
            #     curr_motion.table_trans = to_torch(table_trans)
            #     curr_motion.table_quat = to_torch(table_quat)
            # if "object_contact" in curr_file.keys():
            #     curr_motion.object_contact = to_torch(object_contact)
            #     curr_motion.body_contact = to_torch(body_contact)
            res[curr_id] = (curr_file, curr_motion)


        if not queue is None:
            queue.put(res)
        else:
            return res
        
    def get_motion_state(self, motion_ids, motion_times, offset=None):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        # print("non_interval", frame_idx0, frame_idx1, blend)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        local_rot0 = self.lrs[f0l]
        local_rot1 = self.lrs[f1l]

        body_vel0 = self.gvs[f0l]
        body_vel1 = self.gvs[f1l]

        body_ang_vel0 = self.gavs[f0l]
        body_ang_vel1 = self.gavs[f1l]

        # import ipdb; ipdb.set_trace()
        o_pos0 = self.o_gts[f0l]
        o_pos1 = self.o_gts[f1l]
        o_rot0 = self.o_grs[f0l]
        o_rot1 = self.o_grs[f1l]

        t_pos0 = self.t_gts[f0l]
        t_pos1 = self.t_gts[f1l]
        t_rot0 = self.t_grs[f0l]
        t_rot1 = self.t_grs[f1l]

        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        vals = [local_rot0, local_rot1, body_vel0, body_vel1, body_ang_vel0, body_ang_vel1, rg_pos0, rg_pos1, dof_vel0, dof_vel1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)


        if offset is None:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1  # ZL: apply offset
            o_pos = (1.0 - blend_exp) * o_pos0[:, None] + blend_exp * o_pos1[:, None]  # ZL: apply offset
            t_pos = (1.0 - blend_exp) * t_pos0[:, None] + blend_exp * t_pos1[:, None]  # ZL: apply offset
        else:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1 + offset[..., None, :]  # ZL: apply offset
            o_pos = (1.0 - blend_exp) * o_pos0[:, None] + blend_exp * o_pos1[:, None] + offset[..., None, :]  # ZL: apply offset
            t_pos = (1.0 - blend_exp) * t_pos0[:, None] + blend_exp * t_pos1[:, None] + offset[..., None, :]  # ZL: apply offset
            
        o_pos = o_pos.squeeze(1)
        t_pos = t_pos.squeeze(1)

        body_vel = (1.0 - blend_exp) * body_vel0 + blend_exp * body_vel1
        body_ang_vel = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1
        dof_vel = (1.0 - blend_exp) * dof_vel0 + blend_exp * dof_vel1


        local_rot = torch_utils.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
        dof_pos = self._local_rotation_to_dof_smpl(local_rot)

        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]
        rb_rot = torch_utils.slerp(rb_rot0, rb_rot1, blend_exp)
        o_rot = torch_utils.slerp(o_rot0[:, None], o_rot1[:, None], blend_exp).squeeze(1)
        t_rot = torch_utils.slerp(t_rot0[:, None], t_rot1[:, None], blend_exp).squeeze(1)
        
        if flags.real_traj:
            q_body_ang_vel0, q_body_ang_vel1 = self.q_gavs[f0l], self.q_gavs[f1l]
            q_rb_rot0, q_rb_rot1 = self.q_grs[f0l], self.q_grs[f1l]
            q_rg_pos0, q_rg_pos1 = self.q_gts[f0l, :], self.q_gts[f1l, :]
            q_body_vel0, q_body_vel1 = self.q_gvs[f0l], self.q_gvs[f1l]

            q_ang_vel = (1.0 - blend_exp) * q_body_ang_vel0 + blend_exp * q_body_ang_vel1
            q_rb_rot = torch_utils.slerp(q_rb_rot0, q_rb_rot1, blend_exp)
            q_rg_pos = (1.0 - blend_exp) * q_rg_pos0 + blend_exp * q_rg_pos1
            q_body_vel = (1.0 - blend_exp) * q_body_vel0 + blend_exp * q_body_vel1
            
            rg_pos[:, self.track_idx] = q_rg_pos
            rb_rot[:, self.track_idx] = q_rb_rot
            body_vel[:, self.track_idx] = q_body_vel
            body_ang_vel[:, self.track_idx] = q_ang_vel

        o_contact = self.o_contact[f0l]
        b_contact = self.b_contact[f0l]

        # pos_noise = torch.randn_like(rg_pos) * 0.02
        # rg_pos += pos_noise

        # import ipdb; ipdb.set_trace()
        return {
            "root_pos": rg_pos[..., 0, :].clone(),
            "root_rot": rb_rot[..., 0, :].clone(),
            "dof_pos": dof_pos.clone(),
            "root_vel": body_vel[..., 0, :].clone(),
            "root_ang_vel": body_ang_vel[..., 0, :].clone(),
            "dof_vel": dof_vel.view(dof_vel.shape[0], -1),
            "motion_aa": self._motion_aa[f0l],
            "rg_pos": rg_pos,
            "rb_rot": rb_rot,
            "body_vel": body_vel,
            "body_ang_vel": body_ang_vel,
            "motion_bodies": self._motion_bodies[motion_ids],
            "motion_limb_weights": self._motion_limb_weights[motion_ids],
            "object_pos": o_pos.clone(),
            "object_rot": o_rot.clone(),
            "table_pos": t_pos.clone(),
            "table_rot": t_rot.clone(),
            "object_contact": o_contact.clone(),
            "body_contact": b_contact.clone(),
            "gt_action": self.gta[f0l].clone(),
        }