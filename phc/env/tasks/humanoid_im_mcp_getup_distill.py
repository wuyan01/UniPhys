

from typing import OrderedDict
import torch
import numpy as np
from phc.utils.torch_utils import quat_to_tan_norm
import phc.env.tasks.humanoid_im_getup as humanoid_im_getup
import phc.env.tasks.humanoid_im_mcp as humanoid_im_mcp
import phc.env.tasks.humanoid_im_mcp_getup as humanoid_im_mcp_getup
from phc.env.tasks.humanoid_amp import HumanoidAMP, remove_base_rot
from phc.utils.motion_lib_smpl import MotionLibSMPL 

from phc.utils import torch_utils

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from phc.utils.flags import flags
import joblib
import gc
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
from rl_games.algos_torch import torch_ext
import torch.nn as nn
from phc.learning.network_loader import load_mcp_mlp, load_pnn
from collections import deque
from phc.utils.motion_lib_from_tracking import MotionLibFromTracking
from easydict import EasyDict
from phc.utils.motion_lib_base import FixHeightMode

class HumanoidImMCPGetupDistill(humanoid_im_mcp_getup.HumanoidImMCPGetup):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
        return
    
    def _load_motion(self, motion_train_file, motion_test_file=[]):
        assert (self._dof_offsets[-1] == self.num_dof)

        if self.humanoid_type in ["smpl", "smplh", "smplx"]:
            motion_lib_cfg = EasyDict({
                "motion_file": motion_train_file,
                "device": torch.device("cpu"),
                "fix_height": FixHeightMode.full_fix,
                "min_length": self._min_motion_len,
                "max_length": -1,
                "im_eval": flags.im_eval,
                "multi_thread": True ,
                "smpl_type": self.humanoid_type,
                "randomrize_heading": True,
                "object_type": "all",
                "device": self.device,
                "keyword": None,
            })
            # print(self.humanoid_type)
            motion_eval_file = motion_train_file
            # self._motion_train_lib = MotionLibSMPL(motion_lib_cfg) if self.humanoid_type == "smpl" else MotionLibSMPLX(motion_lib_cfg)  ### todo: merge MotionLibSMPL and MotionLibSMPLX
            self._motion_train_lib = MotionLibFromTracking(motion_lib_cfg)  ### todo: merge MotionLibSMPL and MotionLibSMPLX
            motion_lib_cfg.im_eval = True
            # self._motion_eval_lib = MotionLibSMPL(motion_lib_cfg) if self.humanoid_type == "smpl" else MotionLibSMPLX(motion_lib_cfg)
            self._motion_eval_lib = MotionLibFromTracking(motion_lib_cfg)

            self._motion_lib = self._motion_train_lib
            self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=self.humanoid_shapes.cpu(), limb_weights=self.humanoid_limb_and_weights.cpu(), random_sample=(not flags.test) and (not self.seq_motions), max_len=-1 if flags.test else self.max_len)

        else:
            raise NotImplementedError

        return
    
    def get_action_from_motion_lib_cache(self, time_steps, offset=0):
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        # time_steps = 12
        B = env_ids.shape[0]
        time_internals = torch.arange(time_steps).to(self.device).repeat(B).view(-1, time_steps) * self._traj_sample_timestep
        motion_times_steps = ((self.progress_buf[env_ids, None] + offset) * self.dt + time_internals + self._motion_start_times[env_ids, None] + self._motion_start_times_offset[env_ids, None]).flatten()
        env_ids_steps = self._sampled_motion_ids[env_ids].repeat_interleave(time_steps)
        # import ipdb; ipdb.set_trace()
        motion_res = self._get_state_from_motionlib_cache(env_ids_steps, motion_times_steps, self._global_offset[env_ids].repeat_interleave(time_steps, dim=0).view(-1, 3))  # pass in the env_ids such that the motion is in synced.

        return motion_res["gt_action"]

