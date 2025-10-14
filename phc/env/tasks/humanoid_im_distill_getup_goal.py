# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from typing import OrderedDict
import torch
import numpy as np
from phc.utils.torch_utils import quat_to_tan_norm
import phc.env.tasks.humanoid_im_getup as humanoid_im_getup
import phc.env.tasks.humanoid_im_distill as humanoid_im_distill
import phc.env.tasks.humanoid_im_distill_getup as humanoid_im_distill_getup
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

class HumanoidImDistillGetupGoal(humanoid_im_distill_getup.HumanoidImDistillGetup):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._max_num_goals = 1
        self._num_goals = 0
        self.joint_idx = cfg.env.joint_idx  # [0: Pelvis; 18: L_Hand, 23: R_Hand, 13: Head]
        self.device_type = device_type
        self._goal_pos = None
        self.tar_min = cfg.env.tar_min
        self.tar_max = cfg.env.tar_max
        self.tar_min_height = cfg.env.tar_min_height
        self.tar_max_height = cfg.env.tar_max_height
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
        self._goal_pos = torch.zeros([self.num_envs, 3], dtype=torch.float, device="cuda")

        return

    def _build_marker(self, env_id, env_ptr):
        default_pose = gymapi.Transform()

        for i in range(self._max_num_goals):

            marker_handle = self.gym.create_actor(
                env_ptr,
                self._marker_asset,
                default_pose,
                "marker",
                self.num_envs + 10,
                0,
                0,
            )
            self.gym.set_rigid_body_color(
                env_ptr,
                marker_handle,
                0,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(1.0, 0.0, 0.0),
            )
            self._marker_handles[env_id].append(marker_handle)

    def _build_marker_state_tensors(self):
        num_actors = self.get_num_actors_per_env()
        
        self._marker_states = self._root_states.view(
            self.num_envs, num_actors, self._root_states.shape[-1]
        )[..., 1 : (1 + self._max_num_goals), :]
        self._marker_pos = self._marker_states[..., :3]

        self._marker_actor_ids = self._humanoid_actor_ids.unsqueeze(
            -1
        ) + torch_utils.to_torch(
            self._marker_handles, dtype=torch.int32, device=self.device
        )
        self._marker_actor_ids = self._marker_actor_ids.flatten()


    ###############################################################
    # Helpers
    ###############################################################
    def _load_marker_asset(self):
        asset_root = "phc/data/assets/urdf/"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_angular_velocity = 0.0
        asset_options.density = 0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, asset_root, "traj_marker_large.urdf", asset_options)
        
        self._marker_asset_small = self.gym.load_asset(self.sim, asset_root, "traj_marker_small.urdf", asset_options)

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()

        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1] + 3.0, 5.0)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1], 0.0)

        # cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1], 10.0)
        # cam_target = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1], 1.0)
        if self.viewer:
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._humanoid_root_states[self.viewing_env_idx, 0:3].cpu().numpy()

        if self.viewer:
            cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
            cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        else:
            cam_pos = np.array([char_root_pos[0] + 2.5, char_root_pos[1] + 2.5, char_root_pos[2]])

        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], char_root_pos[2])
        # if np.abs(cam_pos[2] - char_root_pos[2]) > 5:
        cam_pos[2] = char_root_pos[2] + 0.5
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], char_root_pos[1] + cam_delta[1], cam_pos[2])

        self.gym.set_camera_location(self.recorder_camera_handle, self.envs[self.viewing_env_idx], new_cam_pos, new_cam_target)

        if flags.follow:
            self.start = True
        else:
            self.start = False

        if self.start:
            self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return


    def update_goal_state(self, goal_pos):
        """
        goal_pos: if sparse: [B, 3]; else [B, N, 3]
        """
        if goal_pos.ndim == 2:
            self._goal_pos = goal_pos.unsqueeze(1) # [B, 1, 3]
        self._num_goals = self._goal_pos.shape[1]
        assert self._num_goals <= self._max_num_goals

    def set_random_goal(self, x=None, y=None, z=None):
        n = self.num_envs
        # _tar_dist_min, _tar_dist_max = self.state_machine_conditions[task]['tar_dist_range']
        
        goal_pos = torch.zeros([n, 3], dtype=torch.float, device="cuda")

        if x is not None and y is not None and z is not None:
            goal_pos[:, 0] = x
            goal_pos[:, 1] = y
            goal_pos[:, 2] = z
        else:
            _tar_dist_min, _tar_dist_max = self.tar_min, self.tar_max
            _tar_height_min, _tar_height_max = self.tar_min_height, self.tar_max_height
            rand_dist = (_tar_dist_max - _tar_dist_min) * torch.rand([n], dtype=float, device=self.device_type) + _tar_dist_min
            rand_height = (_tar_height_max - _tar_height_min) * torch.rand([n], dtype=float, device=self.device_type) + _tar_height_min
            rand_theta = 0.5 * np.pi * torch.rand([n], dtype=float, device=self.device_type) + 0.5 * np.pi
            goal_pos[:, 0] = rand_dist * torch.cos(rand_theta) + self._humanoid_root_states[:, 0]
            goal_pos[:, 1] = rand_dist * torch.sin(rand_theta) + self._humanoid_root_states[:, 1]
            goal_pos[:, 2] = rand_height

        self.update_goal_state(goal_pos)

        return goal_pos


    def capture_new_goal_from_keyborad(self):
        print("\033[1mCurrent root position: {}\033[0m".format(self._root_states[self._humanoid_actor_ids][..., :3].cpu().numpy()))
        text = input("\033[34mEnter the text prompt: \033[0m")
        goal_position_xyz = input("\033[34mEnter goal xyz position separated by a space: \033[0m")
        new_goal_position = self._root_states[self._humanoid_actor_ids][..., :3].clone()
        try:
            goal_x, goal_y, goal_z = map(float, goal_position_xyz.split())
            new_goal_position[..., 0] = goal_x
            new_goal_position[..., 1] = goal_y
            new_goal_position[..., 2] = goal_z
            print("\033[32mSetting text command: {}\033[0m".format(text))
            print("\033[32mSetting the goal position as {}\033[0m".format(new_goal_position.cpu().numpy()))
            self.update_goal_state(new_goal_position)
        except ValueError:
            print("Invalid input. Please enter two numbers separated by a space. The goal position remains unchanged")
            print("\033[32mSetting the goal position as {}\033[0m".format(new_goal_position.cpu().numpy()))
            self.update_goal_state(new_goal_position)
        return text, new_goal_position

    def set_state(self, root_state, dof_state, goal_pos):
        self._root_states[self._humanoid_actor_ids, ...] = torch.from_numpy(root_state[...]).cuda()
        self._root_states[self._humanoid_actor_ids, 7:] = 0
        self._root_states[self._marker_actor_ids, :3] = torch.from_numpy(goal_pos[...]).cuda()
        
        self._dof_state[...] = torch.from_numpy(dof_state[...]).cuda()
        # self._dof_state[..., 1] = 0
        self._goal_pos[:] = torch.from_numpy(goal_pos).cuda()

        return

    def vis_step(self, root_state, dof_state, goal_pos):
        if not self.paused and self.enable_viewer_sync:

            for _ in range(2):
                self.set_state(root_state, dof_state, goal_pos)
                self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))
                self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self._dof_state))

                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.gym.refresh_rigid_body_state_tensor(self.sim)


                if self.device == 'cpu':
                    self.gym.fetch_results(self.sim, True)
            
                self.render()

        root_state_diff = self._root_states[self._humanoid_actor_ids, ...] - torch.from_numpy(root_state[...]).cuda()
        dof_state_diff = self._dof_state - torch.from_numpy(dof_state[...]).cuda()
        print(root_state_diff[..., :7].mean(), dof_state_diff[:, 0].mean())

        return

    def _update_marker(self):

        if self._goal_pos is not None:
            self._marker_pos[...] = self._goal_pos

            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._root_states),
                gymtorch.unwrap_tensor(self._marker_actor_ids),
                len(self._marker_actor_ids),
            )

    def _draw_task(self):
        # uncomment it if you want to see the marker
        self._update_marker()
        return

    def reset(self, env_ids=None):
        super().reset(env_ids=env_ids)
        return
        

    def draw_task(self):
        self._update_marker()

