import time
import torch
import phc.env.tasks.humanoid_im as humanoid_im

from isaacgym.torch_utils import *
from phc.utils.flags import flags
from rl_games.algos_torch import torch_ext
import torch.nn as nn
from phc.learning.pnn import PNN
from collections import deque
from phc.learning.network_loader import load_mcp_mlp, load_pnn
from isaacgym import gymutil
from isaacgym import gymapi
from isaacgym import gymtorch
import random

class HumanoidImMCP(humanoid_im.HumanoidIm):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.num_prim = cfg["env"].get("num_prim", 3)
        self.discrete_mcp = cfg["env"].get("discrete_moe", False)
        self.has_pnn = cfg["env"].get("has_pnn", False)
        self.has_lateral = cfg["env"].get("has_lateral", False)
        self.z_activation = cfg["env"].get("z_activation", "relu")

        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)

        if self.has_pnn:
            assert (len(self.models_path) == 1)
            pnn_ck = torch_ext.load_checkpoint(self.models_path[0])
            self.pnn = load_pnn(pnn_ck, num_prim = self.num_prim, has_lateral = self.has_lateral, activation = self.z_activation, device = self.device)
            self.running_mean, self.running_var = pnn_ck['running_mean_std']['running_mean'], pnn_ck['running_mean_std']['running_var']
        
        self.fps = deque(maxlen=90)
        
        return

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        self._num_actions = self.num_prim
        return

    def get_task_obs_size_detail(self):
        task_obs_detail = super().get_task_obs_size_detail()
        task_obs_detail['num_prim'] = self.num_prim
        return task_obs_detail

    def get_action_from_weights(self, weights):
        with torch.no_grad():
            if weights.shape[-1] == 3:
                # Apply trained Model.
                curr_obs = ((self.obs_buf - self.running_mean.float()) / torch.sqrt(self.running_var.float() + 1e-05))
                
                curr_obs = torch.clamp(curr_obs, min=-5.0, max=5.0)
                if self.discrete_mcp:
                    max_idx = torch.argmax(weights, dim=1)
                    weights = torch.nn.functional.one_hot(max_idx, num_classes=self.num_prim).float()
                
                if self.has_pnn:
                    _, actions = self.pnn(curr_obs)
                    
                    x_all = torch.stack(actions, dim=1)
                else:
                    x_all = torch.stack([net(curr_obs) for net in self.actors], dim=1)
                # print(weights)
                actions = torch.sum(weights[:, :, None] * x_all, dim=1)

        return actions
    
    def post_physics_step(self):
        super().post_physics_step()

        # if flags.im_eval:
        #     motion_times = (self.progress_buf) * self.dt + self._motion_start_times + self._motion_start_times_offset  # already has time + 1, so don't need to + 1 to get the target for "this frame"
        #     motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids, motion_times, self._global_offset)  # pass in the env_ids such that the motion is in synced.
        #     body_pos = self._rigid_body_pos
        #     self.extras['mpjpe'] = (body_pos - motion_res['rg_pos']).norm(dim=-1).mean(dim=-1)
        #     self.extras['body_pos'] = body_pos.cpu().numpy()
        #     self.extras['body_pos_gt'] = motion_res['rg_pos'].cpu().numpy()


        # if flags.im_eval:
        if True:
            motion_times = (self.progress_buf) * self.dt + self._motion_start_times + self._motion_start_times_offset  # already has time + 1, so don't need to + 1 to get the target for "this frame"
            motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids, motion_times, self._global_offset)  # pass in the env_ids such that the motion is in synced.
            body_pos = self._rigid_body_pos
            dof_pos = self._dof_pos
            self.extras['self_obs'] = self.self_obs_buf.detach().clone().cpu().numpy()
            self.extras['full_obs_t'] = self.obs_buf_t.copy()
            self.extras['mpjpe'] = (body_pos - motion_res['rg_pos']).norm(dim=-1).mean(dim=-1)
            self.extras['body_pos'] = body_pos.cpu().numpy()
            self.extras['dof_pos'] = dof_pos.cpu().numpy()
            self.extras['body_pos_gt'] = motion_res['rg_pos'].cpu().numpy()
            self.extras['dof_pos_gt'] = motion_res['dof_pos'].cpu().numpy()
            self.extras['root_state'] = self._humanoid_root_states.cpu().numpy()
            self.extras['dof_state'] = self._dof_state.cpu().numpy().reshape((self.num_envs, -1, 2))
            try:
                self.extras['action'] = self.actions.clone()
                self.extras['noisy_action'] = self.noisy_actions.clone()
            except:
                pass

            self.extras['pd_tar'] = self.pd_tar
            self.obs_buf_t = self.obs_buf.detach().cpu().numpy()
        # if flags.test:
        #     """
        #     apply random external force
        #     """
        #     # if n % 20 == 0 or n % 21 == 0 or n % 22 == 0 or n % 23 == 0:
        #     print(self.progress_buf)
        #     apply_force = True
        #     forces = torch.zeros((1, self._rigid_body_state.shape[0], 3), device=self.device, dtype=torch.float)
        #     torques = torch.zeros((1, self._rigid_body_state.shape[0], 3), device=self.device, dtype=torch.float)
        #     external_joint_force = {
        #                             # "ROOT": (0, 1500),   # (dof_index, force)
        #                             # "CHEST": (11, 1000),
        #                             "L_ANKLE": (3, 3500),
        #                             "R_ANKLE": (7, 3500),
        #                             # "L_KNEE": (2, 2000),
        #                             # "R_KNEE": (6, 2000),
        #                             }
        #     sample_joint = random.sample(list(external_joint_force), 1)[0]
        #     print("applying external force to: {}".format(sample_joint))
        #     (idx, force) = external_joint_force[sample_joint]

        #     for i in range(self._rigid_body_state.shape[0] // self.num_bodies):
        #         forces[:, i * self.env.task.num_bodies + idx, :] = 1 * force * torch.clip(torch.randn(3), -1, 1)    ### CHEST
        #         # forces[:, i * self.num_bodies + idx, -1] = -1000    ### CHEST

        #     self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)

        return

    def step(self, weights):

        # if self.dr_randomizations.get('actions', None):
        #     actions = self.dr_randomizations['actions']['noise_lambda'](actions)
        # if flags.server_mode:
            # t_s = time.time()
        
        with torch.no_grad():
            if weights.shape[-1] == 3:
                # Apply trained Model.
                curr_obs = ((self.obs_buf - self.running_mean.float()) / torch.sqrt(self.running_var.float() + 1e-05))
                
                curr_obs = torch.clamp(curr_obs, min=-5.0, max=5.0)
                if self.discrete_mcp:
                    max_idx = torch.argmax(weights, dim=1)
                    weights = torch.nn.functional.one_hot(max_idx, num_classes=self.num_prim).float()
                
                if self.has_pnn:
                    _, actions = self.pnn(curr_obs)
                    
                    x_all = torch.stack(actions, dim=1)
                else:
                    x_all = torch.stack([net(curr_obs) for net in self.actors], dim=1)
                # print(weights)
                self.actions = torch.sum(weights[:, :, None] * x_all, dim=1)

                self.noisy_actions = self.actions + 0.00 * torch.randn_like(self.actions)

            else:
                # import ipdb; ipdb.set_trace()
                # print("step with action")
                self.actions = weights
                self.noisy_actions = self.actions# + 0.00 * torch.randn_like(self.actions)

            # ## debug:
            # self.action = torch.randn_like(weights)
            # self.noisy_actions = self.actions

            # print("step with:", self.noisy_actions[:, 0])
        # if weights.shape[-1] != 3:
        #     import ipdb; ipdb.set_trace()
        # actions = x_all[:, 3]  # Debugging
        # apply actions
        self.pre_physics_step(self.noisy_actions)

        # step physics and render each frame
        self._physics_step()

        # if apply_force:
        #     forces = torch.zeros((self.num_envs, self.num_bodies, 2, 3), device=self.device, dtype=torch.float)
        #     torques = torch.zeros((self.num_envs, self.num_bodies, 2, 3), device=self.device, dtype=torch.float)
        #     forces[:, 0, 0, 1] = 300
        #     forces[:, 0, 0, 0] = 300
        #     torques[:, 0, :, 2] = 0
        #     self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces.reshape(-1, 3)), None, gymapi.ENV_SPACE)
        #     # step the physics
        #     self.gym.simulate(self.sim)

        # gym.fetch_results(sim, True)

        # # update the viewer
        # gym.step_graphics(sim)
        # gym.draw_viewer(viewer, sim, True)

        # # Wait for dt to elapse in real time.
        # # This synchronizes the physics simulation with the rendering rate.
        # gym.sync_frame_time(sim)



        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()
        # if flags.server_mode:
        #     dt = time.time() - t_s
        #     print(f'\r {1/dt:.2f} fps', end='')
            
        # dt = time.time() - t_s
        # self.fps.append(1/dt)
        # print(f'\r {np.mean(self.fps):.2f} fps', end='')
        

        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)
