import joblib
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from tqdm import tqdm

from uniphys.utils.clip_utils import load_and_freeze_clip
from uniphys.utils.motion_repr_utils import (
    cano_goal,
    recover_global_joints_position,
    recover_global_root_state,
)

from .df_humanoid import DiffusionForcingHumanoid

printed_messages = set()

def print_once(message):
    if message not in printed_messages:
        print(message)
        printed_messages.add(message)

mujoco_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']


class DiffusionForcingHumanoidGoal(DiffusionForcingHumanoid):
    """
    A video prediction algorithm using Diffusion Forcing.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.pred_dof_pos, self.pred_dof_pos_all = [], []
        self.root_state, self.root_state_all = [], []
        self.dof_state, self.dof_state_all = [], []
        self.action, self.action_all = [], []
        self.zs, self.zs_all = [], []
        self.pred_pos, self.pred_pos_all = [], []
        self.goal_positions_all = []
        self.task_complete_memory = []
        self.curr_steps = 0


    def post_step(self, info, done, save_dir=None, max_episode_length=None):
        if max_episode_length is None:
            max_episode_length = self.max_episode_length

        if save_dir is not None:
            self.pred_pos_all.append(info["body_pos"])
            self.pred_dof_pos_all.append(info["dof_pos"])
            self.root_state_all.append(info["root_state"])
            self.dof_state_all.append(info["dof_state"])

            self.action_all.append(info["action"].clone().detach().cpu().numpy())
            self.goal_positions_all.append(self.global_goal_position.clone().detach().cpu().numpy())

            if "z" in info: 
                self.zs_all.append(info["z"])

        done = done | (self.env.task.progress_buf >= self.max_episode_length) # timeout

        return done
    
    def _compute_goal_cost(
        self, 
        global_pos,          # [n, t, b, 3]
        global_rotmat,       # [n, t, b, 3, 3]
        global_goal_pos,     # [3]
        context_frames,      # int
        curr_pos_diff        # float
    ):
        """
        Compute position + orientation costs toward the global goal.

        Args:
            global_pos (Tensor): Predicted joint positions [n, t, b, 3].
            global_rotmat (Tensor): Predicted root rotations [n, t, b, 3, 3].
            global_goal_pos (Tensor): Target global goal position [3].
            context_frames (int): Number of context frames.
            curr_pos_diff (float): Current distance between controlled joint and goal.

        Returns:
            pos_cost (Tensor): Position cost [n, b].
            orientation_cost (Tensor): Orientation cost [n, b].
            pos_cost_scale (float): Position scaling factor.
            orientation_cost_scale (float): Orientation scaling factor.
        """
        # === Position cost ===
        pos_cost = F.l1_loss(
            global_pos[:, context_frames:], global_goal_pos, reduction="none"
        ).mean(-1).mean(1)  # [n, b]

        # === Orientation cost ===
        forward_vector = global_rotmat[:, context_frames:][..., 0]        # forward dir = first column
        xy_vector = F.normalize(forward_vector[..., :2].cuda(), dim=-1)   # project to XY plane

        goal_vector = global_goal_pos - global_pos[:, context_frames:]
        goal_vector_xy = F.normalize(goal_vector[..., :2], dim=-1)

        cos_theta = torch.sum(xy_vector * goal_vector_xy, dim=-1)  # [n, t, b]
        orientation_cost = (1 - cos_theta).mean(1)                 # [n, b]

        # === Scaling factors ===
        d = 3
        pos_cost_scale = max(d / (d + curr_pos_diff), 0.25)
        orientation_cost_scale = curr_pos_diff / (d + curr_pos_diff)

        return pos_cost, orientation_cost, pos_cost_scale, orientation_cost_scale
    
    def goal_guidance_fn(self, xs_pred=None, verbose=False):
        """
        Guidance loss during sampling: encourages motion toward goal position 
        with correct orientation.
        """

        # === Reshape predictions ===
        xs_pred = rearrange(xs_pred, "t b (fs c) -> (t fs) b c", fs=self.frame_stack).contiguous()
        xs_pred = rearrange(xs_pred, "t (n b) c -> n t b c", n=self.n_samples).contiguous()

        # === Recover state prediction ===
        state_pred = xs_pred[..., self.action_dim:]
        state_pred = (
            state_pred * torch.from_numpy(self.state_std).cuda() 
            + torch.from_numpy(self.state_mean).cuda()
        )

        # === Recover root and joints ===
        global_root_pos, _, global_root_rotmat = recover_global_root_state(
            local_pos=state_pred[..., :3],
            local_rot6d=state_pred[..., 3:9],
            local_vel=state_pred[..., 9:12],
            cano_transf_rotmat=self.cano_transf_rotmat,
            cano_transf_transl=self.cano_transf_transl,
        )
        local_joints_pred = state_pred[..., 15:15+24*3].reshape(*state_pred.shape[:3], -1, 3)
        global_joints_position = recover_global_joints_position(local_joints_pred, global_root_pos, global_root_rotmat)

        global_goal_pos = self.global_goal_position.cuda()
        idx = self.env.task.joint_idx
        print_once(f"Control {mujoco_joint_names[idx]}")

        curr_pos_diff = torch.norm(
            self.env.task._rigid_body_pos[:, idx].clone() - self.global_goal_position
        )

        # === Compute costs ===
        pos_cost, orientation_cost, pos_cost_scale, orientation_cost_scale = self._compute_goal_cost(
            global_pos=global_joints_position[:, :, :, idx],
            global_rotmat=global_root_rotmat,
            global_goal_pos=global_goal_pos,
            context_frames=self.context_frames,
            curr_pos_diff=curr_pos_diff,
        )

        ## [B, N, 3] -> [B]

        # === Final guidance loss ===
        loss = -(pos_cost * pos_cost_scale + orientation_cost * orientation_cost_scale).mean()  ## to fix: batch-wise guidance loss

        if verbose:
            print(loss, pos_cost.mean(), orientation_cost.mean(), curr_pos_diff)

        return loss * 15

    def post_select_cost_fn(self, state_pred):
        """
        Post-hoc cost for evaluating sampled trajectories.
        """

        # === Recover root and joints ===
        global_root_pos, _, global_root_rotmat = recover_global_root_state(
            local_pos=state_pred[..., :3],
            local_rot6d=state_pred[..., 3:9],
            local_vel=state_pred[..., 9:12],
            cano_transf_rotmat=self.cano_transf_rotmat,
            cano_transf_transl=self.cano_transf_transl,
        )
        local_joints_pred = state_pred[..., 15:15+24*3].reshape(*state_pred.shape[:3], -1, 3)
        global_joints_position = recover_global_joints_position(local_joints_pred, global_root_pos, global_root_rotmat)

        global_goal_pos = self.global_goal_position.cuda()
        idx = self.env.task.joint_idx
        print_once(f"Control {mujoco_joint_names[idx]}")

        curr_pos_diff = torch.norm(
            self.env.task._rigid_body_pos[:, idx].clone() - self.global_goal_position
        )

        # === Compute costs ===
        pos_cost, orientation_cost, pos_cost_scale, orientation_cost_scale = self._compute_goal_cost(
            global_pos=global_joints_position[:, :, :, idx],
            global_rotmat=global_root_rotmat,
            global_goal_pos=global_goal_pos,
            context_frames=self.context_frames,
            curr_pos_diff=curr_pos_diff,
        )

        return pos_cost * pos_cost_scale + orientation_cost * orientation_cost_scale


    @torch.no_grad()
    def interact(self, save=False):
        """
        Main interaction loop: runs episodes, samples trajectories, executes actions,
        checks goal completion, and logs results.
        """
        # === Environment setup ===
        self.env.task._termination_distances[:] = 1e6
        self.env.task.termination_mode = "sampling"
        num_envs = self.env.task.num_envs

        # === Guidance & text setup ===
        self.n_samples = 20  # Monte Carlo samples
        self.guidance_params = 0
        self.guidance_fn = self.goal_guidance_fn
        text = "walk"
        self.clip_model = load_and_freeze_clip("ViT-B/32", device="cuda")
        self.text_embedding = torch.zeros((1, 512)) 

        # === Buffers ===
        self._clear_hist_buffer()

        # === Action Mean and Std stats ===
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
        task_complete, episode_length_list = [], []
        self.env.task.completed_episode_lengths = []

        self.max_episode_length = self.env.task.max_episode_length
        self.max_episode = 50
        max_steps = (self.max_episode + 1) * self.max_episode_length

        # === Loop ===
        t, e, count, reset_flag = 0, 0, 0, False
        sequential_goal = False # True - reset env after each goal; False - keep going after reaching each goal
        break_flag = False

        ## === Save dir ===
        if save:
            import os
            save_dir = os.path.join(self.save_dir, "goal_reaching", "step_{}_exec_{}_noise_{}_n_{}_sample_{}_dist_{}".format(self.resume_step, self.exec_step, self.diffusion_model.stabilization_level, self.cfg.diffusion.sampling_timesteps, self.n_samples, self.env.task.tar_max))
            if not self.cfg.skip_text and self.guidance_params > 0:
                save_dir += "_{}".format(text)
                save_dir += "_g{}".format(self.cfg.guidance_params)
            else:
                save_dir += "_uncond"
            if sequential_goal:
                save_dir += "_sequential_goal"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        else:
            save_dir = None

        """
        Interact starts
        """
        while t < max_steps and e < self.max_episode:
            # --- Episode init ---
            done_indices = []
            self.env.task.set_char_color([0, 0.5, 0.8], env_ids=torch.arange(num_envs))

            self.global_goal_position = self.env.task.set_random_goal()

            n = 0

            """
            Reset + initializing buffer
            """
            if e == 0 or not sequential_goal or reset_flag:
                reset_flag = False
                obs_dict = self.player.env_reset()
                self.global_goal_position = self.env.task.set_random_goal()
                self.pbar = tqdm(range(self.max_episode_length))
                self._clear_hist_buffer()

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

                    t += 1
                    n += 1

            while n < self.max_episode_length:
                self.env.task.set_char_color(col=[1, 0.5, 0.0], env_ids=torch.arange(self.env.task.num_envs))  ## orange
                """
                Diffusion with Monte Carlo Guidance
                """
                with torch.no_grad():
                    """
                    Canonicalize the goal to the current frame
                    """
                    joint_position = self.joint_pos_buffer.get().clone().cpu().numpy()[-self.context_frames:]
                    _, self.cano_transf_rotmat, self.cano_transf_transl = cano_goal(joint_position.transpose(1, 0, 2, 3), self.global_goal_position)

                    """
                    Denoiseing diffusion to sample action sequence
                    """
                    action_pred, state_pred = self.policy()

                    """
                    if MONTE CARLO GUIDANCE, choose the best trajectory among n_samples
                    """

                    if self.n_samples > 1:  
                        cost = self.post_select_cost_fn(state_pred)
                        opt_i = cost.argmin(0) 

                        action_pred = action_pred[opt_i, :, torch.arange(action_pred.shape[2])].permute(1, 0, 2)
                        state_pred = state_pred[opt_i, :, torch.arange(state_pred.shape[2])].permute(1, 0, 2)

                    action_exec = action_pred[self.H:].permute(1, 0, 2).float()

                    """
                    interact with the env
                    """

                    for i in range(self.exec_step):
                        a = action_exec[:, i]

                        if self.cfg.action_keys == ["z"]:
                            with torch.no_grad():
                                input_dict = {"obs": obs_dict}
                                a_exec, _ = self.player.dec_action(a, input_dict)

                        else:
                            a_exec = a

                        obs_dict, r, done, info = self.player.env_step(self.env, a_exec)
        

                        self.root_state_buffer.add(self.env.task._root_states[self.env.task._humanoid_actor_ids].clone())
                        self.dof_state_buffer.add(self.env.task._dof_state.reshape(num_envs, -1, 2).clone())
                        self.joint_pos_buffer.add(self.env.task._rigid_body_pos.clone())
                        self.action_buffer.add(a.clone())  ## "z" or "action"

                        t += 1
                        n += 1

                        """
                        Check if goal reached
                        Succ criteria: within 0.3m for 5 consecutive steps
                        """
                        if done.any():
                            break_flag = True
                            # flags.prompt = True
                            print()
                            print("\033[31mFailed! Falled. \033[0m")
                            reset_flag = True
                            task_complete.append(0)
                            episode_length_list.append(n)
                            break

                        distance = torch.norm(self.env.task._rigid_body_pos[:, self.env.task.joint_idx] - self.global_goal_position)
                        if distance < 1:
                            self.exec_step = 4 # Replan more frequently when close to goal
                        else:
                            self.exec_step = 8

                        # succ criteria: within 0.3m for 5 consecutive steps
                        if distance < 0.3:
                            count += 1
                            if count > 5:
                                count = 0
                                done[:] = 1
                                print()
                                print("\033[31mGoal reached!\033[0m")  # Red text
                                break_flag = True
                                task_complete.append(1)
                                episode_length_list.append(n)
                                break

                        done = self.post_step(info, done.clone(), save_dir=save_dir)

                        self.success_rate = sum(task_complete) / len(task_complete) if len(task_complete) > 0 else 0
                        
                        self.pbar.update(1)
                        self.pbar.refresh()
                        update_str = f"Episode: {e+1}/{self.max_episode} | Steps {n}/{self.max_episode_length} | Distance {distance:.3f} | Succ rate: {self.success_rate:.3f}"
                        self.pbar.set_description(update_str)
                    
                    if break_flag:
                        n = 0
                        e += 1
                        done[:] = 0
                        break
            if not break_flag:
                # flags.prompt = True
                print()
                print("\033[31mFailed! Time out.\033[0m")
                reset_flag = True
                task_complete.append(0)
                episode_length_list.append(n)
                n = 0
                e += 1
            break_flag = False

        self.pred_pos_all = np.stack(self.pred_pos_all).squeeze()
        self.pred_dof_pos_all = np.stack(self.pred_dof_pos_all).squeeze()
        self.root_state_all = np.stack(self.root_state_all).squeeze()
        self.dof_state_all = np.stack(self.dof_state_all).squeeze()
        self.action_all = np.stack(self.action_all).squeeze()
        self.zs_all = np.stack(self.zs_all).squeeze()
        self.goal_positions_all = np.stack(self.goal_positions_all).squeeze()

        save_info = {"body_pos": self.pred_pos_all, 
                     "dof_pos": self.pred_dof_pos_all, 
                     "root_state": self.root_state_all, 
                     "dof_state": self.dof_state_all, 
                     "action": self.action_all, 
                     "z": self.zs_all, 
                     "goal_position": self.goal_positions_all,
                     "success_rate": self.success_rate,
                     "task_complete": task_complete,
                     "episode_length": episode_length_list
                     }
        
        joblib.dump(save_info, os.path.join(save_dir, "goal_reaching_e{}_succ{}.pkl".format(len(task_complete), int(sum(task_complete)))))
        # print("Goal reaching task completed. Success rate: {}".format(sum(task_complete) / len(task_complete) if len(task_complete) > 0 else 0))
        print("Saved at {}".format(os.path.join(save_dir, "goal_reaching_e{}_succ{}.pkl".format(len(task_complete), int(sum(task_complete))))))
