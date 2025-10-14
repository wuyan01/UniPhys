import joblib
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import DictConfig
from tqdm import tqdm

from uniphys.utils.clip_utils import load_and_freeze_clip
from .df_humanoid import DiffusionForcingHumanoid

from uniphys.utils.motion_repr_utils import (
    cano_goal,
    recover_global_root_state,
)


class DiffusionForcingHumanoidSteer(DiffusionForcingHumanoid):
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
        self.target_speed_all, self.target_dir_all, self.target_facing_dir_all = [], [], []
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

            self.target_speed_all.append(self.env.task._tar_speed.clone().detach().cpu().numpy())
            self.target_dir_all.append(self.env.task._tar_dir.clone().detach().cpu().numpy())
            self.target_facing_dir_all.append(self.env.task._tar_facing_dir.clone().detach().cpu().numpy())

            if "z" in info: 
                self.zs_all.append(info["z"])

        done = done | (self.env.task.progress_buf >= self.max_episode_length) # timeout

        return done

    def velocity_guidance_fn(self, xs_pred=None, verbose=False):
        """
        Guidance function for diffusion sampling.
        Encourages predicted root velocities and orientations to match
        target velocities and target facing direction.

        Args:
            xs_pred (Tensor): Predicted sequence [T, B, fs*c]
            verbose (bool): If True, prints debug losses.

        Returns:
            Tensor: Scalar guidance loss.
        """

        # === Reshape predictions ===
        xs_pred = rearrange(xs_pred, "t b (fs c) -> (t fs) b c", fs=self.frame_stack).contiguous()
        xs_pred = rearrange(xs_pred, "t (n b) c -> n t b c", n=self.n_samples).contiguous()

        # === Recover state predictions ===
        # Apply normalization (de-standardize)
        state_pred = xs_pred[..., self.action_dim:]
        state_pred = (
            state_pred * torch.from_numpy(self.state_std).cuda()
            + torch.from_numpy(self.state_mean).cuda()
        )

        # Recover global states from predicted features
        global_pos, global_root_linear_vel, global_rotmat = recover_global_root_state(
            state_pred[..., :3],      # root position
            state_pred[..., 3:9],     # root rotation (6D)
            state_pred[..., 9:12],    # root linear velocity
            self.cano_transf_rotmat,
            self.cano_transf_transl,
        )

        # Remove context frames (only keep future predictions)
        global_pos = global_pos[:, self.context_frames:]
        global_rotmat = global_rotmat[:, self.context_frames:]
        global_root_linear_vel = global_root_linear_vel[:, self.context_frames:]

        # === Create interpolated velocity target ===
        T = self.chunk_size         # horizon
        change_t = 1   # number of steps to interpolate

        # Weighting schedule for interpolation
        t_steps = torch.ones(T).cuda()
        t_steps[:change_t] = torch.linspace(1 / change_t, 1, change_t).cuda()

        # Current and target velocity (XY plane only)
        v_current = self.env.task._root_states[self.env.task._humanoid_actor_ids][..., 7:9]
        v_target = self.env.task._tar_dir * self.env.task._tar_speed

        # Split into magnitude and direction for interpolation
        magnitude_current = torch.norm(v_current, dim=-1)
        magnitude_target = torch.norm(v_target, dim=-1)

        direction_current = v_current / magnitude_current[..., None]
        direction_target = v_target / magnitude_target[..., None]

        # Angles of current/target velocities
        angle_current = torch.atan2(direction_current[:, 1], direction_current[:, 0])
        angle_target = torch.atan2(direction_target[:, 1], direction_target[:, 0])

        # Ensure shortest angular interpolation
        delta_angle = (angle_target - angle_current + torch.pi) % (2 * torch.pi) - torch.pi

        # Build interpolated velocities (XY plane)
        interpolated_velocities = []
        for t in t_steps:
            # Interpolate magnitude
            mag_interp = torch.lerp(magnitude_current, magnitude_target, t)

            # Interpolate direction
            angle_interp = angle_current + t * delta_angle
            dir_interp = torch.stack([torch.cos(angle_interp), torch.sin(angle_interp)])

            # Combine magnitude and direction
            v_interp = mag_interp * dir_interp
            interpolated_velocities.append(v_interp)

        # [T, B, 2]
        self.interpolated_velocities = torch.stack(interpolated_velocities)

        # === Velocity matching loss ===
        mag_target_vel = torch.norm(self.interpolated_velocities, dim=1)              # [T, B]
        mag_pred_vel = torch.norm(global_root_linear_vel[..., :2], dim=-1)            # [n, T, B]

        # Magnitude loss
        loss_mag = F.mse_loss(mag_pred_vel, mag_target_vel, reduction="none").mean(-1).mean(-1)

        # Direction loss (cosine similarity)
        cosine_sim = F.cosine_similarity(
            global_root_linear_vel[..., :2],
            self.interpolated_velocities.transpose(1, 2).unsqueeze(0),
            dim=-1,
        )
        loss_dir = 1 - cosine_sim.mean(-1).mean(-1)

        # Balance magnitude vs direction loss
        beta = loss_dir / (loss_dir + 0.02)
        vel_cost = (1 - beta) * loss_mag + beta * loss_dir

        # === Orientation matching loss ===
        # Forward vector is the first column of the root rotation matrix
        forward_vector = global_rotmat[..., 0]  
        xy_vector = F.normalize(forward_vector[..., :2], dim=-1)

        # Target facing direction (XY plane)
        goal_vector_xy = F.normalize(self.env.task._tar_facing_dir, dim=-1)

        # Cosine similarity for orientation
        cos_theta = torch.sum(xy_vector * goal_vector_xy, dim=-1)
        orientation_cost = 1 - cos_theta.mean(-1).mean(1)

        # Balance velocity vs orientation loss
        alpha = orientation_cost / (orientation_cost + 0.01)

        # === Final guidance loss ===
        loss = -((1 - alpha) * vel_cost + alpha * orientation_cost).mean()

        if verbose:
            print(
                loss.item(),
                vel_cost.mean().item(),
                orientation_cost.mean().item(),
                loss_dir.mean().item(),
                loss_mag.mean().item(),
            )

        return loss * 10
    
    def post_select_cost_fn(self, state_pred):
        """
        For monte carlo sampling: compute cost for each sampled trajectory and select the optimal one.
        For velocity guidance task, the cost function is basically same as the velocity guidance function, but you can also customize it based on the task needs.
        """
        global_root_pos_pred, global_root_linear_vel_pred, global_root_rotmat = recover_global_root_state(state_pred[..., :3], state_pred[..., 3:9], state_pred[..., 9:12], self.cano_transf_rotmat, self.cano_transf_transl)
        global_rotmat = global_root_rotmat[:, self.context_frames:]
        global_root_linear_vel = global_root_linear_vel_pred[:, self.context_frames:]

        mag_target_vel = torch.norm(self.interpolated_velocities, dim=1)
        mag_pred_vel = torch.norm(global_root_linear_vel[..., :2], dim=-1)
        loss_mag = F.mse_loss(mag_pred_vel, mag_target_vel.unsqueeze(0), reduction='none').mean(-1).mean(-1)
        cosine_sim = F.cosine_similarity(global_root_linear_vel[..., :2], self.interpolated_velocities.transpose(1, 2).unsqueeze(0), dim=-1).mean(-1).mean(-1)
        loss_dir = 1 - cosine_sim

        beta = loss_dir / (loss_dir + 0.02)
        vel_cost = (1-beta) * loss_mag + beta * loss_dir

        # Extract the forward vector (1st column of the rotation matrix)
        forward_vector = global_rotmat[..., 0]  # Assuming the forward direction is the first column
        # Extract x and y components to project onto the x-y plane
        xy_vector = forward_vector[..., :2].cuda()
        xy_vector = F.normalize(xy_vector, dim=-1)

        goal_vector_xy = self.env.task._tar_facing_dir
        goal_vector_xy = F.normalize(goal_vector_xy, dim=-1)
        goal_vector_xy = F.normalize(goal_vector_xy, dim=-1)

        # Orientation Loss (1 - cosine similarity)
        cos_theta = torch.sum(xy_vector * goal_vector_xy, dim=-1)  # Dot product [n, b, 1]
        orientation_cost = 1 - cos_theta.mean(-1).mean(1)

        alpha = orientation_cost / (orientation_cost + 0.01)

        cost = (1-alpha) * vel_cost + alpha * orientation_cost ** 2

        return cost


    @torch.no_grad()
    def interact(self, save=False):
        # === Environment setup ===
        self.env.task._termination_distances[:] = 1e6
        self.env.task.termination_mode = "sampling"
        num_envs = self.env.task.num_envs

        # === Guidance & text setup ===
        self.n_samples = 20  # Monte Carlo samples
        self.guidance_params = 0.0
        self.guidance_fn = self.velocity_guidance_fn
        text = ""
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
        self.env.task.completed_episode_lengths = []

        self.max_episode_length = self.env.task.max_episode_length
        self.max_episode = 1
        # warm_up_step = self.context_frames + 1
        max_steps = (self.max_episode) * self.max_episode_length
        self.pbar = tqdm(range(self.max_episode_length))

        # === Loop ===
        t, e, reset_flag = 0, 0, False

        ## === Save dir ===
        if save:
            import os
            save_dir = os.path.join(self.save_dir, "steer", "step_{}_exec_{}_noise_{}_n_{}_sample_{}".format(self.resume_step, self.exec_step, self.diffusion_model.stabilization_level, self.cfg.diffusion.sampling_timesteps, self.n_samples))
            if not self.cfg.skip_text and self.guidance_params > 0:
                save_dir += "_{}".format(text)
                save_dir += "_g{}".format(self.cfg.guidance_params)
            else:
                save_dir += "_uncond"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        else:
            save_dir = None


        while t < max_steps and e < self.max_episode:
            # --- Episode init ---
            done_indices = []

            if t == 0 or reset_flag:
                obs_dict = self.player.env_reset()
                reset_flag = False
                self.env.task.set_char_color(col=[0, 0.5, 0.8], env_ids=torch.arange(self.env.task.num_envs))  ## blue

                while len(self.root_state_buffer.buffer) < self.context_frames:
                    """
                    Init with zero actions for few steps
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

            self.global_goal_position = self.env.task.sparse_goal_positions.view(1, -1, self.env.task.num_envs, 3) + self.env.task._root_states[self.env.task._humanoid_actor_ids][..., :3]
            self.global_goal_position[..., -1] = 0.9

            self.env.task.set_char_color(col=[1, 0.5, 0.0], env_ids=torch.arange(self.env.task.num_envs))  ## orange

            """
            Diffusion with Monte Carlo Guidance
            """
            with torch.no_grad():
                joint_position = self.joint_pos_buffer.get().clone().cpu().numpy()[-self.context_frames:]
                _, self.cano_transf_rotmat, self.cano_transf_transl = cano_goal(joint_position.transpose(1, 0, 2, 3), self.global_goal_position)
                
                action_pred, state_pred = self.policy()
                
                """
                if MONTE CARLO GUIDANCE, choose the best trajectory among n_samples
                """

                if self.n_samples > 1:
                    cost = self.post_select_cost_fn(state_pred)
                    opt_i = cost.argmin() 

                    action_pred = action_pred[opt_i]
                    state_pred = state_pred[opt_i]

                action_exec = action_pred[self.H:].permute(1, 0, 2).float()

                """
                interact with the env
                """

                for i in range(self.exec_step):

                    velocity = self.env.task._root_states[self.env.task._humanoid_actor_ids][..., 7:9].cpu().numpy()
                    target_velocity = (self.env.task._tar_dir * self.env.task._tar_speed).cpu().numpy()

                    self.pbar.update(1)
                    self.pbar.refresh()
                    update_str = f"Steps {t}/{max_steps} | Target {target_velocity.squeeze()} | Current: {velocity.squeeze()} | Error: {np.linalg.norm(velocity - target_velocity)}"
                    self.pbar.set_description(update_str)

                    a = action_exec[:, i]

                    if self.cfg.action_keys == ["z"]:
                        with torch.no_grad():
                            input_dict = {"obs": obs_dict}
                            a_exec, _ = self.player.dec_action(a, input_dict)
                    else:
                        a_exec = a  # B, 69


                    obs_dict, r, done, info = self.player.env_step(self.env, a_exec)

                    self.global_goal_position = self.env.task.sparse_goal_positions.view(1, -1, self.env.task.num_envs, 3) + self.env.task._root_states[self.env.task._humanoid_actor_ids][..., :3]
                    self.global_goal_position[..., -1] = 0.9

                    self.root_state_buffer.add(self.env.task._root_states[self.env.task._humanoid_actor_ids].clone())
                    self.dof_state_buffer.add(self.env.task._dof_state.reshape(num_envs, -1, 2).clone())
                    self.joint_pos_buffer.add(self.env.task._rigid_body_pos.clone())
                    self.action_buffer.add(a.clone())  ## "z" or "action"

                    t += 1

                    done = self.post_step(info, done.clone(), save_dir=save_dir)

                    if done.any():
                        reset_flag = True
                        e += 1
                        print("\033[31mFailed!\033[0m")
                        break

        self.pred_pos_all = np.stack(self.pred_pos_all).squeeze()
        self.pred_dof_pos_all = np.stack(self.pred_dof_pos_all).squeeze()
        self.root_state_all = np.stack(self.root_state_all).squeeze()
        self.dof_state_all = np.stack(self.dof_state_all).squeeze()
        self.action_all = np.stack(self.action_all).squeeze()
        self.target_speed_all = np.stack(self.target_speed_all).squeeze()
        self.target_dir_all = np.stack(self.target_dir_all).squeeze()
        self.target_facing_dir_all = np.stack(self.target_facing_dir_all).squeeze()

        save_info = {"body_pos": self.pred_pos_all, 
                     "dof_pos": self.pred_dof_pos_all, 
                     "root_state": self.root_state_all, 
                     "dof_state": self.dof_state_all, 
                     "action": self.action_all, 
                     "target_speed": self.target_speed_all,
                     "target_dir": self.target_dir_all,
                     "target_face_dir": self.target_facing_dir_all,
                     }
        
        joblib.dump(save_info, os.path.join(save_dir, "steering_steps{}_e{}.pkl".format(max_steps, e)))
        print("Saved at {}".format(os.path.join(save_dir, "steering_steps{}_e{}.pkl".format(max_steps, e))))
