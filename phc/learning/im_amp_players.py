

import os
import os.path as osp
# sys.path.append(os.getcwd())

import numpy as np
import torch
from phc.utils.flags import flags

from tqdm import tqdm
import joblib
import time
from smpl_sim.smpllib.smpl_eval import compute_metrics_lite
from rl_games.common.tr_helpers import unsqueeze_obs
import phc.learning.amp_players as amp_players

COLLECT_Z = True

DOF_NAMES= ['L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']

class IMAMPPlayerContinuous(amp_players.AMPPlayerContinuous):
    def __init__(self, config):
        super().__init__(config)

        self.terminate_state = torch.zeros(self.env.task.num_envs, device=self.device)
        self.terminate_memory = []
        self.mpjpe, self.mpjpe_all = [], []
        self.gt_pos, self.gt_pos_all = [], []
        self.pred_pos, self.pred_pos_all = [], []
        self.gt_dof_pos, self.gt_dof_pos_all = [], []
        self.pred_dof_pos, self.pred_dof_pos_all = [], []
        self.gt_body_c, self.gt_obj_c = [], []
        self.gt_body_c_all, self.gt_obj_c_all = [], []
        self.body_force, self.obj_force, self.table_force, self.dof_force = [], [], [], []
        self.body_force_all, self.obj_force_all, self.table_force_all, self.dof_force_all = [], [], [], []
        self.action, self.action_all = [], []
        self.noisy_action, self.noisy_action_all = [], []
        self.pd_tar, self.pd_tar_all = [], []
        self.root_state, self.root_state_all = [], []
        self.dof_state, self.dof_state_all = [], []
        self.self_obs, self.self_obs_all = [], []   # t+1
        self.full_obs_t, self.full_obs_t_all = [], []  # t
        self.frame_labels, self.frame_labels_all = [], []

        self.steps_lists_all = []

        self.curr_stpes = 0

        if COLLECT_Z:
            self.zs, self.zs_all = [], []

        humanoid_env = self.env.task
        humanoid_env._termination_distances[:] = 10 # if not humanoid_env.strict_eval else 0.25 # ZL: use UHC's termination distance
        humanoid_env._recovery_episode_prob, humanoid_env._fall_init_prob = 0, 0

        if flags.im_eval:
            self.success_rate = 0
            self.pbar = tqdm(range(humanoid_env._motion_lib._num_unique_motions // humanoid_env.num_envs))
            humanoid_env.zero_out_far = False
            humanoid_env.zero_out_far_train = False
            
            if len(humanoid_env._reset_bodies_id) > 15:
                humanoid_env._reset_bodies_id = humanoid_env._eval_track_bodies_id  # Following UHC. Only do it for full body, not for three point/two point trackings. 
            
            humanoid_env.cycle_motion = False
            self.print_stats = False
        
        # joblib.dump({"mlp": self.model.a2c_network.actor_mlp, "mu": self.model.a2c_network.mu}, "single_model.pkl") # ZL: for saving part of the model.
        return

    def _post_step(self, info, done):
        super()._post_step(info)

        # modify done such that games will exit and reset.
        if flags.im_eval:

            humanoid_env = self.env.task

            if humanoid_env.cycle_motion:
                steps_lists = torch.ones_like(humanoid_env._motion_lib.get_motion_num_steps()) * humanoid_env.max_episode_length
                steps_lists = torch.maximum(
                    steps_lists,
                    humanoid_env._motion_lib.get_motion_num_steps()
                )
            else:
                steps_lists = humanoid_env._motion_lib.get_motion_num_steps()
            
            termination_state = torch.logical_and(self.curr_stpes <= steps_lists - 1, info["terminate"]) # if terminate after the last frame, then it is not a termination. curr_step is one step behind simulation. 
            # termination_state = info["terminate"]
            self.terminate_state = torch.logical_or(termination_state, self.terminate_state)
            if (~self.terminate_state).sum() > 0:
                max_possible_id = humanoid_env._motion_lib._num_unique_motions - 1
                curr_ids = humanoid_env._motion_lib._curr_motion_ids
                if (max_possible_id == curr_ids).sum() > 0: # When you are running out of motions. 
                    bound = (max_possible_id == curr_ids).nonzero()[0] + 1
                    if (~self.terminate_state[:bound]).sum() > 0:
                        curr_max = steps_lists[:bound][~self.terminate_state[:bound]].max()
                    else:
                        curr_max = (self.curr_stpes - 1)  # the ones that should be counted have teimrated
                else:
                    curr_max = steps_lists[~self.terminate_state].max()

                if self.curr_stpes >= curr_max: curr_max = self.curr_stpes + 1  # For matching up the current steps and max steps. 
            else:
                curr_max = steps_lists.max()

            self.mpjpe.append(info["mpjpe"])
            self.gt_pos.append(info["body_pos_gt"])
            self.pred_pos.append(info["body_pos"])
            self.gt_dof_pos.append(info["dof_pos_gt"])
            self.pred_dof_pos.append(info["dof_pos"])
            self.root_state.append(info["root_state"])
            self.dof_state.append(info["dof_state"])
            self.self_obs.append(info["self_obs"])
            
            ## action
            self.action.append(info["action"].clone().cpu().numpy())

            if COLLECT_Z: self.zs.append(info["z"].clone().cpu().numpy())
            self.curr_stpes += 1

            if self.curr_stpes >= curr_max or self.terminate_state.sum() == humanoid_env.num_envs:
                
                self.terminate_memory.append(self.terminate_state.cpu().numpy())
                self.success_rate = (1 - np.concatenate(self.terminate_memory)[: humanoid_env._motion_lib._num_unique_motions].mean())

                # MPJPE
                all_mpjpe = torch.stack(self.mpjpe)
                try:
                    assert(all_mpjpe.shape[0] == curr_max or self.terminate_state.sum() == humanoid_env.num_envs) # Max should be the same as the number of frames in the motion.
                except:
                    import ipdb; ipdb.set_trace()
                    print('??')


                all_mpjpe = [all_mpjpe[: (i - 1), idx].mean() for idx, i in enumerate(steps_lists)] # -1 since we do not count the first frame. 
                all_body_pos_pred = np.stack(self.pred_pos)
                all_body_pos_pred = [all_body_pos_pred[: (i - 1), idx] for idx, i in enumerate(steps_lists)]
                all_body_pos_gt = np.stack(self.gt_pos)
                all_body_pos_gt = [all_body_pos_gt[: (i - 1), idx] for idx, i in enumerate(steps_lists)]
                all_dof_pos_pred = np.stack(self.pred_dof_pos)
                all_dof_pos_pred = [all_dof_pos_pred[: (i - 1), idx] for idx, i in enumerate(steps_lists)]
                all_dof_pos_gt = np.stack(self.gt_dof_pos)
                all_dof_pos_gt = [all_dof_pos_gt[: (i - 1), idx] for idx, i in enumerate(steps_lists)]

                self.pred_pos, self.gt_pos, self.pred_dof_pos, self.gt_dof_pos = [], [], [], []
                

                if COLLECT_Z:
                    all_zs = np.stack(self.zs)
                    all_zs = [all_zs[: (i - 1), idx] for idx, i in enumerate(steps_lists)]
                    self.zs_all += all_zs


                self.mpjpe_all.append(all_mpjpe)
                self.pred_pos_all += all_body_pos_pred
                self.gt_pos_all += all_body_pos_gt
                self.pred_dof_pos_all += all_dof_pos_pred
                self.gt_dof_pos_all += all_dof_pos_gt


                all_action = np.stack(self.action)
                all_action = [all_action[: (i - 1), idx] for idx, i in enumerate(steps_lists)]
                all_root_state = np.stack(self.root_state)
                all_root_state = [all_root_state[: (i - 1), idx] for idx, i in enumerate(steps_lists)]
                all_dof_state = np.stack(self.dof_state)
                all_dof_state = [all_dof_state[: (i - 1), idx] for idx, i in enumerate(steps_lists)]
                all_self_obs = np.stack(self.self_obs)
                all_self_obs = [all_self_obs[: (i - 1), idx] for idx, i in enumerate(steps_lists)]

                self.action,  self.root_state, self.dof_state, self.self_obs = [], [], [], []

                self.action_all += all_action
                self.root_state_all += all_root_state
                self.dof_state_all += all_dof_state
                self.self_obs_all += all_self_obs

                self.frame_labels_all += list(humanoid_env._motion_lib.get_frame_labels())  # ignore if you do not have frame_labels

                self.steps_lists_all += steps_lists

                N = humanoid_env._motion_lib._num_unique_motions
                if (humanoid_env.start_idx + humanoid_env.num_envs >= humanoid_env._motion_lib._num_unique_motions):
                    del self.gt_pos, self.pred_pos, self.gt_dof_pos, self.pred_dof_pos, self.action, self.pd_tar, self.root_state, self.dof_state, self.self_obs, self.full_obs_t
                    terminate_hist = np.concatenate(self.terminate_memory)
                    succ_idxes = np.nonzero(~terminate_hist[: humanoid_env._motion_lib._num_unique_motions])[0].tolist()

                    pred_pos_all_succ = [(self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions])[i] for i in succ_idxes]
                    gt_pos_all_succ = [(self.gt_pos_all[: humanoid_env._motion_lib._num_unique_motions])[i] for i in succ_idxes]

                    pred_pos_all = self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions]
                    gt_pos_all = self.gt_pos_all[: humanoid_env._motion_lib._num_unique_motions]

                    failed_keys = humanoid_env._motion_lib._motion_data_keys[terminate_hist[: humanoid_env._motion_lib._num_unique_motions]]
                    success_keys = humanoid_env._motion_lib._motion_data_keys[~terminate_hist[: humanoid_env._motion_lib._num_unique_motions]]
                    if flags.real_traj:
                        pred_pos_all = [i[:, humanoid_env._reset_bodies_id] for i in pred_pos_all]
                        gt_pos_all = [i[:, humanoid_env._reset_bodies_id] for i in gt_pos_all]
                        pred_pos_all_succ = [i[:, humanoid_env._reset_bodies_id] for i in pred_pos_all_succ]
                        gt_pos_all_succ = [i[:, humanoid_env._reset_bodies_id] for i in gt_pos_all_succ]
                        
                        
                        
                    metrics = compute_metrics_lite(pred_pos_all, gt_pos_all)
                    metrics_succ = compute_metrics_lite(pred_pos_all_succ, gt_pos_all_succ)

                    metrics_all_print = {m: np.mean(v) for m, v in metrics.items()}
                    metrics_print = {m: np.mean(v) for m, v in metrics_succ.items()}

                    print("------------------------------------------")
                    print("------------------------------------------")
                    print(f"Success Rate: {self.success_rate:.10f}")
                    print("All: ", " \t".join([f"{k}: {v:.3f}" for k, v in metrics_all_print.items()]))
                    print("Succ: "," \t".join([f"{k}: {v:.3f}" for k, v in metrics_print.items()]))
                    print(self.config['network_path'])

                    save_info = {
                        "succ_idxes": succ_idxes,
                        "mpjpe_all": self.mpjpe_all,

                        "pred_pos_all": self.pred_pos_all[:N],
                        # "pred_dof_pos_all": self.pred_dof_pos_all[:N],
                        "action_all": self.action_all[:N],
                        "root_state_all": self.root_state_all[:N],
                        "dof_state_all": self.dof_state_all[:N],
                        # "self_obs_all": self.self_obs_all[:N],
                        "frame_labels_all": self.frame_labels_all[:N],
                        "motion_lengths_all": self.steps_lists_all[:N],
                        "z_all": self.zs_all[:N] if COLLECT_Z else None,
                        "motion_keys": humanoid_env._motion_lib._motion_data_keys[:N],
                        "failed_keys": failed_keys,
                        "success_keys": success_keys,
                    }

                    save_path = "data/amass_tracking_results"
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    save_file = osp.join(save_path, "eval_info_{}_sample.pkl".format(humanoid_env.motion_file_name))
                    
                    print("Saving to {}".format(save_file))
                    joblib.dump(save_info, save_file)

                    import sys; sys.exit()
                    # break


                done[:] = 1  # Turning all of the sequences done and reset for the next batch of eval.

                humanoid_env.forward_motion_samples()
                self.terminate_state = torch.zeros(
                    self.env.task.num_envs, device=self.device
                )

                self.pbar.update(1)
                self.pbar.refresh()
                self.mpjpe, self.gt_pos, self.pred_pos,  = [], [], []
                if COLLECT_Z: self.zs = []
                self.curr_stpes = 0
            

            update_str = f"Terminated: {self.terminate_state.sum().item()} | max frames: {curr_max} | steps {self.curr_stpes} | Start: {humanoid_env.start_idx} | Succ rate: {self.success_rate:.3f} | Mpjpe: {np.mean(self.mpjpe_all) * 1000:.3f}"
            self.pbar.set_description(update_str)

        return done
    
    def get_z(self, obs_dict):
        obs = obs_dict['obs']
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs,
            'rnn_states': self.states
        }
        with torch.no_grad():
            z = self.model.a2c_network.eval_z(input_dict)
            return z
        
    def compute_prior(self, obs_dict):
        obs = obs_dict['obs']
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs,
            'rnn_states': self.states
        }
        with torch.no_grad():
            prior_mu, prior_logvar = self.model.a2c_network.compute_prior(input_dict)
            # a, noise = self.model.a2c_network.reparameterize(prior_mu, prior_logvar); print(f"\r prior_mu {prior_mu.abs().max():.3f} {prior_logvar.exp().max():.3f}",  end='')
            return prior_mu, prior_logvar
        
    def dec_action(self, z, obs_dict):
        obs = obs_dict["obs"]
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs,
            'rnn_states': self.states
        }

        with torch.no_grad():
            mu, sigma = self.model.a2c_network.dec_actor_from_z(z, input_dict)
            return mu, sigma

        return None

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        for t in range(n_games):
            if games_played >= n_games:
                break
            obs_dict = self.env_reset()

            batch_size = 1
            batch_size = self.get_batch_size(obs_dict["obs"], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            print_game_res = False

            done_indices = []

            with torch.no_grad():
                for n in range(self.max_steps):
                    obs_dict = self.env_reset(done_indices)

                    if COLLECT_Z: 
                        z = self.get_z(obs_dict)
                        mu, sigma = self.dec_action(z, obs_dict)  ## for debug

                    if has_masks:
                        masks = self.env.get_action_mask()
                        action = self.get_masked_action(obs_dict, masks, is_determenistic)
                    else:
                        action = self.get_action(obs_dict, is_determenistic)

                    obs_dict, r, done, info = self.env_step(self.env, action)

                    cr += r
                    steps += 1

                    if COLLECT_Z: info['z'] = z
                    done = self._post_step(info, done.clone())

                    # print((action - mu).mean())

                    if render:
                        self.env.render(mode="human")
                        time.sleep(self.render_sleep)
                        
                    all_done_indices = done.nonzero(as_tuple=False)
                    done_indices = all_done_indices[:: self.num_agents]
                    done_count = len(done_indices)
                    games_played += done_count

                    if done_count > 0:
                        if self.is_rnn:
                            for s in self.states:
                                s[:, all_done_indices, :] = (
                                    s[:, all_done_indices, :] * 0.0
                                )

                        cur_rewards = cr[done_indices].sum().item()
                        cur_steps = steps[done_indices].sum().item()

                        cr = cr * (1.0 - done.float())
                        steps = steps * (1.0 - done.float())
                        sum_rewards += cur_rewards
                        sum_steps += cur_steps

                        game_res = 0.0
                        if isinstance(info, dict):
                            if "battle_won" in info:
                                print_game_res = True
                                game_res = info.get("battle_won", 0.5)
                            if "scores" in info:
                                print_game_res = True
                                game_res = info.get("scores", 0.5)
                        if self.print_stats:
                            if print_game_res:
                                print("reward:", cur_rewards / done_count, "steps:", cur_steps / done_count, "w:", game_res,)
                            else:
                                print("reward:", cur_rewards / done_count, "steps:", cur_steps / done_count,)

                        sum_game_res += game_res
                        # if batch_size//self.num_agents == 1 or games_played >= n_games:
                        if games_played >= n_games:
                            break

                    done_indices = done_indices[:, 0]

        print(sum_rewards)
        if print_game_res:
            print(
                "av reward:",
                sum_rewards / games_played * n_game_life,
                "av steps:",
                sum_steps / games_played * n_game_life,
                "winrate:",
                sum_game_res / games_played * n_game_life,
            )
        else:
            print(
                "av reward:",
                sum_rewards / games_played * n_game_life,
                "av steps:",
                sum_steps / games_played * n_game_life,
            )

        return
