import glob
import os
import sys
import pdb
import os.path as osp
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.append(os.getcwd())

from phc.utils.config import set_np_formatting, set_seed, SIM_TIMESTEP
from phc.utils.parse_task import parse_task
from isaacgym import gymapi
from isaacgym import gymutil


from rl_games.algos_torch import players
from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, experiment, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner

from phc.utils.flags import flags

import numpy as np
import copy
import torch
import wandb

from phc.learning import im_amp
from phc.learning import im_amp_players
from phc.learning import amp_agent
from phc.learning import amp_players
from phc.learning import amp_models
from phc.learning import amp_network_builder
from phc.learning import amp_network_mcp_builder
from phc.learning import amp_network_pnn_builder
from phc.learning import amp_network_z_builder
from phc.learning import amp_network_z_reader_builder

from phc.env.tasks import humanoid_amp_task
import hydra
from omegaconf import DictConfig, OmegaConf
from easydict import EasyDict

args = None
cfg = None
cfg_train = None


def parse_sim_params(cfg):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = SIM_TIMESTEP
    sim_params.num_client_threads = cfg.sim.slices
    
    if cfg.sim.use_flex:
        if cfg.sim.pipeline in ["gpu"]:
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.use_flex.shape_collision_margin = 0.01
        sim_params.use_flex.num_outer_iterations = 4
        sim_params.use_flex.num_inner_iterations = 10
    else : # use gymapi.SIM_PHYSX
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = cfg.sim.pipeline in ["gpu"]
        sim_params.physx.num_subscenes = cfg.sim.subscenes
        if flags.test and not flags.im_eval:
            sim_params.physx.max_gpu_contact_pairs = 4 * 1024 * 1024
        else:
            sim_params.physx.max_gpu_contact_pairs = 16 * 1024 * 1024

    sim_params.use_gpu_pipeline = cfg.sim.pipeline in ["gpu"]
    sim_params.physx.use_gpu = cfg.sim.pipeline in ["gpu"]

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if not cfg.sim.use_flex and cfg.sim.physx.num_threads > 0:
        sim_params.physx.num_threads = cfg.sim.physx.num_threads
    
    return sim_params

def create_rlgpu_env(**kwargs):
    use_horovod = cfg_train['params']['config'].get('multi_gpu', False)
    if use_horovod:
        import horovod.torch as hvd

        rank = hvd.rank()
        print("Horovod rank: ", rank)

        cfg_train['params']['seed'] = cfg_train['params']['seed'] + rank

        args.device = 'cuda'
        args.device_id = rank
        args.rl_device = 'cuda:' + str(rank)

        cfg['rank'] = rank
        cfg['rl_device'] = 'cuda:' + str(rank)
    
    sim_params = parse_sim_params(cfg)
    args = EasyDict({
        "task": cfg.env.task, 
        "device_id": cfg.device_id,
        "rl_device": cfg.rl_device,
        "physics_engine": gymapi.SIM_PHYSX if not cfg.sim.use_flex else gymapi.SIM_FLEX,
        "headless": cfg.headless,
        "device": cfg.device,
    }) #### ZL: patch 
    task, env = parse_task(args, cfg, cfg_train, sim_params)

    print(env.num_envs)
    print(env.num_actions)
    print(env.num_obs)
    print(env.num_states)

    frames = kwargs.pop('frames', 1)
    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    return env


class RLGPUAlgoObserver(AlgoObserver):

    def __init__(self, use_successes=True):
        self.use_successes = use_successes
        return

    def after_init(self, algo):
        self.algo = algo
        self.consecutive_successes = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.writer = self.algo.writer
        return

    def process_infos(self, infos, done_indices):
        if isinstance(infos, dict):
            if (self.use_successes == False) and 'consecutive_successes' in infos:
                cons_successes = infos['consecutive_successes'].clone()
                self.consecutive_successes.update(cons_successes.to(self.algo.ppo_device))
            if self.use_successes and 'successes' in infos:
                successes = infos['successes'].clone()
                self.consecutive_successes.update(successes[done_indices].to(self.algo.ppo_device))
        return

    def after_clear_stats(self):
        self.mean_scores.clear()
        return

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.consecutive_successes.current_size > 0:
            mean_con_successes = self.consecutive_successes.get_mean()
            self.writer.add_scalar('successes/consecutive_successes/mean', mean_con_successes, frame)
            self.writer.add_scalar('successes/consecutive_successes/iter', mean_con_successes, epoch_num)
            self.writer.add_scalar('successes/consecutive_successes/time', mean_con_successes, total_time)
        return


class RLGPUEnv(vecenv.IVecEnv):

    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)
        self.use_global_obs = (self.env.num_states > 0)

        self.full_state = {}
        self.full_state["obs"] = self.reset()
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
        return

    def step(self, action):
        next_obs, reward, is_done, info = self.env.step(action)

        # todo: improve, return only dictinary
        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, reward, is_done, info
        else:
            return self.full_state["obs"], reward, is_done, info

    def reset(self, env_ids=None):
        self.full_state["obs"] = self.env.reset(env_ids)
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state
        else:
            return self.full_state["obs"]

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space
        info['amp_observation_space'] = self.env.amp_observation_space
        
        info['enc_amp_observation_space'] = self.env.enc_amp_observation_space
        
        if isinstance(self.env.task, humanoid_amp_task.HumanoidAMPTask):
            info['task_obs_size'] = self.env.task.get_task_obs_size()
        else:
            info['task_obs_size'] = 0

        if self.use_global_obs:
            info['state_space'] = self.env.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info


vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs), 'vecenv_type': 'RLGPU'})


def build_alg_runner(algo_observer):
    runner = Runner(algo_observer)
    runner.player_factory.register_builder('amp_discrete', lambda **kwargs: amp_players.AMPPlayerDiscrete(**kwargs))
    
    runner.algo_factory.register_builder('amp', lambda **kwargs: amp_agent.AMPAgent(**kwargs))
    runner.player_factory.register_builder('amp', lambda **kwargs: amp_players.AMPPlayerContinuous(**kwargs))

    runner.model_builder.model_factory.register_builder('amp', lambda network, **kwargs: amp_models.ModelAMPContinuous(network))
    runner.model_builder.network_factory.register_builder('amp', lambda **kwargs: amp_network_builder.AMPBuilder())
    runner.model_builder.network_factory.register_builder('amp_mcp', lambda **kwargs: amp_network_mcp_builder.AMPMCPBuilder())
    runner.model_builder.network_factory.register_builder('amp_pnn', lambda **kwargs: amp_network_pnn_builder.AMPPNNBuilder())
    runner.model_builder.network_factory.register_builder('amp_z', lambda **kwargs: amp_network_z_builder.AMPZBuilder())
    runner.model_builder.network_factory.register_builder('amp_z_reader', lambda **kwargs: amp_network_z_reader_builder.AMPZReaderBuilder())

    runner.algo_factory.register_builder('im_amp', lambda **kwargs: im_amp.IMAmpAgent(**kwargs))
    runner.player_factory.register_builder('im_amp', lambda **kwargs: im_amp_players.IMAMPPlayerContinuous(**kwargs))
    
    return runner

"""
diffusion-forcing related
"""
import os
import sys
import subprocess
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict

from uniphys.utils.print_utils import cyan
from uniphys.utils.ckpt_utils import download_latest_checkpoint, is_run_id
from uniphys.utils.cluster_utils import submit_slurm_job
from uniphys.utils.distributed_utils import is_rank_zero
from uniphys.experiments import build_experiment
from uniphys.utils.wandb_utils import OfflineWandbLogger, SpaceEfficientWandbLogger


def run_local(cfg: DictConfig):
    # delay some imports in case they are not needed in non-local envs for submission

    # # Get yaml names
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cfg_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)

    if cfg_choice["diffusion_forcing/experiment"] is not None:
        cfg.experiment["_name"] = cfg_choice["diffusion_forcing/experiment"]
    if cfg_choice["diffusion_forcing/dataset"] is not None:
        cfg.dataset["_name"] = cfg_choice["diffusion_forcing/dataset"]
    if cfg_choice["diffusion_forcing/algorithm"] is not None:
        cfg.algorithm["_name"] = cfg_choice["diffusion_forcing/algorithm"]


    # Set up the output directory.
    hydra_output_dir = hydra_cfg.runtime.output_dir.split("/")
    outputs_str_index = hydra_output_dir.index("outputs")
    output_dir = os.path.join('/', *hydra_output_dir[:outputs_str_index+1], cfg.name, *hydra_output_dir[outputs_str_index+1:])
    
    # sys.argv.append('hydra_cfg.runtime.output_dir={}'.format(output_dir))
    output_dir = Path(hydra_cfg.runtime.output_dir)
    if is_rank_zero:
        print(cyan(f"Outputs will be saved to:"), output_dir)
        (output_dir.parents[1] / "latest-run").unlink(missing_ok=True)
        (output_dir.parents[1] / "latest-run").symlink_to(output_dir, target_is_directory=True)

    # Set up logging with wandb.
    if cfg.wandb.mode != "disabled":
        # If resuming, merge into the existing run on wandb.
        resume = cfg.get("resume", None)
        name = f"{cfg.name} ({output_dir.parent.name}/{output_dir.name})" if resume is None else None

        if "_on_compute_node" in cfg and cfg.cluster.is_compute_node_offline:
            logger_cls = OfflineWandbLogger
        else:
            logger_cls = SpaceEfficientWandbLogger

        offline = cfg.wandb.mode != "online"
        logger = logger_cls(
            name=name,
            save_dir=str(output_dir),
            offline=offline,
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            log_model=False, #"all" if not offline else False,
            # config=OmegaConf.to_container(cfg),
            config=cfg,
            id=resume,
        )
    else:
        logger = None
    
    # Load ckpt
    resume = cfg.get("resume", None)
    load = cfg.get("load", None)
    checkpoint_path = None
    load_id = None
    if load and not is_run_id(load):
        checkpoint_path = load
    if resume:
        load_id = resume
    elif load and is_run_id(load):
        load_id = load
    else:
        load_id = None

    if load_id:
        run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{load_id}"
        checkpoint_path = Path("outputs/downloaded") / run_path / "model.ckpt"

    if checkpoint_path and is_rank_zero:
        print(f"Will load checkpoint from {checkpoint_path}")

    # launch experiment
    experiment = build_experiment(cfg, logger, checkpoint_path)

    return experiment


def run_slurm(cfg: DictConfig):
    python_args = " ".join(sys.argv[1:]) + " +_on_compute_node=True"
    project_root = Path.cwd()
    while not (project_root / ".git").exists():
        project_root = project_root.parent
        if project_root == Path("/"):
            raise Exception("Could not find repo directory!")

    slurm_log_dir = submit_slurm_job(
        cfg,
        python_args,
        project_root,
    )

    if "cluster" in cfg and cfg.cluster.is_compute_node_offline and cfg.wandb.mode == "online":
        print("Job submitted to a compute node without internet. This requires manual syncing on login node.")
        osh_command_dir = project_root / ".wandb_osh_command_dir"

        osh_proc = None
        # if click.confirm("Do you want us to run the sync loop for you?", default=True):
        osh_proc = subprocess.Popen(["wandb-osh", "--command-dir", osh_command_dir])
        print(f"Running wandb-osh in background... PID: {osh_proc.pid}")
        print(f"To kill the sync process, run 'kill {osh_proc.pid}' in the terminal.")
        print(
            f"You can manually start a sync loop later by running the following:",
            cyan(f"wandb-osh --command-dir {osh_command_dir}"),
        )

    print(
        "Once the job gets allocated and starts running, we will print a command below "
        "for you to trace the errors and outputs: (Ctrl + C to exit without waiting)"
    )
    msg = f"tail -f {slurm_log_dir}/* \n"
    try:
        while not list(slurm_log_dir.glob("*.out")) and not list(slurm_log_dir.glob("*.err")):
            time.sleep(1)
        print(cyan("To trace the outputs and errors, run the following command:"), msg)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting...")
        print(
            cyan("To trace the outputs and errors, manually wait for the job to start and run the following command:"),
            msg,
        )

@hydra.main(
    version_base=None,
    config_path="config",
    config_name="cfg",
)
def main(cfg_hydra: DictConfig) -> None:
    global cfg_train
    global cfg
    
    cfg_all = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True))
    cfg = cfg_all.phc
    cfg_diffusion = cfg_all.diffusion_forcing

    """
    phc related set-up
    """
    set_np_formatting()

    flags.debug, flags.follow, flags.fixed, flags.divide_group, flags.no_collision_check, flags.fixed_path, flags.real_path,  flags.show_traj, flags.server_mode, flags.slow, flags.real_traj, flags.im_eval, flags.no_virtual_display, flags.render_o3d = \
        cfg.debug, cfg.follow, False, False, False, False, False, True, cfg.server_mode, False, False, cfg.im_eval, cfg.no_virtual_display, cfg.render_o3d

    flags.test = cfg.test
    flags.add_proj = cfg.add_proj
    flags.has_eval = cfg.has_eval
    flags.trigger_input = False

    flags.prompt = getattr(cfg_diffusion.algorithm, "interactive_input_prompt", False)

    if cfg.server_mode:
        flags.follow = cfg.follow = True
        flags.fixed = cfg.fixed = True
        flags.no_collision_check = True
        flags.show_traj = True
        cfg['env']['episode_length'] = 99999999999999

    if cfg.real_traj:
        cfg['env']['episode_length'] = 99999999999999
        flags.real_traj = True
    
    cfg.train = not cfg.test
    project_name = cfg.get("project_name", "egoquest")
    if (not cfg.no_log) and (not cfg.test) and (not cfg.debug):
        wandb.init(
            project=project_name,
            resume=not cfg.resume_str is None,
            id=cfg.resume_str,
            notes=cfg.get("notes", "no notes"),
        )
        wandb.config.update(cfg, allow_val_change=True)
        wandb.run.name = cfg.exp_name
        wandb.run.save()
    
    set_seed(cfg.get("seed", -1), cfg.get("torch_deterministic", False))

    # Create default directories for weights and statistics
    cfg_train = cfg.learning
    cfg_train['params']['config']['network_path'] = cfg.output_path
    cfg_train['params']['config']['train_dir'] = cfg.output_path
    cfg_train["params"]["config"]["num_actors"] = cfg.env.num_envs
    
    if cfg.epoch > 0:
        cfg_train["params"]["load_checkpoint"] = True
        cfg_train["params"]["load_path"] = osp.join(cfg.output_path, cfg_train["params"]["config"]['name'] + "_" + str(cfg.epoch).zfill(8) + '.pth')
    elif cfg.epoch == -1:
        path = osp.join(cfg.output_path, cfg_train["params"]["config"]['name'] + '.pth')
        if osp.exists(path):
            cfg_train["params"]["load_path"] = path
            cfg_train["params"]["load_checkpoint"] = True
        else:
            print(path)
            raise Exception("no file to resume!!!!")

    os.makedirs(cfg.output_path, exist_ok=True)

    """
    diffusion-forcing related setup
    """
    if "_on_compute_node" in cfg_diffusion and cfg_diffusion.cluster.is_compute_node_offline:
        with open_dict(cfg_diffusion):
            if cfg_diffusion.cluster.is_compute_node_offline and cfg_diffusion.wandb.mode == "online":
                cfg.wandb.mode = "offline"

    if "name" not in cfg_diffusion:
        raise ValueError("must specify a name for the run with command line argument '+name=[name]'")

    if not cfg_diffusion.wandb.get("entity", None):
        raise ValueError(
            "must specify wandb entity in 'configurations/config.yaml' or with command line"
            " argument 'wandb.entity=[entity]' \n An entity is your wandb user name or group"
            " name. This is used for logging. If you don't have an wandb account, please signup at https://wandb.ai/"
        )

    if cfg_diffusion.wandb.project is None:
        cfg_diffusion.wandb.project = str(Path(__file__).parent.name)

    # If resuming or loading a wandb ckpt and not on a compute node, download the checkpoint.
    resume = cfg_diffusion.get("resume", None)
    load = cfg_diffusion.get("load", None)

    if resume and load:
        raise ValueError(
            "When resuming a wandb run with `resume=[wandb id]`, checkpoint will be loaded from the cloud"
            "and `load` should not be specified."
        )

    if resume:
        load_id = resume
    elif load and is_run_id(load):
        load_id = load
    else:
        load_id = None

    if load_id and "_on_compute_node" not in cfg_diffusion:
        run_path = f"{cfg_diffusion.wandb.entity}/{cfg_diffusion.wandb.project}/{load_id}"
        download_latest_checkpoint(run_path, Path("outputs/downloaded"))

    if "cluster" in cfg_diffusion and not "_on_compute_node" in cfg_diffusion:
        print(cyan("Slurm detected, submitting to compute node instead of running locally..."))
        run_slurm(cfg_diffusion)
    else:
        experiment = run_local(cfg_diffusion)

    cfg_diffusion.dataset.state_dim = 351
    cfg_diffusion.dataset.repr_name_list = ['local_positions', 'local_vel', 'dof_pose_6d', 'dof_vel']
    if cfg_diffusion.algorithm.state_with_root:
        cfg_diffusion.dataset.state_dim = 351 + 3 + 6 + 3 + 3
        cfg_diffusion.dataset.repr_name_list = ['root_trans', 'root_rot_6d', 'root_trans_vel', 'root_rot_vel', 'local_positions', 'local_vel', 'dof_pose_6d', 'dof_vel']

    cfg_diffusion.dataset.observation_shape = [cfg_diffusion.dataset.state_dim + cfg_diffusion.dataset.action_dim]
    cfg_diffusion.algorithm.state_dim = cfg_diffusion.dataset.state_dim
    cfg_diffusion.algorithm.observation_shape = cfg_diffusion.dataset.observation_shape
    cfg_diffusion.algorithm.x_shape = cfg_diffusion.dataset.observation_shape

    ## create player [optional]
    if cfg_diffusion.play:
        if torch.cuda.current_device() == 0:
            algo_observer = RLGPUAlgoObserver()
            runner = build_alg_runner(algo_observer)
            runner.load(cfg_train)
            runner.reset()
            # player = runner.load_player()
            player = runner.create_player()
            player.restore(cfg_train["params"]["load_path"])
            experiment.create_player(player)

    experiment.exec_task(cfg_diffusion.task)

if __name__ == '__main__':
    main()
