import os
import joblib
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import torch

# ---------------------------------------------------------
# Asset description class
# ---------------------------------------------------------
class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments


# ---------------------------------------------------------
# Define assets
# ---------------------------------------------------------
asset_descriptors = [AssetDesc("smpl_humanoid.xml", False)]


# ---------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------
args = gymutil.parse_arguments(
    description="Visualize motion sequence in Isaac Gym",
    custom_parameters=[
        {
            "name": "--asset_id",
            "type": int,
            "default": 0,
            "help": f"Asset id (0 - {len(asset_descriptors) - 1})",
        },
        {
            "name": "--show_axis",
            "action": "store_true",
            "help": "Visualize DOF axis",
        },
        {
            "name": "--load_motion_path",
            "type": str,
            "default": "./CMU/01/01_01_poses.pkl",
            "help": "Path to motion pickle file",
        },
    ],
)

if not (0 <= args.asset_id < len(asset_descriptors)):
    print(f"*** Invalid asset_id specified. Valid range is 0 to {len(asset_descriptors) - 1}")
    quit()


# ---------------------------------------------------------
# Initialize simulator
# ---------------------------------------------------------
gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

if not args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()


# ---------------------------------------------------------
# Ground and viewer setup
# ---------------------------------------------------------
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()


# ---------------------------------------------------------
# Load asset
# ---------------------------------------------------------
asset_root = "./phc/data/assets/mjcf/"
asset_file = asset_descriptors[args.asset_id].file_name

asset_options = gymapi.AssetOptions()
asset_options.use_mesh_materials = True

print(f"Loading asset '{asset_file}' from '{asset_root}'")
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)


# ---------------------------------------------------------
# Create environment
# ---------------------------------------------------------
num_envs = 1
num_per_row = 1
spacing = 5.0

env_lower = gymapi.Vec3(-spacing, -spacing, 0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

envs, actor_handles = [], []
num_dofs = gym.get_asset_dof_count(asset)

print(f"Creating {num_envs} environment(s)")
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    pose = gymapi.Transform()
    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)

    dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

gym.prepare_sim(sim)

# ---------------------------------------------------------
# Load motion sequence
# ---------------------------------------------------------
load_motion_path = args.load_motion_path
assert os.path.exists(load_motion_path), f"Motion file not found: {load_motion_path}"

motion = joblib.load(load_motion_path)
motion_length = len(motion["root_state"])
is_succ = motion.get("is_succ", True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_states = torch.from_numpy(motion["root_state"]).to(device)
dof_states = torch.from_numpy(motion["dof_state"]).to(device)

print(f"Loaded motion from {load_motion_path} ({motion_length} frames) - success: {is_succ}")


# ---------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------
rigidbody_state = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim)).reshape(num_envs, -1, 13)
actor_root_state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))

cam_pos = gymapi.Vec3(0, -5.0, 3)
cam_target = gymapi.Vec3(0, 0, 1)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

time_step = 0
fps = 30.0

print("Starting playback...")

while not gym.query_viewer_has_closed(viewer):
    motion_time = time_step % motion_length

    if args.show_axis:
        gym.clear_lines(viewer)

    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states[motion_time:motion_time + 1]))
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_states[motion_time]))

    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

    time_step += 1

    print("\r" + " " * 200, end="")
    print(f"\rTime step: {motion_time} / {motion_length}", end="")

    import time; time.sleep(1.0 / fps)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
