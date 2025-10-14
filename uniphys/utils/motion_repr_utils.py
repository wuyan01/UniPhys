
REPR_LIST_DOF = [
             'dof_pose_6d', 'dof_vel',  # smplx-based local pose
]

REPR_LIST_DOF_JOINT = [
             'local_positions', 'local_vel',  # joint-based local pose
             'dof_pose_6d', 'dof_vel',  # smplx-based local pose
]

REPR_LIST_ROOT_DOF = [
             'root_rot_6d', 'root_rot_vel', 'root_trans', 'root_trans_vel',  # smplx-based traj
             'dof_pose_6d', 'dof_vel',  # smplx-based local pose
]
REPR_LIST_ROOT_DOF_JOINT = [
             'root_trans', 'root_rot_6d','root_trans_vel', 'root_rot_vel',  # smplx-based traj
             'local_positions', 'local_vel',  # joint-based local pose
             'dof_pose_6d', 'dof_vel',  # smplx-based local pose
]

REPR_LIST_DOF_NO_VEL = [
             'dof_pose_6d' # smplx-based local pose
]

REPR_LIST_DOF_JOINT_NO_VEL = [
             'local_positions',  # joint-based local pose
             'dof_pose_6d', # smplx-based local pose
]

REPR_LIST_ROOT_DOF_NO_VEL = [
             'root_rot_6d', 'root_rot_vel', 'root_trans', 'root_trans_vel',  # smplx-based traj
             'dof_pose_6d', # smplx-based local pose
]
REPR_LIST_ROOT_DOF_JOINT_NO_VEL = [
             'root_rot_6d', 'root_rot_vel', 'root_trans', 'root_trans_vel',  # smplx-based traj
             'local_positions',  # joint-based local pose
             'dof_pose_6d', # smplx-based local pose
]

# dimension for each categody of the motion representation
REPR_DIM_DICT = {
                 'root_rot_6d': 6,
                 'root_rot_vel': 3,
                 'root_trans': 3,
                 'root_trans_vel': 3,
                 'local_positions': 52 * 3,
                 'local_vel': 52 * 3,
                 'dof_pose_6d': 51 * 6,
                 'dof_vel': 51 * 3,
                 }

# dimension for each categody of the motion representation
REPR_DIM_DICT_SMPLX = {
                 'root_rot_6d': 6,
                 'root_rot_vel': 3,
                 'root_trans': 3,
                 'root_trans_vel': 3,
                 'local_positions': 52 * 3,
                 'local_vel': 52 * 3,
                 'dof_pose_6d': 51 * 6,
                 'dof_vel': 51 * 3,
                 }

# # dimension for each categody of the motion representation
# REPR_DIM_DICT_SMPL = {
#                  'root_rot_6d': 6,
#                  'root_rot_vel': 3,
#                  'root_trans': 3,
#                  'root_trans_vel': 3,
#                  'local_positions': 22 * 3,
#                  'local_vel': 22 * 3,
#                  'dof_pose_6d': 21 * 6,
#                  'dof_vel': 21 * 3,
#                  }

REPR_DIM_DICT_REAL_SMPL = {
                 'root_rot_6d': 6,
                 'root_rot_vel': 3,
                 'root_trans': 3,
                 'root_trans_vel': 3,
                 'local_positions': 24 * 3,
                 'local_vel': 24 * 3,
                 'dof_pose_6d': 23 * 6,
                 'dof_vel': 23 * 3,
                #  'self_obs': 358,
                 }

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from uniphys.utils.quaternion import *

face_joint_indx_robot = [5, 1]  # [r_hip, l_hip]

def cano_seq_smpl_or_smplx_batch(joint_positions_batch, dof_state_batch, root_state_batch):
    """
    canonicalize the sequence such that: 
    - frame 0 of the output sequence faces y+ axis
    - x/y coordinate of frame 0 is located at origin

    Input:
        - joint_positions_batch [B, T, J+1, 3]
        - dof_state_batch [B, T, J*3, 2]
        - root_state_batch [B, T, 3+4+3+3]
        - face_joint_indx_robot: indices for r_hip, l_hip
    Output:
        - cano_joint_positions_batch
        - cano_dof_state_batch
        - cano_root_state_batch
        - transf_matrix_batch
    """
    B, T, J, _ = joint_positions_batch.shape
    r_hip, l_hip = face_joint_indx_robot

    cano_joint_positions_batch = joint_positions_batch.copy()

    ######################## transl such that XY for frame 0 is at origin
    root_pos_init = cano_joint_positions_batch[:, 0]  # [B, 52, 3]
    root_pose_init_xy = root_pos_init[:, 0] * np.array([1, 1, 0])  # [B, 3]
    cano_joint_positions_batch = cano_joint_positions_batch - root_pose_init_xy[:, np.newaxis, np.newaxis, :]  # [B, T, 52, 3]

    ######################## transfrom such that frame 0 faces y+ axis
    joints_frame0 = cano_joint_positions_batch[:, 0]  # [B, 52, 3]
    across1 = joints_frame0[:, r_hip] - joints_frame0[:, l_hip]  # [B, 3]
    x_axis = across1
    x_axis[:, -1] = 0
    x_axis = x_axis / np.linalg.norm(x_axis, axis=1)[:, np.newaxis]
    z_axis = np.array([0, 0, 1.0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis, axis=1)[:, np.newaxis]
    transf_rotmat_batch = np.stack([x_axis, y_axis, np.tile(z_axis, (B, 1))], axis=1)  # [B, 3, 3]
    cano_joint_positions_batch = np.einsum('bij,btkj->btki', transf_rotmat_batch, cano_joint_positions_batch)  # [B, T, 52, 3]

    ######################## canonicalization transf matrix for root state
    root_pos_batch = root_state_batch[:, :, :3].copy()
    root_rot_quat_batch = root_state_batch[:, :, 3:7].copy()
    root_rot_mat_batch = R.from_quat(root_rot_quat_batch.reshape(-1, 4)).as_matrix().reshape(B, T, 3, 3)
    root_vel_batch = root_state_batch[:, :, 7:10].copy()
    root_rot_vel_batch = root_state_batch[:, :, 10:].copy()

    transf_matrix_1_batch = np.tile(np.array([[1.0, 0, 0, 0],
                                              [0, 1.0, 0, 0],
                                              [0, 0, 1.0, 0],
                                              [0, 0, 0, 1.0]]), (B, 1, 1))
    transf_matrix_1_batch[:, 0, 3] = -root_pose_init_xy[:, 0]
    transf_matrix_1_batch[:, 1, 3] = -root_pose_init_xy[:, 1]

    transf_matrix_2_batch = np.zeros([B, 4, 4])
    transf_matrix_2_batch[:, 0:3, 0:3] = transf_rotmat_batch#.transpose(0, 2, 1)
    transf_matrix_2_batch[:, -1, -1] = 1
    transf_matrix_batch = np.matmul(transf_matrix_2_batch, transf_matrix_1_batch)  # [B, 4, 4]

    delta_T_batch = joint_positions_batch[:, 0, 0] - root_pos_batch[:, 0]  ### should be 0

    body_mat_batch = np.zeros([B, T, 4, 4])
    body_mat_batch[:, :, :-1, :-1] = root_rot_mat_batch
    body_mat_batch[:, :, :-1, -1] = root_pos_batch + delta_T_batch[:, np.newaxis, :]
    body_mat_batch[:, :, -1, -1] = 1

    trans_to_target_origin_batch = np.expand_dims(transf_matrix_batch, axis=1)  # [B, 1, 4, 4]
    trans_to_target_origin_batch = np.repeat(trans_to_target_origin_batch, T, axis=1)  # [B, T, 4, 4]

    body_mat_new_batch = np.matmul(trans_to_target_origin_batch, body_mat_batch)  # [B, T, 4, 4]
    root_rot_quat_new_batch = R.from_matrix(body_mat_new_batch[:, :, :-1, :-1].reshape(-1, 3, 3)).as_quat().reshape(B, T, 4)
    root_pos_new_batch = body_mat_new_batch[:, :, :-1, -1].reshape(B, T, 3)

    root_vel_new_batch = np.matmul(root_vel_batch, transf_matrix_batch[:, :-1, :-1].transpose(0, 2, 1))
    root_rot_vel_new_batch = np.matmul(root_rot_vel_batch, transf_matrix_batch[:, :-1, :-1].transpose(0, 2, 1))

    root_state_new_batch = np.concatenate([root_pos_new_batch, root_rot_quat_new_batch, root_vel_new_batch, root_rot_vel_new_batch], axis=-1)
    
    return cano_joint_positions_batch, dof_state_batch, root_state_new_batch, transf_matrix_batch

def cano_seq_smpl_or_smplx(joint_positions, dof_state, root_state):
    """
    canonicalize the sequence such that: 
    - frame 0 of the output sequence faces y+ axis
    - x/y coordinate of frame 0 is located at origin

    Input:
        - joint_positions [T, J+1, 3], +1: root joint
        - dof_state [dof_pos, dof_vel]  [T, J*3, 2]
        - root_state [root_pos, root_rot, root_linear_vel, root_ang_vel]  [T, 3+4+3+3]
    Output:
        - cano_joint_positions
        - cano_dof_state
        - cano_root_state
    """
    cano_joint_positions = joint_positions.copy()
    bs = len(cano_joint_positions)
    r_hip, l_hip = face_joint_indx_robot

    ######################## transl such that XY for frame 0 is at origin
    root_pos_init = cano_joint_positions[0]  # [52, 3]
    root_pose_init_xy = root_pos_init[0] * np.array([1, 1, 0])
    cano_joint_positions = cano_joint_positions - root_pose_init_xy  # [T, 52, 3]

    ######################## transfrom such that frame 0 faces y+ axis
    joints_frame0 = cano_joint_positions[0] # [N, 3] joints of first frame
    across1 = joints_frame0[r_hip] - joints_frame0[l_hip]
    # across2 = joints_frame0[sdr_r] - joints_frame0[sdr_l]
    x_axis = across1# + across2
    x_axis[-1] = 0
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.array([0, 0, 1])
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    transf_rotmat = np.stack([x_axis, y_axis, z_axis], axis=1)  # [3, 3]
    cano_joint_positions = np.matmul(cano_joint_positions, transf_rotmat)  # [T(/bs), 22, 3]

    ######################## canonicalization transf matrix for root state
    root_pos = root_state[:, :3].copy()
    root_rot_quat = root_state[:, 3:7].copy()
    root_rot_mat = R.from_quat(root_rot_quat).as_matrix()
    root_vel = root_state[:, 7:10].copy()
    root_rot_vel = root_state[:, 10:].copy()

    transf_matrix_1 = np.array([[1, 0, 0, -root_pose_init_xy[0]],
                                [0, 1, 0, -root_pose_init_xy[1]],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
    transf_matrix_2 = np.zeros([4, 4])
    transf_matrix_2[0:3, 0:3] = transf_rotmat.T
    transf_matrix_2[-1, -1] = 1
    transf_matrix = np.matmul(transf_matrix_2, transf_matrix_1)

    delta_T = joint_positions[:, 0] - root_pos  ### should be 0

    body_mat = np.zeros([bs, 4, 4])
    body_mat[:, :-1, :-1] = root_rot_mat
    body_mat[:, :-1, -1] = root_pos + delta_T
    body_mat[:, -1, -1] = 1

    trans_to_target_origin = np.expand_dims(transf_matrix, axis=0)  # [1, 4, 4]
    trans_to_target_origin = np.repeat(trans_to_target_origin, bs, axis=0)

    body_mat_new = np.matmul(trans_to_target_origin, body_mat)  # [bs, 4, 4]
    root_rot_quat_new = R.from_matrix(body_mat_new[:, :-1,:-1]).as_quat().reshape(-1, 4)
    root_pos_new = body_mat_new[:, :-1, -1].reshape(-1, 3)
    root_vel_new = np.dot(root_vel, transf_matrix[:-1, :-1].T)
    root_rot_vel_new = np.matmul(root_rot_vel, transf_matrix[:-1, :-1].T)

    root_state_new = np.concatenate([root_pos_new, root_rot_quat_new, root_vel_new, root_rot_vel_new], axis=-1)

    return cano_joint_positions, dof_state, root_state_new, transf_matrix


def get_repr(joint_positions, dof_state, root_state, return_last=False):
    ##################### joint-based repr #####################

    global_joint_positions = joint_positions.copy()

    '''Get Forward Direction'''
    l_hip, r_hip = face_joint_indx_robot # [5, 1]
    across1 = joint_positions[:, r_hip] - joint_positions[:, l_hip]
    # across2 = positions[:, sdr_r] - positions[:, sdr_l]
    across = across1# + across2
    across[:, -1] = 0
    across = across / np.sqrt((across ** 2).sum(axis=-1))[:, np.newaxis]
    forward = np.cross(np.array([[0, 0, 1]]), across, axis=-1)
    forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

    '''Get Root Rotation and rotation velocity'''
    target = np.array([[0, 1, 0]]).repeat(len(forward), axis=0)
    root_rot_quat = qbetween_np(forward, target)
    root_rot_quat[0] = np.array([[0, 0.0, 0.0, 1.0]])  
    ### several frames have nan values
    if np.isnan(root_rot_quat).sum() > 0:

        frame_idx = np.unique(np.where(np.isnan(root_rot_quat) == True)[0])
        for idx in frame_idx:
            root_rot_quat[idx] = root_rot_quat[idx - 1]
    root_rot_quat_vel = qmul_np(root_rot_quat[1:], qinv_np(root_rot_quat[:-1]))

    '''abs root traj '''
    root_l_pos = joint_positions[:, 0]  # [seq_len, 3]
    root_height = joint_positions[:, 0, 2:3] # [seq_len]
    '''Get Root linear velocity'''
    root_l_vel = (joint_positions[1:, 0] - joint_positions[:-1, 0]).copy()
    root_l_vel = qrot_np(root_rot_quat[1:], root_l_vel)

    '''Get Root rotation angle and rot velocity angle'''
    root_rot_angle = np.arctan2(root_rot_quat[:, 3:4], root_rot_quat[:, 0:1])  # rotation angle, is half of the actual angle...
    root_rot_angle_vel = np.arctan2(root_rot_quat_vel[:, 3:4], root_rot_quat_vel[:, 0:1])  # rotation angle velocity


    """ local joint positions """
    '''local joint positions'''
    local_positions = joint_positions.copy()  # [seq_len, 22, 3]
    local_positions[..., 0] -= local_positions[:, 0:1, 0]
    local_positions[..., 1] -= local_positions[:, 0:1, 1]
    '''for each frame, local pose face y+'''
    local_positions = qrot_np(np.repeat(root_rot_quat[:, None], local_positions.shape[1], axis=1), local_positions)

    if np.sum(np.isnan(local_positions)) > 0:
        import ipdb; ipdb.set_trace()

    '''local joint velocity'''
    local_vel = qrot_np(np.repeat(root_rot_quat[:-1, None], global_joint_positions.shape[1], axis=1),
                        global_joint_positions[1:] - global_joint_positions[:-1])
    
    ''' root state '''
    robot_root_rot_quat = root_state[:, 3:7]
    robot_root_rot_mat = R.from_quat(robot_root_rot_quat).as_matrix().reshape(len(robot_root_rot_quat), 3, 3)
    robot_root_rot_6d = robot_root_rot_mat[..., :-1].reshape(robot_root_rot_mat.shape[0], 6)
    # robot_root_rot_6d_v1 = robot_root_rot_mat.reshape(robot_root_rot_mat.shape[0], 9)[..., :6]


    ''' dof pos '''
    dof_pose_aa = dof_state[..., 0].reshape(len(dof_state), -1, 3)
    dof_pose_rot_mat = R.from_rotvec(dof_pose_aa.reshape(-1, 3)).as_matrix().reshape(len(dof_pose_aa), -1, 3, 3)
    dof_pose_6d = dof_pose_rot_mat[..., :-1].reshape(len(dof_pose_aa), -1, 6)
    # dof_pose_6d = dof_pose_rot_mat.reshape(len(dof_pose_aa), -1, 9)[..., :6]


    """ debug to recover rotation representation of root and dof"""
    # recovered_dof_pose_rotmat = rot6d_to_rotmat(torch.from_numpy(dof_pose_6d.reshape(-1, 6)))
    # # recovered_dof_pose_rotmat[:, 1] = -recovered_dof_pose_rotmat[:, 1]
    # # recovered_dof_pose_rotmat[:, 2] = -recovered_dof_pose_rotmat[:, 2]
    # recovered_dof_pose_quat = rotmat_to_quat(recovered_dof_pose_rotmat)
    # recovered_dof_pose_exp_map = quat_to_exp_map(recovered_dof_pose_quat)

    # recovered_root_rot_rotmat = rot6d_to_rotmat(torch.from_numpy(robot_root_rot_6d.reshape(-1, 6)))
    # # recovered_root_rot_rotmat[:, 1] = -recovered_root_rot_rotmat[:, 1]
    # # recovered_root_rot_rotmat[:, 2] = -recovered_root_rot_rotmat[:, 2]
    # recovered_root_rot_quat = rotmat_to_quat(recovered_root_rot_rotmat)


    if return_last:
        data_dict = {
            'root_l_pos': root_l_pos[:],
            'root_height': root_height[:],
            'root_l_vel': root_l_vel,
            'root_rot_angle': root_rot_angle[:],
            'root_rot_angle_vel': root_rot_angle_vel,
            'local_root_rot_quat': root_rot_quat[:],

            'root_rot_6d': robot_root_rot_6d[:],
            'root_rot_vel': root_state[:, 10:],
            'root_trans': root_state[:, 0:3],
            'root_trans_vel': root_state[:, 7:10],
            'local_positions': local_positions[:].reshape(len(local_vel)+1, -1),
            'local_vel': np.concatenate([local_vel[0].reshape(1, -1), local_vel.reshape(len(local_vel), -1)], axis=0),
            'dof_pose_6d': dof_pose_6d[:].reshape(len(local_vel)+1, -1),
            'dof_vel': dof_state[:, ..., 1].reshape(len(local_vel)+1, -1),
            # 'cano_transf_matrix': cano_transf_matrix,
            'dof_pose_aa': dof_pose_aa[:].reshape(len(local_vel)+1, -1),
        }

    else:
        data_dict = {
            'root_l_pos': root_l_pos[0:-1],
            'root_height': root_height[0:-1],
            'root_l_vel': root_l_vel,
            'root_rot_angle': root_rot_angle[0:-1],
            'root_rot_angle_vel': root_rot_angle_vel,
            'local_root_rot_quat': root_rot_quat[0:-1],

            'root_rot_6d': robot_root_rot_6d[0:-1],
            'root_rot_vel': root_state[0:-1, 10:],
            'root_trans': root_state[0:-1, 0:3],
            'root_trans_vel': root_state[0:-1, 7:10],
            'local_positions': local_positions[0:-1].reshape(len(local_vel), -1),
            'local_vel': local_vel.reshape(len(local_vel), -1),
            'dof_pose_6d': dof_pose_6d[0:-1].reshape(len(local_vel), -1),
            'dof_vel': dof_state[0:-1, ..., 1].reshape(len(local_vel), -1),
            # 'cano_transf_matrix': cano_transf_matrix,
            'dof_pose_aa': dof_pose_aa[0:-1].reshape(len(local_vel), -1),
        }

    return data_dict


@torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))

@torch.jit.script
def exp_map_to_angle_axis(exp_map):
    min_theta = 1e-5

    angle = torch.norm(exp_map, dim=-1)
    angle_exp = torch.unsqueeze(angle, dim=-1)
    axis = exp_map / angle_exp
    angle = normalize_angle(angle)

    default_axis = torch.zeros_like(exp_map)
    default_axis[..., -1] = 1

    mask = torch.abs(angle) > min_theta
    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)

    return angle, axis


@torch.jit.script
def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

@torch.jit.script
def quat_unit(a):
    return normalize(a)

@torch.jit.script
def quat_from_angle_axis(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return quat_unit(torch.cat([xyz, w], dim=-1))

@torch.jit.script
def exp_map_to_quat(exp_map):
    angle, axis = exp_map_to_angle_axis(exp_map)
    q = quat_from_angle_axis(angle, axis)
    return q

# @torch.jit.script
def quat_to_rot6d(q):
    r = R.from_quat(q)
    return r.as_matrix().reshape(-1, 9)[:, :6]

def exp_map_to_rot6d(q):
    return quat_to_rot6d(exp_map_to_quat(q))



def compute_rotation_matrix(y_a, y_b):
    """
    Compute the rotation matrix that aligns coordinate system A to B.
    Args:
        y_a (torch.Tensor): y-axis vector of A, shape (B, 3).
        y_b (torch.Tensor): y-axis vector of B, shape (B, 3).
    Returns:
        torch.Tensor: Rotation matrix from A to B, shape (B, 3, 3).
    """
    # Normalize the y-axis vectors
    y_a = torch.nn.functional.normalize(y_a, dim=-1)
    y_b = torch.nn.functional.normalize(y_b, dim=-1)

    # Compute the z-axis (shared between A and B)
    z_a = torch.tensor([0.0, 0.0, 1.0], device=y_a.device).expand_as(y_a)
    z_b = z_a  # Both share the same z-axis

    # Compute the x-axis using the cross product
    x_a = torch.cross(y_a, z_a, dim=-1)
    x_b = torch.cross(y_b, z_b, dim=-1)

    # Normalize the x-axis
    x_a = torch.nn.functional.normalize(x_a, dim=-1)
    x_b = torch.nn.functional.normalize(x_b, dim=-1)

    # Recompute the y-axis to ensure orthonormality
    y_a = torch.cross(z_a, x_a, dim=-1)
    y_b = torch.cross(z_b, x_b, dim=-1)

    # Construct the basis for A and B
    basis_a = torch.stack([x_a, y_a, z_a], dim=-1)  # Shape: (B, 3, 3)
    basis_b = torch.stack([x_b, y_b, z_b], dim=-1)  # Shape: (B, 3, 3)

    # Compute the rotation matrix: R = B @ A^T
    rotation_matrix = torch.bmm(basis_b, basis_a.transpose(1, 2))
    return rotation_matrix


def recover_global_root_state(local_pos, local_rot6d, local_vel, cano_transf_rotmat, cano_transf_transl):
    
    cano_transf_rotmat_T = torch.from_numpy(cano_transf_rotmat.transpose(0, 2, 1)).float().cuda()# [B, 3, 3]
    cano_transf_transl = torch.from_numpy(cano_transf_transl).float().cuda()# [B, 3, 3]
    global_position = torch.einsum('bij,ntbj->ntbi', cano_transf_rotmat_T, local_pos) + cano_transf_transl
    global_velocity = torch.einsum('bij,ntbj->ntbi', cano_transf_rotmat_T, local_vel)

    r1 = local_rot6d[..., ::2].clone()
    r2 = local_rot6d[..., 1::2].clone()

    # Orthogonalize using Gram-Schmidt process
    r1 = F.normalize(r1, dim=-1)  # Normalize r1
    r2 = r2 - torch.sum(r1 * r2, dim=-1, keepdim=True) * r1  # Make r2 orthogonal to r1
    r2 = F.normalize(r2, dim=-1)  # Normalize r2

    # Compute the third column as cross-product of r1 and r2
    r3 = torch.cross(r1, r2, dim=-1)

    local_rotmat = torch.stack([r1, r2, r3], dim=-1).cuda()
    global_rotmat = torch.einsum('bij,ntbjk->ntbik', cano_transf_rotmat_T, local_rotmat)

    return global_position, global_velocity, global_rotmat

def recover_global_joints_position(local_joints_pred, global_pos, global_rotmat):
    ## recover global joint positions
    projected_global_rotmat = torch.zeros_like(global_rotmat)
    projected_global_rotmat[..., 0, 0] = global_rotmat[..., 0, 0]
    projected_global_rotmat[..., 0, 1] = global_rotmat[..., 0, 1]
    projected_global_rotmat[..., 1, 0] = global_rotmat[..., 1, 0]
    projected_global_rotmat[..., 1, 1] = global_rotmat[..., 1, 1]
    projected_global_rotmat[..., 2, 2] = 1.0  # Identity in z-axis

    forward_vector = global_rotmat[..., 0]
    target_vector = torch.zeros_like(forward_vector)
    target_vector[..., 1] = -1  # ? To check
    align_rotation_matrix = compute_rotation_matrix(target_vector.reshape(-1, 3), forward_vector.reshape(-1, 3))
    align_rotation_matrix = align_rotation_matrix.reshape(*global_rotmat.shape[:3], 3, 3)
    recovered_global_joints_pred = torch.matmul(local_joints_pred, align_rotation_matrix.transpose(3, 4))
    recovered_global_joints_pred[..., 0] = recovered_global_joints_pred[..., 0] + global_pos.unsqueeze(3)[..., 0]
    recovered_global_joints_pred[..., 1] = recovered_global_joints_pred[..., 1] + global_pos.unsqueeze(3)[..., 1]

    return recovered_global_joints_pred

def cano_goal(joint_positions_batch, goal_position):
    """
    Transform goal position into a canonical frame based on initial root position
    and facing direction of the character.

    Args:
        joint_positions_batch (np.ndarray): Joint positions of shape [B, T, J, 3],
                                            where B=batch, T=timesteps, J=num_joints.
        goal_position (np.ndarray): Goal positions, shape can be [B, 3], [B, T, 3],
                                    or sometimes with extra dims.

    Returns:
        cano_goal_position (np.ndarray): Goal position in canonical coordinates [B, T, 3].
        transf_rotmat_batch (np.ndarray): Rotation matrices for canonical transform [B, 3, 3].
        root_pose_init_xy (np.ndarray): Initial root position in XY plane [B, 3].
    """

    # ------------------------------------------------------------------
    # Ensure goal_position has shape [B, T, 3]
    # ------------------------------------------------------------------
    if goal_position.ndim == 2:       # [B, 3] → expand to [B, 1, 3]
        goal_position = goal_position.unsqueeze(1)
    elif goal_position.ndim == 4:     # Sometimes comes as [B, T, 1, 3] → squeeze
        goal_position = goal_position.squeeze()
    # otherwise assume [B, T, 3]

    # Indices of hips (used to determine facing direction in canonicalization)
    face_joint_indx_robot = [5, 1]   # [right_hip, left_hip]
    B, T, J, _ = joint_positions_batch.shape
    r_hip, l_hip = face_joint_indx_robot

    cano_joint_positions_batch = joint_positions_batch.copy()

    # ------------------------------------------------------------------
    # Step 1: Translate → put the root (frame 0) at the origin in XY
    # ------------------------------------------------------------------
    root_pos_init = joint_positions_batch[:, 0]              # [B, J, 3] at first frame
    root_pose_init_xy = root_pos_init[:, 0] * np.array([1, 1, 0])  # Take XY only, zero Z
    # Shift all joints so that root starts at origin in XY
    cano_joint_positions_batch = (
        cano_joint_positions_batch - root_pose_init_xy[:, np.newaxis, np.newaxis, :]
    )  # [B, T, J, 3]

    # Apply the same shift to the goal
    goal_position = goal_position - root_pose_init_xy[:, np.newaxis, :]  # [B, T, 3]

    # ------------------------------------------------------------------
    # Step 2: Rotate → align initial facing direction with +Y axis
    # ------------------------------------------------------------------
    joints_frame0 = cano_joint_positions_batch[:, 0]  # [B, J, 3] at first frame

    # Vector across hips defines local X axis (character's left–right direction)
    across1 = joints_frame0[:, r_hip] - joints_frame0[:, l_hip]  # [B, 3]
    x_axis = across1
    x_axis[:, -1] = 0  # Project to XY plane
    x_axis = x_axis / np.linalg.norm(x_axis, axis=1)[:, np.newaxis]  # Normalize

    # Z axis is always up
    z_axis = np.array([0, 0, 1.0])

    # Y axis = cross(Z, X) → character's forward direction
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis, axis=1)[:, np.newaxis]

    # Build rotation matrix [x, y, z] per batch
    transf_rotmat_batch = np.stack(
        [x_axis, y_axis, np.tile(z_axis, (B, 1))], axis=1
    )  # [B, 3, 3]

    # ------------------------------------------------------------------
    # Step 3: Apply transform to goal position
    # ------------------------------------------------------------------
    cano_goal_position = np.einsum(
        'bij,btj->bti', transf_rotmat_batch, goal_position
    )  # [B, T, 3]

    return cano_goal_position, transf_rotmat_batch, root_pose_init_xy
