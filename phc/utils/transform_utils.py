import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation as R

def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))

import torch

def rot6d_to_rotmat(x):
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Args:
        x (torch.Tensor): (B,6) Batch of 6-D rotation representations.
    Returns:
        torch.Tensor: Batch of corresponding rotation matrices with shape (B,3,3).
    """
    # if rot6d_mode == 'prohmr':
    #     x = x.reshape(-1,2,3).permute(0, 2, 1).contiguous()
    # elif rot6d_mode == 'diffusion':
    x = x.reshape(-1, 3, 2)
    ### note: order for 6d feture items different between diffusion and prohmr code!!!
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


# def rot6d_to_rotmat(rot_6d):
#     """
#     Convert a batch of 6D rotation vectors to a batch of 3x3 rotation matrices.
    
#     Parameters:
#     rot_6d (torch.Tensor): A tensor of shape (B, 6) representing the first two rows of the rotation matrices.
    
#     Returns:
#     torch.Tensor: A tensor of shape (B, 3, 3) representing the batch of 3x3 rotation matrices.
#     """
#     assert rot_6d.shape[1] == 6, "Each rotation vector must have 6 elements."
    
#     # Reshape the input to extract the first two rows of the rotation matrices
#     r1 = rot_6d[:, :3]
#     r2 = rot_6d[:, 3:]
    
#     # Normalize r1 and r2 to ensure they are orthonormal
#     r1 = r1 / r1.norm(dim=1, keepdim=True)
#     r2 = r2 / r2.norm(dim=1, keepdim=True)
    
#     # Ensure r2 is orthogonal to r1 by projecting out any component in the direction of r1
#     dot_product = (r2 * r1).sum(dim=1, keepdim=True)
#     r2 = r2 - dot_product * r1
#     r2 = r2 / r2.norm(dim=1, keepdim=True)
    
#     # Compute the third row as the cross product of the first two rows
#     r3 = torch.cross(r1, r2, dim=1)
    
#     # Stack the rows to form the rotation matrices
#     rotation_matrix = torch.stack([r1, r2, r3], dim=1)#.permute(0, 2, 1)
    
#     return rotation_matrix

# def rot6d_to_rotmat(x):
#     x = x.view(-1,3,2)

#     # Normalize the first vector
#     b1 = F.normalize(x[:, :, 0], dim=1, eps=1e-6)

#     dot_prod = torch.sum(b1 * x[:, :, 1], dim=1, keepdim=True)
#     # Compute the second vector by finding the orthogonal complement to it
#     b2 = F.normalize(x[:, :, 1] - dot_prod * b1, dim=-1, eps=1e-6)

#     # Finish building the basis by taking the cross product
#     b3 = torch.cross(b1, b2, dim=1)
#     rot_mats = torch.stack([b1, b2, b3], dim=-1)

#     return rot_mats

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

# def rotmat_to_quat(x):
#     r = R.from_matrix(x)
#     return r.as_quat()
def rotmat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(torch.stack(
        [
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
            1.0 + m00 + m11 + m22,
        ],
        dim=-1,
    ))

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_ijkr = torch.stack(
        [
            torch.stack([q_abs[..., 0]**2, m10 + m01, m02 + m20, m21 - m12], dim=-1),
            torch.stack([m10 + m01, q_abs[..., 1]**2, m21 + m12, m02 - m20], dim=-1),
            torch.stack([m02 + m20, m12 + m21, q_abs[..., 2]**2, m10 - m01], dim=-1),
            torch.stack([m21 - m12, m02 - m20, m10 - m01, q_abs[..., 3]**2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_ijkr / (2.0 * q_abs[..., None].max(flr))

    return quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))  # pyre-ignore[16]



def rot6d_to_quat(x):
    # print(x.shape, x[0])
    matrix = rot6d_to_rotmat(x)
    # print(matrix.shape, matrix[0])
    quat = rotmat_to_quat(matrix)
    # print(quat.shape, quat[0])
    return quat

def quat_to_angle_axis(q):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    # computes axis-angle representation from quaternion q
    # q must be normalized
    min_theta = 1e-5
    qx, qy, qz, qw = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qw] / sin_theta_expand

    mask = torch.abs(sin_theta) > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis


@torch.jit.script
def angle_axis_to_exp_map(angle, axis):
    # type: (Tensor, Tensor) -> Tensor
    # compute exponential map from axis-angle
    angle_expand = angle.unsqueeze(-1)
    exp_map = angle_expand * axis
    return exp_map


@torch.jit.script
def quat_to_exp_map(q):
    # type: (Tensor) -> Tensor
    # compute exponential map from quaternion
    # q must be normalized
    angle, axis = quat_to_angle_axis(q)
    exp_map = angle_axis_to_exp_map(angle, axis)
    return exp_map

def rot6d_to_exp_map(x):
    matrix = rot6d_to_rotmat(x)
    quat = rotmat_to_quat(matrix)
    exp_map = quat_to_exp_map(quat)
    return exp_map

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

# @torch.jit.script
# def exp_map_to_rot6d(q):
#     # type: (Tensor) -> Tensor
#     # compute exponential map from quaternion
#     # q must be normalized
#     angle_axis = exp_map_to_angle_axis(q)
#     matrix = p3dt.angle_axis_to_matrix(angle_axis)
#     rot6d = p3dt.matrix_to_rot6d(matrix)
#     return rot6d

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

# @torch.jit.script
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