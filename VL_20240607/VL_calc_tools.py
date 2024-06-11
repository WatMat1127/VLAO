import numpy as np
import torch

def calc_rot_matrix(rot_vec, rot_angle):
    rot_matrix = torch.tensor(np.zeros((3, 3)))
    rot_matrix[0][0] = torch.cos(rot_angle) + rot_vec[0] ** 2 * (1 - torch.cos(rot_angle))
    rot_matrix[0][1] = rot_vec[0] * rot_vec[1] * (1 - torch.cos(rot_angle)) - rot_vec[2] * torch.sin(rot_angle)
    rot_matrix[0][2] = rot_vec[2] * rot_vec[0] * (1 - torch.cos(rot_angle)) + rot_vec[1] * torch.sin(rot_angle)
    rot_matrix[1][0] = rot_vec[0] * rot_vec[1] * (1 - torch.cos(rot_angle)) + rot_vec[2] * torch.sin(rot_angle)
    rot_matrix[1][1] = torch.cos(rot_angle) + rot_vec[1] ** 2 * (1 - torch.cos(rot_angle))
    rot_matrix[1][2] = rot_vec[1] * rot_vec[2] * (1 - torch.cos(rot_angle)) - rot_vec[0] * torch.sin(rot_angle)
    rot_matrix[2][0] = rot_vec[2] * rot_vec[0] * (1 - torch.cos(rot_angle)) - rot_vec[1] * torch.sin(rot_angle)
    rot_matrix[2][1] = rot_vec[1] * rot_vec[2] * (1 - torch.cos(rot_angle)) + rot_vec[0] * torch.sin(rot_angle)
    rot_matrix[2][2] = torch.cos(rot_angle) + rot_vec[2] ** 2 * (1 - torch.cos(rot_angle))
    return rot_matrix

def calc_rot_matrix_vec2axis(vec_target, axis_target, additional_angle=0):
    cos_rot_angle = torch.dot(vec_target, axis_target) / torch.linalg.norm(vec_target) / torch.linalg.norm(axis_target)
    rot_angle = torch.acos(cos_rot_angle) + additional_angle
    if rot_angle.item() == 0:
        rot_matrix = torch.tensor(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    else:
        rot_vec = torch.cross(vec_target, axis_target) / torch.linalg.norm(torch.cross(vec_target, axis_target))
        rot_matrix = calc_rot_matrix(rot_vec, rot_angle)
    return rot_matrix

def calc_origin_LJpot(vec_axis, vec_p_1, vec_p, theta_val, l_val):
    cos_rot_angle = torch.dot(vec_axis, vec_p_1) / torch.linalg.norm(vec_axis) / torch.linalg.norm(vec_p_1)
    rot_angle = (theta_val - torch.acos(cos_rot_angle))
    rot_vec = torch.cross(vec_axis, vec_p_1) / torch.linalg.norm(torch.cross(vec_axis, vec_p_1))
    rot_matrix = calc_rot_matrix(rot_vec, rot_angle)
    tmp_vec = torch.matmul(rot_matrix, vec_p_1)
    vec_p_origin = tmp_vec / torch.linalg.norm(tmp_vec) * l_val
    vec_origin = vec_p + vec_p_origin
    return vec_origin

def calc_lonepair_axis(vec_p, vec_x_list):
    vec_p_axis = torch.tensor(np.array([0.0,0.0,0.0]))
    for ivec in vec_x_list:
        vec_p_x = ivec - vec_p
        vec_p_axis += vec_p_x / torch.linalg.norm(vec_p_x)
    return vec_p_axis

def calc_affine_rotate(rot_angle, axis):
    if axis == 'z':
        elem_1 = torch.stack([torch.cos(rot_angle), -1.0 * torch.sin(rot_angle), torch.tensor(0.0), torch.tensor(0.0)], dim=0)
        elem_2 = torch.stack([torch.sin(rot_angle),        torch.cos(rot_angle), torch.tensor(0.0), torch.tensor(0.0)], dim=0)
        elem_3 = torch.stack([   torch.tensor(0.0),           torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0)], dim=0)
        elem_4 = torch.stack([   torch.tensor(0.0),           torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)], dim=0)

    elif axis == 'x':
        elem_1 = torch.stack([torch.tensor(1.0),    torch.tensor(0.0),           torch.tensor(0.0), torch.tensor(0.0)], dim=0)
        elem_2 = torch.stack([torch.tensor(0.0), torch.cos(rot_angle), -1.0 * torch.sin(rot_angle), torch.tensor(0.0)], dim=0)
        elem_3 = torch.stack([torch.tensor(0.0), torch.sin(rot_angle),        torch.cos(rot_angle), torch.tensor(0.0)], dim=0)
        elem_4 = torch.stack([torch.tensor(0.0),    torch.tensor(0.0),           torch.tensor(0.0), torch.tensor(1.0)], dim=0)

    elif axis == 'y':
        elem_1 = torch.stack([       torch.cos(rot_angle), torch.tensor(0.0), torch.sin(rot_angle), torch.tensor(0.0)], dim=0)
        elem_2 = torch.stack([          torch.tensor(0.0), torch.tensor(1.0),    torch.tensor(0.0), torch.tensor(0.0)], dim=0)
        elem_3 = torch.stack([-1.0 * torch.sin(rot_angle), torch.tensor(0.0), torch.cos(rot_angle), torch.tensor(0.0)], dim=0)
        elem_4 = torch.stack([          torch.tensor(0.0), torch.tensor(0.0),    torch.tensor(0.0), torch.tensor(1.0)], dim=0)

    rot_matrix = torch.stack([elem_1, elem_2, elem_3, elem_4], dim=0)
    return rot_matrix

def calc_affine_translate(trans_vec):
    elem_1 = torch.stack([torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0), trans_vec[0]], dim=0)
    elem_2 = torch.stack([torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0), trans_vec[1]], dim=0)
    elem_3 = torch.stack([torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0), trans_vec[2]], dim=0)
    elem_4 = torch.stack([torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)], dim=0)
    trans_matrix = torch.stack([elem_1, elem_2, elem_3, elem_4], dim=0)
    return trans_matrix

def calc_affine_scaling(x_scale, y_scale, z_scale):
    elem_1 = torch.stack([          x_scale, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)], dim=0)
    elem_2 = torch.stack([torch.tensor(0.0),           y_scale, torch.tensor(0.0), torch.tensor(0.0)], dim=0)
    elem_3 = torch.stack([torch.tensor(0.0), torch.tensor(0.0),           z_scale, torch.tensor(0.0)], dim=0)
    elem_4 = torch.stack([torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)], dim=0)
    scaling_matrix = torch.stack([elem_1, elem_2, elem_3, elem_4], dim=0)
    return scaling_matrix

def calc_affine_xyz2axis(vec2origin, vec2z, vec2x, add_angle):
    # vec2origin: coordinate which is set for the origin
    # vec2z     : coordinate which is set for the z_axis
    # vec2x     : coordinate which is set for the x_axis
    # add_angle : angle for additional rotation around z-axis
    vec2origin = torch.cat((vec2origin, torch.tensor([1.0])), dim=0)
    vec2z = torch.cat((vec2z, torch.tensor([1.0])), dim=0)
    vec2x = torch.cat((vec2x, torch.tensor([1.0])), dim=0)

    # translation
    matrix_trans_1 = calc_affine_translate(-1 * vec2origin)
    # rotation along y-axis
    vec2z_after1 = torch.matmul(matrix_trans_1, vec2z)[:3]
    cos_y = vec2z_after1[2] / (vec2z_after1[0] ** 2 + vec2z_after1[2] ** 2) ** 0.5
    if vec2z_after1[0] >= 0:
        rot_angle_y = np.pi - torch.acos(cos_y)
    else:
        rot_angle_y = - np.pi + torch.acos(cos_y)
    matrix_rot_2 = calc_affine_rotate(rot_angle_y, 'y')
    matrix_12 = torch.matmul(matrix_rot_2, matrix_trans_1)
    # rotation along x-axis
    vec2z_after12 = torch.matmul(matrix_12, vec2z)[:3]
    cos_x = vec2z_after12[2] / (vec2z_after12[1] ** 2 + vec2z_after12[2] ** 2) ** 0.5
    if vec2z_after12[1] >= 0:
        rot_angle_x = - np.pi + torch.acos(cos_x)
    else:
        rot_angle_x = np.pi - torch.acos(cos_x)
    matrix_rot_3 = calc_affine_rotate(rot_angle_x, 'x')
    matrix_123 = torch.matmul(matrix_rot_3, matrix_12)

    # rotation along z-axis
    vec2x_after123 = torch.matmul(matrix_123, vec2x)[:3]
    cos_z = vec2x_after123[0] / ((vec2x_after123[0] ** 2 + vec2x_after123[1] ** 2) ** 0.5)

    if vec2x_after123[1] >= 0:
        rot_angle_z = -torch.acos(cos_z)
    elif vec2x_after123[1] < 0:
        rot_angle_z = torch.acos(cos_z)
    matrix_rot_4 = calc_affine_rotate(rot_angle_z, 'z')
    matrix_1234 = torch.matmul(matrix_rot_4, matrix_123)

    # rotation along z-axis based on phi
    rot_angle_z = add_angle
    matrix_rot_5 = calc_affine_rotate(rot_angle_z, 'z')
    matrix_12345 = torch.matmul(matrix_rot_5, matrix_1234)

    return matrix_12345
