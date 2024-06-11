import numpy as np
import copy
import math


def calc_f_val_grad(qm_ene_list, grad_list):
    A_b_ene = qm_ene_list[0]
    A_l_ene = qm_ene_list[1]
    B_b_ene = qm_ene_list[2]
    B_l_ene = qm_ene_list[3]
    C_b_ene = qm_ene_list[4]
    C_l_ene = qm_ene_list[5]
    D_b_ene = qm_ene_list[6]
    D_l_ene = qm_ene_list[7]

    A_b_rel_ene = (A_b_ene - D_b_ene) * 627.51
    A_l_rel_ene = (A_l_ene - D_b_ene) * 627.51
    B_b_rel_ene = (B_b_ene - D_b_ene) * 627.51
    B_l_rel_ene = (B_l_ene - D_b_ene) * 627.51
    C_b_rel_ene = (C_b_ene - D_b_ene) * 627.51
    C_l_rel_ene = (C_l_ene - D_b_ene) * 627.51
    D_b_rel_ene = (D_b_ene - D_b_ene) * 627.51
    D_l_rel_ene = (D_l_ene - D_b_ene) * 627.51

    const_r = 8.31446261815324 / 4184
    const_t = 233.15

    A_b_exp = np.exp(-1 * A_b_rel_ene / (const_r * const_t))
    A_l_exp = np.exp(-1 * A_l_rel_ene / (const_r * const_t))
    B_b_exp = np.exp(-1 * B_b_rel_ene / (const_r * const_t))
    B_l_exp = np.exp(-1 * B_l_rel_ene / (const_r * const_t))
    C_b_exp = np.exp(-1 * C_b_rel_ene / (const_r * const_t))
    C_l_exp = np.exp(-1 * C_l_rel_ene / (const_r * const_t))
    D_b_exp = np.exp(-1 * D_b_rel_ene / (const_r * const_t))
    D_l_exp = np.exp(-1 * D_l_rel_ene / (const_r * const_t))

    sum_all = A_b_exp + A_l_exp + B_b_exp + B_l_exp + C_b_exp + C_l_exp + D_b_exp + D_l_exp
    sum_l = A_l_exp + B_l_exp + C_l_exp + D_l_exp
    sum_b = A_b_exp + B_b_exp + C_b_exp + D_b_exp
    sum_b_s = A_b_exp + D_b_exp
    sum_b_r = B_b_exp + C_b_exp

    rr = sum_b / sum_all * 100
    ee = (sum_b_s - sum_b_r) / (sum_b_s + sum_b_r) * 100
    f_val = (rr + ee) * (-1)

    print('rree', rr, ee)

    A_b_grad = grad_list[0] * 627.51
    A_l_grad = grad_list[1] * 627.51
    B_b_grad = grad_list[2] * 627.51
    B_l_grad = grad_list[3] * 627.51
    C_b_grad = grad_list[4] * 627.51
    C_l_grad = grad_list[5] * 627.51
    D_b_grad = grad_list[6] * 627.51
    D_l_grad = grad_list[7] * 627.51

    d_sum_all = -1 / (const_r * const_t) * (
                A_b_exp * A_b_grad + A_l_exp * A_l_grad + B_b_exp * B_b_grad + B_l_exp * B_l_grad + C_b_exp * C_b_grad + C_l_exp * C_l_grad + D_b_exp * D_b_grad + D_l_exp * D_l_grad)
    d_sum_b = -1 / (const_r * const_t) * (
                A_b_exp * A_b_grad + B_b_exp * B_b_grad + C_b_exp * C_b_grad + D_b_exp * D_b_grad)
    d_sum_b_s = -1 / (const_r * const_t) * (A_b_exp * A_b_grad + D_b_exp * D_b_grad)
    d_sum_b_r = -1 / (const_r * const_t) * (B_b_exp * B_b_grad + C_b_exp * C_b_grad)

    d_rr = (d_sum_b * sum_all - sum_b * d_sum_all) / (sum_all) ** 2
    d_ee = ((d_sum_b_s - d_sum_b_r) * (sum_b_s + sum_b_r) - (sum_b_s - sum_b_r) * (d_sum_b_s + d_sum_b_r)) / (
                sum_b_s + sum_b_r) ** 2
    f_grad = (d_rr + d_ee) * (-1) * 100
    return f_val, f_grad
