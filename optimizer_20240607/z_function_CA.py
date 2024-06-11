import numpy as np
import copy
import math


def calc_f_val_grad(qm_ene_list, grad_list):
    A_b_ene = qm_ene_list[0]
    B_b_ene = qm_ene_list[1]

    A_b_rel_ene = (A_b_ene - B_b_ene) * 627.51
    B_b_rel_ene = (B_b_ene - B_b_ene) * 627.51

    const_r = 8.31446261815324 / 4184
    const_t = 300.00

    A_b_exp = np.exp(-1 * A_b_rel_ene / (const_r * const_t))
    B_b_exp = np.exp(-1 * B_b_rel_ene / (const_r * const_t))

    sum_all = A_b_exp + B_b_exp
    sum_b = B_b_exp

    rr = sum_b / sum_all * 100
    f_val = (rr) * (-1)

    A_b_grad = grad_list[0] * 627.51
    B_b_grad = grad_list[1] * 627.51

    d_sum_all = -1 / (const_r * const_t) * (A_b_exp * A_b_grad + B_b_exp * B_b_grad)
    d_sum_b = -1 / (const_r * const_t) * (B_b_exp * B_b_grad)

    d_rr = (d_sum_b * sum_all - sum_b * d_sum_all) / (sum_all) ** 2
    f_grad = (d_rr) * (-1) * 100
    return f_val, f_grad
