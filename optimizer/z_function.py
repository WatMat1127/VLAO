import numpy as np
import copy
import math


def calc_f_val_grad(qm_ene_list, grad_list):
    eq_ene = qm_ene_list[0]
    ts_ene = qm_ene_list[1]

    f_val = (ts_ene - eq_ene) * 627.51
    eq_grad = grad_list[0]
    ts_grad = grad_list[1]

    f_grad = (ts_grad - eq_grad) * 627.51
    return f_val, f_grad
