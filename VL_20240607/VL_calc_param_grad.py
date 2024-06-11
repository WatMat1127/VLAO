import math
import copy
import numpy as np
import torch
import VL_class_files
from VL_keep_pot import calc_keep_pot_pgrad
from VL_keep_pyr_pot import calc_keep_pyr_pot_pgrad
from VL_LJ_asym_ell_pot import calc_LJ_asym_ell_pot_pgrad
from tools_unit_constant import unit_ang2au
from tools_unit_constant import unit_kcal2hartree
from tools_unit_constant import unit_deg2rad


def find_param_tag(info_tag):
    dat_info_tag = {}
    for iligand in info_tag:
        for ipot in info_tag[iligand]:
            for iterm in info_tag[iligand][ipot]:
                if '@@' in str(info_tag[iligand][ipot][iterm]):
                    param_tag_tmp = info_tag[iligand][ipot][iterm]
                    if param_tag_tmp not in dat_info_tag.keys():
                        dat_info_tag[param_tag_tmp] = []
                    dat_info_tag[param_tag_tmp].append([iligand, ipot, iterm])
    return dat_info_tag

def VL_param_grad(xyz, param, param_tag, MM_param_list, path_phi_log):
    dat_param_grad = {}

    ###-----
    # calculate grad for keep pot (the keep potential)
    ###-----
    dat_keep_info_tag = find_param_tag(param_tag.keep_info)

    tensor_list = []
    tensor_tag = []
    for itag in dat_keep_info_tag:
        term_0 = dat_keep_info_tag[itag][0]
        tensor_list.append(torch.tensor(param.keep_info[term_0[0]][term_0[1]][term_0[2]], requires_grad=True))
        tensor_tag.append(itag)

    if len(tensor_tag) >= 1:
        grad_keeppot = torch.func.jacfwd(calc_keep_pot_pgrad, argnums=2)(xyz, param_tag.keep_info, tensor_list, tensor_tag)
        for itag in range(len(tensor_tag)):
            dat_param_grad[tensor_tag[itag]] = grad_keeppot[itag].item()

    ###-----
    # calculate grad for keep pyr pot (the keep angle potential)
    ###-----
    dat_keeppyr_info_tag = find_param_tag(param_tag.keeppyr_info)

    tensor_list = []
    tensor_tag = []
    for itag in dat_keeppyr_info_tag:
        term_0 = dat_keeppyr_info_tag[itag][0]
        tensor_list.append(torch.tensor(param.keeppyr_info[term_0[0]][term_0[1]][term_0[2]], requires_grad=True))
        tensor_tag.append(itag)

    if len(tensor_tag) >= 1:
        grad_keeppyrpot = torch.func.jacfwd(calc_keep_pyr_pot_pgrad, argnums=2)(xyz, param_tag.keeppyr_info, tensor_list, tensor_tag)
        for itag in range(len(tensor_tag)):
            dat_param_grad[tensor_tag[itag]] = grad_keeppyrpot[itag].item()


    ###-----
    # calculate grad for LJpot_asym_ell (the ovoid LJ potential)
    ###-----
    dat_LJ_asym_ell_info_tag = find_param_tag(param_tag.LJ_asym_ell_info)

    tensor_list = []
    tensor_tag = []
    for itag in dat_LJ_asym_ell_info_tag:
        term_0 = dat_LJ_asym_ell_info_tag[itag][0]
        tensor_list.append(torch.tensor(param.LJ_asym_ell_info[term_0[0]][term_0[1]][term_0[2]], requires_grad=True))
        tensor_tag.append(itag)

    if len(tensor_tag) >= 1:
        phi_log = VL_class_files.PhiLog(path_phi_log)
        phi_list = torch.tensor(phi_log.phi_list, requires_grad=True, dtype=torch.float64)
        grad_LJ_pot_asym_ell = torch.func.jacfwd(calc_LJ_asym_ell_pot_pgrad, argnums=4)(xyz, phi_list, MM_param_list, param_tag.LJ_asym_ell_info, tensor_list, tensor_tag)
        for itag in range(len(tensor_tag)):
            dat_param_grad[tensor_tag[itag]] = grad_LJ_pot_asym_ell[itag].item()
        del grad_LJ_pot_asym_ell

    ###-----
    # output a param_grad file
    ###-----
    dat_convert = {}
    dat_convert['@@r0@@'] = unit_ang2au()
    dat_convert['@@a1@@'] = unit_ang2au()
    dat_convert['@@a2@@'] = unit_ang2au()
    dat_convert['@@b1@@'] = unit_ang2au()
    dat_convert['@@b2@@'] = unit_ang2au()
    dat_convert['@@c1@@'] = unit_ang2au()
    dat_convert['@@c2@@'] = unit_ang2au()

    dat_convert['@@r0_1@@'] = unit_ang2au()
    dat_convert['@@ang_1@@'] = unit_deg2rad()
    dat_convert['@@a1_1@@'] = unit_ang2au()
    dat_convert['@@a2_1@@'] = unit_ang2au()
    dat_convert['@@b1_1@@'] = unit_ang2au()
    dat_convert['@@b2_1@@'] = unit_ang2au()
    dat_convert['@@c1_1@@'] = unit_ang2au()
    dat_convert['@@c2_1@@'] = unit_ang2au()
    dat_convert['@@dist_1@@'] = unit_ang2au()

    dat_convert['@@r0_2@@'] = unit_ang2au()
    dat_convert['@@ang_2@@'] = unit_deg2rad()
    dat_convert['@@a1_2@@'] = unit_ang2au()
    dat_convert['@@a2_2@@'] = unit_ang2au()
    dat_convert['@@b1_2@@'] = unit_ang2au()
    dat_convert['@@b2_2@@'] = unit_ang2au()
    dat_convert['@@c1_2@@'] = unit_ang2au()
    dat_convert['@@c2_2@@'] = unit_ang2au()
    dat_convert['@@dist_2@@'] = unit_ang2au()

    dat_convert['@@r0_3@@'] = unit_ang2au()
    dat_convert['@@ang_3@@'] = unit_deg2rad()
    dat_convert['@@a1_3@@'] = unit_ang2au()
    dat_convert['@@a2_3@@'] = unit_ang2au()
    dat_convert['@@b1_3@@'] = unit_ang2au()
    dat_convert['@@b2_3@@'] = unit_ang2au()
    dat_convert['@@c1_3@@'] = unit_ang2au()
    dat_convert['@@c2_3@@'] = unit_ang2au()
    dat_convert['@@dist_3@@'] = unit_ang2au()

    dat_convert['@@r0_4@@'] = unit_ang2au()
    dat_convert['@@ang_4@@'] = unit_deg2rad()
    dat_convert['@@a1_4@@'] = unit_ang2au()
    dat_convert['@@a2_4@@'] = unit_ang2au()
    dat_convert['@@b1_4@@'] = unit_ang2au()
    dat_convert['@@b2_4@@'] = unit_ang2au()
    dat_convert['@@c1_4@@'] = unit_ang2au()
    dat_convert['@@c2_4@@'] = unit_ang2au()
    dat_convert['@@dist_4@@'] = unit_ang2au()

    dat_convert['@@r0_5@@'] = unit_ang2au()
    dat_convert['@@ang_5@@'] = unit_deg2rad()
    dat_convert['@@a1_5@@'] = unit_ang2au()
    dat_convert['@@a2_5@@'] = unit_ang2au()
    dat_convert['@@b1_5@@'] = unit_ang2au()
    dat_convert['@@b2_5@@'] = unit_ang2au()
    dat_convert['@@c1_5@@'] = unit_ang2au()
    dat_convert['@@c2_5@@'] = unit_ang2au()
    dat_convert['@@dist_5@@'] = unit_ang2au()

    dat_convert['@@r0_6@@'] = unit_ang2au()
    dat_convert['@@ang_6@@'] = unit_deg2rad()
    dat_convert['@@a1_6@@'] = unit_ang2au()
    dat_convert['@@a2_6@@'] = unit_ang2au()
    dat_convert['@@b1_6@@'] = unit_ang2au()
    dat_convert['@@b2_6@@'] = unit_ang2au()
    dat_convert['@@c1_6@@'] = unit_ang2au()
    dat_convert['@@c2_6@@'] = unit_ang2au()
    dat_convert['@@dist_6@@'] = unit_ang2au()

    with open(param.path + '_grad', mode='w') as g:
        for itag in dat_param_grad:
            if itag in dat_convert.keys():
                g.write('{0:<15} {1}\n'.format(itag, dat_param_grad[itag] * dat_convert[itag]))
            else:
                g.write('{0:<15} {1} [CAUTION] unit \n'.format(itag, dat_param_grad[itag]))
