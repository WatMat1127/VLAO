#!/path/to/python

import numpy as np
import os
import sys
import torch
import subprocess
import VL_class_files
import VL_class_penarty
import VL_calc_param_grad


#####-----------------
# get paths for target files
#####-----------------
path_prog = sys.argv[0]
dir_prog = os.path.dirname(sys.argv[0])

path_LinkJOB = sys.argv[1]
path_LinkJOB_write = path_LinkJOB + '_final'
path_fn_top = sys.argv[1].split('_LinkJOB')[0]
path_param = path_fn_top + '.param'
path_param_tag = path_fn_top + '.param_tag'
path_param_grad = path_fn_top + '.param_grad'
path_com = path_fn_top + '.com'
path_phi_log = path_fn_top + '.phi_log'
if dir_prog != '':
    path_MM_param = os.path.dirname(sys.argv[0]) + '/MM_param.txt'
else:
    path_MM_param = 'MM_param.txt'

#####-----------------
# read LinkJOB, param and com files
#####-----------------
linkjob = VL_class_files.LinkJOB(path_LinkJOB)
comfile = VL_class_files.ComFile(path_com)
param = VL_class_files.ParamFile(path_param, comfile.natom_all)

#####-----------------
# combine frozen atoms if needed
#####-----------------
if comfile.tag_frozen == True:
    all_xyz_list = linkjob.xyz + comfile.frozen_xyz
else:
    all_xyz_list = linkjob.xyz

atom_order = []
xyz_list = []
for iatom in all_xyz_list:
    atom_order.append(iatom[0])
    xyz_list.append(iatom[1])
xyz = torch.tensor(np.array(xyz_list), requires_grad=True)

#####-----------------
# get UFF_parameters
#####-----------------
MM_param = VL_class_files.MMParam(path_MM_param)
MM_param_list = MM_param.make_list(atom_order)


#####-----------------
# run main　processes
#####-----------------
if len(sys.argv) >= 3 and 'param_grad' in sys.argv[2]:
    #####-----------------
    #   calculate dE/dp_VL
    #####-----------------
    param_tag = VL_class_files.ParamFile(path_param_tag, comfile.natom_all)
    dat_param_grad = VL_calc_param_grad.VL_param_grad(xyz, param, param_tag, MM_param_list, path_phi_log)

else:
    #####-----------------
    #   calculate penalty, gradient and Hessian
    #####-----------------
    keep_pot = VL_class_penarty.Penarty(linkjob.natom_active)
    keep_pot.add_keep_pot(xyz, param.keep_info)

    keeppyr_pot = VL_class_penarty.Penarty(linkjob.natom_active)
    keeppyr_pot.add_keep_pyr_pot(xyz, param.keeppyr_info)

    LJ_asym_ell_pot = VL_class_penarty.Penarty(linkjob.natom_active)
    LJ_asym_ell_pot.add_LJ_asym_ell_pot(xyz, param.LJ_asym_ell_info, MM_param_list, path_phi_log, atom_order)

    total_penarty = VL_class_penarty.Penarty(linkjob.natom_active)
    total_penarty.combine_penarties([keep_pot, keeppyr_pot, LJ_asym_ell_pot])

    linkjob.add_penarty(total_penarty)
    linkjob.write_LinkJOB(path_LinkJOB_write)

    ## ---------------------------
    ## overwrite to the original LinkJOB file
    ## ---------------------------
    cmd_cp_1 = "cp {0} {0}_old".format(path_LinkJOB)
    cmd_cp_2 = "cp %s %s" % (path_LinkJOB_write, path_LinkJOB)
    subprocess.call(cmd_cp_1, shell=True)
    subprocess.call(cmd_cp_2, shell=True)

