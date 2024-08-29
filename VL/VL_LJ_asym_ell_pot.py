import copy
import numpy as np
import os
import VL_class_files
from VL_calc_tools import calc_lonepair_axis
from VL_calc_tools import calc_origin_LJpot
from VL_calc_tools import calc_affine_rotate
from VL_calc_tools import calc_affine_xyz2axis
from tools_unit_constant import unit_au2ang
import torch


def conbine_xyz_phi(xyz, phi_list):
    xyz_vec = torch.reshape(xyz, (-1, 1)).squeeze()
    xyz_phi_tensor = torch.cat((xyz_vec, phi_list), dim=0)
    return xyz_phi_tensor


def calc_ene_phi_tensor(
    xyz,
    phi_list,
    MM_param_list,
    LJ_asym_ell_info,
    log_tag=False,
    atom_order=None,
    path_phi_log=None,
):
    xyz_phi_tensor = conbine_xyz_phi(xyz, phi_list)
    add_ene = calc_ene(
        xyz_phi_tensor,
        MM_param_list,
        LJ_asym_ell_info,
        log_tag,
        atom_order,
        path_phi_log,
    )
    return add_ene


def calc_ene(
    xyz_phi_tensor,
    MM_param_list,
    LJ_asym_ell_info,
    log_tag=False,
    atom_order=None,
    path_phi_log=None,
):
    add_ene = 0

    natom = len(MM_param_list)
    xyz_tmp = []
    for iatom in range(natom):
        tmp_tensor = torch.stack(
            [
                xyz_phi_tensor[3 * iatom + 0],
                xyz_phi_tensor[3 * iatom + 1],
                xyz_phi_tensor[3 * iatom + 2],
                torch.tensor(1.0, dtype=torch.float64),
            ],
            dim=0,
        )
        xyz_tmp.append(tmp_tensor)

    xyz_aff = torch.stack(xyz_tmp, dim=0)
    phi_list = xyz_phi_tensor[3 * natom :]
    dat_matrix = {}
    dat_pot_num = {}
    n_ipot = 0
    pre_rot_matrix_x = calc_affine_rotate(torch.tensor(1.0, dtype=torch.float64), "x")
    pre_rot_matrix_y = calc_affine_rotate(torch.tensor(2.0, dtype=torch.float64), "y")
    pre_rot_matrix_z = calc_affine_rotate(torch.tensor(3.0, dtype=torch.float64), "z")
    pre_rot_matrix = torch.matmul(
        pre_rot_matrix_z, torch.matmul(pre_rot_matrix_y, pre_rot_matrix_x)
    )
    xyz_aff = torch.matmul(pre_rot_matrix, xyz_aff.T).T

    for ilig in LJ_asym_ell_info:
        ######------
        # calculate vec_p and vec_p_axiz
        ######------
        vec_p = xyz_aff[LJ_asym_ell_info[ilig][0]["atom_number_P"]][:3]
        vec_x_list = []
        for jpot in LJ_asym_ell_info[ilig]:
            atom_number_xi = LJ_asym_ell_info[ilig][jpot]["atom_number_xi"]
            vec_xi = xyz_aff[atom_number_xi][:3]
            vec_x_list.append(vec_xi)
        if len(vec_x_list) == 1:
            vec_x_list.append(
                torch.tensor(np.array([1.0, 0.0, 0.0]), dtype=torch.float64)
            )
        vec_p_axis = calc_lonepair_axis(vec_p, vec_x_list)

        for ipot in LJ_asym_ell_info[ilig]:
            ######------
            # read data
            ######------
            ipot_num = n_ipot + ipot
            dat_pot_num[ipot_num] = [ilig, ipot]

            phi = phi_list[n_ipot + ipot]
            dist_p_origin = LJ_asym_ell_info[ilig][ipot]["dist"]

            ######------
            # calculate vec4transformation
            ######------
            vec_1 = xyz_aff[LJ_asym_ell_info[ilig][ipot]["atom_number_xi"]][:3]
            vec_p_1 = vec_1 - vec_p
            cos = (
                torch.dot(vec_p_1, vec_p_axis)
                / torch.linalg.norm(vec_p_1)
                / torch.linalg.norm(vec_p_axis)
            )
            theta_val = torch.acos(cos)
            vec_origin = calc_origin_LJpot(
                vec_p_axis, vec_p_1, vec_p, theta_val, dist_p_origin
            )
            vec_p_origin = vec_origin - vec_p
            vec_p_std = torch.cross(vec_p_origin, vec_p_axis) / torch.linalg.norm(
                torch.cross(vec_p_origin, vec_p_axis)
            )
            vec_std = vec_p_std + vec_p

            ######------
            # coordinate transformation
            ######------
            # translation and rotation
            matrix_tr = calc_affine_xyz2axis(vec_origin, vec_p, vec_std, phi)
            dat_matrix[ipot_num] = matrix_tr
            natom_flg = len(xyz_aff)
            dat_pot_num[ipot_num].append(natom_flg)

            vec_origin_tmp = torch.cat((vec_origin, torch.tensor([1.0])), dim=0)
            vec_origin_tmp = torch.reshape(vec_origin_tmp, (4, 1))
            xyz_aff = torch.cat([xyz_aff, vec_origin_tmp.T], dim=0)

            #####------
            # for visualization
            #####------
            if log_tag:
                a1_val = LJ_asym_ell_info[ilig][ipot]["a1_val"]
                a2_val = LJ_asym_ell_info[ilig][ipot]["a2_val"]
                b1_val = LJ_asym_ell_info[ilig][ipot]["b1_val"]
                b2_val = LJ_asym_ell_info[ilig][ipot]["b2_val"]
                c1_val = LJ_asym_ell_info[ilig][ipot]["c1_val"]
                c2_val = LJ_asym_ell_info[ilig][ipot]["c2_val"]

                xyz_aff = torch.matmul(matrix_tr, xyz_aff.T).T
                xyz_dummy_list = np.array(
                    [
                        [a1_val, 0.0, 0.0, 1.0],
                        [0.0, b1_val, 0.0, 1.0],
                        [0.0, 0.0, c1_val, 1.0],
                        [-1 * a2_val, 0.0, 0.0, 1.0],
                        [0.0, -1 * b2_val, 0.0, 1.0],
                        [0.0, 0.0, -1 * c2_val, 1.0],
                    ]
                )
                xyz_dummy = torch.tensor(xyz_dummy_list, dtype=torch.float64)
                xyz_aff = torch.cat([xyz_aff, xyz_dummy], dim=0)
                xyz_aff = torch.matmul(torch.linalg.inv(matrix_tr), xyz_aff.T).T

        ######------
        # add number of pot
        ######------
        n_ipot = n_ipot + len(LJ_asym_ell_info[ilig])

    ######------
    # penarty calculation
    ######------
    for ipot_num in dat_pot_num:
        ilig = dat_pot_num[ipot_num][0]
        ipot = dat_pot_num[ipot_num][1]

        epsilon_1 = LJ_asym_ell_info[ilig][ipot]["eps"]
        a1_val = LJ_asym_ell_info[ilig][ipot]["a1_val"]
        a2_val = LJ_asym_ell_info[ilig][ipot]["a2_val"]
        b1_val = LJ_asym_ell_info[ilig][ipot]["b1_val"]
        b2_val = LJ_asym_ell_info[ilig][ipot]["b2_val"]
        c1_val = LJ_asym_ell_info[ilig][ipot]["c1_val"]
        c2_val = LJ_asym_ell_info[ilig][ipot]["c2_val"]
        target_atoms = LJ_asym_ell_info[ilig][ipot]["target_atoms"]

        sgm_a1 = 2 * 2 ** (1 / 6.0) * a1_val
        sgm_a2 = 2 * 2 ** (1 / 6.0) * a2_val
        sgm_b1 = 2 * 2 ** (1 / 6.0) * b1_val
        sgm_b2 = 2 * 2 ** (1 / 6.0) * b2_val
        sgm_c1 = 2 * 2 ** (1 / 6.0) * c1_val
        sgm_c2 = 2 * 2 ** (1 / 6.0) * c2_val

        matrix_tr = dat_matrix[ipot_num]
        xyz_aff = torch.matmul(matrix_tr, xyz_aff.T).T

        ######------
        # penarty for LS interaction
        ######------
        tmp_add_ene = torch.tensor(0.0)
        for iatom in range(len(target_atoms)):
            vec_2 = xyz_aff[target_atoms[iatom]][:3]
            epsilon_2 = MM_param_list[target_atoms[iatom]][0]
            sigma_2 = MM_param_list[target_atoms[iatom]][1]

            if vec_2[0] >= 0:
                sgm_a = sgm_a1
            else:
                sgm_a = sgm_a2
            if vec_2[1] >= 0:
                sgm_b = sgm_b1
            else:
                sgm_b = sgm_b2
            if vec_2[2] >= 0:
                sgm_c = sgm_c1
            else:
                sgm_c = sgm_c2

            vec_2_unit = vec_2 / torch.linalg.norm(vec_2)
            dist_tmp_val = 1 / (
                2
                * (
                    (vec_2_unit[0] / sgm_a) ** 2
                    + (vec_2_unit[1] / sgm_b) ** 2
                    + (vec_2_unit[2] / sgm_c) ** 2
                )
                ** 0.5
            )
            dist_norm_1 = torch.linalg.norm(vec_2) / (dist_tmp_val * 2)

            rt_dist_norm_1 = dist_norm_1**0.5
            rt_dist_norm_2 = (torch.linalg.norm(vec_2) / sigma_2) ** 0.5
            dist_norm = rt_dist_norm_1 * rt_dist_norm_2
            epsilon = (epsilon_1 * epsilon_2) ** 0.5

            ene = epsilon * ((1.0 / dist_norm) ** 12 - 2 * (1.0 / dist_norm) ** 6)
            tmp_add_ene = tmp_add_ene + ene
        add_ene = add_ene + tmp_add_ene

        ######------
        # penarty for LL interaction
        ######------
        tmp_add_ene = torch.tensor(0.0)
        for jpot_num in dat_pot_num:
            if jpot_num <= ipot_num:
                pass
            else:
                vec_2 = xyz_aff[dat_pot_num[jpot_num][2]][:3]

                if vec_2[0] >= 0:
                    sgm_a = sgm_a1
                else:
                    sgm_a = sgm_a2
                if vec_2[1] >= 0:
                    sgm_b = sgm_b1
                else:
                    sgm_b = sgm_b2
                if vec_2[2] >= 0:
                    sgm_c = sgm_c1
                else:
                    sgm_c = sgm_c2

                vec_2_unit = vec_2 / torch.linalg.norm(vec_2)
                dist_tmp_val = 1 / (
                    2
                    * (
                        (vec_2_unit[0] / sgm_a) ** 2
                        + (vec_2_unit[1] / sgm_b) ** 2
                        + (vec_2_unit[2] / sgm_c) ** 2
                    )
                    ** 0.5
                )
                dist_norm_1 = torch.linalg.norm(vec_2) / (dist_tmp_val * 2)
                rt_dist_norm_1 = dist_norm_1**0.5

                jlig = dat_pot_num[jpot_num][0]
                jpot = dat_pot_num[jpot_num][1]

                epsilon_2 = LJ_asym_ell_info[jlig][jpot]["eps"]
                a1_val_jpot = LJ_asym_ell_info[jlig][jpot]["a1_val"]
                a2_val_jpot = LJ_asym_ell_info[jlig][jpot]["a2_val"]
                b1_val_jpot = LJ_asym_ell_info[jlig][jpot]["b1_val"]
                b2_val_jpot = LJ_asym_ell_info[jlig][jpot]["b2_val"]
                c1_val_jpot = LJ_asym_ell_info[jlig][jpot]["c1_val"]
                c2_val_jpot = LJ_asym_ell_info[jlig][jpot]["c2_val"]

                sgm_a1_jpot = 2 * 2 ** (1 / 6.0) * a1_val_jpot
                sgm_a2_jpot = 2 * 2 ** (1 / 6.0) * a2_val_jpot
                sgm_b1_jpot = 2 * 2 ** (1 / 6.0) * b1_val_jpot
                sgm_b2_jpot = 2 * 2 ** (1 / 6.0) * b2_val_jpot
                sgm_c1_jpot = 2 * 2 ** (1 / 6.0) * c1_val_jpot
                sgm_c2_jpot = 2 * 2 ** (1 / 6.0) * c2_val_jpot

                matrix_tr_jpot = dat_matrix[jpot_num]
                xyz_aff = torch.matmul(torch.linalg.inv(matrix_tr), xyz_aff.T).T
                xyz_aff = torch.matmul(matrix_tr_jpot, xyz_aff.T).T

                vec_3 = xyz_aff[dat_pot_num[ipot_num][2]][:3]
                if vec_3[0] >= 0:
                    sgm_a_jpot = sgm_a1_jpot
                else:
                    sgm_a_jpot = sgm_a2_jpot
                if vec_3[1] >= 0:
                    sgm_b_jpot = sgm_b1_jpot
                else:
                    sgm_b_jpot = sgm_b2_jpot
                if vec_3[2] >= 0:
                    sgm_c_jpot = sgm_c1_jpot
                else:
                    sgm_c_jpot = sgm_c2_jpot

                vec_3_unit = vec_3 / torch.linalg.norm(vec_3)
                dist_tmp_val_2 = 1 / (
                    2
                    * (
                        (vec_3_unit[0] / sgm_a_jpot) ** 2
                        + (vec_3_unit[1] / sgm_b_jpot) ** 2
                        + (vec_3_unit[2] / sgm_c_jpot) ** 2
                    )
                    ** 0.5
                )
                dist_norm_2 = torch.linalg.norm(vec_3) / (dist_tmp_val_2 * 2)
                rt_dist_norm_2 = dist_norm_2**0.5

                dist_norm = rt_dist_norm_1 * rt_dist_norm_2
                epsilon = (epsilon_1 * epsilon_2) ** 0.5

                ene = epsilon * ((1.0 / dist_norm) ** 12 - 2 * (1.0 / dist_norm) ** 6)
                tmp_add_ene = tmp_add_ene + ene

                xyz_aff = torch.matmul(torch.linalg.inv(matrix_tr_jpot), xyz_aff.T).T
                xyz_aff = torch.matmul(matrix_tr, xyz_aff.T).T
        add_ene = add_ene + tmp_add_ene
        xyz_aff = torch.matmul(torch.linalg.inv(matrix_tr), xyz_aff.T).T

    if log_tag:
        path_view = path_phi_log.split(".phi_log")[0] + "_ovoid.xyz"
        with open(path_view, mode="w") as g:
            atom_list = atom_order
            xyz_aff = xyz_aff * unit_au2ang()
            xyz_norot = torch.matmul(torch.linalg.inv(pre_rot_matrix), xyz_aff.T).T
            g.write("{0}\n\n".format(len(xyz_aff)))
            for iatom in range(len(xyz_aff)):
                try:
                    g.write(
                        "{0:<5}  {1:12.9f}  {2:12.9f}  {3:12.9f}\n".format(
                            atom_list[iatom],
                            xyz_norot[iatom][0].item(),
                            xyz_norot[iatom][1].item(),
                            xyz_norot[iatom][2].item(),
                        )
                    )
                except:
                    g.write(
                        "{0:<5}  {1:12.9f}  {2:12.9f}  {3:12.9f}\n".format(
                            "X",
                            xyz_norot[iatom][0].item(),
                            xyz_norot[iatom][1].item(),
                            xyz_norot[iatom][2].item(),
                        )
                    )
    return add_ene


def phi_system_search(xyz, phi_list, MM_param_list, LJ_asym_ell_info, n_pot, n_step=6):
    min_score = 10**10
    phi_min = None

    phi_list = phi_list.tolist()
    unit_angle = 2 * np.pi / n_step
    for ipot in range(n_pot):
        for jstep in range(n_step):
            phi_list_tmp = copy.deepcopy(phi_list)
            phi_list_tmp[ipot] = phi_list_tmp[ipot] + unit_angle * jstep
            phi_list_tmp = torch.tensor(
                phi_list_tmp, requires_grad=True, dtype=torch.float64
            )

            phi_list_tmp, newton_tag = phi_newton_opt(
                xyz, phi_list_tmp, MM_param_list, LJ_asym_ell_info
            )
            if newton_tag:
                tmp_add_ene = calc_ene_phi_tensor(
                    xyz, phi_list_tmp, MM_param_list, LJ_asym_ell_info
                )
            else:
                phi_list_tmp, CG_tag = phi_CG_opt(
                    xyz, phi_list_tmp, MM_param_list, LJ_asym_ell_info, [-4, -4, -2]
                )
                phi_list_tmp = phi_list_tmp.tolist()
                phi_list_tmp = torch.tensor(
                    phi_list_tmp, requires_grad=True, dtype=torch.float64
                )

                phi_list_tmp, newton_tag = phi_newton_opt(
                    xyz, phi_list_tmp, MM_param_list, LJ_asym_ell_info
                )
                tmp_add_ene = calc_ene_phi_tensor(
                    xyz, phi_list_tmp, MM_param_list, LJ_asym_ell_info
                )

            if tmp_add_ene < min_score:
                min_score = tmp_add_ene
                phi_min = phi_list_tmp

        phi_list = phi_min.tolist()
    return min_score, phi_min


def phi_newton_opt(xyz, phi_list, MM_param_list, LJ_asym_ell_info):
    istop_tag = False

    icount = 0
    start_phi_list = copy.deepcopy(phi_list)
    start_add_ene = calc_ene_phi_tensor(xyz, phi_list, MM_param_list, LJ_asym_ell_info)

    while True:
        tmp_add_ene = calc_ene_phi_tensor(
            xyz, phi_list, MM_param_list, LJ_asym_ell_info
        )
        grad = torch.func.jacfwd(calc_ene_phi_tensor, argnums=1)(
            xyz, phi_list, MM_param_list, LJ_asym_ell_info
        )
        hessian = torch.func.hessian(calc_ene_phi_tensor, argnums=1)(
            xyz, phi_list, MM_param_list, LJ_asym_ell_info
        )
        if icount > 15:
            istop_tag = True
            break
        else:
            cont_tag = False
            for iphi in range(len(phi_list)):
                if torch.abs(grad[iphi]) < 10 ** (-10):
                    pass
                else:
                    cont_tag = True
            if cont_tag == False:
                break

            update_vec = -torch.matmul(torch.linalg.inv(hessian), grad)
            if (
                torch.max(update_vec).item() > np.pi / 2
                or torch.min(update_vec).item() < -1 * np.pi / 2
            ):
                istop_tag = True
                break
            else:
                phi_list = phi_list + update_vec
                icount = icount + 1

    if tmp_add_ene <= start_add_ene and istop_tag is False:
        newton_tag = True

    elif tmp_add_ene <= start_add_ene and istop_tag is True:
        newton_tag = False
    else:
        phi_list = start_phi_list
        newton_tag = False

    del tmp_add_ene
    return phi_list, newton_tag


def phi_CG_opt(
    xyz, phi_list, MM_param_list, LJ_asym_ell_info, threshold_list=[-8, -10, -7]
):
    x_vec_n = phi_list
    f_val_n = calc_ene_phi_tensor(xyz, phi_list, MM_param_list, LJ_asym_ell_info)
    f_grad_n = torch.func.jacfwd(calc_ene_phi_tensor, argnums=1)(
        xyz, phi_list, MM_param_list, LJ_asym_ell_info
    )
    maxitr = 50
    threshold = 10 ** threshold_list[0]
    tag_continue = True

    iitr = 0
    f_grad_m = None
    d_vec_m_tmp = None

    while tag_continue and iitr < maxitr:
        #####-----
        # calculate CG vec and do line search
        #####-----
        d_vec_n, d_vec_n_tmp = calc_CG(f_grad_n, f_grad_m, d_vec_m_tmp)
        f_grad_m = f_grad_n
        d_vec_m_tmp = d_vec_n_tmp

        x_vec_n, f_val_n, f_grad_n = linesearch_safe(
            xyz,
            x_vec_n,
            MM_param_list,
            LJ_asym_ell_info,
            d_vec_n,
            f_val_n,
            threshold_list[1],
            threshold_list[2],
        )
        f_val_n = calc_ene_phi_tensor(xyz, phi_list, MM_param_list, LJ_asym_ell_info)

        #####-----
        # judge if converge or not
        #####-----
        f2 = 0
        for i in f_grad_n:
            if torch.abs(i) > f2:
                f2 = torch.abs(i)
        if f2 > threshold:
            tag_continue = True
        else:
            tag_continue = False
        iitr += 1

    if tag_continue:
        CG_tag = False
    else:
        CG_tag = True

    del f_val_n
    return x_vec_n, CG_tag


def calc_CG(f_grad_n, f_grad_m, d_vec_m_tmp):
    if f_grad_m is None:
        d_vec_n_tmp = -1 * f_grad_n

    else:
        beta_n = torch.dot(f_grad_n, (f_grad_n - f_grad_m)) / torch.dot(
            f_grad_m, f_grad_m
        )
        d_vec_n_tmp = -1 * f_grad_n + beta_n * d_vec_m_tmp

    d_vec_n = d_vec_n_tmp / torch.linalg.norm(d_vec_n_tmp)
    return d_vec_n, d_vec_n_tmp


def linesearch_safe(
    xyz,
    x_vac_n,
    MM_param_list,
    LJ_asym_ell_info,
    d_vec,
    f_val_n,
    f_threshold,
    a_threshold,
):
    ss = torch.tensor(0.001, dtype=torch.float64)

    for jitr in range(100):
        #####-----
        # calculate step 1
        #####-----
        x_vac_n_tmp1 = x_vac_n + 1.0 * ss * d_vec
        f_val_n_tmp1 = calc_ene_phi_tensor(
            xyz, x_vac_n_tmp1, MM_param_list, LJ_asym_ell_info
        )

        #####-----
        # calculate step 2 and define new ss
        #####-----
        if f_val_n_tmp1 > f_val_n:
            x_vac_n_tmp2 = x_vac_n - 1.0 * ss * d_vec
            f_val_n_tmp2 = calc_ene_phi_tensor(
                xyz, x_vac_n_tmp2, MM_param_list, LJ_asym_ell_info
            )

            if f_val_n_tmp2 > f_val_n:
                f1 = -0.5 * (f_val_n_tmp1 - f_val_n_tmp2) * ss
                f2 = f_val_n_tmp1 + f_val_n_tmp2 - 2.0 * f_val_n
                if torch.abs(f2) > 1.0e-16:
                    ss = f1 / f2
                else:
                    if f_val_n_tmp1 < f_val_n_tmp2:
                        ss *= 0.25
                    else:
                        ss *= -0.25
            else:
                ss *= -0.75

        else:
            x_vac_n_tmp2 = x_vac_n + 2.0 * ss * d_vec
            f_val_n_tmp2 = calc_ene_phi_tensor(
                xyz, x_vac_n_tmp2, MM_param_list, LJ_asym_ell_info
            )

            if f_val_n_tmp2 < f_val_n_tmp1:
                ss *= 2.0
            else:
                f1 = -0.5 * (4.0 * f_val_n_tmp1 - f_val_n_tmp2 - 3.0 * f_val_n) * ss
                f2 = f_val_n_tmp2 + f_val_n - 2.0 * f_val_n_tmp1
                if torch.abs(f2) > 1.0e-16:
                    ss = f1 / f2
                else:
                    if f_val_n < f_val_n_tmp2:
                        ss *= 0.75
                    else:
                        ss *= 1.25
        #####-----
        # correct too large ss
        #####-----
        if torch.abs(ss) > 0.10:
            ss = 0.10 * ss / torch.abs(ss)

        #####-----
        # update parameters and calculate next step
        #####-----
        update_vec = ss * d_vec
        x_vac_n = x_vac_n + update_vec
        f_val_n = calc_ene_phi_tensor(xyz, x_vac_n, MM_param_list, LJ_asym_ell_info)
        f_grad_n = torch.func.jacfwd(calc_ene_phi_tensor, argnums=1)(
            xyz, x_vac_n, MM_param_list, LJ_asym_ell_info
        )

        #####-----
        # judge if converge
        #####-----
        a = torch.dot(f_grad_n / torch.linalg.norm(f_grad_n), d_vec)

        if torch.abs(ss) < 10 ** f_threshold or torch.abs(a) < 10**a_threshold:
            end_tag = True
        else:
            end_tag = False

        if end_tag:
            break

    return x_vac_n, f_val_n, f_grad_n


def microiteration_phi(xyz, MM_param_list, LJ_asym_ell_info, path_phi_log):
    search_tag = True
    update_tag = False

    if os.path.isfile(path_phi_log):
        phi_log = VL_class_files.PhiLog(path_phi_log)
        phi_list = torch.tensor(
            phi_log.phi_list, requires_grad=True, dtype=torch.float64
        )
        path_grrm_log = path_phi_log.split(".phi_log")[0] + ".log"
        if os.path.isfile(path_grrm_log):
            try:
                grrm_log = VL_class_files.LogFile(path_grrm_log)
                itr_num = grrm_log.itr_num
                n_pot = len(phi_list)

                if (itr_num + 1) % (n_pot * 2) != 0:
                    search_tag = False

            except:
                pass

        if search_tag:
            # global search
            n_pot = len(phi_list)
            min_score = calc_ene_phi_tensor(
                xyz, phi_list, MM_param_list, LJ_asym_ell_info
            )
            tmp_min_score, tmp_phi_min = phi_system_search(
                xyz, phi_list, MM_param_list, LJ_asym_ell_info, n_pot
            )

            if tmp_min_score < min_score:
                phi_min = tmp_phi_min
                update_tag = True
                phi_list = phi_min

    else:
        n_pot = 0
        for ilig in LJ_asym_ell_info:
            for ipot in LJ_asym_ell_info[ilig]:
                n_pot = n_pot + 1

        phi_list = np.array([0.0 for ipot in range(n_pot)])
        min_score, phi_list = phi_system_search(
            xyz, phi_list, MM_param_list, LJ_asym_ell_info, n_pot
        )
        min_score, phi_list = phi_system_search(
            xyz, phi_list, MM_param_list, LJ_asym_ell_info, n_pot
        )

    #####----------
    # Newton optimization of phi (might be redundant)
    #####----------
    phi_list = phi_list.tolist()
    phi_list = torch.tensor(phi_list, requires_grad=True, dtype=torch.float64)

    phi_list_tmp, newton_tag = phi_newton_opt(
        xyz, phi_list, MM_param_list, LJ_asym_ell_info
    )
    if newton_tag:
        phi_list = phi_list_tmp
        CG_tag = False
    else:
        phi_list, CG_tag = phi_CG_opt(xyz, phi_list, MM_param_list, LJ_asym_ell_info)

    #####----------
    # update phi_log
    #####----------
    with open(path_phi_log, mode="a") as f:
        for ipot in range(len(phi_list)):
            f.write("{} ".format(phi_list[ipot].item()))

        f.write("\n-----")
        if update_tag:
            f.write(" update by global search, ")
        if newton_tag:
            f.write(" Newton success")
        if CG_tag:
            f.write(" CG success")

        f.write("\n")
    return phi_list


def calc_LJ_asym_ell_pot_pgrad(
    xyz, phi_list, MM_param_list, LJ_asym_ell_info, tensor_list, tensor_tag
):
    for itag in range(len(tensor_tag)):
        for iligand in LJ_asym_ell_info:
            for ipot in LJ_asym_ell_info[iligand]:
                for iterm in LJ_asym_ell_info[iligand][ipot]:
                    if LJ_asym_ell_info[iligand][ipot][iterm] == tensor_tag[itag]:
                        LJ_asym_ell_info[iligand][ipot][iterm] = tensor_list[itag]
    add_ene = calc_ene_phi_tensor(xyz, phi_list, MM_param_list, LJ_asym_ell_info)
    return add_ene
