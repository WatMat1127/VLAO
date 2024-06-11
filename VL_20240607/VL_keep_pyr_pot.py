import torch
import VL_calc_tools

def calc_keep_pyr_pot(xyz, keeppyr_info):
    add_ene = 0

    for ilig in keeppyr_info:
        ilig_keeppyr_info = keeppyr_info[ilig]
        for ipot in ilig_keeppyr_info:
            tmp_keeppyr_info = ilig_keeppyr_info[ipot]
            vec_p = xyz[tmp_keeppyr_info['atom_number_P']]
            vec_tgt = xyz[tmp_keeppyr_info['atom_number_xi']]
            vec_sdt_list = []
            for iatom in tmp_keeppyr_info['atom_number_xlist']:
                vec_sdt_list.append(xyz[iatom])
            k_val = tmp_keeppyr_info['k_val']
            a_val = tmp_keeppyr_info['a_val']

            vec_p_tgt = vec_tgt - vec_p
            vec_p_axis = VL_calc_tools.calc_lonepair_axis(vec_p, vec_sdt_list)
            cos = torch.dot(vec_p_tgt, vec_p_axis) / torch.linalg.norm(vec_p_tgt) / torch.linalg.norm(vec_p_axis)
            theta = torch.acos(cos)
            d_theta = theta - a_val
            pot = 0.5 * k_val * d_theta ** 2
            add_ene = add_ene + pot

    return add_ene

def calc_keep_pyr_pot_pgrad(xyz, keeppyr_info, tensor_list, tensor_tag):
    for itag in range(len(tensor_tag)):
        for iligand in keeppyr_info:
            for ipot in keeppyr_info[iligand]:
                for iterm in keeppyr_info[iligand][ipot]:
                    if keeppyr_info[iligand][ipot][iterm] == tensor_tag[itag]:
                        keeppyr_info[iligand][ipot][iterm] = tensor_list[itag]

    add_ene = calc_keep_pyr_pot(xyz, keeppyr_info)
    return add_ene
