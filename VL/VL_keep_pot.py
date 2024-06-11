import torch

def calc_keeppot(xyz, keep_info):
    add_ene = 0

    for ilig in keep_info:
        ilig_keep_info = keep_info[ilig]
        for ipot in ilig_keep_info:
            tmp_keep_info = ilig_keep_info[ipot]
            vec_1 = xyz[tmp_keep_info['atom_number_P']]
            vec_2 = xyz[tmp_keep_info['atom_number_xi']]
            k_val = tmp_keep_info['k_val']
            d_val = tmp_keep_info['d_val']

            vec_12 = vec_2 - vec_1
            dist_12 = torch.linalg.norm(vec_12)
            ene = 0.5 * k_val * (dist_12 - d_val) ** 2
            add_ene = add_ene + ene
    return add_ene


def calc_keep_pot_pgrad(xyz, keep_info, tensor_list, tensor_tag):
    for itag in range(len(tensor_tag)):
        for iligand in keep_info:
            for ipot in keep_info[iligand]:
                for iterm in keep_info[iligand][ipot]:
                    if keep_info[iligand][ipot][iterm] == tensor_tag[itag]:
                        keep_info[iligand][ipot][iterm] = tensor_list[itag]

    add_ene = calc_keeppot(xyz, keep_info)
    return add_ene
