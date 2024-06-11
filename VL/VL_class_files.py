import numpy as np
import copy
from tools_unit_constant import unit_kcal2hartree
from tools_unit_constant import unit_ang2au
from tools_unit_constant import unit_deg2rad



class LinkJOB:
    def __init__(self, path):
        path_LinkJOB_write = f"{path}_rw"
        with open(path, mode='r') as f:
            fdat_linkjob = f.readlines()
            fdat_linkjob_list = [line.strip() for line in fdat_linkjob]
            coordinate_i = [i for i, line in enumerate(fdat_linkjob_list) if 'CURRENT COORDINATE' in line]
            energy_i = [i for i, line in enumerate(fdat_linkjob_list) if 'ENERGY' in line]
            gradient_i = [i for i, line in enumerate(fdat_linkjob_list) if 'GRADIENT' in line]
            hessian_i = [i for i, line in enumerate(fdat_linkjob_list) if 'HESSIAN' in line]
            hessian_end_i = [i for i, line in enumerate(fdat_linkjob_list) if 'DIPOLE DERIVATIVES' in line]
            natom = energy_i[0] - coordinate_i[0] - 1

            # load coordinate
            coordinate_list = []
            for iatom in range(natom):
                element_iatom = fdat_linkjob_list[coordinate_i[0] + 1 + iatom].split()[0]
                x_iatom = float(fdat_linkjob_list[coordinate_i[0] + 1 + iatom].split()[1]) * unit_ang2au()
                y_iatom = float(fdat_linkjob_list[coordinate_i[0] + 1 + iatom].split()[2]) * unit_ang2au()
                z_iatom = float(fdat_linkjob_list[coordinate_i[0] + 1 + iatom].split()[3]) * unit_ang2au()
                coordinate_list.append([element_iatom, np.array([x_iatom, y_iatom, z_iatom])])

            # load energy, gradient and hessian
            tmp_ene = float(fdat_linkjob_list[energy_i[0]].split()[2])
            grad_list = []
            hessian_list = []
            for iatom in range(natom):
                grad_list.append(
                    [float(fdat_linkjob_list[gradient_i[0] + 3 * iatom + 1]),
                     float(fdat_linkjob_list[gradient_i[0] + 3 * iatom + 2]),
                     float(fdat_linkjob_list[gradient_i[0] + 3 * iatom + 3])])

            tmp_d1 = (3 * natom) // 5
            tmp_d2 = (3 * natom) % 5
            tmp_list = []
            for i in range(3 * natom):
                tmp_list.append(fdat_linkjob_list[hessian_i[0] + 1 + i].split())
            for i in range(tmp_d1 - 1):
                for j in range(3 * natom - 5 * (i + 1)):
                    tmp_list[5 * (i + 1) + j] += fdat_linkjob_list[
                        int(hessian_i[0] + 1 + 3 * natom * (i + 1) - 5 * i * (i + 1) / 2 + j)].split()
            for j in range(tmp_d2):
                tmp_list[5 * (tmp_d1) + j] += fdat_linkjob_list[
                    int(hessian_i[0] + 1 + 3 * natom * tmp_d1 - 5 * tmp_d1 * (tmp_d1 - 1) / 2 + j)].split()

            for iatom in range(natom):
                hessian_list_iatom = []
                for jatom in range(iatom + 1):
                    if iatom == jatom:
                        xx = float(tmp_list[3 * iatom][-1])
                        yx = float(tmp_list[3 * iatom + 1][-2])
                        yy = float(tmp_list[3 * iatom + 1][-1])
                        zx = float(tmp_list[3 * iatom + 2][-3])
                        zy = float(tmp_list[3 * iatom + 2][-2])
                        zz = float(tmp_list[3 * iatom + 2][-1])

                        hessian_list_iatom.append([[xx], [yx, yy], [zx, zy, zz]])
                    else:
                        xx = float(tmp_list[3 * iatom][3 * jatom])
                        xy = float(tmp_list[3 * iatom][3 * jatom + 1])
                        xz = float(tmp_list[3 * iatom][3 * jatom + 2])
                        yx = float(tmp_list[3 * iatom + 1][3 * jatom])
                        yy = float(tmp_list[3 * iatom + 1][3 * jatom + 1])
                        yz = float(tmp_list[3 * iatom + 1][3 * jatom + 2])
                        zx = float(tmp_list[3 * iatom + 2][3 * jatom])
                        zy = float(tmp_list[3 * iatom + 2][3 * jatom + 1])
                        zz = float(tmp_list[3 * iatom + 2][3 * jatom + 2])
                        hessian_list_iatom.append([[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]])
                hessian_list.append(hessian_list_iatom)
            self.path_LinkJOB_write = path_LinkJOB_write
            self.all_list = fdat_linkjob
            self.all = fdat_linkjob_list
            self.coordinate_index = coordinate_i[0]
            self.energy_index = energy_i[0]
            self.grad_index = gradient_i[0]
            self.hessian_index = hessian_i[0]
            self.hessian_end_index = hessian_end_i[0]
            self.natom_active = natom
            self.xyz = coordinate_list
            self.ene = tmp_ene
            self.grad = grad_list
            self.hess = hessian_list

    def add_penarty(self, Penarty):
        add_ene = Penarty.add_ene
        add_grad = Penarty.add_grad
        add_hess = Penarty.add_hess
        self.ene += add_ene
        for iatom in range(self.natom_active):
            self.grad[iatom][0] += add_grad[iatom][0]
            self.grad[iatom][1] += add_grad[iatom][1]
            self.grad[iatom][2] += add_grad[iatom][2]

        for iatom in range(self.natom_active):
            for jatom in range(self.natom_active):
                for idim in range(3):
                    for jdim in range(3):
                        if iatom == jatom:
                            if idim >= jdim:
                                self.hess[iatom][jatom][idim][jdim] += add_hess[iatom][jatom][idim][jdim]
                        elif iatom > jatom:
                            self.hess[iatom][jatom][idim][jdim] += add_hess[iatom][jatom][idim][jdim]
                        else:
                            pass

    def write_LinkJOB(self, path_LinkJOB_write):
        natom = self.natom_active
        tmp_list_2 = []
        for iatom in range(natom):
            for ixyz in range(3):
                tmp_d3 = []
                for jatom in range(iatom + 1):
                    tmp_d3 += self.hess[iatom][jatom][ixyz]
                tmp_list_2.append(tmp_d3)

        hessian_write = []
        for i in range(3 * natom // 5):
            hessian_write.append(tmp_list_2[5 * i][5 * i:])
            hessian_write.append(tmp_list_2[5 * i + 1][5 * i:])
            hessian_write.append(tmp_list_2[5 * i + 2][5 * i:])
            hessian_write.append(tmp_list_2[5 * i + 3][5 * i:])
            hessian_write.append(tmp_list_2[5 * i + 4][5 * i:])
            for j in range(3 * natom - 5 * (i + 1)):
                hessian_write.append(tmp_list_2[5 * (i + 1) + j][5 * i: 5 * i + 5])
        for i in range(3 * natom % 5):
            hessian_write.append(tmp_list_2[5 * (3 * natom // 5) + i][5 * (3 * natom // 5):])

        ## ----------------------------------------
        ##   output
        ## ----------------------------------------
        fdat_linkjob_list = self.all_list
        fdat_linkjob_list_write = copy.deepcopy(fdat_linkjob_list)
        fdat_linkjob_list_write[self.energy_index] = f"ENERGY =   {self.ene: 15.12f}    0.000000000000    0.000000000000"
        for iatom in range(natom):
            fdat_linkjob_list_write[self.grad_index + 3 * iatom + 1] = f"{self.grad[iatom][0]: 11.9f}"
            fdat_linkjob_list_write[self.grad_index + 3 * iatom + 2] = f"{self.grad[iatom][1]: 11.9f}"
            fdat_linkjob_list_write[self.grad_index + 3 * iatom + 3] = f"{self.grad[iatom][2]: 11.9f}"
        for iline in range(self.hessian_end_index - self.hessian_index - 1):
            for iterm in range(len(hessian_write[iline])):
                hessian_write[iline][iterm] = f"{hessian_write[iline][iterm]: 11.9f}"
            fdat_linkjob_list_write[self.hessian_index + 1 + iline] = '   ' + '\t   '.join(hessian_write[iline])

        with open(path_LinkJOB_write, mode='w') as linkjob_write:
            for iline in range(len(fdat_linkjob_list_write)):
                if fdat_linkjob_list_write[iline] == fdat_linkjob_list[iline]:
                    linkjob_write.write(self.all_list[iline])
                else:
                    linkjob_write.write(str(fdat_linkjob_list_write[iline]) + '\n')

class ComFile:
    def __init__(self, path):
        with open(path, mode='r') as f:
            fdat_com = f.readlines()
            fdat_com_list = [line.strip() for line in fdat_com]
            start_index = [i for i, line in enumerate(fdat_com_list) if '' == line]
            frozen_index = [i for i, line in enumerate(fdat_com_list) if 'frozen' in line.lower()]
            end_index = [i for i, line in enumerate(fdat_com_list) if 'options' in line.lower()]
            if len(frozen_index) == 0:
                natom_all = end_index[0] - 1 - (start_index[0] + 1)
                self.natom_all = natom_all
                self.tag_frozen = False
            elif len(frozen_index) == 1:
                frosen_struct_list = fdat_com_list[frozen_index[0] + 1:end_index[0]]
                coordinate_list = []
                for iatom in range(len(frosen_struct_list)):
                    element_iatom = frosen_struct_list[iatom].split()[0]
                    x_iatom = float(frosen_struct_list[iatom].split()[1]) * unit_ang2au()
                    y_iatom = float(frosen_struct_list[iatom].split()[2]) * unit_ang2au()
                    z_iatom = float(frosen_struct_list[iatom].split()[3]) * unit_ang2au()
                    coordinate_list.append([element_iatom, np.array([x_iatom, y_iatom, z_iatom])])

                natom_all = end_index[0] - 1 - (start_index[0] + 1) - 1
                self.natom_all = natom_all
                self.tag_frozen = True
                self.frozen_xyz = coordinate_list
                self.natom_frozen = len(coordinate_list)
            else:
                print('[ERROR] len(frozen_index) should be 0 or 1')
                exit()

class LogFile:
    def __init__(self, path):
        with open(path, mode='r') as g:
            fdat_log = g.readlines()
            fdat_log_list = [line.strip() for line in fdat_log]
            itr_index = [i for i, line in enumerate(fdat_log_list) if '# ITR.' in line]
            itr_num = int(fdat_log_list[itr_index[-1]].split()[2])
        self.itr_num = itr_num

class ParamFile:
    def __init__(self, path, natom_all):
        with open(path, mode='r') as f:
            fdat_param = f.readlines()
            fdat_param_list = [line.strip() for line in fdat_param]
            nligand = int(fdat_param_list[0].split("=")[1])
            ligand_param_index = [i for i, line in enumerate(fdat_param_list) if 'LIGLIGLIGLIG' in line]

            ligand_atoms = []
            for iligand in range(nligand):
                atom_number_P = int(fdat_param_list[ligand_param_index[iligand] + 1].split('=')[1]) - 1
                atom_number_xlist = [int(i) - 1 for i in fdat_param_list[ligand_param_index[iligand] + 2].split('=')[1].split(',')]
                atom_number_offtgt = [int(i) - 1 for i in fdat_param_list[ligand_param_index[iligand] + 3].split('=')[1].split(',')]
                ligand_atoms.append([atom_number_P, atom_number_xlist, atom_number_offtgt])

            keep_info = {}
            keeppyr_info = {}
            LJ_asym_ell_info = {}

            for iligand in range(nligand):
                atom_number_P = ligand_atoms[iligand][0]
                atom_number_xlist = ligand_atoms[iligand][1]
                atom_number_offtgt = ligand_atoms[iligand][2]

                tmp_info = fdat_param_list[ligand_param_index[iligand] + 1:ligand_param_index[iligand + 1]]
                keep_idx = [i for i, line in enumerate(tmp_info) if 'keeppot' in line.lower()]
                keeppyr_idx = [i for i, line in enumerate(tmp_info) if 'keepangle' in line.lower() or 'keeppyr' in line.lower()]
                LJ_asym_ell_idx = [i for i, line in enumerate(tmp_info) if 'ovoid_ljpot' in line.lower() or 'ljpot_asym_ell' in line.lower()]

                nkeep = len(keep_idx)
                nkeeppyr = len(keeppyr_idx)
                nLJ_asym_ell = len(LJ_asym_ell_idx)

                # read parameters for keeppot in i-th ligand section
                ilig_keep_info = {}
                for ipot in range(nkeep):
                    ikeep_info = {}
                    keep_tmp_list = tmp_info[keep_idx[ipot]].split('=')[1].split(";")
                    atom_number_xi = int(keep_tmp_list[0]) - 1
                    k_val = self.get_param(keep_tmp_list[1], 1.0)
                    d_val = self.get_param(keep_tmp_list[2], unit_ang2au())
                    ikeep_info['atom_number_P'] = atom_number_P
                    ikeep_info['atom_number_xi'] = atom_number_xi
                    ikeep_info['k_val'] = k_val
                    ikeep_info['d_val'] = d_val

                    ilig_keep_info[ipot] = ikeep_info

                if len(ilig_keep_info) != 0:
                    keep_info[iligand] = ilig_keep_info

                # read parameters for keeppyrpot in i-th ligand section
                ilig_keeppyr_info = {}
                for ipot in range(nkeeppyr):
                    ikeeppyr_info = {}
                    keeppyr_tmp_list = tmp_info[keeppyr_idx[ipot]].split('=')[1].split(";")
                    atom_number_xi = int(keeppyr_tmp_list[0]) - 1

                    k_val = self.get_param(keeppyr_tmp_list[1], 1.0)
                    a_val = self.get_param(keeppyr_tmp_list[2], unit_deg2rad())

                    ikeeppyr_info['atom_number_P'] = atom_number_P
                    ikeeppyr_info['atom_number_xi'] = atom_number_xi
                    ikeeppyr_info['atom_number_xlist'] = atom_number_xlist
                    ikeeppyr_info['k_val'] = k_val
                    ikeeppyr_info['a_val'] = a_val

                    ilig_keeppyr_info[ipot] = ikeeppyr_info

                if len(ilig_keeppyr_info) != 0:
                    keeppyr_info[iligand] = ilig_keeppyr_info

                # read parameters for LJ_asym_ell pot in i-th ligand section
                ilig_LJ_asym_ell_info = {}
                for ipot in range(nLJ_asym_ell):
                    iLJ_asym_ell_info = {}
                    LJ_asym_ell_tmp_list = tmp_info[LJ_asym_ell_idx[ipot]].split('=')[1].split(";")
                    atom_number_xi = int(LJ_asym_ell_tmp_list[0]) - 1

                    epsilon = self.get_param(LJ_asym_ell_tmp_list[1], unit_kcal2hartree())
                    a1_val = self.get_param(LJ_asym_ell_tmp_list[2], unit_ang2au())
                    a2_val = self.get_param(LJ_asym_ell_tmp_list[3], unit_ang2au())
                    b1_val = self.get_param(LJ_asym_ell_tmp_list[4], unit_ang2au())
                    b2_val = self.get_param(LJ_asym_ell_tmp_list[5], unit_ang2au())
                    c1_val = self.get_param(LJ_asym_ell_tmp_list[6], unit_ang2au())
                    c2_val = self.get_param(LJ_asym_ell_tmp_list[7], unit_ang2au())
                    dist = self.get_param(LJ_asym_ell_tmp_list[8], unit_ang2au())

                    target_atoms = []
                    for iatom in range(natom_all):
                        if iatom in atom_number_offtgt or iatom == atom_number_P or iatom in atom_number_xlist:
                            pass
                        else:
                            target_atoms.append(iatom)

                    iLJ_asym_ell_info['atom_number_P'] = atom_number_P
                    iLJ_asym_ell_info['atom_number_xi'] = atom_number_xi
                    iLJ_asym_ell_info['target_atoms'] = target_atoms
                    iLJ_asym_ell_info['eps'] = epsilon
                    iLJ_asym_ell_info['a1_val'] = a1_val
                    iLJ_asym_ell_info['a2_val'] = a2_val
                    iLJ_asym_ell_info['b1_val'] = b1_val
                    iLJ_asym_ell_info['b2_val'] = b2_val
                    iLJ_asym_ell_info['c1_val'] = c1_val
                    iLJ_asym_ell_info['c2_val'] = c2_val
                    iLJ_asym_ell_info['dist'] = dist

                    ilig_LJ_asym_ell_info[ipot] = iLJ_asym_ell_info
                if len(ilig_LJ_asym_ell_info) != 0:
                    LJ_asym_ell_info[iligand] = ilig_LJ_asym_ell_info

            self.path = path
            self.nligand = nligand
            self.keep_info = keep_info
            self.keeppyr_info = keeppyr_info
            self.LJ_asym_ell_info = LJ_asym_ell_info
            self.ligand_atoms = ligand_atoms

    def get_param(self, param_str, unit_constatnt):
        if '@@' not in param_str:
            return float(param_str) * unit_constatnt
        else:
            return param_str.strip()

class MMParam:
    def __init__(self, path):
        MMParam = {}
        with open(path, mode='r') as f:
            fdat = f.readlines()
            fdat_list = [line.strip() for line in fdat]

            for irow in range(len(fdat_list)):
                tmp_list = fdat_list[irow].split()
                element = tmp_list[0]
                epsilon = float(tmp_list[1]) * unit_kcal2hartree()
                sigma = float(tmp_list[2]) * unit_ang2au()
                MMParam[element] = [epsilon, sigma]
        self.dat_all_MM_param = MMParam

    def make_list(self, atom_order):
        MM_param_list = []
        for iatom in atom_order:
            params = self.dat_all_MM_param[iatom]
            MM_param_list.append(params)
        return MM_param_list

class PhiLog:
    def __init__(self, path):
        with open(path, mode='r') as f:
            fdat_phi_log = f.readlines()
            fdat_phi_log_list = [line.strip() for line in fdat_phi_log]
            itr_index = [i for i, line in enumerate(fdat_phi_log_list) if '-----' in line]
            tmp_list = fdat_phi_log_list[itr_index[-1] - 1].split()

            phi_list = []
            for iphi in tmp_list:
                phi_list.append(float(iphi))
        self.phi_list = phi_list