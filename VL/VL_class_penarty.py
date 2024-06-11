import torch
import VL_keep_pot
import VL_keep_pyr_pot
import VL_LJ_asym_ell_pot


class Penarty:
    def __init__(self, natom_active):
        natom = natom_active

        self.natom_active = natom_active
        self.add_ene = 0
        self.add_grad = []
        for iatom in range(natom):
            self.add_grad.append(3 * [0.0])

        self.add_hess = []
        for iatom in range(natom):
            hessian_list_iatom = []
            for jatom in range(iatom + 1):
                if iatom == jatom:
                    hessian_list_iatom.append([[0.0], [0.0, 0.0], [0.0, 0.0, 0.0]])
                else:
                    hessian_list_iatom.append([3 * [0.0], 3 * [0.0], 3 * [0.0]])
            self.add_hess.append(hessian_list_iatom)


    def reshape_torch_grad_hess(self, torch_grad, torch_hess, natom_active=None):
        if natom_active == None:
            natom_active = self.natom_active
        for iatom in range(natom_active):
            self.add_grad[iatom][0] += torch_grad[iatom][0].item()
            self.add_grad[iatom][1] += torch_grad[iatom][1].item()
            self.add_grad[iatom][2] += torch_grad[iatom][2].item()

        for iatom in range(natom_active):
            for idim in range(3):
                for jatom in range(natom_active):
                    for jdim in range(3):
                        if iatom == jatom:
                            if idim >= jdim:
                                self.add_hess[iatom][jatom][idim][jdim] += torch_hess[iatom][idim][jatom][jdim].item()
                        elif iatom > jatom:
                            self.add_hess[iatom][jatom][idim][jdim] += torch_hess[iatom][idim][jatom][jdim].item()
                        else:
                            pass
        return

    def add_keep_pot(self, xyz, keep_info):
        if len(keep_info) != 0:
            ene = VL_keep_pot.calc_keeppot(xyz, keep_info)
            grad = torch.func.jacfwd(VL_keep_pot.calc_keeppot)(xyz, keep_info)
            hessian = torch.func.hessian(VL_keep_pot.calc_keeppot)(xyz, keep_info)
            self.add_ene = ene.item()
            self.reshape_torch_grad_hess(grad, hessian)

    def add_keep_pyr_pot(self, xyz, keep_pyr_info):
        if len(keep_pyr_info) != 0:
            ene = VL_keep_pyr_pot.calc_keep_pyr_pot(xyz, keep_pyr_info)
            grad = torch.func.jacfwd(VL_keep_pyr_pot.calc_keep_pyr_pot)(xyz, keep_pyr_info)
            hessian = torch.func.hessian(VL_keep_pyr_pot.calc_keep_pyr_pot)(xyz, keep_pyr_info)
            self.add_ene = ene.item()
            self.reshape_torch_grad_hess(grad, hessian)

    def add_LJ_asym_ell_pot(self, xyz, LJ_asym_ell_info, MM_param_list, path_phi_log, atom_order):
        if len(LJ_asym_ell_info) != 0:
            phi_list = VL_LJ_asym_ell_pot.microiteration_phi(xyz, MM_param_list, LJ_asym_ell_info, path_phi_log)
            xyz_phi_tensor = VL_LJ_asym_ell_pot.conbine_xyz_phi(xyz, phi_list)
            ene = VL_LJ_asym_ell_pot.calc_ene(xyz_phi_tensor, MM_param_list, LJ_asym_ell_info, log_tag=True, atom_order=atom_order, path_phi_log=path_phi_log)
            self.add_ene = ene.item()
            del ene
            grad = torch.func.jacfwd(VL_LJ_asym_ell_pot.calc_ene)(xyz_phi_tensor, MM_param_list, LJ_asym_ell_info)
            hessian = torch.func.hessian(VL_LJ_asym_ell_pot.calc_ene)(xyz_phi_tensor, MM_param_list, LJ_asym_ell_info)
            natom = len(MM_param_list)

            grad = torch.reshape(grad[:3 * natom], (natom, 3))
            hessian_tmp_1 = hessian[:3 * natom].T
            hessian_rr = hessian_tmp_1[:3 * natom].T
            hessian_rn = hessian_tmp_1[3 * natom:].T

            hessian_tmp_2 = hessian[3 * natom:].T
            hessian_nn = hessian_tmp_2[3 * natom:].T
            hessian_nr = hessian_tmp_2[:3 * natom].T

            hessian_tmp_3 = hessian_rr - torch.matmul(hessian_rn, torch.matmul(torch.linalg.inv(hessian_nn), hessian_nr))
            eff_hessian = torch.reshape(hessian_tmp_3, (natom, 3, natom, 3))
            self.reshape_torch_grad_hess(grad, eff_hessian)

    def combine_penarties(self, penarty_list):
        for ipenarty in penarty_list:
            self.add_ene += ipenarty.add_ene
            for iatom in range(self.natom_active):
                self.add_grad[iatom][0] += ipenarty.add_grad[iatom][0]
                self.add_grad[iatom][1] += ipenarty.add_grad[iatom][1]
                self.add_grad[iatom][2] += ipenarty.add_grad[iatom][2]

            for iatom in range(self.natom_active):
                for jatom in range(self.natom_active):
                    for idim in range(3):
                        for jdim in range(3):
                            if iatom == jatom:
                                if idim >= jdim:
                                    self.add_hess[iatom][jatom][idim][jdim] += ipenarty.add_hess[iatom][jatom][idim][jdim]
                            elif iatom > jatom:
                                self.add_hess[iatom][jatom][idim][jdim] += ipenarty.add_hess[iatom][jatom][idim][jdim]
                            else:
                                pass



