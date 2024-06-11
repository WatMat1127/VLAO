import os
import copy
import numpy as np
from write_read_files import txtfile2list
from write_read_files import list2textfile

class Struct:
    def __init__(self):
        pass

    def list2geom(self,list):
        geom_list = []
        for line in list:
            tmp_atom = line.split()[0]
            tmp_x = float(line.split()[1])
            tmp_y = float(line.split()[2])
            tmp_z = float(line.split()[3])
            tmp_vec = np.array([tmp_x, tmp_y, tmp_z])
            geom_list.append([tmp_atom, tmp_vec])

        self.geom = geom_list
        self.natom = len(geom_list)

    def add_geom(self,geom):
        self.geom = geom
        self.natom = len(geom)

    def geom2list(self):
        geom_list = []
        for iatom in range(self.natom):
            tmp_str = '{:<15}{: 14.8f}{: 14.8f}{: 14.8f}'.format(self.geom[iatom][0], self.geom[iatom][1][0], self.geom[iatom][1][1], self.geom[iatom][1][2])
            geom_list.append(tmp_str)
        self.geom_list = geom_list

    def geom2xyz(self, text='text'):
        xyz_list = []
        xyz_list.append(str(self.natom))
        xyz_list.append(text)
        for iatom in range(self.natom):
            tmp_str = ' {:<15}{: 14.8f}{: 14.8f}{: 14.8f}'.format(self.geom[iatom][0], self.geom[iatom][1][0], self.geom[iatom][1][1], self.geom[iatom][1][2])
            xyz_list.append(tmp_str)
        self.xyz_list = xyz_list

    def opt_pcl_dist(self, keeppot_info):
        for ipot in keeppot_info:
            p_num = ipot[0]
            cl_num = ipot[1]
            d_val = ipot[2]

            a_val = 0.278421317
            b_val = -0.222992424
            c_val = 1.336325835
            dist_tgt = a_val * d_val ** 2 + b_val * d_val + c_val

            p_vec = self.geom[p_num][1]
            cl_vec = self.geom[cl_num][1]
            pcl_vec_raw = cl_vec - p_vec
            pcl_dist_raw = np.linalg.norm(pcl_vec_raw)
            pcl_vec_new = pcl_vec_raw / pcl_dist_raw * dist_tgt
            cl_vec_new = pcl_vec_new + p_vec
            self.geom[cl_num][1] = cl_vec_new

class LogFile:
    def __init__(self, path):
        self.path = path
        self.fn_top = os.path.basename(path)[:-4]

    def read_grrm_opt_log(self):
        if os.path.isfile(self.path):
            l = txtfile2list(self.path)
            opt_idx = [i for i, line in enumerate(l) if '{}'.format('OPTOPTOPTOPTOPTOPTOPTOPTOPTOPTOPTOPTOPTOPT') in line]
            n_opt = len(opt_idx)
            if n_opt == 1:
                tag = os.path.isfile(self.path[:-4] + '_message_STOP.rrm')
                if tag:
                    print(self.path, 'ERROR: The JOB not finished.')
                    result_tag = 'message_STOP'
                    self.result = result_tag

                else:
                    print(self.path, 'ERROR: The JOB not finished, but massage_STOP not found.')
                    result_tag = 'unknown (no message_STOP)'
                    self.result = result_tag

                struct = Struct()
                itr_idx = [i for i, line in enumerate(l) if '# ITR.' in line]
                itr_idx_2 = [i for i, line in enumerate(l) if 'Threshold' in line]
                struct_list = l[itr_idx[-1] + 1: itr_idx_2[-1]]
                struct.list2geom(struct_list)
                self.last_struct = struct


            elif n_opt > 2:
                print('ERROR: More than 1 optimization jobs are included.')
                result_tag = 'more than 1 optimization jobs'
                self.result = result_tag

            else:
                struct_idx = [i for i, line in enumerate(l) if '{}'.format('Optimized') in line]
                e_idx = [i for i, line in enumerate(l) if '{}'.format('ENERGY    =') in line]
                g_idx = [i for i, line in enumerate(l) if '{}'.format('Free Energy   =') in line]

                result_tag = l[opt_idx[-1] - 1]

                if len(struct_idx) == 1 and len(e_idx) == 1:
                    # Normal termination
                    struct_list = l[struct_idx[0] + 1 : e_idx[0]]
                    e_ene = float(l[e_idx[0]].split()[2])

                    struct = Struct()
                    struct.list2geom(struct_list)
                    struct.e_ene = e_ene

                    if len(g_idx) == 2:
                        g_ene = float(l[g_idx[1]].split()[3])
                        struct.g_ene = g_ene
                    else:
                        struct.g_ene = None
                    self.struct = struct
                    self.result = result_tag

                else:
                    print(self.path, 'ERROR: The JOB not finished.')
                    struct = Struct()
                    itr_idx = [i for i, line in enumerate(l) if '# ITR.' in line]
                    itr_idx_2 = [i for i, line in enumerate(l) if 'Threshold' in line]
                    struct_list = l[itr_idx[-1] + 1: itr_idx_2[-1]]
                    struct.list2geom(struct_list)
                    self.last_struct = struct

                    self.result = result_tag
        else:
            result_tag = 'File not found'
            self.result = result_tag

class ComFile:
    def __init__(self, path):
        self.path = path
        self.fn_top = os.path.basename(path)[:-4]

    def read_grrm_com(self):
        l = txtfile2list(self.path)
        brank_idx = [i for i, line in enumerate(l) if '' == line]
        option_idx = [i for i, line in enumerate(l) if 'Options' in line]
        frozen_idx = [i for i, line in enumerate(l) if 'Frozen' in line]

        head = l[:brank_idx[0]]
        charge_spin = l[brank_idx[0] + 1]
        option = [i for i in l[option_idx[0]:] if i != '']

        if len(frozen_idx) == 0:
            free_struct_list = l[brank_idx[0] + 2: option_idx[0]]
            free_struct = Struct()
            free_struct.list2geom(free_struct_list)
            self.struct = free_struct

        else:
            free_struct_list = l[brank_idx[0] + 2: frozen_idx[0]]
            free_struct = Struct()
            free_struct.list2geom(free_struct_list)
            froz_struct_list = l[frozen_idx[0] + 1 : option_idx[0]]
            froz_struct = Struct()
            froz_struct.list2geom(froz_struct_list)
            self.struct = free_struct
            self.frozen_struct = froz_struct
        self.head = head
        self.charge_spin = charge_spin
        self.option = option

    def rename(self, new_path):
        self.path = new_path
        self.fn_top = os.path.basename(new_path)[:-4]

    def write_grrm_com(self):
        tmp_list = self.head + [''] + [self.charge_spin]
        if 'frozen_struct' not in dir(self):
            self.struct.geom2list()
            tmp_list += self.struct.geom_list
        else:
            if self.frozen_struct == None:
                self.struct.geom2list()
                tmp_list += self.struct.geom_list
            else:
                self.struct.geom2list()
                tmp_list += self.struct.geom_list
                tmp_list += ['Frozen Atoms']
                self.frozen_struct.geom2list()
                tmp_list += self.frozen_struct.geom_list
        tmp_list += self.option
        list2textfile(self.path, tmp_list)

    def copy_file_from(self, path_file_old):
        extension = os.path.splitext(path_file_old)
        path_file_new = self.path[:-4] + extension[1]
        tmp_list = txtfile2list(path_file_old)
        list2textfile(path_file_new, tmp_list)

class ParamFile:
    def __init__(self, path):
        self.path = path
        self.fn_top = os.path.basename(path)[:-6]

    def read_param(self):
        l = txtfile2list(self.path)
        self.all_list = l

    def assign_param(self, param_list, param_tag):
        param_tmp = self.all_list
        param_write = []
        for iline in range(len(param_tmp)):
            if '@@' in param_tmp[iline]:
                tmp_list = param_tmp[iline].split('@@')
                for iterm in range(len(tmp_list)):
                    if tmp_list[iterm] in param_tag:
                        idx = param_tag.index(tmp_list[iterm])
                        tmp_list[iterm] = '{}'.format(param_list[idx])
                tmp_str = ''.join(tmp_list)
                param_write.append(tmp_str)
            else:
                param_write.append(param_tmp[iline])
        self.all_list = param_write

    def rename(self, new_path):
        self.path = new_path
        self.fn_top = os.path.basename(new_path)[:-6]

    def write_param(self):
        list2textfile(self.path, self.all_list)

class LinkJOBFile:
    def __init__(self, path):
        self.path = path
        self.fn_top = os.path.basename(path)[:-4]

    def read_ene(self):
        l = txtfile2list(self.path)
        ene_idx = [i for i, line in enumerate(l) if 'ENERGY =' in line]
        ene = float(l[ene_idx[0]].split()[2])
        self.ene = ene
