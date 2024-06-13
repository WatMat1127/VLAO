import numpy as np
import os
import subprocess
import copy
import time
import class_general
import write_read_files

class Target:
    def __init__(self, ftop_target):
        self.fn_target = ftop_target
        self.dir_DAT = ftop_target + '_DAT'
        self.info_log = {}
        self.qm_ene = {}
        self.penarty = {}
        self.grad = {}

    def make_input_files(self, dat_info):
        nstep = dat_info['nstep']
        param = dat_info['param']
        param_tag = dat_info['param_tag']
        expot_tag = dat_info['expot_tag']

        ftop_old = self.fn_target
        ftop_new = self.fn_target + '_step{}'.format(nstep)
        if os.path.isfile(ftop_new + '.com'):
            pass
        else:
            #####-----
            # make param file
            #####-----
            param_file = class_general.ParamFile(ftop_old + '.param')
            param_file.read_param()
            param_file.rename(ftop_new + '.param')
            param_file.assign_param(param, param_tag)
            param_file.write_param()

            keeppot_info = []
            for iline in param_file.all_list:
                if 'atom_number_P' in iline or 'atom_num_center' in iline:
                    p_num = int(iline.split('=')[1]) - 1
                elif 'keeppot' in iline:
                    cl_num = int(iline.split('=')[1].split(';')[0]) - 1
                    d_val = float(iline.split('=')[1].split(';')[2])
                    keeppot_info.append([p_num, cl_num, d_val])

            #####-----
            # make com file
            #####-----
            com_file = class_general.ComFile(ftop_old + '.com')
            com_file.read_grrm_com()
            com_file.rename(ftop_new + '.com')

            if nstep >= 1:
                ftop_best = self.fn_target + '_step{}'.format(self.nstep_best)
                log_file_best = class_general.LogFile(self.fn_target + '_step{}'.format(self.nstep_best) + '.log')
                log_file_best.read_grrm_opt_log()

                com_file.struct = log_file_best.struct
                com_file.struct.opt_pcl_dist(keeppot_info)
                com_file.option.append('MO GUESS = {}_MO.rrm'.format(ftop_best))
                subprocess.run('cp {0}.phi_log {1}.phi_log'.format(ftop_best, ftop_new), shell=True)

            else:
                com_file.struct.opt_pcl_dist(keeppot_info)

            for ioption in range(len(com_file.option)):
                if '@@SubAddExPot@@' == com_file.option[ioption]:
                    com_file.option[ioption] = expot_tag
                else:
                    pass
            com_file.write_grrm_com()


    def copy_result(self, nstep, istep):
        fn_nstep = self.fn_target + '_step{}'.format(nstep)
        fn_istep = self.fn_target + '_step{}'.format(istep)
        subprocess.run('cp {0}.com {0}_save.com'.format(fn_nstep), shell=True)
        subprocess.run('cp {0}.param {0}_save.param'.format(fn_nstep), shell=True)

        subprocess.run('cp {0}.com {1}.com'.format(fn_istep, fn_nstep), shell=True)
        subprocess.run('cp {0}.param {1}.param'.format(fn_istep, fn_nstep), shell=True)
        subprocess.run('cp {0}.log {1}.log'.format(fn_istep, fn_nstep), shell=True)
        subprocess.run('cp {0}.phi_log {1}.phi_log'.format(fn_istep, fn_nstep), shell=True)
        subprocess.run('cp {0}_MO.rrm {1}_MO.rrm'.format(fn_istep, fn_nstep), shell=True)
        subprocess.run('cp {0}_LinkJOB.rrm {1}_LinkJOB.rrm'.format(fn_istep, fn_nstep), shell=True)
        subprocess.run('cp {0}_LinkJOB.rrm_old {1}_LinkJOB.rrm_old'.format(fn_istep, fn_nstep), shell=True)
        subprocess.run('cp {0}_message_END.rrm {1}_message_END.rrm'.format(fn_istep, fn_nstep), shell=True)

        with open('analysis_copy.txt', mode='a') as g:
            g.write('step {0} is copied as step {1}: {2}\n'.format(istep, nstep, self.fn_target))

    def run_calculation(self, nstep):
        if os.path.isfile(self.fn_target + '_step{}.log'.format(nstep)):
            pass
        else:
            if self.run_tag == 'stop':
                #####-----
                # stop running
                #####-----
                print('{} not run'.format(self.fn_target + '_step{}.com'.format(nstep)))
                exit()
            elif self.run_tag == 'run':
                #####-----
                # real run
                #####-----
                ftop_new = self.fn_target + '_step{}'.format(nstep)
                subprocess.run('GRRMsub {0}'.format(ftop_new), shell=True)
            else:
                print('undefined run_tag: {}'.format(self.run_tag))
                print('exit')
                exit()

    def read_QM_struct_ene(self, nstep, ene_tag):
        path_log = self.fn_target + '_step{}.log'.format(nstep)
        path_message_end = self.fn_target + '_step{}_message_END.rrm'.format(nstep)
        path_message_stop = self.fn_target + '_step{}_message_STOP.rrm'.format(nstep)
        path_message_error = self.fn_target + '_step{}_message_ERROR.rrm'.format(nstep)
        path_message_linkerror = self.fn_target + '_step{}_message_LinkERROR.rrm'.format(nstep)

        wait_tag = True
        while wait_tag:
            if os.path.isfile(path_message_end):
                with open(path_log, mode='r') as f:
                    lines = f.readlines()
                    list = [line.strip() for line in lines]
                    struct_idx = [i for i, line in enumerate(list) if 'Optimized' in line]
                    ene_idx = [i for i, line in enumerate(list) if 'ENERGY    =' in line]
                    if len(struct_idx) == 1 and len(ene_idx) == 1:
                        struct = list[struct_idx[0] + 1:ene_idx[0]]
                        self.latest_struct = struct
                        wait_tag = False
                    else:
                        print('Some error happened in QM calculation.', flush=True)
                        print(path_log)
                        exit()
                        
            elif os.path.isfile(path_message_stop) or os.path.isfile(path_message_error) or os.path.isfile(path_message_linkerror):
                print('Some error happened in QM calculation.', flush=True)
                print(path_log)
                exit()
                 
            else:
                time.sleep(60)
                wait_tag = True

        log_file = class_general.LogFile(path_log)
        log_file.read_grrm_opt_log()
        if ene_tag == 'E':
            ene = log_file.struct.e_ene
        elif ene_tag == 'G':
            ene = log_file.struct.g_ene

        self.qm_ene[nstep] = ene
        return ene

    def calc_grad(self, dat_info):
        nstep = dat_info['nstep']
        param_list = dat_info['param']
        param_tag = dat_info['param_tag']
        expot_tag = dat_info['expot_tag']

        ftop = self.fn_target + '_step{}'.format(nstep)
        path_LinkJOB = ftop + '_LinkJOB.rrm'
        path_param_tag = ftop + '.param_tag'
        path_param_grad = ftop + '.param_grad'

        subprocess.run('cp {0} {1}'.format(self.fn_target + '.param', path_param_tag), shell=True)
        path_sterep = expot_tag.split('=')[1]
        if os.path.isfile(path_param_grad):
            pass
        else:
            subprocess.run('python {0} {1} param_grad=True'.format(path_sterep, path_LinkJOB), shell=True)
        grad = []
        l_dat = write_read_files.txtfile2list(path_param_grad)
        param_grad_tmp = {}
        for iline in l_dat:
            param_tag_tmp = iline.split()[0].split('@@')[1]
            grad_elem = float(iline.split()[1])
            param_grad_tmp[param_tag_tmp] = grad_elem

        for iparam in range(len(param_list)):
            param_tag_tmp = param_tag[iparam]
            grad_elem = param_grad_tmp[param_tag_tmp]
            grad.append(grad_elem)
        self.grad[nstep] = np.array(grad)
        return np.array(grad)






