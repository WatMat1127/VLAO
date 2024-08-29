import numpy as np
import class_target
import z_function
import calc_barrier


class OptInfo:
    def __init__(self, input_path, run_tag):
        self.info_path = input_path + ".txt"
        self.run_tag = run_tag
        self.nstep = 0
        self.nstep_iitr = 0
        self.param_log = {}
        self.f_val_log = {}
        self.penalty_log = {}
        self.f_val_best = 99999999999999
        self.f_val_best_tag = -999
        self.f_grad_log = {}
        self.d_vec_log = {}
        self.d_vec_tmp_log = {}
        self.initss_log = {}
        self.ss_log = {}
        self.angle_log = {}

    def read_input_file(self):
        info_path = self.info_path

        target_list = []
        with open(info_path, mode="r") as f:
            lines = f.readlines()
            for line in lines:
                tmp_str = line.strip().split("#")[0]
                if tmp_str == "":
                    pass
                else:
                    if "com" in tmp_str:
                        ftop_target = tmp_str.split()[1].split(",")[0]
                        target = class_target.Target(ftop_target)
                        target.add_ene = float(tmp_str.split()[2])
                        target_list.append(target)

                    elif "ene_read" in tmp_str:
                        ene_tag = tmp_str.split()[1]

                    elif "SubAddExPot" in tmp_str:
                        expot_tag = tmp_str

                    elif "init_param" in tmp_str:
                        x0 = tmp_str.split(":")[1]
                        x0 = [float(i) for i in x0.split(",")]

                    elif "param_tag" in tmp_str:
                        param_tag_tmp = tmp_str.split(":")[1]
                        param_tag = [i.strip() for i in param_tag_tmp.split(",")]

                    elif "param_range" in tmp_str:
                        param_range_tmp = tmp_str.split(":")[1]
                        tmp_list = param_range_tmp.split(",")
                        param_range = []
                        for irange in tmp_list:
                            p_min = float(irange.split("_")[0])
                            p_max = float(irange.split("_")[1])
                            param_range.append([p_min, p_max])

                    elif "max_itr" in tmp_str:
                        max_itr = int(tmp_str.split(":")[1].strip())

                    elif "grad_threshold" in tmp_str:
                        grad_threshold = float(tmp_str.split(":")[1].strip())

                    elif "param_threshold" in tmp_str:
                        param_threshold = float(tmp_str.split(":")[1].strip())

                    elif "f_val_threshold" in tmp_str:
                        f_val_threshold = float(tmp_str.split(":")[1].strip())

                    elif "penalty_std" in tmp_str:
                        penalty_std = float(tmp_str.split(":")[1].strip())

        self.target_list = target_list
        self.ene_tag = "E"
        self.expot_tag = expot_tag
        self.param_log[0] = np.array(x0)
        self.param_tag = param_tag
        self.param_range = param_range
        self.max_itr = max_itr
        self.grad_threshold = grad_threshold
        self.param_threshold = param_threshold
        self.f_val_threshold = f_val_threshold
        self.penalty_std = penalty_std

    def function(self):
        #####-----
        # preparation
        #####-----
        target_list = self.target_list

        nstep = self.nstep
        param = self.param_log[self.nstep]
        penarty_std = self.penalty_std
        dat_info = {}
        dat_info["nstep"] = nstep
        dat_info["param"] = param
        dat_info["param_tag"] = self.param_tag
        dat_info["expot_tag"] = self.expot_tag

        for itgt in target_list:
            itgt.run_tag = self.run_tag

        #####-----
        # calculate E
        #####-----
        for itgt in target_list:
            itgt.make_input_files(dat_info)

        # search previous calculations that have same parameters
        param_idx = []
        for istep in self.param_log:
            bool_vec = param == self.param_log[istep]
            if bool_vec.all() == True:
                param_idx.append(istep)

        # copy result if the calculations have been already done
        if len(param_idx) >= 2:
            istep = param_idx[0]
            for itgt in target_list:
                itgt.copy_result(nstep, istep)
        # run QM calculation
        else:
            for itgt in target_list:
                itgt.run_calculation(nstep)

        # read energy and struct from log file
        qm_ene_list = []
        for itgt in target_list:
            qm_ene = itgt.read_QM_struct_ene(nstep, self.ene_tag)
            qm_ene += itgt.add_ene
            qm_ene_list.append(qm_ene)

        #####-----
        # calculate dE/dp
        #####-----
        grad_list = []
        for itgt in target_list:
            grad = itgt.calc_grad(dat_info)
            grad_list.append(grad)

        #####-----
        # calculate f_val and f_grad
        #####-----
        f_val, f_grad = z_function.calc_f_val_grad(qm_ene_list, grad_list)

        #####-----
        # calculate barrier function
        #####-----

        barrier_val, barrier_grad = calc_barrier.switching_barrier(
            param, self.param_range, penarty_std
        )
        f_val += barrier_val
        f_grad += barrier_grad

        #####-----
        # keep structures if f_val is the best
        #####-----
        if f_val <= self.f_val_best:
            self.f_val_best = f_val
            self.nstep_best = nstep
            for i in range(len(target_list)):
                target_list[i].lowest_struct = target_list[i].latest_struct
                target_list[i].nstep_best = nstep
        return f_val, f_grad, barrier_val

    def calc_CG(self):
        f_grad_n = self.f_grad_log[self.nstep]
        f_grad_m = self.f_grad_log[self.nstep_iitr]
        d_vec_m_tmp = self.d_vec_tmp_log[self.nstep_iitr]

        if f_grad_m is None:
            d_vec_n_tmp = -1 * f_grad_n

        else:
            beta_n = np.dot(f_grad_n, (f_grad_n - f_grad_m)) / np.dot(
                f_grad_m, f_grad_m
            )
            d_vec_n_tmp = -1 * f_grad_n + beta_n * d_vec_m_tmp
        d_vec_n = d_vec_n_tmp / np.linalg.norm(d_vec_n_tmp)

        self.nstep_iitr = self.nstep
        self.d_vec_log[self.nstep_iitr] = d_vec_n
        self.d_vec_tmp_log[self.nstep_iitr] = d_vec_n_tmp
        return d_vec_n

    def calc_initSS(self):
        x_vec_n = self.param_log[self.nstep_iitr]
        d_vec_n = self.d_vec_log[self.nstep_iitr]
        f_val_std = self.f_val_log[self.nstep]
        x_val_std = self.param_log[self.nstep]
        max_param = -1
        max_idx = 99
        for iparam in range(len(d_vec_n)):
            if np.abs(d_vec_n[iparam]) > max_param:
                max_param = np.abs(d_vec_n[iparam])
                max_idx = iparam
        std_size = np.abs(x_vec_n[max_idx]) * 0.020
        initss = std_size / max_param

        cont_tag = True
        while cont_tag:
            self.nstep += 1
            self.param_log[self.nstep] = x_val_std + initss * d_vec_n
            f_val_n, f_grad_n, penarty_val = self.function()
            self.f_val_log[self.nstep] = f_val_n
            self.f_grad_log[self.nstep] = f_grad_n
            self.penalty_log[self.nstep] = penarty_val
            if f_val_n <= f_val_std:
                cont_tag = False
            else:
                initss *= 0.5
        self.initss_log[self.nstep_iitr] = initss

    def linesearch_safe(self):
        ss = self.initss_log[self.nstep_iitr]
        for jitr in range(1000):
            #####-----
            # read current value
            #####-----
            if jitr == 0:
                x_vac_n = self.param_log[self.nstep_iitr]
                d_vec = self.d_vec_log[self.nstep_iitr]
                f_val_n = self.f_val_log[self.nstep_iitr]
            else:
                x_vac_n = self.param_log[self.nstep]
                d_vec = self.d_vec_log[self.nstep_iitr]
                f_val_n = self.f_val_log[self.nstep]

            #####-----
            # calculate step 1
            #####-----
            self.nstep += 1
            x_vac_n_tmp1 = x_vac_n + 1.0 * ss * d_vec
            self.param_log[self.nstep] = x_vac_n_tmp1
            f_val_n_tmp1, f_grad_n_tmp1, penarty_val_n_tmp1 = self.function()
            self.f_val_log[self.nstep] = f_val_n_tmp1
            self.f_grad_log[self.nstep] = f_grad_n_tmp1
            self.penalty_log[self.nstep] = penarty_val_n_tmp1

            #####-----
            # calculate step 2 and define new ss
            #####-----
            if f_val_n_tmp1 > f_val_n:
                self.nstep += 1
                x_vac_n_tmp2 = x_vac_n - 1.0 * ss * d_vec
                self.param_log[self.nstep] = x_vac_n_tmp2
                f_val_n_tmp2, f_grad_n_tmp2, penarty_val_n_tmp2 = self.function()
                self.f_val_log[self.nstep] = f_val_n_tmp2
                self.f_grad_log[self.nstep] = f_grad_n_tmp2
                self.penalty_log[self.nstep] = penarty_val_n_tmp2

                if f_val_n_tmp2 > f_val_n:
                    f1 = -0.5 * (f_val_n_tmp1 - f_val_n_tmp2) * ss
                    f2 = f_val_n_tmp1 + f_val_n_tmp2 - 2.0 * f_val_n
                    if np.abs(f2) > 1.0e-16:
                        ss = f1 / f2
                    else:
                        if f_val_n_tmp1 < f_val_n_tmp2:
                            ss *= 0.25
                        else:
                            ss *= -0.25
                else:
                    ss *= -0.75

            else:
                self.nstep += 1
                x_vac_n_tmp2 = x_vac_n + 2.0 * ss * d_vec
                self.param_log[self.nstep] = x_vac_n_tmp2
                f_val_n_tmp2, f_grad_n_tmp2, penarty_val_n_tmp2 = self.function()
                self.f_val_log[self.nstep] = f_val_n_tmp2
                self.f_grad_log[self.nstep] = f_grad_n_tmp2
                self.penalty_log[self.nstep] = penarty_val_n_tmp2

                if f_val_n_tmp2 < f_val_n_tmp1:
                    ss *= 2.0
                else:
                    f1 = -0.5 * (4.0 * f_val_n_tmp1 - f_val_n_tmp2 - 3.0 * f_val_n) * ss
                    f2 = f_val_n_tmp2 + f_val_n - 2.0 * f_val_n_tmp1
                    if np.abs(f2) > 1.0e-16:
                        ss = f1 / f2
                    else:
                        if f_val_n < f_val_n_tmp2:
                            ss *= 0.75
                        else:
                            ss *= 1.25
            #####-----
            # correct too large ss
            #####-----
            tmp_vec = ss * d_vec
            max_idx = 99
            max_param = -10
            for iparam in range(len(tmp_vec)):
                if np.abs(tmp_vec[iparam]) > max_param:
                    max_idx = iparam
                    max_param = np.abs(tmp_vec[iparam])

            std_param = np.abs(x_vac_n[max_idx])
            if max_param > std_param * 0.1:
                ss = std_param * 0.1 / np.abs(d_vec[max_idx]) * np.sign(ss)

            #####-----
            # update parameters and calculate next step
            #####-----
            self.nstep += 1
            self.ss_log[self.nstep] = ss
            update_vec = ss * d_vec
            self.param_log[self.nstep] = x_vac_n + update_vec
            f_val_p, f_grad_p, penarty_val_p = self.function()
            self.f_val_log[self.nstep] = f_val_p
            self.f_grad_log[self.nstep] = f_grad_p
            self.penalty_log[self.nstep] = penarty_val_p

            #####-----
            # calculate angle between d_vec and grad
            #####-----
            a = np.dot(f_grad_p / np.linalg.norm(f_grad_p), d_vec)
            self.angle_log[self.nstep] = a

            #####-----
            # judge if converge
            #####-----
            f2 = 0
            for iparam in update_vec:
                if np.abs(iparam) > f2:
                    f2 = np.abs(iparam)

            if np.abs(a) < 0.10 or np.abs(ss) < 0.001:
                end_tag = True
            else:
                end_tag = False

            #####-----
            # make report file
            #####-----
            np.set_printoptions(linewidth=10000)

            with open("analysis.txt", mode="w") as g:
                g.write("f_val_log\n")
                g.write(
                    "0 (step0) {0} {1}\n".format(
                        self.f_val_log[0], self.f_val_log[0] - self.penalty_log[0]
                    )
                )
                for i, istep in enumerate(self.ss_log):
                    g.write(
                        "{0} (step{1}) {2} {3}\n".format(
                            i + 1,
                            istep,
                            self.f_val_log[istep],
                            self.f_val_log[istep] - self.penalty_log[istep],
                        )
                    )

                g.write("f_grad_log\n")
                g.write("0 (step0) {0}\n".format(self.f_grad_log[0]))
                for i, istep in enumerate(self.ss_log):
                    g.write(
                        "{0} (step{1}) {2}\n".format(
                            i + 1, istep, self.f_grad_log[istep]
                        )
                    )

                g.write("param_log\n")
                g.write("0 (step0) {0}\n".format(self.param_log[0]))
                for i, istep in enumerate(self.ss_log):
                    g.write(
                        "{0} (step{1}) {2}\n".format(
                            i + 1, istep, self.param_log[istep]
                        )
                    )

                g.write("d_vec_log\n")
                for i, istep in enumerate(self.d_vec_log):
                    g.write(
                        "{0} (step{1}) {2}\n".format(i, istep, self.d_vec_log[istep])
                    )

                g.write("ss_log\n")
                for i, istep in enumerate(self.ss_log):
                    g.write(
                        "{0} (step{1}) {2}\n".format(i + 1, istep, self.ss_log[istep])
                    )

                g.write("angle_log\n")
                for i, istep in enumerate(self.ss_log):
                    g.write(
                        "{0} (step{1}) {2}\n".format(
                            i + 1, istep, self.angle_log[istep]
                        )
                    )

            #####-----
            # judge if f_val is lower than threshold
            #####-----
            if f_val_p <= self.f_val_threshold:
                print("The F value is low enough, terminate calculation")
                exit()

            elif end_tag:
                break
