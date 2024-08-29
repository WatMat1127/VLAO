#!/usr/bin/env python3
import sys
import numpy as np
import class_optinfo

#####-----
# read initial information
#####-----
args = sys.argv
if len(args) != 3:
    print("[VL_optimizer] execute as follows:")
    print("[VL_optimizer] python main.py [xxx.info] [run_type]")
    print("[VL_optimizer] run_type options : dummy, run, stop")
    exit()

elif len(args) == 3:
    input_path = args[1]
    run_tag = args[2]
    optinfo = class_optinfo.OptInfo(input_path, run_tag)
    optinfo.read_input_file()

    #####----------
    # initial evaluation of F and gradient
    #####----------
    x_vec_n = optinfo.param_log[0]
    f_val_n, f_grad_n, penalty_val_n = optinfo.function()
    optinfo.param_log[optinfo.nstep_iitr] = x_vec_n
    optinfo.f_val_log[0] = f_val_n
    optinfo.f_grad_log[0] = f_grad_n
    optinfo.penalty_log[0] = penalty_val_n

    maxitr = optinfo.max_itr
    grad_threshold = optinfo.grad_threshold
    param_threshold = optinfo.param_threshold
    tag_continue = True

    iitr = 0
    optinfo.nstep_iitr = -1
    optinfo.f_grad_log[-1] = None
    optinfo.d_vec_tmp_log[-1] = None

    #####----------
    # CG optimization
    #####----------
    while tag_continue and iitr < maxitr:
        #####-----
        # calculate CG vec and do line search
        #####-----
        d_vec_n = optinfo.calc_CG()
        optinfo.calc_initSS()
        optinfo.linesearch_safe()

        #####-----
        # judge if converge or not
        #####-----
        f2 = 0
        for i in optinfo.f_grad_log[optinfo.nstep]:
            if np.abs(i) > f2:
                f2 = np.abs(i)

        d_param_max = 0
        step_n = optinfo.nstep
        param_n = optinfo.param_log[step_n]
        step_m = list(optinfo.d_vec_log.keys())[-1]
        param_m = optinfo.param_log[step_m]

        d_param = param_n - param_m
        for i in d_param:
            if np.abs(i) > d_param_max:
                d_param_max = np.abs(i)

        if f2 > grad_threshold or d_param_max > param_threshold:
            tag_continue = True
        else:
            tag_continue = False

        iitr += 1

    if tag_continue:
        print("Maximum number of iteration was exceeded.")
    else:
        print("CONVERGED!!!")
