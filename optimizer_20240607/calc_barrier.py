import numpy as np
import copy
import math

def switching_barrier(param, param_range, penarty_val):
    barrier_val = 0
    barrier_grad_tmp = []


    for iparam in range(len(param)):
        iparam_val = param[iparam]
        iparam_min = param_range[iparam][0]
        iparam_max = param_range[iparam][1]
        iparam_range = iparam_max - iparam_min
        iparam_min_low = iparam_min - 0.1 * iparam_range
        iparam_min_high = iparam_min + 0.1 * iparam_range
        iparam_max_low = iparam_max - 0.1 * iparam_range
        iparam_max_high = iparam_max + 0.1 * iparam_range

        if iparam_val <= iparam_min:
            x_val = (iparam_val - iparam_min_low)/(iparam_range * 0.2)
            barrier_val += (-3.75 * x_val + 2.875) * penarty_val

            dbarrier_dx = (-3.75) * penarty_val
            dx_diparam = 1.0 / (iparam_range * 0.2)
            barrier_grad_tmp.append(dbarrier_dx * dx_diparam)

        elif iparam_val > iparam_min and iparam_val <= iparam_min_high:
            x_val = (iparam_val - iparam_min_low) / (iparam_range * 0.2)
            barrier_val += (2.0 - 20 * x_val ** 3 + 30.0 * x_val ** 4 - 12.0 * x_val ** 5) * penarty_val

            dbarrier_dx = (-60.0 * x_val ** 2 + 120.0 * x_val ** 3 - 60.0 * x_val ** 4) * penarty_val
            dx_diparam = 1.0 / (iparam_range * 0.2)
            barrier_grad_tmp.append(dbarrier_dx * dx_diparam)

        elif iparam_val > iparam_min_high and iparam_val <= iparam_max_low:
            barrier_val += 0
            barrier_grad_tmp.append(0.0)

        elif iparam_val > iparam_max_low and iparam_val <= iparam_max:
            x_val = (iparam_max_high - iparam_val) / (iparam_range * 0.2)
            barrier_val += (2.0 - 20 * x_val ** 3 + 30 * x_val ** 4 - 12 * x_val ** 5) * penarty_val

            dbarrier_dx = (-60.0 * x_val ** 2 + 120.0 * x_val ** 3 - 60.0 * x_val ** 4) * penarty_val
            dx_diparam = -1.0 / (iparam_range * 0.2)
            barrier_grad_tmp.append(dbarrier_dx * dx_diparam)

        elif iparam_val > iparam_max:
            x_val = (iparam_max_high - iparam_val) / (iparam_range * 0.2)
            barrier_val += (-3.75 * x_val + 2.875) * penarty_val

            dbarrier_dx = (-3.75) * penarty_val
            dx_diparam = -1.0 / (iparam_range * 0.2)
            barrier_grad_tmp.append(dbarrier_dx * dx_diparam)

    barrier_grad = np.array(barrier_grad_tmp)
    return barrier_val, barrier_grad
