# How to run
## Step 0. Set GRRMs command

## Step 1. Prepare input files
The following files are required. You can find elample files in the _Example_ directrory.  
- _input_.txt  
- _GRRM_job_i_.com (i = 1–_n_)  
- _GRRM_job_i_.param (i = 1–_n_)  


### _input_.txt
An input file for main.py which specifies GRRM jobs to be executed, the absolute path for VL_main.py, parameters to be optimized, and other detailed settings for optimization in the following format:

  ----------
  \# GRRMjobs to be executed  
  com_1: _GRRM_job_1_, (float)   
  com_2: _GRRM_job_2_, (float)   
  •••
  
  \# the absolute path for VL_main.py  
  SubAddExPot=_xxx_/_yyy_/VL_main.py  
  
  \# parameter setting  
  param_tag  : _p_<sub>1</sub><sup>label</sup>, _p_<sub>2</sub><sup>label</sup>, _p_<sub>3</sub><sup>label</sup>, ..., _p<sub>N</sub>_<sup>label</sup>   
  init_param : _p_<sub>1</sub><sup>init</sup>, _p_<sub>2</sub><sup>init</sup>, _p_<sub>3</sub><sup>init</sup>, ..., _p<sub>N</sub>_<sup>init</sup>  
  param_range: _p_<sub>1</sub><sup>low</sup>\__p_<sub>1</sub><sup>high</sup>, _p_<sub>2</sub><sup>low</sup>\__p_<sub>2</sub><sup>high</sup>, _p_<sub>3</sub><sup>low</sup>\__p_<sub>3</sub><sup>high</sup>, ..., _p<sub>N</sub>_<sup>low</sup>\__p<sub>N</sub>_<sup>high</sup>  
  penalty_std: _P_<sub>0</sub>  
  
  \# detailed setting  
  max_itr         : (int)   
  grad_threshold  : (float)   
  param_threshold : (float)   
  f_val_threshold : (float)   

  ----------
The "com\__n_" specifies GRRMjob to be executed (format: com\__n_: _filename_ (no extension), _energy correction_ (in haetree)).  
The "SubAddExPot" specifies the absolute path for VL_main.py.    
The "param_tag" specifies the parameters to be optimized. The label should be consistent with those written in _GRRM_job_i_.param.      
The "init_param" specifies the initial values for each paramter.    
The "param_range" specifies the lower and higher bounds for each paramter (see Figure S6 in the VLAO paper).  
The "penalty_std" specifies the scale of the barrier function (see Figure S6 in the VLAO paper).  
The "max_itr" specifies the maximum number of iteration for the conjugate gradient method.   
The "grad_threshold" and 'param_threshold' specify termination criteria. The calculation will be terminated when the maximum absolute values among the gradient components and the displacement components are smaller than those specified here.  
The "f_val_threshold" specifies another termination criterion. The calculation will be terminated when the objective function is smaller than those specified here.  
   

### GRRM_job_i.com
An input file for the GRRM23 program. The string "@@SubAddExPot@@" needs to be specified in the option part.

### GRRM_job_i.param
An input file for the VL program. See the _README.md_ file in the _VL_ directory for detail. The parameter to be optimized must be specified as "@@_p<sub>j</sub>_<sup>label</sup>@@" instead of corresponding initial values. The label must be consistent with those written in _input_.txt.  

