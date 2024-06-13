# How to Run
## Step 0. Set GRRMsub command
Set up your computer enviroment so that a GRRM job can be executed with the following command: `GRRMs filename`.


[追加追加]scrディレクトリに持っていくこと　　

## Step 1. Prepare input files
The following files are required. You can find elample files in the `Example` directrory.  
- _input_.txt  
- _GRRM_job_i_.com (_i_ = 1–_n_)  
- _GRRM_job_i_.param (_i_ = 1–_n_)  


### _input_.txt
An input file for `main.py` which specifies GRRM jobs to be executed, the absolute path for `VL_main.py`, parameters to be optimized, and other detailed settings for optimization in the following format:

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
The `com_n` statement specifies GRRMjob to be executed (format: com\__n_: _filename_ (no extension), _energy correction_).  
The `SubAddExPot` statement specifies the absolute path for `VL_main.py`.    
The `param_tag` statement specifies the parameters to be optimized. The label should be consistent with those written in _GRRM_job_i_.param.      
The `init_param` statement specifies the initial values for each paramter.    
The `param_range` statement specifies the lower and higher bounds for each paramter (see Figure S6 in the VLAO paper).  
The `penalty_std` statement specifies the scale of the barrier function (see Figure S6 in the VLAO paper).  
The `max_itr` statement specifies the maximum number of iteration for the conjugate gradient method.   
The `grad_threshold` and `param_threshold` statements specify termination criteria. The calculation will be terminated when the maximum absolute values among the gradient components and the displacement components are smaller than those specified here.  
The `f_val_threshold` statement specifies another termination criterion. The calculation will be terminated when the objective function is smaller than those specified here.  
   

### _GRRM_job_i_.com
An input file for the GRRM23 program. See [AFIR-web](https://afir.sci.hokudai.ac.jp) for the detailed format. The `@@SubAddExPot@@` needs to be specified instead of `SubAddExPot=_xxx_/_yyy_/VL_main.py` in the option part.

### _GRRM_job_i_.param
An input file for the VL program. See the `README` in the `VL` directory for detail. The parameter to be optimized must be specified as "@@_p<sub>j</sub>_<sup>label</sup>@@" instead of corresponding initial values. The label must be consistent with those written in _input_.txt.  

## Step 2. Modify z_function.py
An arbitrary objective function can be implemented as the `calc_f_val_grad` function. The `f_val` specified here will be minimized. 
You can find elample files in the `Example` directrory. The format for the `calc_f_val_grad` function is as follows:

    def calc_f_val_grad(qm_ene_list, grad_list):
        •••
        (calculation of f_val and f_grad)
        •••
        return f_val, f_grad

The `f_val` and `f_grad` are an objective function and a list of its gradients with respect to the VL parameters to be optimized.  
The `qm_ene_list` argument is a list containing the electronic energy of _GRRM_job_i_ (corrected by the corresponding _energy correction_ specified in  _input_.txt) as the *i*th component.  
The `grad_list` argument is a list of lists, where the *j*th component of the *i*th list is corresponds to a derivative value of the *i*th electronic energy with respect to the *j*th parameter.

## Step 3. Run calculation
The calculation can be executed with the following command: `python main.py input run`.  
As the optimization usually takes some time, it is recommended to run the calculation background with the following command:  
`nohup python main.py input run > result.log &`.  

Immediately, _GRRM_job_i_\_step0.com and _GRRM_job_i_\_step0.param (_i_ = 1–_n_) will be generated and corresponding GRRM jobs will be executed. After several steps, a file named analysis.txt will appear, summarizing the results of the parameter optimization.

