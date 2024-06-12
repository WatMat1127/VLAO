
# How to Run
## Step 0. Modify the shebang line in the VL main.py
Modify the line as `#!/xxx/yyy/python`.

## Step 1. Prepare input files
The following files are required. You can find example files in the `Example` directory.  
- _GRRM_job_.com
- _GRRM_job_.param

### GRRM_job.com
An input file for the GRRM23 program. See [AFIR-web](https://afir.sci.hokudai.ac.jp) for the detailed format.
The `SubAddExPot=xxx/yyy/VL_main.py` option needs to be specified in the option part to use the VL method.

### GRRM_job.param
An input file for the `VL_main.py` which specifies settings for the VL calculation in the following format:

----------
num_virtual_mol   =(int)  
LIGLIGLIGLIGLIGLIGLIGLIGLIGLIG  
atom_num_center   =(int)  
atom_num_sub      =(int),(int),(int)  
off_target_atom   =(int), ..., (int)  
keeppot           =(int); _k_<sub>keep</sub>; _r_<sub>0</sub>  
keepanglepot      =(int); _k_<sub>keep angle</sub>; _θ_<sub>0</sub>   
ovoid_LJpot       =(int); _ε_; _a_<sub>1</sub>; _a_<sub>2</sub>; _b_<sub>1</sub>; _b_<sub>2</sub>; _c_<sub>1</sub>; _c_<sub>2</sub>; _d_  
•••   
LIGLIGLIGLIGLIGLIGLIGLIGLIGLIG  
•••

----------


The total number of virtual molecules (or sections) should be specified in the `num_virtual_mol` statement.  
This number must be consistent with the number of the ligand sections (blocks between two 'LIGLIGLIGLIGLIGLIGLIGLIGLIGLIG's).  

Each ligand section must include one `atom_num_center`, `atom_num_sub`, and `off_target_atom` statement in this order.  
- The `atom_num_center` statement specifies the phosphorus atom.  
- The `atom_num_sub` statement specifies three atoms adjacent to the phosphorous atom (_e.g._, three Cl* atoms for PCl<sub>3</sub>).  
- The `off_target_atom` statement specifies atoms to be excluded from the calculation of the ovoid-based LJ poitential.
  
Each ligand section can include one or more `keeppot` and `keepanglepot` statements.
- The `keeppot` statement specifies the parameters of the keep potential. The first integer defines the Cl* atom for which the keep potential is calculted; _k_<sub>keep</sub> and _r_<sub>0</sub> define the force constant and equiliblium distance of the keep potential.
- The `keepanglepot` statement specifies the parameters of the keep angle potential. The first integer defines the Cl* atom for which the keep angle potential is calculted; _k_<sub>keep angle</sub> and _θ_<sub>0</sub> define the force constant and equiliblium angle of the keep angle potential.

Each ligand section can include two or more `ovoid_LJpot` statements.
- The `ovoid_LJpot` statement specifies the parameters of the ovoid-based LJ potential. The first integer defines the Cl* atom for which the ovoid-based LJ potential is calculted; _ε_ is the parameter corresponding to the well depth; _a_<sub>1</sub>, _a_<sub>2</sub>, _b_<sub>1</sub>, _b_<sub>2</sub>, _c_<sub>1</sub>, _c_<sub>2</sub> and _d_ are the parameters which define shape and size of the ovoid.

## Step 2. Run GRRM 

The GRRM23 program calls the `VL_main.py` at each iteration of structural optimization. If the above settings are working well, you will find _GRRM_job_\_LinkJOB.rrm_old and _GRRM_job_\_LinkJOB.rrm_final in addition to usual output files of the GRRM23 program. When the ovoid-based LJ potential is used, _GRRM_job_.phi_log and _GRRM_job_\_ovoid.xyz will be also generated.

- _GRRM_job_\_LinkJOB.rrm_old and _GRRM_job_\_LinkJOB.rrm_final include the geometry and the corresponding electronic energy, gradient, and Hessian before and after adding penalty functions, respectively.
- _GRRM_job_.phi_log includes the optimized internal parameters (**_q_**<sup>*</sup><sub>VL</sub>).
- _GRRM_job_\_ovoid.xyz shows each apex of the ovoid by placing dummy atoms (X). In the original VLAO paper, the ovoid was visualized based on this file using Matplotlib.



