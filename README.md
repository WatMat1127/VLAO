# VLAO

This repository contains codes for the paper:   
> Virtual Ligand-Assisted Optimization: A Rational Strategy for Ligand Engineering (DOI: [10.1021/acscatal.4c06003](https://pubs.acs.org/doi/10.1021/acscatal.4c06003)).  

Please cite this version as:

> Matsuoka, W.; Oki, T.; Yamada, R.; Yokoyama, T.; Suda, S.; Saunders, C. M.; Skjelstad, B. B.; Harabuchi, Y.; Fey, N.; Iwata, S.; Maeda, S. _ACS Catal._ **2024**, _14_, 16297. 

It consists of two main components:
- VL : python programs to calculate the penarty functions for the electronic and steric aproximations
- optimizer : python programs to perform optimization of VL parameters by the conjugate gradient method
  
Each component has a readme file with further information.  
The code has been slightly modified from the version used in the original paper, but it has been confirmed that essentially the same results are obtained.   
The code has only been tested in the computer environment below, and may require minor modifications to run in different computer systems.  

## Test environment
- GRRM23 program  
- Gaussian 16
- conda 23.7.4
- Python 3.11.5
- pytorch 2.1.0  
- numpy 1.24.3  

