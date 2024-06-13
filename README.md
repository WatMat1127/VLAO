# VLAO

This repository contains codes for the paper:   
Virtual Ligand-Assisted Optimization: A Rational Strategy for Ligand Engineering (DOI: [10.26434/chemrxiv-2024-6www6](https://chemrxiv.org/engage/chemrxiv/article-details/662856e521291e5d1d87b234)).  

It consists of two main components:
- VL : python programs to calculate the penarty functions for the electronic and steric aproximations
- optimizer : python programs to perform optimization of VL parameters by the conjugate gradient method
  
Each component has a readme file with further information. 
The codes have been modified from the original version used in the original paper, but it has been confirmed that essentially the same results are obtained. 
The codes have only been tested in the computer environment below, and may require minor modifications to run in different computer systems.

## Test environment
- GRRM23 program  
- Gaussian 16
- conda 23.7.4
- Python 3.11.5
- pytorch 2.1.0  
- numpy 1.24.3  

