# cp2k_ML: cp2k script for machine-learning

## Introduction
* the script for carrying out the machine-learning analysis with the computational results obtained with CP2K
* based on ASE (https://wiki.fysik.dtu.dk/ase/index.html)
* the calculation consists of two python files; `make_dataset.py` and `regression.py`

## make_dataset.py
* do DFT calculation for alloy metal surface
* position of the alloy elements are set randomly, and data are generated according to argument `n_samples`
* CO adsorption energy is now treated as the target value
* now the center of the s, p, d-band is recorded so can be used as the machine-learning descriptor
* CP2K version 7.1.0 is confrmed to work on MacOS, and 6.1 is confirmed on LINUX (Redhat)
* one should specify the cp2k binary path in this script
* do not specify the CP2K binary. It should be cp2k_shell.{sopt, popt, ...}

## regression.py
* DFT-calculated data is stored in some JSON file, and this script reads this file
* machine-learning is based on scikit-learn library
* currently LASSO is used for the regression model, but any method can be used

