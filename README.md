# cp2k_ML

* The script for carrying out the machine-learning analysis with the computational results obtained with CP2K
* Based on ASE
* CP2K version is 7.1.0, as this version is stable on MacOS
* One should specify the environment variable for CP2K (e.g. in ~/.bashrc) as
```
export ASE_CP2K_COMMAND="${HOME}/cp2k/cp2k-7.1.0/exe/Darwin-IntelMacintosh-gfortran/cp2k_shell.sopt"
```
* Do not specify the CP2K binary. It should be cp2k_shell.{sopt, popt, ...}
* Machine-learning is based on scikit-learn library. Now it does the simple regression analysis
* CO adsorption energy is now treated as the target value, and the atomic number of the adsorption site is descriptor