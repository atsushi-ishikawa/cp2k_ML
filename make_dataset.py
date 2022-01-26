import numpy as np
import random
import time
import os
import uuid
from ase import Atoms
from ase.build import fcc111, add_adsorbate
from ase.visualize import view
from ase.calculators.emt import EMT
from ase.calculators.cp2k import CP2K
from ase.optimize.bfgs import BFGS
from ase.optimize.fire import FIRE
from ase.constraints import FixAtoms
import pandas as pd
import argparse

random.seed(time.time())

# surface
vacuum = 10.0
onelayer = 9
nlayer = 4

def constraint(surf, indices=None):
	c = FixAtoms(indices=indices)
	surf.set_constraint(c)

def make_base_surface(element="Au"):
	surf = fcc111(element, size=[3, 3, nlayer], vacuum=vacuum)

	indices = list(range(0, onelayer*nlayer // 2))
	constraint(surf, indices=indices)

	surf.translate([0, 0, -vacuum+0.1])
	surf.pbc = False
	return surf

def shuffle(surf, elements=None):
	if elements is None:
		#elements = ["Al", "Ni", "Cu", "Pd", "Ag", "Pt"]
		elements = ["Cu", "Ag", "Pt"]

	surf_copy = surf.copy()
	num_atoms = len(surf_copy.get_atomic_numbers())
	max_replace = int(0.5*num_atoms)
	num_replace = random.choice(range(1, max_replace))

	for iatom in range(num_replace):
		surf_copy[iatom].symbol = random.choice(elements)

	atomic_numbers = surf_copy.get_atomic_numbers()
	atomic_numbers = list(atomic_numbers)
	np.random.shuffle(atomic_numbers)
	surf_copy.set_atomic_numbers(atomic_numbers)
	surf_copy.pbc = True

	indices = list(range(0, onelayer*nlayer // 2))
	constraint(surf_copy, indices=indices)

	return surf_copy

def adsorbate_molecule(surf, ads):
	surf_with_ads = surf.copy()

	add_adsorbate(surf_with_ads, ads, height=1.6)
	surf_with_ads.pbc = True

	indices = list(range(0, onelayer*nlayer // 2))
	constraint(surf_with_ads, indices=indices)

	return surf_with_ads

def get_element_of_adsorption_site(surf, adsorption_pos=0):
	number = surf.get_atomic_numbers()[adsorption_pos]
	symbol = surf.get_chemical_symbols()[adsorption_pos]
	return number, symbol

def get_dos_center(pdos_file=None, save_to=False):
	l_channels = ("s", "p", "d", "f")
	centers = np.zeros(len(l_channels))

	# read in entire file
	f = open(pdos_file, "r")
	data = f.readlines()
	nlines = len(data)

	# find Fermi level in eV
	e_fermi = np.float_(data[0].split(" ")[-2])*27.211

	# find number of angular momentum channels
	nl_channels = len(list(filter(None, data[1].strip("").strip("\n").split(" ")))) - 5

	# read energy vs. DOS data for all components into a single matrix of eV vs. DOS data
	dos_data = []
	for i in range(2, nlines):
		dos_data.append(list(filter(None, data[i].strip(" ").strip("\n").split(" ")))[1:])

	dos_data = np.float_(np.array(dos_data))

	# convert energy axis from Hartree to eV
	dos_data[:,0] = 27.211*dos_data[:,0]

	E1 = e_fermi - 20.0  # inner shells are not necessary
	E2 = e_fermi + 5.0   # take some mergin for unoccupied region
	dE = 1.0e-1
	sigma = 3.0e-1

	for i in range(0, nl_channels):
		dos_energy = np.linspace(E1, E2, int((E2-E1)/dE))
		dos_hist = np.zeros([len(dos_energy), 1])
		j = 0
		for E in dos_energy:
			for (eps, weight) in zip(dos_data[:, 0], dos_data[:, 2+i]):
				dos_hist[j] += 1.0/(np.sqrt(2*np.pi)*sigma)*np.exp(-(E-eps)**2 / (2*sigma**2))*weight
			j += 1
	
		dos_energy = dos_energy[np.newaxis].T - e_fermi
		dos = np.hstack((dos_energy, dos_hist))

		# save to aptly named file
		if save_to is not None:
			outfile = save_to + '_' + l_channels[i] + "_dos" + '.dat'
			print("now saveing dos file to", outfile)
			np.savetxt(outfile, dos)

		centers[i] = np.trapz(dos_energy*dos_hist, dos_energy, axis=0) / np.trapz(dos_hist, dos_energy, axis=0)

	#centers = np.float_(centers).reshape(-1)

	return centers

# ---- start
parser = argparse.ArgumentParser()
parser.add_argument("--jsonfile", help="json file to store data", default="data.json")
parser.add_argument("--nsample",  help="number of samples", default=1, type=int)
args = parser.parse_args()

jsonfile = args.jsonfile
nsample = args.nsample

uid = uuid.uuid1()
uid = str(uid).split("-")[-1]
tmpdir = "tmpdir_" + uid
if not os.path.isdir(tmpdir):
	os.makedirs(tmpdir)

jsonfile = "data.json"

# optimization
steps = 100
max_scf = 20
fmax = 0.1
maxstep = 0.5

ncore = 36
home = os.environ["HOME"]

# cp2k setup
basis_set = "DZVP-MOLOPT-SR-GTH"

cp2k_root  = home + "/" + "cp2k/cp2k-6.1"
cp2k_shell = cp2k_root + "/exe/Linux-x86-64-intel/cp2k_shell.popt"
os.environ["CP2K_DATA_DIR"] = cp2k_root + "/data"
CP2K.command = "mpiexec.hydra -n {0:d} {1:s}".format(ncore, cp2k_shell)

#cp2k_root  = home + "/" + "cp2k/cp2k-7.1.0"
#cp2k_shell = cp2k_root + "/exe/Darwin-IntelMacintosh-gfortran/cp2k_shell.sopt"
#os.environ["CP2K_DATA_DIR"] = cp2k_root + "/data"
#CP2K.command = cp2k_shell

df = pd.DataFrame()
base_surf = make_base_surface()
inp = ''' &FORCE_EVAL
            &DFT
              &XC
                &XC_FUNCTIONAL
				  &PBE
				    PARAMETRIZATION REVPBE
				  &END PBE
                &END XC_FUNCTIONAL
              ! &VDW_POTENTIAL
              !   POTENTIAL_TYPE PAIR_POTENTIAL
              !   &PAIR_POTENTIAL
              !     TYPE DFTD3 # DFTD3(BJ) for Becke-Jonshon dampling
              !     CALCULATE_C9_TERM .TRUE. # optional
              !     REFERENCE_FUNCTIONAL PBE
              !     PARAMETER_FILE_NAME /home/usr6/m70286a/cp2k/cp2k-6.1/data/dftd3.dat
              !     R_CUTOFF 15
              !   &END PAIR_POTENTIAL
              ! &END VDW_POTENTIAL
              &END XC
              &SCF
				SCF_GUESS ATOMIC
                EPS_SCF 1.0E-5
                &OT
                  MINIMIZER DIIS
				  PRECONDITIONER FULL_ALL
                &END OT
              &END SCF
              &POISSON
                PERIODIC XYZ
              &END POISSON
              &PRINT
                &PDOS
                  &LDOS
                    # LIST 1..Natoms
                    LIST surf_start..surf_end
                  &END LDOS
                  FILENAME pdosdir
                &END PDOS
              &END PRINT
            &END DFT
          &END FORCE_EVAL
'''
# adsorbate
ads = Atoms("CO", [[0, 0, 0], [0, 0, 1.2]])
ads.cell = 3*[vacuum]
ads.pbc = True

# calculate for adsorbate molecule
print("calculating energy for adsorbate:", ads.get_chemical_formula())
natoms = len(ads)
label  = tmpdir + "/ads"
#inp_replaced = inp.replace("Natoms", str(natoms))
inp_replaced = inp.replace("surf_start", str(1))
inp_replaced = inp_replaced.replace("surf_end", str(natoms))
inp_replaced = inp_replaced.replace("pdosdir", label)

calc = CP2K(max_scf=max_scf, uks=False,
			basis_set=basis_set, basis_set_file="BASIS_MOLOPT",
			pseudo_potential="GTH-PBE", potential_file="GTH_POTENTIALS", label=label,
			poisson_solver=None, xc=None, print_level="MEDIUM", inp=inp_replaced)
ads.set_calculator(calc)
opt = FIRE(ads, maxstep=maxstep, trajectory=label+".traj")
opt.run(steps=steps, fmax=fmax)
energy_ads = ads.get_potential_energy()
os.system("rm {}*".format(label))

for isample in range(nsample):
	energy_list = np.zeros(2)
	print(" ---- Now {0:d} / {1:d} th sample ---".format(isample+1, nsample))
	#
	# surface
	#
	surf = shuffle(base_surf)
	surf_formula = surf.get_chemical_formula()
	surf_symbols = surf.get_chemical_symbols()
	print("calculating surface:", surf_formula)
	label = tmpdir + "/" + surf_formula

	# optimization
	natoms = len(surf)
	#inp_replaced = inp.replace("Natoms", str(natoms))
	inp_replaced = inp.replace("surf_start", str(onelayer*(nlayer-1)+1))
	inp_replaced = inp_replaced.replace("surf_end", str(natoms))
	inp_replaced = inp_replaced.replace("pdosdir", label)
	calc = CP2K(max_scf=max_scf, uks=True, basis_set=basis_set, basis_set_file="BASIS_MOLOPT",
				pseudo_potential="GTH-PBE", potential_file="GTH_POTENTIALS", label=label,
				poisson_solver=None, xc=None, print_level="MEDIUM", inp=inp_replaced)
	surf.set_calculator(calc)
	opt = FIRE(surf, maxstep=maxstep, trajectory=label+".traj")
	opt.run(steps=steps, fmax=fmax)

	# get dos centers for alpha
	pdos_file = label + "-ALPHA_list1.pdos"
	centers_alpha = get_dos_center(pdos_file=pdos_file, save_to=label+"_alpha")

	# get dos centers for beta
	pdos_file = label + "-BETA_list1.pdos"
	centers_beta = get_dos_center(pdos_file=pdos_file, save_to=label+"_beta")

	centers = (centers_alpha + centers_beta) / 2.0

	energy_list[0] = surf.get_potential_energy()
	#
	# surface + adsorbate
	#
	print("calculating surface + adsorbate:", surf_formula)
	surf_ads = adsorbate_molecule(surf, ads)
	label = tmpdir + "/" + surf_ads.get_chemical_formula()
	calc.set_label(label=label)

	# optimization
	surf_ads.set_calculator(calc)
	opt = FIRE(surf_ads, maxstep=maxstep, trajectory=label+".traj")
	opt.run(steps=steps, fmax=fmax)

	energy_list[1] = surf_ads.get_potential_energy()

	e_ads = energy_list[1] - (energy_list[0] + energy_ads)
	print("{0:s} adsorption energy: {1:5.3f} eV".format(str(ads.get_chemical_formula()), e_ads))

	df_ = pd.DataFrame([[surf_formula, surf_symbols, centers[0], centers[1], centers[2], e_ads]])
	df  = df.append(df_, ignore_index=True)

df.columns = ["surf_formula", "surf_symbols", "s_center", "p_center", "d_center", "ads_energy"]

if os.path.exists(jsonfile):
	df_ = pd.read_json(jsonfile, orient="records", lines=True)
	df = df_.append(df)

df.to_json(jsonfile, orient="records", lines=True)
print("done")

