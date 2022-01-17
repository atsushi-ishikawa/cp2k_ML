import numpy as np
import random
import os
from ase import Atoms
from ase.build import fcc111, add_adsorbate
from ase.visualize import view
from ase.calculators.emt import EMT
from ase.calculators.cp2k import CP2K
from ase.optimize.bfgs import BFGS
from ase.constraints import FixAtoms
import pandas as pd
import argparse

def constraint(surf, indices=None):
	c = FixAtoms(indices=indices)
	surf.set_constraint(c)

def make_base_surface(element="Au"):
	vacuum = 8.0
	surf = fcc111(element, size=[2, 2, 3], vacuum=vacuum)

	indices = list(range(0, 8))
	constraint(surf, indices=indices)

	surf.translate([0, 0, -vacuum+0.1])
	surf.pbc = False
	return surf

def shuffle(surf, elements=None):
	if elements is None:
		#elements = ["Al", "Ni", "Cu", "Pd", "Ag", "Pt"]
		elements = ["Cu", "Pd", "Ag", "Pt"]

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

	indices = list(range(0, 8))
	constraint(surf_copy, indices=indices)

	return surf_copy

def adsorbate_molecule(surf, ads):
	surf_with_ads = surf.copy()

	add_adsorbate(surf_with_ads, ads, height=1.5)
	surf_with_ads.pbc = True
	indices = list(range(0, 8))
	constraint(surf_with_ads, indices=indices)

	return [surf_with_ads, surf]

def get_element_of_adsorption_site(surf, adsorption_pos=0):
	number = surf.get_atomic_numbers()[adsorption_pos]
	symbol = surf.get_chemical_symbols()[adsorption_pos]
	return number, symbol

def get_fermi_energy(pdos_file=None):
	f = open(pdos_file, "r")
	line = f.readline()
	e_fermi = line.split("=")[2].strip().split(" ")[0]
	e_fermi = float(e_fermi)
	return e_fermi

def get_dos_center(pdos_file=None):
	dos = np.loadtxt(pdos_file, skiprows=2)
	energy = dos[:, 1]
	s_dos  = dos[:, 3]
	p_dos  = dos[:, 4] + dos[:, 5] + dos[:, 6]
	d_dos  = dos[:, 7] + dos[:, 8] + dos[:, 9] + dos[:, 10] + dos[:, 11]
	s_center = np.trapz(energy*s_dos, energy) / np.trapz(s_dos, energy)
	p_center = np.trapz(energy*p_dos, energy) / np.trapz(p_dos, energy)
	d_center = np.trapz(energy*d_dos, energy) / np.trapz(d_dos, energy)
	return s_center, p_center, d_center

# ---- start
parser = argparse.ArgumentParser()
parser.add_argument("--jsonfile", help="json file to store data", default="data.json")
parser.add_argument("--nsample",  help="number of samples", default=1, type=int)
args = parser.parse_args()

jsonfile = args.jsonfile
nsample = args.nsample

os.system("rm cp2k* >& /dev/null")
pdos_dir = "pdos"
if not os.path.isdir(pdos_dir):
	os.makedirs("pdos")
else:
	os.system("rm {}/*".format(pdos_dir))

jsonfile = "data.json"

steps = 5
max_scf = 10
ncore = 36

home = os.environ["HOME"]

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
				&SCF
					EPS_SCF 1.0E-3
					&OT
						MINIMIZER DIIS
					&END
				&END SCF
				&POISSON
					PERIODIC XYZ
				&END POISSON
				&PRINT
				  &PDOS
				    &LDOS
				      COMPONENTS .TRUE.
				      LIST 1..Natoms
				    &END LDOS
				    FILENAME ./pdos/
				  &END PDOS
				&END PRINT
			&END DFT
		  &END FORCE_EVAL				 
'''

# adsorbate
ads = Atoms("CO", [[0, 0, 0], [0, 0, 1.3]])
ads.cell = [10.0, 10.0, 10.0]
ads.pbc = True

# calculate for adsorbate molecule
print("calculating energy for adsorbate:", ads.get_chemical_formula())
natoms = len(ads)
inp_replaced = inp.replace("Natoms", str(natoms))
calc = CP2K(max_scf=max_scf, uks=True,
			basis_set="SZV-MOLOPT-SR-GTH", basis_set_file="BASIS_MOLOPT",
			pseudo_potential="GTH-PBE", potential_file="GTH_POTENTIALS",
			poisson_solver=None, xc="PBE", print_level="MEDIUM", inp=inp_replaced)
ads.set_calculator(calc)
opt = BFGS(ads, maxstep=0.1, trajectory="cp2k.traj")
opt.run(steps=steps)
energy_ads = ads.get_potential_energy()

for isample in range(nsample):
	print(" ---- Now {0:d} / {1:d} th sample ---".format(isample, nsample))
	surf = shuffle(base_surf)
	surf_ads = adsorbate_molecule(surf, ads)
	surf_formula = surf_ads[1].get_chemical_formula()
	surf_symbols = surf_ads[1].get_chemical_symbols()

	energy_list = np.zeros(2)
	for imol, mol in enumerate(surf_ads):
		print("now calculating:", mol.get_chemical_formula())

		# pre-opt with EMT
		calc = EMT()
		mol.set_calculator(calc)
		opt = BFGS(mol)
		opt.run(steps=30, fmax=0.05)
		pos = mol.get_positions()

		# cp2k calc
		mol.set_positions(pos)
		natoms = len(mol)
		inp_replaced = inp.replace("Natoms", str(natoms))
		calc = CP2K(max_scf=max_scf, uks=True,
					basis_set="SZV-MOLOPT-SR-GTH", basis_set_file="BASIS_MOLOPT",
					pseudo_potential="GTH-PBE", potential_file="GTH_POTENTIALS",
					poisson_solver=None, xc="PBE", print_level="MEDIUM", inp=inp_replaced)
		mol.set_calculator(calc)
		opt = BFGS(mol, maxstep=0.1, trajectory="cp2k.traj")
		opt.run(steps=steps)
		energy = mol.get_potential_energy()
		energy_list[imol] = energy

		if imol == 1:  # surf
			pdos_file = pdos_dir + "/" + "cp2k-ALPHA_list1.pdos"
			e_fermi = get_fermi_energy(pdos_file=pdos_file)
			s_center, p_center, d_center = get_dos_center(pdos_file=pdos_file)
			s_center -= e_fermi
			p_center -= e_fermi
			d_center -= e_fermi

	e_ads = energy_list[0] - (energy_list[1] + energy_ads)

	df_ = pd.DataFrame([[surf_formula, surf_symbols, s_center, p_center, d_center, e_ads]])
	df  = df.append(df_, ignore_index=True)

df.columns = ["surf_formula", "surf_symbols", "s_center", "p_center", "d_center", "ads_energy"]

if os.path.exists(jsonfile):
	df_ = pd.read_json(jsonfile, orient="records", lines=True)
	df = df_.append(df)

df.to_json(jsonfile, orient="records", lines=True)

