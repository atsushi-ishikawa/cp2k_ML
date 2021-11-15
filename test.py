import numpy as np
import random
import os
import time
from ase import Atoms
from ase.build import fcc111, add_adsorbate
from ase.visualize import view
from ase.calculators.emt import EMT
from ase.calculators.cp2k import CP2K
from ase.optimize.bfgs import BFGS
from ase.constraints import FixAtoms
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.rcParams['backend'] = 'TkAgg'
import matplotlib.pyplot as plt

def constraint(surf, indices=None):
	c = FixAtoms(indices=indices)
	surf.set_constraint(c)

def make_base_surface(element="Au"):
	vacuum = 10.0
	surf = fcc111(element, size=[2, 2, 3], vacuum=vacuum)

	indices = list(range(0, 4))
	constraint(surf, indices=indices)

	surf.translate([0, 0, -vacuum+0.1])
	surf.pbc = False
	return surf

def shuffle(surf, elements=None):
	if elements is None:
		elements = ["Al", "Ni", "Cu", "Pd", "Ag", "Pt"]

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

	indices = list(range(0, 4))
	constraint(surf_copy, indices=indices)

	return surf_copy

def adsorbate_CO(surf):
	surf_with_ads = surf.copy()
	ads = Atoms("CO", [[0, 0, 0], [0, 0, 1.3]])
	ads.cell = [10.0, 10.0, 10.0]
	ads.pbc = True

	add_adsorbate(surf_with_ads, ads, height=1.5)
	surf_with_ads.pbc = True
	indices = list(range(0, 4))
	constraint(surf_with_ads, indices=indices)

	return [surf_with_ads, ads, surf]

def get_element_of_adsorption_site(surf, adsorption_pos=8):
	number = surf.get_atomic_numbers()[adsorption_pos]
	symbol = surf.get_chemical_symbols()[adsorption_pos]
	return number, symbol

def regression(df, x_index=0, do_plot=True):
	x = df.iloc[:, x_index].values.reshape(-1, 1)  # for 1D-array
	y = df.iloc[:, -1].values

	cv = 5
	test_size = 1.0 / cv

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

	lr = LinearRegression()
	lr.fit(x_train, y_train)

	print(pd.DataFrame({"name": "atomic_num", "Coef": lr.coef_}).sort_values(by="Coef"))
	print("Training set score: {:.3f}".format(lr.score(x_train, y_train)))
	print("Test set score: {:.3f}".format(lr.score(x_test, y_test)))
	print("RMSE: {:.3f}".format(np.sqrt(mean_squared_error(y_test, lr.predict(x_test)))))

	if do_plot:
		plt.scatter(x, y)
		plt.plot(x, lr.predict(x), color="red")
		plt.show()

# ---- start
os.system("rm cp2k*")
pdos_dir = "pdos"
if not os.path.isdir(pdos_dir):
	os.makedirs("./pdos")
else:
	os.system("rm {}/*".format(pdos_dir))

os.environ["CP2K_DATA_DIR"] = "/Users/ishi/cp2k/cp2k-7.1.0/data"

df = pd.DataFrame()
base_surf = make_base_surface()
inp = ''' &FORCE_EVAL
			&DFT
				&SCF
					EPS_SCF 1.0E-1
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
				      LIST 1
				    &END LDOS
				    FILENAME ./pdos/
				  &END PDOS
				&END PRINT
			&END DFT
		  &END FORCE_EVAL				 
'''

num_sample = 3
for isample in range(num_sample):
	surf = shuffle(base_surf)
	surf_ads = adsorbate_CO(surf)

	energy_list = np.zeros(3)
	for imol, mol in enumerate(surf_ads):
		# pre-opt with EMT
		calc = EMT()
		mol.set_calculator(calc)
		opt = BFGS(mol)
		opt.run(steps=100, fmax=0.01)
		pos = mol.get_positions()

		# cp2k calc
		mol.set_positions(pos)
		calc = CP2K(max_scf=10, uks=True,
					basis_set="SZV-MOLOPT-SR-GTH", basis_set_file="BASIS_MOLOPT",
					pseudo_potential="GTH-PBE", potential_file="GTH_POTENTIALS",
					poisson_solver=None,
					xc="PBE", print_level="MEDIUM", inp=inp)
		mol.set_calculator(calc)
		opt = BFGS(mol, maxstep=0.1, trajectory="cp2k.traj")
		opt.run(steps=6)
		energy = mol.get_potential_energy()
		energy_list[imol] = energy

	e_ads = energy_list[0] - (energy_list[1] + energy_list[2])  # notice the order
	num, elem = get_element_of_adsorption_site(surf)
	print("replaced_by={0:s}, adsorption energy={1:f}".format(elem, e_ads))
	df2 = pd.DataFrame([[int(num), e_ads]])
	df  = df.append(df2, ignore_index=True)

df.columns = ["atomic_num", "e_ads"]
print(df)
regression(df, x_index=0)
