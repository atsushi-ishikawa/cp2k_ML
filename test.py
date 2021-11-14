import numpy as np
import random
import os
import time
from ase import Atoms
from ase.build import fcc111, add_adsorbate
from ase.visualize import view
from ase.calculators.emt import EMT
from ase.calculators.cp2k import CP2K
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.rcParams['backend'] = 'TkAgg'
import matplotlib.pyplot as plt

def make_base_surface(element="Au"):
	vacuum = 10.0
	surf = fcc111(element, size=[2, 2, 3], vacuum=vacuum)
	surf.translate([0, 0, -vacuum+0.1])
	surf.pbc = False
	return surf

def shuffle(surf, elements=None):
	if elements is None:
		elements = ["Al", "Ni", "Cu", "Pd", "Ag", "Pt"]

	surf_copy = surf.copy()
	num_atoms = len(surf_copy.get_atomic_numbers())
	num_replace = random.choice(range(1, num_atoms))

	for iatom in range(num_replace):
		surf_copy[iatom].symbol = random.choice(elements)

	atomic_numbers = surf_copy.get_atomic_numbers()
	atomic_numbers = list(atomic_numbers)
	np.random.shuffle(atomic_numbers)
	surf_copy.set_atomic_numbers(atomic_numbers)
	surf_copy.pbc = True

	return surf_copy

def adsorbate_CO(surf):
	surf_copy = surf.copy()
	mol = Atoms("CO", [[0, 0, 0], [0, 0, 1.3]])
	mol.cell = [10.0, 10.0, 10.0]
	mol.pbc = True

	add_adsorbate(surf_copy, mol, height=1.5)
	surf_copy.pbc = True
	return [surf_copy, mol, surf]

def get_adsorption_pos_number(surf, adsorption_pos=8):
	return surf.get_atomic_numbers()[adsorption_pos]

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

df = pd.DataFrame()
base_surf = make_base_surface()
num_sample = 2
inp = ''' &FORCE_EVAL
			&DFT
				&SCF
					EPS_SCF 1.0E-2
					&OT
						MINIMIZER DIIS
					&END
				&END SCF
				!&POISSON
				!	PERIODIC XYZ
				!&END POISSON
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

for isample in range(num_sample):
	os.system("rm cp2k*")
	pdos_dir = "pdos"
	if not os.path.isdir(pdos_dir):
		os.makedirs("./pdos")
	else:
		os.system("rm {}/*".format(pdos_dir))

	surf = shuffle(base_surf)
	surf_mol_ads = adsorbate_CO(surf)

	energy_list = []
	for imol in surf_mol_ads:
		#imol.calc = EMT()
		calc = CP2K(max_scf=10, uks=True, basis_set="SZV-MOLOPT-SR-GTH",
					poisson_solver=None,
					xc="PBE", print_level="MEDIUM", inp=inp)
		imol.set_calculator(calc)
		energy = imol.get_potential_energy()
		energy_list.append(energy)
		time.sleep(10)

	e_ads = energy_list[2] - sum(energy_list[0:1])
	num = get_adsorption_pos_number(surf)

	print("site={0:d}, adsorption energy={1:f}".format(num, e_ads))
	df2 = pd.DataFrame([[int(num), e_ads]])
	df  = df.append(df2, ignore_index=True)

df.columns = ["atomic_num", "e_ads"]
print(df)
regression(df, x_index=0)
