import os
import numpy as np
from scipy import sparse
from scipy.integrate import cumulative_trapezoid, trapezoid

def calcula_trapezoid_integration(n_t):
	integrate = np.zeros((n_t*2,n_t))
	# hacer un bucle sobre cada elemento de la base
	for i in range(n_t*2):
		phi = np.zeros(n_t*2)
		if i not in [0, 2*n_t-1]:
			phi[i] = 1
		# Calcular el offset
		# ToDo: esto está pensado con la discretización de Braess, revisar para los otros grafos
		offset = trapezoid(phi, dx=180)
		# calcular la integral sobre cada elemento
		integrate_phi = cumulative_trapezoid(phi-offset/((2*n_t-1)*180), dx=180, initial=0)
		# Guardar los coeficientes de esos delay en la matriz
		integrate[i,:] = integrate_phi[:n_t]
	return integrate

full_path = "/home/njares/Documentos_FIXED/Doctorado/Papers/Dynamic User Equilibrium with Software Implementation/RC-DUE/"

for network_name in ["Braess", "Nguyen"]:
	paths_filename = os.path.join(full_path,network_name,"paths.csv")
	edges_data_filename = os.path.join(full_path,network_name,"edges_data.csv")

	# Cargo archivo de caminos (son los índices de los arcos que usa)
	with open(paths_filename, mode='r') as paths_file:
		paths = np.loadtxt(paths_file, delimiter = ",", dtype=int)
	# Cargo archivo de datos de arcos
	with open(edges_data_filename, mode='r') as edges_data_file:
		edges_data = np.loadtxt(edges_data_file, delimiter = ",")

	path_list = paths
	n_t = 100
	n_arc_path = np.sum(path_list != 0)

	trapezoid_integration = calcula_trapezoid_integration(n_t)

	cum_trap_full_old = sparse.block_diag([trapezoid_integration.T for _ in range(n_arc_path)])
	cum_trap_full_new = sparse.kron(sparse.eye(n_arc_path), trapezoid_integration.T, format='csr')

	#import pdb;pdb.set_trace()
	print(np.allclose( cum_trap_full_new.toarray(), cum_trap_full_old.toarray() ))
