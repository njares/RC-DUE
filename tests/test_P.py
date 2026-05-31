import os
import numpy as np
from scipy import sparse
from scipy.integrate import cumulative_trapezoid, trapezoid

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
	edges = edges_data[:,:2]
	n_t = 100

	mask = (path_list == 0)
	last_cols = mask.argmax(axis=1)-1
	last_elements = path_list[np.arange(path_list.shape[0]), last_cols]
	OD_edges_raw = np.vstack([path_list[:,0],last_elements]).T
	OD_tuples = [(int(edges[row[0]-1,0]),int(edges[row[1]-1,1])) for row in OD_edges_raw]
	OD_unique = list(dict.fromkeys(OD_tuples))
	B_raw = [ [int(odp==OD_pair) for odp in OD_tuples] for OD_pair in OD_unique]
	B = np.array(B_raw)
	P_full = np.eye( B.shape[1] )-B.T@np.linalg.inv(B@B.T)@B

	P_old = sparse.bmat([[P_full[i,j]*sparse.eye(n_t) for j in range(P_full.shape[1])] for i in range(P_full.shape[0])])
	P_new = sparse.kron(P_full, sparse.eye(n_t), format='csr')

	#import pdb;pdb.set_trace()
	print(np.allclose( P_new.toarray(), P_old.toarray() ))
