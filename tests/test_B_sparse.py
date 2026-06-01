import os
import numpy as np
from scipy import sparse
from scipy.integrate import cumulative_trapezoid, trapezoid

from scipy.sparse.linalg import LinearOperator

full_path = "/home/njares/Documentos_FIXED/Doctorado/Papers/Dynamic User Equilibrium with Software Implementation/RC-DUE_con_tests/"

for network_name in ["Braess", "Nguyen", "Sioux"]:
	paths_filename = os.path.join(full_path,network_name,"paths.csv")
	edges_data_filename = os.path.join(full_path,network_name,"edges_data.csv")

	with open(paths_filename, mode='r') as paths_file:
		paths = np.loadtxt(paths_file, delimiter = ",", dtype=int)
	with open(edges_data_filename, mode='r') as edges_data_file:
		edges_data = np.loadtxt(edges_data_file, delimiter = ",")

	path_list = paths
	edges = edges_data[:,:2]

	mask = (path_list == 0)
	last_cols = mask.argmax(axis=1)-1
	last_elements = path_list[np.arange(path_list.shape[0]), last_cols]
	OD_edges_raw = np.vstack([path_list[:,0],last_elements]).T
	OD_tuples = [(int(edges[row[0]-1,0]),int(edges[row[1]-1,1])) for row in OD_edges_raw]
	OD_unique = list(dict.fromkeys(OD_tuples))

	# FORMA VIEJA
	B_raw = [ [int(odp==OD_pair) for odp in OD_tuples] for OD_pair in OD_unique]
	B = np.array(B_raw)
	B_old = B.copy()

	# FORMA NUEVA
	# For each path, find which OD_unique index it belongs to
	OD_unique_dict = {od: i for i, od in enumerate(OD_unique)}
	path_od_indices = np.array([OD_unique_dict[odp] for odp in OD_tuples])  # (n_paths,)

	# Build B as sparse directly
	n_od = len(OD_unique)
	n_paths = len(OD_tuples)
	B = sparse.csr_matrix(
		(np.ones(n_paths), (path_od_indices, np.arange(n_paths))),
		shape=(n_od, n_paths)
	)
	B_new = B.copy()

	print(np.allclose( B_new.toarray(), B_old ))
