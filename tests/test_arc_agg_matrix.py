import os
import numpy as np
from scipy import sparse
from scipy.integrate import cumulative_trapezoid, trapezoid



def calcula_arc_agg_matrix_old(path_list, n_t):
	n_arc_path = np.sum(path_list != 0)
	n_arcs = np.unique(path_list.flatten()).shape[0] - 1
	edge_list = path_list.flatten()
	edge_list = edge_list[edge_list.nonzero()]
	arc_agg_ids = np.eye(n_arcs)[:,edge_list-1]
	bands = []
	for idxs in arc_agg_ids:
		bands.append(sparse.hstack([sparse.eye(n_t) if idx else sparse.csr_matrix((n_t,n_t)) for idx in idxs]))
	arc_agg_matrix = sparse.vstack(bands)
	return arc_agg_matrix

def calcula_arc_agg_matrix_new(path_list, n_t):
	n_arc_path = np.sum(path_list != 0)
	n_arcs = np.unique(path_list.flatten()).shape[0] - 1
	edge_list = path_list.flatten()
	edge_list = edge_list[edge_list.nonzero()]
	arc_indices = edge_list - 1  # shape: (n_arc_path,)
	rows = []
	cols = []
	for arc_idx, edge_idx in enumerate(arc_indices):
		# Each arc-path block contributes n_t rows (arc edge_idx) and n_t cols (arc_path arc_idx)
		r = np.arange(edge_idx * n_t, (edge_idx + 1) * n_t)
		c = np.arange(arc_idx * n_t, (arc_idx + 1) * n_t)
		rows.append(r)
		cols.append(c)
	rows = np.concatenate(rows)
	cols = np.concatenate(cols)
	data = np.ones(len(rows))
	arc_agg_matrix = sparse.csr_matrix(
		(data, (rows, cols)),
		shape=(n_arcs * n_t, n_arc_path * n_t)
	)
	return arc_agg_matrix

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

	arc_agg_matrix_old = calcula_arc_agg_matrix_old(path_list, n_t)
	arc_agg_matrix_new = calcula_arc_agg_matrix_new(path_list, n_t)

	#import pdb;pdb.set_trace()
	print(np.allclose( arc_agg_matrix_new.toarray(), arc_agg_matrix_old.toarray() ))
