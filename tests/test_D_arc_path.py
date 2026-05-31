import os
import numpy as np
from scipy import sparse


def calcula_D_arc_path_new(path_list, D):
	'''
	path_list es una matriz que relaciona rutas con arcos
	D es una lista de matrices ralas
	D[edge]: matriz de 2*n_t x 2*n_t
	D[edge] es la matriz de retraso para el arco i
	'''
	n_t = D[0].shape[0]
	D_arc_path_in_list = []
	D_arc_path_out_list = []
	for path in path_list:
		edgelist = path[path != 0] - 1
		cur_in = []
		cur_out = []
		prev_out = sparse.eye(n_t, format='csr')
		for edge in edgelist:
			cur_in.append(prev_out)
			prev_out = D[edge].dot(prev_out)
			cur_out.append(prev_out)
		cur_D_in_sparse = sparse.vstack(cur_in)
		cur_D_out_sparse = sparse.vstack(cur_out)
		D_arc_path_in_list.append(cur_D_in_sparse)
		D_arc_path_out_list.append(cur_D_out_sparse)
	D_arc_path_in_sparse  = sparse.block_diag(D_arc_path_in_list)
	D_arc_path_out_sparse = sparse.block_diag(D_arc_path_out_list)
	return sparse.vstack([D_arc_path_in_sparse, D_arc_path_out_sparse])


def calcula_D_arc_path_old(path_list, D):
	'''
	path_list es una matriz que relaciona rutas con arcos
	D es una lista de matrices ralas
	D[edge]: matriz de 2*n_t x 2*n_t
	D[edge] es la matriz de retraso para el arco i
	'''
	#print("D[0].shape:", D[0].shape)
	#print("path_list.shape:", path_list.shape)
	#print("n_arc_path:", np.sum(path_list != 0))
	n_arc_path = np.sum(path_list != 0)
	n_t = D[0].shape[0]
	D_arc_path_in_list = []
	D_arc_path_out_list = []
	# AR_id = 0
	for path_i, path in enumerate(path_list):
		edgelist = path[path != 0] - 1
		first = True
		cur_D_in_list = []
		cur_D_out_list = []
		for edge in edgelist:
			if first:
				# AR_flow_in[AR_id] = h[path_i]
				cur_D_in_list.append(sparse.eye(n_t))
				first = False
			else:
				# AR_flow_in[AR_id] = AR_flow_out[AR_id-1]
				cur_D_in_list.append(cur_D_out_list[-1])
			# flow_sparse = sparse.csc_matrix(AR_flow_in[AR_id])
			# AR_flow_out[AR_id] = flow_sparse.dot(D[edge]).toarray()[0]
			cur_D_out_list.append(D[edge].dot(cur_D_in_list[-1]))
			# AR_id += 1
		cur_D_in_sparse = sparse.vstack(cur_D_in_list)
		cur_D_out_sparse = sparse.vstack(cur_D_out_list)
		D_arc_path_in_list.append(cur_D_in_sparse)
		D_arc_path_out_list.append(cur_D_out_sparse)
	D_arc_path_in_sparse = sparse.block_diag(D_arc_path_in_list)
	D_arc_path_out_sparse = sparse.block_diag(D_arc_path_out_list)
	D_arc_path_sparse = sparse.vstack([D_arc_path_in_sparse, D_arc_path_out_sparse])
	#print("in_sparse shape:", D_arc_path_in_sparse.shape)
	#print("out_sparse shape:", D_arc_path_out_sparse.shape)
	return D_arc_path_sparse


def calcula_D(taus):
	n_arcs = taus.shape[0]
	n_t = taus.shape[1]
	n_t2 = n_t * 2
	new_taus = np.empty((n_arcs, n_t2))
	new_taus[:, :n_t] = taus
	new_taus[:, n_t:] = taus[:, -1:] + np.arange(1, n_t + 1)
	phi_matrix = np.eye(n_t2)
	phi_matrix[0, :] = 0
	phi_matrix[2*n_t-1, :] = 0
	t_values = np.arange(n_t2)         # shape (n_t2,)
	D = [[] for _ in range(n_arcs)]
	for edge in range(n_arcs):
		tau = new_taus[edge]
		t_0 = np.searchsorted(tau, t_values, side='right') - 1  # shape (n_t2,)
		valid = t_0 >= 0
		t_0c = np.clip(t_0, 0, n_t2 - 2)
		tau0 = tau[t_0c]
		tau1 = tau[t_0c + 1]
		denom = tau1 - tau0
		alpha = np.where(denom != 0, (t_values - tau0) / denom, 0.0)
		d = ((1 - alpha)[:, None] * phi_matrix[t_0c] +
			 alpha[:, None] * phi_matrix[t_0c + 1])
		d[~valid] = 0.0
		D[edge] = sparse.csc_matrix(d).T.tocsr()
	return D


full_path = "/home/njares/Documentos_FIXED/Doctorado/Papers/Dynamic User Equilibrium with Software Implementation/RC-DUE/"

for network_name in ["Braess", "Nguyen"]:
	edge_times_filename = os.path.join(full_path,network_name,"traversal_time.csv")
	paths_filename = os.path.join(full_path,network_name,"paths.csv")

	# Cargo archivo de delay por arco
	with open(edge_times_filename, mode='r') as edge_times_file:
		edge_delay = np.loadtxt(edge_times_file, delimiter = ",")
	# Cargo archivo de caminos (son los índices de los arcos que usa)
	with open(paths_filename, mode='r') as paths_file:
		paths = np.loadtxt(paths_file, delimiter = ",", dtype=int)

	arc_delay_paper = edge_delay
	n_t = arc_delay_paper.shape[1]
	n_arcs = arc_delay_paper.shape[0]

	for t in range(1,n_t):
		arc_delay_paper[:,t] = np.maximum(arc_delay_paper[:,t-1]-.99, arc_delay_paper[:,t])

	taus = np.tile(np.arange(n_t),(n_arcs,1)) + arc_delay_paper
	D = calcula_D(taus)
	path_list = paths

	D_arc_path_sparse_new = calcula_D_arc_path_new(path_list, D)
	D_arc_path_sparse_old = calcula_D_arc_path_old(path_list, D)

	#import pdb;pdb.set_trace()
	print(np.allclose( D_arc_path_sparse_new.toarray(), D_arc_path_sparse_old.toarray() ))
