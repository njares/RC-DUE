import os
import numpy as np
from scipy import sparse
from scipy.integrate import cumulative_trapezoid, trapezoid

def calcula_af_matrix_new(D_arc_path_sparse, combined_left):
	'''
	path_list es una matriz que relaciona rutas con arcos
	D_arc_path_sparse es una matriz rala que calcula el delay para todas las rutas y arcos
	'''
	return combined_left.dot(D_arc_path_sparse)

def calcula_af_matrix_old(D_arc_path_sparse, cum_trap_full, path_list, n_t):
	'''
	path_list es una matriz que relaciona rutas con arcos
	D_arc_path_sparse es una matriz rala que calcula el delay para todas las rutas y arcos
	'''
	n_arc_path = np.sum(path_list != 0)
	n_arcs = np.unique(path_list.flatten()).shape[0] - 1
	n_AR_flow = n_arc_path*n_t*2
	# matriz de flujo neto por arco-ruta
	AR_flow_matrix = sparse.hstack([sparse.eye(n_AR_flow),-sparse.eye(n_AR_flow)])
	# matriz de agregación en arcos de volumen por arco-ruta
	edge_list = path_list.flatten()
	edge_list = edge_list[edge_list.nonzero()]
	arc_agg_ids = np.eye(n_arcs)[:,edge_list-1]
	bands = []
	for idxs in arc_agg_ids:
		bands.append(sparse.hstack([sparse.eye(n_t) if idx else sparse.csr_matrix((n_t,n_t)) for idx in idxs]))
	arc_agg_matrix = sparse.vstack(bands)
	# multiplico todas las matrices
	full_matrix = D_arc_path_sparse.copy()
	full_matrix = AR_flow_matrix.dot(full_matrix)
	full_matrix = cum_trap_full.dot(full_matrix)
	full_matrix = arc_agg_matrix.dot(full_matrix)
	return full_matrix

def calcula_D_arc_path(path_list, D):
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

def calcula_arc_agg_matrix(path_list, n_t):
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
	n_arc_path = np.sum(path_list != 0)

	D_arc_path_sparse = calcula_D_arc_path(path_list, D)
	trapezoid_integration = calcula_trapezoid_integration(n_t)
	cum_trap_full = sparse.block_diag([trapezoid_integration.T for _ in range(n_arc_path)])
	arc_agg_matrix = calcula_arc_agg_matrix(path_list, n_t)
	n_AR_flow = n_arc_path*n_t*2
	AR_flow_matrix = sparse.hstack([sparse.eye(n_AR_flow),-sparse.eye(n_AR_flow)])
	combined_left = arc_agg_matrix.dot(cum_trap_full).dot(AR_flow_matrix)
	combined_left = combined_left.tocsr()  # ensure fast matmul format
	
	af_new = calcula_af_matrix_new(D_arc_path_sparse, combined_left)
	af_old = calcula_af_matrix_old(D_arc_path_sparse, cum_trap_full, path_list, n_t)

	#import pdb;pdb.set_trace()
	print(np.allclose( af_new.toarray(), af_old.toarray() ))
