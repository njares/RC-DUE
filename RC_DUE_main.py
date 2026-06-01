import os
import argparse
import time
import cProfile
from scipy.linalg import cho_factor, cho_solve

from RC_DUE_helpers import *


def main():
	# profiler = cProfile.Profile()
	# profiler.enable()

	main_time = time.time()
	parser = argparse.ArgumentParser(description="Run RC-DUE.")
	parser.add_argument("network_name", choices=["Braess", "Nguyen", "Sioux", "Anaheim"], help="Name of the network ('Braess', 'Nguyen', 'Sioux', 'Anaheim')")
	args = parser.parse_args()
	
	# Nombre de la red
	network_name = args.network_name
	print(f"Calculando RC-DUE para la red: {network_name}")

	# constantes
	full_path = './'
	flows_filename = os.path.join(full_path,network_name,"flows.csv")
	paths_filename = os.path.join(full_path,network_name,"paths.csv")
	edge_flows_filename = os.path.join(full_path,network_name,"edge_flows.csv")
	edge_times_filename = os.path.join(full_path,network_name,"traversal_time.csv")
	edges_data_filename = os.path.join(full_path,network_name,"edges_data.csv")

	# Cargo archivo de flujos
	with open(flows_filename, mode='r') as flows_file:
		flows = np.loadtxt(flows_file, delimiter = ",")
	# Cargo archivo de caminos (son los índices de los arcos que usa)
	with open(paths_filename, mode='r') as paths_file:
		paths = np.loadtxt(paths_file, delimiter = ",", dtype=int)
	# Cargo archivo de flujos por arco
	with open(edge_flows_filename, mode='r') as edge_flows_file:
		edge_flows = np.loadtxt(edge_flows_file, delimiter = ",")
	# Cargo archivo de delay por arco
	with open(edge_times_filename, mode='r') as edge_times_file:
		edge_delay = np.loadtxt(edge_times_file, delimiter = ",")
	# Cargo archivo de datos de arcos
	with open(edges_data_filename, mode='r') as edges_data_file:
		edges_data = np.loadtxt(edges_data_file, delimiter = ",")

	# cosas necesarias
	h_0 = flows
	path_list = paths
	x_0 = edge_flows
	arc_delay_paper = edge_delay
	edges_capacity = edges_data[:,2] # en unidades veh/s
	edges_fft = edges_data[:,3] # en unidades s
	arc_delay_0 = calcula_arc_delay(x_0, edges_capacity, edges_fft)
	n_t = h_0.shape[1]
	n_arcs = arc_delay_0.shape[0]
	edges = edges_data[:,:2]

	mask = (path_list == 0)
	last_cols = mask.argmax(axis=1)-1
	path_lengths = last_cols + 1  # already computed
	slot_starts = np.concatenate([[0], np.cumsum(path_lengths[:-1])])
	last_elements = path_list[np.arange(path_list.shape[0]), last_cols]
	OD_edges_raw = np.vstack([path_list[:,0],last_elements]).T
	OD_tuples = [(int(edges[row[0]-1,0]),int(edges[row[1]-1,1])) for row in OD_edges_raw]
	OD_unique = list(dict.fromkeys(OD_tuples))
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
	# Precompute once:
	BBT = (B @ B.T).toarray()                    # (1406, 1406)
	BBT_factor = cho_factor(BBT)

	od_path_groups = [np.where(path_od_indices == k)[0] for k in range(n_od)]
	# def P_lambda(h):
	# 	h_out = h.copy()
	# 	for k in range(n_od):
	# 		idx = od_path_groups[k]
	# 		dk = d_full[k]                    # (n_t,)
	# 		h_sub = h[idx, :]                 # (n_paths_k, n_t)
	# 		d0_sub = d0_full[idx, :]          # (n_paths_k, n_t)
	# 		def pBd_sub(x):
	# 			X = x.reshape(-1, n_t)
	# 			# Project onto {h | sum(h) = dk} per time step
	# 			residual = X.sum(axis=0) - dk  # (n_t,)
	# 			# subtract residual equally across paths
	# 			return (X - residual / len(idx)).ravel()
	# 		def pHplus_sub(x):
	# 			return np.maximum(x, 0)
	# 		h_sub_flat = project([pBd_sub, pHplus_sub], h_sub.flatten())
	# 		h_out[idx, :] = h_sub_flat.reshape(-1, n_t)
	# 	return h_out

	def P_lambda(h):
		h_out = h.copy()
		for k in range(n_od):
			idx = od_path_groups[k]
			dk = d_full[k]                    # (n_t,)
			h_sub = h[idx, :]                 # (n_paths_k, n_t)
			# Apply closed-form simplex projection per time step
			h_out[idx, :] = project_simplex_scaled_matrix(h_sub, dk)
		return h_out

	def project_simplex_scaled_matrix(V, c):
		"""Project each column of V onto {x >= 0, sum(x) = c[t]} — closed form"""
		# V: (n_k, n_t), c: (n_t,)
		n_k, n_t = V.shape
		out = np.zeros_like(V)
		for t in range(n_t):
			ct = c[t]
			v = V[:, t]
			if ct <= 0:
				out[:, t] = 0.0
				continue
			u = np.sort(v)[::-1]
			cssv = np.cumsum(u)
			rho_candidates = np.where(u + (ct - cssv) / np.arange(1, n_k + 1) > 0)[0]
			if len(rho_candidates) == 0:
				out[:, t] = np.maximum(v, 0)
				continue
			rho = rho_candidates[-1]
			theta = (cssv[rho] - ct) / (rho + 1)
			out[:, t] = np.maximum(v - theta, 0)
		return out

	#def apply_P(x_flat):
	#	X = x_flat.reshape(-1, n_t)                          # (n_paths, n_t)
	#	# B @ X: sum rows of X grouped by OD pair
	#	BX = np.array([np.bincount(path_od_indices, weights=X[:, t], minlength=n_od) 
    #               for t in range(n_t)]).T                   # (n_od, n_t)
	#	solved = cho_solve(BBT_factor, BX)                   # (n_od, n_t)
	#	# B.T @ solved: scatter solved back to each path
	#	return (X - solved[path_od_indices]).ravel()
	#pB0 = lambda x: apply_P(x)

	d_full = B @ h_0
	d0_full = np.zeros(h_0.shape)
	row_counts = np.diff(B.indptr)  # nnz per row = number of paths per OD pair
	aux_d_index = np.hstack([[0], np.cumsum(row_counts[:-1])])
	d0_full[aux_d_index, :] = d_full
	d0 = d0_full.flatten() # tamaño de h, satisface Bh=d
	#import pdb;pdb.set_trace()


	# pHplus = lambda x : np.maximum(x, 0)
	# pBd = lambda x : d0 + pB0(x-d0)
	# P_lambda_flat = lambda x : project([pBd, pHplus], x)

	#def P_lambda(h):
	#	h_flat = h.flatten()
	#	h_proy_flat = P_lambda_flat(h_flat)
	#	return h_proy_flat.reshape(-1, n_t)

	trapezoid_integration = calcula_trapezoid_integration(n_t)
	n_arc_path = np.sum(path_list != 0)

	A = lambda h , arc_delay : A_delay(h, arc_delay, path_list, edges_capacity, edges_fft, trapezoid_integration, slot_starts)

	for t in range(1,n_t):
		arc_delay_paper[:,t] = np.maximum(arc_delay_paper[:,t-1]-.99, arc_delay_paper[:,t])

	# calcular equilibrio
	rc_due_time = time.time()
	h_next, arc_delay_next, status = rc_due(h_0, arc_delay_paper, P_lambda, A, epsilon = 6e-7)
	#h_next, arc_delay_next, status = rc_due(h_0, arc_delay_paper, P_lambda, A, epsilon = 1e-5)#, max_iter=1)
	rc_due_time = time.time() - rc_due_time
	print(status)

	# calcular flujos por arco finales
	taus = np.tile(np.arange(n_t),(n_arcs,1)) + arc_delay_next
	D = calcula_D(taus)
	af_matrix = make_af_operator(path_list, trapezoid_integration, D, n_arc_path, n_t, n_arcs, slot_starts)
	x_final = arc_flows_matrix(h_next, af_matrix)

	c_final, _ = A_delay(h_next, arc_delay_next, path_list, edges_capacity, edges_fft, trapezoid_integration, slot_starts)

	# guardar costo final
	np.savetxt(network_name+"/route_traversal_time_RC_DUE.csv", c_final, delimiter = ",")
	np.savetxt(network_name+"/traversal_time_RC_DUE.csv", arc_delay_next, delimiter = ",")
	np.savetxt(network_name+"/flows_RC_DUE.csv", h_next, delimiter = ",")
	np.savetxt(network_name+"/edge_flows_RC_DUE.csv", x_final, delimiter = ",")

	main_time = time.time() - main_time
	print(f"RC DUE elapsed time: {rc_due_time:.4f} seconds")
	print(f"Total elapsed time: {main_time:.4f} seconds")

	# profiler.disable()
	# profiler.dump_stats("profile.prof")


if __name__ == "__main__":
	main()
