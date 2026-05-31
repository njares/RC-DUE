import os
import argparse
import time
import cProfile

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
	last_elements = path_list[np.arange(path_list.shape[0]), last_cols]
	OD_edges_raw = np.vstack([path_list[:,0],last_elements]).T
	OD_tuples = [(int(edges[row[0]-1,0]),int(edges[row[1]-1,1])) for row in OD_edges_raw]
	# OD_unique = list(set(OD_tuples))
	OD_unique = list(dict.fromkeys(OD_tuples))
	B_raw = [ [int(odp==OD_pair) for odp in OD_tuples] for OD_pair in OD_unique]
	#B = np.array([[1,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,1,1,0,0],[0,0,0,0,0,0,1,1]])
	B = np.array(B_raw)
	P_full = np.eye( B.shape[1] )-B.T@np.linalg.inv(B@B.T)@B
	#P = sparse.bmat([[P_full[i,j]*sparse.eye(n_t) for j in range(P_full.shape[1])] for i in range(P_full.shape[0])])
	P = sparse.kron(P_full, sparse.eye(n_t), format='csr')
	d_full = B @ h_0
	d0_full = np.zeros(h_0.shape)
	#d0_full[0,:] = d_full[0,:]
	#d0_full[2,:] = d_full[1,:]
	#d0_full[3,:] = d_full[2,:]
	#d0_full[6,:] = d_full[3,:]
	aux_d_index = np.hstack([ np.array(0), np.cumsum([sum(row) for row in B_raw[:-1]])])
	aux_d_matrix = np.zeros((d0_full.shape[0], d_full.shape[0] ))
	aux_d_matrix[aux_d_index, np.arange(aux_d_matrix.shape[1])] = 1
	d0_full = aux_d_matrix @ d_full

	d0 = d0_full.flatten() # tamaño de h, satisface Bh=d

	pHplus = lambda x : np.maximum(x, 0)
	pB0 = lambda x : P@x
	pBd = lambda x : d0 + pB0(x-d0)
	P_lambda_flat = lambda x : project([pBd, pHplus], x)

	def P_lambda(h):
		h_flat = h.flatten()
		h_proy_flat = P_lambda_flat(h_flat)
		return h_proy_flat.reshape(-1, n_t)

	trapezoid_integration = calcula_trapezoid_integration(n_t)
	n_arc_path = np.sum(path_list != 0)
	arc_agg_matrix = calcula_arc_agg_matrix(path_list, n_t)
	#import pdb;pdb.set_trace()

	A = lambda h , arc_delay : A_delay(h, arc_delay, path_list, edges_capacity, edges_fft, trapezoid_integration, arc_agg_matrix)

	for t in range(1,n_t):
		arc_delay_paper[:,t] = np.maximum(arc_delay_paper[:,t-1]-.99, arc_delay_paper[:,t])

	# calcular equilibrio
	rc_due_time = time.time()
	# h_next, arc_delay_next, status = rc_due(h_0, arc_delay_paper, P_lambda, A, epsilon = 6e-7)
	h_next, arc_delay_next, status = rc_due(h_0, arc_delay_paper, P_lambda, A, epsilon = 1e-5)
	rc_due_time = time.time() - rc_due_time
	print(status)

	# calcular flujos por arco finales
	taus = np.tile(np.arange(n_t),(n_arcs,1)) + arc_delay_next
	D = calcula_D(taus)
	af_matrix = make_af_operator(path_list, trapezoid_integration, arc_agg_matrix, D, n_arc_path, n_t, n_arcs)
	x_final = arc_flows_matrix(h_next, af_matrix)

	c_final, _ = A_delay(h_next, arc_delay_next, path_list, edges_capacity, edges_fft, trapezoid_integration, arc_agg_matrix)

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
