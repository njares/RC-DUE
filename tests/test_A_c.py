import os
import numpy as np
from scipy import sparse
from scipy.integrate import cumulative_trapezoid, trapezoid

from scipy.sparse.linalg import LinearOperator

def calcula_A_c_old(path_list, taus):
	n_paths = path_list.shape[0]
	n_t = taus.shape[1]
	A = np.zeros((n_paths,n_t))
	for p, path in enumerate(path_list):
		edgelist = path[path != 0] - 1
		for t in range(n_t):
			tau = t
			for edge in edgelist:
				if tau < n_t-1:
					parte_entera = int(tau)
					mantisa = tau - parte_entera
					tau = (1-mantisa)*taus[edge, parte_entera] + mantisa*taus[edge, parte_entera+1]
					#tau = int(taus[edge, tau])
				else:
					tau = int(np.min(taus[edge])+tau)
			A[p,t] = tau - t
	return A

def calcula_A_c_new(path_list, taus):
    n_paths = path_list.shape[0]
    n_t = taus.shape[1]
    A = np.zeros((n_paths, n_t))
    t_values = np.arange(n_t, dtype=float)

    for p, path in enumerate(path_list):
        edgelist = path[path != 0] - 1
        tau = t_values.copy()          # (n_t,) — all time steps at once

        for edge in edgelist:
            # vectorized interpolation over all t simultaneously
            valid = tau < n_t - 1
            parte_entera = np.floor(tau).astype(int)
            mantisa = tau - parte_entera

            parte_entera_c = np.clip(parte_entera, 0, n_t - 2)
            tau_new = ((1 - mantisa) * taus[edge, parte_entera_c] +
                            mantisa  * taus[edge, parte_entera_c + 1])

            # for tau >= n_t-1: tau = min(taus[edge]) + tau
            tau_invalid = np.min(taus[edge]) + tau

            tau = np.where(valid, tau_new, tau_invalid)

        A[p, :] = tau - t_values

    return A


full_path = "/home/njares/Documentos_FIXED/Doctorado/Papers/Dynamic User Equilibrium with Software Implementation/RC-DUE_con_tests/"

for network_name in ["Braess", "Nguyen", "Sioux"]:
	paths_filename = os.path.join(full_path,network_name,"paths.csv")
	edge_times_filename = os.path.join(full_path,network_name,"traversal_time.csv")

	with open(paths_filename, mode='r') as paths_file:
		paths = np.loadtxt(paths_file, delimiter = ",", dtype=int)
	with open(edge_times_filename, mode='r') as edge_times_file:
		edge_delay = np.loadtxt(edge_times_file, delimiter = ",")

	path_list = paths
	n_t = edge_delay.shape[1]
	n_arcs = edge_delay.shape[0]
	arc_delay_paper = edge_delay

	taus = np.tile(np.arange(n_t),(n_arcs,1)) + arc_delay_paper

	A_h_old = calcula_A_c_old(path_list, taus)
	A_h_new = calcula_A_c_new(path_list, taus)

	print(np.allclose( A_h_new, A_h_old ))
