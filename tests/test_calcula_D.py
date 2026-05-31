import os
import numpy as np
from scipy import sparse


def calcula_D_new(taus):
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
		D[edge] = sparse.csc_matrix(d.T)
	return D


def calcula_D_old(taus):
	n_arcs = taus.shape[0]
	n_t = taus.shape[1]
	new_taus = np.zeros((n_arcs, n_t*2))
	new_taus[:,:n_t] = taus
	for arc in range(n_arcs):
		new_taus[arc,n_t:] = np.arange(n_t)+taus[arc,-1]+1
	D = [[] for _ in range(n_arcs)]
	for edge in range(n_arcs):
		tau = new_taus[edge]
		d = np.zeros((n_t*2,n_t*2))
		# Calcular el delay de cada elemento de la base
		for i in range(n_t*2):
			phi = np.zeros(n_t*2)
			if i not in [0, 2*n_t-1]:
				phi[i] = 1
			phi_tau = flow_delay(phi, tau)
			# Guardar los coeficientes de esos delay en la matriz
			d[i,:] = phi_tau
		D[edge] = sparse.csc_matrix(d)
	return D


def flow_delay(flow, tau):
	if np.all(flow == 0):
		return flow
	n_t = tau.shape[0]
	new_flow = np.zeros(n_t)
	if not np.all(tau[:-1] < tau[1:]):
		print("tau no es estrictamente creciente!")
		arc_delay = tau - np.arange(n_t)
		from scipy.interpolate import UnivariateSpline
		arc_delay_diff = (UnivariateSpline(np.arange(n_t), arc_delay, s=1e-6).derivative())(np.arange(n_t))
		plt.plot(arc_delay)
		plt.plot(arc_delay_diff)
		plt.show()
		import pdb
		pdb.set_trace()
	t_values = np.arange(n_t)
	if tau[-1] < n_t -1:
		print("tau no abarca todo el horizonte temporal!")
		sys.exit()
	t_0_array = np.searchsorted(tau, t_values, side='right') - 1
	valid_mask = t_0_array >= 0
	new_flow[valid_mask] = np.interp(t_values[valid_mask], tau, flow)
	return new_flow


full_path = "/home/njares/Documentos_FIXED/Doctorado/Papers/Dynamic User Equilibrium with Software Implementation/RC-DUE/"

for network_name in ["Braess", "Nguyen"]:
	edge_times_filename = os.path.join(full_path,network_name,"traversal_time.csv")

	# Cargo archivo de delay por arco
	with open(edge_times_filename, mode='r') as edge_times_file:
		edge_delay = np.loadtxt(edge_times_file, delimiter = ",")

	arc_delay_paper = edge_delay
	n_t = arc_delay_paper.shape[1]
	n_arcs = arc_delay_paper.shape[0]

	for t in range(1,n_t):
		arc_delay_paper[:,t] = np.maximum(arc_delay_paper[:,t-1]-.99, arc_delay_paper[:,t])

	taus = np.tile(np.arange(n_t),(n_arcs,1)) + arc_delay_paper
	D_new = calcula_D_new(taus)
	D_old = calcula_D_old(taus)
	
	#import pdb;pdb.set_trace()
	print([np.allclose(d_new.toarray(), d_old.toarray()) for (d_new,d_old) in zip(D_new, D_old)])
