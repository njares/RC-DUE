import os
import numpy as np
from scipy import sparse

def calcula_D(taus, flow_delay):
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

def flow_delay_old(flow, tau):
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
	for t in range(n_t):
		t_0 = np.min(np.where(tau>t)[0])-1
		if t_0 >= 0:
			if t_0 == n_t-1 and tau[t_0]<t:
				print("tau no abarca todo el horizonte temporal!")
				sys.exit()
			c = (t - tau[t_0+1])/(tau[t_0]-tau[t_0+1])
			new_flow[t] = c*flow[t_0] + (1-c)*flow[t_0+1]
	return new_flow

def flow_delay_new(flow, tau):
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
	D_new = calcula_D(taus, flow_delay_new)
	D_old = calcula_D(taus, flow_delay_old)
	
	#import pdb;pdb.set_trace()
	print([np.allclose(d_new.toarray(), d_old.toarray()) for (d_new,d_old) in zip(D_new, D_old)])
