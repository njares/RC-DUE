import os
import numpy as np
from scipy import sparse
from scipy.integrate import cumulative_trapezoid, trapezoid

from scipy.sparse.linalg import LinearOperator


def calcula_af_matrix(D_arc_path_sparse, combined_left): # cum_trap_full, path_list, n_t, arc_agg_matrix, AR_flow_matrix):
	'''
	path_list es una matriz que relaciona rutas con arcos
	D_arc_path_sparse es una matriz rala que calcula el delay para todas las rutas y arcos
	'''
	return combined_left.dot(D_arc_path_sparse)

def arc_flows_matrix_old(h, af_matrix, adjoint = False):
	'''
	función que da flujos por arco en función de los flujos por ruta
	opcionalmente, devuelve el adjunto de ese operador
	donde h es el vector de flujos por ruta
	af_matrix es la matriz que hace toda la magia
	'''
	if adjoint:
		# el input tiene dimensiones de flujos por arco, hago un rename para consistencia de nombres
		x = h
		n_t = x.shape[1]
		n_arcs = x.shape[0]
		x_flat = x.reshape(1,-1)[0]
		path_flow_flat = af_matrix.transpose(copy=True).dot(x_flat)
		path_flow_extended_time = path_flow_flat.reshape(-1, n_t*2)
		path_flow = path_flow_extended_time[:,:n_t]
		return path_flow
	else:
		n_t = h.shape[1]
		n_path = h.shape[0]
		# extender horizonte temporal
		h_new = np.zeros((n_path,n_t*2))
		h_new[:,:n_t] = h
		h_flat = h_new.reshape(1,-1)[0]
		arc_flow_flat = af_matrix.dot(h_flat)
		arc_flow = arc_flow_flat.reshape(-1,n_t)
		return arc_flow

def arc_flows_matrix_new(h, af_matrix, adjoint=False):
    if adjoint:
        x = h
        n_t = x.shape[1]
        x_flat = x.reshape(1, -1)[0]
        path_flow_flat = af_matrix.rmatvec(x_flat)
        path_flow_extended_time = path_flow_flat.reshape(-1, n_t * 2)
        path_flow = path_flow_extended_time[:, :n_t]
        return path_flow
    else:
        n_t = h.shape[1]
        n_path = h.shape[0]
        h_new = np.zeros((n_path, n_t * 2))
        h_new[:, :n_t] = h
        h_flat = h_new.reshape(1, -1)[0]
        arc_flow_flat = af_matrix.matvec(h_flat)
        arc_flow = arc_flow_flat.reshape(-1, n_t)
        return arc_flow

def make_af_operator(path_list, trapezoid_integration, arc_agg_matrix,
                     D, n_arc_path, n_t, n_arcs):
    T  = sparse.csr_matrix(trapezoid_integration.T)  # (n_t, 2*n_t)
    Tt = sparse.csr_matrix(trapezoid_integration)     # (2*n_t, n_t)
    rows, cols = arc_agg_matrix.nonzero()
    slots = cols // n_t
    arcs  = rows // n_t
    order = np.argsort(slots, kind='stable')
    slots_s = slots[order]; arcs_s = arcs[order]
    first_occ = np.concatenate([[True], slots_s[1:] != slots_s[:-1]])
    slot_to_arc = arcs_s[first_occ]  # (n_arc_path,)
    path_edgelists = [path[path != 0] - 1 for path in path_list]
    def matvec(h_flat):
        h = h_flat.reshape(len(path_list), 2 * n_t)
        result = np.zeros(n_arcs * n_t)
        slot = 0
        for path_idx, edgelist in enumerate(path_edgelists):
            prev = h[path_idx].copy()          # (2*n_t,)
            for edge in edgelist:
                h_in  = prev
                h_out = D[edge].dot(prev)
                prev  = h_out
                ar    = h_in - h_out           # (2*n_t,)
                trap  = T.dot(ar)              # (n_t,)
                arc_r = slot_to_arc[slot]
                result[arc_r*n_t : (arc_r+1)*n_t] += trap
                slot += 1
        return result
    def rmatvec(x_flat):
        result = np.zeros(len(path_list) * 2 * n_t)
        slot = 0
        for path_idx, edgelist in enumerate(path_edgelists):
            acc = np.zeros(2 * n_t)
            prevs = [None] * (len(edgelist) + 1)
            arc_xs = []
            for local_k, edge in enumerate(edgelist):
                arc_r = slot_to_arc[slot + local_k]
                x_arc = x_flat[arc_r*n_t : (arc_r+1)*n_t]
                arc_xs.append(Tt.dot(x_arc))   # (2*n_t,) each
            back = np.zeros(2 * n_t)
            for local_k in range(len(edgelist) - 1, -1, -1):
                edge = edgelist[local_k]
                back = D[edge].T.dot(back)      # propagate gradient
                back -= arc_xs[local_k]         # out term: -Tt*x
                back += arc_xs[local_k]         # in term cancels... 
            acc = np.zeros(2 * n_t)
            chain_T_vecs = []
            for local_k, edge in enumerate(edgelist):
                v = arc_xs[local_k]                    # Tt x_{arc_k}: (2*n_t,)
                vv = v - D[edge].T.dot(v)              # (I - D[e_k])^T v
                for e in edgelist[:local_k]:
                    vv = D[e].T.dot(vv)
                acc += vv
            result[path_idx * 2*n_t : (path_idx+1) * 2*n_t] = acc
            slot += len(edgelist)
        return result
    n_paths_ext = len(path_list) * 2 * n_t
    n_out = n_arcs * n_t
    return LinearOperator((n_out, n_paths_ext), matvec=matvec, rmatvec=rmatvec)

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
	return sparse.vstack([D_arc_path_in_sparse, D_arc_path_out_sparse]).tocsr()

def calcula_arc_agg_matrix(path_list, n_t):
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

def calcula_D(taus):
	n_arcs = taus.shape[0]
	n_t = taus.shape[1]
	# extender taus al doble del tamaño
	n_t2 = n_t * 2
	new_taus = np.empty((n_arcs, n_t2))
	new_taus[:, :n_t] = taus
	new_taus[:, n_t:] = taus[:, -1:] + np.arange(1, n_t + 1)
	# Basis matrix: identity of size n_t2, but zero out first and last columns
	phi_matrix = np.eye(n_t2)
	phi_matrix[0, :] = 0
	phi_matrix[2*n_t-1, :] = 0
	t_values = np.arange(n_t2)         # shape (n_t2,)
	# Construir D: lista de sparses
	D = [[] for _ in range(n_arcs)]
	# Iterar sobre arcos
	for edge in range(n_arcs):
		tau = new_taus[edge]
		# forma nueva:
		# flow delay calculado directamente aca
		t_0 = np.searchsorted(tau, t_values, side='right') - 1  # shape (n_t2,)
		valid = t_0 >= 0
		# Interpolation weights
		t_0c = np.clip(t_0, 0, n_t2 - 2)
		tau0 = tau[t_0c]
		tau1 = tau[t_0c + 1]
		denom = tau1 - tau0
		# Avoid division by zero (shouldn't happen if tau strictly increasing)
		alpha = np.where(denom != 0, (t_values - tau0) / denom, 0.0)
		d = ((1 - alpha)[:, None] * phi_matrix[t_0c] +
			 alpha[:, None] * phi_matrix[t_0c + 1])
		d[~valid] = 0.0
		D[edge] = sparse.csc_matrix(d).tocsr()
	# D_T = [d.T.tocsr() for d in D]
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
	flows_filename = os.path.join(full_path,network_name,"flows.csv")
	paths_filename = os.path.join(full_path,network_name,"paths.csv")
	edge_times_filename = os.path.join(full_path,network_name,"traversal_time.csv")

	with open(paths_filename, mode='r') as paths_file:
		paths = np.loadtxt(paths_file, delimiter = ",", dtype=int)
	with open(flows_filename, mode='r') as flows_file:
		flows = np.loadtxt(flows_file, delimiter = ",")
	with open(edge_times_filename, mode='r') as edge_times_file:
		edge_delay = np.loadtxt(edge_times_file, delimiter = ",")

	n_t = flows.shape[1]
	arc_delay_paper = edge_delay
	n_arcs = arc_delay_paper.shape[0]
	for t in range(1,n_t):
		arc_delay_paper[:,t] = np.maximum(arc_delay_paper[:,t-1]-.99, arc_delay_paper[:,t])

	taus = np.tile(np.arange(n_t),(n_arcs,1)) + arc_delay_paper
	trapezoid_integration = calcula_trapezoid_integration(n_t)
	path_list = paths
	n_arc_path = np.sum(path_list != 0)
	n_AR_flow = n_arc_path*n_t*2

	D = calcula_D(taus)
	arc_agg_matrix = calcula_arc_agg_matrix(path_list, n_t)
	cum_trap_full = sparse.kron(sparse.eye(n_arc_path), trapezoid_integration.T, format='csr')
	AR_flow_matrix = sparse.hstack([sparse.eye(n_AR_flow),-sparse.eye(n_AR_flow)])

	D_arc_path_sparse = calcula_D_arc_path(path_list, D)
	combined_left = arc_agg_matrix.dot(cum_trap_full).dot(AR_flow_matrix)
	h = flows

	# FORMA VIEJA
	af_matrix = calcula_af_matrix(D_arc_path_sparse, combined_left)
	x_next_old = arc_flows_matrix_old(h, af_matrix)
	# FORMA NUEVA
	af_operator = make_af_operator(path_list, trapezoid_integration, arc_agg_matrix, D, n_arc_path, n_t, n_arcs)
	x_next_new = arc_flows_matrix_new(h, af_operator)

	print(np.allclose( x_next_new, x_next_old ))
