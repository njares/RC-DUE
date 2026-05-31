# quiero resolver
# < A(h), g-h> >= 0, \forall g \in \Lambda_d
# Como \Lambda_d es cerrado y convexo, las soluciones de VI(\Lambda_d, A) coinciden con las de la ecuación normal:
# h = \Pi_{\Lambda_d} (h - A(h))
# En particular, las soluciones de VI(\Lambda_d, A) coinciden con las de VI(\Lambda_d, c*A), con c>0, pueso eso no cambia el sentido de la desigualdad
# Por lo que también comparte sus soluciones con:
# h = \Pi_{\Lambda_d} (h - c * A(h))
# Luego es punto fijo de un operador, pero no se si ese operador es contractivo.

# El algoritmo entonces va a ser:
# h_{k+1} = \Pi_{\Lambda_d} (h_k - \nu A(h))
# El criterio de parada va a ser:
# ( || h_{k+1} - h_{k} ||^2 / || h_{k} ||^2 ) < \eps
# eps \in [1e-3 , 1e-4]

import numpy as np
from scipy import sparse
from scipy.integrate import cumulative_trapezoid, trapezoid
import sys
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator


def project(P,x0,max_iter=1000,tol=1e-6):
    assert len(x0.shape) == 1, "x0 must be a vector"
    x = x0.copy()
    p = len(P)
    y = np.zeros((p,x0.shape[0]))
    n = 0
    cI = float('inf')
    while n < max_iter and cI >= tol:
        cI = 0
        for i in range(0,p):
            # Update iterate
            prev_x = x.copy()
            x = P[i](prev_x - y[i,:])
            # Update increment
            prev_y = y[i,:].copy()
            y[i,:] = x - (prev_x - prev_y)
            # Stop condition
            cI += np.linalg.norm(prev_y - y[i,:])**2
            n += 1
    return x


pHplus = lambda x : np.maximum(x, 0)
pB0 = lambda x : P@x
pBd = lambda x : d0 + pB0(x-d0)
P_lambda = lambda x : project([pBd, pHplus], x)


def stop_criteria(h_cur, h_next, epsilon):
	num = np.sum((h_cur-h_next)**2)
	den = np.sum((h_cur)**2)
	err = num/den
	stop_crit = (err < epsilon)
	return err, stop_crit


def update_nu(nu, err, alpha, beta, err_hist, patience, min_dec):
	if len(err_hist)<patience:
		# si no hay suficiente historia, no hacer nada con nu
		err_hist = err_hist + [err]
	else:
		# calculo el error promedio
		mean_err = np.mean(err_hist)
		# si creció el error, ir más lento
		if err > mean_err:
			nu = nu*alpha
		# si el error no bajó un mínimo, ir más rápido
		elif err > mean_err*(1-min_dec):
			nu = nu*beta
		# agrego el nuevo error al histórico, y saco el más viejo
		err_hist = err_hist[1:] + [err]
	return nu, err_hist


def rc_due(h_0, arc_delay_0, P_lambda, A_delay, epsilon = 1e-3, alpha = 1/2, beta = 2, max_iter = 100, patience = 2, min_dec = 0.01):
	h_cur = h_0.copy()
	arc_delay_cur = arc_delay_0.copy()
	err_hist = []
	#nu = 1
	nu = 1/4
	for k in range(max_iter):
		A_h, arc_delay_next = A_delay(h_cur, arc_delay_cur)
		h_next = P_lambda( h_cur - nu * A_h )
		err, stop_crit = stop_criteria(h_cur, h_next, epsilon)
		if stop_crit:
			status = "Solution Found"
			break
		nu, err_hist = update_nu(nu, err, alpha, beta, err_hist, patience, min_dec)
		h_cur = h_next.copy()
		arc_delay_cur = arc_delay_next.copy()
		print(f"{k}: err = {err}; step {nu}")
	if k == max_iter-1:
		status = "Max iterations"
	return h_next, arc_delay_next, status


def arc_flows_matrix(h, af_matrix, adjoint=False):
	'''
	función que da flujos por arco en función de los flujos por ruta
	opcionalmente, devuelve el adjunto de ese operador
	donde h es el vector de flujos por ruta
	af_matrix es la matriz que hace toda la magia
	'''
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
		arc_flow_flat = af_matrix.matvec(h_flat)  # or af_matrix @ h_flat, same thing
		arc_flow = arc_flow_flat.reshape(-1, n_t)
		return arc_flow


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

# # Forma sin el for edge in range n_arcs
# # All arcs at once — shapes become (n_arcs, n_t2)
#      t_0 = np.searchsorted(new_taus, t_values, side='right') - 1  # (n_arcs, n_t2)
#      valid = t_0 >= 0
#      t_0c = np.clip(t_0, 0, n_t2 - 2)
#  
#      tau0 = new_taus[np.arange(n_arcs)[:, None], t_0c]          # (n_arcs, n_t2)
#      tau1 = new_taus[np.arange(n_arcs)[:, None], t_0c + 1]
#      denom = tau1 - tau0
#      c = np.where(denom != 0, (t_values - tau1) / denom, 0.0)   # (n_arcs, n_t2)
#  
#      # d[arc, t, :] = c[arc,t]*phi[t_0c[arc,t],:] + (1-c[arc,t])*phi[t_0c[arc,t]+1,:]
#      # phi_matrix[t_0c] -> shape (n_arcs, n_t2, n_t2)
#      d_all = (c[:, :, None] * phi_matrix[t_0c] +
#               (1 - c)[:, :, None] * phi_matrix[t_0c + 1])       # (n_arcs, n_t2, n_t2)
#      d_all[~valid] = 0.0
#  
#      D = [sparse.csc_matrix(d_all[edge]) for edge in range(n_arcs)]
# 
# # Otra forma, supuestamente sin el csr
# def calcula_D(taus):
#     n_arcs = taus.shape[0]
#     n_t = taus.shape[1]
#     n_t2 = n_t * 2
# 
#     new_taus = np.empty((n_arcs, n_t2))
#     new_taus[:, :n_t] = taus
#     new_taus[:, n_t:] = taus[:, -1:] + np.arange(1, n_t + 1)
# 
#     t_values = np.arange(n_t2)  # (n_t2,)
# 
#     # Vectorized over arcs
#     t_0 = np.searchsorted(new_taus, t_values, side='right') - 1  # (n_arcs, n_t2)
#     valid = t_0 >= 0
#     t_0c = np.clip(t_0, 0, n_t2 - 2)
# 
#     tau0 = new_taus[np.arange(n_arcs)[:, None], t_0c]
#     tau1 = new_taus[np.arange(n_arcs)[:, None], t_0c + 1]
#     denom = tau1 - tau0
#     alpha = np.where(denom != 0, (t_values - tau0) / denom, 0.0)  # (n_arcs, n_t2)
# 
#     # Zero out invalid rows
#     alpha[~valid] = 0.0
#     w0 = (1 - alpha)  # weight for t_0c column
#     w1 = alpha        # weight for t_0c+1 column
# 
#     # Apply phi_matrix zeroing: rows 0 and 2*n_t-1 of D are zero
#     # phi_matrix zeros out rows 0 and n_t2-1 of the output matrix
#     # i.e. if t_0c or t_0c+1 == 0 or == n_t2-1, those contributions are 0
#     w0 = np.where((t_0c == 0) | (t_0c == n_t2 - 1), 0.0, w0)
#     w1 = np.where((t_0c + 1 == 0) | (t_0c + 1 == n_t2 - 1), 0.0, w1)
# 
#     # Build all sparse matrices at once using COO
#     # Each matrix is (n_t2 x n_t2) with 2 nonzeros per row
#     row_idx = np.tile(np.arange(n_t2), n_arcs)                    # (n_arcs*n_t2,)
#     col0 = t_0c.ravel()                                            # (n_arcs*n_t2,)
#     col1 = (t_0c + 1).ravel()
#     dat0 = w0.ravel()
#     dat1 = w1.ravel()
# 
#     D = []
#     for edge in range(n_arcs):
#         sl = slice(edge * n_t2, (edge + 1) * n_t2)
#         rows = row_idx[sl]
#         data = np.concatenate([dat0[sl], dat1[sl]])
#         cols = np.concatenate([col0[sl], col1[sl]])
#         mat = sparse.csr_matrix((data, (rows, cols)), shape=(n_t2, n_t2))
#         D.append(mat)
# 
#     return D

def calcula_arc_delay(x, cap, fft):
	'''
	x: arreglo de <# arcos> filas y <n_t> columnas
		Cada fila tiene la cantidad de vehículos en ese arco en cada uno de los n_t instantes de tiempo
	cap: arreglo de <# arcos> filas
		Es la capacidad de cada arco, en veh/s
	fft: arreglo de <# arcos> filas
		Es el free flow time de cada arco, en [s]
	
	Usamos la formula 
	arc delay = free flow time * ( 1 + B * ( flow / capacity ) ^ Power ).
	
	Donde el free flow time es en unidades de discretización temporal (dt = 180)
	y la capacidad es en veh, por lo que usamos cap*fft para nuestra cap.
	B = 0.15, Power = 4
	'''
	# Braess:
	# FFT = 360 s (esto es "2" en el dataset, porque son "2" periodos de "dt=180")
	# capacity = 0.5 veh/s
	dt = 180
	cap_veh = (cap*fft).reshape(-1,1)
	FFT_dt = (fft/dt).reshape(-1,1)
	#arc_delay = 2*(1+0.15*(x/180)**4)
	arc_delay = FFT_dt*(1+0.15*(x/cap_veh)**4)
	n_t = x.shape[1]
	for t in range(1,n_t):
		arc_delay[:,t] = np.maximum(arc_delay[:,t-1]-.99, arc_delay[:,t])
	return arc_delay

def calcula_A_c(path_list, taus):
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

def A_delay(h, arc_delay, path_list, cap, fft, trapezoid_integration, arc_agg_matrix):
	n_t = h.shape[1]
	n_arcs = arc_delay.shape[0]
	n_arc_path = np.sum(path_list != 0)
	# calcular matriz de flujo por arco a partir de los delays por arco
	taus = np.tile(np.arange(n_t),(n_arcs,1)) + arc_delay
	D = calcula_D(taus)
	af_matrix = make_af_operator(path_list, trapezoid_integration, arc_agg_matrix, D, n_arc_path, n_t, n_arcs)
	# calcular flujos por arco a partir de los flujos por ruta y los delays por arco
	x_next = arc_flows_matrix(h, af_matrix)
	# calcular delays por arco a partir de los nuevos flujos por arco
	arc_delay_next = calcula_arc_delay(x_next, cap, fft)
	# calcular delays por ruta a partir de los delays por arco
	taus_next = np.tile(np.arange(n_t),(n_arcs,1)) + arc_delay_next
	A_h = calcula_A_c(path_list, taus_next)
	return A_h, arc_delay_next

def plot_final(h_final, h_inicial, x_inicial, x_final, c_final, c_old):
	# genero la figura
	fig = plt.figure(figsize=(20,9), layout="constrained")
	# ax1 = fig.add_subplot(441) # row-col-num
	row = 3
	col = 5
	# ploteo los flujos por ruta
	#for j in range(h_inicial.shape[0]):
	for j in range(11):
		num = j+1
		cur_ax = fig.add_subplot(row,col,num)
		cur_ax.plot(h_inicial[j], label=f"flujo inicial en la ruta {j}")
		cur_ax.plot(h_final[j], label=f"flujo final en la ruta {j}")
		cur_ax.set_title(f"flujos en la ruta {j}")
		cur_ax.plot(c_final[j], label=f"costo calculado en la ruta {j}")
		cur_ax.plot(c_old[j], label=f"costo paper en la ruta {j}")
		#cur_ax.set_ylim(-0.6, 2.0)
		cur_ax.legend()
	# ploteo los flujos por arco
	#for j in range(x_inicial.shape[0]):
	for j in range(5):
		cur_ax = fig.add_subplot(row,col,j+11)
		cur_ax.plot(x_inicial[j,:], label=f"flujo original en el arco {j+1}")
		cur_ax.plot(x_final[j,:], label=f"flujo final en el arco {j+1}")
		cur_ax.set_title(f"flujo en el arco {j+1}")
		cur_ax.set_ylim(-20, 400)
		cur_ax.legend()
	plt.show()

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

def calcula_A_c_old(path_list, taus):
	n_paths = path_list.shape[0]
	n_t = taus.shape[1]
	A = np.zeros((n_paths,n_t))
	for p, path in enumerate(path_list):
		edgelist = path[path != 0] - 1
		for t in range(n_t):
			tau = t
			for edge in edgelist:
				if tau < n_t:
					tau = int(taus[edge, tau])
				else:
					tau = int(np.min(taus[edge])+tau)
			A[p,t] = tau - t
	return A

def to_int32(m):
	m = m.tocsr()
	m.indices = m.indices.astype(np.int32, copy=False)
	m.indptr = m.indptr.astype(np.int32, copy=False)
	return m


def make_af_operator(path_list, trapezoid_integration, arc_agg_matrix,
                     D, n_arc_path, n_t, n_arcs):
    T  = sparse.csr_matrix(trapezoid_integration.T)  # (n_t, 2*n_t)
    Tt = sparse.csr_matrix(trapezoid_integration)     # (2*n_t, n_t)
    # Precompute slot -> arc lookup
    rows, cols = arc_agg_matrix.nonzero()
    slots = cols // n_t
    arcs  = rows // n_t
    order = np.argsort(slots, kind='stable')
    slots_s = slots[order]; arcs_s = arcs[order]
    first_occ = np.concatenate([[True], slots_s[1:] != slots_s[:-1]])
    slot_to_arc = arcs_s[first_occ]  # (n_arc_path,)
    path_edgelists = [path[path != 0] - 1 for path in path_list]
    def matvec(h_flat):
        # h_flat: (n_paths * 2*n_t,)
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
        # x_flat: (n_arcs * n_t,)
        result = np.zeros(len(path_list) * 2 * n_t)
        slot = 0
        for path_idx, edgelist in enumerate(path_edgelists):
            acc = np.zeros(2 * n_t)
            # forward pass: store intermediates for adjoint
            prevs = [None] * (len(edgelist) + 1)
            # we don't need full intermediates — adjoint via reverse accumulation
            # v_in for slot k = D[e0]^T...D[e_{k-1}]^T applied to Tt*x_arc
            # use reverse: start from last edge, accumulate backwards
            # Adjoint derivation:
            # forward: r += T @ (prev_k - D[e_k] @ prev_k)
            #        = T @ (I - D[e_k]) @ prev_k
            # adjoint wrt h: sum_k (I - D[e_k])^T @ T^T @ x_{arc_k}
            #              propagated back through the chain
            # Pass 1: get arc indices and x_arc for each slot in this path
            arc_xs = []
            for local_k, edge in enumerate(edgelist):
                arc_r = slot_to_arc[slot + local_k]
                x_arc = x_flat[arc_r*n_t : (arc_r+1)*n_t]
                arc_xs.append(Tt.dot(x_arc))   # (2*n_t,) each
            # Pass 2: reverse accumulation
            # grad flows backward: after last edge, grad=0
            # at each step k (going backwards):
            #   grad_in  += arc_xs[k]           (from the (I) term)
            #   grad_out -= arc_xs[k]            (from the (-D[e_k]) term)
            #   grad propagates: grad = D[e_k]^T @ grad + grad_out_contribution
            back = np.zeros(2 * n_t)
            for local_k in range(len(edgelist) - 1, -1, -1):
                edge = edgelist[local_k]
                back = D[edge].T.dot(back)      # propagate gradient
                back -= arc_xs[local_k]         # out term: -Tt*x
                back += arc_xs[local_k]         # in term cancels... 
            # Hmm — let me be more careful. Clean re-derivation:
            # result += sum_k T^T x_{arc_k} applied to (h_in_k - h_out_k)
            # h_in_0  = h,  h_out_0 = D[e0] h
            # h_in_1  = h_out_0,  h_out_1 = D[e1] D[e0] h  ...
            # d(result)/dh = sum_k (I - D[e_k])^T C_k^T T^T x_{arc_k}
            # where C_k = D[e_{k-1}]...D[e_0]  (C_0 = I)
            # = sum_k C_k^T (I - D[e_k])^T Tt x_{arc_k}
            # Accumulate with one forward pass storing C_k^T implicitly:
            acc = np.zeros(2 * n_t)
            # We need C_k^T v = (D[e0]^T ... D[e_{k-1}]^T) v
            # Build this incrementally going forward
            chain_T_vecs = []
            for local_k, edge in enumerate(edgelist):
                v = arc_xs[local_k]                    # Tt x_{arc_k}: (2*n_t,)
                vv = v - D[edge].T.dot(v)              # (I - D[e_k])^T v
                # Now apply C_k^T = D[e0]^T...D[e_{k-1}]^T to vv
                for e in edgelist[:local_k]:
                    vv = D[e].T.dot(vv)
                acc += vv
            result[path_idx * 2*n_t : (path_idx+1) * 2*n_t] = acc
            slot += len(edgelist)
        return result
    n_paths_ext = len(path_list) * 2 * n_t
    n_out = n_arcs * n_t
    return LinearOperator((n_out, n_paths_ext), matvec=matvec, rmatvec=rmatvec)
