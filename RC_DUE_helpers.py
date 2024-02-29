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


def arc_flows_matrix(h, af_matrix, adjoint = False):
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


def calcula_af_matrix(D_arc_path_sparse, trapezoid_integration, path_list, n_t):
	'''
	path_list es una matriz que relaciona rutas con arcos
	D_arc_path_sparse es una matriz rala que calcula el delay para todas las rutas y arcos
	'''
	n_arc_path = np.sum(path_list != 0)
	n_arcs = np.unique(path_list.flatten()).shape[0] - 1
	n_AR_flow = n_arc_path*n_t*2
	# matriz de flujo neto por arco-ruta
	AR_flow_matrix = sparse.hstack([sparse.eye(n_AR_flow),-sparse.eye(n_AR_flow)])
	# matriz de integracion trapecio cumulative
	#cum_trap_matrix = np.triu(np.ones((n_t,n_t))*90) + np.triu(np.ones((n_t,n_t))*90, k=1)
	#cum_trap_matrix[0,:] = cum_trap_matrix[0,:] - 90
	#cum_trap_sparse = sparse.csr_matrix(cum_trap_matrix).transpose(copy=True)
	#cum_trap_full = sparse.block_diag([cum_trap_sparse for _ in range(n_arc_path)])
	cum_trap_full = sparse.block_diag([trapezoid_integration.T for _ in range(n_arc_path)])
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
			cur_D_out_list.append(D[edge].transpose(copy=True).dot(cur_D_in_list[-1]))
			# AR_id += 1
		cur_D_in_sparse = sparse.vstack(cur_D_in_list)
		cur_D_out_sparse = sparse.vstack(cur_D_out_list)
		D_arc_path_in_list.append(cur_D_in_sparse)
		D_arc_path_out_list.append(cur_D_out_sparse)
	D_arc_path_in_sparse = sparse.block_diag(D_arc_path_in_list)
	D_arc_path_out_sparse = sparse.block_diag(D_arc_path_out_list)
	D_arc_path_sparse = sparse.vstack([D_arc_path_in_sparse, D_arc_path_out_sparse])
	return D_arc_path_sparse

def calcula_D(taus):
	n_arcs = taus.shape[0]
	n_t = taus.shape[1]
	# extender taus al doble del tamaño
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
	# chequear que tau sea monótono creciente
	if not np.all(tau[:-1] < tau[1:]):
		print("tau no es estrictamente creciente!")
		arc_delay = tau - np.arange(n_t)
		from scipy.interpolate import UnivariateSpline
		arc_delay_diff = (UnivariateSpline(np.arange(n_t), arc_delay, s=1e-6).derivative())(np.arange(n_t))
		plt.plot(arc_delay)
		plt.plot(arc_delay_diff)
		plt.show()
		#sys.exit()
		import pdb
		pdb.set_trace()
	# calcular la composición con flow
	for t in range(n_t):
		# busco t_1 a donde tau(t_1)=t 
		# primero busco el último t_0 tal que tau[t_0] <= t
		t_0 = np.min(np.where(tau>t)[0])-1
		# si me paso del primero, no hago nada
		if t_0 >= 0:
			# nunca me debería pasar del último
			if t_0 == n_t-1 and tau[t_0]<t:
				print("tau no abarca todo el horizonte temporal!")
				sys.exit()
			# puede ser real y no entero, busco un c tal que:
			# c*tau[t_0]+(1-c)*tau[t_0+1] = t
			# c*tau[t_0]+tau[t_0+1]-c*tau[t_0+1] = t
			# c*(tau[t_0]-tau[t_0+1])+tau[t_0+1] = t
			# c*(tau[t_0]-tau[t_0+1]) = t - tau[t_0+1]
			# c = (t - tau[t_0+1])/(tau[t_0]-tau[t_0+1])
			# está bien definido porque tau[t_0] != tau[t_0+1]
			c = (t - tau[t_0+1])/(tau[t_0]-tau[t_0+1])
			# entonces el t_0 tal que tau(t_0)=t es t_0 = t_0 + (1-c)
			t_1 = t_0 + (1-c)
			# ahora new_flow[t] = flow[t_1], así que tengo que calcular flow[t_1]
			# es la misma combinación convexa
			# c*flow[t_0]+(1-c)*flow[t_0+1]
			# no debería haber out_of_bounds porque tau y flow tienen el mismo tamaño
			new_flow[t] = c*flow[t_0] + (1-c)*flow[t_0+1]
	return new_flow

def calcula_arc_delay(x):
	# por ahora la dejamos fija, pero esta función debería depende de cada grafo
	arc_delay = 2*(1+0.15*(x/100)**4)
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

def A_delay(h, arc_delay, trapezoid_integration, path_list):
	n_t = h.shape[1]
	n_arcs = arc_delay.shape[0]
	# calcular matriz de flujo por arco a partir de los delays por arco
	taus = np.tile(np.arange(n_t),(n_arcs,1)) + arc_delay
	D = calcula_D(taus)
	D_arc_path_sparse = calcula_D_arc_path(path_list, D)
	af_matrix = calcula_af_matrix(D_arc_path_sparse, trapezoid_integration, path_list, n_t)
	# calcular flujos por arco a partir de los flujos por ruta y los delays por arco
	x_next = arc_flows_matrix(h, af_matrix)
	# calcular delays por arco a partir de los nuevos flujos por arco
	arc_delay_next = calcula_arc_delay(x_next)
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
	for j in range(h_inicial.shape[0]):
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
	for j in range(x_inicial.shape[0]):
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
	#import pdb
	#pdb.set_trace()
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
