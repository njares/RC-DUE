import os

from RC_DUE_helpers import *

# Nombre de la red
network_name = "Braess"

# constantes
full_path = './'
flows_filename = os.path.join(full_path,network_name,"flows.csv")
paths_filename = os.path.join(full_path,network_name,"paths.csv")
edge_flows_filename = os.path.join(full_path,network_name,"edge_flows.csv")
edge_times_filename = os.path.join(full_path,network_name,"traversal_time.csv")

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


# cosas necesarias
h_0 = flows
path_list = paths
x_0 = edge_flows
arc_delay_paper = edge_delay
arc_delay_0 = calcula_arc_delay(x_0)
n_t = h_0.shape[1]
n_arcs = arc_delay_0.shape[0]

taus_old = np.tile(np.arange(n_t),(n_arcs,1)) + arc_delay_paper
c_old = calcula_A_c_old(path_list, taus_old)

B = np.array([[1,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,1,1,0,0],[0,0,0,0,0,0,1,1]])
P_full = np.eye(8)-B.T@np.linalg.inv(B@B.T)@B
P = sparse.bmat([[P_full[i,j]*sparse.eye(n_t) for j in range(P_full.shape[1])] for i in range(P_full.shape[0])])

d_full = B @ h_0
d0_full = np.zeros(h_0.shape)
d0_full[0,:] = d_full[0,:]
d0_full[2,:] = d_full[1,:]
d0_full[3,:] = d_full[2,:]
d0_full[6,:] = d_full[3,:]
d0 = d0_full.flatten() # tamaño de h, satisface Bh=d

pHplus = lambda x : np.maximum(x, 0)
pB0 = lambda x : P@x
pBd = lambda x : d0 + pB0(x-d0)
P_lambda_flat = lambda x : project([pBd, pHplus], x)

def P_lambda(h):
	h_flat = h.flatten()
	h_proy_flat = P_lambda_flat(h_flat)
	return h_proy_flat.reshape(-1, 100)

trapezoid_integration = calcula_trapezoid_integration(n_t)

A = lambda h , arc_delay : A_delay(h, arc_delay, trapezoid_integration, path_list)

for t in range(1,n_t):
	arc_delay_paper[:,t] = np.maximum(arc_delay_paper[:,t-1]-.99, arc_delay_paper[:,t])

# calcular equilibrio
#h_next, arc_delay_next, status = rc_due(h_0, arc_delay_0, P_lambda, A, epsilon = 1e-5)
h_next, arc_delay_next, status = rc_due(h_0, arc_delay_paper, P_lambda, A, epsilon = 1e-5)
print(status)

# calcular flujos por arco finales
taus = np.tile(np.arange(n_t),(n_arcs,1)) + arc_delay_next
D = calcula_D(taus)
D_arc_path_sparse = calcula_D_arc_path(path_list, D)
af_matrix = calcula_af_matrix(D_arc_path_sparse, trapezoid_integration, path_list, n_t)
x_final = arc_flows_matrix(h_next, af_matrix)

c_final, _ = A_delay(h_next, arc_delay_next, trapezoid_integration, path_list)

# guardar costo final
np.savetxt("route_traversal_time_RC_DUE.csv", c_final, delimiter = ",")
np.savetxt("traversal_time_RC_DUE.csv", arc_delay_next, delimiter = ",")
np.savetxt("flows_RC_DUE.csv", h_next, delimiter = ",")
np.savetxt("edge_flows_RC_DUE.csv", x_final, delimiter = ",")

# graficar cosas
plot_final(h_next, h_0, x_0, x_final, c_final, c_old)


