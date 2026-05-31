Para cada red, hay que correr en la carpeta del repo DTA de Han, K, Eve, G, Friesz, TL, 2019, en cada carpeta de cada grafo:

# octave
load <Network_name><n_paths>_pp.mat
dlmwrite ("paths.csv", pathList, ",")
dlmwrite ("edges_data.csv", linkData(:,[1,2,3,5]), ",")
clear
load DUE_out.mat
dlmwrite ("flows.csv", h_final, ",")
clear

# bash
octave get_edge_flows_and_times.m

Esto debería generar 5 archivos csv:

paths.csv
edges_data.csv
flows.csv
edge_flows.csv
traversal_time.csv

Esos 5 archivos se tienen que copiar en la carpeta con el mismo nombre del grafo, del repo RC-DUE
