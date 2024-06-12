#**********************************************************************************************************************
#    Title: Ambivert degree identifies crucial brain functional hubs and improves detection of Alzheimer’s Disease
#           and Autism Spectrum Disorder (https://www.sciencedirect.com/science/article/pii/S2213158220300188)
#**********************************************************************************************************************

#! /usr/bin/env python

# We will need some things from several places
from __future__ import division, absolute_import, print_function
import sys
if sys.version_info < (3,):
    range = xrange
import os
import io
import scipy.io
import networkx as nx
import numpy as np
from pylab import *  # for plotting
from numpy.random import *  # for random sampling
import csv
from multiprocessing import Pool
# We need to import the graph_tool module itself
#from graph_tool.all import *
from sklearn import metrics
import math
import random as rm

def get_threshold(graph_data, percent_threshold):
	sorted_graph_data = np.sort(graph_data, axis=None)
	sorted_graph_data = np.array(sorted_graph_data[sorted_graph_data>0])
	#print (sorted_graph_data)
	boundary_element_index = math.floor(sorted_graph_data.size*(100-percent_threshold)/100)
	threshold = sorted_graph_data[int(boundary_element_index)]
	#print (sorted_graph_data.size, boundary_element_index, threshold)
	return threshold

def perform_svd(graph_data, percent_threshold, num_nodes):
	U, s, Vh = np.linalg.svd(graph_data, full_matrices=False)
	assert np.allclose(graph_data, np.dot(U, np.dot(np.diag(s), Vh)))
	sum_s = np.sum(s)
	max_sum = percent_threshold*sum_s/100
	index = num_nodes
	eigen_vals_sum = 0
	for i in range(0, num_nodes):
		if eigen_vals_sum >= max_sum:
			index = i
			break
		else:
			eigen_vals_sum +=  s[i]

	print ('variance s:', 'index:', index, sum_s, max_sum, eigen_vals_sum)
	s[index:] = 0
	new_data = np.dot(U, np.dot(np.diag(s), Vh))
	return new_data

def get_upper_lower(num_entries, percent):

	upper_boundary_element_index = math.floor(num_entries*(100-percent)/100)
	lower_boundary_element_index = math.ceil(num_entries*(percent)/100)

	return upper_boundary_element_index, lower_boundary_element_index

def participation_coefficient(partition_communities, vertices, subject_data): #for graph-tool data
	#calculate participation coefficient for node_1
	pc_dict = {}
	for v in vertices:
		node_degree =  np.sum(subject_data[v])
		node_edges =  subject_data[v]
		node_neighbors = vertices
		#print (v_index, node_degree, node_neighbors)
		if node_degree == 0:
			pc_dict[v] = 0.0
		else:
			pc = 0.0
			for partition in partition_communities.keys():
				community_degree = 0
				if len(partition_communities[partition]) > 1:
					for node_2 in partition_communities[partition]:
						if node_2 in node_neighbors:
							community_degree += 1
					pc = pc + ((float(community_degree)/float(node_degree))**2)
					
					#print ('node 1: ', node_1, 'community_degree: ', community_degree, 'pc: ', pc)
			pc = 1-pc
			pc_dict[v_index] = pc

	return pc_dict

def betweenness_wei(G):
	"""
	Node betweenness centrality is the fraction of all shortest paths in
	the network that contain a given node. Nodes with high values of
	betweenness centrality participate in a large number of shortest paths.
	Parameters
	----------
	L : NxN :obj:`numpy.ndarray`
	directed/undirected weighted connection matrix
	Returns
	-------
	BC : Nx1 :obj:`numpy.ndarray`
	node betweenness centrality vector
	Notes
	-----
	The input matrix must be a connection-length matrix, typically
	obtained via a mapping from weight to length. For instance, in a
	weighted correlation network higher correlations are more naturally
	interpreted as shorter distances and the input matrix should
	consequently be some inverse of the connectivity matrix.
	Betweenness centrality may be normalised to the range [0,1] as
	BC/[(N-1)(N-2)], where N is the number of nodes in the network.
	"""
	n = len(G)
	BC = np.zeros((n,))  # vertex betweenness

	for u in range(n):
		D = np.tile(np.inf, (n,))
		D[u] = 0  # distance from u
		NP = np.zeros((n,))
		NP[u] = 1  # number of paths from u
		S = np.ones((n,), dtype=bool)  # distance permanence
		P = np.zeros((n, n))  # predecessors
		Q = np.zeros((n,), dtype=int)  # indices
		q = n - 1  # order of non-increasing distance

		G1 = G.copy()
		V = [u]
		while True:
			S[V] = 0  # distance u->V is now permanent
			G1[:, V] = 0  # no in-edges as already shortest
			for v in V:
				Q[q] = v
				q -= 1
				W, = np.where(G1[v, :])  # neighbors of v
				for w in W:
					Duw = D[v] + G1[v, w]  # path length to be tested
					if Duw < D[w]:  # if new u->w shorter than old
						D[w] = Duw
						NP[w] = NP[v]  # NP(u->w) = NP of new path
						P[w, :] = 0
						P[w, v] = 1  # v is the only predecessor
					elif Duw == D[w]:  # if new u->w equal to old
						NP[w] += NP[v]  # NP(u->w) sum of old and new
						P[w, v] = 1  # v is also predecessor

			if D[S].size == 0:
				break  # all nodes were reached
			if np.isinf(np.min(D[S])):  # some nodes cannot be reached
				Q[:q + 1], = np.where(np.isinf(D))  # these are first in line
				break
			V, = np.where(D == np.min(D[S]))

		DP = np.zeros((n,))
		for w in Q[:n - 1]:
			BC[w] += DP[w]
			for v in np.where(P[w, :])[0]:
				DP[v] += (1 + DP[w]) * NP[v] / NP[w]

	return BC

def invert(W, copy=True):
	"""
	Inverts elementwise the weights in an input connection matrix.
	In other words, change the from the matrix of internode strengths to the
	matrix of internode distances.
	If copy is not set, this function will *modify W in place.*
	Parameters
	----------
	W : :obj:`numpy.ndarray`
	weighted connectivity matrix
	copy : bool
	if True, returns a copy of the matrix. Otherwise, modifies the matrix
	in place. Default value=True.
	Returns
	-------
	W : :obj:`numpy.ndarray`
	inverted connectivity matrix
	"""
	if copy:
		W = W.copy()
	
	E = np.where(W)
	W[E] = 1. / W[E]
	#W = 1 - W/np.max(W)
	return W

def gateway_coef_sign(W, node_communities, centrality_type='degree'):
	"""
	The gateway coefficient is a variant of participation coefficient.
	It is weighted by how critical the connections are to intermodular
	connectivity (e.g. if a node is the only connection between its
	module and another module, it will have a higher gateway coefficient,
	unlike participation coefficient).
	Parameters
	----------
	W : NxN :obj:`numpy.ndarray`
	undirected signed connection matrix
	node_communities : Nx1 :obj:`numpy.ndarray`
	community affiliation vector
	centrality_type : enum
	'degree' - uses the weighted degree (i.e, node strength)
	'betweenness' - uses the betweenness centrality
	Returns
	-------
	Gpos : Nx1 :obj:`numpy.ndarray`
	gateway coefficient for positive weights
	Gneg : Nx1 :obj:`numpy.ndarray`
	gateway coefficient for negative weights
	References
	----------
	.. [1] Vargas ER, Wahl LM, Eur Phys J B (2014) 87:1-10
	"""
	_, node_communities = np.unique(node_communities, return_inverse=True)
	node_communities += 1
	n = len(W)
	np.fill_diagonal(W, 0)

	def gcoef(W):
		# strength
		node_weights = np.sum(W, axis=1)
		# neighbor community affiliation
		Gc = np.inner((W != 0), np.diag(node_communities))
		#print (np.diag(node_communities), Gc.shape)
		# community specific neighbors
		Sc2 = np.zeros((n,))
		# extra modular weighting
		ksm = np.zeros((n,))
		# intra modular wieghting
		centm = np.zeros((n,))

		if centrality_type == 'degree':
			cent = node_weights.copy()
		elif centrality_type == 'betweenness':
			cent = betweenness_wei(invert(W))

		num_modules = int(np.max(node_communities))

		for module_1 in np.unique(node_communities):
			ks = np.sum(W * (Gc == module_1), axis=1)
			Sc2 += ks ** 2
			
			for module_2 in np.unique(node_communities):
				# calculate extramodular weights
				ksm[node_communities == module_2] += ks[node_communities == module_2] / np.sum(ks[node_communities == module_2])
				# calculate intramodular weights
				centm[node_communities == module_1] = np.sum(cent[node_communities == module_1])

		# print(Gc)
		# print(centm)
		# print(ksm)
		# print(ks)

		centm = centm / max(centm)
		# calculate total weights
		gs = (1 - ksm * centm) ** 2
		Gw = 1 - Sc2 * gs / node_weights ** 2
		Gw[np.where(np.isnan(Gw))] = 0
		Gw[np.where(np.logical_not(Gw))] = 0

		return Gw

	G_pos = gcoef(W * (W > 0))
	G_neg = gcoef(-W * (W < 0))
	return G_pos, G_neg

def participation_coef_sign(W, ci):
	"""
	Participation coefficient is a measure of diversity of intermodular
	connections of individual nodes.
	Parameters
	----------
	W : NxN :obj:`numpy.ndarray`
	undirected connection matrix with positive and negative weights
	ci : Nx1 :obj:`numpy.ndarray`
	community affiliation vector
	Returns
	-------
	Ppos : Nx1 :obj:`numpy.ndarray`
	participation coefficient from positive weights
	Pneg : Nx1 :obj:`numpy.ndarray`
	participation coefficient from negative weights
	"""
	_, ci = np.unique(ci, return_inverse=True)
	ci += 1

	n = len(W)  # number of vertices

	def pcoef(W_):
		S = np.sum(W_, axis=1)  # strength
		# neighbor community affil.
		Gc = np.dot(np.logical_not(W_ == 0), np.diag(ci))
		Sc2 = np.zeros((n,))

		for i in range(1, int(np.max(ci) + 1)):
			Sc2 += np.square(np.sum(W_ * (Gc == i), axis=1))

		P = np.ones((n,)) - Sc2 / np.square(S)
		P[np.where(np.isnan(P))] = 0
		P[np.where(np.logical_not(P))] = 0  # p_ind=0 if no (out)neighbors
		return P

	# explicitly ignore compiler warning for division by zero
	with np.errstate(invalid='ignore'):
		Ppos = pcoef(W * (W > 0))
		Pneg = pcoef(-W * (W < 0))

	return Ppos, Pneg

def module_degree_zscore(W, ci, std=0, flag=0):
	"""
	The within-module degree z-score is a within-module version of degree
	centrality.
	Parameters
	----------
	W : NxN :obj:`numpy.ndarray`
	binary/weighted directed/undirected connection matrix
	ci : Nx1 np.array_like
	community affiliation vector
	flag : int
	Graph type. 0: undirected graph (default)
	1: directed graph in degree
	2: directed graph out degree
	3: directed graph in and out degree
	Returns
	-------
	Z : Nx1 :obj:`numpy.ndarray`
	within-module degree Z-score
	"""
	_, ci = np.unique(ci, return_inverse=True)
	ci += 1

	if flag == 2:
		W = W.copy()
		W = W.T
	elif flag == 3:
		W = W.copy()
		W = W + W.T

	n = len(W)
	Z = np.zeros((n,))  # number of vertices
    
	if std == 1:
		for i in range(1, int(np.max(ci) + 1)):
			Koi = np.sum(W[np.ix_(ci == i, ci == i)], axis=1)
			Z[np.where(ci == i)] = Koi

	else:
		for i in range(1, int(np.max(ci) + 1)):
			Koi = np.sum(W[np.ix_(ci == i, ci == i)], axis=1)
			Z[np.where(ci == i)] = (Koi - np.mean(Koi)) / np.std(Koi)

	Z[np.where(np.isnan(Z))] = 0
	return Z

def percolation_analysis(graph):
	#open('percolation_analysis_' + HCP_set + '.csv', 'w').close()

	for percent_threshold in range(1, 100):
		threshold = get_threshold(graph, percent_threshold)
		graph_t = graph*(graph>threshold)
		sum_rows = np.sum(graph_t, axis=0)
		
		#with open('percolation_analysis_' + HCP_set + '.csv', 'a') as out_stream:
		#	out_stream.write(str(percent_threshold) + ', ' + str(np.count_nonzero(sum_rows)) + '\n')
		
		#print ('Percent:', percent_threshold, 'Size:', np.count_nonzero(sum_rows))

		if 0 not in sum_rows:
			print ('Found threshold:', percent_threshold)
			return percent_threshold

def percolate_network(network):
	percent_threshold = percolation_analysis(network)
	threshold = get_threshold(network, percent_threshold)
	graph = network*(network>threshold)
	np.fill_diagonal(graph, 0)
	return graph


"""
group_labels = np.array([5, 5, 0, 0, 1, 4, 1, 1, 9, 1, 9, 0, 13, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 7, 7, 7, 7, 7, 8, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 12, 12, 12, 12, 12, 8, 12, 12, 12, 12, 12, 12, 12, 12, 8, 12, 8, 8, 12, 8, 7, 12, 12, 8, 8, 12, 8, 13, 4, 4, 2, 11, 6, 13, 6, 6, 11, 11, 6, 4, 11, 4, 4, 13, 4, 4, 13, 13, 13, 9, 4, 11, 11, 11, 4, 6, 6, 11, 4, 6, 4, 3, 4, 4, 3, 11, 3, 11, 6, 4, 6, 6, 9, 11, 6, 3, 6, 13, 13, 13, 6, 6, 6, 9, 6, 6, 11, 3, 3, 13, 6, 6, 6, 5, 5, 5, 2, 10, 2, 2, 10, 2, 10, 10, 2, 2, 10, 10, 2, 2, 10, 5, 2, 2, 10, 2, 2, 10, 5, 2, 2, 5, 10, 2, 5, 10, 10, 0, 0, 0, 11, 13, 0, 9, 9, 1, 1, 11, 11, 0, 0, 0, 0, 9, 0, 9, 9, 9, 11, 9, 11, 11, 0, 9, 0, 0, 12, 9, 12, 1, 9, 0, 0, 3, 3, 3, 12, 3, 3, 3, 1, 9, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 8, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 6, 6, 13, 12, 0, 10, 7, 10, 10, 10, 0, 0, 0, 10, 13, 12])
module_labels = ['Attentional', 'Subcortical', 'Medial_Visual', 'Salience', 'Medial_DMN', 'Occipital_Visual', 'Language', 'Somatomotor_Hand', 'Motor_Auditory', 'Right_FP', 'Lateral_Visual', 'Left_FP', 'Cingulo_Opercular', 'Post_DMN']

HCP_set = 'R227'
directory = '/data/' + HCP_set
#directory = '/home/Codes/' + HCP_set
path = '/home/Codes/HCP_rs_power/'

subject_list_original = np.load(os.path.join(directory, 'Corre_subject_list.npy'), encoding = 'ASCII')
subject_modules = np.genfromtxt('realigned_subject_labels.csv', delimiter=',')

open('node_z_score.csv', 'wb').close()
open('node_participation_coeff.csv', 'wb').close()
open('node_gateway_degree.csv', 'wb').close()
open('node_gateway_betweenness.csv', 'wb').close()

S1_corr_data = np.load(os.path.join(directory, 'S1_all_subjects_corre.npy'))
S2_corr_data = np.load(os.path.join(directory, 'S2_all_subjects_corre.npy'))
S3_corr_data = np.load(os.path.join(directory, 'S3_all_subjects_corre.npy'))
S4_corr_data = np.load(os.path.join(directory, 'S4_all_subjects_corre.npy'))
"""
def single_run(params):
	subject_index, S1_corr_data, S2_corr_data, S3_corr_data, S4_corr_data, subject_modules = params

	percent = 6.7 #z score of 1.5 corresponds to 93.3% 
	connector_hub_threshold = 0.62 #Guimera and Amaral scheme

	num_nodes = S1_corr_data.shape[1];
	subject_data =  np.zeros((num_nodes, num_nodes))
	subject_corr_data = [S1_corr_data, S2_corr_data, S3_corr_data, S4_corr_data]
	num_sessions = len(subject_corr_data)

	#counts the frequency of a node being classified as a hub, last index stores the count (subject info)
	connector_hubs = np.zeros(num_nodes + 1);
	provincial_hubs = np.zeros(num_nodes + 1);

	for index, session_data in enumerate(subject_corr_data):
		graph = percolate_network(session_data)
		subject_data += graph/num_sessions

	subject_data = percolate_network(subject_data)
	subject_distance = 1 - subject_data/np.max(subject_data)
	
	node_module_z_score = module_degree_zscore(subject_data, subject_modules, flag=0) 
	node_participation_coeff, _ = participation_coef_sign(subject_data, subject_modules)
	node_gateway_degree, _  = gateway_coef_sign(subject_data, subject_modules, centrality_type='degree')
	node_gateway_betweenness, _  = gateway_coef_sign(subject_data, subject_modules, centrality_type='betweenness')

	with open('node_z_score.csv', 'ab') as out_stream:
		np.savetxt(out_stream, [np.append(np.asarray([subject_index]), node_module_z_score)],  delimiter=",");
	with open('node_participation_coeff.csv', 'ab') as out_stream:
		np.savetxt(out_stream, [np.append(np.asarray([subject_index]), node_participation_coeff)],  delimiter=",");
	with open('node_gateway_degree.csv', 'ab') as out_stream:
		np.savetxt(out_stream, [np.append(np.asarray([subject_index]), node_gateway_degree)],  delimiter=",");
	with open('node_gateway_betweenness.csv', 'ab') as out_stream:
		np.savetxt(out_stream, [np.append(np.asarray([subject_index]), node_gateway_betweenness)],  delimiter=",");

	return subject_index

	vertices = range(0, num_nodes)

	g_graph_nx = nx.Graph()
	g_graph_nx.add_nodes_from(range(0, num_nodes))
	
	for v1 in vertices:
		for v2 in vertices:
			if(subject_data[v1][v2] > 0 and v2 <= v1):
				g_graph_nx.add_edge(v1, v2, weight=subject_data[v1][v2])

	best = states[index]
	num_blocks = np.unique(subject_modules).size

	block_degree = [[] for i in range(num_blocks)]

	# key is community name and value is list of nodes in it.
	community_nodes = dict() 

	intra_modular_node_degree = np.zeros(num_nodes);

	for v in vertices:
		vertex_status[v] = 0
		v_block = cluster_labels[v]

		if v_block in community_nodes.keys():
			community_nodes[v_block].append(v)
		else:
			community_nodes[v_block] = [v]

	for v in vertices:
		v_block = cluster_labels[v]

		for neighbor in vertices:
			if neighbor in community_nodes[v_block]:
				intra_modular_node_degree[v] += subject_data[v][neighbor]

		block_degree[v_block].append(intra_modular_node_degree[v]);
			
	mean_block, std_block, percent_conn_hubs, path_length_block, path_length_random_block = np.zeros(num_blocks), np.zeros(num_blocks), np.zeros(num_blocks), np.zeros(num_blocks), np.zeros(num_blocks)
	connector_hub_node_list = []
	provincial_hub_node_list = []

	participation_coefficient = participation_coefficient(community_nodes, g)

	'''
	for v in vertices:
		v_index = g.vertex_index[v]
		if v_index in sort_indices and participation_coefficient[v_index] > connector_hub_threshold: 
			connector_hub_node_list.append(v_index)
	'''

	for block in range(num_blocks):
		block_degree[block] =  np.asarray(block_degree[block])
		mean_block[block] = np.mean(block_degree[block])
		std_block[block] = np.std(block_degree[block])
		
		sort_indices = np.asarray(block_degree[block]).argsort()
		block_degree[block] = np.sort(block_degree[block])

		community_nodes[block] =  np.asarray(community_nodes[block])[sort_indices]

		size = community_nodes[block].size
		num_connector_hubs_block = 0
		upper, lower = get_upper_lower(size, percent)
		num_hubs_block = size - upper
		edges_weights_random = []

		g_nx = nx.Graph()
		g_nx.add_nodes_from(community_nodes[block])
		

		#forming a network of only intra-modular edges
		#print( 'community_nodes[block]', community_nodes[block])

		for node in community_nodes[block]:
			for neighbor in vertices:
				if neighbor in community_nodes[block]:
					g_nx.add_edge(node, neighbor, weight = subject_data[node][neighbor])
					edges_weights_random.append(subject_data[node][neighbor])

		#forming a random network of only intra-modular edges
		#g_nx_random = nx.gnm_random_graph(size, len(edges_weights_random)) 
		
		g_nx_random = nx.Graph()
		while len(edges_weights_random) > 0:
			v1 = rm.randint(0, size)
			v2_range = list(range(0, v1)) + list(range(v1+1, 263))
			v2 = rm.choice(v2_range)
			#print (v1, v2, weight)
			if (v1,v2) in g_nx_random.edges() or (v2, v1) in g_nx_random.edges():
				continue
			else:
				weight = edges_weights_random.pop()
				g_nx_random.add_edge(v1, v2, weight = subject_data[v1][v2])
		
		if nx.is_connected(g_nx):
			path_length_block[block] = nx.average_shortest_path_length(g_nx)
		
		if nx.is_connected(g_nx_random):
			path_length_random_block[block] = nx.average_shortest_path_length(g_nx_random)

		for i in range(upper, size):
			print ('Hubs: ', i, 'Node: ', community_nodes[block][i], 'Participation Coefficient: ', participation_coefficient[community_nodes[block][i]])
			if participation_coefficient[community_nodes[block][i]] > connector_hub_threshold: #a connector hub
				vertex_status[community_nodes[block][i]] = 2
				connector_hub_node_list.append(community_nodes[block][i])
				connector_hubs[community_nodes[block][i]] = 1
				num_connector_hubs_block += 1
			else: #provincial hub
				vertex_status[community_nodes[block][i]] = 1
				provincial_hubs[community_nodes[block][i]] = 1

		percent_conn_hubs[block] = num_connector_hubs_block/num_hubs_block


	num_provincial_hubs = np.sum(provincial_hubs)
	num_connector_hubs = np.sum(connector_hubs)


	#print (dc_inds, std_block_dc, std_block)

	#print (std_block_dc, std_block)

	names = np.array(names) 
	inds = names.argsort()
	names = np.sort(names)	
	mean_block = mean_block[inds]
	matches = matches[inds]
	std_block = std_block[inds]

	high_degree_nodes = [] #calculate nodes which are degree hubs but have low participation coefficient.


 	#max number of high degree nodes which are not connector hubs to be removed = 26 (10% of nodes)
	max_hubs_to_remove = 26
	c = 0

	for node_index in sort_indices:
 		if node_index not in connector_hub_node_list and c <= max_hubs_to_remove:
 			#print ('*******************************add node *************************', high_degree_nodes)
 			high_degree_nodes.append(node_index)
 			c +=1
	#print (names_dc, matches_dc, mean_block_dc, std_block_dc, names, matches, mean_block, std_block)

	num_iter = 1
	avg_path_length_hub_removal, avg_path_length_random_removal = np.zeros(12), np.zeros(12)

	#remove x percent of degree hubs which are not connector hubs
	

	for i, percent_hub_removal in  enumerate(range(10, 110, 10)):
		num_of_hubs_remove = math.floor(percent_hub_removal*max_hubs_to_remove/100)
		sum_path_length = 0
		sum_random_path_length = 0

		for iter in range(0, num_iter):
			rm.shuffle(high_degree_nodes)
			edit_g_nx, edit_g_random_nx =  g_graph_nx.copy(), g_graph_nx.copy()
			#print ('remove_nodes_from:', high_degree_nodes[0:num_of_hubs_remove])
			edit_g_nx.remove_nodes_from(high_degree_nodes[0:num_of_hubs_remove])
			#print ('final nodes:', edit_g_nx.nodes())
			random_node_range = list(range(0, 264))
			rm.shuffle(random_node_range)
			edit_g_random_nx.remove_nodes_from(random_node_range[0:num_of_hubs_remove])

			if nx.is_connected(edit_g_nx):
				path_length = nx.average_shortest_path_length(edit_g_nx)
				sum_path_length += path_length
			else:
				sum_path_length = 0
				break;
			'''
			if nx.is_connected(edit_g_random_nx):
				path_length = nx.average_shortest_path_length(edit_g_random_nx)
				sum_random_path_length += path_length
			else:
				sum_random_path_length = 0
				break;
			'''

		avg_path_length_hub_removal[i] = sum_path_length/num_iter
		avg_path_length_random_removal[i] = sum_random_path_length/num_iter

	avg_path_length_hub_replacement = np.zeros(12)
	
	'''
	#for x percent of nodes replace all edges by connecting to random nodes such that each node has average degree of nodes 	
	for i, percent_hub_replacement in  enumerate(range(10, 110, 10)):
		#print (percent_hub_replacement)
		num_of_hubs_replace = math.floor(percent_hub_replacement*num_connector_hubs/100)
		sum_path_length = 0	
		for iter in range(0, num_iter):
			rm.shuffle(connector_hub_node_list)
			edit_g_nx =  g_graph_nx.copy()
			average_degree = math.ceil(edit_g_nx.number_of_edges()/num_nodes)
			
			for node in connector_hub_node_list[0:num_of_hubs_replace]:
				block = cluster_labels[node]
				present_edges = edit_g_nx.edges(node)
				#new_node_neighbors = community_nodes[block]

				#print ('Node being replaced:', node, 'Edges being removed:', present_edges, 'Number of present edges:', num_edges_initial, 'Number of nodes in same module:', len(new_node_neighbors))

				edit_g_nx.remove_edges_from(present_edges)
				
				num_edges_used = 0

				while num_edges_used <= average_degree:
					v2_range = list(range(0, node)) + list(range(node+1, 263))
					v2 = rm.choice(v2_range)

					#print (v1, v2, weight)
					if (node,v2) in edit_g_nx.edges() or (v2, node) in edit_g_nx.edges():
						continue
					else:
						edit_g_nx.add_edge(node, v2)
						num_edges_used += 1

			if nx.is_connected(edit_g_nx):
				path_length = nx.average_shortest_path_length(edit_g_nx)
				sum_path_length += path_length
			else:
				sum_path_length = 0
				break;

		avg_path_length_hub_replacement[i] = sum_path_length/num_iter
	'''
	
	avg_path_length_hub_rewire = np.zeros(12)
	
	'''
	#rewire x percent of edges
	for i, percent_edge_rewire in  enumerate(range(10, 110, 10)):
		#print (percent_hub_replacement)

		sum_path_length = 0	
		for iter in range(0, num_iter):
			edit_g_nx =  g_graph_nx.copy()
			
			for node in connector_hub_node_list:
				block = cluster_labels[node]
				present_edges = edit_g_nx.edges(node)
				rm.shuffle(present_edges)
				new_node_neighbors = community_nodes[block]

				#print ('Node being replaced:', node, 'Edges being removed:', present_edges, 'Number of present edges:', num_edges_initial, 'Number of nodes in same module:', len(new_node_neighbors))
				num_of_edges_replace = math.floor(percent_edge_rewire*len(present_edges)/100)

				edit_g_nx.remove_edges_from(present_edges[0:num_of_edges_replace])
				
				num_edges_used = 0
				
				
			
				if len(new_node_neighbors) > num_of_edges_replace:
					while num_edges_used <= num_of_edges_replace:
						v2 = rm.choice(new_node_neighbors)

						#print (v1, v2, weight)
						#if (node, v2) in edit_g_nx.edges() or (v2, node) in edit_g_nx.edges():
						#	continue
						#else:
						edit_g_nx.add_edge(node, v2)
						num_edges_used += 1

				elif len(new_node_neighbors) <= num_of_edges_replace:
					for v2 in new_node_neighbors:
						if (node, v2) in edit_g_nx.edges() or (v2, node) in edit_g_nx.edges():
							continue
						else:
							edit_g_nx.add_edge(node, v2)
					

			if nx.is_connected(edit_g_nx):
				path_length = nx.average_shortest_path_length(edit_g_nx)
				sum_path_length += path_length
			else:
				sum_path_length = 0
				break;

		avg_path_length_hub_rewire[i] = sum_path_length/num_iter
	'''
	avg_path_length_hub_removal[10], avg_path_length_hub_replacement[10], avg_path_length_hub_rewire[10] = count, count, count
	
	if nx.is_connected(g_graph_nx):
		network_path_length = nx.average_shortest_path_length(g_graph_nx)

	g_random_nx = nx.gnm_random_graph(num_nodes, g_graph_nx.number_of_edges())
	if nx.is_connected(g_random_nx):
		random_path_length = nx.average_shortest_path_length(g_random_nx)

	avg_path_length_hub_removal[11], avg_path_length_hub_replacement[11], avg_path_length_hub_rewire[11] = network_path_length, network_path_length, network_path_length

	print('########################################### Done with this subject #######################################################')
	
	print('Subject: ', count, 'Threshold Percentage: ', percent_threshold, 'Threshold Value: ', threshold, 'LL NDC: ', -best.entropy(), 'Number Connector NDC: ', num_connector_hubs, 'Removal Path Length', avg_path_length_hub_removal, 'Random Removal Path Length', avg_path_length_random_removal, 'Replacement Path Length', avg_path_length_hub_replacement, 'Rewire Path Length', avg_path_length_hub_rewire, 'Original Path Length', network_path_length, 'Random Path Length', random_path_length)
	
	'''
	
	with open('aggregate_info.csv', 'a') as f: 
		f.write( str(count) + ',' + str(percent_threshold) + ',' + str(threshold) + ',' + str(num_blocks) + ',' + str(-best.entropy()) + ', ' +  str(num_connector_hubs/(num_provincial_hubs + num_connector_hubs)) +  ', ' +  str(network_path_length) + ', ' +  str(random_path_length) + '\n');
	
	with open('hubs_removal.csv','ab') as f_hubs_removal:
		np.savetxt(f_hubs_removal, [avg_path_length_hub_removal],  delimiter=",")

	with open('random_removal.csv','ab') as f_random_removal:
		np.savetxt(f_random_removal, [avg_path_length_random_removal],  delimiter=",")

	with open('hubs_replacement.csv','ab') as f_hubs_replacement:
		np.savetxt(f_hubs_replacement, [avg_path_length_hub_replacement],  delimiter=",")

	with open('hubs_rewire.csv','ab') as f_hubs_rewire:
		np.savetxt(f_hubs_rewire, [avg_path_length_hub_rewire],  delimiter=",")
	'''
	return subject_index


if __name__ == '__main__':
	p = Pool(40)
	params = []
	for subject_index, subject in enumerate(subject_list_original):
		params.append(tuple([subject_index, S1_corr_data[subject_index], S2_corr_data[subject_index], S3_corr_data[subject_index], S4_corr_data[subject_index], subject_modules[subject_index]]))
	for val in p.map(single_run, params):
		print ('Completed subject: ', val)
