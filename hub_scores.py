import numpy as np

from utils.subject_module_hub_removal_analysis import *


def generate_hub_scores(dataset, directory, nSubj, nROI, corr_matrix, subj_modules):
    nb = np.zeros((nSubj, nROI))
    nmz = np.zeros((nSubj, nROI))
    na = np.zeros((nSubj, nROI))
    npc = np.zeros((nSubj, nROI))
    ngd = np.zeros((nSubj, nROI))
    ngb = np.zeros((nSubj, nROI))
    
    for i in range(nSubj):
        percnet = percolate_network(corr_matrix[i,:,:])
        
        node_betweenness = betweenness_wei(percnet)
        nb[i,:] = node_betweenness
        
        node_module_z_score = module_degree_zscore(percnet, subj_modules, std=1, flag=0)
        nmz[i,:] = node_module_z_score
        node_module_z_score_nostd = module_degree_zscore(percnet, subj_modules, std=0, flag=0)
        avg_weighted_connection = np.sum(percnet) / np.count_nonzero(percnet > 0)
        node_ambivert = node_module_z_score_nostd * avg_weighted_connection
        na[i,:] = node_ambivert
        
        node_participation_coeff, _ = participation_coef_sign(percnet, subj_modules)
        npc[i,:] = node_participation_coeff
        node_gateway_degree, _  = gateway_coef_sign(percnet, subj_modules, centrality_type='degree')
        ngd[i,:] = node_gateway_degree
        node_gateway_betweenness, _  = gateway_coef_sign(percnet, subj_modules, centrality_type='betweenness')
        ngb[i,:] = node_gateway_betweenness

    np.save('./assets/' + dataset + '_' + directory[1:] + '/node_betweenness.npy', nb)
    np.save('./assets/' + dataset + '_' + directory[1:] + '/node_module_z_score.npy', nmz)
    np.save('./assets/' + dataset + '_' + directory[1:] + '/node_ambivert_degree.npy', na)
    np.save('./assets/' + dataset + '_' + directory[1:] + '/node_participation_coeff.npy', npc)
    np.save('./assets/' + dataset + '_' + directory[1:] + '/node_gateway_degree.npy', ngd)
    np.save('./assets/' + dataset + '_' + directory[1:] + '/node_gateway_betweenness.npy', ngb)
    