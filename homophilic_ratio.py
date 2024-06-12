import numpy as np

from scipy.spatial.distance import cdist


#**************************************************************************************************************************************
#    Title: Assortative mixing in micro-architecturally annotated brain connectomes (https://doi.org/10.1038/s41467-023-38585-4)
#    Author: Vincent Bazinet, Justine Y. Hansen, Reinder Vos de Wael, Boris C. Bernhardt, Martijn P. van den Heuvel & Bratislav Misic
#    Availability: https://github.com/netneurolab/bazinet_assortativity
#**************************************************************************************************************************************
def edge_diff(distance_metric, M, N=None):
    '''
    Function to compute the absolute differences in the annotations of
    individual edges in a network.

    Parameters:
    ----------
    M: (n,) ndarray
        Vector of annotation scores
    N: (n,) ndarray
        Vector of annotation scores

    Returns:
    ----------
    diff: (n, n) ndarray
        Matrix of pairwise absolute differences between the annotation scores
        of pair of nodes.
    '''
    n_nodes = len(M)
    M_in = np.repeat(M[:, np.newaxis], n_nodes, axis=1)
    if N is None:
        M_out = M_in.T
    else:
        M_out = np.repeat(N[np.newaxis, :], n_nodes, axis=0)

    #diff = np.abs(M_in - M_out)
    diff = cdist(M_in, M_out, metric=distance_metric)
        
    return diff


def compute_homophilic_ratio(nSubj, nROI, adj_matrix, mask, distance_metric, fpath):
    results = []
    
    for i in range(nSubj):
        temp = []
        for j in range(nROI):
            M_diff = edge_diff(distance_metric, mask[i,j,:])
            nodal_mean_diff = np.abs(np.average(M_diff, weights=adj_matrix[i,:,:], axis=0))
            temp.append(np.mean(nodal_mean_diff / np.mean(M_diff, axis=0)))
        results.append(temp)
        
    np.save(fpath + '/homophilic_ratio_all_subjects.npy', np.array(results))