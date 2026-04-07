import numpy as np
import pandas as pd


#**************************************************************************************************************************************
#    Title: Assortative mixing in micro-architecturally annotated brain connectomes (https://doi.org/10.1038/s41467-023-38585-4)
#    Author: Vincent Bazinet, Justine Y. Hansen, Reinder Vos de Wael, Boris C. Bernhardt, Martijn P. van den Heuvel & Bratislav Misic
#    Availability: https://github.com/netneurolab/bazinet_assortativity
#**************************************************************************************************************************************
def weighted_assort(A, M, N=None, directed=False, normalize=True):
    '''
    Function to compute the weighted Pearson correlation between the attributes
    of the nodes connected by edges in a network (i.e. weighted assortativity).
    This function also works for binary networks.

    Parameters
    ----------
    A : (n,n) ndarray
        Adjacency matrix of our network.
    M : (n,) ndarray
        Vector of nodal attributes.
    N : (n,) ndarray
        Second vector of nodal attributes (optional)
    directed: bool
        Whether the network is directed or not. When the network is not
        directed, setting this parameter to False will increase the speed of
        the computations.
    normalize: bool
        If False, the adjacency weights won't be normalized to make its weights
        sum to 1. This should only be set to False if the matrix has been
        normalized already. Otherwise, the result will not be the assortativity
        coefficent. This is useful when we want to compute the assortativity
        of thousands annotations in a row. In that case, not having to
        normalize the adjacency matrix each time makes the function much
        faster.

    Returns
    -------
    ga : float
        Weighted assortativity of our network, with respect to the vector
        of attributes
    '''

    if (directed) and (N is None):
        N = M

    # Normalize the adjacency matrix to make weights sum to 1
    if normalize:
        A = A / A.sum(axis=None)

    # zscores of in-annotations
    k_in = A.sum(axis=0)
    mean_in = np.sum(k_in * M)
    sd_in = np.sqrt(np.sum(k_in * ((M-mean_in)**2)))
    z_in = (M - mean_in) / sd_in

    # zscores of out-annotations (if directed or N is not None)
    if N is not None:
        k_out = A.sum(axis=1)
        mean_out = np.sum(k_out * N)
        sd_out = np.sqrt(np.sum(k_out * ((N-mean_out)**2)))
        z_out = (N - mean_out) / sd_out
    else:
        z_out = z_in

    # Compute the weighted assortativity as a sum of z-scores
    ga = (z_in[np.newaxis, :] * z_out[:, np.newaxis] * A).sum()

    return ga


def hub_assortativity_coeff(mask, adj_mat, fpath, hub_score, flag=0, netw_name = None):
    hac_list = []
    for i in range(len(mask)):
        temp = mask[i]
        orig_data = np.sum(temp, axis=0)
                
        """
        subject_id = all_subject_ids[i]
        subject_id = site_name + '_' + subject_id[0]
        
        matching_rows = amb[amb[0] == subject_id]
        tlist1 = matching_rows.values.tolist()
        amb_score = np.array(tlist1[0][1:])

        if hub_score is None:
            hac = weighted_assort(adj_mat[i,:,:], orig_data, amb_score)
            hac_list.append(hac)
        else:
            hac = weighted_assort(adj_mat[i,:,:], orig_data, hub_score[i,:])
            hac_list.append(hac)
        """
        
        hac = weighted_assort(adj_mat[i,:,:], orig_data, hub_score[i,:])
        hac_list.append(hac)
        
    hac_array = np.array(hac_list)
    hac_array = np.nan_to_num(hac_array, nan=0)
    
    if flag==0:
        np.save(fpath + '/hub_assortativity_coeff_all_subjects.npy', hac_array)
    else:
        np.save(fpath + '/' + netw_name + '_hub_assortativity_coeff_all_subjects.npy', hac_array)
