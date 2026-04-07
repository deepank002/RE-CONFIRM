import numpy as np
from sklearn.impute import KNNImputer

def st_fidelity_plus(model, dfc_matrix, dfc_mask, k = 20, t_prime = 60, stride = 1, n_neighbors = 5):
    
    nSubj, nROI, _, T = dfc_matrix.shape
    
    sf_plus_scores = []

    for j in range(nSubj):
        X_j = dfc_matrix[j].copy()           
        S_j = dfc_mask[j].copy() 

        X_plus_j = X_j.copy()

        windows = range(0, T - t_prime + 1, stride)

        for t_start in windows:
            t_end = t_start + t_prime

            S_window = S_j[:, :, t_start:t_end]        
            S_agg = S_window.mean(axis=2)
            node_scores = S_agg.sum(axis=1)

            top_k_nodes = np.argsort(node_scores)[::-1][:k]

            for node in top_k_nodes:
                X_plus_j[node, :, t_start:t_end] = np.nan
                X_plus_j[:, node, t_start:t_end] = np.nan

        X_plus_flat = X_plus_j.reshape(nROI * nROI, T)
        imputer = KNNImputer(n_neighbors=n_neighbors)
        X_plus_imputed_flat = imputer.fit_transform(X_plus_flat)
        X_plus_j_imputed = X_plus_imputed_flat.reshape(nROI, nROI, T)

        pred_original = model(X_j[np.newaxis])
        pred_ablated  = model(X_plus_j_imputed[np.newaxis])

        sf_j = float(pred_original) - float(pred_ablated)
        sf_plus_scores.append(sf_j)

    sf_plus = float(np.mean(sf_plus_scores))
    return sf_plus
