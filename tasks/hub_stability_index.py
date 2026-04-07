import numpy as np

def hub_stability_index(dfc_matrix, amb_score, epsilon = 1e-8, threshold = 1.0):
    nSubj, nROI, _, T = dfc_matrix.shape

    P_H_all = np.zeros((nSubj, T))

    for n in range(nSubj):
        is_hub = amb_score[n] > threshold  

        for t in range(T):
            H_t = set(np.where(is_hub)[0])   
            P_H_all[n, t] = len(H_t) / nROI     

    P_H = P_H_all.mean(axis=0)

    var_P_H = np.var(P_H)
    max_P_H = np.max(P_H)

    sigma2_norm = var_P_H / (max_P_H ** 2 + epsilon)

    HSI = 1.0 - sigma2_norm

    return float(HSI)


def combined_hsi(dfc_matrix, dfc_mask, amb_score_fc, amb_score_attr, epsilon = 1e-8, threshold = 1.0):
    HSI_X = hub_stability_index(dfc_matrix, amb_score_fc, epsilon=epsilon, threshold=threshold)
    HSI_S = hub_stability_index(dfc_mask, amb_score_attr, epsilon=epsilon, threshold=threshold)

    denom = HSI_X + HSI_S
    HSI_combined = (2 * HSI_X * HSI_S / denom) if denom > 0 else 0.0

    return {
        "HSI_X": HSI_X,
        "HSI_S": HSI_S,
        "HSI_combined": HSI_combined,
    }
