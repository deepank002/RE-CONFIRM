import sys

if len(sys.argv) != 5:
    print("usage: have to provide four arguments in total")
    sys.exit(1)

gnn = sys.argv[1]           # type of gnn 
method = sys.argv[2]        # method of explainability
dataset = sys.argv[3]       # dataset type
single_site = sys.argv[4]   # only single site used 

import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gensim.matutils import hellinger
from scipy.stats import wasserstein_distance, spearmanr
from sklearn.metrics.pairwise import cosine_similarity


if single_site == "no" and dataset == "abide":
    directory = "/all_sites" 
if single_site == "yes" and dataset == "abide":
    directory = "/single_site"
if single_site == "no" and dataset == "adhd":
    directory = "/all_sites"
if single_site == "yes" and dataset == "adhd":
    directory = "/single_site"

   
def scores_to_pdf(orig_data):
    #counts, bins, _ = plt.hist(orig_data, bins='auto', density=True)   # density=True for PDF
    counts, bins, _ = plt.hist(orig_data, bins=15, density=True)
    bin_widths = np.diff(bins)
    pdf_values = counts * bin_widths    # Calculate PDF & CDF values
    return pdf_values


def calculate_distance(subfolder, distance_function, orig_data=None, new_data=None):
    if orig_data is None and new_data is None:
        orig_data = np.load('./gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/init/' + method + '/all_subject_scores.npy')
        new_data = np.load('./gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/' + subfolder + '/' + method + '/all_subject_scores.npy')
    
    return distance_function(scores_to_pdf(orig_data), scores_to_pdf(new_data))


def calculate_mean_std(fpath):
    score = np.load(fpath)
    score_tuple = (np.nanmean(score), np.nanstd(score))
    return score_tuple

"""
def modular_ratio(data, file_path):
    keys = data[0].keys()
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
"""

def analysis(gnn, dataset, directory, method):
    path = 'results/' + gnn + '/' + dataset + '/' + directory[1:] + '/' + method
    os.makedirs(path, exist_ok=True)
    
    metrics = ["Model_Parameter_Randomization_Check", "Data_Randomization_Check", "Fidelity", "Stability", "Sensitivity", 
               "Implementation_Invariance", "Ambivert_Scores", "Homophilic_Fidelity", "Hub_Assortativity_Coefficient"]
    rows = ["Values"]
    df = pd.DataFrame(columns=rows, index=metrics)
    
    df["Values"]["Model_Parameter_Randomization_Check"] = (calculate_distance('mprc', wasserstein_distance), calculate_distance('mprc', hellinger))
    
    df["Values"]["Data_Randomization_Check"] = (calculate_distance('drc', wasserstein_distance), calculate_distance('drc', hellinger))

    df["Values"]["Implementation_Invariance"] = (calculate_distance('ii', wasserstein_distance), calculate_distance('ii', hellinger))
    
    orig_data = np.load('./gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/init/' + method + '/all_subject_scores.npy')
    orig_data = orig_data.reshape(1, -1)
    new_data = np.load('./gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/stability/' + method + '/all_subject_scores.npy')
    new_data = new_data.reshape(1, -1)
    df["Values"]["Stability"] = cosine_similarity(orig_data, new_data)
    
    tdc_mean = np.load('./gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/init/' + method + '/tdc_node_mask_mean_scores.npy')
    tdc_mean = np.sum(tdc_mean, axis=1)
    disease_mean = np.load('./gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/init/' + method + '/disease_subj_node_mask_mean_scores.npy')
    disease_mean = np.sum(disease_mean, axis=1)
    df["Values"]["Sensitivity"] = (calculate_distance(None, wasserstein_distance, tdc_mean, disease_mean), calculate_distance(None, hellinger, tdc_mean, disease_mean))
    
    fidelity_mpath = './gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/fidelity/' + method + '/fidelity_individual_score_minus_20.npy'
    fidelity_ppath = './gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/fidelity/' + method + '/fidelity_individual_score_plus_20.npy'
    df["Values"]["Fidelity"] = (calculate_mean_std(fidelity_mpath), calculate_mean_std(fidelity_ppath))
    
    fidelity_mpath = './gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/homfid/' + method + '/fidelity_individual_score_minus_20.npy'
    fidelity_ppath = './gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/homfid/' + method + '/fidelity_individual_score_plus_20.npy'
    df["Values"]["Homophilic_Fidelity"] = (calculate_mean_std(fidelity_mpath), calculate_mean_std(fidelity_ppath))
 
    ambivert_path = './gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/ambivert/' + method + '/corr_ambivert.npy'
    spearman_path = './gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/ambivert/' + method + '/spearman_ambivert.npy'
    df["Values"]["Ambivert_Scores"] = (calculate_mean_std(ambivert_path), calculate_mean_std(spearman_path))
    
    hac_path = './gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/hac/' + method + '/hub_assortativity_coeff_all_subjects.npy'
    df["Values"]["Hub_Assortativity_Coefficient"] = calculate_mean_std(hac_path)
    
    orig_datas = np.load('./gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/init/' + method + '/all_subject_scores.npy')
    hub_files = os.listdir('./assets/' + dataset + '_all_sites')
    hub_files = [file[:-4] for file in hub_files if file.startswith('n')]

    for file in hub_files:
        hub_measure = np.load('./assets/' + dataset + '_all_sites/' + file + '.npy').mean(axis=0)
        df = df._append(pd.DataFrame(index=[file]))
        df["Values"][file] = (np.corrcoef(orig_datas, hub_measure)[0,1], spearmanr(orig_datas.reshape(-1,1), hub_measure.reshape(-1,1)).statistic)
    
    morehac_path = './gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/morehac/' + method
    file = os.listdir(morehac_path)
    file_name = file[0][:-4][:-13]
    df = df._append(pd.DataFrame(index=[file_name]))
    df["Values"][file_name] = calculate_mean_std(morehac_path + '/' + file[0])
    
    df.to_csv(path + '/' + gnn + '_' + dataset + '_' + method + '_' + directory[1:] + '_results.csv')
    
    hdr = np.load('./gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/hdr/' + method + '/homophilic_ratio_all_subjects.npy')
    np.savetxt(path + '/' + gnn + '_' + dataset + '_' + method + '_' + directory[1:] + '_homophilic_distance_ratio.csv', hdr, delimiter=",")
    
    mod_ratio = np.load('./gnn/' + gnn + '/dataset/' + dataset + directory + '/scores/modr/' + method + '/modular_ratio_all_subjects.npy', allow_pickle=True)
    #modular_ratio(mod_ratio, './results/modular_ratio.csv')
    np.savetxt(path + '/' + gnn + '_' + dataset + '_' + method + '_' + directory[1:] + '_modular_ratio.csv', mod_ratio, delimiter=",")
        
    print("Calculated all scores successfully")


if __name__ == "__main__":
    analysis(gnn, dataset, directory, method)