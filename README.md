# Robustness of explainable AI algorithms for disease biomarker discovery from functional connectivity datasets

In this [paper](https://openreview.net/pdf?id=3kti62n63m), we propose quantitative metrics to evaluate the robustness of salient features identified DL models for brain disease classification. 

This repository contains the implementation of the following metrics:-

1. Model Parameter Randomization Check
2. Data Randomization Check
3. Fidelity
4. Stability
5. Target Sensitivity
6. Implementation Invariance
7. Modular Ratio
8. Hub Assortativity Coefficient

We package all these metrics in the RE-CONFIRM framework. 

## Guide to run the code

1.	The `tasks` folder holds the function implementations for the RE-CONFIRM metrics.
2.	The `assets` folder includes all necessary data to execute the scripts in `tasks`. This data includes encoded labels for both the ABIDE and ADHD datasets, as well as the topological properties of functional connectivity, saved in `.npy` format.
3.	The `utils` folder contains wrapper functions and helper code for calculating the metrics.
4.	Finally, `run_experiments.py` and `main.py` provide baseline code for training models, generating saliency scores, and calculating metrics based on these scores.
5.	The script `run_experiments.py` requires four mandatory arguments at runtime: type of GNNs `[gcn, cheb, gat, sage]` to be utilized, explainability method to be applied `[gnnexp, attnexp, ig, pgexp, gbackprop]`, dataset to be analyzed `[abide, adhd]` and choice of analysis site: either a single site (NYU) or a combination of all available sites `[yes, no]`.
7. Additionally, `main.py` requires an extra argument specifying the evaluation metric to be used: `[scores, mprc, drc, stability, invariance, ambivert, hdr, fidelity_individual, homfid, mod]`

8.	You can run the code by running the following command:
    `python run_experiments.py gcn gnnexp abide no`
    Here, we are utilizing the GCN architecture with the GNNExplainer method, applied across all sites of the ABIDE dataset.

More updates to follow.
