import sys

if len(sys.argv) != 5:
    print("usage: have to provide four arguments in total")
    sys.exit(1)

gnn = sys.argv[1]           # type of gnn 
method = sys.argv[2]        # method of explainability
dataset = sys.argv[3]       # dataset type
single_site = sys.argv[4]   # only single site used

import os


tasks = ['scores', 'mprc', 'drc', 'stability', 'invariance', 'ambivert', 'hdr', 'fidelity_individual', 'homfid', 'modr', 'hac', 'morehac']

try:
    #os.system(f'python main.py {gnn} {method} train {dataset} {single_site}')
    
    for task in tasks:
        os.system(f'python main.py {gnn} {method} {task} {dataset} {single_site}')

    os.system(f'python analysis.py {gnn} {method} {dataset} {single_site}')

    print("Successfully finished all experiments")

except Exception as e:
    print("An error occurred:", e)
    exit(0)
