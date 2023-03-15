from collections import Counter

import numpy as np
import pandas as pd
import os

experiments = ['melodia', 'melodia_mix', 'ftanet', 'ftanet_mix', 'melodia_spleeter', 'melodia_spleeter_mix']
annotations = None
for experiment in experiments:
    path = os.path.join(f'data/{experiment}/results', 'annotations.csv')
    df = pd.read_csv(path)
    df.columns = ['tier', 's1', 's2', 'text', f'match_{experiment}', f'group_num_{experiment}', f'occ_num_{experiment}']
    if annotations is None:
        annotations = df
    else:
        annotations = pd.merge(annotations, df, on=['tier', 's1', 's2', 'text'])

match_ix = {}
for experiment in experiments:
    match_ix[experiment] = annotations[annotations[f'match_{experiment}'] != 'no match'].index

comparison_ix = {}
for e1 in experiments:
    for e2 in experiments:
        if e1 not in comparison_ix:
            comparison_ix[e1] = {}
        comparison_ix[e1][e2] = [x for x in match_ix[e1] if x not in match_ix[e2]]


# Some statistics
print(f'There are {len(annotations)} patterns in total')
lengths = Counter([round(x) for x in annotations['s2']-annotations['s1']])
for k,v in lengths.items():
    print(f'    {v} of length {k}s ({round(v*100/len(annotations),1)}%)')
for e1 in experiments:
    for e2 in experiments:
        if e1 == e2:
            continue
        indices = comparison_ix[e1][e2]
        if len(indices) == 0:
            print(f'- Every pattern identified by {e1} was also identified by {e2}')
        else:
            pc = round(len(indices)*100/len(match_ix[e1]),1)
            print(f'- {len(indices)} patterns identified by {e1} were not identified by {e2} ({pc}%)')
            these_indices = annotations[annotations.index.isin(indices)]
            lengths = Counter([round(x) for x in these_indices['s2']-these_indices['s1']])
            for k,v in lengths.items():
                print(f'    {v} of length {k}s ({round(v*100/len(these_indices),1)}%)')


ftanet_v_melodia_mix = annotations[annotations.index.isin(comparison_ix['ftanet_mix']['melodia_mix'])]




annotations[[
    'tier', 's1', 's2', 'text', 
    'match_melodia', 'match_melodia_mix', 'match_ftanet', 'match_ftanet_mix', 'match_melodia_spleeter', 'match_melodia_spleeter_mix',
    'group_num_melodia', 'group_num_melodia_mix', 'group_num_ftanet', 'group_num_ftanet_mix','group_num_melodia_spleeter', 'group_num_melodia_spleeter_mix',
    'occ_num_melodia',  'occ_num_melodia_mix',  'occ_num_ftanet', 'occ_num_ftanet_mix', 'occ_num_melodia_spleeter', 'occ_num_melodia_spleeter_mix'
]]