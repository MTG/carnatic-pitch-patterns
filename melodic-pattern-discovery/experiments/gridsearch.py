%load_ext autoreload
%autoreload 2

annotations_path = 'data/annotations.txt'
tol = 0.3
partial_perc = 0.66

import os
import numpy as np

from src.evaluation import evaluate, load_annotations_brindha
from src.io import load_pkl, load_yaml, write_pkl
from src.core import run_pipeline
from src.visualisation import metrics_plot

## Loading
##########


for experiment in ['ftanet', 'melodia', 'ftanet_mix', 'melodia_mix', 'melodia_spleeter', 'melodia_spleeter_mix', 'ftanet_western', 'ftanet_western_mix']:

    ### Gridsearch
    ##############
    conf = load_yaml(f'conf/pipeline/four_experiments/{experiment}.yaml')
    plot_conf = load_yaml('conf/pipeline/plot_conf.yaml')
    exclusion_conf = load_yaml('conf/pipeline/exclusion_conf.yaml')

    cache = conf['cache']

    pitch_path = conf['pitch_path']
    audio_path = conf['audio_path']
    tonic = plot_conf['tonic']

    # postproc curve
    smooth = conf['smooth']
    gap_interp = conf['gap_interp']

    m_secs = conf['m_secs']
    annotations_raw = load_annotations_brindha(annotations_path, min(m_secs), max(m_secs))
    exclusion_funcs = [x['func'] for x in exclusion_conf['exclusion_funcs']]

    # For Carnatic music, we automatically define yticks_dict
    plot_kwargs = plot_conf
    output_audio = False
    output_plots = False
    output_patterns = False

    # Maximum number of unique motif groups to return
    top_n = conf['top_n']

    # Maximum number of occurrences to return in each motif group
    n_occ = conf['n_occ']

    # Minimum number of occurrences to return in each motif group
    min_occ = conf['min_occ']

    out_dir = conf['out_dir']

    thresh = list(np.arange(0.1, 3, 0.1))

    all_recalls = []
    all_precisions = []
    all_f1 = []
    for t in thresh:
        print('\n\n\n################')
        print(f'Threshold: {t}')
        print('################\n')
        starts, lengths = run_pipeline(
            cache, pitch_path, audio_path, None, tonic, smooth, gap_interp, m_secs, 
            exclusion_funcs, plot_kwargs, output_audio, output_plots, output_patterns, 
            top_n, n_occ, min_occ, tol, partial_perc, t, out_dir
        )

        recall, precision, f1, annotations = evaluate(annotations_raw, starts, lengths, partial_perc)
        
        all_recalls.append(recall)
        all_precisions.append(precision)
        all_f1.append(f1)

        print(f'    Recall: {round(recall*100,2)}%')
        print(f'    Precision: {round(precision*100,2)}%')
        print(f'    f1: {round(f1*100,2)}%')

    write_pkl(all_recalls, f'data/{experiment}/recall.pkl')
    write_pkl(all_precisions, f'data/{experiment}/precisions.pkl')
    write_pkl(all_f1, f'data/{experiment}/f1.pkl')
    write_pkl(thresh, f'data/{experiment}/thresh.pkl')

    plot_path = f'data/{experiment}/gridsearch_plot.png'
    metrics_plot(thresh, all_recalls, all_precisions, all_f1, plot_path, experiment)

