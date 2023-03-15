%load_ext autoreload
%autoreload 2

import os
import numpy as np

from src.evaluation import evaluate, load_annotations_brindha
from src.io import load_pkl, load_yaml, write_pkl
from src.core import run_pipeline, get_timeseries
from src.visualisation import metrics_plot

## Loading
##########


for experiment in ['ftanet', 'melodia', 'ftanet_mix', 'melodia_mix', 'melodia_spleeter', 'melodia_spleeter_mix', 'ftanet_western', 'ftanet_western_mix']:
	starts_path = os.path.join(f'data/{experiment}/results/all_starts.pkl')
	starts = load_pkl(starts_path)
	n_groups = len(starts)
	n_patterns = len([x for y in starts for x in y])
	print(f'{experiment}, {n_patterns}, {n_groups}')





for experiment in ['ftanet', 'melodia', 'ftanet_mix', 'melodia_mix', 'melodia_spleeter', 'melodia_spleeter_mix', 'ftanet_western', 'ftanet_western_mix']:
	pitch_path = os.path.join(f'data/{experiment}/47_Koti_Janmani.csv')
	raw_pitch_, time, timestep = get_timeseries(pitch_path)
	
	coverage = sum(raw_pitch_!=0)/len(raw_pitch_)
	print(f'{experiment}, {coverage}')