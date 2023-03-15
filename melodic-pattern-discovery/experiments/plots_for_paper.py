%load_ext autoreload
%autoreload 2

from src.visualisation import plot_subsequence_paper
from src.utils import get_timestamp, myround
import seaborn as sns

import IPython.display as ipd
import os
import sys

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import essentia.standard as estd
import librosa

from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

from src.utils import get_timestamp, interpolate_below_length
from src.visualisation import plot_all_sequences, double_timeseries, plot_subsequence
from src.iam import unpack_saraga_metadata
from src.io import write_all_sequence_audio, write_json, load_json, load_yaml, load_tonic
from src.matrix_profile import get_matrix_profile
from src.motif import get_motif_clusters, get_exclusion_mask
from src.pitch import cents_to_pitch, pitch_seq_to_cents, pitch_to_cents
from src.core import get_timeseries
from src.io import create_if_not_exists
from src.evaluation import load_annotations_brindha

def compare_pitch_tracks(t_in, t_out, experiments, out_path, title):

    create_if_not_exists(out_path)
    pitch_paths = [f'data/{e}/47_Koti_Janmani.csv' for e in experiments]
    all_timeseries = [get_timeseries(p) for p in pitch_paths]
    
    tonic = 195.99
    gap_interp = 250*0.001 # Interpolate pitch tracks gaps of <gap_interp>seconds or less [set to None to skip]
    svara_cent_path = "conf/svara_cents.yaml"
    svara_cent = load_yaml(svara_cent_path)
    yticks_dict = {k:cents_to_pitch(v, tonic) for k,v in svara_cent.items()}
    yticks_dict = {k:v for k,v in yticks_dict.items() if any([x in k for x in ['S', 'R2', 'G2', 'M1', 'P', 'D2', 'N2', 'S']])}


    plot_kwargs = {
        'yticks_dict':{k:v for k,v in yticks_dict.items() if any([x in k for x in ['S', 'R1', 'R2', 'G2', 'G3', 'M1', 'M2', 'P', 'D1', 'D2', 'N2', 'N3']])},
        'cents':True,
        'tonic':tonic,
        'emphasize':['S', 'S^'],
        'figsize':(15,4),#,
        #'ylim':(0, 1400),
        'title':title 
        #xlabel=None,
        #ylabel=None, 
        #grid=True, 
        #ylim=None, 
        #xlim=None
    }


    plt.figure(figsize=plot_kwargs['figsize'])
    pal = sns.color_palette("Dark2")

    ymax = -300000
    ymin = 300000
    for i, e in enumerate(experiments):
               
        #title = f'Occurence {i} - Pattern Length = {m_secs} seconds, begins at {minutes} minutes {seconds} seconds'
        raw_pitch_, time, timestep = all_timeseries[i]
        raw_pitch = interpolate_below_length(raw_pitch_, 0, int(gap_interp/timestep))

        t1 = int(t_in/timestep)
        t2 = int(t_out/timestep)

        pitch_masked = np.ma.masked_where(raw_pitch==0, pitch_seq_to_cents(raw_pitch, tonic))

        plt.plot(time[t1:t2]-time[t1], pitch_masked[t1:t2], alpha=0.5, label=f'{e}', color=pal[i-1], linewidth=1.1)
        
        ax = plt.gca()
        if yticks_dict:
            tick_names = list(yticks_dict.keys())
            tick_loc = [pitch_to_cents(p, tonic) for p in yticks_dict.values()]
            ax.set_yticks(tick_loc)
            ax.set_yticklabels(tick_names)
        
        non_none_pitch = [x for x in pitch_masked[t1:t2].data if x]
        ymin = min(ymin, min(non_none_pitch))
        ymax = max(ymax, max(non_none_pitch))
        
        ax.set_ylim((ymin, ymax))
    
    labels = np.arange(t_in, t_out, step=1)
    labels = [round(x,1) for x in labels]
    plt.xticks(np.arange(0, len(labels), step=1),labels)  # Set label locations.

    plt.title(title)
    plt.grid()
    plt.legend()
    plt.ylabel(f'Cents above tonic of {round(tonic)}Hz')
    plt.xlabel(f'Time (s)')
    plt.savefig(out_path, pad_inches=0)
    plt.close('all')


def paper_plot(experiment, m_secs, title, out_path, all_t, opacity = 0.5, linestyles='solid'):

    create_if_not_exists(out_path)
    pitch_path = f'data/{experiment}/47_Koti_Janmani.csv'
    raw_pitch_, time, timestep = get_timeseries(pitch_path)
    tonic = 195.99
    gap_interp = 250*0.001 # Interpolate pitch tracks gaps of <gap_interp>seconds or less [set to None to skip]
    svara_cent_path = "conf/svara_cents.yaml"
    svara_cent = load_yaml(svara_cent_path)
    yticks_dict = {k:cents_to_pitch(v, tonic) for k,v in svara_cent.items()}
    yticks_dict = {k:v for k,v in yticks_dict.items() if any([x in k for x in ['S', 'R2', 'G2', 'M1', 'P', 'D2', 'N2', 'S']])}


    plot_kwargs = {
        'yticks_dict':{k:v for k,v in yticks_dict.items() if any([x in k for x in ['S', 'R1', 'R2', 'G2', 'G3', 'M1', 'M2', 'P', 'D1', 'D2', 'N2', 'N3']])},
        'cents':True,
        'tonic':tonic,
        'emphasize':['S', 'S^'],
        'figsize':(15,4),#,
        #'ylim':(0, 1400),
        'title':title 
        #xlabel=None,
        #ylabel=None, 
        #grid=True, 
        #ylim=None, 
        #xlim=None
    }


    raw_pitch = interpolate_below_length(raw_pitch_, 0, int(gap_interp/timestep))


    pitch_masked = np.ma.masked_where(raw_pitch==0, pitch_seq_to_cents(raw_pitch, tonic))
    plt.figure(figsize=plot_kwargs['figsize'])
    pal = sns.color_palette("Dark2")

    ymax = -300000
    ymin = 300000
    for i, (minutes, seconds) in enumerate(all_t, 1):
        
        t = minutes*60+seconds
        m = m_secs/timestep

        t1 = int(t/timestep)
        t2 = int(t1 + m)
        
        #title = f'Occurence {i} - Pattern Length = {m_secs} seconds, begins at {minutes} minutes {seconds} seconds'
        if linestyles != 'solid':
            assert len(linestyles)==len(all_t), "not enough linestyles for each plot"
            l = linestyles[i-1]

        time_str = f'0{minutes}:{round(seconds):02}'
        plt.plot(time[t1:t2]-time[t1], pitch_masked[t1:t2], alpha=opacity, label=f'{i}. starts at {time_str}', color=pal[i-1], linewidth=1.1, linestyle=l)
        
        ax = plt.gca()
        if yticks_dict:
            tick_names = list(yticks_dict.keys())
            tick_loc = [pitch_to_cents(p, tonic) for p in yticks_dict.values()]
            ax.set_yticks(tick_loc)
            ax.set_yticklabels(tick_names)
        
        non_none_pitch = [x for x in pitch_masked[t1:t2].data if x]
        ymin = min(ymin, min(non_none_pitch))
        ymax = max(ymax, max(non_none_pitch))
        
        ax.set_ylim((ymin, ymax))
        
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.ylabel(f'Cents above tonic of {round(tonic)}Hz')
    plt.xlabel(f'Seconds after start of sequence')
    plt.savefig(out_path, pad_inches=0)
    plt.close('all')

### Single pattern plots
experiment = 'ftanet_mix'
m_secs = 4
title = f'MELODIA-S MIX: Motif Group 2, {m_secs} seconds'
out_path = 'paper_plots/melodia_spleeter_mix.png'
# (minute, second)
all_t =  [
    (5, 8.92),
    (3, 19.38),
    (6, 57.06),
    (7, 7.87),
]
linestyles = ['solid', 'solid', 'dashed', 'dashed']

paper_plot(experiment, m_secs, title, out_path, all_t, 0.5, linestyles)

t1 = 180.044
t2 = 183.485

example_ann = [(251.822,257.081),(463.194,467.639),(352.377,357.128),(338.128,340.828)]
example_names = ['grsssnrsnndmgmn', 'ssnsrsndmmgmpmggrs', 'mmmmmpmgrgm','sssnnpn']
example_add_str = [
    ' (ftanet_mix - match, melodia_spleeter_mix - no match)',
    ' (ftanet_mix - no match, melodia_spleeter_mix - match)',
    ' (ftanet_mix - match, melodia_spleeter_mix - no match)',
    ' (ftanet_mix - match, melodia_spleeter_mix - no match)',
    ]

for i in range(len(example_ann)):
    out_path = f'paper_plots/{example_names[i]}_{example_ann[i][0]}_{example_ann[i][1]}.png'
    experiments = ['ftanet_mix', 'melodia_spleeter_mix']
    title = f'{example_names[i]} {example_add_str[i]}'
    compare_pitch_tracks(example_ann[i][0], example_ann[i][1], experiments, out_path, title)





### Plotting all annotations
import tqdm
annotations = pd.read_csv('annotations_joined.csv')

experiments = ['ftanet', 'ftanet_mix', 'melodia_spleeter_mix', 'melodia_spleeter']
for i, df in tqdm.tqdm(list(annotations.iterrows())):
    tier = df['tier']
    s1 = df['s1']
    s2 = df['s2']
    text = df['text']
    match_ftanet_mix = df['match_ftanet_mix'].replace('_',' ')
    match_melodia_spleeter_mix = df['match_melodia_spleeter_mix'].replace('_',' ')

    if 'phrase' in tier:
        tier = 'phrase'
    else:
        tier = 'sancara'

    if match_ftanet_mix=='match' and match_melodia_spleeter_mix=='match':
        folder = 'both_match'
    elif match_ftanet_mix=='match' and not match_melodia_spleeter_mix=='match':
        folder = 'ftanet_match'
    elif match_melodia_spleeter_mix=='match' and not match_ftanet_mix=='match':
        folder = 'melodia_match'
    else:
        folder = 'no_match'

    out_path = f'paper_plots/all_annotations/{folder}/{i}_{text}_({round(s1,1)}s-{round(s2,1)}s).png' 
    experiments = ['ftanet_mix', 'melodia_spleeter_mix']
    title = f'Annotated underlying {tier}, {text} [FTANET-C, MIX - {match_ftanet_mix} | MELODIA-S, MIX - {match_melodia_spleeter_mix}]'
    compare_pitch_tracks(s1, s2, experiments, out_path, title)




