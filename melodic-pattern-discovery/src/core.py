import os
import sys
sys.path.append('../')

import matplotlib.pyplot as plt

import numpy as np
from scipy.ndimage import gaussian_filter1d
import essentia.standard as estd
import librosa

from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

from src.evaluation import evaluate, load_annotations_brindha
from src.utils import get_timestamp, interpolate_below_length
from src.visualisation import plot_all_sequences, double_timeseries, plot_subsequence
from src.iam import unpack_saraga_metadata
from src.io import write_all_sequence_audio, write_json, load_json, load_yaml, load_tonic, write_pkl, load_pkl
from src.sequence import get_silence_stability_mask
from src.matrix_profile import get_matrix_profile
from src.motif import get_motif_clusters, get_exclusion_mask
from src.pitch import cents_to_pitch, pitch_seq_to_cents, pitch_to_cents

from pathlib import Path


def get_timeseries(path):
    pitch = []
    time = []
    with open(path, 'r') as f:
        for i in f:
            t, f = i.replace('/n','').split(',')
            pitch.append(float(f))
            time.append(float(t))
    timestep = time[3]-time[2]
    return np.array(pitch), np.array(time), timestep


def find_motifs(conf, exclusion_conf, plot_conf):
    # Params
    ########
    cache = conf['cache']

    pitch_path = conf['pitch_path']
    audio_path = conf['audio_path']
    annotations_path = conf['annotations_path']
    tonic = plot_conf['tonic']

    # postproc curve
    smooth = conf['smooth']
    gap_interp = conf['gap_interp']

    m_secs = conf['m_secs']

    exclusion_funcs = [x['func'] for x in exclusion_conf['exclusion_funcs']]

    # For Carnatic music, we automatically define yticks_dict
    plot_kwargs = plot_conf
    output_audio = conf['output_audio']
    output_plots = conf['output_plots']
    output_patterns = conf['output_patterns']

    # Maximum number of unique motif groups to return
    top_n = conf['top_n']

    # Maximum number of occurrences to return in each motif group
    n_occ = conf['n_occ']

    # Minimum number of occurrences to return in each motif group
    min_occ = conf['min_occ']

    thresh = conf['thresh'] # patterns with parent distances above this threshold are not considered

    # For evaluation
    tol = conf['tol']
    partial_perc = conf['partial_perc']

    out_dir = conf['out_dir']

    all_starts, all_lengths = run_pipeline(
        cache, pitch_path, audio_path, annotations_path, tonic, smooth, gap_interp, m_secs, 
        exclusion_funcs, plot_kwargs, output_audio, output_plots, output_patterns, 
        top_n, n_occ, min_occ, tol, partial_perc, thresh, out_dir
    )


def run_pipeline(
    cache, pitch_path, audio_path, annotations_path, tonic, smooth, gap_interp, m_secs, 
    exclusion_funcs, plot_kwargs, output_audio, output_plots, output_patterns, 
    top_n, n_occ, min_occ, tol, partial_perc, thresh, out_dir):

    ## Pitch Extraction
    ###################
    raw_pitch_, time, timestep = get_timeseries(pitch_path)

    # Gap interpolation
    if gap_interp:
        print(f'Interpolating gaps of {gap_interp} or less')
        raw_pitch = interpolate_below_length(raw_pitch_, 0, int(gap_interp/timestep))
    else:
        raw_pitch = raw_pitch_

    # Gaussian smoothing
    if smooth:
        print(f'Gaussian smoothing with sigma={smooth}')
        pitch = gaussian_filter1d(raw_pitch, smooth)
    else:
        pitch = raw_pitch[:]

    if not isinstance(m_secs, list):
        m_secs = [m_secs]

    print('Getting stable/silence mask')
    mask_path = os.path.join(Path(pitch_path).parent.absolute(), 'mask.pkl')
    if os.path.exists(mask_path):
        print(f'loading mask from cache, {mask_path}')
        mask = load_pkl(mask_path)
    else:
        mask = get_silence_stability_mask(raw_pitch, 1, 0.2, 10, timestep)
        write_pkl(mask, mask_path)

    all_starts = []
    all_lengths = []
    for M in m_secs:
        if cache:
            cache_dir = os.path.join(Path(pitch_path).parent.absolute(), '.matrix_profile', f'm={M}_gap_interp={gap_interp}_smooth={smooth}','')

        ## Matrix Profile
        #################
        print('#######')
        print(f'Finding motifs of length {M} seconds')
        print('#######')
        print('')
        # Convert to elements
        m_el = int(M/timestep)

        matrix_profile, matrix_profile_length = get_matrix_profile(pitch, m_el, path=cache_dir)
        # Can take some time depending on exclusion_funcs
        if cache:
            exclusion_cache_path = os.path.join(Path(pitch_path).parent.absolute(), '.exclusion_mask', f'm={M}_gap_interp={gap_interp}_smooth={smooth}/', '')
        else: 
            exclusion_cache_path = None
        
        print(f'Applying exclusion funcs: {[x.__name__ for x in exclusion_funcs]}')
        exclusion_mask = get_exclusion_mask(mask, matrix_profile_length, exclusion_funcs, path=exclusion_cache_path)
        print(f'{round(sum(exclusion_mask)*100/len(exclusion_mask), 2)}% subsequences excluded')

        motifs, distances, motif_len = get_motif_clusters(matrix_profile, pitch, matrix_profile_length, top_n, n_occ, exclusion_mask, thresh=thresh, min_occ=min_occ)

        print(f'{len(motifs)} motif groups found')
        all_starts = all_starts + motifs
        all_lengths = all_lengths + [[motif_len[i] for y in x] for i,x in enumerate(motifs)]

    all_starts_sec = [[x*timestep for x in y] for y in all_starts]
    all_lengths_sec = [[x*timestep for x in y] for y in all_lengths]

    if annotations_path:
        annotations_raw = load_annotations_brindha(annotations_path, min(m_secs), max(m_secs))
        recall, precision, f1, annotations = evaluate(annotations_raw, all_starts_sec, all_lengths_sec, partial_perc)
        print(f'Recall: {round(recall, 2)}')
        print(f'Precision: {round(precision, 2)}')
        print(f'F1: {round(f1, 2)}')


    ## Output
    #########
    # If the directory does not exist, it will be created
    out_dir_ = os.path.join(out_dir, 'all_lengths', '')
    print(f'Output directory: {out_dir_}')
    if output_plots:
        print('Writing plots')
        plot_all_sequences(raw_pitch_, time, all_lengths, all_starts, out_dir_, clear_dir=True, plot_kwargs=plot_kwargs)

    if output_audio:
        print('Writing audio')
        write_all_sequence_audio(audio_path, all_starts, all_lengths, timestep, out_dir_)

    parameters = {
        'top_n': top_n,
        'n_occ':n_occ,
        'min_occ':min_occ,
        'thresh':thresh,
        'pitch_path': pitch_path
    }
    print('Writing metadata')
    write_json(parameters, os.path.join(out_dir_, 'parameters.json'))

    if output_patterns:
        all_starts_path = os.path.join(out_dir, 'all_starts.pkl')
        all_lengths_path = os.path.join(out_dir, 'all_lengths.pkl')
        annot_out_path = os.path.join(out_dir, 'annotations.csv')
        metric_path = os.path.join(out_dir, 'metrics.txt')

        if annotations_path:
            print(f'Writing annotations to {annot_out_path}')
            annotations.to_csv(annot_out_path, index=False)
            
            print(f'Writing metrics to {metric_path}')
            with open(metric_path, 'w') as f:
                f.write(f'recall, {recall}\nprecision, {precision}\nf1, {f1}')

        print(f'Writing patterns starts to {all_starts_path}')
        print(f'Writing patterns lengths to {all_lengths_path}')
        write_pkl(all_starts_sec, all_starts_path)
        write_pkl(all_lengths_sec, all_lengths_path)   

    return all_starts_sec, all_lengths_sec
