%load_ext autoreload
%autoreload 2

import IPython.display as ipd
import os
import sys
sys.path.append('../')

import matplotlib.pyplot as plt
%matplotlib inline
#matplotlib.use('TkAgg')

import numpy as np
from scipy.ndimage import gaussian_filter1d
import essentia.standard as estd
import librosa

from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

from src.utils import get_timestamp, interpolate_below_length
from src.visualisation import plot_all_sequences, double_timeseries, plot_subsequence
from src.iam import unpack_saraga_metadata
from src.io import write_all_sequence_audio, write_json, load_json, load_yaml, load_tonic, create_if_not_exists
from src.matrix_profile import get_matrix_profile
from src.motif import get_motif_clusters, get_exclusion_mask
from src.pitch import cents_to_pitch, pitch_seq_to_cents, pitch_to_cents


mir_datasets_dir = '<insert_dir_here>'

# Akkarai Sisters - Koti Janmani
recording_dir = "Akkarai Sisters at Arkay by Akkarai Sisters/Koti Janmani/"

audio_file = "Koti Janmani.mp3.mp3"
metadata_file = "Koti Janmani.json"

audio_path = os.path.join(mir_datasets_dir, recording_dir, audio_file)
metadata_path = os.path.join(mir_datasets_dir, recording_dir, metadata_file)

tonic = 195.99

svara_cent_path = "../conf/svara_cents.yaml"
svara_freq_path = "../conf/svara_lookup.yaml"

svara_cent = load_yaml(svara_cent_path)
svara_freq = load_yaml(svara_freq_path)

sampling_rate = 44100 # defined on load in next cell
frameSize = 2048 # For Melodia pitch extraction
hopSize = 128 # For Melodia pitch extraction
gap_interp = 250*0.001 # Interpolate pitch tracks gaps of <gap_interp>seconds or less [set to None to skip]

# load raw audio for display later
audio_loaded, sr = librosa.load(audio_path, sr=sampling_rate)

# Run spleeter on track to remove the background
separator = Separator('spleeter:2stems')
audio_loader = AudioAdapter.default()
waveform, _ = audio_loader.load(audio_path, sample_rate=sampling_rate)
prediction = separator.separate(waveform=waveform)
clean_vocal = prediction['vocals']

audio_mono = clean_vocal.sum(axis=1) / 2

# Prepare audio for pitch extraction
audio_mono_eqloud = estd.EqualLoudness(sampleRate=sampling_rate)(audio_mono)

# Extract pitch using Melodia algorithm from Essentia
pitch_extractor = estd.PredominantPitchMelodia(frameSize=frameSize, hopSize=hopSize)
raw_pitch, _ = pitch_extractor(audio_mono_eqloud)
raw_pitch_ = np.append(raw_pitch, 0.0)
time = np.linspace(0.0, len(audio_mono_eqloud) / sampling_rate, len(raw_pitch))

timestep = time[4]-time[3] # resolution of time track

raw_pitch = interpolate_below_length(raw_pitch_, 0, int(gap_interp/timestep))
    
pitch = raw_pitch[:]

import csv

out_path = '/Users/thomasnuttall/code/MTG/ASPLAB/cmmr_tismir/data/melodia_spleeter_mix/47_Koti_Janmani.csv'
create_if_not_exists(out_path)

with open(out_path, 'w') as f:
	writer = csv.writer(f)
	for p,t in zip(pitch, time):
		# write a row to the csv file
		writer.writerow([t, p])