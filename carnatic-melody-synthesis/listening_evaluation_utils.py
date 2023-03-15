import os
import glob
import numpy as np
import pandas as pd
#import sounddevice as sd
from scipy.io import wavfile

from pathlib import Path

def get_dataset_mapping(data_path):
    """
    Map the IAM melody dataset into a python dictionary
    data_path: path to the dataset foldeer
    returns dataset_mapping (dict): mapping of the songs and ids in the dataset
    """
    file_list = glob.glob(os.path.join(data_path, 'audio', '*.wav'))
    dataset_mapping = {}
    for track in file_list:
        filename = track.split('/')[-1]
        song_name = '_'.join(filename.split('_')[:-1])
        track_id = filename.split('_')[-1].replace('.wav', '')
        #id_list = [x.split('/')[-1].replace('.wav', '').split('_')[-1] for x in file_list if song_name in x]
        if dataset_mapping.get(song_name) is None:
            dataset_mapping[song_name] = [track_id]
        else:
            dataset_mapping[song_name].append(track_id)
        
    return dataset_mapping


def sinewaveSynth(freqs, amp, H, fs):
    """
    Synthesis of one sinusoid with time-varying frequency
    freqs, amps: array of frequencies and amplitudes of sinusoids
    H: hop size, fs: sampling rate
    returns y: output array sound
    """

    t = np.arange(H)/float(fs)                              # time array
    lastphase = 0                                           # initialize synthesis phase
    lastfreq = freqs[0]                                     # initialize synthesis frequency
    y = np.array([])                                        # initialize output array
    for l in range(freqs.size):                             # iterate over all frames
        if (lastfreq==0) & (freqs[l]==0):                     # if 0 freq add zeros
            A = np.zeros(H)
            freq = np.zeros(H)
        elif (lastfreq==0) & (freqs[l]>0):                    # if starting freq ramp up the amplitude
            A = np.arange(0,amp, amp/H)
            freq = np.ones(H)*freqs[l]
        elif (lastfreq>0) & (freqs[l]>0):                     # if freqs in boundaries use both
            A = np.ones(H)*amp
            if (lastfreq==freqs[l]):
                freq = np.ones(H)*lastfreq
            else:
                freq = np.arange(lastfreq,freqs[l], (freqs[l]-lastfreq)/H)
        elif (lastfreq>0) & (freqs[l]==0):                    # if ending freq ramp down the amplitude
            A = np.arange(amp,0,-amp/H)
            freq = np.ones(H)*lastfreq
        phase = 2*np.pi*freq*t+lastphase                      # generate phase values
        yh = A * np.cos(phase)                                # compute sine for one frame
        lastfreq = freqs[l]                                   # save frequency for phase propagation
        lastphase = np.remainder(phase[H-1], 2*np.pi)         # save phase to be use for next frame
        y = np.append(y, yh)                                  # append frame to previous one
    return y


def synthesize_f0(song, track_id, attenuation=0.1):
    # Build filename
    filename = os.path.join(
        Path().absolute(),
        'resources',
        'Saraga-Synth-Dataset',
        'annotations',
        'melody',
        song + '_' + track_id + '.csv'
    )
    
    # Import annotation
    ycsv = pd.read_csv(filename, names=["time", "freq"])
    gtf = ycsv['freq'].values
    
    y = sinewaveSynth(gtf, attenuation, 128, 44100)
    
    return y


def get_audio_mono(data_path, song, track_id):
    return wavfile.read(
        os.path.join(data_path, 'audio', song + '_' + track_id + '.wav')
    )

def play_output_mix(audio, f0, sample_rate=44100, center=False):
    if len(audio) != len(f0):
        audio = audio[:min(len(audio), len(f0))]
        f0 = f0[:min(len(audio), len(f0))]
        
    if not center:
        output_mix = np.zeros([len(audio), 2])
        output_mix[:, 0] = audio
        output_mix[:, 1] = f0
        sd.play(output_mix, sample_rate)
    else:
        output_mix = audio + f0
        sd.play(output_mix, sample_rate)
   