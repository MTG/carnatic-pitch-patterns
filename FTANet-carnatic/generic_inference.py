import numpy as np
import mirdata
from scipy.signal import resample
from scipy.ndimage import gaussian_filter1d
from cfp import cfp_process
from tensorflow import keras
from pydub import AudioSegment
import tempfile
import soundfile as sf
import os
import math
import argparse
from pathlib import Path

from constant import *
from loader import *

from network.ftanet import create_model
from loader import get_CenFreq

import essentia.standard as estd

def load_audio(filepath, sr=None, mono=True, dtype='float32'):
    if '.mp3' in filepath:
        mp3 = AudioSegment.from_mp3(filepath)
        _, path = tempfile.mkstemp()
        mp3.export(path, format="wav")
        del mp3
        x, fs = sf.read(path)
        os.remove(path)
    else:
        x, fs = sf.read(filepath)
    
    if mono and len(x.shape) > 1:
        x = np.mean(x, axis=1)
    if sr:
        x = scipy.signal.resample_poly(x, sr, fs)
        fs = sr
    x = x.astype(dtype)
    
    return x, fs


def interpolate_below_length(arr, val, gap_len):
    s = np.copy(arr)
    is_zero = s == val
    cumsum = np.cumsum(is_zero).astype('float')
    diff = np.zeros_like(s)
    diff[~is_zero] = np.diff(cumsum[~is_zero], prepend=0)
    for i, d in enumerate(diff):
        if d <= gap_len:
            s[int(i-d):i] = np.nan
    interp = pd.Series(s).interpolate(method='linear', axis=0)\
                            .ffill()\
                            .bfill()\
                            .values
    return interp

def std_normalize(data): 
    # normalize as 64 bit, to avoid numpy warnings
    data = data.astype(np.float64)
    mean = np.mean(data)
    std = np.std(data)
    data = data.copy() - mean
    if std != 0.:
        data = data / std
    return data.astype(np.float32)

def est(output, CenFreq, time_arr):
    # output: (freq_bins, T)
    CenFreq[0] = 0
    est_time = time_arr
    est_freq = np.argmax(output, axis=0)

    for j in range(len(est_freq)):
        est_freq[j] = CenFreq[int(est_freq[j])]

    if len(est_freq) != len(est_time):
        new_length = min(len(est_freq), len(est_time))
        est_freq = est_freq[:new_length]
        est_time = est_time[:new_length]

    est_arr = np.concatenate((est_time[:, None], est_freq[:, None]), axis=1)

    return est_arr

def iseg(data):
    # data: (batch_size, freq_bins, seg_len)
    new_length = data.shape[0] * data.shape[-1]  # T = batch_size * seg_len
    new_data = np.zeros((data.shape[1], new_length))  # (freq_bins, T)
    for i in range(len(data)):
        new_data[:, i * data.shape[-1] : (i + 1) * data.shape[-1]] = data[i]
    return new_data

def get_est_arr(model, x_list, y_list, batch_size):
    for i in range(len(x_list)):
        x = x_list[i]
        y = y_list[i]
        
        # predict and concat
        num = x.shape[0] // batch_size
        if x.shape[0] % batch_size != 0:
            num += 1
        preds = []
        for j in range(num):
            # x: (batch_size, freq_bins, seg_len)
            if j == num - 1:
                X = x[j * batch_size:]
                length = x.shape[0] - j * batch_size
            else:
                X = x[j * batch_size: (j + 1) * batch_size]
                length = batch_size
            
            # for k in range(length): # normalization
            #     X[k] = std_normalize(X[k])
            prediction = model.predict(X, length)
            preds.append(prediction)
        
        # (num*bs, freq_bins, seg_len) to (freq_bins, T)
        preds = np.concatenate(preds, axis=0)
        preds = iseg(preds)
        # trnasform to f0ref
        CenFreq = get_CenFreq(StartFreq=31, StopFreq=1250, NumPerOct=60)
        # CenFreq = get_CenFreq(StartFreq=20, StopFreq=2048, NumPerOct=60)
        # CenFreq = get_CenFreq(StartFreq=81, StopFreq=600, NumPerOct=111)
        # CenFreq = get_CenFreq(StartFreq=81, StopFreq=600, NumPerOct=190)
        est_arr = est(preds, CenFreq, y)
        
    # VR, VFA, RPA, RCA, OA
    return est_arr

def get_pitch_track(model, file_in, file_out):
    xlist = []
    timestamps = []

    # feature = np.load(data_folder + 'cfp/' + fname + '.npy')
    print('CFP process in ' + str(file_in) + ' ... (It may take some times)')
    y, _ = load_audio(file_in, sr=8000)
    audio_len = len(y)
    batch_min = 8000*60*3
    freqs = []
    if len(y) > batch_min:
        iters = math.ceil(len(y)/batch_min)
        for i in np.arange(iters):
            if i < iters-1:
                audio_in = y[batch_min*i:batch_min*(i+1)]
            if i == iters-1:
                audio_in = y[batch_min*i:]
            # Getting feature for five min batch
            feature, _, time_arr = cfp_process(audio_in, sr=8000, hop=80)
            # Batching features for inference
            print('feature', np.shape(feature))
            data = batchize_test(feature, size=128)
            xlist.append(data)
            timestamps.append(time_arr)
            # Getting estimatted pitch
            estimation = get_est_arr(model, xlist, timestamps, batch_size=16)
            if i == 0:
                freqs = estimation[:, 1]
            else:
                freqs = np.concatenate((freqs, estimation[:, 1]))
    else:
        feature, _, time_arr = cfp_process(y, sr=8000, hop=80)
        # Batching features for inference
        print('feature', np.shape(feature))
        data = batchize_test(feature, size=128)
        xlist.append(data)
        timestamps.append(time_arr)
        # Getting estimatted pitch
        estimation = get_est_arr(model, xlist, timestamps, batch_size=16)
        freqs = estimation[:, 1]
        times = estimation[:, 0]

#    freqs = interpolate_below_length(freqs, 0.0, 0.25)
#    freqs_1 = gaussian_filter1d(freqs, sigma=1)
#    times = np.linspace(0, audio_len/8000, len(freqs))

    if file_out:
        save_pitch_track_to_dataset(
            file_out, times, freqs)
    else:
        print(np.shape(times), np.shape(freqs))


def save_pitch_track_to_dataset(filename, est_time, est_freq):
    # Write txt annotation to file
    with open(filename, 'w') as f:
        for i, j in zip(est_time, est_freq):
            f.write("{}, {}\n".format(i, j))
    print('Saved with exit to {}'.format(filename))

def load_ftanet(model_path):
    print('Loading model...')
    model = create_model(input_shape=IN_SHAPE)
    model.load_weights(
        filepath=model_path
    ).expect_partial()
    print('Model loaded!')
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='SCM-S')
    parser.add_argument('--file-in', default=None)
    parser.add_argument('--file-out', default=None)
    args = parser.parse_args()
    model_path = os.path.join(Path().absolute(), 'model', args.model, 'OA')
    model = load_ftanet(model_path)
    if args.file_in is not None:
        get_pitch_track(model, args.file_in, args.file_out)
    else:
        print('Please provide an input file to the --file-in argument')

