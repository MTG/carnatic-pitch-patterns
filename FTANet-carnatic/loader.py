import os
import csv

import scipy.signal
from tqdm import tqdm
import pickle
import librosa
import mir_eval
import numpy as np
import pandas as pd
from scipy import signal
from cfp import cfp_process


def get_CenFreq(StartFreq=80, StopFreq=1000, NumPerOct=48):
    Nest = int(np.ceil(np.log2(StopFreq/StartFreq))*NumPerOct)
    central_freq = []
    for i in range(0, Nest):
        CenFreq = StartFreq*pow(2, float(i)/NumPerOct)
        if CenFreq < StopFreq:
            central_freq.append(CenFreq)
        else:
            break
    return central_freq


def seq2map(seq, CenFreq):
    CenFreq[0] = 0
    gtmap = np.zeros((len(CenFreq), len(seq)))
    for i in range(len(seq)):
        for j in range(len(CenFreq)):
            if seq[i] < 0.1:
                gtmap[0, i] = 1
                break
            elif CenFreq[j] > seq[i]:
                gtmap[j, i] = 1
                break
    return gtmap


def batchize(data, gt, xlist, ylist, size=430):
    if data.shape[-1] != gt.shape[-1]:
        new_length = min(data.shape[-1], gt.shape[-1])
        print('data:', data.shape, ', gt shape:', gt.shape)

        data = data[:, :, :new_length]
        gt = gt[:, :new_length]
    num = int(gt.shape[-1] / size)
    if gt.shape[-1] % size != 0:
        num += 1
    for i in range(num):
        if (i + 1) * size > gt.shape[-1]:
            batch_x = np.zeros((data.shape[0], data.shape[1], size))
            batch_y = np.zeros((gt.shape[0], size))

            tmp_x = data[:, :, i * size:]
            tmp_y = gt[:, i * size:]

            batch_x[:, :, :tmp_x.shape[-1]] += tmp_x
            batch_y[:, :tmp_y.shape[-1]] += tmp_y
            xlist.append(batch_x.transpose(1, 2, 0))
            ylist.append(batch_y)
            break
        else:
            batch_x = data[:, :, i * size:(i + 1) * size]
            batch_y = gt[:, i * size:(i + 1) * size]
            xlist.append(batch_x.transpose(1, 2, 0))
            ylist.append(batch_y)

    return xlist, ylist, num


def batchize_test(data, size=430):
    xlist = []
    num = int(data.shape[-1] / size)
    if data.shape[-1] % size != 0:
        num += 1
    for i in range(num):
        if (i + 1) * size > data.shape[-1]:
            batch_x = np.zeros((data.shape[0], data.shape[1], size))

            tmp_x = data[:, :, i * size:]

            batch_x[:, :, :tmp_x.shape[-1]] += tmp_x
            xlist.append(batch_x.transpose(1, 2, 0))
            break
        else:
            batch_x = data[:, :, i * size:(i + 1) * size]
            xlist.append(batch_x.transpose(1, 2, 0))

    return np.array(xlist)
    

def load_data(track_list, seg_len=430):
    
    xlist = []
    ylist = []
    for wav_file in tqdm(track_list):

        ## Load cfp features (3, 320, T)
        #feature = np.load(data_folder + 'cfp/' + fname + '.npy')
        feature, CenFreq, time_arr = cfp_process(wav_file, sr=8000, hop=80)
#        print('feature', np.shape(feature))

        ## Load f0 frequency
        #pitch = np.loadtxt(data_folder + 'f0ref/' + fname + '.txt')
        '''
        if 'medley' in wav_file:
            ref_arr = csv2ref(wav_file.replace('.wav', 'REF.csv'))
        else:
            if 'train' in wav_file:
                ref_arr = txt2ref_tabs(wav_file.replace('.wav', 'REF.txt'))
            else:
                ref_arr = txt2ref_spaces(wav_file.replace('.wav', 'REF.txt'))
        '''
        ref_arr = csv2ref(wav_file.replace('.wav', '.csv').replace('audio/', 'annotations/melody/'))
        _, pitch_res = resample_melody(ref_arr, np.shape(feature)[-1])
#        print('pitch', np.shape(pitch_res))
        mapping = seq2map(pitch_res, CenFreq)  # (321, T)
        ## Crop to segments
        xlist, ylist, num = batchize(feature, mapping, xlist, ylist, size=seg_len)

    return xlist, ylist, len(ylist)


def load_data_for_test(track_list, seg_len=430):

    xlist = []
    ylist = []
    for wav_file in tqdm(track_list):
        
        ## Load cfp features (3, 320, T)
        # feature = np.load(data_folder + 'cfp/' + fname + '.npy')
        feature, _, time_arr = cfp_process(wav_file, sr=8000, hop=80)
#        print('feature', np.shape(feature))

        ## Load f0 frequency
        # pitch = np.loadtxt(data_folder + 'f0ref/' + fname + '.txt')
        '''
        if 'medley' in wav_file:
            ref_arr = csv2ref(wav_file.replace('.wav', 'REF.csv'))
        else:
            if 'train' in wav_file:
                ref_arr = txt2ref_tabs(wav_file.replace('.wav', 'REF.txt'))
            else:
                ref_arr = txt2ref_spaces(wav_file.replace('.wav', 'REF.txt'))
        '''
        ref_arr = csv2ref(wav_file.replace('.wav', '.csv').replace('audio/', 'annotation/melody/'))
        times, pitch = resample_melody(ref_arr, np.shape(feature)[-1])
        ref_arr_res = np.concatenate((times[:, None], pitch[:, None]), axis=1)
#        print('pitch', np.shape(ref_arr_res))

        data = batchize_test(feature, size=seg_len)
        xlist.append(data)
        ylist.append(ref_arr_res[:, :])

    return xlist, ylist


def load_single_data_for_test(fname, seg_len=430):
    # data_file = 'data/single_' + fname + '_{}test.pkl'.format(seg_len)
    # if os.path.exists(data_file):
    #     with open(data_file, 'rb') as f:
    #         xlist, ylist = pickle.load(f)
    
    # else:
    data_folder = '/data1/project/MCDNN/data/'
    
    xlist = []
    ylist = []

    ## Get file key
    fname = fname.replace('.npy', '').rstrip()

    ## Load cfp features
    feature = np.load(data_folder + 'cfp/' + fname + '.npy') # (3, 320, T)

    ## Load f0 frequency
    ref_arr = np.loadtxt(data_folder + 'f0ref/' + fname + '.txt')  # (T, 2)

    data = batchize_test(feature, seg_len)
    xlist.append(data)
    ylist.append(ref_arr[:, :])

    # dataset = (xlist, ylist)
    # with open(data_file, 'wb') as f:
    #     pickle.dump(dataset, f)
    #     print("Saved {} segments to {}".format(len(xlist), data_file))
        
    return xlist, ylist


def txt_to_pitch(ypath):
    pitch = []
    with open(ypath, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=',')
        for row in reader:
            pitch.append(float(row[1].replace(' ', '')))
        
        return np.array(pitch)


def txt2ref_spaces(ypath):
    times = []
    pitch = []
    with open(ypath, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=' ')
        for row in reader:
            times.append(float(row[0]))
            pitch.append(float(row[-1]))
            
        times = np.array(times)
        pitch = np.array(pitch)
        ref_arr = np.concatenate((times[:, None], pitch[:, None]), axis=1)
        
        return ref_arr


def txt2ref_tabs(ypath):
    times = []
    pitch = []
    with open(ypath, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter='\t')
        for row in reader:
            times.append(float(row[0]))
            pitch.append(float(row[1]))
        
        times = np.array(times)
        pitch = np.array(pitch)
        ref_arr = np.concatenate((times[:, None], pitch[:, None]), axis=1)
        
        return ref_arr
    
    
def csv2ref(ypath):
    ycsv = pd.read_csv(ypath, names=["time", "freq"])
    gtt = ycsv['time'].values
    gtf = ycsv['freq'].values
    ref_arr = np.concatenate((gtt[:, None], gtf[:, None]), axis=1)
    return ref_arr


def resample_melody(pitch, new_len):
    times = pitch[:, 0]
    frequencies = pitch[:, 1]
    #frequencies = [0 if p < 100 else p for p in frequencies]
    
    voicing = []
    for freq in frequencies:
        voicing.append(1) if freq > 0 else voicing.append(0)
    
    #times_new = signal.resample(times, new_len)
    times_new = np.linspace(times[0], times[-1], num=new_len)
    
    frequencies_resampled, voicing_resampled = mir_eval.melody.resample_melody_series(
        times=times,
        frequencies=frequencies,
        voicing=np.array(voicing),
        times_new=times_new,
        kind='linear'
    )
    
    return times_new, frequencies_resampled
