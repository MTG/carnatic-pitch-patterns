import glob
import os
import csv
import random
import pickle
import argparse
import time
from tqdm import tqdm
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.metrics import categorical_accuracy
from pathlib import Path

from constant import *
from generator import create_data_generator
from loader import load_data, load_data_for_test
from evaluator import evaluate

from network.ftanet import create_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

fp = 'PATH-TO-SCMS'
checkpoint_best_OA = os.path.join(Path().absolute(), 'model', 'SCM-S', 'OA')
dataset_filelist = glob.glob(fp + 'audio/*.wav')
with open(fp + 'artists_to_track_mapping.pkl', 'rb') as map_file:
    artists_to_track_mapping = pickle.load(map_file)

artists_to_train = [
    'Angarai V K Rajasimhan',  # Male
    'KP Nandini',  # Female
    'Vidya Subramanian',  # Female
    'Kuldeep Pai',  # Male
    'Salem Gayatri Venkatesan',  # Female
    'Vasundara Rajagopal',  # Female
    'Modhumudi Sudhakar',  # Male
    'Srividya Janakiraman'  # Female
]
# Get tracks to train
tracks_to_train = []
for artist in artists_to_train:
    tracks_to_train.append(artists_to_track_mapping[artist])

# Get filenames to train
training_files = []
validation_files = []
for files_for_artists in tracks_to_train:
#    files_for_artists = random.sample(artist, 87)
    files_to_validate = random.sample([x for x in artist if x not in files_for_artists], 3)
    complete_files_for_artists = [(fp + 'audio/' + track + '.wav') for track in files_for_artists]
    complete_files_to_validate = [(fp + 'audio/' + track + '.wav') for track in files_to_validate]
    training_files = training_files + complete_files_for_artists
    validation_files = validation_files + complete_files_to_validate

##--- 加载数据 ---##
# x: (n, freq_bins, time_frames, 3) extract from audio by cfp_process
# y: (n, freq_bins+1, time_frames) from ground-truth
print('Files to train: ', len(training_files))
print('Files to validate: ', len(validation_files))
train_x, train_y, train_num = load_data(
    track_list=training_files,
    seg_len=SEG_LEN
)
valid_x, valid_y = load_data_for_test(
    track_list=validation_files,
    seg_len=SEG_LEN
)

##--- Data Generator ---##
print('\nCreating generators...')
train_generator = create_data_generator(train_x, train_y, batch_size=BATCH_SIZE)

##--- 网络 ---##
print('\nCreating model...')

model = create_model(input_shape=IN_SHAPE)
model.compile(loss='binary_crossentropy', optimizer=(Adam(lr=LR)))
# model.summary()

##--- 开始训练 ---##
print('\nTaining...')
print('params={}'.format(model.count_params()))

epoch, iteration = 0, 0
best_OA, best_OA_SOTA, best_epoch, best_RPA, best_loss = 0, 0, 0, 0, 10000
best_RPA_eval_arr = np.array([0, 0, 0, 0, 0], dtype='float64')
best_loss_eval_arr = np.array([0, 0, 0, 0, 0], dtype='float64')
best_OA_eval_arr = np.array([0, 0, 0, 0, 0], dtype='float64')
best_OA_eval_arr_sota = np.array([0, 0, 0, 0, 0], dtype='float64')
mean_loss = 0
time_start = time.time()
while epoch < EPOCHS:
    iteration += 1
    print('Epoch {}/{} - {:3d}/{:3d}'.format(
        epoch+1, EPOCHS, iteration%(train_num//BATCH_SIZE), train_num//BATCH_SIZE), end='\r')
    # 取1个batch数据
    X, y = next(train_generator)
    # 训练1个iteration
    loss = model.train_on_batch(X, y)
    mean_loss += loss
    # 每个epoch输出信息
    if iteration % (train_num//BATCH_SIZE) == 0:
        # train meassage
        epoch += 1
        traintime = time.time() - time_start
        mean_loss /= train_num//BATCH_SIZE
        print('', end='\r')
        print('Epoch {}/{} - {:.1f}s - loss {:.4f}'.format(epoch, EPOCHS, traintime, mean_loss))
        # valid results
        avg_eval_arr = evaluate(model, valid_x, valid_y, BATCH_SIZE, cent_tolerance=25)
        avg_eval_arr_sota = evaluate(model, valid_x, valid_y, BATCH_SIZE, cent_tolerance=50)
        
        # save best OA model
        if avg_eval_arr_sota[-1] > best_OA:
            best_OA = avg_eval_arr_sota[-1]
            best_epoch = epoch
            best_OA_eval_arr = avg_eval_arr_sota
            model.save_weights(
                filepath=checkpoint_best_OA,
                overwrite=True,
                save_format='tf'
            )
            print('Saved to ' + checkpoint_best_OA)
        
        # save best OA model
        if avg_eval_arr_sota[-1] > best_OA_SOTA:
            best_OA_SOTA = avg_eval_arr_sota[-1]
            best_OA_eval_arr_sota = avg_eval_arr_sota

        # save best loss model
        if mean_loss <= best_loss:
            best_loss = mean_loss
            best_loss_eval_arr = avg_eval_arr
            print('Best loss detected!')

        # save best RPA model
        if avg_eval_arr[2] > best_RPA:
            best_RPA = avg_eval_arr[2]
            best_RPA_eval_arr = avg_eval_arr
            print('Best RPA detected!')
        
        print('ACTUAL VALIDATION:')
        print('VR {:.2f}% VFA {:.2f}% RPA {:.2f}% RCA {:.2f}% OA {:.2f}% BestOA {:.2f}%'.format(
            avg_eval_arr[0], avg_eval_arr[1], avg_eval_arr[2], avg_eval_arr[3], avg_eval_arr[4], best_OA))
        print('ACTUAL VALIDATION 50 tolerance:')
        print('VR {:.2f}% VFA {:.2f}% RPA {:.2f}% RCA {:.2f}% OA {:.2f}% BestOA {:.2f}%'.format(
            avg_eval_arr_sota[0], avg_eval_arr_sota[1], avg_eval_arr_sota[2], avg_eval_arr_sota[3], avg_eval_arr_sota[4], best_OA_SOTA))
        
        print('\nACTUAL BEST RPA:')
        print('VR {:.2f}% VFA {:.2f}% RPA {:.2f}% RCA {:.2f}% OA {:.2f}%'.format(
            best_RPA_eval_arr[0], best_RPA_eval_arr[1], best_RPA_eval_arr[2], best_RPA_eval_arr[3], best_RPA_eval_arr[4]))
        
        print('\nACTUAL BEST OA:')
        print('VR {:.2f}% VFA {:.2f}% RPA {:.2f}% RCA {:.2f}% OA {:.2f}%'.format(
            best_OA_eval_arr[0], best_OA_eval_arr[1], best_OA_eval_arr[2], best_OA_eval_arr[3], best_OA_eval_arr[4]))
        print('\nACTUAL BEST OA 50 tolerance:')
        print('VR {:.2f}% VFA {:.2f}% RPA {:.2f}% RCA {:.2f}% OA {:.2f}%'.format(
            best_OA_eval_arr_sota[0], best_OA_eval_arr_sota[1], best_OA_eval_arr_sota[2], best_OA_eval_arr_sota[3], best_OA_eval_arr_sota[4]))
        
        print('\nACTUAL BEST LOSS:')
        print('VR {:.2f}% VFA {:.2f}% RPA {:.2f}% RCA {:.2f}% OA {:.2f}%'.format(
            best_loss_eval_arr[0], best_loss_eval_arr[1], best_loss_eval_arr[2], best_loss_eval_arr[3], best_loss_eval_arr[4]))
        # early stopping
        if epoch - best_epoch >= PATIENCE:
            print('Early stopping with best OA {:.2f}%'.format(best_OA))
            break
            
        # initialization
        mean_loss = 0
        time_start = time.time()
