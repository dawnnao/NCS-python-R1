
# coding: utf-8

# # Implementing compressive sensing in a neural network with group sparsity optimization

# In[ ]:

# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 13:43:49 2018

@author: zhiyitang
"""
# %reset

# import os
# os.system('cls')

import numpy as np
import tensorflow as tf
import scipy.io
import hdf5storage
import math
import pandas as pd
from scipy.linalg import dft
from sklearn.preprocessing import normalize
from keras import layers
import keras
from keras.layers import Input, Dense, Activation, Multiply, Add, Subtract, Conv2D, Lambda, Reshape
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD, RMSprop
from keras import regularizers
from keras.constraints import Constraint
from keras import backend as K
from keras import losses

import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'  # Windows
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model
from keras.preprocessing import image
from matplotlib.pyplot import imshow

import pickle
import time

from skimage.util.shape import view_as_windows
from keras.callbacks import LearningRateScheduler

import subprocess

from keras.callbacks import ModelCheckpoint
import h5py
import time
import seaborn as sns

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')

#%% User input starts

#%% earthquake event 1
event_name = 'event_1'
# data_path = "C:/dataArchive/data_special_events/%s/data_full_columns/" % event_name  # XPS
data_path = 'D:/data_special_events/%s/quanzhouwan/data_full_columns/' % event_name  # Box

data_name = ['VIB']
# data_name = ['DPM', 'GPS', 'HPT', 'TLT', 'VIB', 'VIC', 'VIF']

tail = '.mat'

# channel = 'all'  # start with 0
# channel = np.array([1, 2, 3, 4, 8, 9, 11, 16]) - 1
channel = np.array([1, 2, 3, 4, 12, 46, 47, 48, 49]) - 1

# bad_channel = np.array([1])  # !!! index from 0 to len(channel)-1
# bad_sample_ratio = np.array([0.7])

# sample_ratio = np.array([0.75, 0.85, 0.95])  # input sampling ratio here
sample_ratio = np.array([0.7, 0.75, 0.8, 0.85, 0.9, 0.95])  # input sampling ratio here
# sample_ratio = np.array([0.5, 0.6, 0.7, 0.8, 0.9])  # input sampling ratio here
# sample_ratio = np.array([0.7, 0.95, 0.9, 0.85, 0.8, 0.75, 0.65, 0.6, 0.55, 0.5])  # input sampling ratio here
# sample_ratio = np.array([0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05])  # input sampling ratio here
# sample_ratio = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])  # input sampling ratio here

packet = np.array([25])  # number of point in each packet
# packet = np.array([25, 50])  # number of point in each packet

randseed = np.arange(1, 5)  # seeds for random number

regularizer_weight = np.array([10.0])
# regularizer_weight = np.array([1.0, 0.01, 0.05, 0.1, 5.0, 10.0])

batch_size = 128

epochs = 2400

# harmonic_wavelet_interval = np.array([1, 2, 4, 8, 16, 32])
harmonic_wavelet_interval = np.array([2])

loss_weight_rate = 24


#%% typhoon
# event_name = 'event_6'
# data_path = 'D:/data_special_events/%s/quanzhouwan/data_full_columns/' % event_name
#
# data_name = ['VIB']
# # data_name = ['DPM', 'GPS', 'HPT', 'TLT', 'VIB', 'VIC', 'VIF']
#
# tail = '.mat'
#
# # channel = 'all'  # start with 0
# # channel = np.array([1, 2, 3, 4, 8, 9, 11, 16]) - 1
# channel = np.array([1, 2, 3, 4, 12, 46, 47, 48, 49]) - 1
#
# # bad_channel = np.array([0])
# # bad_sample_ratio = np.array([0.1])
#
# sample_ratio = np.array([0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05])  # input sampling ratio here
# # sample_ratio = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])  # input sampling ratio here
#
# packet = 1  # number of point in each packet
#
# randseed = np.arange(0, 1)  # seeds for random number


#%% toy example
# event_name = 'toy'
# data_path = 'D:/data_special_events/%s/data_full_columns/' % event_name
#
# data_name = ['toy_data6']  # ['U%d' % u for u in range(1, 10)]
#
# tail = '.mat'
#
# # duration = 2048
# # overlap = 512
#
# channel = 'all'  # start with 0
#
# sample_ratio = np.array([0.3])
#
# packet = 1  # number of point in each packet
#
# randseed = np.arange(0, 1)  # seeds for random number


#%% simulation
# event_name = 'simulation_impulse'
# # event_name = 'simulation_El_Centro'
# # data_path = "C:/dataArchive/data_special_events/%s/data_full_columns/" % event_name  # XPS
# data_path = "D:/data_special_events/%s/data_full_columns/" % event_name  # Box
#
# data_name = ['DPM']
# # data_name = ['DPM', 'GPS', 'HPT', 'TLT', 'VIB', 'VIC', 'VIF']
#
# tail = '.mat'
#
# channel = 'all'  # start with 0
#
# # bad_channel = np.array([0])
# # bad_sample_ratio = np.array([0.1])
#
# sample_ratio = np.array([0.5, 0.6])  # input sampling ratio here
# # sample_ratio = np.array([0.5, 0.6, 0.7, 0.8, 0.9])  # input sampling ratio here
# # sample_ratio = np.array([0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05])  # input sampling ratio here
# # sample_ratio = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])  # input sampling ratio here
#
# packet = np.array([100, 50])  # number of point in each packet
# # packet = np.array([10, 50, 100])  # number of point in each packet
#
# randseed = np.arange(0, 1)  # seeds for random number
#
# regularizer_weight = np.array([10.0])
#
# batch_size = 128
#
# epochs = 1200  # 2400  # 1200
#
# # harmonic_wavelet_interval = np.array([1, 2, 4, 8, 16, 32])
# # harmonic_wavelet_interval = np.array([64, 128, 256, 512])
# harmonic_wavelet_interval = np.array([2])
#
# loss_weight_rate = 24

#%% haicang
# event_name = 'event_haicang'
# # data_path = "C:/dataArchive/data_special_events/%s/data_full_columns/" % event_name  # XPS
# data_path = "D:/data_special_events/%s/data_full_columns/" % event_name  # Box
# # data_path = "C:/dataArchive/haicang/程序整理/文中模态识别、数据压缩恢复使用的数据 - tidy/"  # Box / PC / XPS
# # data_path = "/Users/zhiyitang/dataArchive/haicang/程序整理/文中模态识别、数据压缩恢复使用的数据 - tidy/"  # Mac
#
# data_name = ['U%d' % u for u in range(1, 5)]  # ['U%d' % u for u in range(1, 10)]
# # data_name = ['toy_data5']  # ['U%d' % u for u in range(1, 10)]
#
# tail = '.mat'
#
# # duration = 2048
# # overlap = 512
#
# channel = 'all'  # start with 0
# # channel = np.array([0])
#
# # bad_channel = np.array([0])
# # bad_sample_ratio = np.array([0.05])
#
# sample_ratio = np.array([0.5, 0.6, 0.7, 0.8, 0.9])  # input sampling ratio here
# # sample_ratio = np.array([0.7, 0.75, 0.8, 0.85, 0.9, 0.95])  # input sampling ratio here
# # sample_ratio = np.array([0.7])
# # sample_ratio = np.array([0.9, 0.8, 0.7, 0.6, 0.5])  # input sampling ratio here
# # sample_ratio = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])  # input sampling ratio here
#
# packet = np.array([1, 25])  # number of point in each packet
#
# randseed = np.arange(0, 1)  # seeds for random number
#
# regularizer_weight = np.array([10.0])
#
# batch_size = 128
#
# epochs = 2400  # 1200
#
# # harmonic_wavelet_interval = np.array([1, 2, 4, 8, 16, 32])
# # harmonic_wavelet_interval = np.array([64, 128, 256, 512])
# harmonic_wavelet_interval = np.array([1])
#
# loss_weight_rate = 24


#%% sutong
# event_name = 'event_9'
# data_path = 'D:/data_special_events/%s/sutong/data_full_columns/' % event_name
#
# data_name = ['RSG']
# # data_name = ['DPM', 'GPS', 'HPT', 'TLT', 'VIB', 'VIC', 'VIF', 'RSG']
#
# tail = '.mat'
#
# channel = 'all'  # start with 0
# # channel = np.array([5, 7, 9, 11, 13, 15, 19, 21, 23]) - 1
#
# # bad_channel = np.array([0])
# # bad_sample_ratio = np.array([0.1])
#
# sample_ratio = np.array([0.7])  # input sampling ratio here
# # sample_ratio = np.array([0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05])  # input sampling ratio here
# # sample_ratio = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])  # input sampling ratio here
#
# packet = 20  # number of point in each packet
#
# randseed = np.arange(0, 1)  # seeds for random number
#
# regularizer_weight = np.array([1.0])
# # regularizer_weight = np.array([1.0, 0.01, 0.05, 0.1, 5.0, 10.0])
#
# batch_size = 128
#
# epochs = 1200
#
# harmonic_wavelet_interval = np.array([16])
# # harmonic_wavelet_interval = np.array([1, 2, 4, 8, 16])
# # harmonic_wavelet_interval = np.array([32, 64, 128, 256, 512])

#%%

def ncs(data_path, data_name, tail, fs, duration, overlap, channel, sample_ratio, result_path, randseed,
        bad_channel=None, bad_sample_ratio=None, packet=1, regularizer_weight=1.0, batch_size=128, epochs=1200,
        harmonic_wavelet_interval=16, loss_weight_rate=32):

    # %% 1 Load data and prepare file and folder names

    # data_raw = scipy.io.loadmat(data_path+data_name+tail)
    data_raw = hdf5storage.loadmat(data_path+data_name+tail)
    data_all = data_raw[data_name]
    del data_raw

    def abbr(vec):
        line = np.full([np.max(vec) + 2], np.nan)
        line[vec] = vec
        edge = []
        if not np.isnan(line[0]):
            edge.append(0)
        for n in np.arange(1, len(line)):
            if not np.isnan(line[n]) and np.isnan(line[n - 1]):
                edge.append(n)
            elif np.isnan(line[n]) and not np.isnan(line[n - 1]):
                edge.append(n - 1)
        edge = np.array(edge).reshape(np.int(len(edge) / 2), 2).T
        return edge

    def tidy_name(edge):
        name_str = []
        for n in np.arange(0, edge.shape[1]):
            if edge[1, n] == edge[0, n]:
                name_str.append('%d' % edge[0, n])
            elif edge[1, n] == edge[0, n] + 1:
                name_str.append('%d,%d' % (edge[0, n], edge[1, n]))
            elif edge[1, n] > edge[0, n] + 1:
                name_str.append('%d-%d' % (edge[0, n], edge[1, n]))
        name_str = '_'.join(name_str)
        return name_str

    if channel == 'all':
        channel = np.arange(0, data_all.shape[1])

    channel_num = len(channel)
    channel_str_abbr = tidy_name(abbr(channel))

    if bad_channel is None:

        result_folder = data_name + '_' + str(duration) + '_' + str(packet) + '_[' + channel_str_abbr + ']_' + \
                        '%.2f' % sample_ratio + '_' + str(randseed) + \
                        '_rw_' + str(regularizer_weight) + '_bs_' + str(batch_size) + '_epo_' + str(epochs) + \
                        '_hw_interval_' + str(hw_interval) + \
                        '_lw_rate_' + '%03d' % loss_weight_rate + '/'

        result_file = data_name + '_' + str(duration) + '_' + str(packet) + '_[' + channel_str_abbr + ']_' + \
                      '%.2f' % sample_ratio + '_' + str(randseed) + \
                      '_rw_' + str(regularizer_weight) + '_bs_' + str(batch_size) + '_epo_' + str(epochs) + \
                      '_hw_interval_' + str(hw_interval) + \
                      '_lw_rate_' + '%03d' % loss_weight_rate
    else:
        bad_channel_str = [str(i) for i in bad_channel]
        bad_channel_str_stack = '_'.join(bad_channel_str)
        bad_sample_ratio_str = ['%.2f' % i for i in bad_sample_ratio]
        bad_sample_ratio_str_stack = '_'.join(bad_sample_ratio_str)

        result_folder = data_name + '_' + str(duration) + '_' + str(packet) + '_[' + channel_str_abbr + ']_' + \
                        '%.2f' % sample_ratio + '_' + str(randseed) + \
                        '_rw_' + str(regularizer_weight) + '_bs_' + str(batch_size) + '_epo_' + str(epochs) + \
                        '_hw_interval_' + str(hw_interval) + \
                        '_lw_rate_' + '%03d' % loss_weight_rate + \
                        '__[' + bad_channel_str_stack + ']_' + bad_sample_ratio_str_stack + '/'

        result_file = data_name + '_' + str(duration) + '_' + str(packet) + '_[' + channel_str_abbr + ']_' + \
                      '%.2f' % sample_ratio + '_' + str(randseed) + \
                      '_rw_' + str(regularizer_weight) + '_bs_' + str(batch_size) + '_epo_' + str(epochs) + \
                      '_hw_interval_' + str(hw_interval) + \
                      '_lw_rate_' + '%03d' % loss_weight_rate + \
                      '__[' + bad_channel_str_stack + ']_' + bad_sample_ratio_str_stack

    def create_folder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)

    create_folder(result_path + result_folder)

    channel_str = ['ch %d' %(i+1) for i in channel]

    #%% 2 Preprocess data

    data_all = data_all[:, channel]
    data_split = view_as_windows(np.ascontiguousarray(data_all), (duration, data_all.shape[1]), duration-overlap)
    data_split = np.squeeze(data_split, axis=1)
    slice_num = data_split.shape[0]  # number of sections, split by duration
    print('Shape of data_split: ', data_split.shape)

    # data_split = data_split[4:6, :, :]  # duration = 4096 !!! 20190420 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    data_split_offset = np.mean(data_split, axis=1)

    # data_split_norm = np.linalg.norm(data_split, axis=, ord=inf)
    # data_split_norm = np.amax(np.abs(data_split - data_split_offset), axis=1)

    data_split_norm = np.zeros(data_split_offset.shape)
    data_split_normalized = np.zeros(data_split.shape)

    for slice in range(slice_num):

        # print('data_split[slice] type', type(data_split[slice]), '\n')
        # print('data_split[slice] shape', data_split[slice].shape, '\n')
        #
        # print('data_split_norm[slice].reshape((1, len(channel))) type', type(data_split_norm[slice].reshape((1, len(channel)))), '\n')
        # print('data_split_norm[slice].reshape((1, len(channel))) shape', data_split_norm[slice].reshape((1, len(channel))).shape, '\n')

        # print(data_split_norm[slice])

        # data_split_temp = data_split[slice]
        # data_split_temp = data_split_temp / data_split_norm[slice]
        # data_split[slice] = data_split_temp

        # data_split[slice] = data_split[slice] / data_split_norm[slice]

        data_split_norm[slice] = np.amax(np.abs(data_split[slice] -
                                                data_split_offset[slice].reshape((1, channel_num))), axis=0)

        data_split_normalized[slice] = np.true_divide(data_split[slice] -
                                                      data_split_offset[slice].reshape((1, channel_num)),
                                                      data_split_norm[slice].reshape((1, channel_num)))

    # data_all_divisible = data_all[0:data_all.shape[0]-np.mod(data_all.shape[0], duration), :]  # ignore the remainder
    # print('Shape of data_all_divisible: ' + str(data_all_divisible.shape))
    # data_split = np.array(np.split(data_all_divisible, slice_num))

    data_hat = np.zeros(data_split.shape, dtype=np.complex128)
    data_masked_ignore_zero_all = np.zeros(data_split.shape)
    mask_matrix_all = np.zeros(data_split.shape)
    weights_complex = np.zeros(data_split.shape, dtype=np.complex128)
    recon_error_time_domain = np.zeros([slice_num, len(channel)])
    recon_error_freq_domain = np.zeros([slice_num, len(channel)])

    for slice in range(slice_num):

        start_time = time.time()

        data_time = data_split_normalized[slice]
        print('shape of data:\n', data_time.shape)
        dt = 1./fs
        t = np.arange(0., duration/fs, dt)

        for f in range(len(channel)):
            fig = plt.figure(figsize=(18,4))
            plt.plot(t, data_time[:,f])
            ax = plt.gca()
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            plt.legend([channel_str[f]], loc=1, fontsize=12)
            plt.xlabel('Time (s)', fontsize=12)

            def data_unit(data_name):
                return {
                    'DPM': 'Displ. (mm)',
                    'GPS': 'Displ. (mm)',
                    'HPT': 'Displ. (mm)',
                    'RHS': 'Tempr. (degree Celsius)',
                    'TLT': 'Tilt (degree)',
                    'UAN': 'Velocity (m/s)',
                    'ULT': ' ',
                    'VIB': 'Accel. (gal)',
                    'VIC': 'Accel. (gal)',
                    'VIF': 'Accel. (gal)',
                    'U1': 'Velocity (mm/s)',
                    'U2': 'Velocity (mm/s)',
                    'U3': 'Velocity (mm/s)',
                    'U4': 'Velocity (mm/s)',
                    'U5': 'Velocity (mm/s)',
                    'U6': 'Velocity (mm/s)',
                    'U7': 'Velocity (mm/s)',
                    'U8': 'Velocity (mm/s)',
                    'U9': 'Velocity (mm/s)'
                }.get(data_name, ' ')  # default if x not found

            plt.ylabel(data_unit(data_name), fontsize=12)
            plt.xlim(0, max(t))
            # plt.xlim(2, 3)  # for toy_data
            # plt.show()
            plt.tight_layout()

            result_folder_signal = 'signal_%02d/' % (channel[f]+1)
            create_folder(result_path + result_folder + result_folder_signal)
            fig.savefig(result_path + result_folder + result_folder_signal + 'signal_%d_slice-%02d.png' % (channel[f]+1, slice + 1))
            plt.close()
            time.sleep(0.1)

        fig = plt.figure(figsize=(18,4))
        plt.plot(t, data_time)
        # matplotlib.rcParams.update({'font.size': 12})
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        # matplotlib.rc('xtick', labelsize=12)
        # matplotlib.rc('ytick', labelsize=12)
        plt.legend(channel_str, loc=1, fontsize=12)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel(data_unit(data_name), fontsize=12)
        plt.xlim(0, max(t))
        # plt.xlim(2, 3)  # for toy_data
        plt.tight_layout()
        # plt.show()
        fig.savefig(result_path + result_folder + 'signal_all_slice-%02d.png' % (slice+1))
        time.sleep(0.1)

        plt.close('all')

        #%% 3 Generate mask matrix

        np.random.seed(randseed + slice)  # generate random seed

        if packet > 1:
            remain = np.mod(duration, packet)
            if bad_channel is None:
                mask_matrix_condensed = np.random.choice([0, 1], size=(data_time.shape[0] // packet, data_time.shape[1]),
                                                         p=[1 - sample_ratio, sample_ratio])
                mask_matrix = np.vstack((np.repeat(mask_matrix_condensed, packet, axis=0), np.ones((remain, data_time.shape[1]))))
            else:
                mask_matrix_condensed = np.random.choice([0, 1], size=(data_time.shape[0] // packet, data_time.shape[1]),
                                                         p=[1 - sample_ratio, sample_ratio])
                for b in range(len(bad_channel)):
                    mask_col = np.random.choice([0, 1], size=(data_time.shape[0] // packet),
                                                p=[1 - bad_sample_ratio[b], bad_sample_ratio[b]])
                    mask_matrix_condensed[:, bad_channel[b]] = mask_col
                mask_matrix = np.vstack((np.repeat(mask_matrix_condensed, packet, axis=0), np.ones((remain, data_time.shape[1]))))
        else:
            if bad_channel is None:
                mask_matrix = np.random.choice([0, 1], size=data_time.shape, p=[1 - sample_ratio, sample_ratio])
            else:
                mask_matrix = np.random.choice([0, 1], size=data_time.shape, p=[1 - sample_ratio, sample_ratio])
                for b in range(len(bad_channel)):
                    mask_col = np.random.choice([0, 1], size=(data_time.shape[0]),
                                                p=[1 - bad_sample_ratio[b], bad_sample_ratio[b]])
                    mask_matrix[:, bad_channel[b]] = mask_col

        mask_matrix_all[slice] = mask_matrix
        # print('mask_matrix:\n', mask_matrix, '\n')
        print('shape of mask_matrix:', mask_matrix.shape, '\n')

        #%% 4 Acquire masked data

        data_masked = data_time * mask_matrix # target of network
        # print('data_masked:\n', data_masked, '\n')

        # print(data_masked[3,:])

        # for f in range(len(channel)):
        #     plt.figure(figsize=(18,4))
        #     plt.plot(t, data_masked[:,f])
        #     plt.legend([channel_str[f]], loc=1)
        #     plt.xlabel('Time (s)')
        #     plt.ylabel('Velocity (mm/s)')
        #     # plt.show()
        #     plt.close()
        #
        # plt.figure(figsize=(18,4))
        # plt.plot(t, data_masked)
        # plt.legend(channel_str, loc=1)
        # plt.xlabel('Time (s)')
        # plt.ylabel('acceleration (cm/s/s)')
        # # plt.show()
        #
        # plt.close('all')

        #%% 5 Generate basis matrix

        # Fourier matrix:

        def dft2d(n):
            m = dft(n)
        #     m = normalize(m, axis=0)
        #     m = np.real(m)
            return m

        def harmonic_wavelet(length, m, n):
            x = np.zeros((length, length), dtype='complex')
            interval = n - m
            piece_num = length / interval
            c = 0
            if piece_num % 1 != 0:
                raise Exception('Given length can not be exact divided by n-m.')
            else:
                piece_num = int(piece_num)
                for k in range(piece_num):
                    w_raw = np.zeros(length, dtype='complex')
                    w_raw[k * interval: (k + 1) * interval] = 1 + 0j
                    x_raw = np.fft.ifft(w_raw)
                    x_raw_normalized = x_raw / np.max(np.abs(x_raw))
                    for p in range(interval):
                        x_mid = np.roll(x_raw_normalized, p * piece_num)
                        x[:, c] = x_mid
                        c += 1
            return x

        basis_folder = 'basis_matrix/'
        basis_file = 'basis_' + str(duration) + '.npy'
        # basis_file = 'basis_' + str(duration) + '.pickle'

        # if os.path.exists(result_path + basis_folder + basis_file):
        #     print('\nMapping basis matrix...\n')
        #
        #     # basis_matrix = np.memmap(result_path + basis_folder + basis_file, dtype='complex', mode='r',
        #     #                          shape=(duration, duration))
        #     basis_matrix = np.load(result_path + basis_folder + basis_file, mmap_mode="r")
        #     # with open(result_path + basis_folder + basis_file, 'rb') as f:
        #     #     basis_matrix = pickle.load(f)
        #     # f.close()
        #
        #     # for d in range(0, duration, 1):
        #     #     fig = plt.figure(figsize=(7, 4))
        #     #     plt.plot(np.real(basis_matrix[:, d]))
        #     #     plt.plot(np.imag(basis_matrix[:, d]))
        #     #     plt.legend(['Real part', 'Imaginary part'])
        #     #     plt.title('Col %d' % d)
        #     #     plt.show()
        #     #     input("Press Enter to continue...")
        #     #     plt.close()
        #
        # else:
        #     print('\nGenerating basis matrix...\n')
        #     m = 0  # 128
        #     n = hw_interval  # 256
        #     basis_matrix = harmonic_wavelet(duration, m, n)
        #     # basis_matrix = dft2d(duration)
        #     create_folder(result_path + basis_folder)
        #
        #     print('\nSaving basis matrix...\n')
        #     np.save(result_path + basis_folder + basis_file, basis_matrix)
        #     # with open(result_path + basis_folder + basis_file, 'wb') as f:
        #     #     pickle.dump(basis_matrix, f, protocol=4)
        #     # f.close()
        #     print('\nBasis matrix saved at:\n' + basis_folder + basis_file)
        #
        #     # for d in range(0, duration, 1):
        #     #     fig = plt.figure(figsize=(7, 4))
        #     #     plt.plot(np.real(basis_matrix[:, d]))
        #     #     plt.plot(np.imag(basis_matrix[:, d]))
        #     #     plt.legend(['Real part', 'Imaginary part'])
        #     #     plt.title('Col %d' % d)
        #     #     plt.show()
        #     #     input("Press Enter to continue...")
        #     #     plt.close()

        print('\nGenerating basis matrix...\n')
        m = 0  # 128
        n = hw_interval  # 256
        basis_matrix = harmonic_wavelet(duration, m, n)
        # basis_matrix = dft2d(duration)
        create_folder(result_path + basis_folder)

        # print('\nSaving basis matrix...\n')
        # np.save(result_path + basis_folder + basis_file, basis_matrix)
        # print('\nBasis matrix saved at:\n' + basis_folder + basis_file)


        print('\nShape of basis_matrix:\n', basis_matrix.shape, '\n')
        print('\nBasis_matrix:\n', basis_matrix, '\n')

        # print(np.dot(matrix_h, matrix_h.T))

        # check basis:

        # plt.figure()
        # plt.plot(np.real(basis_matrix[:,1]))
        # plt.plot(np.imag(basis_matrix[:,1]))
        # plt.legend(['real', 'imag'])
        # # plt.show()
        # plt.close()

        #%% 6 Construct neural network

        data_input1 = np.real(basis_matrix)
        data_input2 = np.imag(basis_matrix)
        data_input3 = mask_matrix
        data_output1 = np.real(data_masked.astype(complex))
        data_output2 = np.imag(data_masked.astype(complex))

        # print('shape of data_input1:\n', data_input1.shape, '\n')
        # print('shape of data_input2:\n', data_input2.shape, '\n')
        # print('shape of data_input3:\n', data_input3.shape, '\n')
        # print('shape of data_output1:\n', data_output1.shape, '\n')
        # print('shape of data_output2:\n', data_output2.shape, '\n')
        #
        # print('type of data_input1:\n', type(data_input1), '\n')
        # print('type of data_input2:\n', type(data_input2), '\n')
        # print('type of data_input3:\n', type(data_input3), '\n')
        # print('type of data_output1:\n', type(data_output1), '\n')
        # print('type of data_output2:\n', type(data_output2), '\n')
        #
        # print('dtype of data_input1:\n', data_input1.dtype, '\n')
        # print('dtype of data_input2:\n', data_input2.dtype, '\n')
        # print('dtype of data_input3:\n', data_input3.dtype, '\n')
        # print('dtype of data_output1:\n', data_output1.dtype, '\n')
        # print('dtype of data_output2:\n', data_output2.dtype, '\n')
        #
        # print('data_input1:\n', data_input1)
        # print('data_input2:\n', data_input2)
        # print('data_input3:\n', data_input3)
        # print('data_output1:\n', data_output1)
        # print('data_output2:\n', data_output2)

        # Expand input dimension for CNN
        data_input1 = np.expand_dims(data_input1, 0)
        data_input1 = np.expand_dims(data_input1, -1)

        data_input2 = np.expand_dims(data_input2, 0)
        data_input2 = np.expand_dims(data_input2, -1)

        data_input3 = np.expand_dims(data_input3, 0)
        data_input3 = np.expand_dims(data_input3, -1)

        data_output1 = np.expand_dims(data_output1, 0)
        data_output1 = np.expand_dims(data_output1, -1)

        data_output2 = np.expand_dims(data_output2, 0)
        data_output2 = np.expand_dims(data_output2, -1)

        np.random.seed(randseed)
        tf.set_random_seed(randseed)

        class Regularizer(object):
            """Regularizer base class.
            """

            def __call__(self, x):
                return 0.

            @classmethod
            def from_config(cls, config):
                return cls(**config)

        class GroupSparseRegularizer(Regularizer):

            def __init__(self, regularizer_weight):
                self.regularizer_weight = regularizer_weight

            def __call__(self, weight_matrix):
                # return self.regularizer_weight * tf.norm(tf.norm(weight_matrix, ord='euclidean', axis=1),
                #                                          ord='euclidean')  # L-2 norm
                return self.regularizer_weight * K.sum(tf.norm(weight_matrix,
                                                               ord='euclidean', axis=3))  # Frobenius norm

            def get_config(self):
                return {'regularizer_weight': float(self.regularizer_weight)}

        def model(data_input1, data_input2, data_input3, data_output1, data_output2, duration, num_channel):

            # CNN
            # input layer
            basis_real = Input(shape=(duration, duration, 1), name='Basis_real')
            basis_imag = Input(shape=(duration, duration, 1), name='Basis_imag')

            # Convolutional layer
            coeff_real = Conv2D(num_channel, kernel_size=(1, duration), activation='linear',
                                input_shape=(duration, duration, 1), name='Coeff_real', use_bias=False,
                                # kernel_regularizer=GroupSparseRegularizer(regularizer_weight=regularizer_weight))
                                kernel_regularizer=regularizers.l1(regularizer_weight))  # regularizers.l1(0.1)
            coeff_imag = Conv2D(num_channel, kernel_size=(1, duration), activation='linear',
                                input_shape=(duration, duration, 1), name='Coeff_imag', use_bias=False,
                                # kernel_regularizer=GroupSparseRegularizer(regularizer_weight=regularizer_weight))
                                kernel_regularizer=regularizers.l1(regularizer_weight))

            mask = Input(shape=(duration, num_channel, 1), name='Mask')

            # Feature maps as intermediate layer (need permute dimensions)
            item_ac = coeff_real(basis_real)
            item_ad = coeff_imag(basis_real)
            item_bc = coeff_real(basis_imag)
            item_bd = coeff_imag(basis_imag)

            item_ac_permuted = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 3, 2)))(item_ac)
            item_ad_permuted = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 3, 2)))(item_ad)
            item_bc_permuted = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 3, 2)))(item_bc)
            item_bd_permuted = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 3, 2)))(item_bd)

            # Recovery
            signal_real = Subtract(name='Signal_real')([item_ac_permuted, item_bd_permuted])
            signal_imag = Add(name='Signal_imag')([item_ad_permuted, item_bc_permuted])

            # masking
            signal_real_masked = Multiply(name='Sparse_signal_real')([signal_real, mask])
            signal_imag_masked = Multiply(name='Sparse_signal_imag')([signal_imag, mask])

            # reshape into more square
            def more_square(num_row, num_col):
                product = num_row * num_col
                square_root = math.sqrt(product)
                for n in range(math.floor(square_root), 0, -1):
                    if product % n == 0:
                        d1_new = n
                        d2_new = int(product / n)
                        return d1_new, d2_new

            new_row, new_col = more_square(signal_real_masked.get_shape().as_list()[1],
                                           signal_real_masked.get_shape().as_list()[2])
            signal_real_masked_reshaped = Reshape((new_row, new_col))(signal_real_masked)
            signal_imag_masked_reshaped = Reshape((new_row, new_col))(signal_imag_masked)

            # singular_value_real = tf.svd(signal_real_masked_reshaped, compute_uv=False)
            # singular_value_imag = tf.svd(signal_imag_masked_reshaped, compute_uv=False)
            #
            # truncated_nuclear_regularization = 1.0 * (tf.reduce_sum(singular_value_real[30:-1]) +
            #                                           tf.reduce_sum(singular_value_imag[30:-1]))

            model = Model(inputs=[basis_real, basis_imag, mask], outputs=[signal_real_masked, signal_imag_masked])

            return model

        model = model(data_input1, data_input2, data_input3,
                      data_output1, data_output2,
                      duration, data_input3.shape[2])
        model.summary()

        if not os.path.exists(result_path + result_folder + 'model_v6.png'):
            plot_model(model, to_file='model_v6.png', show_shapes=True, show_layer_names=True)

        SVG(model_to_dot(model).create(prog='dot', format='svg'))

        #%% 7 Solving

        def mean_squared_error_sampled_points(y_true, y_pred):
            if y_true == 0:
                pass
            else:
                loss = losses.mean_squared_error(y_true, y_pred)
            return loss

        #%%
        opt = Adam(lr=0.0001)  # 0.0001
        ##
        lw_real = K.variable(1.)
        lw_imag = K.variable(1.)
        model.compile(optimizer=opt, loss="mean_squared_error",
                      loss_weights=[lw_real, lw_imag], metrics=['mae'])

        #%%
        def lr_scheduler(epoch):
            if epoch == 301:  # epoch == 301
                model.optimizer.lr.set_value(0.00001)
            return K.get_value(model.optimizer.lr)

        #%%
        # duration_float = float(duration)

        # duration_keras = K.variable(duration)
        class LossWeightsScheduler(keras.callbacks.Callback):
            # def __init__(self, lw_real, lw_imag):
            def __init__(self, lw_real, lw_imag, epochs, loss_weight_rate):
                self.lw_real = lw_real
                self.lw_imag = lw_imag
                self.epochs = epochs
                self.loss_weight_rate = loss_weight_rate

            # customize your behavior
            def on_epoch_begin(self, epoch, logs={}):
                if epoch <= self.epochs:
                    # K.set_value(self.lw_real, math.pow(2, (epoch//50 + 1)))
                    K.set_value(self.lw_real, math.pow(2, (epoch/self.epochs * self.loss_weight_rate + 1)))
                    K.set_value(self.lw_imag, math.pow(2, (epoch/self.epochs * self.loss_weight_rate + 1)))

        class LossHistory(keras.callbacks.Callback):
            def on_train_begin(self, logs={}):
                self.losses = []

            def on_batch_end(self, batch, logs={}):
                self.losses.append(logs.get('loss'))

        change_lr = LearningRateScheduler(lr_scheduler)
        loss_history = LossHistory()

        weights_folder = 'weights_history/'
        weights_slice_folder = 'slice_%02d/' % (slice+1)
        create_folder(result_path + result_folder + weights_folder + weights_slice_folder)

        file_path_of_weights = result_path + result_folder + weights_folder + weights_slice_folder + \
                               'weights_slice-%02d-{epoch:03d}.hdf5' % (slice+1)
        checkpoint = ModelCheckpoint(file_path_of_weights, monitor='loss', verbose=0, save_best_only=False,
                                     save_weights_only=True, mode='auto', period=50)

        history = model.fit(x=[data_input1, data_input2, data_input3], y=[data_output1, data_output2],
                            epochs=epochs, batch_size=batch_size,
                            callbacks=[loss_history,
                                       LossWeightsScheduler(lw_real, lw_imag, epochs, loss_weight_rate),
                                       checkpoint
                                       ])  # change_lr

        #%% 8 Check results

        #%% Plot training history
        print(history.history.keys())
        # print(len(loss_history.losses))

        # summarize history for loss
        fig = plt.figure(figsize=(6,4))
        plt.semilogy(history.history['Sparse_signal_real_loss'], linestyle='-', color='red')
        plt.semilogy(history.history['Sparse_signal_imag_loss'], linestyle='--', color='blue')
        # plt.semilogy(history.history['Signal_real_loss'])
        # plt.semilogy(history.history['Signal_imag_loss'])
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        # plt.title('Model loss')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(['real part', 'imaginary part'], loc='upper right', fontsize=12)
        plt.tight_layout()
        plt.show()
        file_name = result_path + result_folder + 'training_history_slice-%02d' % (slice+1)
        fig.savefig(file_name + '.png')
        fig.savefig(file_name + '.svg')
        fig.savefig(file_name + '.eps')
        fig.savefig(file_name + '.pdf')
        subprocess.call(
            'C:/Program Files/Inkscape/inkscape.exe ' + file_name + '.svg ' '--export-emf=' + file_name + '.emf')
        time.sleep(0.1)

        #%% Get basis coeffs

        # weights_real = np.zeros(model.get_weights()[0].shape)
        weights_real = np.squeeze(model.get_weights()[0])
        # print('type of weights_real:', type(weights_real), '\n')
        print('shape of weights_real:', weights_real.shape, '\n')

        weights_imag = np.squeeze(model.get_weights()[1])
        # print('type of weights_imag:', type(weights_imag), '\n')
        print('shape of weights_imag:', weights_imag.shape, '\n')

        weights_complex[slice] = weights_real + weights_imag * 1j
        # print('type of weights_complex:', type(weights_complex), '\n')
        # print('dtype of weights_complex:', weights_complex.dtype, '\n')
        print('shape of weights_complex:', weights_complex.shape, '\n')

        # Plot heatmap of real part of coefficients
        fig = plt.figure(figsize=(3, 60))  # figsize=(6, 6)
        ax = sns.heatmap(weights_imag, cmap='coolwarm', square=True, xticklabels=1, yticklabels=128)
        sns.set(font_scale=1.0)
        # cbar_axes = ax.figure.axes[-1]
        # ax.figure.axes[-1].yaxis.label.set_size(12)
        # ax.tick_params(axis='both', which='major', labelsize=12)
        # ax.tick_params(axis='both', which='minor', labelsize=12)
        # plt.title('Imaginary part of basis coefficients')  # , fontsize=12
        # plt.xlabel('Column')
        # plt.ylabel('Row')
        # plt.xticks(np.arange(0, 2048, 32))
        plt.tight_layout()
        # plt.show()
        fig.savefig(result_path + result_folder +
                    'basis_coeffs_signal_all_imag_slice-%02d_heatmap.pdf' % (slice + 1))
        plt.close()
        sns.reset_orig()

        # Plot heatmap of imaginary part of coefficients
        fig = plt.figure(figsize=(3, 60))  # figsize=(6, 6)
        ax = sns.heatmap(weights_real, cmap='coolwarm', square=True, xticklabels=1, yticklabels=128)
        # sns.set(font_scale=0.05)
        # cbar_axes = ax.figure.axes[-1]
        # ax.figure.axes[-1].yaxis.label.set_size(12)
        # ax.tick_params(axis='both', which='major', labelsize=12)
        # ax.tick_params(axis='both', which='minor', labelsize=12)
        # plt.title('Real part of basis coefficients')  # , fontsize=12
        # plt.xlabel('Column')
        # plt.ylabel('Row')
        # plt.xticks(np.arange(0, 2048, 32))
        plt.tight_layout()
        # plt.show()
        fig.savefig(result_path + result_folder +
                    'basis_coeffs_signal_all_real_slice-%02d_heatmap.pdf' % (slice+1))
        plt.close()
        sns.reset_orig()

        # Plot real part of coefficients
        fig = plt.figure(figsize=(17,4))
        plt.plot(weights_real)
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        plt.title('Real part', fontsize=12)
        plt.xlabel('Point', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.legend(channel_str, loc=9, ncol=len(channel_str), fontsize=12)
        plt.tight_layout()
        # plt.show()
        fig.savefig(result_path + result_folder + 'basis_coeffs_signal_all_real_slice-%02d.png' % (slice+1))
        time.sleep(0.1)

        # Plot imaginary part of coefficients
        fig = plt.figure(figsize=(17,4))
        plt.plot(weights_imag)
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        plt.title('Imaginary part', fontsize=12)
        plt.xlabel('Point', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.legend(channel_str, loc=9, ncol=len(channel_str), fontsize=12)
        plt.tight_layout()
        # plt.show()
        fig.savefig(result_path + result_folder + 'basis_coeffs_signal_all_imag_slice-%02d.png' % (slice+1))
        time.sleep(0.1)

        # Plot coefficients of each channel respectively
        for f in range(len(channel)):
            fig = plt.figure(figsize=(12,8))

            plt.subplot(2, 1, 1)
            plt.plot(weights_real[:,f])
            ax = plt.gca()
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            plt.title('Real part', fontsize=12)
            plt.legend([channel_str[f]], loc=1, fontsize=12)
            plt.xlabel('Point', fontsize=12)
            plt.ylabel('Amplitude', fontsize=12)
            plt.xlim(0, duration)

            plt.subplot(2, 1, 2)
            plt.plot(weights_imag[:, f])
            ax = plt.gca()
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            plt.title('Imaginary part', fontsize=12)
            plt.legend([channel_str[f]], loc=1, fontsize=12)
            plt.xlabel('Point', fontsize=12)
            plt.ylabel('Amplitude', fontsize=12)
            plt.xlim(0, duration)

            # plt.show()
            plt.tight_layout()

            result_folder_signal = 'signal_%02d/' % (channel[f]+1)
            create_folder(result_path + result_folder + result_folder_signal)
            fig.savefig(result_path + result_folder + result_folder_signal + 'basis_coeffs_signal_%d_slice-%02d.png' % (channel[f]+1, slice + 1))
            plt.close()
            time.sleep(0.1)

        #%% Check reconstructed data

        data_hat_time = np.matmul(basis_matrix, weights_complex[slice])

        data_hat_time = data_hat_time * data_split_norm[slice] + data_split_offset[slice]  # 20190422
        data_time = data_time * data_split_norm[slice] + data_split_offset[slice]  # 20190422
        data_masked = data_masked * data_split_norm[slice] + data_split_offset[slice]  # 20190422

        data_hat[slice] = data_hat_time
        print('shape of data_hat_time:', data_hat_time.shape)

        recon_error_time_domain[slice] = np.linalg.norm((np.real(data_hat_time - data_time)), ord=2, axis=0) / \
                                         np.linalg.norm(np.real(data_time), ord=2, axis=0)
        print('\nrecon_error_time_domain:', recon_error_time_domain[slice])

        # recon_error_time_domain_txt = recon_error_time_domain[slice]
        # recon_error_time_domain_txt = np.append(recon_error_time_domain_txt,
        #                                         np.mean(recon_error_time_domain[slice])).reshape((1, len(channel)+1))
        #
        # np.savetxt(result_path + result_folder + result_file + '.txt', recon_error_time_domain_txt * 100, fmt='%.2f',
        #            header='row: slice, column: channel')
        # print('\nResults saved: ' + result_file + '.txt')

        #%% Time domain:

        data_masked_ignore_zero = np.array(data_masked)
        data_masked_ignore_zero[data_masked_ignore_zero == 0] = np.nan  # data ignored zero for plotting
        data_masked_ignore_zero_all[slice] = data_masked_ignore_zero

        # idx_missing_section = np.zeros(data_masked_ignore_zero.shape)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # for f in channel:
        #     # for p in range(data_masked_ignore_zero.shape[0]):
        #     for p in range(duration):
        #         if np.isnan(data_masked_ignore_zero[p, list(channel).index(f)]):
        #             idx_missing_section[p+1, list(channel).index(f)] = 1


        for f in channel:

            print('\nPlotting signal ' + str(f) + '\n')

            # fig = plt.figure(figsize=(17,4))
            # plt.plot(t[section], data[section,f])
            # plt.title('Signal %s - original'%(f))
            # plt.xlabel('Time (sec)')
            # plt.ylabel('Velocity (mm/s)')
            # plt.grid(True)
            # # plt.show()
            # fig.savefig(result_path+ result_folder + 'signal_' + str(f+1) + '_orig_slice-%03d.png' % (slice+1))
            # time.sleep(0.1)
            #
            # fig = plt.figure(figsize=(18,4))
            # plt.plot(t[section], data_masked[section,f])
            # plt.legend([channel_str[f]], loc=1)
            # plt.title('Signal %s - original - masked'%(f))
            # plt.xlabel('Time (sec)')
            # plt.ylabel('Velocity (mm/s)')
            # plt.grid(True)
            # # plt.show()
            # fig.savefig(result_path + result_folder + result_folder_signal + 'signal_' + str(f+!) + '_orig_mask_slice-%03d.png' % (slice+1))
            # time.sleep(0.1)
            #
            # fig = plt.figure(figsize=(17,4))
            # plt.plot(t[section], np.real(data_hat[section,f]))
            # plt.title('Signal %s - hat - real'%(f))
            # plt.xlabel('Time (sec)')
            # plt.ylabel('Velocity (mm/s)')
            # plt.grid(True)
            # # plt.show()
            # fig.savefig(result_path + result_folder + result_folder_signal + 'signal_' + str(f+1) + '_hat_real_slice-%03d.png' % (slice+1))
            # time.sleep(0.1)
            #
            # fig = plt.figure(figsize=(17,4))
            # plt.plot(t[section], np.imag(data_hat[section,f]))
            # plt.title('Signal %s - hat - imag'%(f))
            # plt.xlabel('Time (sec)')
            # plt.ylabel('Velocity (mm/s)')
            # plt.grid(True)
            # # plt.show()
            # fig.savefig(result_path + result_folder + result_folder_signal + 'signal_' + str(f+1) + '_hat_imag_slice-%03d.png' % (slice+1))
            # time.sleep(0.1)

            fig = plt.figure(figsize=(17,4))
            plot_origin = plt.plot(t, np.real(data_time[:,list(channel).index(f)]), 'red')  # xkcd:lightish green

            # if packet == 1:
            #     plot_sampled = plt.scatter(t, np.real(data_masked_ignore_zero[:,list(channel).index(f)]), c='xkcd:lightish green')

            plot_recover = plt.plot(t, np.real(data_hat_time[:,list(channel).index(f)]), 'blue')
            # plot_error = plt.plot(t, np.real(data_time[:,list(channel).index(f)] - data_hat_time[:,list(channel).index(f)]), 'r')  # error

            p = 0
            while p < duration:
                if mask_matrix[p, list(channel).index(f)] == 0:
                    # plot section background
                    plot_section = plt.axvspan(p / fs, (p + packet) / fs, facecolor='red', alpha=0.08)
                    p = p + packet
                else:
                    p = p + 1

            ax = plt.gca()
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            plt.title('Signal %s - Slice %03d - Comparison' % (f+1, slice+1), fontsize=12)
            plt.xlabel('Time (sec)', fontsize=12)
            plt.ylabel(data_unit(data_name), fontsize=12)
            plt.legend(['Original', 'Recovered'], fontsize=12)
            # plt.legend(['Original', 'Recovered', 'Error'], fontsize=12)
            plt.xlim(0, max(t))
            # plt.xlim(2, 3)  # for toy_data
            plt.grid(True)
            plt.tight_layout()

            result_folder_signal = 'signal_%02d/' % (f + 1)
            create_folder(result_path + result_folder + result_folder_signal)
            fig.savefig(result_path + result_folder + result_folder_signal + 'compare_time_signal_' + str(f+1) + '_slice-%03d.png' % (slice+1))
            fig.savefig(result_path + result_folder + result_folder_signal + 'compare_time_signal_' + str(f+1) + '_slice-%03d.pdf' % (slice+1))
            time.sleep(0.1)

            plt.close('all')

        #%% Frequency domain:

        data_freq = np.fft.fft(np.real(data_time), axis=0)
        data_hat_freq = np.fft.fft(np.real(data_hat_time), axis=0)

        recon_error_freq_domain[slice] = np.linalg.norm((data_hat_freq - data_freq), ord=2, axis=0) / \
                                         np.linalg.norm(data_freq, ord=2, axis=0)
        # print('recon_error_freq_domain:', recon_error_freq_domain[slice])

        x_axis_freq = np.arange(0, fs, fs/duration)
        for f in channel:

            print('\nPlotting signal ' + str(f) + '\n')

            # plt.figure(figsize=(18,4))
            # plt.plot(np.abs(data_freq[:,list(channel).index(f)]))
            # plt.title('signal %s original'%(f))
            # # plt.show()
            #
            # plt.figure(figsize=(18,4))
            # plt.plot(np.abs(data_hat_freq[:,list(channel).index(f)]))
            # plt.title('signal %s hat'%(f))
            # # plt.show()

            fig = plt.figure(figsize=(17,4))
            plt.plot(x_axis_freq, np.abs(data_freq[:,list(channel).index(f)]), 'red')  # xkcd:lightish green
            plt.plot(x_axis_freq, np.abs(data_hat_freq[:,list(channel).index(f)]), 'blue')
            # plt.plot(x_axis_freq, np.abs(data_freq[:,list(channel).index(f)] - data_hat_freq[:,list(channel).index(f)]), 'r')
            ax = plt.gca()
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            plt.title('Signal %s - Slice %03d - Comparison' % (f+1, slice+1), fontsize=12)
            plt.xlabel('Frequency (Hz)', fontsize=12)
            plt.ylabel(data_unit(data_name), fontsize=12)
            plt.legend(['Original', 'Recovered'], fontsize=12)
            # plt.legend(['Original', 'Recovered', 'Error'], fontsize=12)
            plt.xlim(0, fs/2+1)
            plt.grid(True)
            plt.tight_layout()

            result_folder_signal = 'signal_%02d/' % (f + 1)
            create_folder(result_path + result_folder + result_folder_signal)
            fig.savefig(result_path + result_folder + result_folder_signal + 'compare_freq_signal_' + str(f+1) + '_slice-%03d.png' % (slice+1))
            fig.savefig(result_path + result_folder + result_folder_signal + 'compare_freq_signal_' + str(f+1) + '_slice-%03d.pdf' % (slice+1))
            time.sleep(0.1)

            plt.close('all')

        elapsed_time = time.time() - start_time
        print(time.strftime("Elapsed time: %H:%M:%S\n", time.gmtime(elapsed_time)))

    #%% 9 Save variables

    # recon_error_time_domain_global = np.mean(recon_error_time_domain)
    # print('recon_error_time_domain_global shape: ', recon_error_time_domain_global.shape)
    # print('recon_error_time_domain_global: ', recon_error_time_domain_global)

    results = {
        'data_split': data_split,
        'data_hat': data_hat,
        'data_masked_ignore_zero_all': data_masked_ignore_zero_all,
        'mask_matrix_all': mask_matrix_all,
        'weights_complex': weights_complex,
        'recon_error_time_domain': recon_error_time_domain,
        'recon_error_freq_domain': recon_error_freq_domain  # ,
        # 'history': history  # cannot be pickled because it is not a variable?
    }

    # collect metrics for parallel plot
    collect_results_folder = 'collect_results/'
    collect_results_file = 'collect_results'
    create_folder(result_path + collect_results_folder)

    recon_error_time_domain_mean_all = float(np.mean(recon_error_time_domain))

    with open(result_path + collect_results_folder + collect_results_file + '.txt', 'a+') as f:
        f.write('%5.2f, %03d, %03d, %02d, %04d, %7.5f\n' % (regularizer_weight, batch_size,
                                                          harmonic_wavelet_interval, loss_weight_rate,
                                                          epochs, recon_error_time_domain_mean_all))
    f.close()

    # save metrics locally
    np.savetxt(result_path + result_folder + result_file + '.txt', recon_error_time_domain * 100, fmt='%.2f',
               header='row: slice, column: channel')
    print('\nResults saved: ' + result_file + '.txt')



    # recon_error_time_domain_txt = recon_error_time_domain[slice]
    # recon_error_time_domain_txt = np.append(recon_error_time_domain_txt,
    #                                         np.mean(recon_error_time_domain[slice])).reshape((1, len(channel)+1))
    #
    # np.savetxt(result_path + result_folder + result_file + '.txt', recon_error_time_domain_txt*100, fmt='%.2f',
    #            header='row: slice | first column: mean value, subsequent column: channel')
    # print('\nMetrics saved at: ' + result_file + '.txt')




    # save results file locally
    with open(result_path + result_folder + result_file + '.pickle', 'wb') as f:
        pickle.dump(results, f)
    f.close()
    print('\nResults saved at: ' + result_file + '.pickle')
    K.clear_session()


#%%
for pac in packet:

    for d_name in data_name:

        def data_fs(name):
            return {
                'DPM': 100,
                'GPS': 1,
                'HPT': 1,
                'RHS': 10,
                'TLT': 1,
                'UAN': 10,
                'ULT': 1,
                'VIB': 50,
                'VIC': 50,
                'VIF': 40,
                'toy_data5': 400,
                'RSG': 20,
                'U1': 100,
                'U2': 100,
                'U3': 100,
                'U4': 100,
                'U5': 100,
                'U6': 100,
                'U7': 100,
                'U8': 100,
                'U9': 100
            }.get(name, 1)  # 1 is default if x not found

        fs = data_fs(d_name)

        # for pac in packet:

        # earthquake event 1
        duration = 1024 * 8  # 1024 * 8
        overlap = 0  # 0
        result_path = "C:/dataArchive/quanzhouwan/NCS_results_group_sparse_test_v6_l1/%s/Data_%d_%d/" % (
        # result_path = "C:/dataArchive/quanzhouwan/NCS_results_group_sparse_test_v6/%s/Data_%d_%d/" % (
            event_name, duration, pac)  # Box / PC / XPS

        # # typhoon event 5
        # duration = 2048
        # overlap = 512
        # result_path = "C:/dataArchive/quanzhouwan/NCS_results_group_sparse/%s/Data_%d_%d/" % (event_name, duration, packet)  # Box / PC / XPS

        # # toy
        # duration = 2048
        # overlap = 512
        # result_path = "C:/dataArchive/quanzhouwan/NCS_results_group_sparse/%s/Data_%d_%d/" % (event_name, duration, packet)  # Box / PC / XPS

        # # simulation
        # duration = 2048*2
        # overlap = 0  # 512
        # result_path = "C:/dataArchive/simulation/NCS_results_group_sparse_test_v6/%s/Data_%d_%d/" % (
        #     event_name, duration, pac)  # Box / PC / XPS

        # # sutong
        # duration = 2048 * 4
        # overlap = 0  # 512
        # result_path = "C:/dataArchive/sutong/NCS_results_group_sparse/%s/Data_%d_%d/" % (event_name, duration, packet)  # Box / PC / XPS

        # # haicang
        # duration = 4096
        # overlap = 0  # 512
        # result_path = "C:/dataArchive/haicang/NCS_results_group_sparse_test_v6/%s/Data_%d_%d/" % (
        #     event_name, duration, pac)  # Box / PC / XPS

        for sratio in sample_ratio:
            for rseed in randseed:
                for rw in regularizer_weight:
                    for hw_interval in harmonic_wavelet_interval:
                        ncs(data_path=data_path, data_name=d_name, tail=tail, fs=fs, duration=duration, overlap=overlap,
                            channel=channel, sample_ratio=sratio, result_path=result_path, randseed=rseed, packet=pac,
                            regularizer_weight=rw, batch_size=batch_size,
                            epochs=epochs, harmonic_wavelet_interval=hw_interval, loss_weight_rate=loss_weight_rate)
                        # regularizer_weight=regularizer_weight
                        # bad_channel=bad_channel, bad_sample_ratio=bad_sample_ratio
                        print('\nSeed %d done.\n' % rseed)
                        time.sleep(1)
