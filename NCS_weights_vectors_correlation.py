# -*- coding: utf-8 -*-
"""
Created on Tue April 2 09:43:49 2019

@author: zhiyitang
"""
# %reset

# import os
# os.system('cls')


import numpy as np
import tensorflow as tf
import scipy.io
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.linalg import dft
from sklearn.preprocessing import normalize
from keras import layers
from keras.layers import Input, Dense, Activation, Multiply, Add, Subtract
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD, RMSprop
from keras import regularizers
from keras import backend as K
from keras import losses

import os
# os.environ["PATH"] += os.pathsep + "C:/Program Files (x86)/Graphviz2.38/bin/"
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model
from keras.preprocessing import image
from matplotlib.pyplot import imshow

import pickle
import time
import subprocess

from skimage.util.shape import view_as_windows

import h5py
import numpy as np
from scipy.stats.stats import pearsonr
from itertools import combinations
import seaborn as sns
from cycler import cycler

#%%
def data_fs(name):
    return {
        'DPM': 1,
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

#%% 1 - Gather results
#%% User input starts
#%% regular example
# result_path = "C:/dataArchive/haicang/NCS_results_group_sparse/"  # Box / PC / XPS
#
# # # regular example of group sparse
# # result_path = "C:/dataArchive/haicang/NCS_results/"  # Box / PC / XPS
# # result_path = "C:/dataArchive/haicang/NCS_results_growth/"  # Box / PC / XPS
# # result_path = "/Users/zhiyitang/dataArchive/haicang/NCS_results/"  # Mac
#
# data_name = ['U1']  # ['U%d' % n for n in range(1, 10)]  # ['U2']
# duration = 2048
# overlap = 512
# # total_duration = 1536*10  # 16384-512-512
# total_duration = 2048*8
# fs = 100
#
# # sample_ratio = np.arange(0.05, 0.55, 0.05)
# sample_ratio = np.array([0.05])  # results_growth
#
# packet = 1
# randseed = np.arange(0, 1)
# tail = '.hdf5'
# # results_gather_path = 'results_gather/'
# # User input ends
#
# # number_of_channel = np.array([7, 7, 9, 9, 9, 9, 7, 9, 5])
# number_of_channel = np.array([7])
#
# epoch = 600

#%% haicang
# result_path = "C:/dataArchive/haicang/NCS_results_group_sparse/"  # XPS / Box
result_path = "C:/dataArchive/haicang/NCS_results_group_sparse_test_v6/event_haicang/"  # XPS / Box

# # regular example of group sparse
# result_path = "C:/dataArchive/haicang/NCS_results/"  # Box / PC / XPS
# result_path = "C:/dataArchive/haicang/NCS_results_growth/"  # Box / PC / XPS
# result_path = "/Users/zhiyitang/dataArchive/haicang/NCS_results/"  # Mac

# data_name = ['U%d' % n for n in range(1, 10)]  # ['U%d' % n for n in range(1, 10)]  # ['U2']
data_name = ['U%d' % n for n in range(4, 5)]  # ['U%d' % n for n in range(1, 10)]  # ['U2']
duration = 4096
overlap = 0
total_duration = 4096 * 4

# sample_ratio = np.arange(0.5, 0.1, 1.0)
# sample_ratio = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
sample_ratio = np.array([0.5])

packet = 100
randseed = np.arange(0, 1)
regularizer_weight = 10.0
loss_weight_rate = 24
batch_size = 128
epochs = 2400
harmonic_wavelet_interval = 1
tail = '.hdf5'
# results_gather_path = 'results_gather/'
# User input ends

# number_of_channel = np.array([7, 7, 9, 9, 9, 9, 7, 9, 5])
number_of_channel = np.array([9])
# number_of_channel = np.array([71])


#%% quanzhouwan
# # result_path = "C:/dataArchive/haicang/NCS_results_group_sparse/"  # XPS / Box
# result_path = "C:/dataArchive/quanzhouwan/NCS_results_group_sparse_test_v6/event_1/"  # XPS / Box
# # result_path = "C:/dataArchive/quanzhouwan/NCS_results_group_sparse_test_v6_l1/event_1/"  # XPS / Box
# # result_path = "C:/dataArchive/quanzhouwan/NCS_results_group_sparse_test_v5/event_1/"  # XPS / Box
#
# # # regular example of group sparse
# # result_path = "C:/dataArchive/haicang/NCS_results/"  # Box / PC / XPS
# # result_path = "C:/dataArchive/haicang/NCS_results_growth/"  # Box / PC / XPS
# # result_path = "/Users/zhiyitang/dataArchive/haicang/NCS_results/"  # Mac
#
# data_name = ['VIB']  # ['U%d' % n for n in range(1, 10)]  # ['U2']
# duration = 8192
# overlap = 0
# total_duration = 8192
#
# # sample_ratio = np.arange(0.5, 0.1, 1.0)
# # sample_ratio = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
# sample_ratio = np.array([0.7])
#
# packet = 25
# randseed = np.arange(0, 1)
# regularizer_weight = 10.0
# loss_weight_rate = 24
# batch_size = 128
# epochs = 2400
# harmonic_wavelet_interval = 2
# tail = '.hdf5'
# # results_gather_path = 'results_gather/'
# # User input ends
#
# number_of_channel = np.array([9])
# # number_of_channel = np.array([71])


#%%
total_duration_idx = np.array(range(total_duration))
data_split = view_as_windows(total_duration_idx, duration, duration-overlap)
slice_num = data_split.shape[0]  # number of sections for each channel split by duration
# slice_num = 3

#%% gather information

def abbr(vec):
    line = np.full([np.max(vec) + 2], np.nan)
    line[vec] = vec
    edge = []
    if not np.isnan(line[0]):
        edge.append(0)
    for n in np.arange(1, len(line)):
        if line[n] > 0 and np.isnan(line[n - 1]):
            edge.append(n)
        elif np.isnan(line[n]) and line[n - 1] > 0:
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


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)


recon_error_time_domain_new_mean_all = {}

for dn in data_name:

    channel = np.arange(0, number_of_channel[data_name.index(dn)])
    # channel = np.array([1, 2, 3, 4, 12, 46, 47, 48, 49]) - 1

    channel_str = ['ch ' + str(i+1) for i in channel]

    sample_ratio_str = ["%.2f" % s for s in sample_ratio]
    print('\nReading results:\n')

    for sr in sample_ratio:
        for rseed in randseed:

            channel_str_abbr = tidy_name(abbr(channel))

            # result_folder = 'U_%d_%d/' % (duration, packet) + dn + '_' + str(duration) + '_' + str(packet) + \
            #                 '_[' + channel_str_abbr + ']_' + '%.3f' % sr + '_' + str(rseed) + '/weights_history/'

            result_folder = 'Data_%d_%d/' % (duration, packet) + \
                            dn + '_' + str(duration) + '_' + str(packet) + '_[' + channel_str_abbr + ']_' + \
                            '%.2f' % sr + '_' + str(rseed) + \
                            '_rw_' + str(regularizer_weight) + '_bs_' + str(batch_size) + '_epo_' + str(epochs) + \
                            '_hw_interval_' + str(harmonic_wavelet_interval) + \
                            '_lw_rate_' + '%03d' % loss_weight_rate + '/'

            cor_vector_3d_real = np.array([])
            cor_vector_3d_imag = np.array([])
            cor_vector_1d_real = np.array([])
            cor_vector_1d_imag = np.array([])
            for s_num in range(1, slice_num+1):

                if s_num == 1:
                    group_name_real = 'Coeff_real'
                    group_name_imag = 'Coeff_imag'
                else:
                    group_name_real = 'Coeff_real_%d' % ((s_num-1) * 2)
                    group_name_imag = 'Coeff_imag_%d' % ((s_num-1) * 2)

                epochs_breakpoint_num = 0
                epochs_breakpoints = np.array([])
                for e_num in range(1, epochs+1):
                    if e_num % 50 == 0:
                        epochs_breakpoint_num += 1
                        epochs_breakpoints = np.append(epochs_breakpoints, e_num)
                        print('data_name: %s    sr: %s    rseed: %s    slice number: %02d    epoch: %03d'
                              % (dn, sr, rseed, s_num, e_num))

                        weights_folder = 'weights_history/'
                        weights_slice_folder = 'slice_%02d/' % (s_num)
                        result_file = 'weights_slice-%02d-%03d%s' % (s_num, e_num, tail)
                        data_0 = h5py.File(result_path + result_folder + weights_folder + weights_slice_folder +
                                           result_file, 'r')
                        data_1_real = data_0['Coeff_real']
                        data_1_imag = data_0['Coeff_imag']

                        data_2_real = data_1_real[group_name_real]
                        data_2_imag = data_1_imag[group_name_imag]

                        kernel_real = np.array(data_2_real['kernel:0'])
                        kernel_imag = np.array(data_2_imag['kernel:0'])

                        kernel_real = np.squeeze(kernel_real)
                        kernel_imag = np.squeeze(kernel_imag)

                        # print('kernel real shape: ', kernel_real.shape)
                        # print('kernel imag shape: ', kernel_imag.shape)

                        # print('shape of data_split:\n')
                        # print(results['data_split'].shape)

                        cor_matrix_real = np.zeros([kernel_real.shape[1], kernel_real.shape[1]])
                        cor_matrix_imag = np.zeros([kernel_imag.shape[1], kernel_imag.shape[1]])

                        # combinations
                        combis_real = [c for c in combinations(range(cor_matrix_real.shape[1]), 2)]
                        combis_imag = [c for c in combinations(range(cor_matrix_imag.shape[1]), 2)]
                        # print(combis)

                        combis_arr = np.array(combis_real) + 1
                        combis_str = []
                        combis_str = ['ch %d, %d' % (i, j) for i, j in zip(combis_arr[:, 0], combis_arr[:, 1])]


                        def unit_vector(vector):
                            """ Returns the unit vector of the vector.  """
                            return vector / np.linalg.norm(vector)


                        def angle_between(v1, v2):
                            """ Returns the angle in radians between vectors 'v1' and 'v2'::
                                    # >>> angle_between((1, 0, 0), (0, 1, 0))
                                    # 1.5707963267948966
                                    # >>> angle_between((1, 0, 0), (1, 0, 0))
                                    # 0.0
                                    # >>> angle_between((1, 0, 0), (-1, 0, 0))
                                    # 3.141592653589793
                            """
                            v1_u = unit_vector(v1)
                            v2_u = unit_vector(v2)
                            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

                        for co1, co2 in combis_real:
                            # print('co1: %d, co2: %d' % (co1, co2))
                            temp = pearsonr(kernel_real[:, co1], kernel_real[:, co2])[0]
                            # temp = angle_between(kernel_real[:, co1], kernel_real[:, co2])
                            cor_matrix_real[co1, co2] = temp
                            cor_vector_1d_real = np.append(cor_vector_1d_real, temp)

                        for co1, co2 in combis_imag:
                            # print('co1: %d, co2: %d' % (co1, co2))
                            temp = pearsonr(kernel_imag[:, co1], kernel_imag[:, co2])[0]
                            cor_matrix_imag[co1, co2] = temp
                            cor_vector_1d_imag = np.append(cor_vector_1d_imag, temp)

                        # print(cor_matrix_real)
                        # print(cor_vector_1d)

                        # cor_vector_2d = np.append(cor_vector_2d, cor_vector_1d, axis=[1])
                        # cor_vector_2d = np.append(cor_vector_2d, cor_vector_1d)

                        # print('shape of cor_vector_2d: ', cor_vector_2d.shape)

            # print('cor_vector_1d shape: ', cor_vector_1d.shape)
            cor_vector_3d_real = cor_vector_1d_real
            cor_vector_3d_real = np.reshape(cor_vector_3d_real, [slice_num, epochs_breakpoint_num, len(combis_real)], order='C')
            cor_vector_3d_real = np.transpose(cor_vector_3d_real, axes=[0, 2, 1])

            cor_vector_3d_imag = cor_vector_1d_imag
            cor_vector_3d_imag = np.reshape(cor_vector_3d_imag, [slice_num, epochs_breakpoint_num, len(combis_imag)], order='C')
            cor_vector_3d_imag = np.transpose(cor_vector_3d_imag, axes=[0, 2, 1])

            for s in range(slice_num):
                colors = sns.color_palette("bright", len(combis_real) // 4)  # hls Sets1 Sets2 Paired bright
                plt.rc('axes', prop_cycle=(cycler(linestyle=['-', '--', ':', '-.']) * cycler(color=colors)))



                abc = cor_vector_3d_real[s, :, :]
                abc1 = cor_vector_3d_imag[s, :, :]



                # real part
                fig = plt.figure(figsize=(13, 8))
                plt.plot(epochs_breakpoints, cor_vector_3d_real[s, :, :].T)
                ax = plt.gca()
                ax.tick_params(axis='both', which='major', labelsize=18)
                ax.tick_params(axis='both', which='minor', labelsize=18)
                plt.title('Correlation history - real part', fontsize=18)
                # plt.legend(combis_str, loc="lower right", ncol=6, fontsize=12)  # bbox_to_anchor=(1.04, 1)
                plt.legend(combis_str, loc="upper left", ncol=2, fontsize=16)  # bbox_to_anchor=(1.04, 1)
                plt.xlabel('Epoch', fontsize=18)
                plt.ylabel('Correlation coefficient', fontsize=18)
                plt.grid(linestyle='--')
                plt.xlim(0, 2400)

                plt.show()
                plt.tight_layout()
                file_name = result_path + result_folder + 'weights_correlation_history_slice-%02d_real' % (s + 1)
                fig.savefig(file_name + '.png')
                fig.savefig(file_name + '.svg')
                fig.savefig(file_name + '.eps')
                fig.savefig(file_name + '.pdf')
                # subprocess.call(
                #     'C:/Program Files/Inkscape/inkscape.exe ' + file_name + '.svg ' '--export-emf=' + file_name + '.emf')
                # time.sleep(0.1)

                # imaginary part
                fig = plt.figure(figsize=(13, 8))
                plt.plot(epochs_breakpoints, cor_vector_3d_imag[s, :, :].T)
                ax = plt.gca()
                ax.tick_params(axis='both', which='major', labelsize=18)
                ax.tick_params(axis='both', which='minor', labelsize=18)
                plt.title('Correlation history - imaginary part', fontsize=18)
                # plt.legend(combis_str, loc="upper right", ncol=6, fontsize=12)  # bbox_to_anchor=(1.04, 1)
                plt.legend(combis_str, loc="upper left", ncol=2, fontsize=16)  # bbox_to_anchor=(1.04, 1)
                plt.xlabel('Epoch', fontsize=18)
                plt.ylabel('Correlation coefficient', fontsize=18)
                plt.grid(linestyle='--')
                plt.xlim(0, 2400)

                plt.show()
                plt.tight_layout()
                file_name = result_path + result_folder + 'weights_correlation_history_slice-%02d_imag' % (s + 1)
                fig.savefig(file_name + '.png')
                fig.savefig(file_name + '.svg')
                fig.savefig(file_name + '.eps')
                fig.savefig(file_name + '.pdf')
                # subprocess.call(
                #     'C:/Program Files/Inkscape/inkscape.exe ' + file_name + '.svg ' '--export-emf=' + file_name + '.emf')
                # time.sleep(0.1)
