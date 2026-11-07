''' 
#decode from different numbers of contacts, to make sure that we're not getting better decoding in some patients/networks purely because of more contacts 

'''

import numpy as np
from neo import io
import quantities as pq
import glob   
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
import mat73
import itertools as it
import random
import seaborn as sns
import re
import mne
import scipy
from scipy.stats.stats import pearsonr

import DMN_LFP_funcs
import beh_funcs

import sklearn
from sklearn import discriminant_analysis
from sklearn.svm import SVC 
from sklearn.svm import LinearSVC
from sklearn.metrics import  roc_curve, auc, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler  # to standardize the features
from sklearn.decomposition import PCA  # to apply PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import feature_selection

import sys
sys.path.append('../')

import patient_params
patients_data = patient_params.patients_data

import decoding_funcs

import itertools

# plt.ion()

# from mpl_helpers import PdfPlotter

network = input("Choose cortical network (DMN, FrontoParietal, SomatoMotor, Limbic, Visual, DorsalAttention, VentralAttention, all): ")

if network=='all':
    networks = ['DMN', 'FrontoParietal', 'SomatoMotor', 'Limbic', 'Visual', 'DorsalAttention', 'VentralAttention']
else:
    networks = [network]

scores_all_networks_AUT_BAT = {}

colors = ['orange', 'blue', 'red', 'yellow', 'green', 'purple', 'grey']


weights_by_channel_all = []
originality_by_trial_all = []

f_nets,ax_nets = plt.subplots(7)

f,ax = plt.subplots()

scores_across_n_all_nets = []

for net_idx, network in enumerate(networks):
    all_dat = pd.read_csv('pkl_files/store_neural_data_list_with_ripple' + network + '.csv', header=[0, 1], index_col=[0, 1, 2, 3])

    patient_dfs = [group for _, group in all_dat.groupby(level='patient')]
    cleaned_dfs = [df.dropna(axis=1, how='all') for df in patient_dfs]

    neural_data_list_fixed = cleaned_dfs
    df_by_freq_band_all = [neural_data.unstack('time') for neural_data in neural_data_list_fixed]

    #for svm (decode categorical originality)
    cm_all_patients_AUT_BAT = [] 
    scores_all_patients_AUT_BAT = [] 
    scores_across_bins_all_patients = []


    scores_across_n_all_patients = []

    for p, df in enumerate(df_by_freq_band_all):
        if df.index.get_level_values('patient').unique()[0]=='UIC202420':
            continue
        if df.index.get_level_values('patient').unique()[0]=='UIC202422':
            continue
        print(p)
        df_by_freq_band_to_decode = df.T
        df_by_freq_band_to_decode.dropna(inplace=True)


        scores_across_n = [] #for subsampling (to check that number of features isnt what makes DMN decoding better)
        channel_ids = df_by_freq_band_to_decode.index.get_level_values('channel').unique()

        for n in np.arange(len(channel_ids))[1:]:
            scores_across_repeats = [] #for subsampling (to check that number of features isnt what makes DMN decoding better)

            for i in itertools.repeat(None, 10): #repeat each subselection of contacts
                channels_subset = random.sample(list(channel_ids), n) #pick n contacts
                print(len(channels_subset))
                df_by_freq_band_to_decode_sub = df_by_freq_band_to_decode.loc[:,channels_subset,:] #subsampled


                clf = SVC(C=1, kernel='linear', tol=1e-6)
                kf = KFold(n_splits=4)
                answers_trial_type = []
                scores = []



                for train_idx, test_idx in kf.split(df_by_freq_band_to_decode_sub.columns.get_level_values('trial_idx')):
                    train = df_by_freq_band_to_decode_sub.iloc[:,list(train_idx)]
                    test = df_by_freq_band_to_decode_sub.iloc[:,list(test_idx)]
                    clf.fit(train.T, train.columns.get_level_values('trial_type'))
                    answers_trial_type.extend(clf.predict(test.T))
                    score = clf.score(test.T, test.columns.get_level_values('trial_type'))
                    scores.append(score)
                scores_across_repeats.append(np.mean(scores))
            scores_across_n.append(np.mean(scores_across_repeats))
        ax.plot(scores_across_n, color = colors[net_idx])

        # scores_across_n_all_patients.append(scores_across_n)
        # ax_nets[net_idx].plot(scores_across_n)
plt.show()

