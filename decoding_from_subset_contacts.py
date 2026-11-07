''' 
#decode from subsampled contacts: 
get numner of DAN contacts, and subsample that many contacts from DMN and decode only from that number of DMN contacts
iterate 
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


import sys
sys.path.append('../')

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


import patient_params
patients_data = patient_params.patients_data

import decoding_funcs

import itertools

# plt.ion()

# from mpl_helpers import PdfPlotter


scores_all_networks_AUT_BAT = {}

colors = ['orange', 'blue', 'red', 'yellow', 'green', 'purple', 'grey']


weights_by_channel_all = []
originality_by_trial_all = []

f_nets,ax_nets = plt.subplots(7)

f,ax = plt.subplots()

scores_across_n_all_nets = []

#using "DAN" to stand in for whatever network we're trying to compare to DMN here
all_dat_DA = pd.read_csv('pkl_files/store_neural_data_list_with_rippleDorsalAttention.csv', header=[0, 1], index_col=[0, 1, 2, 3])
all_dat_DMN = pd.read_csv('pkl_files/store_neural_data_list_with_rippleDMN.csv', header=[0, 1], index_col=[0, 1, 2, 3])
patients_subset = all_dat_DA.index.get_level_values('patient').unique().values
all_dat = all_dat_DMN.loc[patients_subset]



patient_dfs = [group for _, group in all_dat.groupby(level='patient')]
cleaned_dfs = [df.dropna(axis=1, how='all') for df in patient_dfs]

neural_data_list_fixed = cleaned_dfs
df_by_freq_band_all = [neural_data.unstack('time') for neural_data in neural_data_list_fixed]

patient_dfs_DAN = [group for _, group in all_dat_DA.groupby(level='patient')]
cleaned_dfs_DAN = [df.dropna(axis=1, how='all') for df in patient_dfs_DAN]


#for svm (decode categorical originality)
cm_all_patients_AUT_BAT = [] 
scores_all_patients_AUT_BAT = [] 
scores_across_bins_all_patients = []


scores_all_patients_DMN = []
scores_all_patients_DAN = []
scores_all_patients_all_repeats_DMN = []
scores_all_patients_all_repeats_DAN = []

for p, df in enumerate(df_by_freq_band_all):
    patient = df.index.get_level_values('patient').unique()[0]

    if df.index.get_level_values('patient').unique()[0]=='UIC202420':
        continue
    if df.index.get_level_values('patient').unique()[0]=='UIC202422':
        continue


    df_DAN = all_dat_DA.loc[patient].dropna(axis=1)
    df_by_freq_band_to_decode_DAN = df_DAN.unstack('time').T

    chans_DAN = df_DAN.columns.get_level_values('channel').unique()
    len_chans = len(chans_DAN)
    print(len_chans)
    if (len_chans < 2):
        continue 


    df_by_freq_band_to_decode = df.T
    df_by_freq_band_to_decode.dropna(inplace=True)
    # import pdb;pdb.set_trace()



    channel_ids = df_by_freq_band_to_decode.index.get_level_values('channel').unique()

    len_chans = min(len(chans_DAN), len(channel_ids))
    n_iter = 500

    scores_across_repeats_DMN = [] #for subsampling (to check that number of features isnt what makes DMN decoding better)
    scores_across_repeats_DAN = [] #for subsampling (to check that number of features isnt what makes DMN decoding better)

    for i in itertools.repeat(None, n_iter): #repeat each subselection of contacts
        channels_subset = random.sample(list(channel_ids), len_chans) #pick n contacts
        # print(len(channels_subset))
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
        scores_across_repeats_DMN.append(np.mean(scores))



        channels_subset = random.sample(list(chans_DAN), len_chans) #pick n contacts
        # print(len(channels_subset))
        df_by_freq_band_to_decode_sub = df_by_freq_band_to_decode_DAN.loc[:,channels_subset,:] #subsampled

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
        scores_across_repeats_DAN.append(np.mean(scores))

    scores_across_repeats_avg_DMN = np.mean(scores_across_repeats_DMN)
    scores_across_repeats_avg_DAN = np.mean(scores_across_repeats_DAN)
    scores_all_patients_DMN.append(scores_across_repeats_avg_DMN)
    scores_all_patients_DAN.append(scores_across_repeats_avg_DAN)
    scores_all_patients_all_repeats_DMN.append(scores_across_repeats_DMN)
    scores_all_patients_all_repeats_DAN.append(scores_across_repeats_DAN)

scipy.stats.ttest_1samp(scores_all_patients_DMN, 0.5) #DMN: t=5.79 5.27; p=0.00067 0.0012
scipy.stats.ttest_1samp(scores_all_patients_DAN, 0.5) #DAN: t=1.046 1.068; p=0.33 0.32

scipy.stats.ttest_1samp(np.concatenate(scores_all_patients_all_repeats_DMN), 0.5) #DMN: t=5.79 5.27; p=0.00067 0.0012
scipy.stats.ttest_1samp(np.concatenate(scores_all_patients_all_repeats_DAN), 0.5) #DMN: t=5.79 5.27; p=0.00067 0.0012

scipy.stats.mannwhitneyu(scores_all_patients_DMN, scores_all_patients_DAN) #pvalue=0.0379 
scipy.stats.wilcoxon(scores_all_patients_DMN, scores_all_patients_DAN) 
scipy.stats.mannwhitneyu(np.concatenate(scores_all_patients_all_repeats_DMN), np.concatenate(scores_all_patients_all_repeats_DAN)) #pvalue=0.0379


[np.mean(scores_all_patients_DAN[i]>scores_all_patients_all_repeats_DMN[i]) for i in range(len(scores_all_patients_all_repeats_DAN))]
## --> [0.0, 0.0, 0.124, 0.118] for patients w >2 contacts in DAN  
