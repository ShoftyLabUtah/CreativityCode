''' DECODING TRIAL TYPE LEVEL FROM LFP POWER
    LDA Decode AUT vs OCT

    Makes Figure 3d
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
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler  # to standardize the features
from sklearn.decomposition import PCA  # to apply PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import feature_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


import patient_params
patients_data = patient_params.patients_data

import decoding_funcs



from mpl_helpers import PdfPlotter
PdfPlotter('Figures/decoding_trial_type_AUT_OCT.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))

network = input("Choose cortical network (DMN, FrontoParietal, SomatoMotor, Limbic, Visual, DorsalAttention, VentralAttention, all): ")

if network=='all':
    networks = ['DMN', 'FrontoParietal', 'SomatoMotor', 'Limbic', 'Visual', 'DorsalAttention', 'VentralAttention']
else:
    networks = [network]

scores_all_networks_AUT_OCT = {}




####DO THIS IF WANT TO JUST CHECK DMN DECODING WHEN CONSIDER ONLY SUBSET OF PATIENTS WITH DAN COVERAGE
# all_dat_DA = pd.read_csv('pkl_files/store_neural_data_list_with_rippleDorsalAttention.csv', header=[0, 1], index_col=[0, 1, 2, 3])
# all_dat_DMN = pd.read_csv('pkl_files/store_neural_data_list_with_rippleDMN.csv', header=[0, 1], index_col=[0, 1, 2, 3])
# patients_subset = all_dat_DA.index.get_level_values('patient').unique().values


for network in networks:
    all_dat = pd.read_csv('pkl_files/store_neural_data_list_all_trialtypes_' + network + '.csv', header=[0, 1], index_col=[0, 1, 2, 3])

    ####DO THIS IF WANT TO JUST CHECK DMN DECODING WHEN CONSIDER ONLY SUBSET OF PATIENTS WITH DAN COVERAGE
    # if network == 'DMN':
    #     all_dat = all_dat_DMN.loc[patients_subset]

    patient_dfs = [group for _, group in all_dat.groupby(level='patient')]
    cleaned_dfs = [df.dropna(axis=1, how='all') for df in patient_dfs]

    neural_data_list_fixed_AUC_OCT = [neural_data.iloc[np.where((neural_data.index.get_level_values('trial_type')=='AUT') | (neural_data.index.get_level_values('trial_type')=='OCT'))[0]] for neural_data in cleaned_dfs]
    trial_type_float_labels = [1.0*(neural_data.index.get_level_values('trial_type')=='AUT') for neural_data in neural_data_list_fixed_AUC_OCT] 

    df_by_freq_band_all = [neural_data.unstack('time') for neural_data in neural_data_list_fixed_AUC_OCT]
    
    #for svm (decode categorical originality)
    cm_all_patients_AUT_OCT = [] 
    scores_all_patients_AUT_OCT = [] 

    true_labels_numeric_all_patients = []
    decision_scores_all_patients = []

    for p, df in enumerate(df_by_freq_band_all):
        if df.index.get_level_values('patient').unique()[0]=='UIC202420':
            continue
        if df.index.get_level_values('patient').unique()[0]=='UIC202422':
            continue
        print(p)
        trial_type_float_labels = 1.0*(df.index.get_level_values('trial_type')=='AUT')


        clf = LinearDiscriminantAnalysis()
        kf = StratifiedKFold(n_splits=4)
        answers_trial_type = []
        scores = []
        true_labels = []
        decision_scores = []
        
        for train_idx, test_idx in kf.split(df.index.get_level_values('trial_idx'), df.index.get_level_values('trial_type')):
            print(len(train_idx))
            train =  df.iloc[train_idx]
            test =  df.iloc[test_idx]
            clf.fit(train, train.index.get_level_values('trial_type'))
            answers_trial_type.extend(clf.predict(test))
            score = clf.score(test, test.index.get_level_values('trial_type'))
            scores.append(score)
            # Get true labels and decision function scores
            true_labels.extend(test.index.get_level_values('trial_type'))



        cm = confusion_matrix(df.index.get_level_values('trial_type'), answers_trial_type)
        avg_score = np.mean(scores)
        cm_all_patients_AUT_OCT.append(cm)
        scores_all_patients_AUT_OCT.append(avg_score)

        
    scores_all_networks_AUT_OCT.update({network: np.array(scores_all_patients_AUT_OCT)*100})


    avg_cm_all_patients_AUT_OCT =np.mean(cm_all_patients_AUT_OCT,axis=0)
    cm_norm_all_patients_AUT_OCT = [cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] for cm in cm_all_patients_AUT_OCT]
    avg_cm_norm_AUT_OCT = avg_cm_all_patients_AUT_OCT.astype('float') / avg_cm_all_patients_AUT_OCT.sum(axis=1)[:, np.newaxis]




import statsmodels.stats.multitest as multi
p_values = []
for net in scores_all_networks_AUT_OCT:
    t,p = scipy.stats.ttest_1samp(scores_all_networks_AUT_OCT[net], 50, alternative='greater')
    t,p = scipy.stats.ttest_1samp(scores_all_networks_AUT_OCT[net], 50)

    print(scipy.stats.ttest_1samp(scores_all_networks_AUT_OCT[net], 50))
    p_values.append(p)


colors = ['orange', 'blue', 'red', 'yellow', 'green', 'purple', 'grey']
p_values_both = []
p_values_corr_both = []
scores_all_networks = scores_all_networks_AUT_OCT
f,ax = plt.subplots()
ax.boxplot(list(scores_all_networks.values()), showfliers=False, labels = list(scores_all_networks.keys()))
for i, scores in enumerate(list(scores_all_networks.values())):
    x = np.random.normal(i+1, scale = 0.04 ,size=len(scores))
    ax.scatter(x, scores, s = 10, color = colors[i], edgecolors='k', linewidths=0.5)
ax.axhline(50,color='k',linewidth = 0.5, linestyle='--')
# ax.set_ylim(1,100)
sns.despine(top = True, right = True)
ax.set_ylabel('% decoding accuracy')
ax.set_title('Decoding AUT vs BAT vs OCT')

p_values = []
for i, scores in enumerate(list(scores_all_networks.values())):
    t,p = scipy.stats.ttest_1samp(scores, 50, alternative='greater')
    t,p = scipy.stats.ttest_1samp(scores, 50)
    print(scipy.stats.ttest_1samp(scores, 50))
    p_values.append(p)
    # print(p)
p_values_both.append(p_values)
#FDR correction:
rej, corr_p = multi.fdrcorrection(p_values, alpha=0.05, method='indep')
p_values_corr_both.append(corr_p)


