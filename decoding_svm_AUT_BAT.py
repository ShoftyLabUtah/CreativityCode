''' SVM DECODING TRIAL TYPE LEVEL FROM LFP POWER

    1. Decode AUT vs BAT; ; compare scores for networks
    2. decode high vs low; compare scores for networks 

    Then make boxplots for both types of decoding to compare performance across networks

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


# plt.ion()

from mpl_helpers import PdfPlotter
PdfPlotter('Figures/decoding_creativity_and_trial_type_all_networks.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))

network = input("Choose cortical network (DMN, FrontoParietal, SomatoMotor, Limbic, Visual, DorsalAttention, VentralAttention, all): ")

if network=='all':
    networks = ['DMN', 'FrontoParietal', 'SomatoMotor', 'Limbic', 'Visual', 'DorsalAttention', 'VentralAttention']
else:
    networks = [network]

scores_all_networks_AUT_BAT = {}
scores_all_networks_creativity = {}

f_auc, ax_auc = plt.subplots() # Plot ROC curve


for network in networks:
    all_dat = pd.read_csv('pkl_files/store_neural_data_list_' + network + '.csv', header=[0, 1], index_col=[0, 1, 2, 3])
    
    patient_dfs = [group for _, group in all_dat.groupby(level='patient')]
    cleaned_dfs = [df.dropna(axis=1, how='all') for df in patient_dfs]

    neural_data_list_fixed = cleaned_dfs
    df_by_freq_band_all = [neural_data.unstack('time') for neural_data in neural_data_list_fixed]

    #for svm (decode categorical originality)
    cm_all_patients_AUT_BAT = [] 
    scores_all_patients_AUT_BAT = [] 
    cm_all_patients_creativity = [] 
    scores_all_patients_creativity = [] 
    scores_across_bins_all_patients = []

    most_important_features=[]

    for p, df in enumerate(df_by_freq_band_all):
        print(p)
        df_by_freq_band_to_decode = df.T
        # import pdb;pdb.set_trace()

        '''#try only decoding from stimulus period'''
        # df_by_freq_band_to_decode = df_by_freq_band_to_decode[df_by_freq_band_to_decode.index.get_level_values('time')<10]
        # df = df_by_freq_band_to_decode.T


        clf = SVC(C=1, kernel='linear', tol=1e-6)
        kf = KFold(n_splits=4)
        answers_trial_type = []
        scores = []
        true_labels = []
        decision_scores = []
        for train_idx, test_idx in kf.split(df_by_freq_band_to_decode.columns.get_level_values('trial_idx')):
            print(len(train_idx))

            train = df_by_freq_band_to_decode.iloc[:,list(train_idx)]
            test = df_by_freq_band_to_decode.iloc[:,list(test_idx)]
            clf.fit(train.T, train.columns.get_level_values('trial_type'))
            answers_trial_type.extend(clf.predict(test.T))
            score = clf.score(test.T, test.columns.get_level_values('trial_type'))
            scores.append(score)
            # Get true labels and decision function scores
            true_labels.extend(test.columns.get_level_values('trial_type'))
            decision_scores.extend(clf.decision_function(test.T))  # Use decision function instead of predict

        cm = confusion_matrix(df_by_freq_band_to_decode.columns.get_level_values('trial_type'), answers_trial_type)
        avg_score = np.mean(scores)
        cm_all_patients_AUT_BAT.append(cm)
        scores_all_patients_AUT_BAT.append(avg_score)

        # Compute ROC curve
        true_labels_numeric = [{'AUT': 0, 'BAT': 1}[label] for label in true_labels]
        fpr, tpr, _ = roc_curve(true_labels_numeric, decision_scores)
        roc_auc = auc(fpr, tpr)
        ax_auc.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')


        '''####ONLY LOOK AT AUT TRIALS; DECODE CREATIVITY LEVEL'''

        neural_data_list_fixed_AUT = df.iloc[np.where(df.index.get_level_values('trial_type')=='AUT')[0]]
        neural_data_list_fixed_originality = []
    
        patient = neural_data_list_fixed_AUT.index.get_level_values('patient').unique().values[0] #get patient from df because out of order
        patient_processed_AUT_datafile = 'scored_responses/' + patient + '_processed_data.csv'  #file that includes responses and originality scores
        AUT_scores_df = pd.read_csv(patient_processed_AUT_datafile)
        AUT_scores_df.drop(np.where(AUT_scores_df.Start.isna())[0],inplace=True) #drop rows that have nans in the times columns (those trials didnt happen)
        originality_by_trial = AUT_scores_df.originality
        ### SPLIT ORIGINALITY INTO HIGH VS LOW (1 VS 0) REL TO MEDIAN
        mid_originality = np.nanmedian(originality_by_trial)
        originality_categ = np.array(originality_by_trial>mid_originality).astype(int) #categorical 


        # ###add new index to df (originality score)

        df_by_freq_band_to_decode_AUT = neural_data_list_fixed_AUT.T
        # ###add new index to df (originality score)
        cols_frame = df_by_freq_band_to_decode_AUT.columns.to_frame()
        cols_frame['originality_cont'] = np.array(originality_by_trial)
        cols_frame['originality_cat'] = originality_categ
        df_by_freq_band_to_decode_AUT.columns = pd.MultiIndex.from_frame(cols_frame[['trial_idx', 'trial_type', 'originality_cont', 'originality_cat']]) #reorder and turn into index

        ###remove where orig = nan (where patient didnt answer, for instance)
        nan_inds = np.where(np.isnan(df_by_freq_band_to_decode_AUT.columns.get_level_values('originality_cont').values))[0]
        nan_trials = df_by_freq_band_to_decode_AUT.columns.get_level_values('trial_idx')[nan_inds]
        df_by_freq_band_to_decode_AUT = df_by_freq_band_to_decode_AUT.drop(columns = nan_trials)



        '''DECODE HIGH VS LOW ORIGINALITY (CATEGORICAL): USE SVM'''
        clf = SVC(C=1, kernel='linear', tol=1e-6)
        # clf = linear_model.RidgeClassifier()
        kf = KFold(n_splits=4)
        predictions = []
        scores = []

        clf = SVC(C=1, kernel='linear', tol=1e-6)
        kf = KFold(n_splits=5)
        answers_trial_type = []
        scores = []
        # for train_idx, test_idx in kf.split(df_by_freq_band_to_decode_AUT.columns.get_level_values('trial_idx')):
        #     print(len(train_idx))
        for trial in df_by_freq_band_to_decode_AUT.columns.get_level_values('trial_idx'):
            train = df_by_freq_band_to_decode_AUT.drop(columns=trial)
            test = df_by_freq_band_to_decode_AUT[trial]

            # train = df_by_freq_band_to_decode_AUT.iloc[:,list(train_idx)]
            # test = df_by_freq_band_to_decode_AUT.iloc[:,list(test_idx)]
            
            clf.fit(train.T, train.columns.get_level_values('originality_cat'))
            predictions.extend(clf.predict(test.T))
            score = clf.score(test.T, test.columns.get_level_values('originality_cat'))
            scores.append(score)
        cm = confusion_matrix(df_by_freq_band_to_decode_AUT.columns.get_level_values('originality_cat'), predictions)
        avg_score = np.mean(scores)
        cm_all_patients_creativity.append(cm)
        scores_all_patients_creativity.append(avg_score)
        
    scores_all_networks_AUT_BAT.update({network: np.array(scores_all_patients_AUT_BAT)*100})
    scores_all_networks_creativity.update({network: np.array(scores_all_patients_creativity)*100})

import statsmodels.stats.multitest as multi

p_values_both = []
p_values_corr_both = []
for j, scores_all_networks in enumerate([scores_all_networks_AUT_BAT, scores_all_networks_creativity]):
    f,ax = plt.subplots()
    ax.boxplot(list(scores_all_networks.values()), showfliers=False, labels = list(scores_all_networks.keys()))
    for i, scores in enumerate(list(scores_all_networks.values())):
        x = np.random.normal(i+1, scale = 0.04 ,size=len(scores))
        ax.scatter(x, scores, s = 10, color = 'k', edgecolors='k', linewidths=0.5)
    ax.axhline(50,color='k',linewidth = 0.5, linestyle='--')
    # ax.set_ylim(1,100)
    sns.despine(top = True, right = True)
    ax.set_ylabel('% decoding accuracy')
    ax.set_title(['Decoding AUT vs BAT', 'Decoding high vs low creativity'][j])
    
    p_values = []
    for i, scores in enumerate(list(scores_all_networks.values())):
        t,p = scipy.stats.ttest_1samp(scores, 50, alternative='greater')
        p_values.append(p)
        print(p)
    p_values_both.append(p_values)
    #FDR correction:
    rej, corr_p = multi.fdrcorrection(p_values, alpha=0.05, method='indep')
    p_values_corr_both.append(corr_p)


ax_auc.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
ax_auc.set_xlabel('False Positive Rate')
ax_auc.set_ylabel('True Positive Rate')
ax_auc.set_title('ROC Curve')
ax_auc.legend(loc='lower right')





avg_cm_all_patients_AUT_BAT =np.mean(cm_all_patients_AUT_BAT,axis=0)
cm_norm_all_patients_AUT_BAT = [cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] for cm in cm_all_patients_AUT_BAT]
avg_cm_norm_AUT_BAT = avg_cm_all_patients_AUT_BAT.astype('float') / avg_cm_all_patients_AUT_BAT.sum(axis=1)[:, np.newaxis]


AUT_accuracy = np.array(cm_norm_all_patients_AUT_BAT)[:,0,0] * 100
BAT_accuracy = np.array(cm_norm_all_patients_AUT_BAT)[:,1,1] * 100
accuracy_data = [AUT_accuracy, BAT_accuracy]
colors = ['darkorange', 'cornflowerblue']
f,ax = plt.subplots()
ax.boxplot(accuracy_data, showfliers=False, labels = ['AUT', 'BAT'])
for i, d in enumerate(accuracy_data):
    ax.scatter(np.random.normal(i+1, scale = 0.04 ,size=len(d)), d, s = 10, edgecolors='k', linewidths=0.5, color = colors[i])
ax.axhline(50,color='k',linewidth = 0.5, linestyle='--')
# ax.set_ylim(1,100)
sns.despine(top = True, right = True)
ax.set_ylabel('% decoding accuracy')
ax.set_title('Decode trial type from ' + network)

AUT_p = scipy.stats.ttest_1samp(AUT_accuracy, 50, alternative='greater')
BAT_p = scipy.stats.ttest_1samp(BAT_accuracy, 50, alternative='greater')
trans = ax.get_yaxis_transform()
ax.text(0.25,10,f"{AUT_p[1]:.2}",transform=trans)
ax.text(0.75,10,f"{BAT_p[1]:.2}",transform=trans)


cm_norm_all_patients_creativity = [cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] for cm in cm_all_patients_creativity]

low_accuracy = np.array(cm_norm_all_patients_creativity)[:,0,0] * 100
high_accuracy = np.array(cm_norm_all_patients_creativity)[:,1,1] * 100
accuracy_data = [low_accuracy, high_accuracy]
colors = ['sienna', 'orangered']
f,ax = plt.subplots()
ax.boxplot(accuracy_data, showfliers=False, labels = ['Low', 'High'])
for i, d in enumerate(accuracy_data):
    ax.scatter(np.random.normal(i+1, scale = 0.04 ,size=len(d)), d, s = 10, edgecolors='k', linewidths=0.5, color = colors[i])
ax.axhline(50,color='k',linewidth = 0.5, linestyle='--')
# ax.set_ylim(1,100)
sns.despine(top = True, right = True)
ax.set_ylabel('% decoding accuracy')
ax.set_title('Decode creativity from ' + network)

low_p = scipy.stats.ttest_1samp(low_accuracy, 50, alternative='greater')
high_p = scipy.stats.ttest_1samp(high_accuracy, 50, alternative='greater')
trans = ax.get_yaxis_transform()
ax.text(0.25,10,f"{high_p[1]:.2}",transform=trans)
ax.text(0.75,10,f"{low_p[1]:.2}",transform=trans)

plt.show()
plt.close()


###STATS
patients = {}
for network in networks:
    all_dat = pd.read_csv('pkl_files/store_neural_data_list_' + network + '.csv', header=[0, 1], index_col=[0, 1, 2, 3])
    pts = all_dat.index.get_level_values('patient').unique().values
    patients.update({network: pts})


scipy.stats.f_oneway(scores_all_networks_AUT_BAT['DMN'], scores_all_networks_AUT_BAT['FrontoParietal'], 
                     scores_all_networks_AUT_BAT['SomatoMotor'], scores_all_networks_AUT_BAT['Limbic'],
                     scores_all_networks_AUT_BAT['Visual'], scores_all_networks_AUT_BAT['DorsalAttention'],
                     scores_all_networks_AUT_BAT['VentralAttention'])
                     #pvalue=0.0044843209416001095
p_values = []
for key in list(scores_all_networks_AUT_BAT.keys())[1:]:
    t,p = scipy.stats.mannwhitneyu(scores_all_networks_AUT_BAT['DMN'], scores_all_networks_AUT_BAT[key])
    p_values.append(p)
rej, corr_p = multi.fdrcorrection(p_values, alpha=0.05, method='indep')


p_values = []
scores_all = []
for key in list(scores_all_networks_AUT_BAT.keys())[1:]:
    scores_all.extend(scores_all_networks_AUT_BAT[key])
t,p = scipy.stats.mannwhitneyu(scores_all_networks_AUT_BAT['DMN'], scores_all)  #pvalue=0.009




##################### DOWNSAMPLE CONTACTS: TRY DECODING FROM DIFFERENT SUBSETS OF CONTACTS ################
import random

n_contacts = 5
n_repeats = 50
from mpl_helpers import PdfPlotter
PdfPlotter('Figures/decoding_across_num_contacts.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))

for network in networks:
    all_dat = pd.read_csv('pkl_files/store_neural_data_list_' + network + '.csv', header=[0, 1], index_col=[0, 1, 2, 3])
    all_dat = pd.read_csv('pkl_files/store_neural_data_list_with_ripple' + network + '.csv', header=[0, 1], index_col=[0, 1, 2, 3])

    patient_dfs = [group for _, group in all_dat.groupby(level='patient')]
    cleaned_dfs = [df.dropna(axis=1, how='all') for df in patient_dfs]

    neural_data_list_fixed = cleaned_dfs
    df_by_freq_band_all = [neural_data.unstack('time') for neural_data in neural_data_list_fixed]


    scores_downsampled_all_patients = []
    for p, df in enumerate(df_by_freq_band_all):
        print(p)
        df_by_freq_band_to_decode = df.T


        all_channels = df_by_freq_band_to_decode.index.get_level_values('channel').unique()

        scores_downsampled_for_diff_num_contacts = []
        for n_contacts in range(1, len(all_channels)+1):

            scores_downsampled = []


            for i in range(n_repeats):
                sampled_channels = random.sample(list(all_channels), n_contacts)
                subset = df_by_freq_band_to_decode.loc[df_by_freq_band_to_decode.index.get_level_values('channel').isin(sampled_channels)]
                
                answers_trial_type = []
                scores = []
                true_labels = []
                
                kf = KFold(n_splits=4)
                for train_idx, test_idx in kf.split(subset.columns.get_level_values('trial_idx')):
                    train = subset.iloc[:, list(train_idx)]
                    test = subset.iloc[:, list(test_idx)]
                    clf = SVC(C=1, kernel='linear', tol=1e-6)
                    clf.fit(train.T, train.columns.get_level_values('trial_type'))
                    answers_trial_type.extend(clf.predict(test.T))
                    scores.append(clf.score(test.T, test.columns.get_level_values('trial_type')))
                
                scores_downsampled.append(np.mean(scores))
            scores_downsampled_for_diff_num_contacts.append(np.mean(scores_downsampled))
        scores_downsampled_all_patients.append(scores_downsampled_for_diff_num_contacts)


    f,ax = plt.subplots()
    for scores in scores_downsampled_all_patients:
        ax.plot(scores)
    ax.set_xlabel('number contacts')
    ax.set_ylabel('decoding accuracy')
    ax.set_title('decoding performance for subsets of contacts (' + network + ')')

plt.show()
plt.close()