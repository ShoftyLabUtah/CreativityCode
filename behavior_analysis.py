'''
CHECK PERFORMANCE ACROSS PATIENTS; VISUALIZE BEHAVIORAL DATA with polar plot
For Figure 1 b and c
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
import re
import mne
import scipy
import DMN_LFP_funcs
import beh_funcs

import sys
sys.path.append('../')



from mpl_helpers import PdfPlotter
PdfPlotter('Figures/behavior_visualization.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))



import patient_params
patients_data = patient_params.patients_data
include_ripple = input("Include ripple patients (y/n): ")
if include_ripple:
    patients_info_ripple = patient_params.patients_data_ripple
    patients_data_ripple = patients_info_ripple
    patients_data = patients_data + patients_data_ripple


def normalize(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data

f,ax = plt.subplots(figsize=(6, 3))
originality_all_patients = []
for patient, data_loc in patients_data:
    print(patient)
    if patient=='UIC202420':
        continue
    if '20250103-115359' in data_loc:
        continue
    patient_processed_AUT_datafile = 'scored_responses/' + patient + '_processed_data.csv'  #file that includes responses and originality scores:
    if '20241231-144952' in data_loc:
        patient_processed_AUT_datafile = 'scored_responses/' + patient + '_processed_data_day1.csv'  

    ####get trial info from matlab file
    trial_info_df = beh_funcs.get_trial_info_df(data_loc)

    ###Find where practice trials end and real session starts. Practice AUT and OCT trials start with promot "key"; BAT trials start with '10 + 1 = '
    AUT_trial_df = beh_funcs.trial_type_subset_df(trial_info_df, 'AUT', practice_prompts=[['key'], ['10 + 1 = ']]) #just the AUT trials
    #get responses and originality scores: (generated from https://openscoring.du.edu)
    try:
        AUT_scores_df = pd.read_csv(patient_processed_AUT_datafile)
    except:
        print('Not yet scored... generating template and breaking here!')
        beh_funcs.create_csv_template(patient, data_loc, 'AUT')
        1/0
    # import pdb; pdb.set_trace()
    AUT_scores_df_dropna = AUT_scores_df.dropna()
    AUT_scores_df_sorted = AUT_scores_df_dropna.sort_values('originality') #sort by originality 
    num_trials_per_third = int(len(AUT_scores_df_sorted)/3) #low will be bottom 1/3; high is top 1/3
    low_orig_df = AUT_scores_df_sorted[0:num_trials_per_third] #low creative trials
    high_orig_df = AUT_scores_df_sorted[-num_trials_per_third:] #high creative trials

    originality_by_trial = AUT_scores_df.originality
    originality_by_trial_norm = normalize(AUT_scores_df.originality)

    originality_all_patients.append(originality_by_trial.values)

    '''###HISTOGRAM'''
    # sns.histplot(data = originality_by_trial,kde=True,kde_kws={'bw_method':0.1},bins=20,ax = ax, alpha = 0.25)
    # sns.histplot(data = originality_by_trial_norm,kde=True,kde_kws={'bw_method':0.1},bins=20,ax = ax2, alpha = 0.25)

    '''KDE PLOT'''
    # sns.kdeplot(data=originality_by_trial, bw_adjust=1, ax=ax)



    # f,ax=plt.subplots()
    # ax.scatter(originality_by_trial,np.zeros_like(originality_by_trial))

    for i, orig in enumerate(originality_by_trial):
        if np.isnan(orig):
            print('orig is nan')
            continue
        item = AUT_scores_df.get('Item', AUT_scores_df.get('item')).iloc[i].lower()
        if item == 'pencil':
            print('pencil!')
            ax.scatter(orig, 0, color='b')
            plt.annotate(item + ': '+ AUT_scores_df.response[i], [originality_by_trial[i],0.01], ha='center', va='bottom', rotation=90, fontsize=7)
        # if item == 'saxophone':
        #     ax.scatter(orig, 0, color='orange')
        #     plt.annotate(item + ': '+ AUT_scores_df.response[i], [originality_by_trial[i],0.01], ha='center', va='bottom', rotation=90, fontsize=7)
        # if item == 'lens':
        #     ax.scatter(orig, 0, color='green')
        #     plt.annotate(item + ': '+ AUT_scores_df.response[i], [originality_by_trial[i],0.01], ha='center', va='bottom', rotation=90, fontsize=7)
        # if item == 'rope':
        #     ax.scatter(orig, 0, color='grey')
        #     plt.annotate(item + ': '+ AUT_scores_df.response[i], [originality_by_trial[i],0.01], ha='center', va='bottom', rotation=90, fontsize=7)

ax.set_yticks([])  # remove y-axis ticks
ax.set_xlabel("Semantic distance")
ax.set_title("Originality scores")
ax.set_ylim([-0.01,0.1])
ax.set_xlim([0.5,1])
ax.axhline(0,linestyle='--',color='k')
plt.show()

flattened_scores = np.concatenate([arr[~np.isnan(arr)] for arr in originality_all_patients])

# Calculate global mean and std
mean_score = np.mean(flattened_scores)
std_score = np.std(flattened_scores)



f, ax = plt.subplots(subplot_kw={'projection': 'polar'})
x = np.arange(len(originality_all_patients))
theta_full_circle = np.linspace(0.0, 2 * np.pi, len(x), endpoint=False)  # Convert x to radians

for i, orig in enumerate(originality_all_patients):
    orig_clean = np.where(np.isnan(orig), 0, orig)
    theta = theta_full_circle[i]
    theta += np.random.normal(0, 0.1, size=len(orig_clean))

    ax.scatter(theta, orig_clean, s = 10, edgecolors='k', linewidths=0.5)
    # ax.scatter(np.random.normal(i+1, scale = 0.04 ,size=len(orig_clean)), orig_clean, s = 10, edgecolors='k', linewidths=0.5)
# plt.axis('off')
plt.show()


