'''add patients to all_dat file
This is the file that contains the LFP power across frequency bands, for a given network. 
Code loops through networks and adds patient data to each

Choose "ripple" if include patients recorded with Ripple. Choose "blackrock" if only include blackrock recordings.

'''


import numpy as np
from neo import io
import quantities as pq
import glob   
import matplotlib.pyplot as plt
import pandas as pd
import itertools as it
import random
import seaborn as sns
import mne
import scipy

import DMN_LFP_funcs
import beh_funcs


import decoding_funcs

import sys
sys.path.append('../')


blackrock_or_ripple = input("add to normal dat df (b), or to ripple (r)? : ") #


for network in ['DMN', 'FrontoParietal', 'SomatoMotor', 'Limbic', 'Visual', 'DorsalAttention', 'VentralAttention']:

    if blackrock_or_ripple == 'r':
        csv_file = 'pkl_files/store_neural_data_list_with_ripple' + network + '.csv'
    else:
        csv_file = 'pkl_files/store_neural_data_list_' + network + '.csv'
    all_dat = pd.read_csv(csv_file, header=[0, 1], index_col=[0, 1, 2, 3])
    patient_dfs = [group for _, group in all_dat.groupby(level='patient')]
    cleaned_dfs = [df.dropna(axis=1, how='all') for df in patient_dfs]

    neural_data_list_fixed = cleaned_dfs
    discrete_labels = [1*(neural_data.index.get_level_values('trial_type')=='AUT') for neural_data in neural_data_list_fixed] 

    if patient in all_dat.index.get_level_values('patient').unique().values:
        continue

    #get mne file
    raw = DMN_LFP_funcs.get_raw(patient,data_loc) #get raw mne (with already marked bad channels that were selected manually)
    raw =  DMN_LFP_funcs.do_preprocessing(patient,data_loc,raw)


    # df_by_freq_band = make_df(raw, trial_info_df, good_chans_DMN, regions_list)
    df_by_freq_band = decoding_funcs.make_power_df(patient,data_loc, raw=raw, network = network, rotated = False)
    if df_by_freq_band.empty:
        continue     
    neural_data = df_by_freq_band.T


    ####FIX LABELS TO MAKE SURE ONLY LOOKING AT TRIALS THAT ARE AUT OR BAT
    inds_to_decode = np.where((neural_data.index.get_level_values('trial_type')=='AUT') | 
        (neural_data.index.get_level_values('trial_type')=='BAT'))[0]
    neural_data_to_decode = neural_data.iloc[inds_to_decode]
    discrete_labels.append(1*(neural_data_to_decode.index.get_level_values('trial_type')=='AUT'))

    new_dat = pd.concat([neural_data_to_decode], keys=np.arange(len(neural_data_to_decode))) 
    pt_index = pd.Series(patient, name='patient').reindex(new_dat.index.get_level_values(0))
    old_index = new_dat.index.to_frame()
    pt_index.index = old_index.index
    # new_index = pd.MultiIndex.from_frame(pd.concat((pt_index, old_index.drop('patient', axis=1)), axis=1))
    new_index = pd.MultiIndex.from_frame(pd.concat((pt_index, old_index.drop(0, axis=1)), axis=1))
    new_dat.index = new_index

    neural_data_list_fixed.append(new_dat)




    all_dat = pd.concat(neural_data_list_fixed, keys=np.arange(len(neural_data_list_fixed))) #concat patients

    fixed_idx = pd.MultiIndex.from_frame(all_dat.index.to_frame().drop(0, axis=1))
    all_dat.index = fixed_idx
    all_dat.to_csv(csv_file)



###remove ripple patients if added accidentally 

# for network in ['DMN', 'FrontoParietal', 'SomatoMotor', 'Limbic', 'Visual', 'DorsalAttention', 'VentralAttention']:

#     all_dat = pd.read_csv('pkl_files/store_neural_data_list_' + network + '.csv', header=[0, 1], index_col=[0, 1, 2, 3])

#     for rip_pat in ['UIC202418', 'UIC202420', 'UIC202421', 'UIC202422b']:
#         try:
#             all_dat.drop(rip_pat, inplace=True)
#         except:
#             continue

#     all_dat.to_csv('pkl_files/store_neural_data_list_' + network + '.csv')
