'''add patients to all_dat file.
   ***** Include AUT, OCT, and BAT trials (add_patient_to_dat.py only includes AUT and BAT, and includes ripple patients)
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

import sys
sys.path.append('../')

import DMN_LFP_funcs
import beh_funcs


import decoding_funcs



network = input("Choose cortical network (DMN, FrontoParietal, SomatoMotor, Limbic, Visual, DorsalAttention, VentralAttention, or all): ")
if network == 'all':
    networks = ['DMN', 'FrontoParietal', 'SomatoMotor', 'Limbic', 'Visual', 'DorsalAttention', 'VentralAttention']
else:
    networks = [network]

# networks = ['FrontoParietal', 'SomatoMotor', 'Limbic', 'Visual', 'DorsalAttention', 'VentralAttention']


for network in networks:
    for patient, data_loc in patients_data:
        print(patient)
        

        csv_file = 'pkl_files/store_neural_data_list_all_trialtypes_' + network + '.csv'

        try:
            all_dat = pd.read_csv(csv_file, header=[0, 1], index_col=[0, 1, 2, 3])

            patient_dfs = [group for _, group in all_dat.groupby(level='patient')]
            cleaned_dfs = [df.dropna(axis=1, how='all') for df in patient_dfs]

            neural_data_list_fixed = cleaned_dfs

            if patient in all_dat.index.get_level_values('patient').unique().values:
                print('patient already added')
                continue

        except: #all_dat file doesn't exist yet!
            print('all_dat file does not exist')
            neural_data_list_fixed = []


        #get mne file
        raw = DMN_LFP_funcs.get_raw(patient,data_loc) #get raw mne (with already marked bad channels that were selected manually)
        raw =  DMN_LFP_funcs.do_preprocessing(patient,data_loc,raw)


        # df_by_freq_band = make_df(raw, trial_info_df, good_chans_DMN, regions_list)
        df_by_freq_band = decoding_funcs.make_power_df(patient,data_loc, raw=raw, network = network, rotated = False)
        if df_by_freq_band.empty:
            continue     
        neural_data = df_by_freq_band.T


        new_dat = pd.concat([neural_data], keys=np.arange(len(neural_data))) 
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

