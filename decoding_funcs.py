''' Useful functions for decoding scripts
Makes df_by_freq_band, which gets used to make the all_dat files


make_df Makes dataframe to use for decoding trial type:

index names : ['freq_band', 'channel', 'time'] (time is time in trial (0-15))

e.g.,

trial_idx                                               0             1             2   ...             42            43            44
trial_type                                             OCT           OCT           OCT  ...            AUT           AUT           AUT
      channel                           time                                            ...                                           
theta 0 Left AIns anterior insula       0     56437.557444  20612.540977  30721.997656  ...  132979.529555  64204.854464  19158.431594
                                        1     20085.593652  11864.705225  11510.959299  ...   34504.037810  50158.252125  37734.619064
                                        2      9535.451832  45406.779732  20554.661469  ...   31691.459110  21038.246566  33984.179758
                                        3     52114.725906  16826.697765  24499.870373  ...   21857.705510   3924.577116  20610.412262
                                        4     26308.542113  49267.989851  48755.589347  ...   38215.113341  17311.936710  17730.546448
...                                                    ...           ...           ...  ...            ...           ...           ...
beta  9 Right MTG middle temporal gyrus 11     5563.981269   4380.107032   7785.509212  ...    5007.482129  14762.877776   3631.211797
                                        12     8733.654955   4896.477330   7164.917879  ...    4114.029605   4538.921574   5872.684740
                                        13     7542.463533   2750.735639   9434.092167  ...    2964.721891   5633.601424   7023.151855
                                        14     6648.394825   5441.813152   7645.881827  ...    4799.363576   4569.798841   8836.329321
                                        15     4121.378455   9294.390548   8866.310176  ...    6895.392857   7798.737007   9954.490441

This is the rotated dataframe. I.e. default (rotated = True). 

If rotated = False, then each column is a timepoint:
trial_idx                          0                                               ...             44                                            
trial_type                        AUT                                              ...            OCT                                            
time                               0              1             2              3   ...             12             13            14             15
           channel                                                                 ...                                                           
theta      0 Right MTG  342518.681371  509285.244999  1.320108e+06  996447.630980  ...  799594.096670  832234.353646  1.132081e+06  798549.771785
           1 Right MTG   79243.621134   65493.878535  1.867797e+05   95219.137670  ...   73832.688467  153299.058206  1.355485e+05  130290.566987
           2 Left MTG    18212.339256   12641.595111  5.196461e+04   26940.050240  ...   42463.749270   13392.488205  4.645091e+04   23389.913881
           3 Left MTG    40297.253131   17018.827894  5.114820e+04   29314.416539  ...   50028.997996   20653.275743  6.962250e+04   36984.229217
           4 Left MTG    42914.749527   25737.836090  4.427464e+04   32684.013215  ...   41038.519064   27068.782809  9.338796e+04   53192.687945
           5 Left MTG    11727.638568   17810.730090  2.757599e+04   21807.088056  ...   33094.002956   17763.166359  4.372844e+04   41950.446971

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
import sys
sys.path.append('../')

import DMN_LFP_funcs
import beh_funcs

from sklearn.svm import SVC 
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn import linear_model




import patient_params
patients_data = patient_params.patients_data




def make_power_df(patient,data_loc,raw=None, network = "DMN", window_size=1, rotated=True):
    
    if raw is None:
        #get mne file if not passed to function
        raw = DMN_LFP_funcs.get_raw(patient,data_loc) #get raw mne (with already marked bad channels that were selected manually)
        raw = DMN_LFP_funcs.do_preprocessing(patient,data_loc,raw)

    good_chans_all, good_chans_subset, good_regions_subset, good_networks_subset = DMN_LFP_funcs.get_label_info_network(patient,data_loc,raw, which_chans_str = network)

    if good_chans_subset==[]: #no channels of this network
        print('There are no channels in ' + network + ' network')
        return(pd.DataFrame())
 

    trial_info_df = beh_funcs.get_trial_info_df(data_loc)

    
    trial_info_df_fixed = beh_funcs.trial_type_subset_df(trial_info_df, 'all') #cut out practice trials
    start_times_ms, end_times_ms, trial_start_times_ms = beh_funcs.get_start_end_times_ms(trial_info_df_fixed, 'all', data_loc=data_loc)
    start_times_sec = start_times_ms/1000
    end_times_sec = end_times_ms/1000 + 6

    #get LFP power for each freq band
    theta_sig, theta_power, _ = DMN_LFP_funcs.get_power_over_time_from_unfiltered(raw, start_times_ms, end_times_ms + (6*1000),good_chans_subset,'theta',window_size=window_size)
    high_gamma_sig, high_gamma_power, _ = DMN_LFP_funcs.get_power_over_time_from_unfiltered(raw, start_times_ms, end_times_ms + (6*1000),good_chans_subset,'high_gamma',window_size=window_size)
    gamma_sig, gamma_power, _ = DMN_LFP_funcs.get_power_over_time_from_unfiltered(raw, start_times_ms, end_times_ms + (6*1000),good_chans_subset,'gamma',window_size=window_size)
    beta_sig, beta_power, _ = DMN_LFP_funcs.get_power_over_time_from_unfiltered(raw, start_times_ms, end_times_ms + (6*1000),good_chans_subset,'beta',window_size=window_size)

    bins_in_trial = theta_power.shape[0]
    num_trials = theta_power.shape[1]
    len_session_bins = bins_in_trial * num_trials


    #make array that is the length of the total num frames in the session
    #so that I can label each frame as being in a given trial
    trial_numbers_array = np.tile(np.nan, len_session_bins) #initiaze to nans
    frame = np.tile(np.nan, len_session_bins) #initiaze to nans

    trial_types_array_original = np.concatenate([[trial_type] * bins_in_trial*3 for trial_type in trial_info_df_fixed.stimType])
    trial_types_array = trial_types_array_original
    

    ###try decoding AUT vs other
    # trial_types_array = ['AUT' if tr=='AUT' else 'other' for tr in trial_types_array_original]

    trial_numbers_array = np.repeat(np.arange(num_trials),bins_in_trial)
    time_in_trial = np.tile(np.arange(bins_in_trial),num_trials)

    
    if int(data_loc[31:37]) >= 202413 and int(data_loc[31:37]) < 202418: ##after patient 9, when remove OCT, trial_info_df has extra "[]" brackets...
        trial_types_array = [stim[0] for stim in trial_types_array]
    
    
    #take the arrays and make them into pandas indices for the dataframe
    #first make MultiIndex object
    columns = pd.MultiIndex.from_tuples(list(zip(trial_numbers_array, trial_types_array, time_in_trial)), names = ['trial_idx', 'trial_type', 'time'])

    #get region names for indices of df
    chan_names = [good_chans_subset[i] +' ' + name for i, name in enumerate(good_regions_subset)]


    theta_power_reshape = theta_power.transpose(1,0,2).reshape(len_session_bins, theta_power.shape[-1]) #concat all timepoints for all trials
    theta_dataframe = pd.DataFrame(theta_power_reshape.T, index = chan_names, columns = columns) #make dataframe: rows are cell ids, columns are frame number (and also trial ind and response type)
    theta_dataframe.index.names = ['channel']
    theta_dataframe_rotated = theta_dataframe.T.stack().unstack(['trial_idx', 'trial_type']).groupby(['channel', 'time']).sum()

    high_gamma_power_reshape = high_gamma_power.transpose(1,0,2).reshape(len_session_bins, high_gamma_power.shape[-1]) #concat all timepoints for all trials
    high_gamma_dataframe = pd.DataFrame(high_gamma_power_reshape.T, index = chan_names, columns = columns) #make dataframe: rows are cell ids, columns are frame number (and also trial ind and response type)
    high_gamma_dataframe.index.names = ['channel']
    high_gamma_dataframe_rotated = high_gamma_dataframe.T.stack().unstack(['trial_idx', 'trial_type']).groupby(['channel', 'time']).sum()
    
    gamma_power_reshape = gamma_power.transpose(1,0,2).reshape(len_session_bins, gamma_power.shape[-1]) #concat all timepoints for all trials
    gamma_dataframe = pd.DataFrame(gamma_power_reshape.T, index = chan_names, columns = columns) #make dataframe: rows are cell ids, columns are frame number (and also trial ind and response type)
    gamma_dataframe.index.names = ['channel']
    gamma_dataframe_rotated = gamma_dataframe.T.stack().unstack(['trial_idx', 'trial_type']).groupby(['channel', 'time']).sum()

    beta_power_reshape = beta_power.transpose(1,0,2).reshape(len_session_bins, beta_power.shape[-1]) #concat all timepoints for all trials
    beta_dataframe = pd.DataFrame(beta_power_reshape.T, index = chan_names, columns = columns) #make dataframe: rows are cell ids, columns are frame number (and also trial ind and response type)
    beta_dataframe.index.names = ['channel']
    beta_dataframe_rotated = beta_dataframe.T.stack().unstack(['trial_idx', 'trial_type']).groupby(['channel', 'time']).sum()

    if rotated: #default
        df_by_freq_band= pd.concat([theta_dataframe_rotated, high_gamma_dataframe_rotated, gamma_dataframe_rotated, beta_dataframe_rotated], keys=['theta','high_gamma', 'gamma', 'beta'])
        df_by_freq_band.index.names = ['freq_band', 'channel', 'time']

    else: 
        df_by_freq_band = pd.concat([theta_dataframe, high_gamma_dataframe, gamma_dataframe, beta_dataframe], keys=['theta','high_gamma', 'gamma', 'beta'])

    return (df_by_freq_band)








def make_power_df_ripple(patient,data_loc,NIP_start_ms, raw=None, network = "DMN", window_size=1, rotated=True):
    
    if raw is None:
        #get mne file if not passed to function
        raw = DMN_LFP_funcs.get_raw(patient,data_loc) #get raw mne (with already marked bad channels that were selected manually)
        raw = DMN_LFP_funcs.do_preprocessing(patient,data_loc,raw)

    good_chans_all, good_chans_subset, good_regions_subset, good_networks_subset = DMN_LFP_funcs.get_label_info_network(patient,data_loc,raw, which_chans_str = network)

    if good_chans_subset==[]: #no channels of this network
        print('There are no channels in ' + network + ' network')
        return(pd.DataFrame())
 

    trial_info_df = beh_funcs.get_trial_info_df(data_loc)
    if patient=='UIC202421':
        trial_info_df = trial_info_df.loc[4:]
    trial_info_df = trial_info_df[~trial_info_df.stimFlag]
    if patient=='UIC202420':
        trial_info_df.drop(12, inplace=True) #outlier trial; something weird happened.



    
    trial_info_df_fixed = beh_funcs.trial_type_subset_df(trial_info_df, 'all') #cut out practice trials
    start_times_ms, end_times_ms, trial_start_times_ms = beh_funcs.get_start_end_times_ms(trial_info_df_fixed, 'all')
    start_times_sec = start_times_ms/1000
    end_times_sec = end_times_ms/1000 + 6

    theta_sig, theta_power, _ = DMN_LFP_funcs.get_power_over_time_from_unfiltered(raw, start_times_ms - NIP_start_ms, end_times_ms - NIP_start_ms + (6*1000),good_chans_subset,'theta',window_size=window_size)
    high_gamma_sig, high_gamma_power, _ = DMN_LFP_funcs.get_power_over_time_from_unfiltered(raw, start_times_ms - NIP_start_ms, end_times_ms - NIP_start_ms+ (6*1000),good_chans_subset,'high_gamma',window_size=window_size)
    gamma_sig, gamma_power, _ = DMN_LFP_funcs.get_power_over_time_from_unfiltered(raw, start_times_ms -NIP_start_ms, end_times_ms -NIP_start_ms + (6*1000),good_chans_subset,'gamma',window_size=window_size)
    beta_sig, beta_power, _ = DMN_LFP_funcs.get_power_over_time_from_unfiltered(raw, start_times_ms-NIP_start_ms, end_times_ms-NIP_start_ms + (6*1000),good_chans_subset,'beta',window_size=window_size)

    bins_in_trial = theta_power.shape[0]
    num_trials = theta_power.shape[1]
    len_session_bins = bins_in_trial * num_trials


    #make array that is the length of the total num frames in the session
    #so that I can label each frame as being in a given trial
    trial_numbers_array = np.tile(np.nan, len_session_bins) #initiaze to nans
    frame = np.tile(np.nan, len_session_bins) #initiaze to nans

    trial_types_array_original = np.concatenate([[trial_type] * bins_in_trial*3 for trial_type in trial_info_df_fixed.stimType])
    trial_types_array = trial_types_array_original
    

    ###try decoding AUT vs other
    # trial_types_array = ['AUT' if tr=='AUT' else 'other' for tr in trial_types_array_original]

    trial_numbers_array = np.repeat(np.arange(num_trials),bins_in_trial)
    time_in_trial = np.tile(np.arange(bins_in_trial),num_trials)

    
    if int(data_loc[31:37]) >= 202413 and int(data_loc[31:37]) < 202418: ##after patient 9, when remove OCT, trial_info_df has extra "[]" brackets...
        trial_types_array = [stim[0] for stim in trial_types_array]
    
    
    #take the arrays and make them into pandas indices for the dataframe
    #first make MultiIndex object
    columns = pd.MultiIndex.from_tuples(list(zip(trial_numbers_array, trial_types_array, time_in_trial)), names = ['trial_idx', 'trial_type', 'time'])

    #get region names for indices of df
    chan_names = [good_chans_subset[i] +' ' + name for i, name in enumerate(good_regions_subset)]

    theta_power_reshape = theta_power.transpose(1,0,2).reshape(len_session_bins, theta_power.shape[-1]) #concat all timepoints for all trials
    theta_dataframe = pd.DataFrame(theta_power_reshape.T, index = chan_names, columns = columns) #make dataframe: rows are cell ids, columns are frame number (and also trial ind and response type)
    theta_dataframe.index.names = ['channel']
    theta_dataframe_rotated = theta_dataframe.T.stack().unstack(['trial_idx', 'trial_type']).groupby(['channel', 'time']).sum()

    high_gamma_power_reshape = high_gamma_power.transpose(1,0,2).reshape(len_session_bins, high_gamma_power.shape[-1]) #concat all timepoints for all trials
    high_gamma_dataframe = pd.DataFrame(high_gamma_power_reshape.T, index = chan_names, columns = columns) #make dataframe: rows are cell ids, columns are frame number (and also trial ind and response type)
    high_gamma_dataframe.index.names = ['channel']
    high_gamma_dataframe_rotated = high_gamma_dataframe.T.stack().unstack(['trial_idx', 'trial_type']).groupby(['channel', 'time']).sum()
    
    gamma_power_reshape = gamma_power.transpose(1,0,2).reshape(len_session_bins, gamma_power.shape[-1]) #concat all timepoints for all trials
    gamma_dataframe = pd.DataFrame(gamma_power_reshape.T, index = chan_names, columns = columns) #make dataframe: rows are cell ids, columns are frame number (and also trial ind and response type)
    gamma_dataframe.index.names = ['channel']
    gamma_dataframe_rotated = gamma_dataframe.T.stack().unstack(['trial_idx', 'trial_type']).groupby(['channel', 'time']).sum()

    beta_power_reshape = beta_power.transpose(1,0,2).reshape(len_session_bins, beta_power.shape[-1]) #concat all timepoints for all trials
    beta_dataframe = pd.DataFrame(beta_power_reshape.T, index = chan_names, columns = columns) #make dataframe: rows are cell ids, columns are frame number (and also trial ind and response type)
    beta_dataframe.index.names = ['channel']
    beta_dataframe_rotated = beta_dataframe.T.stack().unstack(['trial_idx', 'trial_type']).groupby(['channel', 'time']).sum()

    df_by_freq_band = pd.concat([theta_dataframe_rotated, high_gamma_dataframe_rotated, gamma_dataframe_rotated, beta_dataframe_rotated], keys=['theta','high_gamma', 'gamma', 'beta'])

    df_by_freq_band.index.names = ['freq_band', 'channel', 'time']
    return (df_by_freq_band)