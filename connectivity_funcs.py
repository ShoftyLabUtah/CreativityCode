"""LFP analysis for AUT task
POWER CORRELLATIONS AND COHERENCE

turned analysis entirely into functions, so that can call for each patient and frequency band

"""

import numpy as np
from neo import io
import quantities as pq
import glob   
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import mat73
import itertools as it
import random
import seaborn as sns
import re
import mne
import scipy
import neurodsp
from neurodsp import timefrequency

import DMN_LFP_funcs
import beh_funcs

import sys
sys.path.append('../')


def get_base_corr_power(raw, start_times, end_times, trial_start_times, freq_band,which_chans):
    '''#filter the sig into the desired freq band; get avg baseline power for each trial; then baseline correct'''

    filtered_signal, power_over_time, power_over_time_db = DMN_LFP_funcs.get_power_over_time_from_unfiltered(raw, start_times, end_times, which_chans, freq_band) ##if want end times to be end of response, pass in "end_times_ms + (6*1000)"
    baseline_power, baseline_power_db = DMN_LFP_funcs.get_power_for_trials_subset(filtered_signal, trial_start_times - (2 * 1000), trial_start_times - (1*1000), which_chans, freq_band)
    ##BASELINE CORRECT:
    baseline_correction_method = DMN_LFP_funcs.baseline_correct_percent_method_1
    power_base_corrected_over_time = baseline_correction_method(power_over_time, baseline_power, trial_start_times, start_times)
    return (power_base_corrected_over_time)

def reorder(all_chans_data, good_chans_all, good_chans_DMN, good_regions):
    '''###REORDER SO DMN IS FIRST, BUT OTHERWISE, ALL CONTACTS IN SAME REGION ARE GROUPED TOGETHER '''
    
    ###SEPERATE DMN FROM REMAINING REGIONS AND DATA
    DMN_inds = [list(good_chans_all).index(g) for g in good_chans_DMN]
    remaining_inds = [list(good_chans_all).index(g) for g in good_chans_all if g not in good_chans_DMN]
    good_regions_DMN = np.array(good_regions)[DMN_inds]
    good_regions_remaining = np.array(good_regions)[remaining_inds]
    DMN_data = all_chans_data[:,:,DMN_inds]
    remaining_data = all_chans_data[:,:,remaining_inds]

    ###REORDER BOTH DMN AND REMAINING REGIONS/DATA SO THAT CONTACTS IN SAME REGION ARE GROUPED TOGETHER 
    DMN_regions_reorder_idx = np.argsort(good_regions_DMN)
    DMN_regions_reorder = good_regions_DMN[DMN_regions_reorder_idx]
    DMN_data_reorder = DMN_data[:,:,DMN_regions_reorder_idx]
    remining_regions_reorder_idx = np.argsort(good_regions_remaining)
    remaining_regions_reorder = good_regions_remaining[remining_regions_reorder_idx]
    remaining_data_reorder = remaining_data[:,:,remining_regions_reorder_idx]

    return (DMN_regions_reorder,DMN_data_reorder,remaining_regions_reorder,remaining_data_reorder)

def get_PPL_df_from_epochs(signal,index,columns,freq_band,fs):
    '''####COHERENCE / PHASE-LOCKING
    ####GENERATES DATAFRAME WITH PPL VALUE FOR EACH PAIR (ROW,COL) OF CHANNELS
    ####SIGNAL IS ALREADY FILTERED IF NECESSARY. INDEX AND COLUMNS ARE ALL THE CHANNELS
    ####SIGNAL FORM IS like EPOCHS: time x trials x channels
    
    ###Returns:
        df of PPL values
        df of angle values (the angle between the phases: retains sign (ie retain info about which signal leads)
            use this for circular stats
        '''
        

    low_f, high_f = DMN_LFP_funcs.get_freq_range(freq_band)

    #empty df, same shape/rows/cols as corr_df (ie rows and cols are all contacts)
    phase_df = pd.DataFrame(index = index, columns = columns)
    angle_df = pd.DataFrame(index = index, columns = columns)

    #get phase difference for each pair of chans
    counter = 0
    for i,ch1 in enumerate(phase_df):
        counter+=1
        for j,ch2 in enumerate(phase_df):
            if j<i:
                continue
            print(counter)
            phase1_all_trials = []
            phase2_all_trials = []
            for t in np.arange(signal.shape[1]):
                sig1 = signal[:,t,i] #1 channel; current trial
                pha1 = neurodsp.timefrequency.phase_by_time(sig1,fs,f_range=(low_f, high_f)) 
                phase1_all_trials.append(pha1)
                sig2 = signal[:,t,j] #2 channel; current trial
                pha2 = neurodsp.timefrequency.phase_by_time(sig2,fs,f_range=(low_f, high_f)) 
                phase2_all_trials.append(pha2)
            phase_diff = np.array(phase1_all_trials) - np.array(phase2_all_trials)

            #get circular difference 
            phase_diff_circ = []
            for d in np.concatenate(phase_diff):
                if d > np.pi:
                    d = -(2*np.pi - d) #negative to flip sign, and therefore keep the same position on the unit circle
                if d < -np.pi:
                    d = -(-2*np.pi - d)
                phase_diff_circ.append(d)
            # PPL = np.squeeze(np.abs(np.nanmean(np.squeeze(np.exp(1j*np.array(phase_diff_circ))),axis=0))) ##dont actually need squeeze
            PPL = np.abs(np.nanmean(np.exp(1j*np.array(phase_diff_circ)))) 
            angle = np.angle(np.nanmean(np.exp(1j*np.array(phase_diff_circ)))) ##the angle between the phases: retains sign (ie retain info about which signal leads)
            phase_df.iat[i,j] = PPL #fill in df with PPL value
            angle_df.iat[i,j] = angle #fill in df with PPL value
    phase_df.fillna(phase_df.T, inplace = True) ##since phase_df was only the upper right triangle, use transpose to fill in the rest symmetrically
    phase_df.drop(columns='Unknown', inplace=True, errors="ignore") #remove "unknown" network
    phase_df.drop(index='Unknown', inplace=True, errors="ignore") #remove "unknown" network

    angle_df.fillna(angle_df.T, inplace = True) ##since phase_df was only the upper right triangle, use transpose to fill in the rest symmetrically
    angle_df.drop(columns='Unknown', inplace=True, errors="ignore") #remove "unknown" network
    angle_df.drop(index='Unknown', inplace=True, errors="ignore") #remove "unknown" network

    return (phase_df, angle_df)

def get_PPL_by_trial(signal,networks,freq_band,fs):
    '''####COHERENCE / PHASE-LOCKING
    ####do this by trials (get PPL value for each trial for each pair of chans)

    Do this by applying calv_plv func to df of arguments (args_df), which contains combinations of channel a, channel b, and trial
    '''
    contacts = np.arange(signal.shape[-1])
    trials = np.arange(signal.shape[1])

    df = (pd.DataFrame(1, index=contacts, columns=trials).cumsum(1)-1)
    # df = (pd.DataFrame(1, index=np.arange(48), columns=np.arange(15)).cumsum(1)-1)
    df.index.name = 'contact'
    df.columns.name = 'trial'

    # df=df.stack()

    ###GET ALL COMBINATIONS OF CHANNELS (EACH CHANNEL PAIR)
    contact_comb = list(it.combinations(df.index.unique('contact'), 2))
    # df = df.unstack('trial')
    df = pd.DataFrame(contact_comb) #makes df where channel pairs are the columns (column 0 = contact a; col 1 = contact b)

    df['val']  = 0 #adds column of zeros
    df = df.set_index([0, 1]) #MAKES THE 2 CHANS IN EACH PAIR THE INDICES
    df.index.names = ['contact_a', 'contact_b']
    # df = df.apply(lambda x: pd.Series(np.arange(15)), axis=1)
    df = df.apply(lambda x: pd.Series(trials), axis=1) ###PUTS TRIALS IN THE DF, EACH COLUMN IS A TRIAL
    df.columns.name = 'trial'

    # args_df = df.stack().index.to_frame()

    args_df = df.stack().index.to_frame().reset_index(drop=True) #now df is just 3 columnns: contact a, contact b, and trial. So every combination is included

    ### MAKE THE FUNCTION THAT CALCULATES PHASE LOCKING; WILL BE APPLIED TO ARGS_DF RATHER THAN LOOPING THROUGH EACH CHAN PAIR AND TRIAL
    def calc_ppl(args_df_row, sig = signal, freq_band = freq_band, fs=fs):
        # print(args_df_row.name)
        low_f, high_f = DMN_LFP_funcs.get_freq_range(freq_band)
        sig1 = sig[:, args_df_row['trial'], args_df_row['contact_a']]  
        pha1 = neurodsp.timefrequency.phase_by_time(sig1,fs,f_range=(low_f, high_f)) 
        sig2 = sig[:, args_df_row['trial'], args_df_row['contact_b']]  
        pha2 = neurodsp.timefrequency.phase_by_time(sig2,fs,f_range=(low_f, high_f)) 

        phase_diff = pha1 - pha2

        phase_diff_circ = []
        for d in phase_diff:
            if d > np.pi:
                d = -(2*np.pi - d) #negative to flip sign, and therefore keep the same position on the unit circle
            if d < -np.pi:
                d = -(-2*np.pi - d)
            phase_diff_circ.append(d)
        PPL = np.abs(np.nanmean(np.exp(1j*np.array(phase_diff_circ)))) 
        return (PPL)

    res = args_df.apply(calc_ppl, sig = signal, freq_band =freq_band, fs = fs, axis=1) ###RESULTING DF AFTER RUNNING THE FUNCTION
    res.index = pd.MultiIndex.from_frame(args_df) 
    coherence_by_trial = res.unstack('trial') ### DF WITH ALL CHANNEL PAIRS, COLUMNS ARE TRIALS (i.e., many rows, 15 columns). Will return this (maybe for decoding purposes)

    ### TURN INTO SQUARE DF, + NAMING THE CHANNELS BASED ON THE NETWORKS, for visualization purposes and for matching network names
    id2  = pd.Series(networks)
    res_square = res.unstack('contact_b')
    res_square.columns = pd.MultiIndex.from_tuples(list(zip(res_square.columns, id2))) #add network names to columns
    indexframe = res_square.index.to_frame()
    indexframe['networks'] = id2.reindex(res_square.index.get_level_values('contact_a')).values #add networks to this temp df that will be the index of res_square
    res_square.index = pd.MultiIndex.from_frame(indexframe[['contact_a','networks','trial']]) #reorder and turn into index


    return(coherence_by_trial, res_square)


def get_PPL_df_continuous(cont_signal,index,columns,freq_band,fs):
    '''####COHERENCE / PHASE-LOCKING, 
    ####SIGNAL IS CONTINUOUS, NOT EPOCHS: eg for baseline signal (not epochs)
    ####GENERATES DATAFRAME WITH PPL VALUE FOR EACH PAIR (ROW,COL) OF CHANNELS
    ####SIGNAL IS ALREADY FILTERED IF NECESSARY. INDEX AND COLUMNS ARE ALL THE CHANNELS
    SIGNAL IS CONTINUOUS, NOT EPOCHS'''


    low_f, high_f = DMN_LFP_funcs.get_freq_range(freq_band)

    #empty df, same shape/rows/cols as corr_df (ie rows and cols are all contacts)
    PPL_df = pd.DataFrame(index = index, columns = columns)
    angle_df = pd.DataFrame(index = index, columns = columns)

    #get phase difference for each pair of chans
    counter = 0
    for i,ch1 in enumerate(PPL_df):
        counter+=1
        for j,ch2 in enumerate(PPL_df):
            if j<i:
                continue
            print(counter)
            sig1 = cont_signal[i] #1 channel; current trial
            pha1 = neurodsp.timefrequency.phase_by_time(sig1,fs,f_range=(low_f, high_f)) 
            sig2 = cont_signal[j] #2 channel; current trial
            pha2 = neurodsp.timefrequency.phase_by_time(sig2,fs,f_range=(low_f, high_f)) 
            phase_diff = np.array(pha1) - np.array(pha2)
            #get circular difference 
            phase_diff_circ = []
            for d in phase_diff:
                if d > np.pi:
                    d = -(2*np.pi - d)
                if d < -np.pi:
                    d = -(-2*np.pi - d)
                phase_diff_circ.append(d)
            PPL = np.squeeze(np.abs(np.nanmean(np.squeeze(np.exp(1j*np.array(phase_diff_circ))),axis=0)))
            angle = np.angle(np.nanmean(np.exp(1j*np.array(phase_diff_circ)))) ##the angle between the phases: retains sign (ie retain info about which signal leads)
            PPL_df.iat[i,j] = PPL #fill in df with PPL value
            angle_df.iat[i,j] = angle #fill in df with PPL value

    PPL_df.fillna(PPL_df.T, inplace = True) ##since PPL_df was only the upper right triangle, use transpose to fill in the rest symmetrically
    PPL_df.drop(columns='Unknown', inplace=True, errors="ignore") #remove "unknown" network
    PPL_df.drop(index='Unknown', inplace=True, errors="ignore") #remove "unknown" network
    angle_df.fillna(angle_df.T, inplace = True) ##since PPL_df was only the upper right triangle, use transpose to fill in the rest symmetrically
    angle_df.drop(columns='Unknown', inplace=True, errors="ignore") #remove "unknown" network
    angle_df.drop(index='Unknown', inplace=True, errors="ignore") #remove "unknown" network

    return (PPL_df, angle_df)

def df_mean_by_group(df):
    '''###group by region or network and get mean (so 1 value per region or network)'''

    np.fill_diagonal(df.values, np.nan) #change diagonals from 1 to nan, so that don't get higher avg corr for regions with fewer contacts
    mean_df = df.groupby(level=0, axis=0).mean()
    df_avg = mean_df.groupby(level=0, axis=1).mean()
    # df_avg = df_avg.reindex(index=sorted(df_avg.index, key=lambda x: x.endswith('_'), reverse=True), columns=sorted(df_avg.columns, key=lambda x: x.endswith('_'), reverse=True)) #only if reorder such that DMN regions end with _
    return(df_avg)
    

def do_shuffle(corr_df):
    '''####SHUFFLE NETWORK LABELS: Shuffle labels to see if network is more correlated than expected by chance'''

    correlation_all_shuffles_all_groups = {} #eg all networks
    for label in np.unique(corr_df.columns):
        correlations_all_shuffles = []
        for _ in range(1000):
            shuffled_labels = random.sample(list(corr_df.columns),len(corr_df.columns)) #shuffle
            corr_df_shuffle = pd.DataFrame(corr_df.values, index=shuffled_labels, columns = shuffled_labels)
            corr_avg = df_mean_by_group(corr_df_shuffle) #mean corr for each network
            correlation_shuffle = corr_avg[label][label] #just the mean DMN corr (or whichever network), for the shuffled data
            correlations_all_shuffles.append(correlation_shuffle) #dmn (or other network) corr distribution for shuffles

        correlation_all_shuffles_all_groups[label] = correlations_all_shuffles
        df_correlation_all_shuffles_all_groups = pd.DataFrame.from_dict(correlation_all_shuffles_all_groups) ###turn shuffle dict into df

    return (df_correlation_all_shuffles_all_groups)
    


def plot_matshow(matrix, title = None, cmap='coolwarm', vmin=-1, vmax=1):
    fig,ax = plt.subplots()
    cax = ax.matshow(matrix,cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(cax)
    ticks = np.arange(0,len(matrix.columns),1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(matrix.columns)
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    return(fig,ax)



def connectivity_for_patient(patient, data_loc, freq_band,raw=None, trial_type = 'AUT', window = 'entire', group = 'network', drop_unknown = True, shuffle = False):
    ''' find power correlations and coherence for a given patient. Do this for frequency'''

    if raw is None:
        #get mne file if not passed to function
        raw = DMN_LFP_funcs.get_raw(patient,data_loc) #get raw mne (with already marked bad channels that were selected manually)
        raw = DMN_LFP_funcs.do_preprocessing(patient,data_loc,raw)

    # chans = raw.ch_names
    fs = raw.info['sfreq'] #sampling freq
    # bads = raw.info['bads']

    # #match electrodes to regions
    # matfile = glob.glob('/Volumes/155.100.91.44/Data/' + patient + '/Imaging/Registered/ChannelMap.mat')
    good_chans_all, good_regions, good_networks = DMN_LFP_funcs.get_label_info_all_networks(patient,data_loc,raw)

    # raw.set_eeg_reference(ref_channels = 'average') #re-reference. bads automatically excluded
    # raw.notch_filter(freqs=[60,120],picks = good_chans_all) #filter out electrical noise

    # trial_info_df = DMN_LFP_funcs.get_trial_info_df(data_loc)  # get trial info from matlab file
    trial_info_df = beh_funcs.get_trial_info_df(data_loc)  # get trial info from matlab file

    # start_times_ms, end_times_ms = DMN_LFP_funcs.get_start_end_times_ms(trial_info_df, trial_type)
    start_times_ms, end_times_ms, trial_start_times_ms = beh_funcs.get_start_end_times_ms(trial_info_df, trial_type, data_loc=data_loc)
    
    win_start, win_end = beh_funcs.get_window_times(window,start_times_ms,end_times_ms)

    ###POWER CORRELATIONS
    all_chans_power_base_corrected_over_time = get_base_corr_power(raw, win_start, win_end, trial_start_times_ms, freq_band,good_chans_all)
    power_data = all_chans_power_base_corrected_over_time
    data = pd.DataFrame(np.reshape(power_data,(power_data.shape[0]*power_data.shape[1],power_data.shape[-1])), columns = good_regions) 

    # corr_df = data.corr()  
    corr_df = data.corr(method='spearman')  

    # ####GET AVERAGE POWER CORRELATIONS FOR EACH BRAIN REGION (AVG CONTACTS WITHIN EACH REGION)
    # ####rename the DMN regions so that they dont get averaged with contacts in same region but diff network assigned
    # corr_df.index = [index + '_' if i < len(DMN_regions_reorder) else index for i, index in enumerate(corr_df.index)]
    # corr_df.columns = [column + '_' if i < len(DMN_regions_reorder) else column for i, column in enumerate(corr_df.columns)]

    corr_avg = df_mean_by_group(corr_df.copy())
    # corr_avg[np.isnan(corr_avg)] = 0 #do this because clustermap cant plot nans. ###THIS WILL GET RETURNED, if group==region

    if group == 'network':
    ###GROUP BY NETWORK
        data_networks = pd.DataFrame(np.reshape(power_data,(power_data.shape[0]*power_data.shape[1],power_data.shape[-1])), columns = good_networks)
        if drop_unknown:
            data_networks.drop(columns='Unknown', inplace=True) #remove "unknown" network
        corr_df_networks = data_networks.corr()  
        corr_avg_networks = df_mean_by_group(corr_df_networks.copy())
        # corr_avg_networks[np.isnan(corr_avg_networks)] = 0
        corr_avg = corr_avg_networks  ###THIS WILL GET RETURNED if group==network
        corr_df = corr_df_networks
    if shuffle:
        ####SHUFFLE NETWORK LABELS: Shuffle labels to see if network is more correlated than expected by chance
        print('SHUFFLING!!!')
        correlation_all_shuffles_all_networks = do_shuffle(corr_df) ###THIS WILL GET RETURNED if shuffle==True
    else:
        correlation_all_shuffles_all_networks='N/A'  

    return (raw, corr_df, corr_avg, correlation_all_shuffles_all_networks)

   



def coherence(patient,data_loc, freq_band, raw=None, trial_type = 'AUT', window = 'entire', baseline=None):

    if raw is None:
        #get mne file if not passed to function
        raw = DMN_LFP_funcs.get_raw(patient,data_loc) #get raw mne (with already marked bad channels that were selected manually)
        raw = DMN_LFP_funcs.do_preprocessing(patient,data_loc,raw)
    
    fs = raw.info['sfreq'] #sampling freq
    good_chans_all, good_regions, good_networks = DMN_LFP_funcs.get_label_info_all_networks(patient,data_loc,raw)

    low_f, high_f = DMN_LFP_funcs.get_freq_range('all')
    sig_filtered = raw.copy().filter(l_freq = low_f, h_freq = high_f, picks=good_chans_all)  
    
    trial_info_df = beh_funcs.get_trial_info_df(data_loc)  # get trial info from matlab file
    start_times_ms, end_times_ms, trial_start_times_ms = beh_funcs.get_start_end_times_ms(trial_info_df, trial_type)
    win_start, win_end = beh_funcs.get_window_times(window,start_times_ms,end_times_ms)
    signal_epochs = DMN_LFP_funcs.split_into_epochs(sig_filtered, win_start, win_end)
    signal = signal_epochs.get_data(picks=good_chans_all)
    signal = np.transpose(signal, (2, 0, 1))

    PPL_df, angle_df = get_PPL_df_from_epochs(signal, good_networks, good_networks, freq_band,fs)

    ###GET AVERAGE PPL FOR EACH REGION NETWORK
    PPL_avg = df_mean_by_group(PPL_df.copy())
    # PPL_avg[np.isnan(PPL_avg)] = 0 #do this because clustermap cant plot nans

    if baseline is None:
        baseline_phase_avg = None
        baseline_angle = None
        baseline_epochs_PPL = None
    
    elif baseline == 'all': #baseline is before each trial, hence epochs
        baseline_epochs = DMN_LFP_funcs.split_into_epochs(sig_filtered, trial_start_times_ms - (2 * 1000), trial_start_times_ms - (1*1000))
        baseline_epochs = baseline_epochs.get_data(picks=good_chans_all)
        baseline_epochs = np.transpose(baseline_epochs, (2, 0, 1))
        baseline_PPL_df, baseline_angle = get_PPL_df_from_epochs(baseline_epochs, good_networks, good_networks, freq_band,sfreq)
        baseline_PPL_avg = df_mean_by_group(baseline_PPL_df.copy())
        # baseline_PPL_avg[np.isnan(baseline_PPL_avg)] = 0 #do this because clustermap cant plot nans

    else: #baseline is the very beginning of the session
        #baseline PPL. passed argument "baseline" is the time of start and stop of baseline from which to calc ppl
        baseline_data = sig_filtered.get_data(picks=good_chans_all)[:,baseline[0]:baseline[1]]
        baseline_PPL_df, baseline_angle = get_PPL_df_continuous(baseline_data,good_networks, good_networks, freq_band, fs)
        baseline_PPL_avg = df_mean_by_group(baseline_PPL_df.copy())
        # baseline_PPL_avg[np.isnan(baseline_PPL_avg)] = 0 #do this because clustermap cant plot nans

    return(PPL_df, PPL_avg, angle_df, baseline_PPL_df, baseline_PPL_avg, baseline_angle)




def connectivity_for_patient_trial_avg(patient, data_loc, freq_band,raw=None, trial_type = 'AUT', window = 'entire', group = 'network', drop_unknown = True, shuffle = False):
    ''' find power correlations and coherence for a given patient. Do this for frequency
        avg corr per trial (i.e. 15 values per electrode to correlate, not 150)'''

    if raw is None:
        #get mne file if not passed to function
        raw = DMN_LFP_funcs.get_raw(patient,data_loc) #get raw mne (with already marked bad channels that were selected manually)
        raw = DMN_LFP_funcs.do_preprocessing(patient,data_loc,raw)

    fs = raw.info['sfreq'] #sampling freq

    # #match electrodes to regions
    good_chans_all, good_regions, good_networks = DMN_LFP_funcs.get_label_info_all_networks(patient,data_loc,raw)

    trial_info_df = beh_funcs.get_trial_info_df(data_loc)  # get trial info from matlab file

    start_times_ms, end_times_ms, trial_start_times_ms = beh_funcs.get_start_end_times_ms(trial_info_df, trial_type)
    
    win_start, win_end = beh_funcs.get_window_times(window,start_times_ms,end_times_ms)

    ###POWER CORRELATIONS
    all_chans_power_base_corrected_over_time = get_base_corr_power(raw, win_start, win_end, trial_start_times_ms, freq_band,good_chans_all)
    power_data = all_chans_power_base_corrected_over_time

    trial_avg_power_data = np.mean(power_data,axis=0) #avg power for each trial (across time window provided)
    trial_avg_df = pd.DataFrame(trial_avg_power_data,columns = good_regions)
    trial_avg_corr_df = trial_avg_df.corr(method='spearman')

    # ####GET AVERAGE POWER CORRELATIONS FOR EACH BRAIN REGION (AVG CONTACTS WITHIN EACH REGION)
    corr_avg = df_mean_by_group(trial_avg_corr_df.copy())

    if group == 'network':
    ###GROUP BY NETWORK
        trial_avg_data_networks = pd.DataFrame(trial_avg_power_data, columns = good_networks)
        if drop_unknown:
            trial_avg_data_networks.drop(columns='Unknown', inplace=True) #remove "unknown" network
        trial_avg_corr_df_networks = trial_avg_data_networks.corr()  
        corr_avg_networks = df_mean_by_group(trial_avg_corr_df_networks.copy())
        corr_avg = corr_avg_networks  ###THIS WILL GET RETURNED if group==network
        corr_df = trial_avg_corr_df_networks
    if shuffle:
        ####SHUFFLE NETWORK LABELS: Shuffle labels to see if network is more correlated than expected by chance
        print('SHUFFLING!!!')
        correlation_all_shuffles_all_networks = do_shuffle(corr_df) ###THIS WILL GET RETURNED if shuffle==True
    else:
        correlation_all_shuffles_all_networks='N/A'  

    return (raw, corr_df, corr_avg, correlation_all_shuffles_all_networks)






def plot_connectivity(signal,good_chans_all,good_regions):

    from mne_connectivity import spectral_connectivity_epochs
    from mne_connectivity.viz import plot_sensors_connectivity
    from mne_connectivity.viz import plot_connectivity_circle

    con = spectral_connectivity_epochs(
        signal,
        method="pli",
        mode="multitaper",
        sfreq=fs,
        fmin=4,
        fmax=8,
        faverage=True,
        tmin=0,
        mt_adaptive=False,
        n_jobs=1,
    )

    label_names = [name + ' ' + str(i) for i, name in enumerate(good_regions)]
    order = list(np.sort(label_names))
    from mne.viz import circular_layout
    node_angles = circular_layout(label_names, order, start_pos=90)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black", subplot_kw=dict(polar=True))

    plot_connectivity_circle(con.get_data(output="dense")[:, :, 0],good_regions,n_lines=300,node_angles=node_angles,ax=ax)


    fname_fig = 'Figures/plot_mixed_connect.pdf'
    fig.savefig(fname_fig, facecolor=fig.get_facecolor())


def calc_sliding_ppl_each_row(args_df_row, sig, freq_band, fs, sliding_win_width = 2000, step = 200):
    ''' ### MAKE THE FUNCTION THAT CALCULATES PHASE LOCKING; WILL BE APPLIED TO ARGS_DF RATHER THAN LOOPING THROUGH EACH CHAN PAIR AND TRIAL
        calculate the phase diff between the signals from 2 channels 
        args_df_row is a single row of args_df (because this func gets applied to args_df)
        thus it is 1 pair of channels and 1 trial'''
    # print(args_df_row.name)
    low_f, high_f = DMN_LFP_funcs.get_freq_range(freq_band)
    sig1 = sig[:, args_df_row['trial'], args_df_row['contact_a']]  
    pha1 = neurodsp.timefrequency.phase_by_time(sig1,fs,f_range=(low_f, high_f)) 
    sig2 = sig[:, args_df_row['trial'], args_df_row['contact_b']]  
    pha2 = neurodsp.timefrequency.phase_by_time(sig2,fs,f_range=(low_f, high_f)) 

    phase_diff = pha1 - pha2

    phase_diff_circ = []
    for d in phase_diff:
        if d > np.pi:
            d = -(2*np.pi - d) #negative to flip sign, and therefore keep the same position on the unit circle
        if d < -np.pi:
            d = -(-2*np.pi - d)
        phase_diff_circ.append(d)

    valid_phase_diff_circ = np.array(phase_diff_circ)[np.isfinite(phase_diff_circ)] #remove nans

    sliding_PPL = []
    i=0
    while(i < len(valid_phase_diff_circ)-sliding_win_width):
        phase_diff_in_win = valid_phase_diff_circ[i:i+sliding_win_width]
        PPL_in_win = np.abs(np.nanmean(np.exp(1j*np.array(phase_diff_in_win)))) 
        sliding_PPL.append(PPL_in_win)
        i+=step
    return (sliding_PPL)

def get_PPL_sliding_window(signal,networks,freq_band,fs, sliding_win_width = 2000, step = 200):
    '''####COHERENCE / PHASE-LOCKING
    ####do this over a sliding window for each trial (get PPL value for each pair of chans over a sliding window)

    like get_PPL_by_trial   but with sliding window '''
    contacts = np.arange(signal.shape[-1])
    trials = np.arange(signal.shape[1])

    df = (pd.DataFrame(1, index=contacts, columns=trials).cumsum(1)-1)
    # df = (pd.DataFrame(1, index=np.arange(48), columns=np.arange(15)).cumsum(1)-1)
    df.index.name = 'contact'
    df.columns.name = 'trial'

    # df=df.stack()

    ###GET ALL COMBINATIONS OF CHANNELS (EACH CHANNEL PAIR)
    contact_comb = list(it.combinations(df.index.unique('contact'), 2))
    # df = df.unstack('trial')
    df = pd.DataFrame(contact_comb) #makes df where channel pairs are the columns (column 0 = contact a; col 1 = contact b)

    df['val']  = 0 #adds column of zeros
    df = df.set_index([0, 1]) #MAKES THE 2 CHANS IN EACH PAIR THE INDICES
    df.index.names = ['contact_a', 'contact_b']
    # df = df.apply(lambda x: pd.Series(np.arange(15)), axis=1)
    df = df.apply(lambda x: pd.Series(trials), axis=1) ###PUTS TRIALS IN THE DF, EACH COLUMN IS A TRIAL
    df.columns.name = 'trial'

    # args_df = df.stack().index.to_frame()

    args_df = df.stack().index.to_frame().reset_index(drop=True) #now df is just 3 columnns: contact a, contact b, and trial. So every combination is included

    res = args_df.apply(calc_sliding_ppl_each_row, sig = signal, freq_band =freq_band, fs = fs, sliding_win_width = sliding_win_width, step = step, axis=1) ###RESULTING DF AFTER RUNNING THE FUNCTION
    res.index = pd.MultiIndex.from_frame(args_df) 


    return(res)




######## alt version of phase code, from chatgpt:

import numpy as np
import pandas as pd
import itertools as it

def get_low_high(freq_band):
    if isinstance(freq_band, tuple):
        return freq_band
    else:
        return DMN_LFP_funcs.get_freq_range(freq_band)


def compute_phase_tensor(signal, freq_band, fs):
    """
    signal shape: time x trial x contact
    returns phase shape: time x trial x contact
    """
    
    low_f, high_f = get_low_high(freq_band)

    n_time, n_trials, n_contacts = signal.shape
    phase = np.empty((n_time, n_trials, n_contacts), dtype=np.float32)

    for tr in range(n_trials):
        for ch in range(n_contacts):
            phase[:, tr, ch] = neurodsp.timefrequency.phase_by_time(
                signal[:, tr, ch],
                fs,
                f_range=(low_f, high_f)
            )

    return phase


def calc_sliding_ppl_from_phase(args_df_row, phase, sliding_win_width=2000, step=200):
    """
    phase shape: time x trial x contact
    args_df_row contains: contact_a, contact_b, trial
    """
    tr = int(args_df_row['trial'])
    ch1 = int(args_df_row['contact_a'])
    ch2 = int(args_df_row['contact_b'])

    pha1 = phase[:, tr, ch1]
    pha2 = phase[:, tr, ch2]

    phase_diff = pha1 - pha2

    valid = np.isfinite(phase_diff)

    # Easiest conservative option:
    # keep original time indexing, but ignore NaNs inside each window
    starts = np.arange(0, len(phase_diff) - sliding_win_width, step)

    sliding_PPL = []
    for s in starts:
        win = phase_diff[s:s + sliding_win_width]
        win = win[np.isfinite(win)]

        if len(win) == 0:
            sliding_PPL.append(np.nan)
        else:
            sliding_PPL.append(np.abs(np.nanmean(np.exp(1j * win))))

    return sliding_PPL


def get_PPL_sliding_window_fast(signal, networks, freq_band, fs,
                                sliding_win_width=2000, step=200):
    """
    Faster version of get_PPL_sliding_window.
    Computes phase once per trial/contact, then reuses it.
    """

    n_time, n_trials, n_contacts = signal.shape

    phase = compute_phase_tensor(signal, freq_band, fs)

    contacts = np.arange(n_contacts)
    trials = np.arange(n_trials)

    contact_comb = list(it.combinations(contacts, 2))

    args = []
    for contact_a, contact_b in contact_comb:
        for tr in trials:
            args.append((contact_a, contact_b, tr))

    args_df = pd.DataFrame(args, columns=['contact_a', 'contact_b', 'trial'])

    res = args_df.apply(
        calc_sliding_ppl_from_phase,
        phase=phase,
        sliding_win_width=sliding_win_width,
        step=step,
        axis=1
    )

    res.index = pd.MultiIndex.from_frame(args_df)

    return res


def get_phase_diff_across_time(signal, networks, freq_band, fs,
                                sliding_win_width=2000, step=200):


    n_time, n_trials, n_contacts = signal.shape

    phase = compute_phase_tensor(signal, freq_band, fs)

    contacts = np.arange(n_contacts)
    trials = np.arange(n_trials)

    contact_comb = list(it.combinations(contacts, 2))

    args = []
    for contact_a, contact_b in contact_comb:
        for tr in trials:
            args.append((contact_a, contact_b, tr))

    args_df = pd.DataFrame(args, columns=['contact_a', 'contact_b', 'trial'])

    def get_phase_diff(args_df_row, phase):
        tr = int(args_df_row['trial'])
        ch1 = int(args_df_row['contact_a'])
        ch2 = int(args_df_row['contact_b'])

        pha1 = phase[:, tr, ch1]
        pha2 = phase[:, tr, ch2]

        phase_diff = pha1 - pha2

        valid = np.isfinite(phase_diff)
        return(phase_diff[valid])


    res = args_df.apply(
        get_phase_diff,
        phase=phase,
        axis=1
    )

    res.index = pd.MultiIndex.from_frame(args_df)

    return res




# def get_PPL_by_trial_old(signal,index,columns,freq_band,fs):
#     '''####COHERENCE / PHASE-LOCKING
#     ####do this by trials (get PPL value for each trial for each pair of chans)
#     '''
        

#     low_f, high_f = DMN_LFP_funcs.get_freq_range(freq_band)

#     #empty df, same shape/rows/cols as corr_df (ie rows and cols are all contacts)
#     phase_df = pd.DataFrame(index = index, columns = columns)

#     num_trials = signal.shape[1]
#     chan_locs_array = np.concatenate([[row] * num_trials for row in index])
#     trial_idx_array = np.concatenate([np.arange(num_trials)]*len(index))
#     chan_idx_array = np.repeat(np.arange(len(columns)),num_trials)
#     new_rows = pd.MultiIndex.from_tuples(list(zip(chan_idx_array, chan_locs_array, trial_idx_array)), names = ['chan_idx', 'network', 'trial_idx'])
    
#     phase_df_trials = pd.DataFrame(index = new_rows, columns = columns)



#     #get phase difference for each pair of chans
#     counter = 0
#     for i in np.unique(chan_idx_array):
#         counter+=1
#         for j in np.unique(chan_idx_array):
#             if j<i:
#                 continue
#             # print(counter)
#             phase1_all_trials = []
#             phase2_all_trials = []
#             PPL_all_trials = []
#             phase_diff_circ_all_trials = []
#             # fig1, ax1 = plt.subplots()
#             # fig2, ax2 = plt.subplots(subplot_kw=dict(projection='polar'))
#             for t in np.arange(num_trials):
#                 sig1 = signal[:,t,i] #1 channel; current trial
#                 pha1 = neurodsp.timefrequency.phase_by_time(sig1,fs,f_range=(low_f, high_f)) 
#                 phase1_all_trials.append(pha1)
#                 sig2 = signal[:,t,j] #2 channel; current trial
#                 pha2 = neurodsp.timefrequency.phase_by_time(sig2,fs,f_range=(low_f, high_f)) 
#                 phase2_all_trials.append(pha2)
            
#                 phase_diff = pha1 - pha2

#                 # fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
#                 # ax.hist(phase_diff, bins = 1000, density=True)

#                 #get circular difference 
#                 phase_diff_circ = []
#                 for d in phase_diff:
#                     if d > np.pi:
#                         d = -(2*np.pi - d) #negative to flip sign, and therefore keep the same position on the unit circle
#                     if d < -np.pi:
#                         d = -(-2*np.pi - d)
#                     phase_diff_circ.append(d)
#                 # ax1.plot(phase_diff_circ)
#                 phase_diff_circ_all_trials.append(phase_diff_circ)

#                 PPL = np.abs(np.nanmean(np.exp(1j*np.array(phase_diff_circ)))) 
#                 # PPL = np.abs(np.nanmean(np.exp(1j*np.array(phase_diff)))) 
#                 # angle = np.angle(np.nanmean(np.exp(1j*np.array(phase_diff_circ)))) ##the angle between the phases: retains sign (ie retain info about which signal leads)

#                 PPL_all_trials.append(PPL)
#             print(PPL_all_trials)
#             PPL_timecourse = np.abs(np.nanmean(np.exp(1j*np.array(phase_diff_circ_all_trials)),axis=0)) 
#             PPL_by_trials = np.abs(np.nanmean(np.exp(1j*np.array(phase_diff_circ_all_trials)),axis=1)) 
#             phase_df_trials.loc[i].iloc[:,j] = PPL_by_trials

#     return (phase_df_trials)
