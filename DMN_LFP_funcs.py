#Functions for DMN LFP analysis
#see connectivity_funcs.py for connectivity-specific functions 


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

import beh_funcs 

import re

import mne

import scipy
import patient_params


def choose_patient(patients_info):
    choose_patient = input("Enter patient ID or type 'all': ")
    check=0
    while check==0:
        if choose_patient!='all': ###if want to pick 1 patient. Otherwise, do analysis on all patients in patients_info
            single_patient_info = list(filter(lambda p: p[0]==choose_patient, patients_info))
            check=1
            if single_patient_info:
                patients_info = single_patient_info
            else:
                choose_patient = input("Invalid patient ID. Enter patient ID or type 'all': ")
                check=0
        else:
            break
    return(patients_info)

def choose_freq():
    choose_freq = input("Choose frequency band (spacebar for both gamma and theta): ")
    check=0
    while check==0:
        if choose_freq==' ': 
            freqs_list=['gamma', 'theta']
            check=1
        else: 
            freq_options = ['theta', 't', 'alpha', 'a', 'beta', 'b','gamma', 'g', 'high_gamma', 'h', 'high gamma', 'high_and_gamma', 'high and gamma']
            if choose_freq in freq_options:
                freqs_list = [choose_freq]
                check=1 
            else:
                input("Invalid frequency name. Choose frequency band or first letter ('theta' (t), 'alpha' (a), 'beta' (b),'gamma' (g), 'high gamma' (h), 'high and gamma')")
                check=0
    return(freqs_list)

def get_freq_range(freq_band):
    """get high and low freq bounds for the chosen freq band"""

    if freq_band == 'theta' or freq_band == 't':
        low_f = 4
        high_f = 8    
    if freq_band == 'alpha' or freq_band == 'a':
        low_f = 8
        high_f = 13
    if freq_band == 'beta' or freq_band == 'b':
        low_f = 13
        high_f = 30
    if freq_band == 'gamma' or freq_band == 'g':
        low_f = 30
        high_f = 70    
    if freq_band == 'high_gamma' or freq_band == 'high gamma' or freq_band == 'h':
        low_f = 70
        high_f = 140
    if freq_band == 'high_and_gamma' or freq_band == 'high and gamma':
        low_f = 30
        high_f = 140
    if freq_band == 'all':
        low_f = 1
        high_f = 150

    return(low_f, high_f)


   
def split_into_epochs(signal, start_times, end_times):
    """ Split signal into epochs given the start and end times of each trial (or epoch)
        If the trials are not all the same length, truncate so that all epochs are the same length"""

    min_trial_len = int(min(end_times - start_times))
    ## Get list of signals for subtrials of interest
    try:
        signal_subtrial_list = [signal[:,int(start_times[i]):int(start_times[i]+min_trial_len)][0] for i in np.arange(len(start_times))]
    except: #start times is series
        start_times = start_times.values
        signal_subtrial_list = [signal[:,int(start_times[i]):int(start_times[i]+min_trial_len)][0] for i in np.arange(len(start_times))]

    signal_epochs = mne.EpochsArray(signal_subtrial_list, signal.info) #turn back into mne "raw"
    return (signal_epochs)

    
def get_power_for_trials_subset(freq_band_signal, start_times, end_times, picked_chans, freq_band):
    """ Get average power for subset of trials
        start with FILTERED signal (i.e. already in chosen freq band)
        This is power averaged across time, so it does NOT give a timecourse"""

    low_f, high_f = get_freq_range(freq_band)
    signal_epochs = split_into_epochs(freq_band_signal, start_times, end_times)

    signal_spec = signal_epochs.compute_psd(fmin = 0.1, fmax = 150, picks=picked_chans).get_data(fmin = low_f, fmax = high_f, picks=picked_chans)
    signal_power = np.mean(signal_spec,axis=-1) #avg spectrum over frequencies 
    #convert to dB
    signal_power_db = 10 * np.log10(signal_power)
 
    return (signal_power, signal_power_db)

def get_power_over_time_from_unfiltered(signal, start_times, end_times, picked_chans, freq_band, window_size=1):
    """ get power over time, from the orignal signal before it was split into freq bands. 
        i.e., this function first filters out the relevant band before getting power in that band"""


    low_f, high_f = get_freq_range(freq_band)

    signal_filtered = signal.copy().filter(l_freq = low_f, h_freq = high_f, picks=picked_chans) # define low- and high-frequencies   
    signal_epochs = split_into_epochs(signal_filtered, start_times, end_times)

    signal_spec_AUT_over_time = [signal_epochs.compute_psd(fmin = 0.1, fmax = 150, tmin = 0+i, tmax = window_size+i, picks=picked_chans).get_data(fmin = low_f, fmax = high_f, picks=picked_chans) for i in np.arange(0, np.array(signal_epochs).shape[-1]//1000, window_size)] #pick time window

    #temp fix if extra freq bin. Was never an issue until try plotting entire block rather than trial  
    if signal_spec_AUT_over_time[-1].shape[-1] > signal_spec_AUT_over_time[0].shape[-1]:
        signal_spec_AUT_over_time[-1] = signal_spec_AUT_over_time[-1][..., :-1]


    signal_power_over_time = np.mean(signal_spec_AUT_over_time,axis=-1) #avg spectrum over frequencies 
    #convert to dB
    signal_power_over_time_db = 10 * np.log10(signal_power_over_time)
    return (signal_filtered, signal_power_over_time, signal_power_over_time_db)



def get_power_over_time_from_unfiltered_old(signal, start_times, end_times, picked_chans, freq_band):
    """ get power over time, from the orignal signal before it was split into freq bands. 
        i.e., this function first filters out the relevant band before getting power in that band
        1 SECOND TIME BINS"""


    low_f, high_f = get_freq_range(freq_band)

    signal_filtered = signal.copy().filter(l_freq = low_f, h_freq = high_f, picks=picked_chans) # define low- and high-frequencies   
    signal_epochs = split_into_epochs(signal_filtered, start_times, end_times)

    signal_spec_AUT_over_time = [signal_epochs.compute_psd(fmin = 0.1, fmax = 150, tmin = 0+i, tmax = 1+i, picks=picked_chans).get_data(fmin = low_f, fmax = high_f, picks=picked_chans) for i in np.arange(np.array(signal_epochs).shape[-1]//1000)] #pick time window

    signal_power_over_time = np.mean(signal_spec_AUT_over_time,axis=-1) #avg spectrum over frequencies 
    #convert to dB
    signal_power_over_time_db = 10 * np.log10(signal_power_over_time)
    return (signal_filtered, signal_power_over_time, signal_power_over_time_db)



def get_power_over_time_from_prefiltered(freq_band_signal, start_times, end_times, picked_chans, freq_band):
    """get power over time, from the filtered signal (already just the chosen freq band)"""

    low_f, high_f = get_freq_range(freq_band)
    signal_epochs = split_into_epochs(freq_band_signal, start_times, end_times)

    signal_spec_AUT_over_time = [signal_epochs.compute_psd(fmin = 0.1, fmax = 150, tmin = 0+i, tmax = 1+i, picks=picked_chans).get_data(fmin = low_f, fmax = high_f, picks=picked_chans) for i in np.arange(np.array(signal_epochs).shape[-1]//1000)] #pick time window

    signal_power_over_time = np.mean(signal_spec_AUT_over_time,axis=-1) #avg spectrum over frequencies 
    #convert to dB
    signal_power_over_time_db = 10 * np.log10(signal_power_over_time)
 
    return (signal_power_over_time, signal_power_over_time_db)



def baseline_correct_percent_method_1(signals_all_subtrails, pre_trial_baselines, trial_start_times=[], subtrial_start_times=[], pre_trial_buffer=0):
    """BASELINE CORRECTION METHOD 1:
    average all pre-trial baselines and use same average for baseline correction across all trials
    divide sigs by baseline to get percent of baseline
    don't actually need (trial_start_times, subtrial_start_times) params, but include them so the call matches method 2"""

    baseline_power_avg = np.mean(pre_trial_baselines,axis=0) #avg across trials so use same baseline for all trials
    # signal_power_baseline_corr_percent = (signals_all_subtrails / baseline_power_avg)*100  ##percent of baseline 
    signal_power_baseline_corr_percent = ((signals_all_subtrails-baseline_power_avg) / baseline_power_avg)*100 ##percent change from baseline
    return (signal_power_baseline_corr_percent)



def baseline_correct_percent_method_2(signals_all_subtrails, pre_trial_baselines, trial_start_times, subtrial_start_times, pre_trial_buffer=0):
    """ BASELINE CORRECTION METHOD 2:
        divide each trial's signal by it's pre-trial baseline   """

    subtrial_start_times = subtrial_start_times + pre_trial_buffer  #because these times have the buffer while trial_start_times does not, so ends up with first subtrial appearing to precede the whole trial

    #find trial inds (each consisting of 3 items) in which each high_orig_response_start_times exists
    inds_temp = [np.where((subtrial_start_times < trial_start_times[i+1]) & (subtrial_start_times > trial_start_times[i]))[0] for i in range(len(trial_start_times)-1)]
    # #the above step didnt include the subtrial start times AFTER the last trial start
    inds_temp.append(np.where(subtrial_start_times > trial_start_times[-1])[0])

    #reorder baseline signals, so that each corresponds to signal for each sub-trial
    inds_temp_df = pd.DataFrame(inds_temp)
    inds_for_reordered_baselines = [int(np.where(inds_temp_df==ind)[0]) for ind in range(len(subtrial_start_times))]
    reordered_baselines = np.array(pre_trial_baselines)[inds_for_reordered_baselines]

    #baseline correct each channel
    # power_baseline_corr = (signals_all_subtrails / reordered_baselines)*100 ##percent of baseline 
    power_baseline_corr = (signals_all_subtrails / reordered_baselines)*100 - 100 ##percent change from baseline
    return(power_baseline_corr)



def get_raw_from_server(patient, data_loc):
    ''' Get raw mne object, and bad channels, manually picked if haven't already been chosen
        GET FILE FROM SERVER... slow!. Use get_raw instead, which gets the local file if it exists'''

    save_file = '/Volumes/155.100.91.44/Data/Rikki/' + patient + '/raw.fif'
    try:
        raw_standard = mne.io.read_raw(save_file, preload=True)
    except: 
        print ("need to get data")
        import save_mne_file
        raw_standard = save_mne_file.save_mne_raw(data_loc,patient)

    #make another mne raw obj, turning channels into "seeg" type
    info = mne.create_info(ch_names=raw_standard.info['ch_names'], sfreq=raw_standard.info['sfreq'], ch_types='seeg')
    raw = mne.io.RawArray(raw_standard.get_data(), info)

    chans = raw.ch_names
    fs = raw.info['sfreq'] #sampling freq

    # #designate "bad" channels to exclude from analysis. Determine bad channels from looking at plot: raw.plot()
    if len(raw_standard.info['bads'])==0:#means we probably haven't picked bad channels, so plot and find bads now
        print('plotting')
        raw_standard.plot(block=True) #manually select bad channels in gui
    bads = raw_standard.info['bads']
    raw.info['bads'] = bads
    return (raw)


def get_raw(patient, data_loc):
    ''' Get raw mne object, and bad channels, manually picked if haven't already been chosen
        Get the file from the LOCAL dir'''

    # save_file = '/Volumes/155.100.91.44/Data/Rikki/' + patient + '/raw.fif'
    save_file_local = 'mne_files/'+patient + '/raw.fif'
    if data_loc=='/Volumes/155.100.91.44/Data/UIC202422/DMN/20241231-144952':
        save_file_local = 'mne_files/'+patient + '/day1_raw.fif'

    try:
        raw_standard = mne.io.read_raw(save_file_local, preload=True)
    except: 
        print ("need to get data")
        import save_mne_file

        if ((patient=='UIC202521') | (patient=='UIC202522')):
            raw_standard = save_mne_file.save_mne_raw_gemini(data_loc,patient)  #saves both locally and on server
        else:
            raw_standard = save_mne_file.save_mne_raw(data_loc,patient)  #saves both locally and on server


    #make another mne raw obj, turning channels into "seeg" type
    info = mne.create_info(ch_names=raw_standard.info['ch_names'], sfreq=raw_standard.info['sfreq'], ch_types='seeg')
    raw = mne.io.RawArray(raw_standard.get_data(), info)

    chans = raw.ch_names
    fs = raw.info['sfreq'] #sampling freq

    # #designate "bad" channels to exclude from analysis. Determine bad channels from looking at plot: raw.plot()
    if len(raw_standard.info['bads'])==0:#means we probably haven't picked bad channels, so plot and find bads now
        print('plotting')
        raw_standard.plot(block=True) #manually select bad channels in gui
    bads = raw_standard.info['bads']
    raw.info['bads'] = bads
    return (raw)

# def get_raw_preprocessed(patient, data_loc):
#     ''' Get raw mne object, and bad channels, manually picked if haven't already been chosen
#         Get the file from the LOCAL dir'''

#     # save_file = '/Volumes/155.100.91.44/Data/Rikki/' + patient + '/raw.fif'
#     save_file_local = 'mne_files/'+patient + '/raw.fif'
#     try:
#         raw_standard = mne.io.read_raw(save_file_local, preload=True)
#     except: 
#         print ("need to get data")
#         import save_mne_file
#         raw_standard = save_mne_file.save_mne_raw(data_loc,patient)


#     #make another mne raw obj, turning channels into "seeg" type
#     info = mne.create_info(ch_names=raw_standard.info['ch_names'], sfreq=raw_standard.info['sfreq'], ch_types='seeg')
#     raw = mne.io.RawArray(raw_standard.get_data(), info)

#     chans = raw.ch_names
#     fs = raw.info['sfreq'] #sampling freq

#     # #designate "bad" channels to exclude from analysis. Determine bad channels from looking at plot: raw.plot()
#     if len(raw_standard.info['bads'])==0:#means we probably haven't picked bad channels, so plot and find bads now
#         print('plotting')
#         raw_standard.plot(block=True) #manually select bad channels in gui
#     bads = raw_standard.info['bads']
#     raw.info['bads'] = bads

#     raw.set_eeg_reference(ref_channels = 'average') #re-reference. bads automatically excluded
#     raw.notch_filter(freqs=[60,120]) #filter out electrical noise
#     return(raw)


def remove_micros_bool(data_loc):
    remove_micros = True
    if '20240115-101116' in data_loc: #didn't record micros; use ChannelMap2 instead
        matfile = glob.glob('/Volumes/155.100.91.44/Data/UIC202401/Imaging/Registered/ChannelMap2.mat')
        remove_micros = False
    if 'UIC202408' in data_loc: #this patient has grid, and no micros (still use ChannelMap1 though)
        remove_micros = False
    if 'UIC202414' in data_loc: #no micros (still use ChannelMap1 though)
        remove_micros = False
    if 'UIC202418' in data_loc: #used ripple. Use channelmap2
        matfile = glob.glob('/Volumes/155.100.91.44/Data/UIC202418/Imaging/Registered/ChannelMap2.mat')
        remove_micros = False
    if 'UIC202418' in data_loc: 
        remove_micros = False
    if 'UIC202420' in data_loc: 
        remove_micros = False
    if 'UIC202421' in data_loc: 
        remove_micros = False
    if '20241231-144952' in data_loc: 
        remove_micros = False
    if 'UIC202506' in data_loc:
        remove_micros = False
    return(remove_micros)

def get_matfile(patient, data_loc):
    matfile = glob.glob('/Volumes/155.100.91.44/Data/' + patient + '/Imaging/Registered/ChannelMap.mat')
    if '20240115-101116' in data_loc: #didn't record micros; use ChannelMap2 instead
        matfile = glob.glob('/Volumes/155.100.91.44/Data/UIC202401/Imaging/Registered/ChannelMap2.mat')
    if 'UIC202418' in data_loc: #used ripple. Use channelmap2
        matfile = glob.glob('/Volumes/155.100.91.44/Data/UIC202418/SEEG/Imaging/Registered/ChannelMap2.mat')
    if 'UIC202420' in data_loc: #used ripple. Use channelmap2
        matfile = glob.glob('/Volumes/155.100.91.44/Data/UIC202420/Imaging/Registered/ChannelMap2.mat')
    if 'UIC202421' in data_loc: #used ripple. Use channelmap2
        matfile = glob.glob('/Volumes/155.100.91.44/Data/UIC202421/Imaging/Registered/ChannelMap2.mat')
    if '20241231-144952' in data_loc: #used ripple. Use channelmap2
        matfile = glob.glob('/Volumes/155.100.91.44/Data/UIC202422/Imaging/Registered/ChannelMap2.mat')
    if 'UIC202504' in data_loc: #used blackrock. Use channelmap2 for macros; channelmap1 for micros
        matfile = glob.glob('/Volumes/155.100.91.44/Data/UIC202504/Imaging/Registered/ChannelMap2.mat')
    return(matfile)

def get_label_info_all_networks(patient,data_loc ,raw, which_chans_str = 'DMN'):
    '''
    Returns all good channel names, as well as the corresponding regions and networks
    '''

    # matfile = glob.glob('/Volumes/155.100.91.44/Data/' + patient + '/Imaging/Registered/ChannelMap.mat')
    matfile = get_matfile(patient, data_loc)
    chans = raw.ch_names
    chan_info = mat73.loadmat(matfile[0])
    regions = pd.DataFrame(chan_info['ElecAtlasProj'])[0]
    regions_list = [reg[0] if reg is not None else 'None' for reg in regions]
    networks = pd.DataFrame(chan_info['ElecAtlasProj'])[pd.DataFrame(chan_info['ElecAtlasProj']).columns.size-1]
    networks_list = [net[0] if net is not None else 'None' for net in networks]
    
    good_chans_all, good_chans_subset = get_chan_info(raw, chan_info, which_chans_str, remove_micros=remove_micros_bool(data_loc), remove_white=True, patient=patient)

    good_chans_inds = [chans.index(g) for g in good_chans_all]
    good_regions = [' '.join(label.split(' ')[:2]) for label in np.array(regions_list)[good_chans_inds]]   
    good_networks = np.array(networks_list)[good_chans_inds]
    
    return(good_chans_all, good_regions, good_networks)

def get_label_info_network(patient,data_loc ,raw, which_chans_str = 'DMN'):
    '''
    Returns good channel names for the desired channel subset, as well as the corresponding regions and networks
    '''
    # matfile = glob.glob('/Volumes/155.100.91.44/Data/' + patient + '/Imaging/Registered/ChannelMap.mat')
    matfile = get_matfile(patient, data_loc)
    chans = raw.ch_names
    chan_info = mat73.loadmat(matfile[0])
    regions = pd.DataFrame(chan_info['ElecAtlasProj'])[0]
    regions_list = [reg[0] if reg is not None else 'None' for reg in regions]
    networks = pd.DataFrame(chan_info['ElecAtlasProj'])[pd.DataFrame(chan_info['ElecAtlasProj']).columns.size-1]
    networks_list = [net[0] if net is not None else 'None' for net in networks]
    
    good_chans_all, good_chans_subset = get_chan_info(raw, chan_info, which_chans_str, remove_micros=remove_micros_bool(data_loc), remove_white=True, patient=patient)

    good_chans_inds = [chans.index(g) for g in good_chans_subset]
    good_regions_subset = [' '.join(label.split(' ')[:2]) for label in np.array(regions_list)[good_chans_inds]]   
    good_networks_subset = np.array(networks_list)[good_chans_inds]
    
    return(good_chans_all, good_chans_subset, good_regions_subset, good_networks_subset)
    
def get_chan_info(raw, chan_map_info, which_chans_str = 'DMN', remove_micros=True, remove_white=True, patient=None):
    '''
    Get the all good channels and also a desired subset. 
    Find and remove the rest of bad channels (remove things that are 'None').
    
    raw: the raw mne object, with bads and chans already defined in info
    chan_map_info: ChannelMap matlab file
    which_chans_str: Desired subset of electrodes. Can be "DMN" or "ROI". Otherwise, subset will be empty
        for example:
            good_chans_all, good_chans_DMN = get_chan_info(raw, chan_map_info, "DMN", remove_micros=True, remove_white=True)
            good_chans_all, good_chans_ROIs = get_chan_info(raw, chan_map_info, "ROI", remove_micros=True, remove_white=True)
    remove_micros: Do we remove microwires channels. Default is True
    remove_white: Do we remove white matter channels. Default is True

    '''
    chans = raw.ch_names
    bads = raw.info['bads']

    ###get regions and networks from ChannelMap mat file
    regions = pd.DataFrame(chan_map_info['ElecAtlasProj'])[0]
    networks = pd.DataFrame(chan_map_info['ElecAtlasProj'])[pd.DataFrame(chan_map_info['ElecAtlasProj']).columns.size-1]
    regions_list = [reg[0] if reg is not None else 'None' for reg in regions]
    networks_list = [net[0] if net is not None else 'None' for net in networks]

    if ((patient=='UIC202521') | (patient=='UIC202522')):
        regions_list.extend(['None']*(224-len(regions_list))) #when no micros, sometimes last few "contacts" are empty, but matfile doesnt list them as none, as theyre at the end
    else:
        regions_list.extend(['None']*(128-len(regions_list))) #when no micros, sometimes last few "contacts" are empty, but matfile doesnt list them as none, as theyre at the end


    ###check to make sure all "None" channels are excluded
    extra_bads_inds = np.where(np.array(networks_list)=='None')[0]
    extra_bads = list(map(lambda x: str(chans[x]),extra_bads_inds)) 
    
    ###REMOVE MICROS
    if remove_micros: #remove channels that are the microwires. Default is True.
        micros_inds = chan_map_info['MicroElec'].astype(int)-1 #microelectrodes, to exclude from LFP analysis (-1 bc data came from matlab, starting from 1)
        bads_all = list(set(bads) | set(extra_bads) | set(np.array(chans)[micros_inds]))
    else: #don't remove any micros
        bads_all = list(set(bads) | set(extra_bads))
    raw.info['bads'] = bads_all
    good_chans_all = [c for c in chans if c not in bads_all] #pick good channels, excluding bads, for picks in next step
    good_chans_inds_all = [chans.index(c) for c in chans if c not in bads_all] #pick good channels, excluding bads, for picks in next step
    
    ####REMOVE WHITE MATTER CHANNELS
    if remove_white: #Default = True
        nonwhite_bools = ['White' not in r and 'Unknown' not in r for r in np.array(regions_list)[good_chans_inds_all]]
        good_chans_all = np.array(good_chans_all)[nonwhite_bools]

    ####GET SUBSET OF CHANNELS THAT ARE IN DMN
    if which_chans_str=='DMN':
        ###white matter electrodes already removed if remove_white=True.
        DMN_chans_inds_all = np.where(np.array(networks_list)=='DMN')[0] 
        DMN_chans_all = list(map(lambda x: str(chans[x]),DMN_chans_inds_all))
        good_chans_DMN = [c for c in good_chans_all if c in DMN_chans_all] #only keep DMN electrodes
        good_chans_subset = good_chans_DMN
    ####GET SUBSET OF CHANNELS THAT ARE IN SPECIFIED REGIONS. 
    elif which_chans_str=='ROI':
        # ROIs_list = ['Hipp', 'parahipp', 'SPL', 'ACgG', 'MFC', 'MFG', 'GRe', 'SFG']
        ROIs_list = input("Enter list of ROIs (e.g., ['Hipp', 'parahipp', 'SPL', 'ACgG', 'MFC', 'MFG', 'GRe', 'SFG']): ")
        ROI_chans = [chans[idx] for idx, s in enumerate(regions_list) if any(ROI in s for ROI in ROIs_list)]
        good_chans_ROIs = [c for c in good_chans_all if c in ROI_chans] #only keep ROI electrodes
        good_chans_subset = good_chans_ROIs
    else:
        # print('test')
        # good_chans_subset = []

        network_chans_inds_all = np.where(np.array(networks_list)==which_chans_str)[0] 
        network_chans_all = list(map(lambda x: str(chans[x]),network_chans_inds_all))
        good_chans_subset = [c for c in good_chans_all if c in network_chans_all] 

    return (good_chans_all, good_chans_subset)




####preprocessing
''' DEAL WITH BAD CHANNELS & REMOVE MICROS BEFORE DOING RE-REFERENCING
'''
def do_preprocessing(patient,data_loc,raw):
    fs = raw.info['sfreq'] #sampling freq
    _, _, _ = get_label_info_all_networks(patient,data_loc,raw) #REMOVES EXTRA BAD CHANNELS
    raw.notch_filter(freqs=[60,120])
    raw.set_eeg_reference(ref_channels = 'average') #re-reference. bads automatically excluded
    # raw.notch_filter(freqs=[60,120]) #filter out electrical noise
    return(raw)




def trial_type_signal(patient,data_loc, raw=None, window = 'entire', trial_type = 'AUT', pre_trial_buffer=0):

    if raw is None:
        #get mne file if not passed to function
        raw = get_raw(patient,data_loc) #get raw mne (with already marked bad channels that were selected manually)
        raw = do_preprocessing(patient,data_loc,raw)
    
    fs = raw.info['sfreq'] #sampling freq
    good_chans_all, good_regions, good_networks = get_label_info_all_networks(patient,data_loc,raw)

    low_f, high_f = get_freq_range('all')
    sig_filtered = raw.copy().filter(l_freq = low_f, h_freq = high_f, picks=good_chans_all)  
    
    trial_info_df = beh_funcs.get_trial_info_df(data_loc)  # get trial info from matlab file

    start_times_ms, end_times_ms, trial_start_times_ms = beh_funcs.get_start_end_times_ms(trial_info_df, trial_type, pre_trial_buffer=pre_trial_buffer, data_loc=data_loc)
    win_start, win_end = beh_funcs.get_window_times(window,start_times_ms,end_times_ms)
    signal_epochs = split_into_epochs(sig_filtered, win_start, win_end)
    signal = signal_epochs.get_data(picks=good_chans_all)
    signal = np.transpose(signal, (2, 0, 1))
    
    return(signal, good_chans_all, good_regions, good_networks)


def trial_type_signal_ripple(patient,data_loc, raw=None, window = 'entire', trial_type = 'AUT', pre_trial_buffer=0, NIP_start_ms = 0):


    patients_data_ripple = patient_params.patients_data_ripple

    if raw is None:
        #get mne file if not passed to function
        raw = get_raw(patient,data_loc) #get raw mne (with already marked bad channels that were selected manually)
        raw = do_preprocessing(patient,data_loc,raw)
    
    fs = raw.info['sfreq'] #sampling freq
    good_chans_all, good_regions, good_networks = get_label_info_all_networks(patient,data_loc,raw)

    low_f, high_f = get_freq_range('all')
    sig_filtered = raw.copy().filter(l_freq = low_f, h_freq = high_f, picks=good_chans_all)  
    
    trial_info_df = beh_funcs.get_trial_info_df(data_loc)  # get trial info from matlab file

    if patient=='UIC202421':
        trial_info_df = trial_info_df.loc[4:]
    if data_loc in np.array(patients_data_ripple)[:,1]:
        trial_info_df = trial_info_df[~trial_info_df.stimFlag]
    if patient=='UIC202420':
        trial_info_df.drop([12], inplace=True) #outlier trial; something weird happened. (no need to drop this, or 8, because only look at trials with AUT responses anyway)
    

    start_times_ms, end_times_ms, trial_start_times_ms = beh_funcs.get_start_end_times_ms(trial_info_df, trial_type, pre_trial_buffer=pre_trial_buffer, data_loc=data_loc)
    start_times_ms = start_times_ms - NIP_start_ms
    end_times_ms = end_times_ms - NIP_start_ms
    win_start, win_end = beh_funcs.get_window_times(window,start_times_ms,end_times_ms)
    signal_epochs = split_into_epochs(sig_filtered, win_start, win_end)
    signal = signal_epochs.get_data(picks=good_chans_all)
    signal = np.transpose(signal, (2, 0, 1))
    
    return(signal, good_chans_all, good_regions, good_networks)





def get_power_over_time_baylor_data(signal, start_times_30khz, end_times_30khz, picked_chans, freq_band):
    low_f, high_f = get_freq_range(freq_band)

    ## Get list of signals for subtrials of interest
    signal_subtrial_list = [signal[:,int(start_times_30khz[i]/30.):int(end_times_30khz[i]/30.)][0] for i in range(len(start_times_30khz))]

    #truncate each array in signals list so that all trials the same length 
    min_trial_len = np.min([list_element.shape[1] for list_element in signal_subtrial_list])
    signal_list_trunc = [[list_element[i][:min_trial_len] for i in range(signal_subtrial_list[0].shape[0])] for list_element in signal_subtrial_list]

    signal_epochs = mne.EpochsArray(signal_list_trunc, signal.info) #turn back into mne "raw"
    
    signal_spec_AUT_over_time = [signal_epochs.compute_psd(fmin = 0.1, fmax = 150, tmin = 0+i, tmax = 1+i, picks=picked_chans).get_data(fmin = low_f, fmax = high_f, picks=picked_chans) for i in np.arange(np.array(signal_epochs).shape[-1]//1000)[:-1]] #pick time window



    signal_power_over_time = np.mean(signal_spec_AUT_over_time,axis=-1) #avg spectrum over frequencies 
    #convert to dB
    signal_power_over_time_db = 10 * np.log10(signal_power_over_time)
 
    return (signal_power_over_time, signal_power_over_time_db)
