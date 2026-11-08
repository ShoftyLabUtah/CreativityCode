#Spectrograms for creativity task


import numpy as np
# from neo import io
import quantities as pq
import glob   
import matplotlib.pyplot as plt
# import pickle
import pandas as pd
# import seaborn as sns
import mat73

import itertools as it
# import random


import sys
sys.path.append('../')

import DMN_LFP_funcs
import beh_funcs

import re

import mne

import scipy

import sys
sys.path.append('../')
from mpl_helpers import PdfPlotter


# data_loc = '/Volumes/155.100.91.44/Data/UIC202307/DMN/20230529-115352'
# patient = 'UIC202307'
# data_loc = '/Volumes/155.100.91.44/Data/UIC202314/DMN/20231208-133604'
# patient = 'UIC202314'
# data_loc = '/Volumes/155.100.91.44/Data/UIC202404/DMN/20240306-134703' 
# patient = 'UIC202404'
# data_loc = '/Volumes/155.100.91.44/Data/UIC202308/DMN/20230802-122445'
# patient = 'UIC202308'
# data_loc = '/Volumes/155.100.91.44/Data/UIC202401/DMN/20240115-101116' ###USE CHANNELMAP2
# patient = 'UIC202401'
data_loc = '/Volumes/155.100.91.44/Data/UIC202409/DMN/20240510-142059' 
patient = 'UIC202409'

trial_type = input("AUT or BAT or OCT? ")
network = input("Choose cortical network (DMN, FrontoParietal, SomatoMotor, Limbic, Visual, DorsalAttention, VentralAttention): ")

PdfPlotter('Figures/' + patient+ '_spectrograms_' + trial_type + '_' + network + '.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))

#########       TRIAL INFO         ##########

##### get trial info from matlab file
matfile = glob.glob(data_loc + '/TrialInfo*')
try:
    trial_info = scipy.io.loadmat(matfile[0])
    trial_info_df = pd.DataFrame(trial_info['TrialInfo'][0])
except:
    trial_info = mat73.loadmat(matfile[0])
    trial_info_df = pd.DataFrame(trial_info['TrialInfo'])

#Find where practice trials end and real session starts. Practice AUT and OCT trials start with promot "key"; BAT trials start with '10 + 1 = '
# AUT_trial_df = beh_funcs.trial_type_subset_df(trial_info_df, 'AUT', practice_prompts=[['key'], ['10 + 1 = ']]) #just the AUT trials
AUT_trial_df = beh_funcs.trial_type_subset_df(trial_info_df, trial_type, practice_prompts=[['key'], ['10 + 1 = ']]) #just the AUT trials

#find time slices when AUT trials are happening
AUT_trial_start_times = AUT_trial_df['respDurStartNIP']
AUT_trial_end_times = AUT_trial_df['respDurFinishNIP']

#get times for each sub-trial (3 items per trial)
AUT_start_end_info = AUT_trial_df['respRepStartNIP']
subtrial_start_times = np.concatenate([AUT_start_end_info.iloc[i][:,0] for i in np.arange(len(AUT_start_end_info))])
subtrial_end_times = np.concatenate([AUT_start_end_info.iloc[i][:,1] for i in np.arange(len(AUT_start_end_info))])


#######   LOAD LFP DATA   #######
raw = DMN_LFP_funcs.get_raw(patient,data_loc) #get raw mne (with already marked bad channels that were selected manually)

chans = raw.ch_names
fs = raw.info['sfreq'] #sampling freq

#######      CURATE ELECTRODES    ##########
### 1. ID and remove bad channels
### 2. find and remove "empty" channels (e.g, not in brain, so no region labels)
### 3. PICK NETWORK (DMN) OR REGIONS OF INTEREST  

# #match electrodes to regions
matfile = glob.glob('/Volumes/155.100.91.44/Data/' + patient + '/Imaging/Registered/ChannelMap.mat')
remove_micros = True
if '20240115-101116' in data_loc: #didn't record micros; use ChannelMap2 instead
    matfile = glob.glob('/Volumes/155.100.91.44/Data/' + patient + '/Imaging/Registered/ChannelMap2.mat')
    remove_micros = False

chan_info = mat73.loadmat(matfile[0])
regions = pd.DataFrame(chan_info['ElecAtlasProj'])[0]
regions_list = [reg[0] if reg is not None else 'None' for reg in regions]
networks = pd.DataFrame(chan_info['ElecAtlasProj'])[pd.DataFrame(chan_info['ElecAtlasProj']).columns.size-1]
networks_list = [net[0] if net is not None else 'None' for net in networks]

good_chans_all, good_chans_DMN = DMN_LFP_funcs.get_chan_info(raw, chan_info, network, remove_micros=remove_micros, remove_white=True)

raw.notch_filter(freqs=[60,120],picks = good_chans_all) #filter out electrical noise
raw.set_eeg_reference(ref_channels = 'average') #re-reference. Probably variable should be renamed something more meaningful than "test", but oh well

# # raw_standard.plot(block=True) #manually select bad channels in gui
# bads = raw_standard.info['bads']

# #check to make sure all "None" channels are excluded
# extra_bads_inds = np.where(np.array(networks_list)=='None')[0]
# extra_bads = list(map(lambda x: str(chans[x]),extra_bads_inds)) 
# bads_all = list(set(bads) | set(extra_bads))
# raw.info['bads'] = bads_all #bad channels are now marked in the info
# good_chans_all = [c for c in chans if c not in bads_all] #pick good channels (excluding bads)

####   PICK NETWORK (DMN) OR REGIONS OF INTEREST; match electrodes to regions
# Find DMN electrodes
# DMN_electrodes_inds = np.where(np.array(networks_list)=='DMN')[0] 
# DMN_electrodes = list(map(lambda x: str(chans[x]),DMN_electrodes_inds)) #add +1 because electrodes numbering starts with 1, not 0, then turn to string


# 1/0

# find electrodes in given region (e.g. MTG)
# ROI = 'MOrG'
# ROIs_list = ['Hipp', 'parahipp', 'SPL', 'ACgG', 'MFC', 'MFG', 'GRe', 'SFG']
# # ROI_electrodes = [str(chans[idx]) for idx, s in enumerate(regions_list) if ROI in s and str(chans[idx]) in good_chans_all]
# # ROI_electrodes_inds = [idx for idx, s in enumerate(regions_list) if ROI in s and str(chans[idx]) in good_chans_all]
# electrodes_and_inds = [(chans[idx],idx) for idx, s in enumerate(regions_list) if any(ROI in s and chans[idx] in good_chans_all for ROI in ROIs_list)]
# ROI_electrodes = np.array(electrodes_and_inds, dtype=[('col1', '<U21'), ('col2', int)])['col1'] #have to specify dtype, or else np.array turns the inds into strings!
# ROI_electrodes_inds = np.array(electrodes_and_inds, dtype=[('col1', '<U21'), ('col2', int)])['col2']




#######   SPECTROGRAMS     ###########



########    Avg spectrogram for multiple chans  #########
#spectrogram for channels in a specific region

#####
#do wavelet transform on entire signal before trial averaging
#get signal for just AUT trials (and pre-trial baselines)
subtrial_sig_list = [raw[:,int((subtrial_start_times - (0*30000))[i]/30.):int((subtrial_start_times + (18*30000))[i]/30.)][0] for i in np.arange(len(subtrial_start_times))]
ITI_list = [raw[:,int((AUT_trial_start_times.values - (5 * 30000))[i]/30.):int((AUT_trial_start_times.values - (0 * 30000))[i]/30.)][0] for i in np.arange(len(AUT_trial_start_times))]

#adds 5 sec pre-block baseline to the start of each trial
composite_list = np.array([np.concatenate((ITI_list[t//3],subtrial_sig_list[t]),axis=-1) for t in np.arange(len(subtrial_sig_list))])

# for ROI in ['PCgG']:
#     print(ROI)
#     ROI_electrodes = [str(chans[idx]) for idx, s in enumerate(regions_list) if ROI in s and str(chans[idx]) in good_chans_all]
#     ROI_electrodes_inds = [idx for idx, s in enumerate(regions_list) if ROI in s and str(chans[idx]) in good_chans_all]

DMN_electrodes_inds = [idx for idx, s in enumerate(networks_list) if network == s and str(chans[idx]) in good_chans_all]

ROI_electrodes_inds = DMN_electrodes_inds

signal_ROI_chans = composite_list[:,ROI_electrodes_inds,:]
# baseline_chans = composite_list[:,ROI_electrodes_inds,4000:4500]

concatenated_trials = np.concatenate(composite_list,axis=1)
entire_signal_ROI_chans = concatenated_trials[ROI_electrodes_inds,:]

w = 10.
# freq = np.linspace(1, fs/10, 150)
freq = np.linspace(1, 150, int(fs/10))
widths = w*fs / (2*freq*np.pi)
t, dt = np.linspace(0, entire_signal_ROI_chans.shape[-1]/1000, entire_signal_ROI_chans.shape[-1], retstep=True)


n_trials = len(subtrial_sig_list)
n_roi_chans = len(ROI_electrodes_inds)
n_freqs = len(freq)

# Get one composite trial to infer trial length
example_trial = np.concatenate(
    (ITI_list[0], subtrial_sig_list[0]),
    axis=-1
)
n_time = example_trial.shape[-1]

# Accumulators only. These are much smaller than all trial CWTs.
sum_spect_signal = np.zeros(
    (n_roi_chans, n_freqs, n_time),
    dtype=np.float32
)

sum_baseline = np.zeros(
    (n_roi_chans, n_freqs),
    dtype=np.float64
)

n_baseline_timepoints_total = 0

for t in range(n_trials):
    composite_trial = np.concatenate(
        (ITI_list[t // 3], subtrial_sig_list[t]),
        axis=-1
    )

    # restrict to ROI channels for this trial only
    trial_ROI = composite_trial[ROI_electrodes_inds, :]

    for ch_i in range(n_roi_chans):
        cwt = scipy.signal.cwt(
            trial_ROI[ch_i],
            scipy.signal.morlet2,
            widths,
            w=w
        )

        spect = np.abs(cwt).astype(np.float32)

        # Accumulate full trial spectrogram
        sum_spect_signal[ch_i] += spect

        # Accumulate baseline
        sum_baseline[ch_i] += spect[:, 3000:4000].sum(axis=-1)

        del cwt, spect

    n_baseline_timepoints_total += 1000

# Average over trials
split_spect_signal_trial_avg = sum_spect_signal / n_trials

# Average baseline over trials and baseline time
avg_base_spect = sum_baseline / n_baseline_timepoints_total

# Baseline correction
baseline_corrected_signal = (
    split_spect_signal_trial_avg / avg_base_spect[:, :, np.newaxis]
) * 100 - 100

mean_signal = np.mean(baseline_corrected_signal,axis=0) #average across chans


# #baseline correct: divide all trial sigs by baseline avg
bounds = max(np.abs(np.percentile(baseline_corrected_signal,1)),np.abs(np.percentile(baseline_corrected_signal,99))) #determine colorbar min and max
t_plot = np.linspace(0, baseline_corrected_signal.shape[-1]/1000, baseline_corrected_signal.shape[-1])
fig,ax=plt.subplots(figsize=(12,4))
cmap = plt.colormaps["turbo"]
s = ax.pcolormesh(t_plot-5,freq[:150],np.mean(baseline_corrected_signal,axis=0)[:150,:],vmin=-65,vmax=65,cmap=cmap,rasterized=True)

plt.colorbar(s,ax=ax)

ax.axvline(x=10,color='k',linewidth = 1)
ax.axvline(x=0,color='k',linewidth = 1)
trans = ax.get_xaxis_transform()
ax.text(0,0.5,'stimulus',rotation=90,transform=trans)
ax.text(10,0.5,'response',rotation=90,transform=trans)
ax.set_xlim(-2,16)
ax.set_title(network + ' ' + trial_type +  ' Spectrogram (baseline corrected)')

plt.show()
