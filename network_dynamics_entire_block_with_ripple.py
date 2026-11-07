"""LFP analysis for AUT task ENTIRE BLOCK, include ripple patients
NETWORK DYNAMICS AVERAGED ACROSS PATIENTS
How does gamma vs theta evolve over the ENTIRE timecourse of a BLOCK, for different trial types. (ALL 3 TRIALS IN BLOCK)
    AUT vs. BAT power dynamics for theta & high gamma
Excludes micros

Makes Supp. Fig. 1a.
"""

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

import fastNFxRead


from scipy.interpolate import interp1d



import patient_params
patients_info = patient_params.patients_data
patients_data = DMN_LFP_funcs.choose_patient(patients_info) #choose all or 1 patient

###DEFINE WHICH FREQUENCIES (E.G. THETA, GAMMA), GROUPS OF CHANNELS (ALL CHANS, DMN, CHOSEN ROIS), AND ROIS TO CONSIDER 
freqs_list=['high_gamma', 'theta'] 
# freqs_list=['high_gamma', 'gamma', 'theta', 'beta', 'alpha'] 
trial_types = ['AUT', 'BAT']
# trial_types = ['AUT', 'BAT', 'OCT']
# chans_subsets = ['power_all', 'power_DMN', 'power_ROIs']
ROIs_list = [] ###no ROIs chosen. #alt: ROIs_list = ['Hipp', 'parahipp', 'SPL', 'ACgG', 'MFC', 'MFG', 'GRe', 'SFG'] 
###create empty df, which will be population for each patient, per freq, for group of channels
index = pd.MultiIndex.from_product([np.array(patients_data)[:,0], freqs_list, ['power_all', 'power_DMN', 'power_ROIs'], ['AUT', 'BAT', 'OCT']], names=['person', 'frequency', 'chan_group', 'trial_type'])

window_size=1 #window size for power calculation (passed to compute_psd)
pre_trial_buffer=0 #adding a bit before prompt appears

percent_power_change_all_patients_df = pd.DataFrame(index=index, columns=range(51))  # Assuming 16 timepoints

network = input("Choose cortical network (DMN, FrontoParietal, SomatoMotor, Limbic, Visual, DorsalAttention, VentralAttention): ")

include_ripple = input("Include ripple patients (y/n): ")
if include_ripple:
    patients_info_ripple = patient_params.patients_data_ripple
    patients_data_ripple = patients_info_ripple
    patients_data =  patients_data_ripple + patients_data 

for patient, data_loc in patients_data:
    print(patient)

    if patient=='UIC202420':
        continue
    if patient=='UIC202521':
        continue
    if patient=='UIC202522':
        continue
    if '20250103-115359' in data_loc:
        continue


    raw = DMN_LFP_funcs.get_raw(patient,data_loc) #get raw mne (with already marked bad channels that were selected manually)
    chans = raw.ch_names
    if data_loc in np.array(patients_data_ripple)[:,1]:
        nf3_file = glob.glob(data_loc + '/' + '*.nf3')[0] #
        print('loading ripple data')
        header, data = fastNFxRead.fastNFxRead(nf3_file)
        NIP_start_ms = header['NIPStart']/30. #recording start time. need to subtract this from the NIP timestamps (only for Ripple recordings)
        raw = raw.resample(sfreq=1000)
    else:
        NIP_start_ms = 0
    raw =  DMN_LFP_funcs.do_preprocessing(patient,data_loc,raw)

    # #match electrodes to regions
    matfile = DMN_LFP_funcs.get_matfile(patient, data_loc)
    remove_micros = DMN_LFP_funcs.remove_micros_bool(data_loc)

    chan_info = mat73.loadmat(matfile[0])
    regions = pd.DataFrame(chan_info['ElecAtlasProj'])[0]
    regions_list = [reg[0] if reg is not None else 'None' for reg in regions]

    good_chans_all, good_chans_DMN = DMN_LFP_funcs.get_chan_info(raw, chan_info, network, remove_micros=remove_micros, remove_white=True)
    good_chans_ROIs=[]

    # get trial info from matlab file
    trial_info_df = beh_funcs.get_trial_info_df(data_loc)
    if patient=='UIC202421':
        trial_info_df = trial_info_df.loc[4:]
        trial_info_df.drop(18, inplace=True) 
    if data_loc in np.array(patients_data_ripple)[:,1]: #this is a ripple session, remove stimulation trials
        trial_info_df = trial_info_df[~trial_info_df.stimFlag]
        if patient=='UIC202422': #this patient during the ripple session
            patient = 'UIC202422b' #rename for dataframe later, so can include both sessions for this patient 


    for trial_type in trial_types:    ## compare trial types
        # ##Find where practice trials end and real session starts. Practice AUT and OCT trials start with promot "key"; BAT trials start with '10 + 1 = '
        start_times_ms, end_times_ms, trial_start_times_ms = beh_funcs.get_start_end_times_ms(trial_info_df, trial_type, pre_trial_buffer=pre_trial_buffer, data_loc=data_loc)
        _, _, trial_start_times_ms_all = beh_funcs.get_start_end_times_ms(trial_info_df, 'all', pre_trial_buffer=pre_trial_buffer, data_loc = data_loc)

        start_times_ms = start_times_ms - NIP_start_ms
        end_times_ms = end_times_ms - NIP_start_ms
        trial_start_times_ms = trial_start_times_ms - NIP_start_ms
        trial_start_times_ms_all = trial_start_times_ms_all - NIP_start_ms
        
        trial_subset_df = beh_funcs.trial_type_subset_df(trial_info_df, trial_type, practice_prompts=[['key'], ['10 + 1 = ']]) 
        trial_end_times_30khz = trial_subset_df['respDurFinishNIP']
        trial_end_times_ms = trial_end_times_30khz.values/30. - NIP_start_ms


        inds_temp = [np.where((start_times_ms < trial_start_times_ms[i+1]) & (start_times_ms > trial_start_times_ms[i]))[0] for i in range(len(trial_start_times_ms)-1)]
        # #the above step didnt include the subtrial start times AFTER the last trial start
        inds_temp.append(np.where(start_times_ms > trial_start_times_ms[-1])[0])
        inds_temp_df = pd.DataFrame(inds_temp)
        inds_for_reorder = [int(np.where(inds_temp_df==ind)[0]) for ind in range(len(start_times_ms))]
        block_starts_repeated = trial_start_times_ms[inds_for_reorder]
        trial_starts_rel_to_block = start_times_ms - block_starts_repeated

        #if want to blot entire blocks  
        # start_times_ms = trial_start_times_ms
        # end_times_ms = trial_start_times_ms + 56000

        for i, freq_band in enumerate(freqs_list):
            for j, which_chans in enumerate([[], good_chans_DMN, good_chans_ROIs]):
                if len(which_chans)==0:
                    continue

                block_signals = []
                for i, starts_in_block in enumerate(start_times_ms[inds_temp]):
                    #filter the sig into the desired freq band; get avg baseline power for each trial; then baseline correct
                    filtered_signal, power_over_time, power_over_time_db = DMN_LFP_funcs.get_power_over_time_from_unfiltered(raw, starts_in_block-2000, starts_in_block + 15000, which_chans,freq_band, window_size=window_size)
                    baseline_power, baseline_power_db = DMN_LFP_funcs.get_power_for_trials_subset(filtered_signal, trial_start_times_ms - (2 * 1000), trial_start_times_ms - (1*1000), which_chans, freq_band)
                    avg_uncorrected_signal = np.mean(power_over_time,axis=(1,2))

                    ##BASELINE CORRECT:
                    baseline_correction_method = DMN_LFP_funcs.baseline_correct_percent_method_1
                    power_base_corrected_over_time = baseline_correction_method(power_over_time, baseline_power, trial_start_times_ms_all, start_times_ms)
                    avg_across_chans = np.mean(power_base_corrected_over_time,axis=(2)) #avg across   chans


                    block_signals.append(avg_across_chans.T.flatten())


                mean_block_signal = np.mean(block_signals,axis=0)
                percent_power_change_all_patients_df.loc[(patient,freq_band,['power_all', 'power_DMN', 'power_ROIs'][j], trial_type)] = mean_block_signal

#avg across patients:
# percent_power_change_avg = percent_power_all_patients_df.groupby(level=['frequency', 'chan_group', 'trial_type']).mean()-100 #-100 if want it to be percent CHANGE from baseline
percent_power_change_avg = percent_power_change_all_patients_df.groupby(level=['frequency', 'chan_group', 'trial_type']).mean() # no longer -100 because function now does percent change
error_patients = percent_power_change_all_patients_df.groupby(level=['frequency', 'chan_group', 'trial_type']).sem() # no longer -100 because function now does percent change


#plot 2 figs: 1. just AUT power dynamics; 2. ALl 3 trial types.
#make and use annotate function, so that can apply same annotations to both figs.
from mpl_helpers import PdfPlotter
PdfPlotter('Figures/network_dynamics_entire_block_' + network + '_plus_ripple.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))
chans_subsets = ['power_all', 'power_DMN'] 
# chans_subsets = ['power_DMN'] 
f1,ax1 = plt.subplots(len(freqs_list),len(chans_subsets), sharex=True, sharey='row')
f2,ax2 = plt.subplots(len(freqs_list),len(chans_subsets), sharex=True, sharey='row')

#if mess with window size of pre-trial buffer, need this to get accurate x axis
x = np.linspace(0-pre_trial_buffer/1000,percent_power_change_avg.shape[-1]*window_size-pre_trial_buffer/1000,percent_power_change_avg.shape[-1],endpoint=False)
for i, freq_band in enumerate(freqs_list):
    for j, chans_sub in enumerate(chans_subsets):
        ax1[i][j].plot(x, percent_power_change_avg.loc[freq_band].loc[chans_sub].iloc[0]) #plot just AUT
        ax2[i][j].plot(x, percent_power_change_avg.loc[freq_band].loc[chans_sub].T) #plot 3 lines (AUT, BAT, OCT). .T so that it makes data columns instead of rows and can plot all 3 at once
ax2[0][0].legend(percent_power_change_avg.loc[freq_band].loc['power_DMN'].T.columns.values)


# annotate function, so that can apply same annotations to both figs.
def annotate1(f, ax, freqs_list, chans_subsets):
    for i, freq_band in enumerate(freqs_list):
        for j, chans_sub in enumerate(chans_subsets):
            ax[i][j].axvline(x=1,color='k',linewidth = 0.5, linestyle='--') 
            ax[i][j].axvline(x=18,color='k',linewidth = 0.5, linestyle='--')
            ax[i][j].axvline(x=35,color='k',linewidth = 0.5, linestyle='--')
            # ax[0][j].set_title(['all chans','DMN network chans', 'DMN ROIs'][j],fontsize=10, loc='left')
            left = ax[0][j].get_position().bounds[0]
            f.text(left+0.1,0.9,['all chans','DMN network chans', 'DMN ROIs'][j],ha='center', va='center',fontsize=10)
            ax[-1][j].set_xlabel('time (s)')
            sns.despine(top = True, right = True)
        ax[i][0].set_title(freq_band + ' power dynamics',fontsize=14, loc='left')
    # ax[0][0].set_ylabel('% power change')
    f.text(0.07,0.5,'% power change',rotation=90, ha='center', va='center',fontsize=10)

    f.subplots_adjust(hspace=0.5)

# annotate function, for plots with only 1 group of chans plotted
def annotate2(f, ax, freqs_list):
    for i, freq_band in enumerate(freqs_list):
            ax[i].axvline(x=1,color='k',linewidth = 0.5, linestyle='--') 
            ax[i].axvline(x=18,color='k',linewidth = 0.5, linestyle='--')
            ax[i].axvline(x=35,color='k',linewidth = 0.5, linestyle='--')
            trans = ax[i].get_xaxis_transform()

            ax[i].set_title(freq_band,fontsize=14, loc='left')
            ax[i].set_xlabel('time (s)')
            ax[0].set_ylabel('% power change',rotation=90, ha='center', va='center',fontsize=10)
            sns.despine(top = True, right = True)
        # ax[i][0].set_title(freq_band + ' power dynamics',fontsize=14, loc='left')
    # ax[0][0].set_ylabel('% power change')
    # f.text(0.07,0.5,'% power change',rotation=90, ha='center', va='center',fontsize=10)

    f.subplots_adjust(hspace=0.5)

def annotate2(f, ax, freqs_list):
    ax.axvline(x=1,color='k',linewidth = 0.5, linestyle='--') 
    ax.axvline(x=18,color='k',linewidth = 0.5, linestyle='--')
    ax.axvline(x=35,color='k',linewidth = 0.5, linestyle='--')
    trans = ax.get_xaxis_transform()

    ax.set_title(freq_band,fontsize=14, loc='left')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('% power change',rotation=90, ha='center', va='center',fontsize=10)
    sns.despine(top = True, right = True)

    f.subplots_adjust(hspace=0.5)

annotate1(f1, ax1, freqs_list, chans_subsets)
annotate1(f2, ax2, freqs_list, chans_subsets)





chans_subsets = ['power_DMN'] 
# freqs_list = ['high_gamma']
f1,ax1 = plt.subplots(1, len(freqs_list))
f2,ax2 = plt.subplots(1, len(freqs_list))
colors = ['darkorange','cornflowerblue','limegreen']
x = np.linspace(0-pre_trial_buffer/1000,percent_power_change_avg.shape[-1]*window_size-pre_trial_buffer/1000,percent_power_change_avg.shape[-1],endpoint=False)
for i, freq_band in enumerate(freqs_list):

        
    values = percent_power_change_avg.loc[freq_band].loc['power_DMN']
    errors = error_patients.loc[freq_band].loc['power_DMN']
    
    for t,trial_type in enumerate(trial_types):
        values_numeric = pd.to_numeric(values.loc[trial_type], errors='coerce') #because getting isfinite error... needs go be float not object
        errors_numeric = pd.to_numeric(errors.loc[trial_type], errors='coerce')

        ax2[i].plot(x, values_numeric, color = colors[t],label=trial_type)
        ax2[i].fill_between(x, 
                            values_numeric - errors_numeric, 
                            values_numeric + errors_numeric, 
                            color=colors[t], alpha=0.25)

ax2[-1].legend(trial_types)
annotate2(f2, ax2, freqs_list)


plt.show()
plt.close()
