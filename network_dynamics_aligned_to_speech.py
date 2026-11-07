"""LFP analysis for AUT task
NETWORK DYNAMICS AVERAGED ACROSS PATIENTS
RELATIVE TO SPEECH ONSET

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

import librosa


import sys
sys.path.append('../')

import brpylib
import fastNFxRead

import DMN_LFP_funcs
import beh_funcs

import patient_params
patients_info = patient_params.patients_data
patients_data = DMN_LFP_funcs.choose_patient(patients_info) #choose all or 1 patient

patients_data = patients_info
# patients_data = patients_info[-2:]

###DEFINE WHICH FREQUENCIES (E.G. THETA, GAMMA), GROUPS OF CHANNELS (ALL CHANS, DMN, CHOSEN ROIS), AND ROIS TO CONSIDER 
freqs_list=['high_gamma', 'theta'] 
trial_types = ['AUT', 'BAT']
ROIs_list = [] ###no ROIs chosen. #alt: ROIs_list = ['Hipp', 'parahipp', 'SPL', 'ACgG', 'MFC', 'MFG', 'GRe', 'SFG'] 
###create empty df, which will be population for each patient, per freq, for group of channels
index = pd.MultiIndex.from_product([np.array(patients_data)[:,0], freqs_list, ['power_all', 'power_DMN', 'power_ROIs'], ['AUT', 'BAT', 'OCT']], names=['person', 'frequency', 'chan_group', 'trial_type'])

window_size=1 #window size for power calculation (passed to compute_psd)
pre_trial_buffer=10000 #adding a bit before prompt appears
AUDIO_OFFSET_S = 0.0

percent_power_change_all_patients_df = pd.DataFrame(index=index, columns=range(int((6 + pre_trial_buffer//1000)/window_size)))  # Assuming 16 timepoints

network = input("Choose cortical network (DMN, FrontoParietal, SomatoMotor, Limbic, Visual, DorsalAttention, VentralAttention): ")

from mpl_helpers import PdfPlotter
PdfPlotter('Figures/network_dynamics_aligned_to_speech_AUT_BAT.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))

def detect_first_speech_onset(
    wav_path,
    window_start_s,
    window_end_s,
    sr=16000,
    frame_ms=10,
    smooth_ms=50,
    threshold_z=4.0,
    min_speech_ms=80,
    ignore_first_ms=100,
    plot=False,
    title=None,
)   :
    """
    Detect first sustained acoustic onset in a response window.

    Returns
    -------
    onset_s : float or np.nan
        Detected onset time in WAV seconds.
    """

    duration = window_end_s - window_start_s

    y, _ = librosa.load(
        wav_path,
        sr=sr,
        mono=True,
        offset=max(0, window_start_s),
        duration=duration
    )

    if len(y) == 0:
        return np.nan

    frame_len = int(sr * frame_ms / 1000)
    hop_len = frame_len

    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_len,
        hop_length=hop_len
    )[0]

    # Smooth RMS envelope
    smooth_n = max(1, int(smooth_ms / frame_ms))
    kernel = np.ones(smooth_n) / smooth_n
    rms_smooth = np.convolve(rms, kernel, mode="same")

    times_s = (
        librosa.frames_to_time(
            np.arange(len(rms_smooth)),
            sr=sr,
            hop_length=hop_len
        )
        + window_start_s
    )

    # Ignore the immediate visual transition / click / button release if needed
    valid = times_s >= (window_start_s + ignore_first_ms / 1000)

    # Estimate noise floor from quieter half of the response window
    quiet_vals = rms_smooth[valid]
    quiet_vals = quiet_vals[quiet_vals <= np.percentile(quiet_vals, 50)]

    if len(quiet_vals) < 5:
        return np.nan

    baseline = np.median(quiet_vals)
    noise_sd = np.std(quiet_vals)
    threshold = baseline + threshold_z * noise_sd

    above = (rms_smooth > threshold) & valid

    min_frames = max(1, int(min_speech_ms / frame_ms))

    onset_s = np.nan
    for i in range(0, len(above) - min_frames):
        if np.all(above[i:i + min_frames]):
            onset_s = times_s[i]
            break

    if plot:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(times_s, rms_smooth)
        ax.axhline(threshold, linestyle="--")
        if not np.isnan(onset_s):
            ax.axvline(onset_s, linestyle="--")
        ax.axvline(window_start_s, linestyle=":")
        ax.axvline(window_end_s, linestyle=":")
        ax.set_xlabel("WAV time (s)")
        ax.set_ylabel("Smoothed RMS")
        ax.set_title(title or f"{window_start_s:.2f}–{window_end_s:.2f} s")
        plt.show()

    return onset_s


include_ripple = input("Include ripple patients (y/n): ")
if include_ripple=='y':
    patients_info_ripple = patient_params.patients_data_ripple
    patients_data_ripple = patients_info_ripple
    patients_data = patients_data_ripple + patients_data 
else:
    NIP_start_ms = 0

for patient, data_loc in patients_data:
    print(patient)

    # if patient=='UIC202420':
    #     trial_info_df.drop([8,12], inplace=True) #repeat trial (8) and outlier trial (12)
    if patient=='UIC202420':
        continue
    if patient=='UIC202521':
        continue
    if patient=='UIC202522':
        continue
    if '20250103-115359' in data_loc:
        continue

    print(patient)

    trial_info_df = beh_funcs.get_trial_info_df(data_loc)
    wav_path = data_loc + '/TaskAudio_' + data_loc[-15:] + '.wav'


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

    # good_chans_DMN is actually good chan subset for chosen network, not necessarily DMN
    good_chans_all, good_chans_DMN = DMN_LFP_funcs.get_chan_info(raw, chan_info, network, remove_micros=remove_micros, remove_white=True) 
    if len(good_chans_DMN)==0:
        continue
    

    # get trial info from matlab file
    # trial_info_df = DMN_LFP_funcs.get_trial_info_df(data_loc)
    trial_info_df = beh_funcs.get_trial_info_df(data_loc)
    if patient=='UIC202421':
        trial_info_df = trial_info_df.loc[4:]
        trial_info_df.drop(18, inplace=True) 
    if data_loc in np.array(patients_data_ripple)[:,1]: #this is a ripple session, remove stimulation trials
        trial_info_df = trial_info_df[~trial_info_df.stimFlag]
        if patient=='UIC202422': #this patient during the ripple session
            patient = 'UIC202422b' #rename for dataframe later, so can include both sessions for this patient 



    good_chans_ROIs=[]

    # get trial info from matlab file

    for trial_type in trial_types:    ## compare trial types
        # ##Find where practice trials end and real session starts. Practice AUT and OCT trials start with promot "key"; BAT trials start with '10 + 1 = '
        # trial_subset_df = DMN_LFP_funcs.trial_type_subset_df(trial_info_df, trial_type, practice_prompts=[['key'], ['10 + 1 = ']]) 
        start_times_ms, end_times_ms, trial_start_times_ms = beh_funcs.get_start_end_times_ms(trial_info_df, trial_type, pre_trial_buffer=pre_trial_buffer, data_loc=data_loc)


        start_times_ms = start_times_ms - NIP_start_ms
        end_times_ms = end_times_ms - NIP_start_ms
        trial_start_times_ms = trial_start_times_ms - NIP_start_ms

        speech_window_start_ms = end_times_ms
        speech_window_end_ms = end_times_ms + 6000

        speech_window_start_s = speech_window_start_ms / 1000 + AUDIO_OFFSET_S
        speech_window_end_s = speech_window_end_ms / 1000 + AUDIO_OFFSET_S


        audio_onsets_s = []

        for i, (ws, we) in enumerate(zip(speech_window_start_s, speech_window_end_s)):
            onset = detect_first_speech_onset(
                wav_path,
                ws,
                we,
                threshold_z=4.0,
                min_speech_ms=80,
                ignore_first_ms=200,
                plot=False,
                title=f"Trial {i + 1}"
            )
            if ~np.isnan(onset):
                audio_onsets_s.append(onset)

        audio_onsets_s = np.array(audio_onsets_s)

        speech_onset_ms = np.round((audio_onsets_s - AUDIO_OFFSET_S) * 1000)

        # response_onset_eeg_sample = np.round(response_onset_ms)


        for i, freq_band in enumerate(freqs_list):
            for j, which_chans in enumerate([[], good_chans_DMN, []]):
                if len(which_chans)==0:
                    continue
                #filter the sig into the desired freq band; get avg baseline power for each trial; then baseline correct
                filtered_signal, power_over_time, power_over_time_db = DMN_LFP_funcs.get_power_over_time_from_unfiltered(raw, speech_onset_ms - pre_trial_buffer, speech_onset_ms + (6*1000),which_chans,freq_band, window_size=window_size)
                baseline_power, baseline_power_db = DMN_LFP_funcs.get_power_for_trials_subset(filtered_signal, trial_start_times_ms - (2 * 1000), trial_start_times_ms - (1*1000), which_chans, freq_band)
                avg_uncorrected_signal = np.mean(power_over_time,axis=(1,2))

                ##BASELINE CORRECT:
                baseline_correction_method = DMN_LFP_funcs.baseline_correct_percent_method_1
                power_base_corrected_over_time = baseline_correction_method(power_over_time, baseline_power, trial_start_times_ms, start_times_ms, pre_trial_buffer = pre_trial_buffer)
                avg_power_base_corrected_over_time = np.mean(power_base_corrected_over_time,axis=(1,2)) #avg across trials and chans

                percent_power_change_all_patients_df.loc[(patient,freq_band,['power_all', 'power_DMN', 'power_ROIs'][j], trial_type)] = avg_power_base_corrected_over_time



percent_power_change_avg = percent_power_change_all_patients_df.groupby(level=['frequency', 'chan_group', 'trial_type']).mean() # no longer -100 because function now does percent change
error_patients = percent_power_change_all_patients_df.groupby(level=['frequency', 'chan_group', 'trial_type']).sem() # no longer -100 because function now does percent change

std_patients = percent_power_change_all_patients_df.groupby(level=['frequency', 'chan_group', 'trial_type']).std() # no longer -100 because function now does percent change


chans_subsets = ['power_DMN'] 
freqs_list = ['high_gamma', 'theta']
# f1,ax1 = plt.subplots(1, len(freqs_list))
f2,ax2 = plt.subplots(1, len(freqs_list))
colors = ['darkorange','cornflowerblue','limegreen']
x = np.linspace(0-pre_trial_buffer/1000,percent_power_change_avg.shape[-1]*window_size-pre_trial_buffer/1000,percent_power_change_avg.shape[-1],endpoint=False)
for i, freq_band in enumerate(freqs_list):
    lines = []
        
    values = percent_power_change_avg.loc[freq_band].loc['power_DMN']
    errors = error_patients.loc[freq_band].loc['power_DMN']
    
    for t,trial_type in enumerate(trial_types):
        values_numeric = pd.to_numeric(values.iloc[t], errors='coerce') #because getting isfinite error... needs go be float not object
        errors_numeric = pd.to_numeric(errors.iloc[t], errors='coerce')

        line = ax2[i].plot(x, values_numeric, color = colors[t],label=trial_type)
        lines.append(line)
        ax2[i].fill_between(x, 
                            values_numeric - errors_numeric, 
                            values_numeric + errors_numeric, 
                            color=colors[t], alpha=0.25)
                            
    ax2[i].axvline(x=0,color='k',linewidth = 0.5, linestyle='--')
ax2[-1].legend()
