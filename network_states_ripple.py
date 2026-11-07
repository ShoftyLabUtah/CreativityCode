'''
analysis from Utah AUT task
determines network states: finds signal power at different recording sites for high vs low creative states,
for any frequency band (specify at the beginning)

Excludes micros

includes ripple patients (option to include)

Makes Figure 4a and Supp. Fig. 6a


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

import fastNFxRead

import sys
sys.path.append('../')


import patient_params
patients_info = patient_params.patients_data
patients_data = DMN_LFP_funcs.choose_patient(patients_info) #choose all or 1 patient

from mpl_helpers import PdfPlotter

network = input("Choose cortical network (DMN, FrontoParietal, SomatoMotor, Limbic, Visual, DorsalAttention, VentralAttention): ")
window_of_interest = input("Choose window ('stimulus' or 'response'): ")
freq_band = input("Choose frequency band: ")
include_ripple = input("Include ripple patients (y/n): ")
if include_ripple=='y':
    patients_info_ripple = patient_params.patients_data_ripple
    patients_data_ripple = patients_info_ripple
    patients_data = patients_data + patients_data_ripple
    PdfPlotter('Figures/all_patients_' + window_of_interest + '_' + freq_band + '_network states_reorder_' + network + '_ripple.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))
else:
    PdfPlotter('Figures/all_patients_' + window_of_interest + '_' + freq_band + '_network states_reorder_' + network + '.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))


###create empty df, which will be population for each patient, per freq, for group of channels
index = pd.MultiIndex.from_product([np.array(patients_data)[:,0], []], names=['person', 'regions'])
results_all_patients_df = pd.DataFrame(index=index, columns=range(2))  # 2 datapts: high creativity power, low creativity power
values_all_patients = []
values_all_patients_uncorr = []
regions_all_patients = []

high_creative_avg_values = []
low_creative_avg_values = []

high_creative_values_unavg = []
low_creative_values_unavg = []


#for plotting over time
power_change_over_time_all_patients_high_orig = []
power_change_over_time_all_patients_low_orig = []



for patient, data_loc in patients_data:
    print(patient)
    if patient=='UIC202420':
        continue
    if '20250103-115359' in data_loc:
        continue
    patient_processed_AUT_datafile = 'scored_responses/' + patient + '_processed_data.csv'  #file that includes responses and originality scores:
    if '20241231-144952' in data_loc:
        patient_processed_AUT_datafile = 'scored_responses/' + patient + '_processed_data_day1.csv'  

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
    if len(good_chans_DMN)==0:
        continue

    # get trial info from matlab file
    trial_info_df = beh_funcs.get_trial_info_df(data_loc)
    if patient=='UIC202421':
        trial_info_df = trial_info_df.loc[4:]
    if data_loc in np.array(patients_data_ripple)[:,1]:
        trial_info_df = trial_info_df[~trial_info_df.stimFlag]
    if patient=='UIC202420':
        trial_info_df.drop([12], inplace=True) #outlier trial; something weird happened. (no need to drop this, or 8, because only look at trials with AUT responses anyway)
    
    ###Find where practice trials end and real session starts. Practice AUT and OCT trials start with promot "key"; BAT trials start with '10 + 1 = '
    AUT_trial_df = beh_funcs.trial_type_subset_df(trial_info_df, 'AUT', practice_prompts=[['key'], ['10 + 1 = ']]) #just the AUT trials

    #find time slices when AUT trials are happening
    AUT_trial_start_times_30khz = AUT_trial_df['respDurStartNIP']
    AUT_trial_end_times_30khz = AUT_trial_df['respDurFinishNIP']
    AUT_trial_start_times_ms = AUT_trial_start_times_30khz.values/30. - NIP_start_ms
    AUT_trial_end_times_ms = AUT_trial_end_times_30khz.values/30. - NIP_start_ms

    #get responses and originality scores: (generated from https://openscoring.du.edu)
    try:
        AUT_scores_df = pd.read_csv(patient_processed_AUT_datafile)
    except:
        print('Not yet scored... generating template and breaking here!')
        # DMN_LFP_funcs.create_csv_template(AUT_trial_df,patient)
        # beh_funcs.create_csv_template(AUT_trial_df,patient)
        beh_funcs.create_csv_template(patient, data_loc, 'AUT')
        1/0

    AUT_scores_df.drop(np.where(AUT_scores_df.Start.isna())[0],inplace=True) #drop rows that have nans in the times columns (those trials didnt happen)
    if data_loc in np.array(patients_data_ripple)[:,1]:
        AUT_scores_df.drop(AUT_scores_df.index[np.where(AUT_scores_df.Stim==1)[0]],inplace=True) #drop rows that have nans in the times columns (those trials didnt happen)


    AUT_scores_df_dropna = AUT_scores_df.dropna()
    AUT_scores_df_sorted = AUT_scores_df_dropna.sort_values('originality') #sort by originality or originality_AI
    # AUT_scores_df_sorted = AUT_scores_df_dropna.sort_values('originality_AI') #sort by originality or originality_AI
    num_trials_per_third = int(len(AUT_scores_df_sorted)/3) #low will be bottom 1/3; high is top 1/3
    low_orig_df = AUT_scores_df_sorted[0:num_trials_per_third] #low creative trials
    high_orig_df = AUT_scores_df_sorted[-num_trials_per_third:] #high creative trials

    #start and end times of high & low creative sub-trials (each item presentation). 
    #These are start and end times of STIMULUS period. Doesn't include verbal response period (despite the variable/column names...)
    high_orig_response_start_times_30khz = high_orig_df.Start #start times of items/sub-trials with high creative responses
    high_orig_response_end_times_30khz = high_orig_df.End #end times of those sub-trials (end of STIMULUS period)
    low_orig_response_start_times_30khz = low_orig_df.Start #start times of items/sub-trials with low creative responses
    low_orig_response_end_times_30khz = low_orig_df.End #end times
    high_orig_response_start_times_ms = high_orig_response_start_times_30khz.values/30 - NIP_start_ms
    high_orig_response_end_times_ms = high_orig_response_end_times_30khz.values/30- NIP_start_ms
    low_orig_response_start_times_ms = low_orig_response_start_times_30khz.values/30- NIP_start_ms
    low_orig_response_end_times_ms = low_orig_response_end_times_30khz.values/30- NIP_start_ms


    low_f, high_f = DMN_LFP_funcs.get_freq_range(freq_band) #define low- and high-frequencies
    signal = raw.copy().filter(l_freq = low_f, h_freq = high_f, picks=chans) # filter signal

    #GET POWER for high and low creative subtrials
    ## only use AUT subset
    ###for stimulus period:
    if window_of_interest == 'stimulus':
        high_creative_power_AUT, high_creative_power_AUT_db = DMN_LFP_funcs.get_power_for_trials_subset(signal, high_orig_response_start_times_ms, high_orig_response_end_times_ms, good_chans_DMN, freq_band)
        low_creative_power_AUT, low_creative_power_AUT_db = DMN_LFP_funcs.get_power_for_trials_subset(signal, low_orig_response_start_times_ms, low_orig_response_end_times_ms, good_chans_DMN, freq_band)
    #####Do this for the response window (from end of stim window to 6 sec later 
    if window_of_interest == 'response':
        high_creative_power_AUT, high_creative_power_AUT_db = DMN_LFP_funcs.get_power_for_trials_subset(signal, high_orig_response_end_times_ms, high_orig_response_end_times_ms + (6*1000), good_chans_DMN, freq_band)
        low_creative_power_AUT, low_creative_power_AUT_db = DMN_LFP_funcs.get_power_for_trials_subset(signal, low_orig_response_end_times_ms, low_orig_response_end_times_ms + (6*1000), good_chans_DMN, freq_band)


    #BASELINE CORRECTION
    #baseline power (for all AUT trials)
    baseline_power_AUT, baseline_power_AUT_db = DMN_LFP_funcs.get_power_for_trials_subset(signal, AUT_trial_start_times_ms - (2 * 1000), AUT_trial_start_times_ms - (1*1000), good_chans_DMN, freq_band)

    baseline_corr_method = DMN_LFP_funcs.baseline_correct_percent_method_1
    high_orig_power_base_corr_percent = baseline_corr_method(high_creative_power_AUT, baseline_power_AUT, AUT_trial_start_times_ms, high_orig_response_start_times_ms)
    high_creative_power_baseline_corr = np.mean(high_orig_power_base_corr_percent,axis=0) #avg across trials
    low_orig_power_base_corr_percent = baseline_corr_method(low_creative_power_AUT, baseline_power_AUT, AUT_trial_start_times_ms, low_orig_response_start_times_ms)
    low_creative_power_baseline_corr = np.mean(low_orig_power_base_corr_percent,axis=0) #avg across trials

    ##Then also try not baseline correcting, since we are comparing trial types to each other
    high_creative_power_uncorr = np.mean(high_creative_power_AUT,axis=0) #avg across trials
    low_creative_power_uncorr = np.mean(low_creative_power_AUT,axis=0) #avg across trials

    


    ####TRY ALSO PLOTTING HIGH VS LOW ORIG POWER OVER TIME
    filtered_signal, power_over_time_high_orig, _ = DMN_LFP_funcs.get_power_over_time_from_unfiltered(raw, high_orig_response_start_times_ms, high_orig_response_end_times_ms + (6*1000),good_chans_DMN,freq_band)
    filtered_signal, power_over_time_low_orig, _ = DMN_LFP_funcs.get_power_over_time_from_unfiltered(raw, low_orig_response_start_times_ms, low_orig_response_end_times_ms + (6*1000),good_chans_DMN,freq_band)
    baseline_power, _ = DMN_LFP_funcs.get_power_for_trials_subset(filtered_signal, AUT_trial_start_times_ms - (2 * 1000), AUT_trial_start_times_ms - (1*1000), good_chans_DMN, freq_band)
    ##BASELINE CORRECT:
    baseline_correction_method = DMN_LFP_funcs.baseline_correct_percent_method_1
    power_base_corrected_over_time_high_orig = baseline_correction_method(power_over_time_high_orig, baseline_power, AUT_trial_start_times_ms, high_orig_response_start_times_ms)
    avg_power_base_corrected_over_time_high_orig = np.mean(power_base_corrected_over_time_high_orig,axis=(1,2)) #avg across trials and chans
    power_base_corrected_over_time_low_orig = baseline_correction_method(power_over_time_low_orig, baseline_power, AUT_trial_start_times_ms, low_orig_response_start_times_ms)
    avg_power_base_corrected_over_time_low_orig = np.mean(power_base_corrected_over_time_low_orig,axis=(1,2)) #avg across trials and chans
    power_change_over_time_all_patients_high_orig.append(avg_power_base_corrected_over_time_high_orig)
    power_change_over_time_all_patients_low_orig.append(avg_power_base_corrected_over_time_low_orig)

    high_creative_values_unavg.append(power_base_corrected_over_time_high_orig)
    low_creative_values_unavg.append(power_base_corrected_over_time_low_orig)
    

    good_chans_inds = [chans.index(g) for g in good_chans_DMN]
    regions_cut = np.array(regions_list)[good_chans_inds]



    #high - low creativity for each channel
    difference = np.array(high_creative_power_baseline_corr) - np.array(low_creative_power_baseline_corr)
    difference_uncorr = np.array(high_creative_power_uncorr) - np.array(low_creative_power_uncorr)
    values_all_patients.append(difference)
    values_all_patients_uncorr.append(difference_uncorr)
    regions_all_patients.append(regions_cut)

    high_creative_avg_values.append(high_creative_power_baseline_corr)
    low_creative_avg_values.append(low_creative_power_baseline_corr)

#sort
regions_concat = np.concatenate(regions_all_patients)
values_concat = np.concatenate(values_all_patients)


desired_order = ['Left SFG superior frontal gyrus',
'Left MSFG superior frontal gyrus medial segment',
'Left MFG middle frontal gyrus',
'Left TrIFG triangular part of the inferior frontal gyrus',
'Left OrIFG orbital part of the inferior frontal gyrus',
'Left LOrG lateral orbital gyrus', 
'Left POrG posterior orbital gyrus',
'Left AIns anterior insula',
'Left STG superior temporal gyrus',
'Left MTG middle temporal gyrus',
'Left ACgG anterior cingulate gyrus',
'Left PHG parahippocampal gyrus',
'Left Hippocampus',
'Right PHG parahippocampal gyrus',
'Right PCgG posterior cingulate gyrus',
'Right MCgG middle cingulate gyrus',
'Right ACgG anterior cingulate gyrus',
'Right AnG angular gyrus',
'Right ITG inferior temporal gyrus',
'Right MTG middle temporal gyrus',
'Right STG superior temporal gyrus',
'Right POrG posterior orbital gyrus',
'Right LOrG lateral orbital gyrus', 
'Right OpIFG opercular part of the inferior frontal gyrus',
'Right MSFG superior frontal gyrus medial segment',
'Right SFG superior frontal gyrus',
'Right MFC medial frontal cortex']


order_dict = {region: i for i, region in enumerate(desired_order)} 
sorted_regions_and_inds = sorted(enumerate(regions_concat), key = lambda x: order_dict.get(x[1],float('inf')))

regions_reorder_idx = [row[0] for row in sorted_regions_and_inds]
regions_reorder = [row[1] for row in sorted_regions_and_inds]
# regions_reorder = regions_concat[regions_reorder_idx]
values_reorder = values_concat[regions_reorder_idx]


regions_unique_random_order,idx, counts_random_order = np.unique(regions_reorder,return_index=True,return_counts=True)
regions_unique = regions_unique_random_order[np.argsort(idx)]
counts = counts_random_order[np.argsort(idx)]

tic_locs = counts/2 + np.concatenate([[0],np.cumsum(counts)])[:-1] #location of ticks, if we want just 1 label per brain region
# colors = list(map(plt.get_cmap(name='gist_rainbow'),np.linspace(0,1,16)))
colors = list(map(plt.get_cmap(name='gist_rainbow'),np.linspace(0,1,len(tic_locs))))
random.Random(4).shuffle(colors)
color_repeat = [[colors[i]]*counts[i] for i in range(len(counts))]

####CIRCULAR  PLOTS

f, ax = plt.subplots(subplot_kw={'projection': 'polar'})
x = np.arange(len(values_reorder))
theta = np.linspace(0.0, 2 * np.pi, len(x), endpoint=False)  # Convert x to radians
width = 2*np.pi / len(values_reorder)

bars = ax.bar(
    x=theta, 
    height=values_reorder,
    width=width, 
    bottom=0,
    color = list(it.chain.from_iterable(color_repeat)),
    linewidth=1,
    edgecolor ='k'
)
plt.axis('off')
# ticks = ax.set_xticks(np.interp(tic_locs, x, theta))
# Add labels
for bar, angle, height, label, color in zip(bars,theta, values_reorder, regions_reorder, list(it.chain.from_iterable(color_repeat))):
    rotation = np.rad2deg(angle)    # Rotation must be specified in degrees
    alignment = ""    # Flip some labels upside down
    if angle >= np.pi/2 and angle < 3*np.pi/2:
        alignment = "right"
        rotation = rotation + 180
    else: 
        alignment = "left"
    yloc = height + 4 if height>0 else 4
    ax.text(
        x=angle, 
        y=yloc,
        s=' '.join(label.split(' ')[:2]),
        ha=alignment, 
        va='center', 
        rotation=rotation, 
        rotation_mode="anchor",
        color=color)



###GET AVERAGE OF EACH BRAIN REGION
split_by_region = np.split(values_reorder, np.cumsum(counts)[:-1])
mean_by_region = [np.mean(section) for section in split_by_region]
f, ax = plt.subplots(subplot_kw={'projection': 'polar'})
x = np.arange(len(mean_by_region))
theta = np.linspace(0.0, 2 * np.pi, len(x), endpoint=False)  # Convert x to radians
width = 2*np.pi / len(mean_by_region)

bars = ax.bar(
    x=theta, 
    height=mean_by_region,
    width=width, 
    bottom=0,
    color = colors,
    linewidth=1,
    edgecolor ='k'
)
plt.axis('off')
# Add labels
for bar, angle, height, label, color in zip(bars,theta, mean_by_region, regions_unique, colors):
    rotation = np.rad2deg(angle)    # Rotation must be specified in degrees
    alignment = ""    # Flip some labels upside down
    if angle >= np.pi/2 and angle < 3*np.pi/2:
        alignment = "right"
        rotation = rotation + 180
    else: 
        alignment = "left"
    yloc = height + 4 if height>0 else 4
    ax.text(
        x=angle, 
        y=yloc,
        s=' '.join(label.split(' ')[:2]),
        ha=alignment, 
        va='center', 
        rotation=rotation, 
        rotation_mode="anchor",
        color=color)


#"Spider plot" (polar plot with lines)
f, ax = plt.subplots(subplot_kw={'projection': 'polar'})
#need to add final point to join ends (so last datapoint is a copy of 1st datapoint)
points = ax.plot(list(theta)+[0], mean_by_region + [mean_by_region[0]])
ax.set_xticklabels([])
# ax.set_yticklabels([])
# Add labels
for angle, height, label in zip(theta, mean_by_region, regions_unique):
    rotation = np.rad2deg(angle)    # Rotation must be specified in degrees
    alignment = ""    # Flip some labels upside down
    if angle >= np.pi/2 and angle < 3*np.pi/2:
        alignment = "right"
        rotation = rotation + 180
    else: 
        alignment = "left"
    yloc = max(mean_by_region) + 4
    ax.text(
        x=angle, 
        y=yloc,
        s=' '.join(label.split(' ')[:2]),
        ha=alignment, 
        va='center', 
        rotation=rotation, 
        rotation_mode="anchor")


power_change_over_time_avg_across_patients_high_orig = np.mean(power_change_over_time_all_patients_high_orig,axis=0)
power_change_over_time_avg_across_patients_low_orig = np.mean(power_change_over_time_all_patients_low_orig,axis=0)
f1,ax1 = plt.subplots()
ax1.plot(power_change_over_time_avg_across_patients_high_orig,color='magenta',label='high originality') #plot just AUT
ax1.plot(power_change_over_time_avg_across_patients_low_orig,color='cyan',label='low originality') #plot just AUT
ax1.axvline(x=9,color='k',linewidth = 0.5,linestyle='--')
ax1.axvline(x=0,color='k',linewidth = 0.5,linestyle='--')
trans = ax1.get_xaxis_transform()
ax1.text(0,0.1,'stimulus',rotation=90,transform=trans)
ax1.text(9,0.1,'response',rotation=90,transform=trans)
plt.legend()
sns.despine(top = True, right = True)

errors_high_orig = scipy.stats.sem(power_change_over_time_all_patients_high_orig, axis=0)
errors_low_orig = scipy.stats.sem(power_change_over_time_all_patients_low_orig, axis=0)
x = np.linspace(0,np.array(power_change_over_time_all_patients_high_orig).shape[1],np.array(power_change_over_time_all_patients_high_orig).shape[1],endpoint=False)

f1,ax1 = plt.subplots()

colors = ['magenta', 'cyan']
for t,values in enumerate([power_change_over_time_all_patients_high_orig,power_change_over_time_all_patients_low_orig]):
    errors = scipy.stats.sem(values, axis=0)


    ax1.plot(x, np.mean(values, axis=0), color = colors[t],label=['high originality', 'low originality'][t])
    ax1.fill_between(x, 
                        np.mean(values, axis=0) - errors, 
                        np.mean(values, axis=0) + errors, 
                        color=colors[t], alpha=0.25)


#compare high vs low at each time bin: 1 p value per bin
p_each_bin = []
for bin in range(np.array(power_change_over_time_all_patients_high_orig).shape[1]):
    high_orig_current_bin = np.array(power_change_over_time_all_patients_high_orig)[:,bin]
    low_orig_current_bin = np.array(power_change_over_time_all_patients_low_orig)[:,bin]

    t,p_bin = scipy.stats.ttest_rel(high_orig_current_bin,low_orig_current_bin)
    p_each_bin.append(p_bin)

#compare high vs low for each patient. 1 p value per patient
p_all_bins = []
for pt in range(np.array(power_change_over_time_all_patients_high_orig).shape[0]):
    high_orig_current_patient = np.array(power_change_over_time_all_patients_high_orig)[pt]
    low_orig_current_patient = np.array(power_change_over_time_all_patients_low_orig)[pt]

    t,p_pt = scipy.stats.ttest_ind(high_orig_current_patient,low_orig_current_patient)
    p_all_bins.append(p_pt)

#compare everything, all patients, all bins
t,p_all = scipy.stats.ttest_ind(np.concatenate(power_change_over_time_all_patients_high_orig),np.concatenate(power_change_over_time_all_patients_low_orig)) #p=0.02


#do repeated measures anova to compare high vs low creativity across time
high_creativity_power = np.array(power_change_over_time_all_patients_high_orig)
low_creativity_power = np.array(power_change_over_time_all_patients_low_orig)
patients = np.array(patients_data)[:,0]
trial_len = np.array(power_change_over_time_all_patients_high_orig).shape[1]
df = pd.DataFrame({
    'Patient': np.repeat(patients, 16 * 2),  # patients ids repeated for each time point for 2 (high + low orig) conditions
    'Timepoint': np.tile(np.tile(np.arange(trial_len), len(patients)), 2),  # 16 timepoints repeated for each subject
    'Condition': np.tile(['High'] * trial_len + ['Low'] * trial_len, len(patients)),  # Condition label
    'Power': np.concatenate([high_creativity_power.flatten(), low_creativity_power.flatten()])  # Flattened power values
})

import pingouin as pg
anova_results = pg.rm_anova(dv='Power', within=['Timepoint', 'Condition'], subject='Patient', data=df, detailed=True)
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm

# Fit a linear mixed-effects model
model = mixedlm("Power ~ Condition * Timepoint", df, groups=df["Patient"]).fit()
print(model.summary())


#compare high vs low creativity values (like the bars values)
t,p = scipy.stats.ttest_rel(np.concatenate(high_creative_avg_values),np.concatenate(low_creative_avg_values), alternative='greater') 
#0.0015 with ripple patients  (high gamma)
#0.045 exlcuding 202420 ripple and 202422 blackrock

#redo plot etc but do sem before averaging across trials
high_creativity_values_all_trials = [np.mean(high_val,axis=2) for high_val in high_creative_values_unavg]
low_creativity_values_all_trials = [np.mean(low_val,axis=2) for low_val in low_creative_values_unavg]
high_creativity_values_all_trials_all_patients = np.concatenate(high_creativity_values_all_trials,axis=-1) #concat all patients and trials
low_creativity_values_all_trials_all_patients = np.concatenate(low_creativity_values_all_trials,axis=-1)




errors_high_orig = scipy.stats.sem(high_creativity_values_all_trials_all_patients, axis=1)
errors_low_orig = scipy.stats.sem(low_creativity_values_all_trials_all_patients, axis=1)


x = np.linspace(0,np.array(high_creativity_values_all_trials_all_patients).shape[0],np.array(high_creativity_values_all_trials_all_patients).shape[0],endpoint=False)
f,ax = plt.subplots()
colors = ['magenta', 'cyan']
for t,values in enumerate([high_creativity_values_all_trials_all_patients,low_creativity_values_all_trials_all_patients]):
    errors = scipy.stats.sem(values, axis=1)


    ax.plot(x, np.mean(values, axis=1), color = colors[t],label=['high originality', 'low originality'][t])
    ax.fill_between(x, 
                        np.mean(values, axis=1) - errors, 
                        np.mean(values, axis=1) + errors, 
                        color=colors[t], alpha=0.25)


patients = np.array(patients_data)[:,0]
num_trials_per_patient = [val.shape[1] for val in high_creative_values_unavg]
trial_len = np.array(power_change_over_time_all_patients_high_orig).shape[1]
patient_idx = np.repeat(np.array(patients), [num_trials_per_patient[p] for p in range(len(patients))])
total_trials = sum(num_trials_per_patient)
df = pd.DataFrame({
    'Patient' : np.repeat(patient_idx, trial_len * 2),  # Each trial has 16 timepoints, repeated for both conditions
    'Timepoint': np.tile(np.tile(np.arange(trial_len), total_trials), 2),  # 16 timepoints repeated for each subject
    'Condition': np.tile(['High'] * trial_len + ['Low'] * trial_len, total_trials),  # Condition label
    'Power': np.concatenate([high_creativity_values_all_trials_all_patients.T.flatten(), low_creativity_values_all_trials_all_patients.T.flatten()])  # Flattened power values
})

anova_results = pg.rm_anova(dv='Power', within=['Timepoint', 'Condition'], subject='Patient', data=df, detailed=True)
# Fit a linear mixed-effects model
model = mixedlm("Power ~ Condition * Timepoint", df, groups=df["Patient"]).fit()
print(model.summary())



plt.show()
plt.close()


