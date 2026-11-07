"""network dynamics with option to include Ripple patients 
LFP analysis for AUT task
NETWORK DYNAMICS AVERAGED ACROSS PATIENTS
How does gamma vs theta evolve over the timecourse of a trial, for different trial types. Excludes micros
    AUT vs BAT power dynamics for theta & gamma
    Stats, including cluster-based permutation tests
        compare AUT and BAT
        compare BAT with baseline

    Makes Figure 3a
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



import patient_params
patients_info = patient_params.patients_data
patients_data = DMN_LFP_funcs.choose_patient(patients_info) #choose all or 1 patient

###DEFINE WHICH FREQUENCIES (E.G. THETA, GAMMA), GROUPS OF CHANNELS (ALL CHANS, DMN, CHOSEN ROIS), AND ROIS TO CONSIDER 
freqs_list=['high_gamma', 'theta'] 
trial_types = ['AUT', 'BAT']

ROIs_list = [] ###no ROIs chosen. #alt: ROIs_list = ['Hipp', 'parahipp', 'SPL', 'ACgG', 'MFC', 'MFG', 'GRe', 'SFG'] 
###create empty df, which will be population for each patient, per freq, for group of channels
index = pd.MultiIndex.from_product([np.array(patients_data)[:,0], freqs_list, ['power_all', 'power_DMN', 'power_ROIs'], ['AUT', 'BAT', 'OCT']], names=['person', 'frequency', 'chan_group', 'trial_type'])

window_size=1 #window size for power calculation (passed to compute_psd)
pre_trial_buffer=2000 #adding a bit before prompt appears

percent_power_change_all_patients_df = pd.DataFrame(index=index, columns=range(int((16 + pre_trial_buffer//1000)/window_size)))  # Assuming 16 timepoints

network = input("Choose cortical network (DMN, FrontoParietal, SomatoMotor, Limbic, Visual, DorsalAttention, VentralAttention): ")
include_ripple = input("Include ripple patients (y/n): ")
if include_ripple=='y':
    patients_info_ripple = patient_params.patients_data_ripple
    patients_data_ripple = patients_info_ripple
    patients_data = patients_data_ripple + patients_data


for patient, data_loc in patients_data:
    print(patient)

    if patient=='UIC202420': #excluded patient (interruptions)
        continue
    if patient=='UIC202521': #patient for OCT analysis
        continue
    if patient=='UIC202522': #patient for OCT analysis
        continue
    if '20250103-115359' in data_loc: #use the ripple sesion for this patient
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

    # good_chans_DMN is actually good chan subset for chosen network, not necessarily DMN
    good_chans_all, good_chans_DMN = DMN_LFP_funcs.get_chan_info(raw, chan_info, network, remove_micros=remove_micros, remove_white=True) 
    if len(good_chans_DMN)==0:
        continue
    
    # good_chans_all, good_chans_ROIs = DMN_LFP_funcs.get_chan_info(raw, chan_info, "PCgG", remove_micros=remove_micros, remove_white=True) 


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
        # trial_subset_df = DMN_LFP_funcs.trial_type_subset_df(trial_info_df, trial_type, practice_prompts=[['key'], ['10 + 1 = ']]) 
        start_times_ms, end_times_ms, trial_start_times_ms = beh_funcs.get_start_end_times_ms(trial_info_df, trial_type, pre_trial_buffer=pre_trial_buffer, data_loc=data_loc)


        start_times_ms = start_times_ms - NIP_start_ms
        end_times_ms = end_times_ms - NIP_start_ms
        trial_start_times_ms = trial_start_times_ms - NIP_start_ms


        for i, freq_band in enumerate(freqs_list):
            for j, which_chans in enumerate([[], good_chans_DMN, []]):
                if len(which_chans)==0:
                    continue
                #filter the sig into the desired freq band; get avg baseline power for each trial; then baseline correct
                filtered_signal, power_over_time, power_over_time_db = DMN_LFP_funcs.get_power_over_time_from_unfiltered(raw, start_times_ms, end_times_ms + (6*1000),which_chans,freq_band, window_size=window_size)
                baseline_power, baseline_power_db = DMN_LFP_funcs.get_power_for_trials_subset(filtered_signal, trial_start_times_ms - (2 * 1000), trial_start_times_ms - (1*1000), which_chans, freq_band)
                avg_uncorrected_signal = np.mean(power_over_time,axis=(1,2))

                ##BASELINE CORRECT:
                baseline_correction_method = DMN_LFP_funcs.baseline_correct_percent_method_1
                power_base_corrected_over_time = baseline_correction_method(power_over_time, baseline_power, trial_start_times_ms, start_times_ms, pre_trial_buffer = pre_trial_buffer)
                avg_power_base_corrected_over_time = np.mean(power_base_corrected_over_time,axis=(1,2)) #avg across trials and chans

                percent_power_change_all_patients_df.loc[(patient,freq_band,['power_all', 'power_DMN', 'power_ROIs'][j], trial_type)] = avg_power_base_corrected_over_time


    

#avg across patients:
# percent_power_change_avg = percent_power_all_patients_df.groupby(level=['frequency', 'chan_group', 'trial_type']).mean()-100 #-100 if want it to be percent CHANGE from baseline
percent_power_change_avg = percent_power_change_all_patients_df.groupby(level=['frequency', 'chan_group', 'trial_type']).mean() # no longer -100 because function now does percent change
error_patients = percent_power_change_all_patients_df.groupby(level=['frequency', 'chan_group', 'trial_type']).sem() # no longer -100 because function now does percent change


#plot 2 figs: 1. just AUT power dynamics; 2. ALl 3 trial types.
#make and use annotate function, so that can apply same annotations to both figs.
# from mpl_helpers import PdfPlotter
# PdfPlotter('Figures/all_patients_network_dynamics_across_trialtypes_' + network + 'ripple_.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))
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
            ax[i][j].axvline(x=9,color='k',linewidth = 0.5, linestyle='--') #annotations, for instance
            ax[i][j].axvline(x=0,color='k',linewidth = 0.5, linestyle='--')
            trans = ax[i][j].get_xaxis_transform()
            ax[0][j].text(0,0.1,'stimulus',rotation=90,transform=trans)
            ax[0][j].text(9,0.1,'response',rotation=90,transform=trans)
            # ax[0][j].set_title(['all chans','DMN network chans', 'DMN ROIs'][j],fontsize=10, loc='left')
            left = ax[0][j].get_position().bounds[0]
            f.text(left+0.1,0.9,['all chans','DMN network chans', 'DMN ROIs'][j],ha='center', va='center',fontsize=10)
            ax[-1][j].set_xlabel('time (s)')
            sns.despine(top = True, right = True)
        ax[i][0].set_title(freq_band + ' power dynamics',fontsize=14, loc='left')
    f.text(0.07,0.5,'% power change',rotation=90, ha='center', va='center',fontsize=10)

    f.subplots_adjust(hspace=0.5)

# annotate function, for plots with only 1 group of chans plotted
def annotate2(f, ax, freqs_list):
    for i, freq_band in enumerate(freqs_list):
            ax[i].axvline(x=9,color='k',linewidth = 0.5, linestyle='--') 
            ax[i].axvline(x=0,color='k',linewidth = 0.5, linestyle='--')
            trans = ax[i].get_xaxis_transform()
            ax[0].text(0,0.1,'stimulus',rotation=90,transform=trans)
            ax[0].text(9,0.1,'response',rotation=90,transform=trans)
            # ax[0][j].set_title(['all chans','DMN network chans', 'DMN ROIs'][j],fontsize=10, loc='left')
            ax[i].set_title(freq_band,fontsize=14, loc='left')
            ax[i].set_xlabel('time (s)')
            ax[0].set_ylabel('% power change',rotation=90, ha='center', va='center',fontsize=10)
            sns.despine(top = True, right = True)

    f.subplots_adjust(hspace=0.5)

annotate1(f1, ax1, freqs_list, chans_subsets)
annotate1(f2, ax2, freqs_list, chans_subsets)





chans_subsets = ['power_DMN'] 
freqs_list = ['high_gamma', 'theta']
f1,ax1 = plt.subplots(1, len(freqs_list))
f2,ax2 = plt.subplots(1, len(freqs_list))
colors = ['darkorange','cornflowerblue','limegreen']
x = np.linspace(0-pre_trial_buffer/1000,percent_power_change_avg.shape[-1]*window_size-pre_trial_buffer/1000,percent_power_change_avg.shape[-1],endpoint=False)
for i, freq_band in enumerate(freqs_list):
        
    values = percent_power_change_avg.loc[freq_band].loc['power_DMN']
    errors = error_patients.loc[freq_band].loc['power_DMN']
    
    for t,trial_type in enumerate(trial_types):
        values_numeric = pd.to_numeric(values.iloc[t], errors='coerce') #because getting isfinite error... needs go be float not object
        errors_numeric = pd.to_numeric(errors.iloc[t], errors='coerce')

        ax2[i].plot(x, values_numeric, color = colors[t],label=trial_type)
        ax2[i].fill_between(x, 
                            values_numeric - errors_numeric, 
                            values_numeric + errors_numeric, 
                            color=colors[t], alpha=0.25)
ax2[-1].legend(trial_types)

annotate2(f2, ax2, freqs_list)


plt.show()
plt.close()



################################## stats ##################################
import statsmodels.stats.multitest as multi

df_gamma_AUT = percent_power_change_all_patients_df.loc[:, 'high_gamma', 'power_DMN', 'AUT'].dropna()
df_gamma_BAT = percent_power_change_all_patients_df.loc[:, 'high_gamma', 'power_DMN', 'BAT'].dropna()   

p_gamma_over_time_AUT_BAT = scipy.stats.ttest_rel(df_gamma_AUT, df_gamma_BAT)[1]
rej, corr_p = multi.fdrcorrection(p_gamma_over_time_AUT_BAT, alpha=0.05, method='indep')

scipy.stats.ttest_rel(np.concatenate(df_gamma_AUT.iloc[:, 5:].values), np.concatenate(df_gamma_BAT.iloc[:, 5:].values)) #t =8.76, pvalue=3.653e-16, df=239)

#average for participant first, then t test,  stim + response windows
scipy.stats.ttest_rel(df_gamma_AUT.iloc[:, 5:].T.mean().values, df_gamma_BAT.iloc[:, 5:].T.mean()) #t=3.0147, pvalue=0.00928, df=14)
scipy.stats.ttest_rel(df_gamma_AUT.iloc[:, 15:].T.mean().values, df_gamma_BAT.iloc[:, 15:].T.mean()) #statistic=3.6081189726985126, pvalue=0.002852166453187442, df=14)

#average for participant first, then t test. Just stimulus window.
scipy.stats.ttest_rel(df_gamma_AUT.iloc[:, 5:15].T.mean().values, df_gamma_BAT.iloc[:, 5:15].T.mean().values) #t=2.2777, pvalue=0.03896, df=14)

scipy.stats.ttest_1samp(list(np.concatenate(df_gamma_BAT.values)), 0) #t = 5.85 p=1.1924e-08
scipy.stats.ttest_1samp(list(np.concatenate(df_gamma_AUT.values)), 0) #t = 5.85 p=1.1924e-08


df_gamma_BAT_num = df_gamma_BAT.apply(pd.to_numeric, errors="coerce")
tvals, pvals = scipy.stats.ttest_1samp(df_gamma_BAT_num, popmean=0, axis=0, nan_policy="omit", alternative='greater')
rej, corr_p_BAT_bins = multi.fdrcorrection(pvals, alpha=0.05, method='indep')

df_gamma_AUT_num = df_gamma_AUT.apply(pd.to_numeric, errors="coerce")
tvals, pvals = scipy.stats.ttest_1samp(df_gamma_AUT_num, popmean=0, axis=0, nan_policy="omit", alternative='greater')
rej, corr_p_AUT_bins = multi.fdrcorrection(pvals, alpha=0.05, method='indep')

df_theta_AUT = percent_power_change_all_patients_df.loc[:, 'theta', 'power_DMN', 'AUT'].dropna()
df_theta_BAT = percent_power_change_all_patients_df.loc[:, 'theta', 'power_DMN', 'BAT'].dropna()   

p_theta_over_time_AUT_BAT = scipy.stats.ttest_rel(df_theta_AUT, df_theta_BAT)[1]
rej, corr_p = multi.fdrcorrection(p_theta_over_time_AUT_BAT, alpha=0.05, method='indep')


#average for participant first, then t test,  stim + response windows
scipy.stats.ttest_rel(df_theta_AUT.iloc[:, 5:].T.mean().values, df_theta_BAT.iloc[:, 5:].T.mean()) #TtestResult(statistic=-3.75892448184811, pvalue=0.0021157249145460025, df=14)

#average for participant first, then t test. Just stimulus window.
scipy.stats.ttest_rel(df_theta_AUT.iloc[:, 5:15].T.mean().values, df_theta_BAT.iloc[:, 5:15].T.mean()) # TtestResult(statistic=-3.7860395905052053, pvalue=0.002005343005175262, df=14)



p_values = [0.0194, 0.78501, 0.23695, 0.5008, 0.07188, 6.9838e-05, 0.00835]
import statsmodels.stats.multitest as multi
rej, corr_p = multi.fdrcorrection(p_values, alpha=0.05, method='indep')




#more stats: CLUSTER-BASED PERMUTATION TESTS

from scipy import stats

def get_patient_time_matrix(df, freq, chan_group, trial_type):
    """
    Returns:
        data: patients x timepoints array
        patients: patient labels
    """
    sub = df.xs(
        (freq, chan_group, trial_type),
        level=("frequency", "chan_group", "trial_type")
    )

    sub = sub.dropna(how="all")
    data = sub.to_numpy(dtype=float)
    patients = sub.index.to_numpy()

    return data, patients

def get_paired_difference_matrix(df, time_sec, freq, chan_group, trial_type_a, trial_type_b):
    data_a, patients_a = get_patient_time_matrix(df, freq, chan_group, trial_type_a)
    data_b, patients_b = get_patient_time_matrix(df, freq, chan_group, trial_type_b)

    df_a = pd.DataFrame(data_a, index=patients_a, columns=time_sec)
    df_b = pd.DataFrame(data_b, index=patients_b, columns=time_sec)

    common_patients = df_a.index.intersection(df_b.index)

    diff = df_a.loc[common_patients] - df_b.loc[common_patients]

    return diff.to_numpy(dtype=float), common_patients


def find_clusters(t_vals, threshold):
    """
    Find contiguous clusters where abs(t) exceeds threshold.
    Returns list of arrays of time indices.
    """
    supra = np.abs(t_vals) > threshold

    clusters = []
    current = []

    for i, is_supra in enumerate(supra):
        if is_supra:
            current.append(i)
        else:
            if len(current) > 0:
                clusters.append(np.array(current))
                current = []

    if len(current) > 0:
        clusters.append(np.array(current))

    return clusters


def cluster_permutation_1samp(data, n_perm=10000, cluster_alpha=0.05, seed=0):
    """
    One-sample cluster-based permutation test against 0.

    Parameters
    ----------
    data : array, shape n_subjects x n_times
        Baseline-corrected patient-level data.
    n_perm : int
        Number of sign-flip permutations.
    cluster_alpha : float
        Cluster-forming alpha, uncorrected.
    seed : int
        Random seed.

    Returns
    -------
    result : dict
        Observed t-values, clusters, cluster masses, corrected p-values.
    """

    rng = np.random.default_rng(seed)

    data = np.asarray(data, dtype=float)

    # Drop subjects with any NaNs in the tested time range
    data = data[~np.isnan(data).any(axis=1)]

    n_subj, n_times = data.shape

    # Observed t-values
    t_obs, _ = stats.ttest_1samp(data, popmean=0.0, axis=0)

    # Cluster-forming threshold
    df = n_subj - 1
    threshold = stats.t.ppf(1 - cluster_alpha / 2, df)

    clusters = find_clusters(t_obs, threshold)
    cluster_masses = np.array([
        np.sum(np.abs(t_obs[c])) for c in clusters
    ])

    # Permutation null: max cluster mass per permutation
    max_masses = np.zeros(n_perm)

    for p in range(n_perm):
        signs = rng.choice([-1, 1], size=(n_subj, 1))
        perm_data = data * signs

        t_perm, _ = stats.ttest_1samp(perm_data, popmean=0.0, axis=0)
        perm_clusters = find_clusters(t_perm, threshold)

        if len(perm_clusters) > 0:
            max_masses[p] = np.max([
                np.sum(np.abs(t_perm[c])) for c in perm_clusters
            ])
        else:
            max_masses[p] = 0

    # Corrected p-values
    cluster_pvals = np.array([
        (np.sum(max_masses >= mass) + 1) / (n_perm + 1)
        for mass in cluster_masses
    ])

    return {
        "t_obs": t_obs,
        "threshold": threshold,
        "clusters": clusters,
        "cluster_masses": cluster_masses,
        "cluster_pvals": cluster_pvals,
        "n_subjects": n_subj,
    }




data, patients = get_patient_time_matrix(
    percent_power_change_all_patients_df,
    freq="high_gamma",
    chan_group="power_DMN",
    trial_type="AUT"
)

n_time = data.shape[1]

time_sec = (
    np.arange(n_time) * window_size
    - pre_trial_buffer / 1000
)

window_mask = (time_sec >= (10)) & (time_sec < (15))

window_data = data[:, window_mask]
window_time = time_sec[window_mask]

cluster_res = cluster_permutation_1samp(
    window_data,
    n_perm=10000,
    cluster_alpha=0.05,
    seed=1
)

for cluster, p in zip(cluster_res["clusters"], cluster_res["cluster_pvals"]):
    print(
        f"Cluster {window_time[cluster[0]]} to {window_time[cluster[-1]] + window_size} s, "
        f"p = {p:.4f}"
    )


for cluster, p in zip(cluster_res["clusters"], cluster_res["cluster_pvals"]):
    start = window_time[cluster[0]]
    end = window_time[cluster[-1]] + window_size
    cluster_mean = np.nanmean(window_data[:, cluster])

    direction = "suppression" if cluster_mean < 0 else "increase"

    print(
        f"{direction}: {start:.1f} to {end:.1f} s, "
        f"mean = {cluster_mean:.2f}%, "
        f"p = {p:.4f}"
    )

#BAT vs baseline (stimulus)
#suppression: 7.0 to 10.0 s, mean = -13.93%, p = 0.0157 ((stimulus))
#suppression: 10.0 to 11.0 s, mean = -12.27%, p = 0.0316 (response)

# suppression: 0.0 to 0.1 s, mean = -17.49%, p = 0.2392
# suppression: 0.2 to 1.8 s, mean = -21.69%, p = 0.0124
# suppression: 1.9 to 2.0 s, mean = -16.68%, p = 0.2838
# suppression: 2.8 to 3.0 s, mean = -16.52%, p = 0.1867
# suppression: 3.1 to 3.8 s, mean = -18.08%, p = 0.0659
# suppression: 3.9 to 4.9 s, mean = -18.72%, p = 0.0367
# suppression: 5.0 to 5.5 s, mean = -17.88%, p = 0.0836
# suppression: 5.6 to 6.1 s, mean = -19.39%, p = 0.0876
# suppression: 6.5 to 6.8 s, mean = -19.55%, p = 0.1134
# suppression: 6.9 to 10.0 s, mean = -26.24%, p = 0.0002

#AUT vs baseline 
# increase: 0.0 to 2.0 s, mean = 17.53%, p = 0.0770 (stimulus)
# increase: 11.0 to 15.0 s, mean = 27.41%, p = 0.0050 (response)





diff_data, common_patients = get_paired_difference_matrix(
    percent_power_change_all_patients_df,
    time_sec,
    freq="high_gamma",
    chan_group="power_DMN",
    trial_type_a="AUT",
    trial_type_b="BAT"
)

window_mask = (time_sec >= 0) & (time_sec < 10)

diff_window = diff_data[:, window_mask]
diff_window_time = time_sec[window_mask]

cluster_res_diff = cluster_permutation_1samp(
    diff_window,
    n_perm=10000,
    cluster_alpha=0.05,
    seed=2
)

for cluster, p in zip(cluster_res_diff["clusters"], cluster_res_diff["cluster_pvals"]):
    cluster_mean = np.nanmean(diff_window[:, cluster])

    print(
        f"AUT - BAT cluster {diff_window_time[cluster[0]]} to "
        f"{diff_window_time[cluster[-1]] + window_size} s, "
        f"mean difference = {cluster_mean:.2f}%, "
        f"cluster p = {p:.4f}"
    )

#AUT - BAT cluster 0.0 to 2.0 s, mean difference = 23.98%, cluster p = 0.0474
# AUT - BAT cluster 7.0 to 10.0 s, mean difference = 17.52%, cluster p = 0.0344
#AUT - BAT cluster 11.0 to 15.0 s, mean difference = 32.93%, cluster p = 0.0113

# AUT - BAT cluster 0.2 to 0.8 s, mean difference = 23.82%, cluster p = 0.0540
# AUT - BAT cluster 1.0 to 1.8 s, mean difference = 17.96%, cluster p = 0.0378
# AUT - BAT cluster 4.4 to 4.5 s, mean difference = 10.22%, cluster p = 0.4291
# AUT - BAT cluster 7.2 to 8.5 s, mean difference = 16.75%, cluster p = 0.0182
# AUT - BAT cluster 8.6 to 8.8 s, mean difference = 14.96%, cluster p = 0.2181
# AUT - BAT cluster 8.9 to 9.0 s, mean difference = 16.27%, cluster p = 0.3435
# AUT - BAT cluster 9.2 to 9.4 s, mean difference = 14.00%, cluster p = 0.2461
# AUT - BAT cluster 9.8 to 9.9 s, mean difference = 22.13%, cluster p = 0.4651

# AUT - BAT cluster 11.2 to 11.8 s, mean difference = 17.12%, cluster p = 0.0557
# AUT - BAT cluster 12.1 to 12.2 s, mean difference = 18.40%, cluster p = 0.3050
# AUT - BAT cluster 12.5 to 14.5 s, mean difference = 33.04%, cluster p = 0.0030
# AUT - BAT cluster 14.7 to 15.0 s, mean difference = 46.10%, cluster p = 0.1112
