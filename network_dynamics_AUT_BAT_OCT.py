"""LFP analysis for AUT task
NETWORK DYNAMICS AVERAGED ACROSS PATIENTS
How does gamma vs theta evolve over the timecourse of a trial, for different trial types. Excludes micros
    AUT vs OCT power dynamics for theta & gamma (option to include BAT trials too)
    Stats including cluster-based permutation tests.

    Makes Figure 3b
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

import brpylib

import DMN_LFP_funcs
import beh_funcs

import patient_params
patients_info = patient_params.patients_data
patients_data = DMN_LFP_funcs.choose_patient(patients_info) #choose all or 1 patient

patients_data = patients_info[-2:] + patients_info[0:8] #just the patients who did OCT

###DEFINE WHICH FREQUENCIES (E.G. THETA, GAMMA), GROUPS OF CHANNELS (ALL CHANS, DMN, CHOSEN ROIS), AND ROIS TO CONSIDER 
freqs_list=['high_gamma', 'theta'] 
trial_types = ['AUT', 'OCT'] #use this one if want to only compare AUT and OCT
# trial_types = ['AUT', 'BAT', 'OCT'] #Use this one if want to compare all three trial types in 1 plot
ROIs_list = [] ###no ROIs chosen. #alt: ROIs_list = ['Hipp', 'parahipp', 'SPL', 'ACgG', 'MFC', 'MFG', 'GRe', 'SFG'] 
###create empty df, which will be population for each patient, per freq, for group of channels
index = pd.MultiIndex.from_product([np.array(patients_data)[:,0], freqs_list, ['power_all', 'power_DMN', 'power_ROIs'], ['AUT', 'BAT', 'OCT']], names=['person', 'frequency', 'chan_group', 'trial_type'])

window_size=1 #window size for power calculation (passed to compute_psd)
pre_trial_buffer=2000 #adding a bit before prompt appears

percent_power_change_all_patients_df = pd.DataFrame(index=index, columns=range(int((16 + pre_trial_buffer//1000)/window_size)))  # Assuming 16 timepoints

network = input("Choose cortical network (DMN, FrontoParietal, SomatoMotor, Limbic, Visual, DorsalAttention, VentralAttention): ")


rows = []

for patient, data_loc in patients_data:
    print(patient)
    #get mne file
    raw = DMN_LFP_funcs.get_raw(patient,data_loc) #get raw mne (with already marked bad channels that were selected manually)
    chans = raw.ch_names
    # fs = raw.info['sfreq'] #sampling freq

    # #match electrodes to regions
    matfile = DMN_LFP_funcs.get_matfile(patient, data_loc)
    remove_micros = DMN_LFP_funcs.remove_micros_bool(data_loc)

    chan_info = mat73.loadmat(matfile[0])
    regions = pd.DataFrame(chan_info['ElecAtlasProj'])[0]
    regions_list = [reg[0] if reg is not None else 'None' for reg in regions]
    # networks = pd.DataFrame(chan_info['ElecAtlasProj'])[pd.DataFrame(chan_info['ElecAtlasProj']).columns.size-1]
    # networks_list = [net[0] if net is not None else 'None' for net in networks]

    # good_chans_DMN is actually good chan subset for chosen network, not necessarily DMN
    good_chans_all, good_chans_DMN = DMN_LFP_funcs.get_chan_info(raw, chan_info, network, remove_micros=remove_micros, remove_white=True, patient=patient) 
    
    # if patient == 'UIC202521': #networks disappeared from channelmap, so use this workaround from saved all-dat
    #     all_dat = pd.read_csv('pkl_files/store_neural_data_list_all_trialtypes_' + network + '.csv', header=[0, 1], index_col=[0, 1, 2, 3])
    #     patient_dfs = [group for _, group in all_dat.groupby(level='patient')]
    #     cleaned_dfs = [df.dropna(axis=1, how='all') for df in patient_dfs]

    #     chans_fullnames = cleaned_dfs[-1].T.index.get_level_values('channel').unique()
    #     good_chans_DMN = [c.split(" ")[0] for c in chans_fullnames]
    
    if len(good_chans_DMN)==0:
        continue
    

    good_chans_ROIs=[]

    raw.notch_filter(freqs=[60,120],picks = good_chans_all) #filter out electrical noise
    raw.set_eeg_reference(ref_channels = 'average') #re-reference. bads automatically excluded

    # get trial info from matlab file
    # trial_info_df = DMN_LFP_funcs.get_trial_info_df(data_loc)
    trial_info_df = beh_funcs.get_trial_info_df(data_loc)

    for trial_type in trial_types:    ## compare trial types
        # ##Find where practice trials end and real session starts. Practice AUT and OCT trials start with promot "key"; BAT trials start with '10 + 1 = '
        # trial_subset_df = DMN_LFP_funcs.trial_type_subset_df(trial_info_df, trial_type, practice_prompts=[['key'], ['10 + 1 = ']]) 
        start_times_ms, end_times_ms, trial_start_times_ms = beh_funcs.get_start_end_times_ms(trial_info_df, trial_type, pre_trial_buffer=pre_trial_buffer, data_loc = data_loc)

        for i, freq_band in enumerate(freqs_list):
            for j, which_chans in enumerate([[], good_chans_DMN, good_chans_ROIs]):
                if len(which_chans)==0:
                    continue
                #filter the sig into the desired freq band; get avg baseline power for each trial; then baseline correct
                filtered_signal, power_over_time, power_over_time_db = DMN_LFP_funcs.get_power_over_time_from_unfiltered(raw, start_times_ms, end_times_ms + (6*1000),which_chans,freq_band, window_size=window_size)
                baseline_power, baseline_power_db = DMN_LFP_funcs.get_power_for_trials_subset(filtered_signal, trial_start_times_ms - (2 * 1000), trial_start_times_ms - (1*1000), which_chans, freq_band)
                avg_uncorrected_signal = np.mean(power_over_time,axis=(1,2))

                ##BASELINE CORRECT:
                baseline_correction_method = DMN_LFP_funcs.baseline_correct_percent_method_1
                power_base_corrected_over_time = baseline_correction_method(power_over_time, baseline_power, trial_start_times_ms, start_times_ms)
                avg_power_base_corrected_over_time = np.mean(power_base_corrected_over_time,axis=(1,2)) #avg across trials and chans
                percent_power_change_all_patients_df.loc[(patient,freq_band,['power_all', 'power_DMN', 'power_ROIs'][j], trial_type)] = avg_power_base_corrected_over_time


                #for mixed effects model
                trial_level_power = np.nanmean(power_base_corrected_over_time,axis=(-1))

                n_time = trial_level_power.shape[0] #num timebins
                time_sec = (np.arange(n_time) * window_size
                    - pre_trial_buffer / 1000)

                for trial_idx in range(trial_level_power.shape[1]): #loop over trials
                    for time_idx, t in enumerate(time_sec):
                        rows.append({
                            "patient": patient,
                            "trial": trial_idx,
                            "trial_type": trial_type,
                            "frequency": freq_band,
                            "time_sec": t,
                            "power": trial_level_power[time_idx, trial_idx],
                        })

lmm_df = pd.DataFrame(rows)

################################## stats ##################################
import statsmodels.stats.multitest as multi

df_gamma_AUT = percent_power_change_all_patients_df.loc[:, 'high_gamma', 'power_DMN', 'AUT'].dropna()
df_gamma_OCT = percent_power_change_all_patients_df.loc[:, 'high_gamma', 'power_DMN', 'OCT'].dropna()
df_gamma_BAT = percent_power_change_all_patients_df.loc[:, 'high_gamma', 'power_DMN', 'BAT'].dropna()   

p_gamma_over_time_AUT_OCT = scipy.stats.ttest_rel(df_gamma_AUT, df_gamma_OCT)[1]
rej, corr_p = multi.fdrcorrection(p_gamma_over_time_AUT_OCT, alpha=0.05, method='indep')

#average for participant first, then t test,  stim + response windows
scipy.stats.ttest_rel(df_gamma_AUT.iloc[:, 2:].T.mean().values, df_gamma_OCT.iloc[:, 2:].T.mean()) 

#average for participant first, then t test. Just stimulus window.
scipy.stats.ttest_rel(df_gamma_AUT.iloc[:, 2:12].T.mean().values, df_gamma_OCT.iloc[:, 2:12].T.mean()) 

#average for participant first, then t test. Just response window.
scipy.stats.ttest_rel(df_gamma_AUT.iloc[:,12:].T.mean().values, df_gamma_OCT.iloc[:,12:].T.mean()) 

scipy.stats.ttest_rel(df_gamma_AUT.iloc[:,2:].mean().values, df_gamma_OCT.iloc[:,2:].mean()) #entire trial

scipy.stats.ttest_rel(np.concatenate(df_gamma_AUT.iloc[:, 2:].values), np.concatenate(df_gamma_OCT.iloc[:, 2:].values)) #DMN: p=0.0194. FPN: p=0.78501. DAN: p=0.23695. SMN: p=0.5008. Limbic: 0.07188. Visual (OCT>AUT): p=6.9838e-05. VAN: p=0.00835
print(scipy.stats.ttest_rel(np.concatenate(df_gamma_AUT.iloc[:, 2:].values), np.concatenate(df_gamma_OCT.iloc[:, 2:].values)))
#DMN: TtestResult(statistic=2.847987748305343, pvalue=0.004981359886603283, df=159)
#FPN: TtestResult(statistic=1.8204372473834975, pvalue=0.0705727756876544, df=159)
#SMN: TtestResult(statistic=0.5870819868150319, pvalue=0.5581908832490585, df=127)
#Limbic: TtestResult(statistic=-1.1336207468208053, pvalue=0.2590890719239137, df=127)
#Visual: TtestResult(statistic=-3.538960085235268, pvalue=0.0005879197149385501, df=111) #OCT > AUT
#DAN: TtestResult(statistic=-0.9706801457152443, pvalue=0.33467019768646356, df=79)
#VAN: TtestResult(statistic=2.745283607406576, pvalue=0.007054042925150098, df=111)

df_theta_AUT = percent_power_change_all_patients_df.loc[:, 'theta', 'power_DMN', 'AUT'].dropna()
df_theta_OCT = percent_power_change_all_patients_df.loc[:, 'theta', 'power_DMN', 'OCT'].dropna()
# df_theta_BAT = percent_power_change_all_patients_df.loc[:, 'theta', 'power_DMN', 'BAT'].dropna()   


scipy.stats.ttest_rel(df_theta_AUT.iloc[:, 2:].T.mean().values, df_theta_OCT.iloc[:, 2:].T.mean().values) 

#average across participant first, then t test. Just stimulus window.
scipy.stats.ttest_rel(df_theta_AUT.iloc[:, 2:12].T.mean().values, df_theta_OCT.iloc[:, 2:12].T.mean().values) 

#average across participant first, then t test. Just response window.
scipy.stats.ttest_rel(df_theta_AUT.iloc[:,12:].T.mean().values, df_theta_OCT.iloc[:,12:].T.mean().values) 



p_theta_over_time_AUT_OCT = scipy.stats.ttest_rel(df_theta_AUT, df_theta_OCT)[1]
rej, corr_p = multi.fdrcorrection(p_theta_over_time_AUT_OCT, alpha=0.05, method='indep')

# p_theta_over_time_AUT_BAT = scipy.stats.ttest_rel(df_theta_AUT, df_theta_BAT)[1]
scipy.stats.ttest_rel(np.concatenate(df_theta_AUT.iloc[:, 2:].values), np.concatenate(df_theta_OCT.iloc[:, 2:].values)) #DMN: p=2.47849e-08. FPN: p=4.06226e-11. DAN: p=0.0001169. SMN: p=0.0165. Limbic: 0.02426. Visual: p=0.00253
# scipy.stats.ttest_rel(np.concatenate(df_theta_AUT.values), np.concatenate(df_theta_BAT.values)) #DMN: p=4.32735e-16. FPN: p=7.69778e-18. DAN: p=8.11419e-07. SMN: p=6.8294e-12. Limbic: 1.21911e-10. Visual: p=0.34788


p_values = [0.0194, 0.78501, 0.23695, 0.5008, 0.07188, 6.9838e-05, 0.00835]
rej, corr_p = multi.fdrcorrection(p_values, alpha=0.05, method='indep')

##CORRECTED P VALUES (HIGH GAMMA OCT VS AUT)
#DMN: 0.0452666667, FPN: 0.78501, DAN: 0.33173, SMN: 0.584266667, Limbic: 0.12579, Vis: 0.000488866, VAN: 0.029225

    ################################## plots ##################################


#avg across patients:
# percent_power_change_avg = percent_power_all_patients_df.groupby(level=['frequency', 'chan_group', 'trial_type']).mean()-100 #-100 if want it to be percent CHANGE from baseline
percent_power_change_avg = percent_power_change_all_patients_df.groupby(level=['frequency', 'chan_group', 'trial_type']).mean() # no longer -100 because function now does percent change
error_patients = percent_power_change_all_patients_df.groupby(level=['frequency', 'chan_group', 'trial_type']).sem() # no longer -100 because function now does percent change

std_patients = percent_power_change_all_patients_df.groupby(level=['frequency', 'chan_group', 'trial_type']).std() # no longer -100 because function now does percent change


#plot 2 figs: 1. just AUT power dynamics; 2. ALl 3 trial types.
#make and use annotate function, so that can apply same annotations to both figs.
from mpl_helpers import PdfPlotter
PdfPlotter('Figures/all_patients_network_dynamics_' + network + '_AUT_OCT.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))
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
    # ax[0][0].set_ylabel('% power change')
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
    
    for t,trial_type in enumerate(['AUT', 'BAT', 'OCT']):
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

std_patients = percent_power_change_all_patients_df.groupby(level=['frequency', 'chan_group', 'trial_type']).std()
std_patients.loc['high_gamma', 'power_DMN']
stim_period_std = std_patients.loc['high_gamma', 'power_DMN'].iloc[:, np.where((x<10) & (x>0))[0]]
resp_period_std = std_patients.loc['high_gamma', 'power_DMN'].iloc[:, np.where(x>10)[0]]





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

diff_data, common_patients = get_paired_difference_matrix(
    percent_power_change_all_patients_df,
    time_sec,
    freq="high_gamma",
    chan_group="power_DMN",
    trial_type_a="AUT",
    trial_type_b="OCT"
)

window_mask = (time_sec >= 0) & (time_sec < 15)

diff_window = diff_data[:, window_mask]
diff_window_time = time_sec[window_mask]

cluster_res_diff = cluster_permutation_1samp(
    diff_window,
    n_perm=10000,
    cluster_alpha=0.05,
    seed=42
)

for cluster, p in zip(cluster_res_diff["clusters"], cluster_res_diff["cluster_pvals"]):
    cluster_mean = np.nanmean(diff_window[:, cluster])

    print(
        f"AUT - OCT cluster {diff_window_time[cluster[0]]} to "
        f"{diff_window_time[cluster[-1]] + window_size} s, "
        f"mean difference = {cluster_mean:.2f}%, "
        f"cluster p = {p:.4f}"
    )



###mixed effects model
import statsmodels.formula.api as smf

def run_lmm_condition_at_each_time(
    lmm_df,
    freq="high_gamma",
    condition_a="AUT",
    condition_b="OCT"
):
    sub = lmm_df[
        (lmm_df["frequency"] == freq) &
        (lmm_df["trial_type"].isin([condition_a, condition_b]))
    ].copy()

    # Set reference category
    sub["trial_type"] = pd.Categorical(
        sub["trial_type"],
        categories=[condition_a, condition_b]
    )

    results = []

    for t in sorted(sub["time_sec"].unique()):
        d = sub[sub["time_sec"] == t].dropna(subset=["power"])
        if d["patient"].nunique() < 3:
            continue

        try:
            model = smf.mixedlm(
                "power ~ trial_type",
                data=d,
                groups=d["patient"]
            )

            fit = model.fit(reml=False)

            coef_name = f"trial_type[T.{condition_b}]"

            results.append({
                "time_sec": t,
                "comparison": f"{condition_b} - {condition_a}",
                "beta": fit.params[coef_name],
                "se": fit.bse[coef_name],
                "z": fit.tvalues[coef_name],
                "p": fit.pvalues[coef_name],
                "n_patients": d["patient"].nunique(),
                "n_trials": len(d),
            })

        except Exception as e:
            results.append({
                "time_sec": t,
                "comparison": f"{condition_b} - {condition_a}",
                "beta": np.nan,
                "se": np.nan,
                "z": np.nan,
                "p": np.nan,
                "n_patients": d["patient"].nunique(),
                "n_trials": len(d),
                "error": str(e),
            })

    return pd.DataFrame(results)


oct_vs_aut_lmm = run_lmm_condition_at_each_time(
    lmm_df,
    freq="high_gamma",
    condition_a="AUT",
    condition_b="OCT"
)


rej, corr_p = multi.fdrcorrection(oct_vs_aut_lmm.p, alpha=0.05, method='indep')


import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

def run_window_lmm(
    lmm_df,
    freq="high_gamma",
    condition_a="AUT",
    condition_b="OCT",
    windows=None
):
    if windows is None:
        windows = {
            "pre_stim": (-1, 0),
            "early_stim": (0, 5),
            "late_stim": (5, 10),
            "response": (10, 16),
            "full_trial": (0, 16),
        }

    sub = lmm_df[
        (lmm_df["frequency"] == freq) &
        (lmm_df["trial_type"].isin([condition_a, condition_b]))
    ].copy()

    sub["trial_type"] = pd.Categorical(
        sub["trial_type"],
        categories=[condition_a, condition_b]
    )

    results = []

    for win_name, (tmin, tmax) in windows.items():
        d = sub[(sub["time_sec"] >= tmin) & (sub["time_sec"] < tmax)].copy()

        # Average within trial/window first so each trial contributes one value
        # d_win = (
        #     d.groupby(["patient", "trial_type", "trial"], as_index=False)["power"]
        #     .mean()
        # )

        d_win = (
            d.groupby(["patient", "trial_type", "trial"], as_index=False)["power"]
            .mean()
            .dropna(subset=["power", "patient", "trial_type"])
            .reset_index(drop=True)
        )
        model = smf.mixedlm(
            "power ~ trial_type",
            data=d_win,
            groups=d_win["patient"]
        )

        fit = model.fit(reml=False)

        coef_name = f"trial_type[T.{condition_b}]"

        results.append({
            "window": win_name,
            "comparison": f"{condition_b} - {condition_a}",
            "beta": fit.params[coef_name],
            "se": fit.bse[coef_name],
            "z": fit.tvalues[coef_name],
            "p": fit.pvalues[coef_name],
            "n_patients": d_win["patient"].nunique(),
            "n_trials": len(d_win),
        })

    results = pd.DataFrame(results)
    results["p_fdr"] = multipletests(results["p"], method="fdr_bh")[1]
    results["sig_fdr"] = results["p_fdr"] < 0.05

    return results

oct_vs_aut_lmm = run_window_lmm(
    lmm_df,
    freq="high_gamma",
    condition_a="AUT",
    condition_b="OCT", 
    windows=None
)
