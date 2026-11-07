
"""CONNECTIVITY analysis for AUT task
COHERENCE (PPL) across time

Mixed effects models and patient level analysis.

Makes Supp. Fig. 9

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy
import seaborn as sns
import glob

import sys
sys.path.append('../')

import DMN_LFP_funcs
import connectivity_funcs
import beh_funcs
import fastNFxRead

from sklearn.manifold import MDS



from sklearn.svm import SVC 
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import linear_model
from sklearn.preprocessing import KBinsDiscretizer


import cebra
from cebra import CEBRA
import cebra.models

import patient_params
patients_data = patient_params.patients_data   ###patients_info and patients_data refer to lists of patient IDs and file locs

# patient, data_loc = patients_data[-2]
window = input("Choose window ('stimulus', 'response', or 'entire'): ")
freq_band = input("Choose frequency band: ")



def normalize(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data

#####LOOK AT PPL COHERENCE OVER TIME
df_max_coh_long_all_pts = []
df_peak_diff_long_all_pts = []
originality_predictions_all_patients = []
originality_real_all_patients = []
embeddings_all = []
labels_all = []
coherence_data_all = []

from mpl_helpers import PdfPlotter
PdfPlotter('Figures/connectivity_over_time_' + freq_band + '_' + window + '.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))

include_ripple = input("Include ripple patients (y/n): ")
if include_ripple=='y':
    patients_data_ripple = patient_params.patients_data_ripple
    patients_data =  patients_data_ripple + patients_data
else:
    patients_data_ripple = []
patient_ids = []

for patient, data_loc in patients_data:
    print(patient)
    if patient=='UIC202420': #excluded patient
        continue
    if patient=='UIC202521': #patient for OCT results
        continue
    if patient=='UIC202522': #patient for OCT patient
        continue
    if '20250103-115359' in data_loc: #UIC202422 blackrock session (use ripple session for analysis)
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


    #look at originality scores
    patient_processed_AUT_datafile = 'scored_responses/' + patient + '_processed_data.csv'  #file that includes responses and originality scores
    AUT_scores_df = pd.read_csv(patient_processed_AUT_datafile)
    originality_by_trial = AUT_scores_df.originality
    originality_by_trial = originality_by_trial.values
    # originality_by_trial = normalize(originality_by_trial) ###TRY NORMALIZING CREATIVIITY VALUES FOR EACH PATIENT
    

    AUT_signal, good_chans_all, good_regions, good_networks = DMN_LFP_funcs.trial_type_signal_ripple(patient,data_loc, raw = raw, window = window, trial_type = 'AUT', pre_trial_buffer=0, NIP_start_ms = NIP_start_ms)

    AUT_signal_DMN = AUT_signal[:,:,np.where(good_networks=='DMN')[0]]
    AUT_signal_DAN = AUT_signal[:,:,np.where(good_networks=='DorsalAttention')[0]]
    AUT_signal_FPN = AUT_signal[:,:,np.where(good_networks=='FrontoParietal')[0]]

    network_subset_DMN = good_networks[np.where(good_networks=='DMN')[0]]
    network_subset_DAN = good_networks[np.where(good_networks=='DorsalAttention')[0]]
    network_subset_FPN = good_networks[np.where(good_networks=='FrontoParietal')[0]]

    AUT_signal_network_subset = np.concatenate([AUT_signal_DMN, AUT_signal_DAN, AUT_signal_FPN],axis=2)
    network_subset = np.concatenate([network_subset_DMN, network_subset_DAN, network_subset_FPN])

    #sliding PPL (PPL over time, in sliding window)
    # res = connectivity_funcs.get_PPL_sliding_window(AUT_signal_network_subset, network_subset, freq_band, raw.info['sfreq'], sliding_win_width = 2000, step = 200)
    res = connectivity_funcs.get_PPL_sliding_window_fast(AUT_signal_network_subset, network_subset, freq_band, raw.info['sfreq'], sliding_win_width = 2000, step = 200)


    network_ids = pd.Series(network_subset) #so that can label contacts as networks

    fixed = res.explode().reset_index(name='value')
    fixed['elem_index'] = fixed.groupby(['contact_a', 'contact_b', 'trial']).cumcount()
    fixed = fixed.set_index(['contact_a', 'contact_b', 'trial', 'elem_index'])
    fixed_unstack = fixed.unstack('elem_index')



    DMN_range = np.where(network_ids=='DMN')[0]
    DAN_range = np.where(network_ids=='DorsalAttention')[0]
    FPN_range = np.where(network_ids=='FrontoParietal')[0]

    DMN_a = fixed[fixed.index.get_level_values('contact_a').isin(DMN_range)] #contact a is DMN 
    DMN_DAN_coherence = DMN_a[DMN_a.index.get_level_values('contact_b').isin(DAN_range)] #contact b is DAN 
    DMN_FPN_coherence = DMN_a[DMN_a.index.get_level_values('contact_b').isin(FPN_range)] #contact b is FPN

    DAN_a = fixed[fixed.index.get_level_values('contact_a').isin(DAN_range)] #contact a is DAN 
    DAN_FPN_coherence = DAN_a[DAN_a.index.get_level_values('contact_b').isin(FPN_range)] #contact b is FPN

    FPN_a = fixed[fixed.index.get_level_values('contact_a').isin(FPN_range)] #contact a is DAN 


    fixed_unstack_DMN_FPN = DMN_FPN_coherence.unstack('elem_index')
    avg_across_contact_pairs_DMN_FPN = fixed_unstack_DMN_FPN.unstack('trial').mean().unstack('elem_index') #for each trial, avg across contact pairs to get avg coherence over time

    fixed_unstack_DMN_DAN = DMN_DAN_coherence.unstack('elem_index')
    avg_across_contact_pairs_DMN_DAN = fixed_unstack_DMN_DAN.unstack('trial').mean().unstack('elem_index') #for each trial, avg across contact pairs to get avg coherence over time



    DMN_coherence = DMN_a[DMN_a.index.get_level_values('contact_b').isin(DMN_range)] #contact b is also DMN 
    DAN_coherence = DAN_a[DAN_a.index.get_level_values('contact_b').isin(DAN_range)] #contact b is also DAN 
    FPN_coherence = FPN_a[FPN_a.index.get_level_values('contact_b').isin(FPN_range)] #contact b is also DAN 
    fixed_unstack_DMN = DMN_coherence.unstack('elem_index')
    avg_across_contact_pairs_DMN = fixed_unstack_DMN.unstack('trial').mean().unstack('elem_index') #for each trial, avg across contact pairs to get avg coherence over time
    fixed_unstack_DAN = DAN_coherence.unstack('elem_index')
    avg_across_contact_pairs_DAN = fixed_unstack_DAN.unstack('trial').mean().unstack('elem_index') #for each trial, avg across contact pairs to get avg coherence over time
    fixed_unstack_FPN = FPN_coherence.unstack('elem_index')
    avg_across_contact_pairs_FPN = fixed_unstack_FPN.unstack('trial').mean().unstack('elem_index') #for each trial, avg across contact pairs to get avg coherence over time

    # for tr in res.index.get_level_values('trial').unique(): #look over trials, plot PPL over time (avg across each pair of chans), new plot of each trial
    #     f,ax = plt.subplots()
    #     ax.plot(avg_across_contact_pairs_DMN.values[tr], color='orange')
    #     ax.plot(avg_across_contact_pairs_FPN.values[tr], color='blue')
    #     ax.plot(avg_across_contact_pairs_DMN_FPN.values[tr], color='black')

    #     ax.set_title(originality_by_trial[tr])


    ### compare max coherence with originality, for diff networks
    y_FPN = [max(a) for a in avg_across_contact_pairs_FPN.values]
    y_DMN = [max(a) for a in avg_across_contact_pairs_DMN.values]
    y_DMN_FPN = [max(a) for a in avg_across_contact_pairs_DMN_FPN.values]
    df_max_coh = pd.DataFrame([originality_by_trial, y_FPN, y_DMN, y_DMN_FPN], index = ['originality', 'FPN', 'DMN', 'DMN-FPN']).T
    df_max_coh.index.names = ['trial']

    #### "melt" so that becomes long df that can be used for sns lmplot
    df_max_coh_long = df_max_coh.reset_index().melt(id_vars=['trial', 'originality'], value_vars=['FPN', 'DMN', 'DMN-FPN'], var_name='network', value_name='max_coherence')

    # g = sns.lmplot(data=df_max_coh_long, x='originality', y='max_coherence', hue='network', palette=['blue','orange', 'pink'])
    # plt.title('max coherence (' + freq_band + '; ' + patient + ')')


    df_max_coh_long_all_pts.append(df_max_coh_long)
    patient_ids.append(patient)

    i_FPN = [np.argmax(a) for a in avg_across_contact_pairs_FPN.values]
    i_DMN = [np.argmax(a) for a in avg_across_contact_pairs_DMN.values]
    diff = np.array(i_FPN) - np.array(i_DMN)
    df_peak_diff = pd.DataFrame([originality_by_trial, diff], index = ['originality', 'diff_peak_coherence']).T
    df_peak_diff.index.names = ['trial']
    df_peak_diff_long = df_peak_diff.reset_index().melt(id_vars=['trial', 'originality'])

    g = sns.lmplot(data=df_peak_diff_long, x='originality', y='value')
    plt.title('diff time of peak coherence FPN-DMN (' + freq_band + '; ' + patient + ')')
    
    df_peak_diff_long_all_pts.append(df_peak_diff_long)



g = sns.lmplot(data=pd.concat(df_max_coh_long_all_pts), x='originality', y='max_coherence', hue='network', palette=['blue','orange', 'pink'])
plt.title('max coherence (' + freq_band + ')')
plt.show()


g = sns.lmplot(data=pd.concat(df_peak_diff_long_all_pts), x='originality', y='value')
plt.title('diff time of peak coherence FPN-DMN (' + freq_band + ')')


for pt_data in df_peak_diff_long_all_pts:
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(pt_data.dropna().originality, pt_data.dropna().value)
    print('slope: ' + str(slope) + '\n' + 'p: ' + str(p_value))


#do regressions by network for max coherence vs originality

#make dfs with pt ids
dfs_with_ids = []
for patient_id, d in zip(patient_ids, df_max_coh_long_all_pts):
    tmp = d.copy()
    tmp["patient_id"] = patient_id
    dfs_with_ids.append(tmp)
df_all = pd.concat(dfs_with_ids, ignore_index=True)


#do linear regression by network; pool across subjects
import statsmodels.formula.api as smf
df = pd.concat(df_max_coh_long_all_pts).copy()
rows = []

for network, d in df.groupby("network"):
    model = smf.ols("max_coherence ~ originality", data=d).fit()
    rows.append({
        "network": network,
        "n": int(model.nobs),
        "intercept": model.params["Intercept"],
        "slope_originality": model.params["originality"],
        "p_originality": model.pvalues["originality"],
        "r_squared": model.rsquared,
        "ci_low": model.conf_int().loc["originality", 0],
        "ci_high": model.conf_int().loc["originality", 1],
    })

regression_results = pd.DataFrame(rows)


#mixed effects model b/c data w/in subject not independent
df = df_all.dropna(subset=["originality", "max_coherence"]).copy()

rows = []

for network, d in df.groupby("network"):
    model = smf.mixedlm(
        "max_coherence ~ originality",
        data=d,
        groups=d["patient_id"]
    ).fit()
    
    rows.append({
        "network": network,
        "n": d.shape[0],
        "n_patients": d["patient_id"].nunique(),
        "intercept": model.params["Intercept"],
        "slope_originality": model.params["originality"],
        "p_originality": model.pvalues["originality"],
        "ci_low": model.conf_int().loc["originality", 0],
        "ci_high": model.conf_int().loc["originality", 1],
    })

mixed_results = pd.DataFrame(rows)


#do patient level analysis to see how coherence depends on mean originality across subjects rather than across trials
patient_level = (
    df_all
    .dropna(subset=["originality", "max_coherence"])
    .groupby(["patient_id", "network"], as_index=False)
    .agg(
        mean_originality=("originality", "mean"),
        mean_max_coherence=("max_coherence", "mean")
    )
)

rows = []
for network, d in patient_level.groupby("network"):
    model = smf.ols("mean_max_coherence ~ mean_originality", data=d).fit()
    
    rows.append({
        "network": network,
        "n_patients": d["patient_id"].nunique(),
        "intercept": model.params["Intercept"],
        "slope_mean_originality": model.params["mean_originality"],
        "p_mean_originality": model.pvalues["mean_originality"],
        "r_squared": model.rsquared,
        "ci_low": model.conf_int().loc["mean_originality", 0],
        "ci_high": model.conf_int().loc["mean_originality", 1],
    })

patient_level_results = pd.DataFrame(rows)

label_to_color = {'DMN': 'orange', 'DMN-FPN': 'purple', 'FPN': 'blue'}

#plot this (coherence vs subject originality)
g = sns.lmplot(
    data=patient_level,
    x="mean_originality",
    y="mean_max_coherence",
    hue="network",
    col="network",
    palette = label_to_color,
    col_wrap=3,
    height=4,
    facet_kws={"sharex": True, "sharey": False}
)

for ax, network in zip(g.axes.flat, patient_level["network"].unique()):
    d = patient_level[patient_level["network"] == network]
    for _, row in d.iterrows():
        ax.text(
            row["mean_originality"],
            row["mean_max_coherence"],
            row["patient_id"],
            fontsize=8
        )

plt.show()
plt.close()