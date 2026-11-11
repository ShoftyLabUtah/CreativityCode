'''SINGLE UNIT ANALYSIS for DMN AUT task

AUT vs BAT vs OCT pie chart (Figure 5e)
'''





import numpy as np
from neo import io
import quantities as pq
import glob   
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
import sys
sys.path.append('../')
import beh_funcs
import DMN_LFP_funcs
import mat73
import scipy
import statsmodels.stats.multitest as multi
import statsmodels.api as sm
from scipy.stats import f_oneway

from sklearn.metrics import mutual_info_score

from mpl_helpers import PdfPlotter


import patient_params
patients_info = patient_params.patients_data_spikes
patients_data = DMN_LFP_funcs.choose_patient(patients_info) #choose all or 1 patient
patients_data = patients_data[:6]

#take timestamps and figure out what they are relative to the time of trial start
    #for all electrodes (concatenated)
def func_get_spike_times_rel_to_events(timestamps_all_units, start_times_sec, end_times_sec):
    timestamps_all_units_concat = np.concatenate(timestamps_all_units)
    spike_times_rel_to_trial_start_all_events = []
    spike_times_rel_to_trial_start_split_by_trial = []
    for [trial_start, trial_end] in zip(start_times_sec,end_times_sec):
        spike_times_currrent_trial = timestamps_all_units_concat[np.where(np.logical_and(timestamps_all_units_concat > trial_start, timestamps_all_units_concat< trial_end))[0]]
        spike_times_rel_to_trial_start = spike_times_currrent_trial - trial_start
        spike_times_rel_to_trial_start_all_events.extend(spike_times_rel_to_trial_start)
        spike_times_rel_to_trial_start_split_by_trial.append(spike_times_rel_to_trial_start)
    spike_times_rel_to_trial_start_all_events_sorted = np.sort(spike_times_rel_to_trial_start_all_events)
    return [spike_times_rel_to_trial_start_split_by_trial, spike_times_rel_to_trial_start_all_events_sorted]


#same as func_get_spike_times_rel_to_events but split by individual electrodes
def func_get_spike_times_rel_to_events_split_by_unit(timestamps_all_units, start_times_sec, end_times_sec):
    spike_times_rel_to_trial_start_all_events_sorted_split_by_units = []
    for timestamps_current_unit in timestamps_all_units:
        spike_times_rel_to_trial_start_all_events_current_unit = []
        for [trial_start, trial_end] in zip(start_times_sec,end_times_sec):
            spike_times_currrent_trial = np.array(timestamps_current_unit)[np.where(np.logical_and(timestamps_current_unit > trial_start, timestamps_current_unit< trial_end))[0]]
            spike_times_rel_to_trial_start = spike_times_currrent_trial - trial_start
            spike_times_rel_to_trial_start_all_events_current_unit.extend(spike_times_rel_to_trial_start)
        spike_times_rel_to_trial_start_all_events_current_unit_sorted = np.sort(spike_times_rel_to_trial_start_all_events_current_unit)
        #now append, so get all channels, but in separate lists
        spike_times_rel_to_trial_start_all_events_sorted_split_by_units.append(spike_times_rel_to_trial_start_all_events_current_unit_sorted)
    return spike_times_rel_to_trial_start_all_events_sorted_split_by_units

#same as func_get_spike_times_rel_to_events_split_by_unit, but keep trials separarte
def func_get_spike_times_per_trial_split_by_unit(timestamps_all_units, start_times_sec, end_times_sec):
    spike_times_per_trial_split_by_units = []
    counter = 0
    for timestamps_current_unit in timestamps_all_units:
        spike_times_rel_to_trial_start_all_events_current_unit = []
        for [trial_start, trial_end] in zip(start_times_sec,end_times_sec):
            spike_times_currrent_trial = np.array(timestamps_current_unit)[np.where(np.logical_and(timestamps_current_unit > trial_start, timestamps_current_unit< trial_end))[0]]
            spike_times_rel_to_trial_start = spike_times_currrent_trial - trial_start
            spike_times_rel_to_trial_start_all_events_current_unit.append(spike_times_rel_to_trial_start)
        spike_times_per_trial_split_by_units.append(spike_times_rel_to_trial_start_all_events_current_unit)
        counter+=1
    return spike_times_per_trial_split_by_units



all_networks = []
all_regions = []


total_cell_num_all_patients = []

num_AUT_suppressed_all_pts = []
num_AUT_enhanced_all_pts = []
num_BAT_suppressed_all_pts = []
num_BAT_enhanced_all_pts = []
num_OCT_suppressed_all_pts = []
num_OCT_enhanced_all_pts = []

all_cell_results = []

for patient, data_loc, filename in patients_data:
    print(patient)



    '''####################    LOADING THE NEV FILE AND GETTING THE UNITS!     ##################'''

    NEV_file = glob.glob(filename)[0] #data file with spike info

    #load nev file
    print('loading NEV file')
    r = io.BlackrockIO(NEV_file)
    # block = r.read_block()
    block = r.read_block(load_waveforms=True) #load waveforms so can plot them

    #get data: data for each unit
    data_units = block.segments[0].data_children[3:]
    #get units information (255 means rejected)
    units = [note.annotations['unit_id'] for note in data_units]
    good_units_inds = np.where(np.array(units)!=255)[0] #channels with non-rejected units
    good_units = [data_units[c] for c in good_units_inds]
    good_channel_ids = [c.annotations['channel_id'] for c in good_units]
    good_unit_ids = [c.annotations['id'] for c in good_units]

    timestamps_all_units = [] #timestamps of all waveforms, for the good channels
    for j in range(len(good_units)):
        timestamps_current_unit = [good_units[j].data[i] for i in range(len(good_units[j].data))]
        timestamps_all_units.append(timestamps_current_unit)


    trial_info_df = beh_funcs.get_trial_info_df(data_loc)


    ###BY TRIAL TYPE
    spikes_all_trialtypes = {'AUT':[],'BAT':[], 'OCT': []}
    spikes_all_trialtypes_per_trial = {'AUT':[],'BAT':[], 'OCT':[]}
    for trial_type in ['AUT','BAT','OCT']:
        start_times_ms, end_times_ms, trial_start_times_ms = beh_funcs.get_start_end_times_ms(trial_info_df, trial_type, data_loc=data_loc)
        start_times_sec = start_times_ms/1000
        end_times_sec = end_times_ms/1000 + 6

        spike_times_rel_to_trial_start_all_events_sorted_split_by_units = func_get_spike_times_rel_to_events_split_by_unit(timestamps_all_units, start_times_sec, end_times_sec)
        spikes_all_trialtypes[trial_type] = spike_times_rel_to_trial_start_all_events_sorted_split_by_units

        spike_times_per_trial_split_by_units = func_get_spike_times_per_trial_split_by_unit(timestamps_all_units, start_times_sec, end_times_sec)
        spikes_all_trialtypes_per_trial[trial_type] = spike_times_per_trial_split_by_units

    matfile = DMN_LFP_funcs.get_matfile(patient, data_loc)
    if patient=='UIC202504': #this patient uses channelmap2 for macros but channelmap1 for micros
        matfile = glob.glob('/Volumes/Data/UIC202504/Imaging/Registered/ChannelMap.mat')

    remove_micros = False
    chan_info = mat73.loadmat(matfile[0])
    regions = pd.DataFrame(chan_info['ElecAtlasProj'])[0]
    regions_list = [reg[0] if reg is not None else 'None' for reg in regions]

    #if white matter, replace with second most likely region, since the micros aren't in white matter
    regions_prob = pd.DataFrame(chan_info['ElecAtlasProbProj'])[0]
    white_inds = np.where(['White' in r for r in np.array(regions_list)])[0]
    regions_fixed = [reg[0] if reg is not None else 'None' for reg in regions]
    for r in white_inds:
        if type(regions_prob.loc[r][0][1])==str:
            continue
        if regions_prob.loc[r][0][1]<0.6:
            regions_fixed[r] = regions_prob.loc[r][1][0]
    regions_list = regions_fixed
    regions = pd.Series(regions_fixed)

    net_column_number = list(pd.DataFrame(chan_info['ElecAtlasProj']).columns)[-1] #because different number of atlases for diff patients
    networks = pd.DataFrame(chan_info['ElecAtlasProj'])[net_column_number]
    networks_list = [net[0] if net is not None else 'None' for net in networks]
    networks_prob = pd.DataFrame(chan_info['ElecAtlasProbProj'])[net_column_number]
    unknown_inds = np.where(['Unknown' in n for n in np.array(networks_list)])[0]
    networks_fixed = [net[0] if net is not None else 'None' for net in networks]
    for n in unknown_inds:
        if type(networks_prob.loc[n][0][1])==str:  ##?
            continue
        if networks_prob.loc[n][0][1]<0.9: #likelihood of classified as "unknown"<0.6
            networks_fixed[n] = networks_prob.loc[n][1][0]
    networks_list = networks_fixed
    networks = pd.Series(networks_fixed)

    good_chan_inds = [chan - 1 for chan in good_channel_ids]
    good_regions = regions[good_chan_inds]
    # good_networks = [n[0] for n in networks[good_chan_inds]]
    good_networks = networks[good_chan_inds]
    all_networks.append(good_networks.values)  ###Get all the networks in which these units reside
    all_regions.append(good_regions.values)  ###Get all the regions in which these units reside


    
    ####FOR PIE CHART
    
    bins = np.linspace(0, 16.1, 51)
    p_vals_all_trials = []
    p_vals_AUT_BAT = []
    p_vals_AUT_OCT = []
    p_vals_bins_anova = []
    preference = []

    preference_results = {'label': [], 'direction': []}

    spikes_all_trialtypes = {'AUT':[],'BAT':[], 'OCT': []}
    spikes_all_trialtypes_per_trial = {'AUT':[],'BAT':[], 'OCT':[]}
    for trial_type in ['AUT','BAT','OCT']:
        start_times_ms, end_times_ms, trial_start_times_ms = beh_funcs.get_start_end_times_ms(trial_info_df, trial_type, data_loc=data_loc)
        start_times_sec = start_times_ms/1000
        end_times_sec = end_times_ms/1000 + 6

        spike_times_rel_to_trial_start_all_events_sorted_split_by_units = func_get_spike_times_rel_to_events_split_by_unit(timestamps_all_units, start_times_sec, end_times_sec)
        spikes_all_trialtypes[trial_type] = spike_times_rel_to_trial_start_all_events_sorted_split_by_units

        spike_times_per_trial_split_by_units = func_get_spike_times_per_trial_split_by_unit(timestamps_all_units, start_times_sec, end_times_sec)
        spikes_all_trialtypes_per_trial[trial_type] = spike_times_per_trial_split_by_units


    cell_results_patient = []
    for i in range(len(good_channel_ids)):
        #compare firing rates, paired ttest across bins
        AUT_counts_hist,_ = np.histogram(spikes_all_trialtypes['AUT'][i],bins=bins)
        BAT_counts_hist,_ = np.histogram(spikes_all_trialtypes['BAT'][i],bins=bins)
        OCT_counts_hist,_ = np.histogram(spikes_all_trialtypes['OCT'][i],bins=bins)


        F, p = f_oneway(AUT_counts_hist,
                        BAT_counts_hist,
                        OCT_counts_hist)

        pairs = {
            "AUT vs BAT": scipy.stats.ttest_rel(AUT_counts_hist, BAT_counts_hist).pvalue,
            "AUT vs OCT": scipy.stats.ttest_rel(AUT_counts_hist, OCT_counts_hist).pvalue,
            "BAT vs OCT": scipy.stats.ttest_rel(BAT_counts_hist, OCT_counts_hist).pvalue
        }

        pvals = list(pairs.values())
        
        means = {
            "AUT": AUT_counts_hist.mean(),
            "BAT": BAT_counts_hist.mean(),
            "OCT": OCT_counts_hist.mean()
        }

        preference.append(max(means, key=means.get))

        # p_vals_all_trials.append(p_trials)
        p_vals_bins_anova.append(p)
        p_vals_AUT_BAT.append(pairs['AUT vs BAT'])
        p_vals_AUT_OCT.append(pairs['AUT vs OCT'])


        sig = lambda p: p < 0.1 #looser cutoff because use anova p value to decide if sig selectivity. this is just to choose preference

        AB = pairs['AUT vs BAT']
        AC = pairs['AUT vs OCT']
        BC = pairs['BAT vs OCT']

        if sig(AB) and sig(AC) and not sig(BC):
            label = "AUT"
            direction = "suppressed" if means['AUT'] < np.mean([means['BAT'], means['OCT']]) else "enhanced"

        elif sig(AB) and sig(BC) and not sig(AC):
            label = "BAT"
            direction = "suppressed" if means['BAT'] < np.mean([means['AUT'], means['OCT']]) else "enhanced"

        elif sig(AC) and sig(BC) and not sig(AB):
            label = "OCT"
            direction = "suppressed" if means['OCT'] < np.mean([means['BAT'], means['AUT']]) else "enhanced"

        elif sig(AB) and sig(AC) and sig(BC):
            label =  max(means, key=means.get)
            direction = "enhanced"  # it is the condition with the greatest mean bin count


        else:
            # import pdb; pdb.set_trace()
            label = "ambiguous"

        preference_results['label'].append(label)
        preference_results['direction'].append(direction)
        

        cell_results_patient.append({
            'patient': patient,
            'cell_id': f'{patient}_ch{good_channel_ids[i]}_unit{good_unit_ids[i]}',
            'channel_id': good_channel_ids[i],
            'unit_id': good_unit_ids[i],
            'region': good_regions.iloc[i],
            'network': good_networks.iloc[i],
            'anova_p': p,
            'aut_vs_bat_p': AB,
            'aut_vs_oct_p': AC,
            'bat_vs_oct_p': BC,
            'aut_mean_bin_count': means['AUT'],
            'bat_mean_bin_count': means['BAT'],
            'oct_mean_bin_count': means['OCT'],
            'preferred_condition': label,
            'direction': direction
        })

    rej, corr_p_bins = multi.fdrcorrection(p_vals_bins_anova, alpha=0.05, method='indep')  #makes sure only consider cells that show sig diff between trial types


    cell_results_patient = pd.DataFrame(cell_results_patient)
    cell_results_patient['anova_p_fdr'] = corr_p_bins
    cell_results_patient['passes_anova_fdr'] = corr_p_bins <= 0.05

    # This column is exactly what determines the pie wedges.
    cell_results_patient['pie_category'] = 'none'

    for condition, short_name in [('AUT', 'aut'), ('BAT', 'bat'), ('OCT', 'oct')]:
        for direction_name, suffix in [('suppressed', 'sup'), ('enhanced', 'enh')]:
            inds = (
                cell_results_patient['passes_anova_fdr']
                & (cell_results_patient['preferred_condition'] == condition)
                & (cell_results_patient['direction'] == direction_name)
            )
            cell_results_patient.loc[inds, 'pie_category'] = f'{short_name}_{suffix}'

    all_cell_results.append(cell_results_patient)

    prefs = np.array(preference_results['label'])[corr_p_bins<=0.05]
    dirs = np.array(preference_results['direction'])[corr_p_bins<=0.05]
    
    AUT_inds = np.where(prefs=='AUT')
    num_AUT_suppressed = len(np.where(dirs[np.where(prefs=='AUT')[0]]=='suppressed')[0])
    num_AUT_enhanced = len(np.where(dirs[np.where(prefs=='AUT')[0]]=='enhanced')[0])
    num_BAT_suppressed = len(np.where(dirs[np.where(prefs=='BAT')[0]]=='suppressed')[0])
    num_BAT_enhanced = len(np.where(dirs[np.where(prefs=='BAT')[0]]=='enhanced')[0])
    num_OCT_suppressed = len(np.where(dirs[np.where(prefs=='OCT')[0]]=='suppressed')[0])
    num_OCT_enhanced = len(np.where(dirs[np.where(prefs=='OCT')[0]]=='enhanced')[0])

    num_AUT_suppressed_all_pts.append(num_AUT_suppressed)
    num_AUT_enhanced_all_pts.append(num_AUT_enhanced)
    num_BAT_suppressed_all_pts.append(num_BAT_suppressed)
    num_BAT_enhanced_all_pts.append(num_BAT_enhanced)
    num_OCT_suppressed_all_pts.append(num_OCT_suppressed)
    num_OCT_enhanced_all_pts.append(num_OCT_enhanced)

    total_cell_num_all_patients.append(len(good_channel_ids))


####Pie chart

PdfPlotter('Figures/proportions_unit_trial_preference_.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))

num_types = {'aut_sup': sum(num_AUT_suppressed_all_pts), 
                'aut_enh': sum(num_AUT_enhanced_all_pts),
                'bat_sup': sum(num_BAT_suppressed_all_pts), 
                'bat_enh': sum(num_BAT_enhanced_all_pts),
                'oct_sup': sum(num_OCT_suppressed_all_pts), 
                'oct_enh': sum(num_OCT_enhanced_all_pts),
                'none': sum(total_cell_num_all_patients) - 
                    (sum(num_AUT_suppressed_all_pts)+
                    sum(num_AUT_enhanced_all_pts)+
                    sum(num_BAT_suppressed_all_pts)+
                    sum(num_BAT_enhanced_all_pts)+
                    sum(num_OCT_suppressed_all_pts)+
                    sum(num_OCT_enhanced_all_pts))}


f, ax = plt.subplots()
wedges1, texts1, autotexts1 = ax.pie(num_types.values(), labels=num_types.keys(), autopct='%1.1f%%', colors=['tab:orange', 'darkorange', 'tab:blue', 'darkblue', 'tab:green', 'darkgreen','grey'], startangle=0, rotatelabels=True )
plt.show()
plt.close()



###save source data:

pie_source_data = pd.concat(all_cell_results, ignore_index=True)

pie_source_data.to_csv(
    'Figure5e_unit_trial_preference.csv',
    index=False
)


#get pie from source data:
pie_order = [
    'aut_sup', 'aut_enh',
    'bat_sup', 'bat_enh',
    'oct_sup', 'oct_enh',
    'none'
]

num_types = (
    pie_source_data['pie_category']
    .value_counts()
    .reindex(pie_order, fill_value=0)
)