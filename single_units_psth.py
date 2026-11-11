'''SINGLE UNIT ANALYSIS for DMN AUT task

PSTH (Figure 5a-d)
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


from sklearn.metrics import mutual_info_score

from mpl_helpers import PdfPlotter


import patient_params
patients_info = patient_params.patients_data_spikes
patients_data = DMN_LFP_funcs.choose_patient(patients_info) #choose all or 1 patient

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


source_data_rows = []

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
    spikes_all_trialtypes = {'AUT':[],'BAT':[]}
    spikes_all_trialtypes_per_trial = {'AUT':[],'BAT':[]}
    for trial_type in ['AUT','BAT']:
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


    #save source data: 
    for i, (channel_id, unit_id) in enumerate(zip(good_channel_ids, good_unit_ids)):
        cell_id = f'{patient}_ch{channel_id}_unit{unit_id}'

        for trial_type in ['AUT', 'BAT']:
            unit_trials = spikes_all_trialtypes_per_trial[trial_type][i]

            for trial_num, spike_times in enumerate(unit_trials):
                spike_times = np.asarray(spike_times)

                # Keep silent trials in the CSV as one row with no spike time
                if len(spike_times) == 0:
                    source_data_rows.append({
                        'patient': patient,
                        'cell_id': cell_id,
                        'unit_id': unit_id,
                        'channel_id': channel_id,
                        'region': good_regions.iloc[i],
                        'network': good_networks.iloc[i],
                        'trial_type': trial_type,
                        'trial_num_within_type': trial_num,
                        'spike_time_sec': np.nan
                    })

                # One row per observed spike
                else:
                    for spike_time in spike_times:
                        source_data_rows.append({
                            'patient': patient,
                            'cell_id': cell_id,
                            'unit_id': unit_id,
                            'channel_id': channel_id,
                            'region': good_regions.iloc[i],
                            'network': good_networks.iloc[i],
                            'trial_type': trial_type,
                            'trial_num_within_type': trial_num,
                            'spike_time_sec': spike_time
                        })
                        
    
    ####PLOT PSTH
    PdfPlotter('Figures/spike_psth_' + patient + '.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))

    bins = np.linspace(0, 16.1, 51)
    p_vals_all_trials = []
    p_vals_all_bins = []
    aut_vs_bat_preference = []
    for i in range(len(good_channel_ids)):
        f, ax = plt.subplots(2,1)

        sns.histplot(data = spikes_all_trialtypes['AUT'][i], kde=True,kde_kws={'bw_method':0.1},bins=bins,ax = ax[0],label='AUT', color = 'darkorange')
        sns.histplot(data = spikes_all_trialtypes['BAT'][i], kde=True,kde_kws={'bw_method':0.1},bins=bins, ax = ax[0], label='BAT', color = 'cornflowerblue')

        ax[0].axvline(x=10, color = 'k', linestyle = ':', linewidth = 0.5)
        ax[0].legend()

        ##rasterplot
        ax[1].eventplot(spikes_all_trialtypes_per_trial['AUT'][i],color="orange")
        ax[1].eventplot(spikes_all_trialtypes_per_trial['BAT'][i],color="blue")

        ax[0].set_title('channel' + str(good_channel_ids[i]) + ' ' + regions_list[good_channel_ids[i]-1] + ' p=' + str(p_bins))

        # plt.show()



source_data_df = pd.DataFrame(source_data_rows)

source_data_df.to_csv(
    'PSTH_spike_times_SourceData.csv',
    index=False
)