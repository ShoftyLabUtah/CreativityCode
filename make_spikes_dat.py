'''make single unit dat file

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
import single_unit_analysis
import mat73
import scipy
import statsmodels.stats.multitest as multi

from sklearn.metrics import mutual_info_score
from sklearn.svm import SVC 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

import cebra
from cebra import CEBRA

import cebra.models
from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

from mpl_helpers import PdfPlotter
PdfPlotter('Figures/spikes_lfp_cebra_creativity_projections.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))

import patient_params
patients_info = patient_params.patients_data_spikes
patients_data = DMN_LFP_funcs.choose_patient(patients_info) #choose all or 1 patient

regions_spikes_dict = {}
df_spiketimes_list = []
raw_spike_dataframe_all_patients = []
firing_rates_all_patients = []    

for patient, data_loc, filename in patients_data:
    print(patient)

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
    

    matfile = DMN_LFP_funcs.get_matfile(patient, data_loc)
    if patient=='UIC202504': #this patient uses channelmap2 for macros but channelmap1 for micros
        matfile = glob.glob('/Volumes/155.100.91.44/Data/UIC202504/Imaging/Registered/ChannelMap.mat')

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

    try:
        networks = pd.DataFrame(chan_info['ElecAtlasProj'])[7]
    except: 
        networks = pd.DataFrame(chan_info['ElecAtlasProj'])[6]
    networks_list = [net[0] if net is not None else 'None' for net in networks]
    
    good_chan_inds = [chan - 1 for chan in good_channel_ids]
    good_regions = regions[good_chan_inds]
    good_networks = [n[0] for n in networks[good_chan_inds]]

    timestamps_all_units = [] #timestamps of all waveforms, for the good channels
    for j in range(len(good_units)):
        timestamps_current_unit = [good_units[j].data[i] for i in range(len(good_units[j].data))]
        timestamps_all_units.append(timestamps_current_unit)


    trial_info_df = beh_funcs.get_trial_info_df(data_loc)


    ###BY TRIAL TYPE
    spikes_all_trialtypes = {'AUT':[],'BAT':[]}
    spikes_all_trialtypes_per_trial = {'AUT':[],'BAT':[]}
    for trial_type in ['AUT','BAT']:
        start_times_ms, end_times_ms, trial_start_times_ms = beh_funcs.get_start_end_times_ms(trial_info_df, trial_type)
        start_times_sec = start_times_ms/1000
        end_times_sec = end_times_ms/1000 + 6

        spike_times_rel_to_trial_start_all_events_sorted_split_by_units = single_unit_analysis.func_get_spike_times_rel_to_events_split_by_unit(timestamps_all_units, start_times_sec, end_times_sec)
        spikes_all_trialtypes[trial_type] = spike_times_rel_to_trial_start_all_events_sorted_split_by_units

        spike_times_per_trial_split_by_units = single_unit_analysis.func_get_spike_times_per_trial_split_by_unit(timestamps_all_units, start_times_sec, end_times_sec)
        spikes_all_trialtypes_per_trial[trial_type] = spike_times_per_trial_split_by_units

    matfile = DMN_LFP_funcs.get_matfile(patient, data_loc)
    if patient=='UIC202504': #this patient uses channelmap2 for macros but channelmap1 for micros
        matfile = glob.glob('/Volumes/155.100.91.44/Data/UIC202504/Imaging/Registered/ChannelMap.mat')



    data_loc_local = 'mne_files/'+patient
    raw = DMN_LFP_funcs.get_raw(patient,data_loc) #get raw mne (with already marked bad channels that were selected manually)
    raw = DMN_LFP_funcs.do_preprocessing(patient,data_loc,raw)


    ### MAKE SPIKE DATAFRAME: 1 WHERE SPIKE, 0 ELSEWHERE

    #ALL TRIAL TYPES
    trial_info_df_fixed = beh_funcs.trial_type_subset_df(trial_info_df, 'all') #cut out practice trials

    start_times_ms, end_times_ms, trial_start_times_ms = beh_funcs.get_start_end_times_ms(trial_info_df, 'all')
    start_times_sec = start_times_ms/1000
    end_times_sec = end_times_ms/1000 + 6

    start_frames = start_times_ms
    start_frames = start_frames.astype(int)
    frames_in_trial = 16*1000
    len_session_frames = len(raw) ### USE LFP SIGNAL IN ORDER TO DETERMINE DF LENGTH



    #make array that is the length of the total num frames in the session
    #so that I can label each frame as being in a given trial
    trial_numbers_array = np.tile(np.nan, len_session_frames) #initiaze to nans
    frame = np.tile(np.nan, len_session_frames) #initiaze to nans
    trial_types_array = np.tile('', len_session_frames).astype(object) #initiaze to empty strings

    for trial_num, start_frame in enumerate(start_frames):
        trial_numbers_array[start_frame : start_frame + frames_in_trial] = trial_num
        trial_types_array[start_frame : start_frame + frames_in_trial] = trial_info_df_fixed.stimType.iloc[int(trial_num/3)]
        frame[start_frame : start_frame + frames_in_trial] = np.arange(frames_in_trial)



    #take the arrays and make them into pandas indices for the dataframe
    #first make MultiIndex object
    columns = pd.MultiIndex.from_tuples(list(zip(trial_numbers_array, trial_types_array, frame)), names = ['trial_idx', 'trial_type', 'frame'])
    index = pd.MultiIndex.from_tuples(list(zip(good_unit_ids, good_regions, good_networks)), names = ['unit', 'region', 'network'])


    #make dataframe: rows are cell ids, columns are frame number (and also trial ind and response type)
    dataframe = pd.DataFrame(0, index = index, columns = columns)
    # dataframe.index.names = ['unit']

    #fill dataframe with the transient onset frames
    for i, unit_data in enumerate(timestamps_all_units):
        # spike_inds = list(map(lambda t: int(t*1000), unit_data)) #convert to ms; then convert to integers, so can use as indices
        spike_ms = np.array(unit_data)*1000
        spike_inds = spike_ms.astype(int)
        dataframe.iloc[i,np.array(spike_inds)[np.where(spike_inds<len_session_frames)]] = 1

    #drop values not in trials (along axis = 1 (ie columns) bc it doesnt know if i want to remove rows or columns of that index)
    #try decoding trial type??
    dataframe_fixed = dataframe.iloc[:, np.where(dataframe.columns.get_level_values('trial_idx').notnull())[0]]
    dataframe_rotated = dataframe_fixed.T.stack('unit').stack('region').stack('network').unstack(['trial_idx', 'trial_type']).groupby(['unit', 'region', 'network', 'frame']).sum()



    '''###ONLY INCLUDE AUT vs BAT'''
    inds_to_decode = np.where((dataframe_rotated.columns.get_level_values('trial_type')=='AUT') | 
        (dataframe_rotated.columns.get_level_values('trial_type')=='BAT'))[0]
    dataframe_rotated_to_decode = dataframe_rotated[inds_to_decode]

    #bin this
    dataframe_rotated_to_decode['bin'] = dataframe_rotated_to_decode.index.get_level_values('frame') // 1000

    dataframe_rotated_to_decode = dataframe_rotated_to_decode.groupby(['unit', 'region', 'network', 'bin']).sum() #sum all spikes in 1 bin
    dataframe_rotated_to_decode_unstack = dataframe_rotated_to_decode.unstack('bin')
    # dataframe_rotated_to_decode_unstack = dataframe_rotated_to_decode_unstack.T.loc[neural_data_originality.index.get_level_values('trial_idx').unique().values].T
    firing_rates = dataframe_rotated_to_decode_unstack.T
    raw_spike_dataframe_all_patients.append(dataframe)
    firing_rates_all_patients.append(firing_rates)



#save spikes for AUT and BAT trials
patients = np.array(patients_data)[:,0]
all_spikes_dat = pd.concat(firing_rates_all_patients, keys=patients, names = ['patient', 'trial_idx', 'trial_type', 'bin']) #concat patients
all_spikes_dat.to_csv('pkl_files/store_spikes_aut_bat_df.csv')


