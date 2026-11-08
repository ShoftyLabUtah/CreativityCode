'''SINGLE UNIT ANALYSIS for DMN AUT task: AUT vs BAT vs OCT
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
PdfPlotter('Figures/single_unit_cebra_AUT_OCT.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))



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




df_spiketimes_list_all = []

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
    
    timestamps_all_units = [] #timestamps of all waveforms, for the good channels
    for j in range(len(good_units)):
        timestamps_current_unit = [good_units[j].data[i] for i in range(len(good_units[j].data))]
        timestamps_all_units.append(timestamps_current_unit)


    trial_info_df = beh_funcs.get_trial_info_df(data_loc)


    ###BY TRIAL TYPE
    spikes_all_trialtypes = {'AUT':[],'BAT':[], 'OCT':[]}
    spikes_all_trialtypes_per_trial = {'AUT':[],'BAT':[],'OCT':[]}
    for trial_type in ['AUT','BAT','OCT']:
        start_times_ms, end_times_ms, trial_start_times_ms = beh_funcs.get_start_end_times_ms(trial_info_df, trial_type, data_loc=data_loc)
        start_times_sec = start_times_ms/1000
        end_times_sec = end_times_ms/1000 + 6

        spike_times_rel_to_trial_start_all_events_sorted_split_by_units = single_unit_analysis.func_get_spike_times_rel_to_events_split_by_unit(timestamps_all_units, start_times_sec, end_times_sec)
        spikes_all_trialtypes[trial_type] = spike_times_rel_to_trial_start_all_events_sorted_split_by_units

        spike_times_per_trial_split_by_units = single_unit_analysis.func_get_spike_times_per_trial_split_by_unit(timestamps_all_units, start_times_sec, end_times_sec)
        spikes_all_trialtypes_per_trial[trial_type] = spike_times_per_trial_split_by_units

    matfile = DMN_LFP_funcs.get_matfile(patient, data_loc)
    if patient=='UIC202504': #this patient uses channelmap2 for macros but channelmap1 for micros
        matfile = glob.glob('/Volumes/155.100.91.44/Data/UIC202504/Imaging/Registered/ChannelMap.mat')

    remove_micros = False
    chan_info = mat73.loadmat(matfile[0])
    regions = pd.DataFrame(chan_info['ElecAtlasProj'])[0]
    regions_list = [reg[0] if reg is not None else 'None' for reg in regions]


    data_loc_local = 'mne_files/'+patient
    raw = DMN_LFP_funcs.get_raw(patient,data_loc) #get raw mne (with already marked bad channels that were selected manually)
    raw = DMN_LFP_funcs.do_preprocessing(patient,data_loc,raw)


    ### MAKE SPIKE DATAFRAME: 1 WHERE SPIKE, 0 ELSEWHERE

    #ALL TRIAL TYPES
    trial_info_df_fixed = beh_funcs.trial_type_subset_df(trial_info_df, 'all') #cut out practice trials

    start_times_ms, end_times_ms, trial_start_times_ms = beh_funcs.get_start_end_times_ms(trial_info_df, 'all', data_loc=data_loc)
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

    #make dataframe: rows are cell ids, columns are frame number (and also trial ind and response type)
    dataframe = pd.DataFrame(0, index = good_unit_ids, columns = columns)
    dataframe.index.names = ['unit']

    #fill dataframe with the transient onset frames
    for i, unit_data in enumerate(timestamps_all_units):
        # spike_inds = list(map(lambda t: int(t*1000), unit_data)) #convert to ms; then convert to integers, so can use as indices
        spike_ms = np.array(unit_data)*1000
        spike_inds = spike_ms.astype(int)
        dataframe.iloc[i,np.array(spike_inds)[np.where(spike_inds<len_session_frames)]] = 1

    #drop values not in trials (along axis = 1 (ie columns) bc it doesnt know if i want to remove rows or columns of that index)
    #try decoding trial type??
    #do this by firing rate instead of blurring

    dataframe_fixed_unblurrred = dataframe.iloc[:, np.where(dataframe.columns.get_level_values('trial_idx').notnull())[0]]
    dataframe_rotated_to_decode = dataframe_fixed_unblurrred.T.stack().unstack(['trial_idx', 'trial_type']).groupby(['unit', 'frame']).sum()


    dataframe_rotated_to_decode['bin'] = dataframe_rotated_to_decode.index.get_level_values('frame') // 1000
    dataframe_rotated_to_decode = dataframe_rotated_to_decode.groupby(['unit', 'bin']).sum()
    dataframe_rotated_to_decode_unstack = dataframe_rotated_to_decode.unstack('bin')
    # dataframe_rotated_to_decode_unstack = dataframe_rotated_to_decode_unstack.T.loc[neural_data_originality.index.get_level_values('trial_idx').unique().values].T
    firing_rates = dataframe_rotated_to_decode_unstack.T
    # firing_rates_all_patients.append(firing_rates)

    X = firing_rates

    df_spiketimes_list_all.append(X)


############################ CEBRA ############################

n_splits = 6
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
# kf = StratifiedKFold(n_splits=n_splits)
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)


neural_data_list_fixed_AUC_OCT = [neural_data.iloc[np.where((neural_data.index.get_level_values('trial_type')=='AUT') | (neural_data.index.get_level_values('trial_type')=='OCT'))[0]] for neural_data in df_spiketimes_list_all]

kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

df_spiketimes_list = neural_data_list_fixed_AUC_OCT

#pre-make all the train/test splits, so that can train cebra for each patient, for each split. Ie train 1 model per patient, align, decode. Do this K times.
all_dat = pd.concat(df_spiketimes_list, keys=np.arange(len(df_spiketimes_list))) #concat patients
all_dat.index.names = ['patient']+all_dat.index.names[1:]
all_dat_unique_trial_inds = all_dat.index.to_frame()['trial_idx'].groupby('patient').unique()
#make a df of the splits (5 splits per patient)


splits_per_patient = all_dat_unique_trial_inds.apply(
    lambda x: pd.Series([(x[train], x[test]) for train, test in kf.split(x)])
    ).stack().apply(pd.Series, index=['train', 'test'])


splits_df = splits_per_patient.reorder_levels([1, 0]).sort_index() #reorder so first index is split #, and next index is patient #
discrete_labels = [1*(neural_data.index.get_level_values('trial_type')=='AUT') for neural_data in df_spiketimes_list] 

accuracy_all_folds = []
cm_all_folds = []

for i in range(n_splits):
    train_test_split = splits_df.loc[i] #get the current split, for all patients
    train_embeddings = []
    test_embeddings = []
    train_labels = []
    test_labels = []
    for p, (X, y) in enumerate(zip(df_spiketimes_list, discrete_labels)):    # Single session training. p = patient index
        print(p)
        # X = X.unstack('time')
        X['label']=y #add labels to X df so can index the same way to get the train and test labels
        train_idx = train_test_split.loc[p].train
        test_idx = train_test_split.loc[p].test
        X.loc[train_idx]
        train_data, test_data = X.loc[train_idx].drop('label',axis=1).values, X.loc[test_idx].drop('label',axis=1).values
        train_discrete_label, test_discrete_label = X.loc[train_idx].label.values, X.loc[test_idx].label.values


        cebra_model = CEBRA(model_architecture='offset10-model',
                            batch_size=512,
                            learning_rate=0.0003,
                            temperature=1,
                            output_dimension=3,
                            max_iterations=1000,
                            distance='cosine',
                            conditional='time_delta',
                            device='cuda_if_available',
                            verbose=True,
                            time_offsets=10)

        # 2. Train a CEBRA-Behavior model on training data only
        cebra_model.fit(train_data, train_discrete_label)

        # 3. Get embedding for training and validation data
        train_embedding = cebra_model.transform(train_data)
        test_embedding = cebra_model.transform(test_data)

        train_embeddings.append(train_embedding)
        test_embeddings.append(test_embedding)

        train_labels.append(train_discrete_label)
        test_labels.append(test_discrete_label)

    # Align the single session embeddings to the first patient
    alignment = cebra.data.helper.OrthogonalProcrustesAlignment()

    for j, em in enumerate(train_embeddings[1:]):
        train_embeddings[j+1] = alignment.fit_transform(train_embeddings[0], em, train_labels[0], train_labels[j+1]) #j+1 because start with embeddings[1] while j=0
    for j, em in enumerate(test_embeddings):
        test_embeddings[j] = alignment.fit_transform(train_embeddings[0], em, train_labels[0], test_labels[j]) 




    # 4. Train the decoder on training embedding and labels
    decoder = cebra.KNNDecoder()
    decoder.fit(np.concatenate(train_embeddings), np.concatenate(train_labels))

    predicted_labels = decoder.predict(np.concatenate(test_embeddings))
    accuracy = np.mean(predicted_labels == np.concatenate(test_labels))
    accuracy_all_folds.append(accuracy)
    cm = confusion_matrix(np.concatenate(test_labels),predicted_labels)
    cm_all_folds.append(cm)

    #plot train and validation embeddings (all patients, current )
    f, ax = plt.subplots(1, 3, figsize=(15, 5)) 
    ax = [f.add_subplot(131, projection='3d'),
        f.add_subplot(132, projection='3d')]

    embeddings_concat = np.concatenate(train_embeddings, axis=0)
    cebra.plot_embedding(embeddings_concat, embedding_labels=np.concatenate(train_labels), idx_order=(0,1,2), markersize=2, alpha=0.8, ax=ax[0])

    embeddings_concat = np.concatenate(test_embeddings, axis=0)
    cebra.plot_embedding(embeddings_concat, embedding_labels=np.concatenate(test_labels), idx_order=(0,1,2), markersize=2, alpha=0.8, ax=ax[1])
    sns.despine(top = True, right = True, left = True, bottom = True)

    
scipy.stats.ttest_1samp(accuracy_all_folds, 0.5)

f,ax = plt.subplots()
ax.bar(['AUT','OCT'], [np.mean(AUT_accuracy),np.mean(OCT_accuracy)], color=['tab:orange','tab:green'])
ax.set_ylim(0,80)
ax.axhline(y=50, color = 'k', linestyle = '--', linewidth = 0.5)
ax.set_ylabel('decoder accuracy (%)')
sns.despine(top = True, right = True)
ax.set_title('Decoding AUT vs OCT')

AUT_p = scipy.stats.ttest_1samp(AUT_accuracy, 50, alternative='greater') 
OCT_p = scipy.stats.ttest_1samp(OCT_accuracy, 50, alternative='greater') 


plt.show()





