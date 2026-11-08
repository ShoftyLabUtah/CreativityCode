'''SINGLE UNIT + LFP CEBRA 

mostly does same thing as spikes_and_LFP_cebra_creativity_projections.py, but also does CCA at the end.

1. Make spiketimes df for each patient, set up df for future cebra analysis

2. multi CEBRA for creatiivty level

3. LFP cebra

4. cca to compare embeddings

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

import patient_params
patients_info = patient_params.patients_data_spikes
patients_data = DMN_LFP_funcs.choose_patient(patients_info) #choose all or 1 patient


patients = [patient for patient, data_loc, filename in patients_data]

network = input("Choose cortical network (DMN, FrontoParietal, SomatoMotor, Limbic, Visual, DorsalAttention, VentralAttention): ")

PdfPlotter('Figures/spikes_lfp_cebra_cca_' + network+ '_.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))


#load from csv. saved file with spikes for aut trials
all_dat_spikes = pd.read_csv('pkl_files/store_spikes_aut_df.csv', index_col=[0, 1, 2, 3, 4])

patient_spike_dfs = [group for _, group in all_dat_spikes.groupby(level='patient')]
cleaned_spike_dfs = [df.dropna(axis=1, how='all') for df in patient_spike_dfs]

spike_list_fixed = cleaned_spike_dfs
spike_dfs = spike_list_fixed

if len(spike_dfs)!=len(patients_data):
    print('need to actually run the stuff to add new patients')

    '''1. Make spiketimes df for each patient, set up df for future cebra analysis'''

    regions_spikes_dict = {}

    df_spiketimes_list = []
    raw_spike_dataframe_all_patients = []
    firing_rates_all_patients = []
        
    #get firing rates

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
        dataframe_blurry = dataframe.rolling(500, win_type='gaussian', center=True, min_periods=0, axis=1).mean(std=15)
        dataframe_fixed = dataframe_blurry.iloc[:, np.where(dataframe.columns.get_level_values('trial_idx').notnull())[0]]
        dataframe_rotated = dataframe_fixed.T.stack().unstack(['trial_idx', 'trial_type']).groupby(['unit', 'frame']).sum()



        '''###ONLY INCLUDE AUT vs BAT'''
        inds_to_decode = np.where((dataframe_rotated.columns.get_level_values('trial_type')=='AUT') | 
            (dataframe_rotated.columns.get_level_values('trial_type')=='BAT'))[0]
        dataframe_rotated_to_decode = dataframe_rotated[inds_to_decode]

        #bin this
        dataframe_rotated_to_decode['bin'] = dataframe_rotated_to_decode.index.get_level_values('frame') // 1000
        dataframe_rotated_to_decode = dataframe_rotated_to_decode.groupby(['unit', 'bin']).mean()



        #make list of dataframes for further cebra analysis
        X = dataframe_rotated_to_decode.unstack('bin').T

        df_spiketimes_list.append(X)
        
        for r, region in enumerate(regions[np.array(good_channel_ids)-1]):
            region = region[0]
            try:
                regions_spikes_dict[region].append([X.T.iloc[r]])
            except:
                # regions_spikes_dict.update({region: X.T.iloc[r]})
                regions_spikes_dict[region] =  [X.T.iloc[r]]

        dataframe_fixed_unblurrred = dataframe.iloc[:, np.where(dataframe.columns.get_level_values('trial_idx').notnull())[0]]
        dataframe_rotated_unblurred = dataframe_fixed_unblurrred.T.stack().unstack(['trial_idx', 'trial_type']).groupby(['unit', 'frame']).sum()
    #  inds_to_decode = np.where(dataframe_rotated_unblurred.columns.get_level_values('trial_type')=='AUT')[0]
        dataframe_rotated_to_decode = dataframe_rotated_unblurred[inds_to_decode]
        dataframe_rotated_to_decode['bin'] = dataframe_rotated_to_decode.index.get_level_values('frame') // 1000
        dataframe_rotated_to_decode = dataframe_rotated_to_decode.groupby(['unit', 'bin']).sum()
        dataframe_rotated_to_decode_unstack = dataframe_rotated_to_decode.unstack('bin')
        # dataframe_rotated_to_decode_unstack = dataframe_rotated_to_decode_unstack.T.loc[neural_data_originality.index.get_level_values('trial_idx').unique().values].T
        firing_rates = dataframe_rotated_to_decode_unstack.T
        raw_spike_dataframe_all_patients.append(dataframe)
        firing_rates_all_patients.append(firing_rates)
        

    df_spiketimes_list_AUT = [neural_data.iloc[np.where(neural_data.index.get_level_values('trial_type')=='AUT')[0]] for neural_data in firing_rates_all_patients]

    #get originality labels
    # df_spiketimes_list_AUT = [neural_data.iloc[np.where(neural_data.index.get_level_values('trial_type')=='AUT')[0]] for neural_data in df_spiketimes_list]
    neural_data_list_fixed_originality_single_unit = []
    patients = np.array(patients_data)[:, 0]

    for p, df in enumerate(df_spiketimes_list_AUT):
        patient = patients[p]
        patient_processed_AUT_datafile = 'scored_responses/' + patient + '_processed_data.csv'  #file that includes responses and originality scores
        AUT_scores_df = pd.read_csv(patient_processed_AUT_datafile)
        AUT_scores_df.drop(np.where(AUT_scores_df.Start.isna())[0],inplace=True) #drop rows that have nans in the times columns (those trials didnt happen)
        originality_by_trial = AUT_scores_df.originality

        # ###add new index to df (originality score)
        neural_data = df_spiketimes_list_AUT[p].unstack('bin')
        cols_frame = neural_data.index.to_frame()
        cols_frame['originality_cont'] = np.array(originality_by_trial)
        neural_data.index = pd.MultiIndex.from_frame(cols_frame[['trial_idx', 'trial_type', 'originality_cont']]) #reorder and turn into index

        cols_frame['patient'] = [patient]*len(originality_by_trial)
        neural_data.index = pd.MultiIndex.from_frame(cols_frame[['trial_idx', 'trial_type', 'originality_cont', 'patient']]) #reorder and turn into index
        neural_data = neural_data.reorder_levels(['patient', 'trial_type', 'trial_idx', 'originality_cont'])

        ###remove where orig = nan (where patient didnt answer, for instance)
        nan_inds = np.where(np.isnan(neural_data.index.get_level_values('originality_cont').values))[0]
        nan_trials = neural_data.index.get_level_values('trial_idx')[nan_inds]
        neural_data = neural_data.drop(index = nan_trials, level = 'trial_idx')    

        neural_data_list_fixed_originality_single_unit.append(neural_data.stack('bin'))


    #reorder index so it matches the LFP one from later
    spike_dfs = []
    for df in neural_data_list_fixed_originality_single_unit:
        new_index = df.reorder_levels(['patient','trial_idx','trial_type','originality_cont','bin']).index.set_names(['patient', 'trial_idx', 'trial_type', 'originality_cont','time'])
        df.index = new_index
        spike_dfs.append(df)

    # spike_dfs = neural_data_list_fixed_originality_single_unit


cont_orig_values = [neural_data.index.get_level_values('originality_cont').values for neural_data in spike_dfs]
time_in_trial = [neural_data.index.get_level_values('time').values for neural_data in spike_dfs] #not used atm



##MULTI CEBRA
multi_cebra_model_creativity_spikes = cebra.CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=3,
                        max_iterations=1000,
                        max_adapt_iterations=500,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

multi_cebra_model_creativity_spikes.fit(spike_dfs,cont_orig_values) #train model

#get embeddings for each patient
embeddings_single_unit = []
embeddings_single_unit_df = []
creativity_values_single_unit_list = []
for i, neural_data in enumerate(spike_dfs):
    embedding = multi_cebra_model_creativity_spikes.transform(neural_data,session_id=i)
    embedding_df = pd.DataFrame(embedding, index = neural_data.index)
    embeddings_single_unit.append(embedding)
    embeddings_single_unit_df.append(embedding_df)
    creativity_values_single_unit_list.append(neural_data.index.get_level_values('originality_cont').values)

creativity_values_single_unit = np.concatenate(creativity_values_single_unit_list)

embed_df_single_unit = pd.concat(embeddings_single_unit_df)


embeddings_concat_single_unit = np.concatenate(embeddings_single_unit_df, axis=0)

# cebra.plot_embedding(embeddings_concat_single_unit, embedding_labels=np.concatenate(cont_orig_values), idx_order=(0,1,2), markersize=5, alpha=0.5)
# plt.title('Multi-subject embedding; single unit data')

# for i, embedding in enumerate(embeddings_all_single_unit):
#     cebra.plot_embedding(embedding, embedding_labels=cont_orig_values[i], idx_order=(0,1,2), markersize=5, alpha=0.5)



'''GET LFP embeddings'''

all_dat_LFP = pd.read_csv('pkl_files/store_neural_data_list_' + network + '.csv', header=[0, 1], index_col=[0, 1, 2, 3])


####only look patients with coverage both in DA and DMN (ie subset of DMN)
# if network=='DMN':
    # all_dat_DA = pd.read_csv('pkl_files/store_neural_data_list_DorsalAttention.csv', header=[0, 1], index_col=[0, 1, 2, 3])
    # all_dat_DMN = pd.read_csv('pkl_files/store_neural_data_list_DMN.csv', header=[0, 1], index_col=[0, 1, 2, 3])
    # patients_subset = all_dat_DA.index.get_level_values('patient').unique().values
    # all_dat_LFP = all_dat_DMN.loc[patients_subset]

patient_dfs = [group for _, group in all_dat_LFP.groupby(level='patient')]
cleaned_dfs = [df.dropna(axis=1, how='all') for df in patient_dfs]

neural_data_list_fixed = cleaned_dfs



neural_data_list_fixed_AUT = [neural_data.iloc[np.where(neural_data.index.get_level_values('trial_type')=='AUT')[0]] for neural_data in neural_data_list_fixed]
neural_data_list_fixed_originality = []

for p, df in enumerate(neural_data_list_fixed_AUT):
    patient = neural_data_list_fixed_AUT[p].index.get_level_values('patient').unique().values[0] #get patient from df because out of order
    patient_processed_AUT_datafile = 'scored_responses/' + patient + '_processed_data.csv'  #file that includes responses and originality scores
    AUT_scores_df = pd.read_csv(patient_processed_AUT_datafile)
    AUT_scores_df.drop(np.where(AUT_scores_df.Start.isna())[0],inplace=True) #drop rows that have nans in the times columns (those trials didnt happen)
    originality_by_trial = AUT_scores_df.originality

    # ###add new index to df (originality score)
    neural_data = neural_data_list_fixed_AUT[p].unstack('time')
    cols_frame = neural_data.index.to_frame()
    cols_frame['originality_cont'] = np.array(originality_by_trial)
    neural_data.index = pd.MultiIndex.from_frame(cols_frame[['patient', 'trial_idx', 'trial_type', 'originality_cont']]) #reorder and turn into index

    ###remove where orig = nan (where patient didnt answer, for instance)
    nan_inds = np.where(np.isnan(neural_data.index.get_level_values('originality_cont').values))[0]
    nan_trials = neural_data.index.get_level_values('trial_idx')[nan_inds]
    neural_data = neural_data.drop(index = nan_trials, level = 'trial_idx')

    neural_data_list_fixed_originality.append(neural_data.stack('time'))

lfp_dfs = neural_data_list_fixed_originality

#creativity values for LFP dfs
cont_orig_values_LFP = [neural_data.index.get_level_values('originality_cont').values for neural_data in lfp_dfs]

#multi cebra for LFP
multi_cebra_model_creativity_lfp = cebra.CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=3,
                        max_iterations=1000,
                        max_adapt_iterations=500,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

multi_cebra_model_creativity_lfp.fit(lfp_dfs,cont_orig_values_LFP)

embeddings_lfp = []
embeddings_lfp_df = []
creativity_values_lfp_list = []
for i, neural_data in enumerate(lfp_dfs):
    embedding = multi_cebra_model_creativity_lfp.transform(neural_data,session_id=i)
    embedding_df = pd.DataFrame(embedding, index = neural_data.index)
    embeddings_lfp.append(embedding)
    embeddings_lfp_df.append(embedding_df)
    creativity_values_lfp_list.append(neural_data.index.get_level_values('originality_cont').values)

creativity_values_lfp = np.concatenate(creativity_values_lfp_list)

embed_df_lfp = pd.concat(embeddings_lfp_df)


#get coding axis and projcet onto it
#1. spike coding axis
model = LinearRegression()
model.fit(embed_df_single_unit, creativity_values_single_unit)

# Get the spike coding axis 
coding_axis = model.coef_ / np.linalg.norm(model.coef_)
creativity_projection_single_unit = embed_df_single_unit @ coding_axis

#plot projection vs actual creativity scatter plot
f,ax = plt.subplots()
plt.scatter(creativity_projection_single_unit, creativity_values_single_unit)
plt.xlabel("Spike creativity projection")
plt.ylabel("Creativity Value")
plt.title("Coding Axis Projection vs. Creativity")
plt.show()

#2. LFP coding axis
model = LinearRegression()
model.fit(embed_df_lfp, creativity_values_lfp)

# Get the LFP coding axis 
coding_axis = model.coef_ / np.linalg.norm(model.coef_)
creativity_projection_lfp = embed_df_lfp @ coding_axis

f,ax = plt.subplots()
plt.scatter(creativity_projection_lfp, creativity_values_lfp)
plt.xlabel("LFP creativity projection")
plt.ylabel("Creativity Value")
plt.title("Coding Axis Projection vs. Creativity")
plt.show()

# r2 = model.score(embed_df_single_unit, creativity_values_single_unit) #r2 = 0.37
r2 = model.score(embed_df_lfp, creativity_values_lfp) #r2 = 0.77
corr = np.corrcoef(creativity_projection_lfp, creativity_values_lfp)


#plot creativity projection, for LFP and for single units
for patient in creativity_projection_single_unit.index.get_level_values('patient').unique():
    print(patient)
    if patient not in creativity_projection_lfp.index.get_level_values('patient').unique().values:
        continue
    projection_single_unit = creativity_projection_single_unit.loc[patient]
    projection_lfp = creativity_projection_lfp.loc[patient]
    # 1/0
    f,ax = plt.subplots()
    ax.plot(projection_single_unit.values, color='k')
    ax.plot(projection_lfp.values, color='orange')
    for x in np.where(projection_single_unit.index.get_level_values('time').values==0)[0]:
        ax.axvline(x=x, color='k',linewidth = 0.5, linestyle='--') 
    ax.set_ylabel('Creatity Projections (spikes and LFP)')
    ax.set_xlabel('time (s)')

plt.show()




'''NOW DO CCA TO FIND RELATIONSHIOP BETWEEN SINGLE UNIT AND LFP EMBEDDINGS'''


#remake LPF embeddings so that only include sessions that also have single units
embeddings_all_LFP = []
creativity_values = []
counter = 0
for i, neural_data in enumerate(lfp_dfs):
    if lfp_dfs[i].index.get_level_values('patient').unique().values[0] in patients:
        print(patients[counter])
        counter+=1
        embedding = multi_cebra_model_creativity_lfp.transform(neural_data,session_id=i)
        embeddings_all_LFP.append(embedding)
        # import pdb;pdb.set_trace()

        creativity_values.append(neural_data.index.get_level_values('originality_cont').values)
    else:
        continue
embeddings_concat_LFP = np.concatenate(embeddings_all_LFP, axis=0)
# cebra.plot_embedding(embeddings_concat_LFP, embedding_labels=np.concatenate(creativity_values), idx_order=(0,1,2), markersize=5, alpha=0.5)

patients_LFP = all_dat_LFP.index.get_level_values('patient').unique()

embeddings_all_spikes = []
creativity_values = []
counter = 0
for i, neural_data in enumerate(spike_dfs):
    if spike_dfs[i].index.get_level_values('patient').unique().values[0] in patients_LFP:
        print(patients[counter])
        counter+=1
        embedding = multi_cebra_model_creativity_spikes.transform(neural_data,session_id=i)
        embeddings_all_spikes.append(embedding)
        # import pdb;pdb.set_trace()

        creativity_values.append(neural_data.index.get_level_values('originality_cont').values)
    else:
        continue
embeddings_concat_spikes = np.concatenate(embeddings_all_spikes, axis=0)
# cebra.plot_embedding(embeddings_concat_LFP, embedding_labels=np.concatenate(creativity_values), idx_order=(0,1,2), markersize=5, alpha=0.5)




### CCA
from sklearn.cross_decomposition import CCA
cca = CCA(n_components=3)
spike_cca, lfp_cca = cca.fit_transform(embeddings_concat_spikes, embeddings_concat_LFP)
correlations = [np.corrcoef(spike_cca[:, i], lfp_cca[:, i])[0, 1] for i in range(spike_cca.shape[1])]
print(correlations)



#plot spike vs lfp cca, with creativty as colorbar
f,ax = plt.subplots()
plt.scatter(spike_cca[:, 0], lfp_cca[:, 0], s=np.concatenate(creativity_values) * 10, c=np.concatenate(creativity_values), cmap='cool')
plt.colorbar(label='Creativity')
plt.xlabel("single unit embeddings CCA")
plt.ylabel("LFP embeddings CCA")
plt.show()
