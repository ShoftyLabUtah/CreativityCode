''' CEBRA
Makes Supp Fig 6b
decode creativity from LFP level using cebra. 
also Shuffle labels + do permutations
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
import seaborn as sns
import re
import mne
import scipy

import DMN_LFP_funcs
import beh_funcs


import decoding_funcs

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import KBinsDiscretizer

import cebra
from cebra import CEBRA

import cebra.models
from sklearn.model_selection import KFold

import sys
sys.path.append('../')

from mpl_helpers import PdfPlotter

import patient_params
patients_info = patient_params.patients_data
patients_data = DMN_LFP_funcs.choose_patient(patients_info) #choose all or 1 patient

# patients = [patient for (patient,data_loc) in patients_data]
mean_accuracy_all_networks_creativity = {}
all_folds_accuracy_all_networks_creativity = {}
mean_accuracy_all_networks_creativity_shuffle = {}
all_folds_accuracy_all_networks_creativity_shuffle = {}

for network in ['DMN', 'FrontoParietal', 'SomatoMotor', 'Limbic', 'Visual', 'DorsalAttention', 'VentralAttention']:
# for network in ['DMN', 'DorsalAttention']:


    try:
        all_dat = pd.read_csv('pkl_files/store_neural_data_list_with_ripple' + network + '.csv', header=[0, 1], index_col=[0, 1, 2, 3])

        # #add patient ID as index
        # pt_index = pd.Series(patients, name='patient').reindex(all_dat.index.get_level_values(0))
        # old_index = all_dat.index.to_frame()
        # pt_index.index = old_index.index
        # new_index = pd.MultiIndex.from_frame(pd.concat((pt_index, old_index.drop('patient', axis=1)), axis=1))
        # all_dat.index = new_index

        
        patient_dfs = [group for _, group in all_dat.groupby(level='patient')]
        cleaned_dfs = [df.dropna(axis=1, how='all') for df in patient_dfs]

        neural_data_list_fixed = cleaned_dfs

        # discrete_labels = [neural_data.index.get_level_values('trial_type') for neural_data in neural_data_list_fixed]
        discrete_labels = [1*(neural_data.index.get_level_values('trial_type')=='AUT') for neural_data in neural_data_list_fixed] 


    except:
        neural_data_list = []
        patients = []
        for patient, data_loc in patients_data:
            print(patient)

            #get mne file
            raw = DMN_LFP_funcs.get_raw(patient,data_loc) #get raw mne (with already marked bad channels that were selected manually)
            raw =  DMN_LFP_funcs.do_preprocessing(patient,data_loc,raw)


            # df_by_freq_band = make_df(raw, trial_info_df, good_chans_DMN, regions_list)
            df_by_freq_band = decoding_funcs.make_power_df(patient,data_loc, raw=raw, network = network, rotated = False)
            neural_data_list.append(df_by_freq_band.T)
            if df_by_freq_band.empty:
                continue 
            patients.append(patient)



        ####FIX LABELS TO MAKE SURE ONLY LOOKING AT TRIALS THAT ARE AUT OR BAT
        discrete_labels = []
        neural_data_list_fixed = []
        for neural_data in neural_data_list:
            if neural_data.empty:
                continue
            inds_to_decode = np.where((neural_data.index.get_level_values('trial_type')=='AUT') | 
                (neural_data.index.get_level_values('trial_type')=='BAT'))[0]
            neural_data_to_decode = neural_data.iloc[inds_to_decode]
            discrete_labels.append(1*(neural_data_to_decode.index.get_level_values('trial_type')=='AUT'))
            neural_data_list_fixed.append(neural_data_to_decode)

        all_dat = pd.concat(neural_data_list_fixed, keys=np.arange(len(neural_data_list_fixed))) #concat patients

        pt_index = pd.Series(patients, name='patient').reindex(all_dat.index.get_level_values(0))
        old_index = all_dat.index.to_frame()
        pt_index.index = old_index.index
        # new_index = pd.MultiIndex.from_frame(pd.concat((pt_index, old_index.drop('patient', axis=1)), axis=1))
        new_index = pd.MultiIndex.from_frame(pd.concat((pt_index, old_index.drop(0, axis=1)), axis=1))
        all_dat.index = new_index
        
        all_dat.to_csv('pkl_files/store_neural_data_list_' + network + '.csv')

        patient_dfs = [group for _, group in all_dat.groupby(level='patient')]
        cleaned_dfs = [df.dropna(axis=1, how='all') for df in patient_dfs]
        neural_data_list_fixed = cleaned_dfs



    ####FIX LABELS TO MAKE SURE ONLY LOOKING AT TRIALS THAT ARE AUT
    neural_data_list_fixed_AUT = [neural_data.iloc[np.where(neural_data.index.get_level_values('trial_type')=='AUT')[0]] for neural_data in neural_data_list_fixed]
    neural_data_list_fixed_originality = []
    
    for p, df in enumerate(neural_data_list_fixed_AUT):
        patient = neural_data_list_fixed_AUT[p].index.get_level_values('patient').unique().values[0] #get patient from df because out of order
        patient_processed_AUT_datafile = 'scored_responses/' + patient + '_processed_data.csv'  #file that includes responses and originality scores
        AUT_scores_df = pd.read_csv(patient_processed_AUT_datafile)
        AUT_scores_df.drop(np.where(AUT_scores_df.Start.isna())[0],inplace=True) #drop rows that have nans in the times columns (those trials didnt happen)
        if 'Stim' in AUT_scores_df.columns:
            AUT_scores_df = AUT_scores_df[AUT_scores_df.Stim==0]
            
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


    cont_orig_values = [neural_data.index.get_level_values('originality_cont').values for neural_data in neural_data_list_fixed_originality]


    # Convert continuous labels into discrete bins
    n_bins = 10  # Adjust this based on your label range
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')

    # Apply discretization to each label array
    discretized_labels = [discretizer.fit_transform(labels.reshape(-1, 1)).flatten().astype(int) for labels in cont_orig_values]




    ###DECODER

    n_splits = 4
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    #pre-make all the train/test splits, so that can train cebra for each patient, for each split. Ie train 1 model per patient, align, decode. Do this K times.
    # all_dat = pd.concat(neural_data_list_fixed, keys=np.arange(len(neural_data_list_fixed))) #concat patients
    # all_dat.index.names = ['patient']+all_dat.index.names[1:]

    # all_dat_AUT = pd.concat(neural_data_list_fixed_originality, keys=np.arange(len(neural_data_list_fixed_originality))) #concat patients
    all_dat_AUT = pd.concat(neural_data_list_fixed_originality) #concat patients
    all_dat_unique_trial_inds = all_dat_AUT.index.to_frame()['trial_idx'].groupby('patient').unique()
    #make a df of the splits (5 splits per patient)
    splits_per_patient = all_dat_unique_trial_inds.apply(lambda x: pd.Series([(x[train], x[test]) for train, test in kf.split(x)])).stack().apply(pd.Series, index=['train', 'test'])
    splits_df = splits_per_patient.reorder_levels([1, 0]).sort_index() #reorder so first index is split #, and next index is patient #

    #now loop through the 5 splits. For each split, train cebra for each patient.
    
    accuracy_all_perms_shuffle = [] #for permuations
    for perm in range(100): #permutations
    
        cm_all_folds = []
        accuracy_all_folds = []
        cm_all_folds_shuffle = []
        accuracy_all_folds_shuffle = []
        for i in range(n_splits):
            train_test_split = splits_df.loc[i] #get the current split, for all patients

            #loop through patients, train model and get train/test embeddings for each patient 
            train_embeddings = []
            test_embeddings = []
            train_labels = []
            test_labels = []

            train_embeddings_shuffle = []
            test_embeddings_shuffle = []

            for p, (X, y) in enumerate(zip(neural_data_list_fixed_originality, discretized_labels)):    # Single session training. p = patient index
                print(p)
                X['label']=y #add labels to X df so can index the same way to get the train and test labels
                
                '''#uncomment if want to only decode from stimulus period'''
                # X = X[X.index.get_level_values('time')<10]
                
                train_idx = train_test_split.iloc[p].train
                test_idx = train_test_split.iloc[p].test
                train_data, test_data = X.loc[:,train_idx,:,:,:].drop('label',axis=1).values, X.loc[:,test_idx,:,:,:].drop('label',axis=1).values
                train_label, test_label = X.loc[:,train_idx,:,:,:].label.values, X.loc[:,test_idx,:,:,:].label.values



                train_labels.append(train_label)
                test_labels.append(test_label)

                #for shuffled data
                cebra_model_creativity_shuffle = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=0.001,
                        temperature=1,
                        output_dimension=5,
                        max_iterations=500,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)
                # 2. Train a CEBRA-Behavior model on training data only
                cebra_model_creativity_shuffle.fit(train_data, np.random.permutation(train_label))

                # 3. Get embedding for training and validation data
                train_embedding_shuffle = cebra_model_creativity_shuffle.transform(train_data)
                test_embedding_shuffle = cebra_model_creativity_shuffle.transform(test_data)

                train_embeddings_shuffle.append(train_embedding_shuffle)
                test_embeddings_shuffle.append(test_embedding_shuffle)


            # Align the single session embeddings to the first patient
            alignment = cebra.data.helper.OrthogonalProcrustesAlignment()

            for j, em in enumerate(train_embeddings_shuffle[1:]):
                train_embeddings_shuffle[j+1] = alignment.fit_transform(train_embeddings_shuffle[0], em, train_labels[0], train_labels[j+1]) #j+1 because start with embeddings[1] while j=0
            for j, em in enumerate(test_embeddings):
                test_embeddings_shuffle[j] = alignment.fit_transform(train_embeddings_shuffle[0], em, train_labels[0], test_labels[j]) 

            # # 4. Train the decoder on shuffled training embedding and labels

            decoder_shuffle = cebra.KNNDecoder()
            decoder_shuffle.fit(np.concatenate(train_embeddings_shuffle), np.concatenate(train_labels))

            predicted_labels_shuffle = decoder_shuffle.predict(np.concatenate(test_embeddings_shuffle))
            accuracy_shuffle = np.mean(predicted_labels_shuffle == np.concatenate(test_labels))
            accuracy_all_folds_shuffle.append(accuracy_shuffle)
            cm_shuffle = confusion_matrix(np.concatenate(test_labels),predicted_labels_shuffle)
            cm_all_folds_shuffle.append(cm_shuffle)
            accuracy_all_perms_shuffle.append(accuracy_shuffle) #permutations



    mean_accuracy_shuffle = np.mean(accuracy_all_perms_shuffle) 
    # if network=='DMN' or network=='DorsalAttention':
    #     avg_cm_shuffle = np.mean(cm_all_folds_shuffle,axis=0)  

    #     cm_norm_shuffle = avg_cm_shuffle.astype('float') / avg_cm_shuffle.sum(axis=1)[:, np.newaxis]
    #     fig, ax = plt.subplots()
    #     cm_plot = ax.imshow(cm_norm_shuffle, interpolation = 'nearest', aspect = 'equal')
    #     plt.colorbar(cm_plot)

    # plt.show()
    # plt.close()


    mean_accuracy_all_networks_creativity_shuffle.update({network: mean_accuracy_shuffle})
    # all_folds_accuracy_all_networks_creativity_shuffle.update({network: accuracy_all_folds_shuffle})
    all_folds_accuracy_all_networks_creativity_shuffle.update({network: accuracy_all_perms_shuffle})




    cm_all_folds = []
    accuracy_all_folds = []
    for i in range(n_splits):
        train_test_split = splits_df.loc[i] #get the current split, for all patients

        #loop through patients, train model and get train/test embeddings for each patient 
        train_embeddings = []
        test_embeddings = []
        train_labels = []
        test_labels = []

        for p, (X, y) in enumerate(zip(neural_data_list_fixed_originality, discretized_labels)):    # Single session training. p = patient index
            print(p)
            X['label']=y #add labels to X df so can index the same way to get the train and test labels
            
            '''#try only decoding from stimulus period'''
            X = X[X.index.get_level_values('time')<10]
            
            train_idx = train_test_split.iloc[p].train
            test_idx = train_test_split.iloc[p].test
            train_data, test_data = X.loc[:,train_idx,:,:,:].drop('label',axis=1).values, X.loc[:,test_idx,:,:,:].drop('label',axis=1).values
            train_label, test_label = X.loc[:,train_idx,:,:,:].label.values, X.loc[:,test_idx,:,:,:].label.values


            cebra_model_creativity = CEBRA(model_architecture='offset10-model',
                                batch_size=512,
                                learning_rate=0.001,
                                temperature=1,
                                output_dimension=5,
                                max_iterations=500,
                                distance='cosine',
                                conditional='time_delta',
                                device='cuda_if_available',
                                verbose=True,
                                time_offsets=10)

            # 2. Train a CEBRA-Behavior model on training data only
            cebra_model_creativity.fit(train_data, train_label)

            # 3. Get embedding for training and validation data
            train_embedding = cebra_model_creativity.transform(train_data)
            test_embedding = cebra_model_creativity.transform(test_data)

            train_embeddings.append(train_embedding)
            test_embeddings.append(test_embedding)

            train_labels.append(train_label)
            test_labels.append(test_label)


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


    mean_accuracy = np.mean(accuracy_all_folds) 

    mean_accuracy_all_networks_creativity.update({network: mean_accuracy})
    all_folds_accuracy_all_networks_creativity.update({network: accuracy_all_folds})

PdfPlotter('Figures/cebra_creativity_decoding_across_networks.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))



f,ax = plt.subplots()
ax.bar(list(mean_accuracy_all_networks_creativity.keys()), list(mean_accuracy_all_networks_creativity.values()))
ax.set_ylabel('decoder accuracy (real)')
plt.show()


f,ax = plt.subplots()
ax.bar(list(mean_accuracy_all_networks_creativity_shuffle.keys()), list(mean_accuracy_all_networks_creativity_shuffle.values()))
ax.set_ylabel('decoder accuracy (shuffled cebra)')
plt.show()
plt.close()

#compare real to shuffle
p_vals_compare_to_shuffle
for key, score in mean_accuracy_all_networks_creativity.keys():
    # t,p = scipy.stats.ttest_1samp(mean_accuracy_all_networks_creativity_shuffle[key], score, alternative='greater')
    null_accuracies = mean_accuracy_all_networks_creativity_shuffle[key]
    p_value = np.mean(null_accuracies >= score)
    p_vals_compare_to_shuffle.append(p_value)


1/0


p_values_shuffle = []
for i, scores in enumerate(list(all_folds_accuracy_all_networks_creativity_shuffle.values())):
    t,p = scipy.stats.ttest_1samp(scores, 0.1, alternative='greater')
    p_values_shuffle.append(p)
    print(p)
import statsmodels.stats.multitest as multi
rej, corr_p = multi.fdrcorrection(p_values_shuffle, alpha=0.05, method='indep')

# shuffled p values
# [0.22822487723648502,
#  0.9219527442643268,
#  0.28710838560536145,
#  0.7374619866471982,
#  0.28710838560536145,
#  0.7374619866471982,
#  0.7374619866471982]


