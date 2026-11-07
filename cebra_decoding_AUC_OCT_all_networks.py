''' CEBRA

DECODE AUT vs OCT!!

TRAIN MULTIPLE MODELS THEN ALIGN
Train decoder on resulting embeddings.
Do this is k fold cross validation

#hold out TRIALS, (entire trials, based on trial idx, not just dataframe rows, which are timepoints!)
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


import cebra
from cebra import CEBRA

import cebra.models
from sklearn.model_selection import KFold

import sys
sys.path.append('../')

import patient_params
patients_info = patient_params.patients_data
patients_data = DMN_LFP_funcs.choose_patient(patients_info) #choose all or 1 patient


from mpl_helpers import PdfPlotter
PdfPlotter('Figures/cebra_AUT_OCT_decoding.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))


mean_accuracy_all_networks_AUT_OCT = {}
all_folds_accuracy_all_networks_AUT_OCT = {}

f,ax = plt.subplots(1,3)

for network in ['DMN', 'FrontoParietal', 'SomatoMotor', 'Limbic', 'Visual', 'DorsalAttention', 'VentralAttention']:
    all_dat = pd.read_csv('pkl_files/store_neural_data_list_all_trialtypes_' + network + '.csv', header=[0, 1], index_col=[0, 1, 2, 3])

    patient_dfs = [group for _, group in all_dat.groupby(level='patient')]
    cleaned_dfs = [df.dropna(axis=1, how='all') for df in patient_dfs]

    neural_data_list_fixed = cleaned_dfs

    discrete_labels = [neural_data.index.get_level_values('trial_type') for neural_data in neural_data_list_fixed]

    #just AUT vs OCT
    neural_data_list_fixed_AUC_OCT = [neural_data.iloc[np.where((neural_data.index.get_level_values('trial_type')=='AUT') | (neural_data.index.get_level_values('trial_type')=='OCT'))[0]] for neural_data in cleaned_dfs]
    trial_type_float_labels = [1*(neural_data.index.get_level_values('trial_type')=='AUT') for neural_data in neural_data_list_fixed_AUC_OCT] #not supported :(


    ###DECODER

    n_splits = 4
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    #pre-make all the train/test splits, so that can train cebra for each patient, for each split. Ie train 1 model per patient, align, decode. Do this K times.
    AUT_OCT_unique_trial_inds = pd.concat(neural_data_list_fixed_AUC_OCT).index.to_frame()['trial_idx'].groupby('patient').unique()

    #make a df of the splits (5 splits per patient)
    splits_per_patient = AUT_OCT_unique_trial_inds.apply(lambda x: pd.Series([(x[train], x[test]) for train, test in kf.split(x)])).stack().apply(pd.Series, index=['train', 'test'])
    splits_df = splits_per_patient.reorder_levels([1, 0]).sort_index() #reorder so first index is split #, and next index is patient #

    #now loop through the 5 splits. For each split, train cebra for each patient.
    cm_all_folds = []
    accuracy_all_folds = []
    for i in range(n_splits):
        train_test_split = splits_df.loc[i] #get the current split, for all patients

        #loop through patients, train model and get train/test embeddings for each patient 
        train_embeddings = []
        test_embeddings = []
        train_labels = []
        test_labels = []
        for p, (X, y) in enumerate(zip(neural_data_list_fixed_AUC_OCT, trial_type_float_labels)):    # Single session training. p = patient index
            patient = neural_data_list_fixed[p].index.get_level_values('patient').unique().values[0] #get patient from df because out of order
            if patient=='UIC202420':
                continue
            if patient=='UIC202422':
                continue
            print(p)
            # X = X.unstack('time')
            X['label']=y #add labels to X df so can index the same way to get the train and test labels
            train_idx = train_test_split.iloc[p].train
            test_idx = train_test_split.iloc[p].test
            # X.loc[:,train_idx,:,:,:]
            train_data, test_data = X.loc[:,train_idx,:,:,:].drop('label',axis=1).values, X.loc[:,test_idx,:,:,:].drop('label',axis=1).values
            train_discrete_label, test_discrete_label = X.loc[:,train_idx,:,:,:].label.values, X.loc[:,test_idx,:,:,:].label.values


            cebra_model = CEBRA(model_architecture='offset10-model',
                                batch_size=512,
                                learning_rate=3e-4,
                                temperature=1,
                                output_dimension=5,
                                max_iterations=500,
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


        # AUT_vs_other_train_labels = [1*(label==1) for label in train_labels] #AUT vs rest (BAT and OCT)
        # AUT_vs_other_test_labels = [1*(label==1) for label in test_labels] #AUT vs rest (BAT and OCT)

        # 4. Train the decoder on training embedding and labels
        decoder = cebra.KNNDecoder()
        decoder.fit(np.concatenate(train_embeddings), np.concatenate(train_labels))

        predicted_labels = decoder.predict(np.concatenate(test_embeddings))
        accuracy = np.mean(predicted_labels == np.concatenate(test_labels))
        accuracy_all_folds.append(accuracy)
        cm = confusion_matrix(np.concatenate(test_labels),np.round(predicted_labels))
        cm_all_folds.append(cm)

        # #plot train and validation embeddings (all patients, current )
        # f, ax = plt.subplots(1, 3, figsize=(15, 5)) 
        # ax = [f.add_subplot(131, projection='3d'),
        #     f.add_subplot(132, projection='3d')]

        embeddings_concat = np.concatenate(train_embeddings, axis=0)
        cebra.plot_embedding(embeddings_concat, embedding_labels=np.concatenate(train_labels), idx_order=(0,2,3), markersize=2, alpha=0.8)

        embeddings_concat = np.concatenate(test_embeddings, axis=0)
        cebra.plot_embedding(embeddings_concat, embedding_labels=np.concatenate(test_labels), idx_order=(0,2,3), markersize=2, alpha=0.8)
        sns.despine(top = True, right = True, left = True, bottom = True)




    mean_accuracy = np.mean(accuracy_all_folds)  #0.718973141837066
    avg_cm = np.mean(cm_all_folds,axis=0)  
    cm_norm = avg_cm.astype('float') / avg_cm.sum(axis=1)[:, np.newaxis]
        # [0.75599747, 0.24400253]
        # [0.31858974, 0.68141026]
    class1_accuracy = np.array(cm_norm)[:,0][0] * 100
    class2_accuracy = np.array(cm_norm)[:,1][1] * 100

    mean_accuracy_all_networks_AUT_OCT.update({network: mean_accuracy})
    all_folds_accuracy_all_networks_AUT_OCT.update({network: accuracy_all_folds})
plt.show()

p_values = []
for i, scores in enumerate(list(all_folds_accuracy_all_networks_AUT_OCT.values())):
    # t,p = scipy.stats.ttest_1samp(scores, 0.5, alternative='greater')
    t,p = scipy.stats.ttest_1samp(scores, 0.5)
    p_values.append(p)
    print(p)

#FDR correction:
import statsmodels.stats.multitest as multi
rej, corr_p = multi.fdrcorrection(p_values, alpha=0.05, method='indep') 

f,ax = plt.subplots()
ax.bar(list(mean_accuracy_all_networks_AUT_OCT.keys()), list(mean_accuracy_all_networks_AUT_OCT.values()))
ax.set_ylabel('decoder accuracy')
plt.show()
plt.close()

#  'DMN': 0.5917724656867055,
#  'FrontoParietal': 0.5729074337572159,
#  'SomatoMotor': 0.5707948452513669,
#  'Limbic': 0.5484761030123046,
#  'Visual': 0.5865947002379273,
#  'DorsalAttention': 0.6004642722473604,
#  'VentralAttention': 0.586912371482684}
# FDR corrected p = 0.00853026, 0.00853026, 0.00860549, 0.0509065 , 0.00860549, 0.01146736, 0.00853026

