
''' CEBRA

TRAIN MULTI SESSION MODEL; I.E. Pseudo-subject

AUT vs BAT vs OCT discrete labels
or AUT vs OCT
or AUT vs BAT/OCT

Makes Supp Fig 4

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

from sklearn.metrics import confusion_matrix
# from mpl_helpers import PdfPlotter


import decoding_funcs

import plotting_funcs

import sys
sys.path.append('../')

import patient_params
patients_info = patient_params.patients_data
patients_data = DMN_LFP_funcs.choose_patient(patients_info) #choose all or 1 patient



network = input("Choose cortical network (DMN, FrontoParietal, SomatoMotor, Limbic, Visual, DorsalAttention, VentralAttention): ")

import cebra
from cebra import CEBRA


import cebra.models




# all_dat = pd.read_csv('pkl_files/store_neural_data_list_' + network + '.csv', header=[0, 1], index_col=[0, 1, 2, 3])
# all_dat = pd.read_csv('pkl_files/store_neural_data_list_with_ripple' + network + '.csv', header=[0, 1], index_col=[0, 1, 2, 3])
all_dat = pd.read_csv('pkl_files/store_neural_data_list_all_trialtypes_' + network + '.csv', header=[0, 1], index_col=[0, 1, 2, 3])




patient_dfs = [group for _, group in all_dat.groupby(level='patient')]
cleaned_dfs = [df.dropna(axis=1, how='all') for df in patient_dfs]

neural_data_list_fixed = cleaned_dfs

discrete_labels = [neural_data.index.get_level_values('trial_type') for neural_data in neural_data_list_fixed]
# discrete_labels = [1*(neural_data.index.get_level_values('trial_type')=='AUT') for neural_data in neural_data_list_fixed] 

#change labels to numbers
trial_type_dict = {'AUT': 1.0, 'BAT': 0.0, 'OCT': 2.0}
trial_type_float_labels = [np.array([trial_type_dict[label] for label in labels]) for labels in discrete_labels]

# trial_type_float_labels = [1.0*(neural_data.index.get_level_values('trial_type')=='AUT') for neural_data in neural_data_list_fixed] #not supported :(

# neural_data_list_fixed_AUC_OCT = [neural_data.iloc[np.where((neural_data.index.get_level_values('trial_type')=='AUT') | (neural_data.index.get_level_values('trial_type')=='OCT'))[0]] for neural_data in cleaned_dfs]
# trial_type_float_labels = [1.0*(neural_data.index.get_level_values('trial_type')=='AUT') for neural_data in neural_data_list_fixed_AUC_OCT] #not supported :(


'''MULTI SESSION MODEL'''
'''AUT vs BAT vs OCT'''


#change labels to numbers
trial_type_dict = {'AUT': 1.0, 'BAT': 0.0, 'OCT': 2.0}
trial_type_float_labels = [np.array([trial_type_dict[label] for label in labels]) for labels in discrete_labels]

multi_cebra_model = cebra.CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=5,
                        max_iterations=500,
                        max_adapt_iterations=500,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)


multi_cebra_model.fit(neural_data_list_fixed,trial_type_float_labels)
embeddings_all = []

for i, neural_data in enumerate(neural_data_list_fixed):
    embedding = multi_cebra_model.transform(neural_data,session_id=i)
    embeddings_all.append(embedding)

embeddings_concat = np.concatenate(embeddings_all, axis=0)

session_id_labels = []
for i, em in enumerate(embeddings_all):
    session_id_labels.append([i]* len(em))

from mpl_helpers import PdfPlotter
PdfPlotter('Figures/cebra_multi_session_AUT_OCT_BAT' + network + '_.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))

cebra.plot_embedding(embeddings_concat, embedding_labels=np.concatenate(trial_type_float_labels), idx_order=(1,2,3), markersize=5, alpha=0.5)

#plot each patient
## ----------PLOT GRID OF SUBPLOTS ------------
# plotting_funcs.plot_embeddings_grid(list(embeddings_all), list(trial_type_float_labels), titles=list([]), colors=['tab:blue', 'tab:orange', 'tab:green'], idx_order=[1,2,3])

label_to_color = {1.0: 'tab:orange', 0.0: 'tab:blue', 2.0: 'tab:green'}
# label_to_color = {'AUT': 'tab:orange', 'BAT': 'tab:blue', 'OCT': 'tab:green'}
plotting_funcs.plot_embeddings_grid_fixed(list(embeddings_all), list(trial_type_float_labels), label_to_color, idx_order=[1,2,3])
# plotting_funcs.plot_embeddings_grid_fixed(list(embeddings_all), list(discrete_labels), label_to_color, idx_order=[1,2,3])


########################################## AUT vs OCT ##########################################

neural_data_list_fixed_AUC_OCT = [neural_data.iloc[np.where((neural_data.index.get_level_values('trial_type')=='AUT') | (neural_data.index.get_level_values('trial_type')=='OCT'))[0]] for neural_data in cleaned_dfs]
trial_type_float_labels = [1.0*(neural_data.index.get_level_values('trial_type')=='AUT') for neural_data in neural_data_list_fixed_AUC_OCT] #not supported :(



multi_cebra_model = cebra.CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=5,
                        max_iterations=500,
                        max_adapt_iterations=500,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)


# multi_cebra_model.fit(neural_data_list_fixed,trial_type_float_labels)
multi_cebra_model.fit(neural_data_list_fixed_AUC_OCT,trial_type_float_labels)

embeddings_all = []

for i, neural_data in enumerate(neural_data_list_fixed_AUC_OCT):
    embedding = multi_cebra_model.transform(neural_data,session_id=i)
    embeddings_all.append(embedding)

embeddings_concat = np.concatenate(embeddings_all, axis=0)

session_id_labels = []
for i, em in enumerate(embeddings_all):
    session_id_labels.append([i]* len(em))

PdfPlotter('Figures/cebra_multi_session_AUT_OCT' + network + '_.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))

cebra.plot_embedding(embeddings_concat, embedding_labels=np.concatenate(trial_type_float_labels), idx_order=(1,2,3), markersize=5, alpha=0.5)
plt.title('AUT vs OCT')

#plot each patient
## ----------PLOT GRID OF SUBPLOTS ------------
# plotting_funcs.plot_embeddings_grid(list(embeddings_all), list(trial_type_float_labels), titles=list([]), colors=['tab:blue', 'tab:orange', 'tab:green'], idx_order=[1,2,3])

label_to_color = {1.0: 'tab:orange', 0.0: 'tab:green'}
# label_to_color = {'AUT': 'tab:orange', 'BAT': 'tab:blue', 'OCT': 'tab:green'}
plotting_funcs.plot_embeddings_grid_fixed(list(embeddings_all), list(trial_type_float_labels), label_to_color, idx_order=[0,2,3])
# plotting_funcs.plot_embeddings_grid_fixed(list(embeddings_all), list(discrete_labels), label_to_color, idx_order=[1,2,3])

# 1/0



plt.show()
# plt.close()





embeddings = []
# Single session training
for X, y in zip(neural_data_list_fixed_AUC_OCT, trial_type_float_labels):
    # Fit one CEBRA model per session
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

    cebra_model.fit(X, y)
    embeddings.append(cebra_model.transform(X))


# Align the single session embeddings to the first patient
alignment = cebra.data.helper.OrthogonalProcrustesAlignment()

for j, em in enumerate(embeddings[1:]):
    embeddings[j+1] = alignment.fit_transform(embeddings[0], em, trial_type_float_labels[0], trial_type_float_labels[j+1]) #j+1 because start with embeddings[1] while j=0
plotting_funcs.plot_embeddings_grid_fixed(list(embeddings), list(trial_type_float_labels), label_to_color, idx_order=[0,2,3])


embeddings = []
trial_type_float_labels = [np.array([trial_type_dict[label] for label in labels]) for labels in discrete_labels]
# Single session training
for X, y in zip(neural_data_list_fixed, trial_type_float_labels):
    # Fit one CEBRA model per session
    cebra_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=5,
                        max_iterations=1000,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

    cebra_model.fit(X, y)
    embeddings.append(cebra_model.transform(X))


# Align the single session embeddings to the first patient
alignment = cebra.data.helper.OrthogonalProcrustesAlignment()

for j, em in enumerate(embeddings[1:]):
    embeddings[j+1] = alignment.fit_transform(embeddings[0], em, trial_type_float_labels[0], trial_type_float_labels[j+1]) #j+1 because start with embeddings[1] while j=0
label_to_color = {1.0: 'tab:orange', 0.0: 'tab:blue', 2.0: 'tab:green'}
plotting_funcs.plot_embeddings_grid_fixed(list(embeddings), list(trial_type_float_labels), label_to_color, idx_order=[0,2,4])
