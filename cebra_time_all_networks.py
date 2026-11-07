''' CEBRA
Run cebra-time (no labels during training)
plot embeddings and color code by trial type

decode trial type

find coding axis and project

Makes Fig. 3f-h and Supp. Fig. 5.
'''

import numpy as np
# from neo import io
# import quantities as pq
# import glob   
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

import pandas as pd
import seaborn as sns
import mat73
import itertools as it
import random
# import re
# import mne
import scipy

import DMN_LFP_funcs
import beh_funcs
import decoding_funcs
import plotting_funcs

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import KBinsDiscretizer

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

# patient = patients_data[0][0]

network = input("Choose cortical network (DMN, FrontoParietal, SomatoMotor, Limbic, Visual, DorsalAttention, VentralAttention) or 'all': ")

PdfPlotter('Figures/cebra_train_time_' + patient + '.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))

embeddings_all_nets = {}
embeddings_all_nets_dfs = {}
labels_all_nets = {}
accuracy_all_patients_all_nets = {}
patients = []

if network=='all':
    networks = ['DMN', 'FrontoParietal', 'SomatoMotor', 'Limbic', 'Visual', 'DorsalAttention', 'VentralAttention']
else:
    networks = [network]

for network in networks:

    all_dat = pd.read_csv('pkl_files/store_neural_data_list_with_ripple' + network + '.csv', header=[0, 1], index_col=[0, 1, 2, 3])

    embeddings = []
    embeddings_df = []
    labels_all = []
    try:
        all_dat.drop('UIC202420', inplace = True)
    except:
        all_dat=all_dat
    for patient in all_dat.index.get_level_values('patient').unique():

        df = all_dat.loc[patient]
        print(patient)
        neural_data_df = df.dropna(axis=1, how='all')
        discrete_labels = 1*(neural_data_df.index.get_level_values('trial_type')=='AUT')

        continuous_time_labels = neural_data_df.reset_index().groupby("trial_idx")["time"].transform(lambda x: (x  / x.max())) #normalize time


        single_cebra_model_time = cebra.CEBRA(model_architecture='offset10-model',
                                batch_size=512,
                                learning_rate=3e-4,
                                temperature=1,
                                output_dimension=5,
                                max_iterations=500,
                                distance='cosine',
                                # conditional='time_delta',
                                device='cuda_if_available',
                                verbose=True,
                                time_offsets=10)


        # single_cebra_model_AUT_BAT.fit(neural_data_df, continuous_time_labels)
        single_cebra_model_time.fit(neural_data_df)
        single_cebra_model_time.save('saved_cebra_models/single_cebra_model_time_' + network + '_' + patient + '.pt')


        embedding = single_cebra_model_time.transform(neural_data_df)

        embedding_df = pd.DataFrame(embedding, index = neural_data_df.index)
        embeddings.append(embedding)
        embeddings_df.append(embedding_df)
        labels_all.append(discrete_labels)


    alignment = cebra.data.helper.OrthogonalProcrustesAlignment()

    
    for j, em in enumerate(embeddings[1:]):
        embeddings[j+1] = alignment.fit_transform(embeddings[0], em, labels_all[0], labels_all[j+1]) #j+1 because start with embeddings[1] while j=0
    embeddings_df_aligned = []
    for e, em in enumerate(embeddings): #now convert to dfs
        embeddings_df_aligned.append(pd.DataFrame(em, index = embeddings_df[e].index))


    scatter = cebra.plot_embedding(np.concatenate(embeddings), embedding_labels=np.concatenate(labels_all), idx_order=(0,1,2), markersize=3, alpha=0.75)
    plt.title(network)
    plotting_funcs.remove_ticks_and_labels_3d(scatter)

    embeddings_all_nets.update({network: np.concatenate(embeddings)})
    embeddings_all_nets_dfs.update({network: pd.concat(embeddings_df_aligned, keys = all_dat.index.get_level_values('patient').unique())})
    labels_all_nets.update({network: np.concatenate(labels_all)})

    #decode, holding 1 patient out and training on the rest
    accuracy_all_patients = []
    for p in range(len(embeddings)): #hold 1 patient out
        test = embeddings[p]
        train = embeddings[:p] + embeddings[p+1:]
        test_labels = labels_all[p]
        train_labels = labels_all[:p] + labels_all[p+1:]

        decoder = cebra.KNNDecoder()
        decoder.fit(np.concatenate(train), np.concatenate(train_labels))
        predicted_labels = decoder.predict(test)

        accuracy = np.mean(predicted_labels == test_labels)
        accuracy_all_patients.append(accuracy*100)

    accuracy_all_patients_all_nets.update({network: accuracy_all_patients})

## ----------PLOT GRID OF SUBPLOTS ------------
plotting_funcs.plot_embeddings_grid(list(embeddings_all_nets.values()), list(labels_all_nets.values()), titles=list(embeddings_all_nets.keys()), colors=['tab:blue', 'tab:orange'], idx_order=[1,2,3])

## ----------STATS + BOXPLOT ------------
import statsmodels.stats.multitest as multi
p_values = []
for key in accuracy_all_patients_all_nets.keys():
    t,p = scipy.stats.ttest_1samp(accuracy_all_patients_all_nets[key], 50)
    print(scipy.stats.ttest_1samp(accuracy_all_patients_all_nets[key], 50))
    p_values.append(p)
    # print(key + ': ' + str(p))
rej, corr_p = multi.fdrcorrection(p_values, alpha=0.05, method='indep') #FDR correction

p_values = []
for key in list(accuracy_all_patients_all_nets.keys())[1:]:
    t,p = scipy.stats.mannwhitneyu(accuracy_all_patients_all_nets['DMN'], accuracy_all_patients_all_nets[key])
    p_values.append(p)
rej, corr_p = multi.fdrcorrection(p_values, alpha=0.05, method='indep') #FDR correction
1/0
# decoding holding out patients:
#p vals

#DMN: 5.11869510520471e-06,
#Frontoparietal:  0.000405113830774921,
#Somatomotor:  0.016842148197514357,
#Limbic:  0.005503440611374851,
#Visual:  0.005503440611374851,
#Dorsal Attention:  0.005503440611374851,
#Ventral Attention:  0.7560575371944769]


# DMN - Frontoparietal: 0.002955125040905343,
# DMN - Somatomotor: 0.0008554901611268958,
# DMN - Limbic: 0.00046744680036484536,
# DMN - Visual: 0.0010151462701815665,
# DMN - Dorsal Attention 0.0032387408884918647,
# DMN - Ventral Attention: 0.00046744680036484536

colors = ['orange', 'blue', 'red', 'yellow', 'green', 'purple', 'grey']
f,ax = plt.subplots()
ax.boxplot(list(accuracy_all_patients_all_nets.values()), showfliers=False, labels = list(accuracy_all_patients_all_nets.keys()))
for i, scores in enumerate(list(accuracy_all_patients_all_nets.values())):
    x = np.random.normal(i+1, scale = 0.04 ,size=len(scores))
    ax.scatter(x, scores, s = 10, color = colors[i], edgecolors='k', linewidths=0.5)
ax.axhline(50,color='k',linewidth = 0.5, linestyle='--')
# ax.set_ylim(1,100)
sns.despine(top = True, right = True)
ax.set_ylabel('% decoding accuracy')
ax.set_title('Unsupervised training: Decoding AUT vs BAT')

# median performance:
# 67.29166666666667
# 56.041666666666664
# 53.94886363636364
# 54.16666666666667
# 53.720238095238095
# 55.83333333333333
# 50.83333333333333


# LFP coding axis
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(embeddings_all_nets_dfs[network], labels_all_nets[network])

# Get the LFP coding axis 
coding_axis = model.coef_ / np.linalg.norm(model.coef_)
trial_type_projection_lfp = embeddings_all_nets_dfs[network] @ coding_axis


f,ax = plt.subplots()
plt.scatter(trial_type_projection_lfp, labels_all_nets[network])
plt.xlabel("DMN trial type projection")
plt.ylabel("Trial Type Value")
plt.title("Coding Axis Projection vs. Trial Type")
plt.show()


#plot creativity projection, for LFP and for single units
for patient in trial_type_projection_lfp.index.get_level_values('patient').unique():
    print(patient)
    projection_lfp = trial_type_projection_lfp.loc[patient]
    f,ax =plt.subplots(figsize=(8,4))
    ax.plot(projection_lfp.values, color='orange')
    
    norm_trial_type = 2 * (1*(projection_lfp.index.get_level_values('trial_type')=='AUT')) - 1

    ax.plot(norm_trial_type)
    for x in np.where(projection_lfp.index.get_level_values('time').values==0)[0]:
        ax.axvline(x=x, color='k',linewidth = 0.5, linestyle='--') 
    ax.set_xlabel('time (s)')
    ax.set_title(patient)

plt.show()