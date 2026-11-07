'''DIMENSIONALITY REDUCGION TO CLUSTTER NETWORKS/REGIONS

For Figure 3e

uses cebra: 

index: channel      network        network_scalar time
cols:  patient, trial_idx, trial_type, freq_band (col name is '0')

Only uses AUT trials, and trains to separate networks 

'''

import numpy as np
import quantities as pq
import glob   
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools as it
import random
import seaborn as sns
import scipy
from scipy.stats.stats import pearsonr

import DMN_LFP_funcs
import beh_funcs

import sys
sys.path.append('../')

import patient_params
patients_data = patient_params.patients_data

import decoding_funcs

import cebra
from cebra import CEBRA
import cebra.models

import plotting_funcs

from mpl_helpers import PdfPlotter
PdfPlotter('Figures/cebra_cluster_networks_c.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))


# plt.ion()

network = input("Choose cortical network (DMN, FrontoParietal, SomatoMotor, Limbic, Visual, DorsalAttention, VentralAttention, all): ")

if network=='all':
    networks = ['DMN', 'FrontoParietal', 'SomatoMotor', 'Limbic', 'Visual', 'DorsalAttention', 'VentralAttention']
else:
    networks = [network]

#concat all networks dfs
all_dfs = []
for n, network in enumerate(networks):
    all_dat = pd.read_csv('pkl_files/store_neural_data_list_with_ripple' + network + '.csv', header=[0, 1], index_col=[0, 1, 2, 3])

    #add network label
    df_cols = all_dat.columns.to_frame() #make df so can add network 
    df_cols['network']=network
    df_cols['network_scalar']=n
    all_dat.columns = pd.MultiIndex.from_frame(df_cols) #turn back into multiindex for cols

    all_dfs.append(all_dat)

# Concatenate along columns, aligning on index (patient, trial_idx, trial_type, time)
merged_df = pd.concat(all_dfs, axis=1, join="outer") #merged df with all the networks 

#split by patient
patient_dfs = [group for _, group in merged_df.groupby(level='patient')]
cleaned_dfs = [df.dropna(axis=1, how='all') for df in patient_dfs]

#just AUT trials
cleaned_dfs_AUT = [neural_data.iloc[np.where(neural_data.index.get_level_values('trial_type')=='AUT')[0]] for neural_data in cleaned_dfs]


#set up df so contacts are observations, not features

#index: netwrk, chan/region, trial_idx, trial type. Cols: freq band, timepoint
neural_data_list_channel_obs = [df.stack(0).T.stack(['trial_idx','trial_type'])for df in cleaned_dfs] #channels (region, network) are the observations 
neural_data_list_channel_obs_2 = [df.stack(0).T.stack(['time'])for df in cleaned_dfs] #channels (region, network) at each timept in trial are the observations

# index: channel      network        network_scalar time
# cols:  patient, trial_idx, trial_type, freq_band (col name is '0')
neural_data_list_channel_obs_AUT = [df.stack(0).T.stack(['time'])for df in cleaned_dfs_AUT] #channels (region, network) at each timept in trial are the observations


trial_type_float_labels = [1.0*(neural_data.index.get_level_values('trial_type')=='AUT') for neural_data in neural_data_list_channel_obs] 
network_labels_scalar = [neural_data.index.get_level_values('network_scalar').astype(float) for neural_data in neural_data_list_channel_obs_AUT] 
# network_labels_scalar = [neural_data.index.get_level_values('network_scalar') for neural_data in neural_data_list_channel_obs_AUT] 

#train to separate networks
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

multi_cebra_model.fit(neural_data_list_channel_obs_AUT,network_labels_scalar)

embeddings_all = []
for i, neural_data in enumerate(neural_data_list_channel_obs_AUT):
    embedding = multi_cebra_model.transform(neural_data,session_id=i)
    embeddings_all.append(embedding)

embeddings_concat = np.concatenate(embeddings_all, axis=0)

# cebra.plot_embedding(embeddings_concat, embedding_labels=np.concatenate(network_labels_scalar), cmap = 'Set1', idx_order=(1,2,3), markersize=5, alpha=0.5)
# for i, embedding in enumerate(embeddings_all):
#     cebra.plot_embedding(embedding, embedding_labels=network_labels_scalar[i], cmap = 'Set1', idx_order=(1,2,3), markersize=5, alpha=0.5)

colors = ['orange', 'blue', 'red', 'yellow', 'green', 'purple', 'grey'] #custom colors!
unique_labels = np.unique(np.concatenate(network_labels_scalar))

# Create a dictionary mapping each label to a specific color
label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
label_to_network = {scalar: network for scalar, network in zip(unique_labels, networks)}

# Convert labels to colors
embedding_colors = [label_to_color[label] for label in np.concatenate(network_labels_scalar)]

# Plot with custom colors
scatter = cebra.plot_embedding(
    embeddings_concat, 
    embedding_labels=None,  # Remove automatic colormap assignment
    idx_order=(1,2,3), 
    markersize=5, 
    alpha=0.5, 
    color=embedding_colors  # Pass custom colors directly
)
plt.title('Network clusters')
plotting_funcs.remove_ticks_and_labels_3d(scatter)


# Create legend handles
legend_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=network) 
                  for (_, network), color in zip(label_to_network.items(), colors)]

# Add legend
plt.legend(handles=legend_patches, title="Networks", loc="best")



scatter = cebra.plot_embedding(embeddings_concat, embedding_labels=None, idx_order=(2,1,3), markersize=5, alpha=0.5, color=embedding_colors)
plotting_funcs.remove_ticks_and_labels_3d(scatter)
plt.title('Network clusters')

# for i, embedding in enumerate(embeddings_all):
#     embedding_colors = [label_to_color[label] for label in network_labels_scalar[i]]
#     scatter_pt = cebra.plot_embedding(embedding, embedding_labels=None, idx_order=(1,2,3), markersize=5, alpha=0.5, color=embedding_colors)
#     plotting_funcs.remove_ticks_and_labels_3d(scatter_pt)

# plt.show()



#plot grid of subplots
import math

# Determine grid size based on number of embeddings
num_embeddings = len(embeddings_all)
cols = math.ceil(math.sqrt(num_embeddings))  # Set columns based on square root
rows = math.ceil(num_embeddings / cols)  # Compute necessary rows

# Create figure and 3D subplots
fig, ax = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), subplot_kw={'projection': '3d'})

# Flatten `ax` to handle cases where rows/cols > 1
ax = np.ravel(ax)

# Loop through embeddings and plot
for i, embedding in enumerate(embeddings_all):
    embedding_colors = [label_to_color[label] for label in network_labels_scalar[i]]
    
    # Plot embedding in corresponding subplot
    scatter_pt = cebra.plot_embedding(
        embedding, 
        embedding_labels=None, 
        idx_order=(1,2,3), 
        markersize=5, 
        alpha=0.5, 
        color=embedding_colors, 
        ax=ax[i]
    )
    plotting_funcs.remove_ticks_and_labels_3d(scatter_pt)
    ax[i].set_title(merged_df.index.get_level_values('patient').unique()[i])

plt.subplots_adjust(wspace=0.01, hspace=0.01)  # Reduce spacing between subplots
plt.tight_layout(pad=1.0)  # Reduce padding around figure




#quantify separation

from scipy.spatial.distance import cdist
import numpy as np

unique_labels = np.unique(np.concatenate(network_labels_scalar))
centroids = [embeddings_concat[np.concatenate(network_labels_scalar) == label].mean(axis=0) for label in unique_labels]
pairwise_distances = cdist(centroids, centroids, metric='euclidean')

print("Pairwise centroid distances:")
print(pairwise_distances)


fig, ax = plt.subplots()
cm_plot = ax.imshow(pairwise_distances, interpolation = 'nearest', aspect = 'equal')
ax.set_title('Pairwise centroid distances')
plt.colorbar(cm_plot)




1/0

#train to separate by trial type and then color by network 

multi_cebra_model2 = cebra.CEBRA(model_architecture='offset10-model',
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

multi_cebra_model2.fit(neural_data_list_channel_obs_2,trial_type_float_labels)

embeddings_all = []
for i, neural_data in enumerate(neural_data_list_channel_obs_2):
    embedding = multi_cebra_model2.transform(neural_data,session_id=i)
    embeddings_all.append(embedding)

embeddings_concat = np.concatenate(embeddings_all, axis=0)

network_labels = pd.factorize(np.concatenate(networks))[0]
cebra.plot_embedding(embeddings_concat, embedding_labels=np.concatenate(network_labels_scalar), cmap = 'Set1', idx_order=(1,2,3), markersize=5, alpha=0.5)
for i, embedding in enumerate(embeddings_all):
    cebra.plot_embedding(embedding, embedding_labels=network_labels_scalar[i], idx_order=(1,2,3), markersize=5, alpha=0.5)


