'''SINGLE UNIT ANALYSIS for DMN AUT task

CEBRA for creatiivty level (Figure 5j)
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
from sklearn.metrics import  roc_curve, auc, confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

import cebra
from cebra import CEBRA

import cebra.models
from sklearn.model_selection import KFold

from mpl_helpers import PdfPlotter

from mpl_helpers import PdfPlotter
PdfPlotter('Figures/single_unit_cebra_.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))

import patient_params
patients_info = patient_params.patients_data_spikes
patients_data = DMN_LFP_funcs.choose_patient(patients_info) #choose all or 1 patient




cm_all_patients = []
scores_all_patients = []
df_spiketimes_list = []

regions_spikes_dict = {}

all_dat_spikes = pd.read_csv('pkl_files/store_spikes_aut_df.csv', index_col=[0, 1, 2, 3, 4])

patient_spike_dfs = [group for _, group in all_dat_spikes.groupby(level='patient')]
cleaned_spike_dfs = [df.dropna(axis=1, how='all') for df in patient_spike_dfs]

spike_list_fixed = cleaned_spike_dfs
spike_dfs = spike_list_fixed

cont_orig_values = [neural_data.index.get_level_values('originality_cont').values for neural_data in spike_dfs]

'''CREATIVITY LEVEL DECODING FROM SINGLE UNITS'''

neural_data_list_fixed_originality = spike_dfs

from sklearn.preprocessing import KBinsDiscretizer
# Convert continuous labels into discrete bins
n_bins = 10  # Adjust this based on your label range
discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')

# Apply discretization to each label array
discretized_labels = [discretizer.fit_transform(labels.reshape(-1, 1)).flatten().astype(int) for labels in cont_orig_values]



n_splits = 5
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
PdfPlotter('Figures/cebra_single_cell_creativity_level_.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))
patients = np.array(patients_data)[:, 0]

label_type = input("Continuous (a) or discrete (b) creativity labels? ")

if label_type=='a':
    labels = cont_orig_values #for smooth embeddings, good for plots
else:
    labels = discretized_labels #for decoding and confusion matrix


from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

scores = []  # Store R² scores per fold
all_true_labels = []
all_pred_labels = []


# decision_scores = []

cm_all_folds = []
accuracy_all_folds = []
for i in range(n_splits):
    train_test_split = splits_df.loc[i] #get the current split, for all patients

    #loop through patients, train model and get train/test embeddings for each patient 
    train_embeddings = []
    test_embeddings = []
    train_labels = []
    test_labels = []

    for p, (X, y) in enumerate(zip(neural_data_list_fixed_originality, labels)):    # Single session training. p = patient index
        print(p)
        X['label']=y #add labels to X df so can index the same way to get the train and test labels
        
        '''#try only decoding from stimulus period'''
        # X = X[X.index.get_level_values('time')<10]
        
        train_idx = splits_df.loc[i,patients[p]].train
        test_idx = splits_df.loc[i,patients[p]].test
        train_data, test_data = X.loc[patients[p]].loc[train_idx].drop('label',axis=1).values, X.loc[patients[p]].loc[test_idx].drop('label',axis=1).values
        train_label, test_label = X.loc[patients[p]].loc[train_idx].label.values, X.loc[patients[p]].loc[test_idx].label.values


        cebra_model_creativity = CEBRA(model_architecture='offset10-model',
                            batch_size=512,
                            learning_rate=0.001,
                            temperature=1,
                            output_dimension=3,
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

        # X = CEBRA embeddings, y = continuous creativity scores




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


mean_accuracy = np.mean(accuracy_all_folds) 
avg_cm = np.mean(cm_all_folds,axis=0)  

cm_norm = avg_cm.astype('float') / avg_cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots()
cm_plot = ax.imshow(cm_norm, interpolation = 'nearest', aspect = 'equal')
ax.set_title('creativity decoding')
ax.set_ylabel('actual originality value')
ax.set_xlabel('predicted originality value')
plt.colorbar(cm_plot)


t,p = scipy.stats.ttest_1samp(accuracy_all_folds, 0.1, alternative='greater') #p = 0.025 (5D); p=0.0002 (3D)


plt.show()
