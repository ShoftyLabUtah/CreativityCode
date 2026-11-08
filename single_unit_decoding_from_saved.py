'''SINGLE UNIT ANALYSIS for DMN AUT task

1. Decode AUT vs BAT

2. CEBRA for AUT vs BAT (Fig 5f-g)

3. CEBRA for creatiivty level (Fig 5j)
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

from mpl_helpers import PdfPlotter

from mpl_helpers import PdfPlotter
PdfPlotter('Figures/single_unit_cebra_AUT_BAT.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))

import patient_params
patients_info = patient_params.patients_data_spikes
patients_data = DMN_LFP_funcs.choose_patient(patients_info) #choose all or 1 patient


'''1. Make spiketimes df for each patient, decode AUT vs BAT for each patient, set up df for future cebra analysis'''
cm_all_patients = []
scores_all_patients = []
df_spiketimes_list = []

regions_spikes_dict = {}

all_dat_spikes = pd.read_csv('pkl_files/store_spikes_aut_bat_df_cut.csv', header=[0, 1, 2], index_col=[0, 1, 2, 3])

patient_spike_dfs = [group for _, group in all_dat_spikes.groupby(level='patient')]
cleaned_spike_dfs = [df.dropna(axis=1, how='all') for df in patient_spike_dfs]

spike_list_fixed = cleaned_spike_dfs
spike_dfs = spike_list_fixed
df_spiketimes_list = spike_dfs

for p, (patient, data_loc, filename) in enumerate(patients_data):
    print(patient)

    dataframe_rotated_to_decode = spike_dfs[p].unstack('bin').T

    #decode with SVM
    clf = SVC(C=1, kernel='linear', tol=1e-6,class_weight='balanced',random_state=42)
    kf = KFold(n_splits=5)
    answers_trial_type = []
    scores = []
    for train_idx, test_idx in kf.split(dataframe_rotated_to_decode.columns.get_level_values('trial_idx')):
        print(len(train_idx))
        train = dataframe_rotated_to_decode.iloc[:,list(train_idx)]
        test = dataframe_rotated_to_decode.iloc[:,list(test_idx)]
        clf.fit(train.T, train.columns.get_level_values('trial_type'))
        answers_trial_type.extend(clf.predict(test.T))
        score = clf.score(test.T, test.columns.get_level_values('trial_type'))
        scores.append(score)
    cm = confusion_matrix(dataframe_rotated_to_decode.columns.get_level_values('trial_type'), answers_trial_type)
    avg_score = np.mean(scores)

    scores_all_patients.append(avg_score)
    cm_all_patients.append(cm)



#avg score:65%
#p-value: 0.00888

# scores all patients: 
# [0.8428571428571429,
#  0.5000000000000001,
#  0.6666666666666666,
#  0.6333333333333333,
#  0.5,
#  0.9,
#  0.7999999999999999,
#  0.4833333333333333,
#  0.5666666666666667,
#  0.55]



''' 2. CEBRA DECODING AUT VS BAT'''

all_dat_spikes = pd.read_csv('pkl_files/store_spikes_aut_bat_df_cut.csv', header=[0, 1, 2], index_col=[0, 1, 2, 3])

patient_spike_dfs = [group for _, group in all_dat_spikes.groupby(level='patient')]
cleaned_spike_dfs = [df.dropna(axis=1, how='all') for df in patient_spike_dfs]

spike_list_fixed = cleaned_spike_dfs
spike_dfs = spike_list_fixed
df_spiketimes_list = spike_dfs

n_splits = 6
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

#pre-make all the train/test splits, so that can train cebra for each patient, for each split. Ie train 1 model per patient, align, decode. Do this K times.
all_dat = pd.concat(spike_dfs, keys=np.arange(len(spike_dfs))) #concat patients


all_dat_spikes.index.names = ['patient']+all_dat_spikes.index.names[1:]
all_dat_unique_trial_inds = all_dat_spikes.index.to_frame()['trial_idx'].groupby('patient').unique()
#make a df of the splits (5 splits per patient)
splits_per_patient = all_dat_unique_trial_inds.apply(lambda x: pd.Series([(x[train], x[test]) for train, test in kf.split(x)])).stack().apply(pd.Series, index=['train', 'test'])
splits_df = splits_per_patient.reorder_levels([1, 0]).sort_index() #reorder so first index is split #, and next index is patient #

discrete_labels = [1*(neural_data.index.get_level_values('trial_type')=='AUT') for neural_data in spike_dfs] 

accuracy_all_folds = []
cm_all_folds = []

for i in range(n_splits):
    train_test_split = splits_df.loc[i] #get the current split, for all patients
    train_embeddings = []
    test_embeddings = []
    train_labels = []
    test_labels = []
    for p, (X, y) in enumerate(zip(spike_dfs, discrete_labels)):    # Single session training. p = patient index
        print(p)
        # X = X.unstack('time')
        X['label']=y #add labels to X df so can index the same way to get the train and test labels
        train_idx = train_test_split.iloc[p].train
        test_idx = train_test_split.iloc[p].test
        train_data, test_data = X.loc[:,train_idx,:,:,:].drop('label',axis=1).values, X.loc[:,test_idx,:,:,:].drop('label',axis=1).values
        train_discrete_label, test_discrete_label = X.loc[:,train_idx,:,:,:].label.values, X.loc[:,test_idx,:,:,:].label.values


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

scipy.stats.ttest_1samp(accuracy_all_folds, 0.5, alternative='greater') #pvalue=0.0003
scipy.stats.ttest_1samp(accuracy_all_folds, 0.5) #pvalue= pvalue=0.0013358 (from subset)



avg_cm_all_folds =np.mean(cm_all_folds,axis=0)
cm_norm_all_folds = [cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] for cm in cm_all_folds]

AUT_accuracy = np.array(cm_norm_all_folds)[:,0,0] * 100
BAT_accuracy = np.array(cm_norm_all_folds)[:,1,1] * 100

accuracy_data = [AUT_accuracy, BAT_accuracy]

f,ax = plt.subplots()
ax.bar(['AUT','BAT'], [np.mean(AUT_accuracy),np.mean(BAT_accuracy)], color=['tab:orange','tab:blue'])
ax.set_ylim(0,80)
ax.axhline(y=50, color = 'k', linestyle = '--', linewidth = 0.5)
ax.set_ylabel('decoder accuracy (%)')
sns.despine(top = True, right = True)
ax.set_title('Decoding AUT vs BAT')

AUT_p = scipy.stats.ttest_1samp(AUT_accuracy, 50) #pvalue=0.00078. now pvalue=0.0314
BAT_p = scipy.stats.ttest_1samp(BAT_accuracy, 50) #pvalue=0.02  now pvalue=0.0379
# ax.text(0.25,10,f"{AUT_p[1]:.2}",transform=trans)
# ax.text(0.75,10,f"{BAT_p[1]:.2}",transform=trans)

plt.show()


'''3. CREATIVITY LEVEL DECODING FROM SINGLE UNITS'''


df_spiketimes_list_AUT = [neural_data.iloc[np.where(neural_data.index.get_level_values('trial_type')=='AUT')[0]] for neural_data in df_spiketimes_list]
neural_data_list_fixed_originality = []
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
    neural_data = neural_data.drop('label',axis=1)
    neural_data_list_fixed_originality.append(neural_data.stack('bin'))


cont_orig_values = [neural_data.index.get_level_values('originality_cont').values for neural_data in neural_data_list_fixed_originality]

from sklearn.preprocessing import KBinsDiscretizer
# Convert continuous labels into discrete bins
n_bins = 10  # Adjust this based on your label range
discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')

# Apply discretization to each label array
discretized_labels = [discretizer.fit_transform(labels.reshape(-1, 1)).flatten().astype(int) for labels in cont_orig_values]



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
PdfPlotter('Figures/cebra_single_cell_creativity_level.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))

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
        train_data, test_data = X.loc[:,:,train_idx,:,:].drop('label',axis=1).values, X.loc[:,:,test_idx,:,:].drop('label',axis=1).values
        train_label, test_label = X.loc[:,:,train_idx,:,:].label.values, X.loc[:,:,test_idx,:,:].label.values


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


plt.show()

1/0
##MULTI CEBRA FOR VISUALIZATION

##redoing this for the moment, because neural_data_list_fixed_originality now has labels added

neural_data_list_fixed_originality = []
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

    neural_data_list_fixed_originality.append(neural_data.stack('bin'))



multi_cebra_model_creativity = cebra.CEBRA(model_architecture='offset10-model',
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

multi_cebra_model_creativity.fit(neural_data_list_fixed_originality,cont_orig_values)
embeddings_all = []

for i, neural_data in enumerate(neural_data_list_fixed_originality):
    embedding = multi_cebra_model_creativity.transform(neural_data,session_id=i)
    embeddings_all.append(embedding)

embeddings_concat = np.concatenate(embeddings_all, axis=0)

cebra.plot_embedding(embeddings_concat, embedding_labels=np.concatenate(cont_orig_values), idx_order=(0,1,2), markersize=5, alpha=0.5)
plt.title('Multi-subject embedding; single unit data')

for i, embedding in enumerate(embeddings_all):
    cebra.plot_embedding(embedding, embedding_labels=cont_orig_values[i], idx_order=(0,1,2), markersize=5, alpha=0.5)



1/0

multi_cebra_model_creativity.save('saved_cebra_models/multi_cebra_model_creativity_single_unit.pt')


embeddings_single_unit = []
embeddings_single_unit_df = []
creativity_values_single_unit_list = []
for i, neural_data in enumerate(neural_data_list_fixed_originality):
    embedding = multi_cebra_model_creativity.transform(neural_data,session_id=i)
    embedding_df = pd.DataFrame(embedding, index = neural_data.index)
    embeddings_single_unit.append(embedding)
    embeddings_single_unit_df.append(embedding_df)
    creativity_values_single_unit_list.append(neural_data.index.get_level_values('originality_cont').values)

creativity_values_single_unit = np.concatenate(creativity_values_single_unit_list)

embed_df_single_unit = pd.concat(embeddings_single_unit_df)

model = LinearRegression()
model.fit(embed_df_single_unit, creativity_values_single_unit)

# Get the LFP coding axis 
coding_axis = model.coef_ / np.linalg.norm(model.coef_)

creativity_projection_single_unit = embed_df_single_unit @ coding_axis

f,ax = plt.subplots()
plt.scatter(creativity_projection_single_unit, creativity_values_single_unit)
plt.xlabel("Spike creativity projection")
plt.ylabel("Creativity Value")
plt.title("Coding Axis Projection vs. Creativity")
plt.show()


r2 = model.score(embed_df_single_unit, creativity_values_single_unit)


for patient in creativity_projection_single_unit.index.get_level_values('patient').unique():
    print(patient)
    f,ax = plt.subplots()
    projection = creativity_projection_single_unit.loc[patient]
    ax.plot(projection.values, color='k')
    for x in np.where(projection.index.get_level_values('bin').values==0)[0]:
        ax.axvline(x=x, color='k',linewidth = 0.5, linestyle='--') 
    ax.set_ylabel('Single Unit Projection onto LFP Coding Axis')
    ax.set_xlabel('time (s)')





1/0

###CODING AXIS


from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D



embed_df = pd.DataFrame(embeddings_concat)
creativity_values = np.concatenate(cont_orig_values)

# Assuming embed_df is your embeddings DataFrame and creativity_values is a NumPy array
model = LinearRegression()
model.fit(embed_df, creativity_values)

# Get the coding axis as a unit vector
coding_axis = model.coef_ / np.linalg.norm(model.coef_)


creativity_projection = embed_df @ coding_axis

f,ax = plt.subplots()
plt.scatter(creativity_projection, creativity_values)
plt.xlabel("Projection onto Coding Axis")
plt.ylabel("Creativity Value")
plt.title("Coding Axis Projection vs. Creativity")
plt.show()

coding_axis_3d = coding_axis[:3]  # Select the first 3 dimensions

min_proj, max_proj = creativity_projection.min(), creativity_projection.max()
center = embed_df.mean(axis=0).values  # Mean embedding
start_point = center + min_proj * coding_axis
end_point = center + max_proj * coding_axis

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with gradient colors
scatter = ax.scatter(embed_df.iloc[:, 0], embed_df.iloc[:, 1], embed_df.iloc[:, 2], 
                     c=creativity_values, cmap="cool", alpha=0.5)


# Draw the coding axis
ax.plot(
    [start_point[0], end_point[0]],
    [start_point[1], end_point[1]],
    [start_point[2], end_point[2]],
    color="black", linewidth=2, label="Coding Axis"
)

# Labels and color bar
ax.set_xlabel("Embedding Dim 1")
ax.set_ylabel("Embedding Dim 2")
ax.set_zlabel("Embedding Dim 3")
# ax.view_init(elev=40, azim=40)  # Change angles for a better view
plt.colorbar(scatter, label="Creativity Value")
ax.legend()
plt.title("3D Embeddings with Full Coding Axis")
plt.show()

r2 = model.score(embed_df, creativity_values)
# R² of Spiking Coding Axis: 0.54



