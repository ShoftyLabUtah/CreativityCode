
''' CEBRA

TRAIN MULTI SESSION MODEL; I.E. Pseudo-subject
creativity level (continuous label)
creativity coding axis/projection , fitting POLYNOMIAL regression

Checks regression model performance for different polynomial degrees

Makes Supp. Fig. 8a

'''

import numpy as np
# from neo import io
import quantities as pq
import glob   
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
# import itertools as it
# import random
# import re
import mne
import scipy

import DMN_LFP_funcs
import beh_funcs
import plotting_funcs


import sys
sys.path.append('../')

from mpl_helpers import PdfPlotter

import patient_params
patients_info = patient_params.patients_data
patients_data = DMN_LFP_funcs.choose_patient(patients_info) #choose all or 1 patient



# network = input("Choose cortical network (DMN, FrontoParietal, SomatoMotor, Limbic, Visual, DorsalAttention, VentralAttention): ")

import cebra
from cebra import CEBRA


import cebra.models

embeddings_all_nets = {}
creativity_values_all_nets = {}
for network in ['DMN','DorsalAttention']:
    # all_dat = pd.read_csv('pkl_files/store_neural_data_list_' + network + '.csv', header=[0, 1], index_col=[0, 1, 2, 3])
    all_dat = pd.read_csv('pkl_files/store_neural_data_list_with_ripple' + network + '.csv', header=[0, 1], index_col=[0, 1, 2, 3])



    try:
        all_dat.drop('UIC202420', inplace = True)
    except:
        print('not dropping doesnt exist')
    try:    
        all_dat.drop('UIC202422', inplace = True)
    except:
        print('not dropping doesnt exist')


    patient_dfs = [group for _, group in all_dat.groupby(level='patient')]
    cleaned_dfs = [df.dropna(axis=1, how='all') for df in patient_dfs]

    neural_data_list_fixed = cleaned_dfs

    # discrete_labels = [neural_data.index.get_level_values('trial_type') for neural_data in neural_data_list_fixed]
    discrete_labels = [1*(neural_data.index.get_level_values('trial_type')=='AUT') for neural_data in neural_data_list_fixed] 




    '''MULTI SESSION MODEL: CREATIVITY LEVEL'''

    neural_data_list_fixed_AUT = [neural_data.iloc[np.where(neural_data.index.get_level_values('trial_type')=='AUT')[0]] for neural_data in neural_data_list_fixed]
    neural_data_list_fixed_originality = []

    for p, df in enumerate(neural_data_list_fixed_AUT):
        patient = neural_data_list_fixed_AUT[p].index.get_level_values('patient').unique().values[0] #get patient from df because out of order
        
        
        patient_processed_AUT_datafile = 'scored_responses/' + patient + '_processed_data.csv'  #file that includes responses and originality scores
        if patient=='UIC202422b': #ripple recording from this pt (day 1)
            patient_processed_AUT_datafile = 'scored_responses/UIC202422_processed_data_day1.csv'  #file that includes responses and originality scores:
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

    ####  try normalizing the originality scores
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    norm_orig_values = [scaler.fit_transform((vals).reshape(-1, 1)).flatten() for vals in cont_orig_values]
    # cont_orig_values = norm_orig_values


    multi_cebra_model_creativity = cebra.CEBRA(model_architecture='offset10-model',
                            batch_size=512,
                            learning_rate=3e-4,
                            temperature=1,
                            output_dimension=5,
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




    ######  coding axis
    from sklearn.linear_model import LinearRegression
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    embed_df = pd.DataFrame(embeddings_concat)
    creativity_values = np.concatenate(cont_orig_values)

    embeddings_all_nets.update({network: embed_df})
    creativity_values_all_nets.update({network: creativity_values})

PdfPlotter('Figures/cebra_poly_regression_different_degrees.pdf', fixed_margins = dict(left=110, right=90, top=90, bottom=90))

####### POLYNOMIAL REGRESSION ########
f1,ax1 = plt.subplots()
f2,ax2 = plt.subplots()

colors = ['orange', 'purple']
for k, key in enumerate(creativity_values_all_nets.keys()):
    fits = []
    corrs = []
    for degree in np.arange(1,12):
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(embeddings_all_nets[key], creativity_values_all_nets[key])
        # Compute the coding axis as a function
        poly_creativity_projection = model.predict(embeddings_all_nets[key])
        r2 = model.score(embeddings_all_nets[key], creativity_values_all_nets[key]) #DMN: 0.84        norm: DMN: 0.908; DA: 0.753
        corr = np.corrcoef(poly_creativity_projection, creativity_values_all_nets[key]) #DMN: 0.92         norm: DMN: 0.953; DA: 0.868
        fits.append(r2)
        corrs.append(corr[0, 1])


    ax1.plot(np.linspace(1, 11, 11), fits, color=colors[k])
    ax2.plot(np.linspace(1, 11 , 11), corrs, color=colors[k])
ax1.set_title('regression goodness of fit')
ax2.set_title('correlation')
plt.show()


###### note: for DA r2 to be as good as DMN, polynomial degree has to be >8! 

###### plot creativity projection vs real values
f,ax = plt.subplots()
plt.scatter(poly_creativity_projection, creativity_values)
plt.xlabel("Projection onto Polynomial Coding Axis")
plt.ylabel("Creativity Value")
plt.title("Nonlinear Coding Axis Projection vs. Creativity")
plt.show()

###### Plot embeddings with coding axis

plotting_funcs.plot_polynomial_coding_axis(embed_df, creativity_values, 
                dimensions=multi_cebra_model_creativity.output_dimension, idx_order=[0,1,2], angle=[35,-55])



1/0


#only compare patients with coverage both in DA and DMN
all_dat_DA = pd.read_csv('pkl_files/store_neural_data_list_with_rippleDorsalAttention.csv', header=[0, 1], index_col=[0, 1, 2, 3])
all_dat_DMN = pd.read_csv('pkl_files/store_neural_data_list_with_rippleDMN.csv', header=[0, 1], index_col=[0, 1, 2, 3])
patients_subset = all_dat_DA.index.get_level_values('patient').unique().values
all_dat = all_dat_DMN.loc[patients_subset]