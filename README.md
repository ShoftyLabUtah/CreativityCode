This is the code used for the paper "Originality in creative ideation evolves along nonlinear axes in the human brain".

Here are some notes about the included scripts: 

**beh_funcs.py**: 
Functions related to subject behavior. Used in a number of other scripts.

behavior_analysis.py: 
Visualize subject behavior for AUT trials. For Figure 1 b and c.

**add_patient_to_dat_all_trialtypes.py**: Loops through patients, adds each patient's LFP power dataframe (separated into frequency bands) to all_dat, and saves CSV file for each network.
Includes AUT (alternate uses), OCT (object characterization), and BAT (basic arithmetic) trial types.

**add_patient_to_dat.py**: Only includes AUT and BAT trial types. Allows for adding ripple recordings as well. (3 patients recorded with Ripple instead of Blackrock. They only did AUT and BAT.)

**brain_vis_all_patients.py**: Generates gif of brain with electrode locations. For Figure 1d.

**cebra_cluster_networks.py**: Trains CEBRA model on AUT trials to separate networks. For Figure 3e and Supp. Fig. 3. 

**cebra_decoding_AUT_OCT_all_networks.py**: Trains CEBRA models to separate AUT and OCT trials. Cross validated training, then decode from embeddings. Gets stats for this. 

**cebra_decoding_AUT_BAT_all_networks.py**: Trains CEBRA models to separate AUT and BAT trials. Makes Supp. Fig. 2d.

**cebra_decoding_creativity_all_networks.py**: Trains CEBRA models and decodes originality score. Makes Figure 4c.

**cebra_decoding_creativity_shuffle.py**: Train CEBRA on data with shuffled labels. Makes Supp Fig 6b.

**cebra_multi_AUT_BAT_OCT.py**: TRAIN MULTI SESSION MODEL; i.e., Pseudo-subject. Visualization of AUT vs. BAT vs. OCT as well as just AUT vs OCT embeddings (Supp. Fig. 4).

**cebra_multi_creativity_axis_and_polynomial_regression.py**: Multi-session CEBRA to get creativity-level embedding. Makes creativity coding axis/projection, fitting LINEAR and then POLYNOMIAL regressions. Checks regression model performance for different polynomial degrees. Makes Fig. Fig. 4e-g; 4h-i (Linear regression) and Supp. Fig. 8a (polynomial regression). 
