This is the code used for the paper "Originality in creative ideation evolves along nonlinear axes in the human brain".

Here are some notes about the included scripts: 

---

FUNCTIONS FILES AND OTHER USEFUL SCRIPTS (e.g., making and saving files to use elsewhere)

**beh_funcs.py**: Functions related to subject behavior. Used in a number of other scripts.

**decoding_funcs.py**: Functions to makes df_by_freq_band (LFP power df), which gets used to make the all_dat files (add_patient_to_dat). Used for decoding analysis, but other things too. make_power_df for Blackrock and and make_power_df_ripple for Ripple.

**connectivity_funcs.py**: Functions related to connectivity, power correlations, and spectral coherence (phase-phase locking). Used in connectivity_over_time.py (e.g., for Supp. Fig. 9)

**add_patient_to_dat_all_trialtypes.py**: Loops through patients, adds each patient's LFP power dataframe (separated into frequency bands) to all_dat, and saves CSV file for each network. When prompted, choose to run this script for a specific network, or type "all" to run all networks. 
Includes AUT (alternate uses), OCT (object characterization), and BAT (basic arithmetic) trial types.

**add_patient_to_dat.py**: Only includes AUT and BAT trial types. Prompt asks whether to include ripple recordings as well. (3 patients recorded with Ripple instead of Blackrock. They only did AUT and BAT.)

---

FIGURE 1 

**behavior_analysis.py**: Visualize subject behavior for AUT trials. For Figure 1 b and c.

**brain_vis_all_patients.py**: Generates gif of brain with electrode locations. For Figure 1d.

---

FIGURE 2: SPECTROGRAMS

---

FIGURE 3 (and related supplementary figures, analyses, and stats): ANALYSIS OF COGNITIVE STATE (CREATIVITY VS. CONTROL TASKS)

decoding_svm_AUT_BAT.py

decoding_LDA_AUT_OCT.py


**cebra_decoding_AUT_BAT_all_networks.py**: Trains CEBRA models to separate AUT and BAT trials. Makes Supp. Fig. 2d.

**cebra_decoding_AUT_OCT_all_networks.py**: Trains CEBRA models to separate AUT and OCT trials. Cross validated training, then decode from embeddings. Gets stats for this. 

**cebra_multi_AUT_BAT_OCT.py**: TRAIN MULTI SESSION MODEL; i.e., Pseudo-subject. Visualization of AUT vs. BAT vs. OCT as well as just AUT vs OCT embeddings (Supp. Fig. 4).

**cebra_cluster_networks.py**: Trains CEBRA model on AUT trials to separate networks. For Figure 3e and Supp. Fig. 3. 


---

**cebra_decoding_creativity_all_networks.py**: Trains CEBRA models and decodes originality score. Makes Figure 4c.

**cebra_decoding_creativity_shuffle.py**: Train CEBRA on data with shuffled labels. Makes Supp Fig 6b.


**cebra_multi_creativity_axis_and_polynomial_regression.py**: Multi-session CEBRA to get creativity-level embedding. Makes creativity coding axis/projection, fitting LINEAR and then POLYNOMIAL regressions. Checks regression model performance for different polynomial degrees. Makes Fig. Fig. 4e-g; 4h-i (Linear regression) and Supp. Fig. 8a (polynomial regression). 

**cebra_time_all_networks.py**: Run self-supervised CEBRA; plot embeddings and color code by trial type (AUT vs BAT). Decode trial type, holding out by patient. Find coding axis and project. Makes Fig. 3f-h and Supp. Fig. 5.

**connectivity_over_time.py**: Connectivity analysis: COHERENCE (PPL) across time. Mixed effects models and patient level analysis. Makes Supp. Fig. 9.

decoding_from_n_contacts.py: decode from different numbers of contacts, to make sure that we're not getting better decoding in some patients/networks purely because of more contacts.

decoding_from_subset_contacts.py: decode from subsampled contacts. Get numner of DAN contacts, and subsample that many contacts from DMN and decode only from that number of DMN contacts



