This is the code used for the paper "Originality in creative ideation evolves along nonlinear axes in the human brain".

Here are some notes about the included scripts: 

---

FUNCTIONS FILES AND OTHER USEFUL SCRIPTS (e.g., making and saving files to use elsewhere)

**patient_params.py**: Contains list of patient IDs and directories, referenced in all other code.

**beh_funcs.py**: Functions related to subject behavior. Used in a number of other scripts.

**DMN_LFP_funcs.py**: Functions related to LFP analysis. Used in most scripts. 

**decoding_funcs.py**: Functions to makes df_by_freq_band (LFP power df), which gets used to make the all_dat files (add_patient_to_dat). Used for decoding analysis, but other things too. make_power_df for Blackrock and and make_power_df_ripple for Ripple.

**connectivity_funcs.py**: Functions related to connectivity, power correlations, and spectral coherence (phase-phase locking). Used in connectivity_over_time.py (e.g., for Supp. Fig. 9)

**add_patient_to_dat_all_trialtypes.py**: Loops through patients, adds each patient's LFP power dataframe (separated into frequency bands) to all_dat, and saves CSV file for each network. When prompted, choose to run this script for a specific network, or type "all" to run all networks. 
Includes AUT (alternate uses), OCT (object characterization), and BAT (basic arithmetic) trial types.

**add_patient_to_dat.py**: Only includes AUT and BAT trial types. Prompt asks whether to include ripple recordings as well. (3 patients recorded with Ripple instead of Blackrock. They only did AUT and BAT.)

**plotting_funcs.py**: Occasionally used for plotting. (e.g., plot grid of embeddings).

**save_mne_file.py**: save MNE "raw.fif" files, from the ns2 Blackrock recordings. 

**save_mne_from_ripple.py**: same as above, but for Ripple recordings.

---

FIGURE 1 

**behavior_analysis.py**: Visualize subject behavior for AUT trials. For Figure 1 b and c.

**brain_vis_all_patients.py**: Generates gif of brain with electrode locations. For Figure 1d.

---

SPECTROGRAMS (FIGURE 2)

---

ANALYSIS OF COGNITIVE STATE (CREATIVITY VS. CONTROL TASKS) (FIGURE 3 and related supplementary figures, analyses, and stats)

**network_dynamics_AUT_BAT_include_ripple.py**: LFP power dynamics for high gamma and theta, comparing AUT and BAT. Includes Ripple patients. Makes Figure 3a.

**network_dynamics_AUT_BAT_OCT.py**: LFP power dynamics for high gamma and theta, comparing AUT and OCT. Also option to compare AUT, BAT, and OCT (excluding Ripple patients). Makes Figure 3b.

**network_dynamics_entire_block_with_ripple.py**: Makes Supp. Fig. 1a (similar to Fig. 3a, but covering entire block: all 3 trials in block). 

**network_dynamics_aligned_to_speech.py**: Makes Supp. Fig. 1b

**decoding_svm_AUT_BAT.py**: Linear decoding of creativity (AUT) vs. arithmetic control (BAT) trial types. Makes Figure 3c. (Also does linear decoding of high vs. low originality.)

**decoding_LDA_AUT_OCT.py**: Linear decoding of creativity (AUT) vs. object characterization control (OCT) trial types. Makes Figure 3d.

**cebra_cluster_networks.py**: Trains CEBRA model on AUT trials to separate networks. For Figure 3e and Supp. Fig. 3. 

**cebra_decoding_AUT_BAT_all_networks.py**: Trains CEBRA models to separate AUT and BAT trials. Makes Supp. Fig. 2d.

**cebra_decoding_AUT_OCT_all_networks.py**: Trains CEBRA models to separate AUT and OCT trials. Cross validated training, then decode from embeddings. Gets stats for this. 

**cebra_multi_AUT_BAT_OCT.py**: TRAIN MULTI SESSION MODEL; i.e., Pseudo-subject. Visualization of AUT vs. BAT vs. OCT as well as just AUT vs OCT embeddings (Supp. Fig. 4).

decoding_from_n_contacts.py: decode trial type from different numbers of contacts, to make sure that we're not getting better decoding in some patients/networks purely because of more contacts.

decoding_from_subset_contacts.py: decode from subsampled contacts. Get numner of DAN contacts, and subsample that many contacts from DMN and decode only from that number of DMN contacts

**cebra_time_all_networks.py**: Run self-supervised CEBRA; plot embeddings and color code by trial type (AUT vs BAT). Decode trial type, holding out by patient. Find coding axis and project. Makes Fig. 3f-h and Supp. Fig. 5.


---

ANALYSIS OF CREATIVITY LEVEL (FIGURE 4)

**network_states_ripple.py**: High vs. low originality states: finds signal power at different recording sites for high vs low creative states. Makes Figure 4a and Supp. Fig. 6a.

**cebra_decoding_creativity_all_networks.py**: Trains CEBRA models and decodes originality score. Makes Figure 4c.

**cebra_decoding_creativity_shuffle.py**: Train CEBRA on data with shuffled labels. Makes Supp Fig 6b.

**cebra_multi_creativity_axis_and_polynomial_regression.py**: Multi-session CEBRA to get creativity-level embedding. Makes creativity coding axis/projection, fitting LINEAR and then POLYNOMIAL regressions. Checks regression model performance for different polynomial degrees. Makes Fig. Fig. 4e-g; 4h-i (Linear regression) and Supp. Fig. 8a (polynomial regression). 



**connectivity_over_time.py**: Connectivity analysis: COHERENCE (PPL) across time. Mixed effects models and patient level analysis. Makes Supp. Fig. 9.


---

SINGLE UNIT ANALYSIS (FIGURE 5)

**make_spikes_dat.py**: makes and saves a spikes df so don't have to load NEV files each time. 

**single_units_psth_and_pie.py**: PSTHs (Figure 5a-d) and proportions pie chart (Figure 5e)

**single_unit_decoding_from_saved.py**: 1. SVM decode AUT vs BAT from single units. 2. CEBRA AUT vs BAT (Figure 5f-g). 3. CEBRA creativity level (Fig. 5j)

**single_unit_decode_AUT_OCT.py**: CEBRA AUT vs OCT (Figure 5h-i)

