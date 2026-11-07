'''Generate Brain images with electrodes visualized
    Adapted from Justin/Mark's InsulaVisBrain code

    Makes Figure 1d
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
import re
import mne
import scipy


import sys
sys.path.append('../')


import DMN_LFP_funcs
import beh_funcs
import os


from visbrain.objects import BrainObj, SceneObj, SourceObj
from visbrain.io import write_fig_canvas, mpl_preview, dialog_save


import sys
sys.path.append('../')

import patient_params
patients_info = patient_params.patients_data
patients_data = DMN_LFP_funcs.choose_patient(patients_info) #choose all or 1 patient

include_ripple = input("Include ripple patients (y/n): ")
if include_ripple=='y':
    patients_info_ripple = patient_params.patients_data_ripple
    patients_data_ripple = patients_info_ripple
    patients_data = patients_data + patients_data_ripple

xyz_all_patients = []
colors_all_patients = []
for patient, data_loc in patients_data:


    # patient_processed_AUT_datafile = 'scored_responses/' + patient + '_processed_data.csv'  #file that includes responses and originality scores:

    # raw = DMN_LFP_funcs.get_raw(patient,data_loc) #get raw mne (with already marked bad channels that were selected manually)

    # chans = raw.ch_names
    # fs = raw.info['sfreq'] #sampling freq
    # bads = raw.info['bads']

    # #match electrodes to regions
    matfile = DMN_LFP_funcs.get_matfile(patient, data_loc)
    remove_micros = DMN_LFP_funcs.remove_micros_bool(data_loc)

    chan_map_info = mat73.loadmat(matfile[0])
    regions = pd.DataFrame(chan_map_info['ElecAtlasProj'])[0]
    regions_list = [reg[0] if reg is not None else 'None' for reg in regions]
    networks = pd.DataFrame(chan_map_info['ElecAtlasProj'])[pd.DataFrame(chan_map_info['ElecAtlasProj']).columns.size-1]
    networks_list = [net[0] if net is not None else 'None' for net in networks]

    # good_chans_all, good_chans_DMN = DMN_LFP_funcs.get_chan_info(raw, chan_map_info, "DMN", remove_micros=remove_micros, remove_white=True)

    MNI_coords = pd.DataFrame(chan_map_info['ElecXYZMNIProj'])
    xyz_coords = pd.DataFrame(chan_map_info['ElecXYZProj'])

    # xyz = electrodesDF[['MNI_X', 'MNI_Y', 'MNI_Z']].to_numpy()
    xyz = chan_map_info['ElecXYZMNIProj']
    xyz_all_patients.extend(xyz)


    colors = ['#E4E4E4'] * len(xyz)
    # sizes = [20] * len(xyz)

    MTG_inds = np.where(['MTG' in name for name in regions_list])[0]
    DMN_inds = np.where(['DMN' in name for name in networks_list])[0]

    for i in DMN_inds:
        colors[i]='orange'

    colors_all_patients.extend(colors)

xyz_all_patients = np.array(xyz_all_patients)

xyz_L = xyz_all_patients[xyz_all_patients[:,0] < 0] #anything that is negative is left
colors_L = [colors_all_patients[i] for i in range(len(colors_all_patients)) if xyz_all_patients[i,0] < 0]
xyz_R =  xyz_all_patients[xyz_all_patients[:,0] > 0] #anything that is positive is right
colors_R = [colors_all_patients[i] for i in range(len(colors_all_patients)) if xyz_all_patients[i,0] > 0]


# xyz_L = xyz[xyz[:,0] < 0] #anything that is negative is left
# colors_L = [colors[i] for i in range(len(colors)) if xyz[i,0] < 0]
# xyz_R = xyz[xyz[:,0] > 0] #anything that is positive is right
# colors_R = [colors[i] for i in range(len(colors)) if xyz[i,0] > 0]

elec_size = 10
edge_width = 2
animate = False
save = True

# Scene object
scene_obj = SceneObj(bgcolor='white', size=(1000, 1000)) #create the scene i.e. background

'''# # Brain object(s) For two hemispheres'''
brain_obj_L = BrainObj('B2', translucent = True, hemisphere = 'left') # create left hemisphere
brain_obj_R = BrainObj('B2', translucent = True, hemisphere = 'right')
scene_obj.add_to_subplot(brain_obj_L, row = 0, col = 1) # add something to the scene i.e. left brain object
scene_obj.add_to_subplot(brain_obj_R, row = 0, col = 0)
# brain_obj_L.rotate(fixed = 'right') #rotate to correct orientation
# brain_obj_R.rotate(fixed = 'left')

brain_obj_L.rotate(fixed = 'left') #swap rotation direction so can see MTG
brain_obj_R.rotate(fixed = 'right')

#Source object(s) for two hemispheres
if len(xyz_L) > 0:
    iEEG_obj_L = SourceObj(name = 'iEEG', xyz = xyz_L, color = colors_L, radius_min = elec_size, edge_color = 'black', edge_width = edge_width) #if electrode on left add to left 
    scene_obj.add_to_subplot(iEEG_obj_L, row = 0, col = 1) #actually adding it to the image
if len(xyz_R) > 0:
    iEEG_obj_R = SourceObj(name = 'iEEG', xyz = xyz_R, color = colors_R, radius_min = elec_size, edge_color = 'black', edge_width = edge_width)
    scene_obj.add_to_subplot(iEEG_obj_R, row = 0, col = 0)

###VISUALIZE
scene_obj.preview()
# brain_obj_L.animate(step = 0.5) # video
# brain_obj_R.animate(step = 0.5) # video
# scene_obj.preview()
scene_obj.record_animation(os.path.join('iEEGModel.gif'), n_pic=100) # save video as gif


'''# # Single Brain Object'''
scene_obj = SceneObj(bgcolor='white', size=(1000, 1000)) #create the scene i.e. background
brain_obj = BrainObj('B2', translucent=True)
scene_obj.add_to_subplot(brain_obj, row=0, col=0)  # Add the brain object to the scene
brain_obj.rotate(fixed='left')  # Rotate to correct orientation
# Combine left and right hemisphere data
xyz_all = np.vstack((xyz_L, xyz_R))
colors_all = colors_L + colors_R

# Source objects for single brain
iEEG_obj = SourceObj(name='iEEG', xyz=xyz_all, color=colors_all, radius_min=elec_size, edge_color='black', edge_width=edge_width)
scene_obj.add_to_subplot(iEEG_obj, row=0, col=0)

brain_obj.animate(step=0.5)
scene_obj.preview()
scene_obj.record_animation('iEEGModel.gif', n_pic=100)


scene_obj.preview()


brain_obj_L.animate(step = 0.5) # video
brain_obj_R.animate(step = 0.5) # video
# scene_obj.preview()
scene_obj.record_animation(os.path.join('iEEGModel.gif'), n_pic=100) # save video as gif

