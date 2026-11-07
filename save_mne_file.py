#save MNE file (do not pickle
import numpy as np
from neo import io
import fastNFxRead


import quantities as pq
import glob   
import matplotlib.pyplot as plt
import pickle

import re

import mne
import os

import sys
sys.path.append('../')

import brpylib

def save_mne_raw(data_loc,patient):
    #ns2 files are the lfp data files for blackrock
    #nf3 are the files for ripple

    # NEV_file = glob.glob(data_loc + '/' + '*.nev')[0] #data file
    # ns5_file = glob.glob(data_loc + '/' + '*.ns5')[0] #

    # if ((patient=='UIC202521') | (patient=='UIC202522')):
    #     ns2_file = glob.glob(data_loc + '/Hub1' + '*.ns2')[0] #
    # else:
    #     ns2_file = glob.glob(data_loc + '/' + '*.ns2')[0] #
    ns2_file = glob.glob(data_loc + '/' + '*.ns2')[0] #

    print('loading data')
    read_ns2 = io.BlackrockIO(ns2_file)
    block = read_ns2.read_block()
    # read_nev = io.BlackrockIO(NEV_file)
    # block = read_nev.read_block()
    seg = block.segments[0]
    data = seg.analogsignals[0].T
    ch_names = re.findall(r'\d+', seg.analogsignals[0].name)

    sfreq = int(seg.analogsignals[0].sampling_rate.magnitude)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq)

    raw_standard = mne.io.RawArray(np.array(data), info)
    # raw_standard.load_data() #do I need this line? And why?
    raw_standard.plot(block=True)

    try:
        save_file = '/Volumes/155.100.91.44/Data/Rikki/' + patient + '/raw.fif'
        print('saving raw file: ' + save_file)
        raw_standard.save(save_file, overwrite=True)

        #save locally too:
        save_file_local = 'mne_files/'+patient + '/raw.fif'
        print('saving raw file, locally: ' + save_file_local)
        raw_standard.save(save_file_local,overwrite=True)

    except: #no folder yet
        print('making patient folder: ' + patient)
        os.mkdir('/Volumes/155.100.91.44/Data/Rikki/' + patient + '/') 
        save_file = '/Volumes/155.100.91.44/Data/Rikki/' + patient + '/raw.fif'
        print('saving raw file: ' + save_file)
        raw_standard.save(save_file)

        #save locally too:
        print('making local patient folder: ' + patient)
        os.mkdir('mne_files/' + patient + '/')         
        save_file_local = 'mne_files/'+patient + '/raw.fif'
        print('saving raw file, locally: ' + save_file_local)
        raw_standard.save(save_file_local)


    # info_for_seeg_raw_file = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='seeg')

    return(raw_standard)




def save_mne_raw_gemini(data_loc,patient):
    #ns2 files are the lfp data files for blackrock
    #nf3 are the files for ripple

    # NEV_file = glob.glob(data_loc + '/' + '*.nev')[0] #data file
    # ns5_file = glob.glob(data_loc + '/' + '*.ns5')[0] #

    # if ((patient=='UIC202521') | (patient=='UIC202522')):
    #     ns2_file = glob.glob(data_loc + '/Hub1' + '*.ns2')[0] #
    # else:
    #     ns2_file = glob.glob(data_loc + '/' + '*.ns2')[0] #
    ns2_file = glob.glob(data_loc + '/Hub1' + '*.ns2')[0] #

    print('loading data')

    ns2_file_loaded = brpylib.NsxFile(ns2_file)
    ns2_data = ns2_file_loaded.getdata()
    timestamp = ns2_data['data_headers'][0]['Timestamp']

    ch_names = ns2_data['elec_ids']
    sfreq = ns2_data['samp_per_s']

    info = mne.create_info(ch_names=list(np.array(ch_names).astype(str)), sfreq=sfreq)

    raw_standard = mne.io.RawArray(np.array(ns2_data['data'][0]), info)


    # raws = []

    # for seg in block.segments:
    #     sig = seg.analogsignals[0]
    #     data = np.asarray(sig).T

    #     ch_names = [name.split('chan')[-1] for name in sig.array_annotations['channel_names']]

    #     ch_names = sig.array_annotations['channel_names'].tolist()

    #     sfreq = int(sig.sampling_rate.magnitude)

    #     info = mne.create_info(ch_names=ch_names, sfreq=sfreq)
    #     raws.append(mne.io.RawArray(data, info))

    # raw_standard = mne.concatenate_raws(raws)

    raw_standard.plot(block=True)

    try:
        save_file = '/Volumes/155.100.91.44/Data/Rikki/' + patient + '/raw.fif'
        print('saving raw file: ' + save_file)
        raw_standard.save(save_file, overwrite=True)

        #save locally too:
        save_file_local = 'mne_files/'+patient + '/raw.fif'
        print('saving raw file, locally: ' + save_file_local)
        raw_standard.save(save_file_local,overwrite=True)

    except: #no folder yet
        print('making patient folder: ' + patient)
        os.mkdir('/Volumes/155.100.91.44/Data/Rikki/' + patient + '/') 
        save_file = '/Volumes/155.100.91.44/Data/Rikki/' + patient + '/raw.fif'
        print('saving raw file: ' + save_file)
        raw_standard.save(save_file)

        #save locally too:
        print('making local patient folder: ' + patient)
        os.mkdir('mne_files/' + patient + '/')         
        save_file_local = 'mne_files/'+patient + '/raw.fif'
        print('saving raw file, locally: ' + save_file_local)
        raw_standard.save(save_file_local)


    # info_for_seeg_raw_file = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='seeg')

    return(raw_standard)




# ns5_pkl_file = '/Volumes/155.100.91.44/Data/shoftyBaylorDMN/DMN_micro/pkl_files/store_' + patient + '_' + num + '_ns5_info.pkl'
# with open(ns5_pkl_file, 'wb') as f:
#     pickle.dump(block_ns5.segments[0], f)
photo_trace = block_ns5.segments[0].analogsignals[2].as_array()[:,0] #photodiode trace
phototrace_pkl_file = 'store_' + patient + '_' + num + '_phototrace.pkl'

def save_the_file_from_baylor(patient, data_file):

    read_file = io.BlackrockIO(data_file)
    block = read_file.read_block()
    # read_nev = io.BlackrockIO(NEV_file)
    # block = read_nev.read_block()
    seg = block.segments[0]
    data = seg.analogsignals[0].T
    ch_names = [str(c) for c in range(len(data))]

    sfreq = int(seg.analogsignals[0].sampling_rate.magnitude)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq)

    raw_standard = mne.io.RawArray(np.array(data), info)
    raw_standard.load_data()
    raw_standard.plot()

    try:
        save_file = '/Volumes/155.100.91.44/Data/Rikki/' + patient + '/raw.fif'
        print('saving raw file: ' + save_file)

        raw_standard.save(save_file)
    except: #no folder yet
        print('making patient folder: ' + patient)
        os.mkdir('/Volumes/155.100.91.44/Data/Rikki/' + patient + '/') 
        save_file = '/Volumes/155.100.91.44/Data/Rikki/' + patient + '/raw.fif'
        print('saving raw file: ' + save_file)
        raw_standard.save(save_file)

    info_for_seeg_raw_file = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='seeg')

    return(raw_standard)
