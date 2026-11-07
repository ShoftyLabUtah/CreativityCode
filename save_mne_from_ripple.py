#save MNE file ripple patients

import numpy as np
import fastNFxRead
import quantities as pq
import glob   
import matplotlib.pyplot as plt
import pickle

import re

import mne
import os

def save_mne_file(data_loc,patient):
    #ns2 files are the lfp data files for blackrock
    #nf3 are the files for ripple

    # NEV_file = glob.glob(data_loc + '/' + '*.nev')[0] #data file
    # ns5_file = glob.glob(data_loc + '/' + '*.ns5')[0] #
    nf3_file = glob.glob(data_loc + '/' + '*.nf3')[0] #

    print('loading data')


    header, data = fastNFxRead.fastNFxRead(nf3_file)

    ch_names = [str(ch) for ch in header['ChannelID']]

    sfreq = header['Fs']
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq)

    raw_standard = mne.io.RawArray(data.T, info)
    # raw_standard.load_data() #do I need this line? And why?
    raw_standard.plot(block=True)

    try:
        save_file = '/Volumes/155.100.91.44/Data/Rikki/' + patient + '/raw.fif'
        print('saving raw file: ' + save_file)
        raw_standard.save(save_file,overwrite=True)

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


