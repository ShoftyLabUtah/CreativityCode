# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 08:01:36 2023

@author: Tyler

To run, add fastNFxRead.py to the python path (either set console working directory to 
file location or sys.path.append(pathtofile)). Input is a string of the full 
path to the nfx file.

1) import fastNFxRead as fr
2) H, D = fr.fastNFxRead(NFxFile)

"""

import struct
import numpy as np
import os

def fastNFxRead(File=''):
    # breakpoint()
    
    if not os.path.exists(File):
        print('File does not exist!')
        return
    
    FID = open(File,'rb')
    
    H = {}
    
    H['FileID'] = FID.read(8).decode('utf-8')
    H['FileSpec'] = struct.unpack('bb',FID.read(2))
    H['HeaderBytes'] = struct.unpack('I',FID.read(4))[0]
    H['FsStr'] = FID.read(16).decode('utf-8')
    H['Comment'] = FID.read(252).decode('utf-8')
    H['NIPStart'] = struct.unpack('I',FID.read(4))[0]
    H['Period'] = struct.unpack('I',FID.read(4))[0]
    H['Resolution'] = struct.unpack('I',FID.read(4))[0]
    H['TimeOrigin'] = struct.unpack('8H',FID.read(16))
    H['ChannelCount'] = struct.unpack('I',FID.read(4))[0]
    H['Fs'] = np.double(H['Resolution'])/np.double(H['Period'])
    
    H['Type'] = []; H['ChannelID'] = []; H['ChannelLabel'] = []; H['PhysConnector'] = []
    H['ConnectorPin'] = []; H['MinDigVal'] = []; H['MaxDigVal'] = []; H['MinAnlgVal'] = []
    H['MaxAnlgVal'] = []; H['Units'] = []; H['HighFreqCorner'] = []; H['HighFreqOrder'] = []
    H['HighFiltType'] = []; H['LowFreqCorner'] = []; H['LowFreqOrder'] = []; H['LowFiltType'] = []
    for k in range(H['ChannelCount']):
        H['Type'].append(FID.read(2).decode('utf-8'))
        H['ChannelID'].append(struct.unpack('H',FID.read(2))[0])
        H['ChannelLabel'].append(FID.read(16).decode('utf-8'))
        H['PhysConnector'].append(struct.unpack('B',FID.read(1))[0])
        H['ConnectorPin'].append(struct.unpack('B',FID.read(1))[0])
        H['MinDigVal'].append(struct.unpack('h',FID.read(2))[0])
        H['MaxDigVal'].append(struct.unpack('h',FID.read(2))[0])
        H['MinAnlgVal'].append(struct.unpack('h',FID.read(2))[0])
        H['MaxAnlgVal'].append(struct.unpack('h',FID.read(2))[0])
        H['Units'].append(FID.read(16).decode('utf-8'))
        H['HighFreqCorner'].append(struct.unpack('I',FID.read(4))[0])
        H['HighFreqOrder'].append(struct.unpack('I',FID.read(4))[0])
        H['HighFiltType'].append(struct.unpack('H',FID.read(2))[0])
        H['LowFreqCorner'].append(struct.unpack('I',FID.read(4))[0])
        H['LowFreqOrder'].append(struct.unpack('I',FID.read(4))[0])
        H['LowFiltType'].append(struct.unpack('H',FID.read(2))[0])
        
    BegOfDataHeader = FID.tell()
    FID.seek(0, os.SEEK_END)
    EndOfFile = FID.tell()
    FID.seek(BegOfDataHeader)
    
    DataHeader = struct.unpack('B',FID.read(1))[0] #always 1
    DataTimestamp = struct.unpack('I',FID.read(4))[0]
    ChannelSamples = struct.unpack('I',FID.read(4))[0]
    BegOfData = FID.tell()
    
    if ((EndOfFile-BegOfData)/4!=ChannelSamples*H['ChannelCount']):
        print('There is a pause in this data. Only loading data before the pause.')
 
    
    #This is much faster than struct.unpack! Consider replacing above...
    D = np.frombuffer(FID.read(ChannelSamples*H['ChannelCount']*4), 
                       dtype=np.float32).reshape((-1,H['ChannelCount'])) 
        
        
    FID.close()
    
    return H, D


    
    
if __name__ == '__main__':
    
    print('This code must be run within a python console!')