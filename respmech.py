#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:11:39 2019

@author: Emil S. Walsted 
------------------------------------------------------------------------ 
Respiratory mechanics and work of breathing calculation
------------------------------------------------------------------------ 
(c) Copyright 2019 Emil Schwarz Walsted <emilwalsted@gmail.com>
This is a rewrite of my MATLAB code from 2016, with some changes/additions.
The latest version of this code is available at GitHub: 
https://github.com/emilwalsted/respmech

This code calculates respiratory mechanics and/or in- and expiratory work of 
breathing and/or diaphragm EMG entropy from a time series of 
pressure/flow/volume recordings obtained  e.g. from LabChart.

Currently, supported input formats are: MATLAB, Excel and CSV. You can extend
this yourself by modifying the 'load()' function.

For usage, please see the example file 'example.py'.

*** Note to respiratory scientists: ***
I created this code for the purpose of my own 
work and shared it hoping that someone else might find it useful. If you have 
any suggestions that will make the code more useful to you or generally, please 
email me to discuss.

*** Note to software engineers/computer scientists: ***
I realise that this code is not as concise or computationally efficient as it 
could be. I have focused on readability in an attempt to enable respiratory 
scientists with basic programming skills to understand, debug and modify/extend
the code themselves.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
 
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
"""

VERSION = "0.9.5"
CREATED = "Created with RespMech.py version " + VERSION + " (www.github.com/emilwalsted/respmech)"

import os
import glob
import ntpath
import math
import datetime
from os.path import join as pjoin
import numpy as np
from pandas.core.construction import array 
import scipy as sp
import scipy.io as sio  
from collections import OrderedDict 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Polygon, Rectangle
import json
from types import SimpleNamespace
from collections import namedtuple

import seaborn as sns; sns.set()
plt.style.use('seaborn-white')

def import_file(full_name, path):
    from importlib import util

    spec = util.spec_from_file_location(full_name, path)
    mod = util.module_from_spec(spec)

    spec.loader.exec_module(mod)
    return mod

def checknotset(setting, text, validate):
    if setting is None or len(s) == 0: raise ValueError(text)

def validatesettings(s):
    checknotset(setting=s.input.inputfolder, text="Input folder not specified in settings")
    checknotset(s.input.files, "Input file(s) not specified in settings")
    checknotset(s.input.format.samplingfrequency, "Sampling frequency not specified in settings")
    checknotset(s.input.format.matlabfileformat, "MatLab file format not specified in settings")
   
    checknotset(s.input.data,column_flow, "Flow data column not specified in settings")
    
    checknotset(s.input.data,column_poes, "Oesophageal pressure data column not specified in settings")
    checknotset(s.input.data,column_pgas, "Gastric pressure data column not specified in settings")
    checknotset(s.input.data,column_pdi, "PDI data column not specified in settings")

    if not settings.processing.mechanics.integratevolumefromflow:
        checknotset(s.input.data,column_volume, "Volume data column not specified in settings – please set 'integratevolumefromflow' to True to allow for automated volume integration.")
    
#TODO:
#     "processing": {
#         "mechanics": {
#             "breathseparationbuffer": 800,

#             "volumetrendadjustmethod": "linear",
#             "excludebreaths": [],
#             "breathcounts": []
#         },
#         "wob": {
#             "calcwobfrom": "average",
#             "avgresamplingobs": 500
#         },
#         "emg": {
#             "windowsize": 0.4, 
#             "remove_noise": false, 
#             "save_sound": false
#         },
#         "entropy": {
#             "entropy_epochs": 2,
#             "entropy_tolerance": 0.1
#         }
#     },
#     "output": {
#         "outputfolder": "/Users/emilnielsen/Documents/Medicin/Forskning/Code/Respiratory mechanics/test/output",
#         "data": {
#             "saveaveragedata": true,
#             "savebreathbybreathdata": true,
#             "saveprocesseddata": false,
#             "includeignoredbreaths": true
#         },
#         "diagnostics": {
#             "savepvaverage": true,
#             "savepvoverview": true,
#             "savepvindividualworkload": true,
#             "savedataviewraw": true,
#             "savedataviewtrimmed": true,
#             "savedataviewdriftcor": true
#         }
#     }
# }

def validatedata(flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns):
    #TODO
    print("")

def load(filepath, settings):
    
    _, fext = os.path.splitext(filepath)
    
    def loadmat(path):
        if settings.input.format.matlabfileformat == 2:
            flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns = loadmatmac(filepath)
        else:
            flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns = loadmatwin(filepath)               
        return flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns

    def loadmatmac(filename):
        rawdata = sio.loadmat(filename)
        data = OrderedDict(rawdata)
        cols = list(data.items())
        
        flow = cols[settings.input.data.column_flow-1][1]
        if np.isnan(settings.input.data.column_volume):
            volume = []
        else:
            volume = cols[settings.input.data.column_volume-1][1]
        poes = cols[settings.input.data.column_poes-1][1]
        pgas = cols[settings.input.data.column_pgas-1][1]
        pdi = cols[settings.input.data.column_pdi-1][1]
        
        if len(settings.input.data.columns_entropy)==0:
            entropycolumns = []
        else:
            entropycolumns = np.empty([len(cols[settings.input.data.columns_entropy[0]][1]), len(settings.input.data.columns_entropy)])
            for i in range(0, len(settings.input.data.columns_entropy)):
                entropycolumns[:,i] = cols[settings.input.data.columns_entropy[i]-1][1].squeeze()

        if len(settings.input.data.columns_emg)==0:
            emgcolumns = []
        else:
            emgcolumns = np.empty([len(cols[settings.input.data.columns_emg[0]][1]), len(settings.input.data.columns_emg)])
            for i in range(0, len(settings.input.data.columns_emg)):
                emgcolumns[:,i] = cols[settings.input.data.columns_emg[i]-1][1].squeeze()
                        
        return flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns
        
    def loadmatwin(filename):
        rawdata = sio.loadmat(filename)
        data = OrderedDict(rawdata)
        cols = list(data["data_block1"])
        
        flow = cols[settings.input.data.column_flow-1]
        if np.isnan(settings.input.data.column_volume):
            volume = []
        else:
            volume = cols[settings.input.data.column_volume-1]
        poes = cols[settings.input.data.column_poes-1]
        pgas = cols[settings.input.data.column_pgas-1]
        pdi = cols[settings.input.data.column_pdi-1]
        
        if len(settings.input.data.columns_entropy)==0:
            entropycolumns = []
        else:
            entropycolumns = np.empty([len(cols[settings.input.data.columns_entropy[0]]), len(settings.input.data.columns_entropy)])
            for i in range(0, len(settings.input.data.columns_entropy)):
                entropycolumns[:,i] = cols[settings.input.data.columns_entropy[i]-1]
        
        if len(settings.input.data.columns_emg)==0:
            emgcolumns = []
        else:
            emgcolumns = np.empty([len(cols[settings.input.data.columns_entcolumns_emgropy[0]]), len(settings.input.data.columns_emg)])
            for i in range(0, len(settings.input.data.columns_emg)):
                emgcolumns[:,i] = cols[settings.input.data.columns_emg[i]-1]
        
        return flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns
    
    def loadxls(filename):
        data = pd.read_excel(filename)
        
        flow = data.iloc[:,settings.input.data.column_flow-1].to_numpy()
        if np.isnan(settings.input.data.column_volume):
            volume = []
        else:
            volume = data.iloc[:,settings.input.data.column_volume-1].to_numpy()
        poes = data.iloc[:,settings.input.data.column_poes-1].to_numpy()
        pgas = data.iloc[:,settings.input.data.column_pgas-1].to_numpy()
        pdi = data.iloc[:,settings.input.data.column_pdi-1].to_numpy()
        
        if len(settings.input.data.columns_entropy)==0:
            entropycolumns = []
        else:
            entropycolumns = data.iloc[:,np.array(settings.input.data.columns_entropy)-1].to_numpy()

        if len(settings.input.data.columns_emg)==0:
            emgcolumns = []
        else:
            emgcolumns = data.iloc[:,np.array(settings.input.data.columns_emg)-1].to_numpy()
        
        return flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns

    def loadcsv(filename):
        data = pd.read_csv(filename)
        
        flow = data.iloc[:,settings.input.data.column_flow-1].to_numpy()
        if np.isnan(settings.input.data.column_volume):
            volume = []
        else:
            volume = data.iloc[:,settings.input.data.column_volume-1].to_numpy()
        poes = data.iloc[:,settings.input.data.column_poes-1].to_numpy()
        pgas = data.iloc[:,settings.input.data.column_pgas-1].to_numpy()
        pdi = data.iloc[:,settings.input.data.column_pdi-1].to_numpy()
        
        if len(settings.input.data.columns_entropy)==0:
            entropycolumns = []
        else:
            entropycolumns = data.iloc[:,np.array(settings.input.data.columns_entropy)-1].to_numpy().squeeze()

        if len(settings.input.data.columns_emg)==0:
            emgcolumns = []
        else:
            emgcolumns = data.iloc[:,np.array(settings.input.data.columns_emg)-1].to_numpy().squeeze()
        
        return flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns

    def loadtxt(filename):
        data = pd.read_csv(filename, sep='\t', decimal=settings.input.format.decimalcharacter)
        
        flow = data.iloc[:,settings.input.data.column_flow-1].to_numpy()
        if np.isnan(settings.input.data.column_volume):
            volume = []
        else:
            volume = data.iloc[:,settings.input.data.column_volume-1].to_numpy()
        poes = data.iloc[:,settings.input.data.column_poes-1].to_numpy()
        pgas = data.iloc[:,settings.input.data.column_pgas-1].to_numpy()
        pdi = data.iloc[:,settings.input.data.column_pdi-1].to_numpy()
        
        if len(settings.input.data.columns_entropy)==0:
            entropycolumns = []
        else:
            entropycolumns = data.iloc[:,np.array(settings.input.data.columns_entropy)-1].to_numpy().squeeze()

        if len(settings.input.data.columns_emg)==0:
            emgcolumns = []
        else:
            emgcolumns = data.iloc[:,np.array(settings.input.data.columns_emg)-1].to_numpy().squeeze()
        
        return flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns
    
    def loadfext(x):
        return {
            #Add new file extensions and a handler function here.
            '.xls': loadxls,
            '.xlsx': loadxls,
            '.csv': loadcsv,
            '.mat': loadmat,
            '.txt': loadtxt
        }.get(x) #, default)
    
    flow = []
    volume = []
    poes = []
    pgas = []
    pdi = []
    
    loadfunc = loadfext(fext)
    flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns = loadfunc(filepath)
    flow = flow.squeeze()
    if len(volume)>0:
        volume = volume.squeeze()
    poes = poes.squeeze()
    pgas = pgas.squeeze()
    pdi = pdi.squeeze()
    
    if settings.processing.mechanics.inverseflow:
        flow = -flow
    
    if settings.processing.mechanics.integratevolumefromflow:
        xval = np.linspace(0, len(flow)/settings.input.format.samplingfrequency, len(flow))
        volume = np.concatenate([-sp.integrate.cumtrapz(flow, xval), [0.0000001]])

    if settings.processing.mechanics.inversevolume:
        volume = -volume
            
    return flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns

def zero(indata):
    return indata-(indata[0])

def correctdrift(volume, settings):
    xno = len(volume)-1
      
    a = ((volume[xno])-volume[0])/xno

    val = a
    corvol=np.zeros(len(volume))
    for i in range(0, xno):
        corvol[i] = volume[i] + val
        val = val - a
        
    return corvol

def correcttrend(title, volume, settings):
    
    from scipy import signal
    
    vol = volume.squeeze()
    peaks = signal.find_peaks((vol*-1)+max(vol), height=settings.processing.mechanics.volumetrendpeakminheight, distance=settings.processing.mechanics.volumetrendpeakmindistance * settings.input.format.samplingfrequency)[0]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(21,29.7))
    title = title + " - Volume trend adjustment (" + settings.processing.mechanics.volumetrendadjustmethod + ")"
    plt.suptitle(title, fontsize=48)
    
    f = sp.interpolate.interp1d(peaks, vol[peaks], settings.processing.mechanics.volumetrendadjustmethod, fill_value="extrapolate")
    peaksresampled = f(np.linspace(0, vol.size-1, vol.size))
            
    axes[0].plot(volume)
    axes[0].plot(peaks, vol[peaks], "x", markersize=20)
    
    axes[1].plot(volume, linewidth=1.5)
    axes[1].plot(peaksresampled, linewidth=1.5)
    
    corvol = volume - peaksresampled
            
    axes[2].plot(corvol)
    axes[2].plot(np.zeros(corvol.size), '--', linewidth=1.5)
    
    savefile = pjoin(settings.output.outputfolder, "plots", title + ".pdf")
    plt.figtext(0.99, 0.01, CREATED + " on " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), horizontalalignment='right')       
    fig.savefig(savefile) 
    
    
    return corvol

def trim(timecol, flow, volume, poes, pgas, pdi, emgcolumns, settings):
    startix = np.argmax(flow <= 0)
    
    posflow = np.argwhere(flow >= 0)
    endix = posflow[:,0][len(posflow[:,0])-1]
    
    return timecol[startix:endix], flow[startix:endix], volume[startix:endix], poes[startix:endix], pgas[startix:endix], pdi[startix:endix], emgcolumns[startix:endix]

def ignorebreaths(curfile, settings):
    allignore = settings.processing.mechanics.excludebreaths
    d = dict(allignore)
    if curfile in d:
        return d[curfile]
    else:
        return []

   
def separateintobreathsbyflow(filename, timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns, settings):
    breaths = OrderedDict()
    j = len(flow)
    bufferwidth = settings.processing.mechanics.breathseparationbuffer

    ib = ignorebreaths(filename, settings)
    
    i = 0
    breathno = 0
    breathcnt = 0
    while i<j:
        breathcnt += 1
        instart = i
        while (i<j and ((flow[i]<0)  or (np.mean(flow[i:min(j,i+bufferwidth)])<0))):
            i += 1
        inend = i-1
        
        exstart = i
        while (i<j and ((flow[i]>0) or (np.mean(flow[i:min(j,i+bufferwidth)])>0))):
            i += 1
        exend = i-1
        exend = min(exend, j)
        
        exp = {'time':timecol[exstart:exend].squeeze(), 'flow':flow[exstart:exend].squeeze(), 'poes':poes[exstart:exend].squeeze(),
                   'pgas': pgas[exstart:exend].squeeze(), 'pdi':pdi[exstart:exend].squeeze(), 'volume':volume[exstart:exend].squeeze()}
        
        insp = {'time':timecol[instart:inend].squeeze(), 'flow':flow[instart:inend].squeeze(), 'poes':poes[instart:inend].squeeze(), 
                    'pgas':pgas[instart:inend].squeeze(), 'pdi':pdi[instart:inend].squeeze(), 'volume':volume[instart:inend].squeeze()}
            
        if breathcnt in ib:
            ignored = True
        else:
            breathno += 1
            ignored = False
            
        entlen = exend-instart
        if len(entropycolumns) > 0:
            entcols = np.zeros([entlen, entropycolumns.shape[1]])
            for ix in range(0, entropycolumns.shape[1]):      
                entcols[:,ix] = entropycolumns[instart:exend,ix]
        else:
            entcols = []

        if len(emgcolumns) > 0:
            emgcols = np.zeros([entlen, emgcolumns.shape[1]])
            for ix in range(0, emgcolumns.shape[1]):      
                emgcols[:,ix] = emgcolumns[instart:exend,ix]
        else:
            emgcols = []
        
        breath = OrderedDict([('number', breathcnt),
                             ('name','Breath #' + str(breathcnt)), 
                             ('expiration', exp),
                             ('inspiration', insp), 
                             ('time', np.concatenate((insp["time"], exp["time"])).squeeze()), 
                             ('flow', np.concatenate((insp["flow"], exp["flow"])).squeeze()),
                             ('volume', np.concatenate((insp["volume"], exp["volume"])).squeeze()),
                             ('poes', np.concatenate((insp["poes"], exp["poes"])).squeeze()),
                             ('pgas', np.concatenate((insp["pgas"], exp["pgas"])).squeeze()),
                             ('pdi', np.concatenate((insp["pdi"], exp["pdi"])).squeeze()),
                             ('breathcnt', breathcnt), 
                             ('ignored', ignored),
                             ('entcols', entcols),
                             ('emgcols', emgcols),
                             ('filename', filename)])
        breaths[breathcnt] = breath
        
    return breaths

def separateintobreathsbyvolume(filename, timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns, settings):
    from scipy import signal

    breaths = OrderedDict()
    j = len(flow)
    bufferwidth = settings.processing.mechanics.breathseparationbuffer

    ib = ignorebreaths(filename, settings)
    
    breathno = 0
    breathcnt = 0

    invol = volume
    exvol = -1 * volume 
    exvol = exvol + min(exvol)*-1
    samplingfrequency = settings.input.format.samplingfrequency
    peakheight = 0.1
    peakdistance = 0.1
    peakwidth = 0.1
    inpeaks, props = signal.find_peaks(invol, height=peakheight, distance=peakdistance*samplingfrequency, width=peakwidth*samplingfrequency)
    expeaks, props = signal.find_peaks(exvol, height=peakheight, distance=peakdistance*samplingfrequency, width=peakwidth*samplingfrequency)

    for inpeak in inpeaks:
        
        breathcnt += 1
                
        if breathcnt == 1:
            instart = 0
            inend = inpeak - 1
        else:
            instart = expeaks[breathcnt-2]
            inend = inpeak - 1
        
        exstart = inend + 1 
        if breathcnt < len(inpeaks):
            expeak = expeaks[breathcnt-1]
            exend = expeak - 1
        else:
            exend = len(invol)-1
       
        exp = {'time':timecol[exstart:exend].squeeze(), 'flow':flow[exstart:exend].squeeze(), 'poes':poes[exstart:exend].squeeze(),
                   'pgas': pgas[exstart:exend].squeeze(), 'pdi':pdi[exstart:exend].squeeze(), 'volume':volume[exstart:exend].squeeze()}
        
        insp = {'time':timecol[instart:inend].squeeze(), 'flow':flow[instart:inend].squeeze(), 'poes':poes[instart:inend].squeeze(), 
                    'pgas':pgas[instart:inend].squeeze(), 'pdi':pdi[instart:inend].squeeze(), 'volume':volume[instart:inend].squeeze()}
            
        if breathcnt in ib:
            ignored = True
        else:
            breathno += 1
            ignored = False
            
        entlen = exend-instart
        if len(entropycolumns) > 0:
            entcols = np.zeros([entlen, entropycolumns.shape[1]])
            for ix in range(0, entropycolumns.shape[1]):      
                entcols[:,ix] = entropycolumns[instart:exend,ix]
        else:
            entcols = []

        if len(emgcolumns) > 0:
            emgcols = np.zeros([entlen, emgcolumns.shape[1]])
            for ix in range(0, emgcolumns.shape[1]):      
                emgcols[:,ix] = emgcolumns[instart:exend,ix]
        else:
            emgcols = []
#        print("instart:" + str(instart) + " inend:" + str(inend) + " exstart:" + str(exstart) + " exend:" + str(exend))

        breath = OrderedDict([('number', breathcnt),
                             ('name','Breath #' + str(breathcnt)), 
                             ('expiration', exp),
                             ('inspiration', insp), 
                             ('time', np.concatenate((insp["time"], exp["time"])).squeeze()), 
                             ('flow', np.concatenate((insp["flow"], exp["flow"])).squeeze()),
                             ('volume', np.concatenate((insp["volume"], exp["volume"])).squeeze()),
                             ('poes', np.concatenate((insp["poes"], exp["poes"])).squeeze()),
                             ('pgas', np.concatenate((insp["pgas"], exp["pgas"])).squeeze()),
                             ('pdi', np.concatenate((insp["pdi"], exp["pdi"])).squeeze()),
                             ('breathcnt', breathcnt), 
                             ('ignored', ignored),
                             ('entcols', entcols),
                             ('emgcols', emgcols),
                             ('filename', filename)])
        breaths[breathcnt] = breath
        
    return breaths

def separateintobreaths(method, filename, timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns, settings):
    if str(str.lower(method)) == "volume":
        return separateintobreathsbyvolume(filename, timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns, settings)
    else:
        return separateintobreathsbyflow(filename, timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns, settings)

def calculatemechanics(breath, bcnt, vefactor, avgvolumein, avgvolumeex, avgpoesin, avgpoesex, settings):
    retbreath = breath
    print(' Mechanics', end="")
    retbreath["inspiration"]["volumeavg"] = avgvolumein
    retbreath["expiration"]["volumeavg"] = avgvolumeex
    retbreath["volumeavg"] = np.concatenate([avgvolumein, avgvolumeex])
    
    retbreath["inspiration"]["poesavg"] = avgpoesin
    retbreath["expiration"]["poesavg"] = avgpoesex
    retbreath["poesavg"] = np.concatenate([avgpoesin, avgpoesex])
    
    retbreath["eilv"] = [retbreath["inspiration"]["volume"][len(retbreath["inspiration"]["volume"])-1], retbreath["inspiration"]["poes"][len(retbreath["inspiration"]["poes"])-1]]
    retbreath["eelv"] = [retbreath["expiration"]["volume"][len(retbreath["expiration"]["volume"])-1], retbreath["expiration"]["poes"][len(retbreath["expiration"]["poes"])-1]]

    retbreath["eilvavg"] = [retbreath["inspiration"]["volumeavg"][len(retbreath["inspiration"]["volumeavg"])-1], retbreath["inspiration"]["poesavg"][len(retbreath["inspiration"]["poesavg"])-1]]
    retbreath["eelvavg"] = [retbreath["expiration"]["volumeavg"][len(retbreath["expiration"]["volumeavg"])-1], retbreath["expiration"]["poesavg"][len(retbreath["expiration"]["poesavg"])-1]]
  
    exp = retbreath["expiration"]  
    
    poes_maxexp = max(exp["poes"])
    poes_endexp = exp["poes"][len(exp["poes"])-1]
    pdi_minexp = min(exp["pdi"])
    pdi_endexp = exp["pdi"][len(exp["pdi"])-1]
    pgas_endexp = exp["pgas"][len(exp["pgas"])-1]
    pgas_maxexp = max(exp["pgas"])
    pgas_minexp = min(exp["pgas"])
        
    midvolexp = min(exp["volume"]) + ((max(exp["volume"])-min(exp["volume"]))/2)
    midvolexpix = np.where(exp["volume"] <= midvolexp)[0][0]
    poes_midvolexp = exp["poes"][midvolexpix]
    flow_midvolexp = -exp["flow"][midvolexpix]
    
    insp = retbreath["inspiration"]
    poes_mininsp = min(insp["poes"])
    poes_endinsp = insp["poes"][len(insp["poes"])-1]
    pdi_maxinsp = max(insp["pdi"])
    pdi_endinsp = insp["pdi"][len(insp["pdi"])-1]
    pgas_endinsp = insp["pgas"][len(insp["pgas"])-1]
    
    poes_tidal_swing = abs(max(retbreath["poes"]) - min(retbreath["poes"]))
    pgas_tidal_swing = abs(max(retbreath["pgas"]) - min(retbreath["pgas"]))
    pdi_tidal_swing = abs(max(retbreath["pdi"]) - min(retbreath["pdi"]))

    midvolinsp = min(insp["volume"]) + ((max(insp["volume"])-min(insp["volume"]))/2)
    midvolinspix = np.where(insp["volume"] >= midvolinsp)[0][0]

    poes_midvolinsp = insp["poes"][midvolinspix]
    flow_midvolinsp = -insp["flow"][midvolinspix]
    
    vol_endinsp = insp["volume"][len(insp["volume"])-1]
    vol_endexp = exp["volume"][len(exp["volume"])-1]
    
    ti = len(insp["flow"])/settings.input.format.samplingfrequency
    te = len(exp["flow"])/settings.input.format.samplingfrequency
    ttot = len(retbreath["flow"])/settings.input.format.samplingfrequency                        
    ti_ttot = ti/ttot
    
    vt = max(retbreath["volume"])-min(retbreath["volume"])
    
    ve = vt * bcnt * vefactor
    
    vmrnumerator = (pgas_endinsp-pgas_endexp)
    vmrdenominator = (poes_endinsp-poes_endexp)
    vmr = np.divide(vmrnumerator, vmrdenominator, out=np.zeros_like(vmrnumerator), where=vmrdenominator!=0)
    
    tlr_insp = abs((poes_midvolexp-poes_midvolinsp)/(flow_midvolexp-flow_midvolinsp))
    insp_pdi_rise = pdi_maxinsp - min(insp["pdi"])
    exp_pgas_rise = pgas_maxexp - min(exp["pgas"])

    poesinsp = adjustforintegration(-insp["poes"])
    ptp_oesinsp, int_oesinsp = calcptp(poesinsp, bcnt, vefactor, settings.input.format.samplingfrequency)
    
    pdiinsp = adjustforintegration(insp["pdi"])
    ptp_pdiinsp, int_pdiinsp = calcptp(pdiinsp, bcnt, vefactor, settings.input.format.samplingfrequency)
    
    pgasexp = adjustforintegration(exp["pgas"])
    ptp_pgasexp, int_pgasexp = calcptp(pgasexp, bcnt, vefactor, settings.input.format.samplingfrequency)
    
    max_in_flow = min(insp["flow"]) * -1
    max_ex_flow = max(exp["flow"])
    inflowmidvol = insp["flow"][midvolinspix] * -1
    exflowmidvol = exp["flow"][midvolexpix]
    
    print(', WOB', end="")
    wob = calculatewob(breath, bcnt, vefactor, avgvolumein, avgvolumeex, avgpoesin, avgpoesex, settings)
    retbreath["wob"] = wob

    if len(breath["emgcols"]) > 0:
        print(', EMG RMS', end="")
        emglib = import_file("emglib", "emg.py")
        retbreath["rms"] = emglib.calculate_rms(breath["emgcols"], settings.processing.emg.rms_s, settings.input.format.samplingfrequency)
    
    if len(settings.input.data.columns_entropy) > 0:
        print(', Entropy', end="")
        entropy = calculateentropy(breath, settings)
        retbreath["entropy"] = np.append(entropy.T, [max(entropy.T), min(entropy.T), np.mean(entropy.T)])
    else:
        retbreath["entropy"] = []
    
    mechs = OrderedDict([    ('poes_maxexp', poes_maxexp),
                             ('poes_mininsp', poes_mininsp),
                             ('poes_endinsp', poes_endinsp), 
                             ('poes_endexp',poes_endexp),
                             ('poes_midvolexp',poes_midvolexp),
                             ('poes_midvolinsp',poes_midvolinsp),
                             ('int_oesinsp',int_oesinsp),
                             ('ptp_oesinsp',ptp_oesinsp),
                             ('poes_tidal_swing',poes_tidal_swing),                             
                             ('pgas_endinsp',pgas_endinsp),
                             ('pgas_endexp',pgas_endexp),
                             ('pgas_maxexp',pgas_maxexp),
                             ('pgas_minexp',pgas_minexp),
                             ('exp_pgas_rise',exp_pgas_rise),
                             ('int_pgasexp',int_pgasexp),
                             ('ptp_pgasexp',ptp_pgasexp),
                             ('pgas_tidal_swing',pgas_tidal_swing),
                             ('int_pdiinsp',int_pdiinsp),
                             ('ptp_pdiinsp',ptp_pdiinsp),
                             ('pdi_minexp',pdi_minexp),
                             ('pdi_maxinsp',pdi_maxinsp),
                             ('pdi_endinsp',pdi_endinsp),
                             ('pdi_endexp',pdi_endexp),
                             ('insp_pdi_rise',insp_pdi_rise),
                             ('pdi_tidal_swing',pdi_tidal_swing),                             
                             ('flow_midvolexp',flow_midvolexp),
                             ('flow_midvolinsp',flow_midvolinsp),
                             ('vol_endinsp',vol_endinsp),
                             ('vol_endexp',vol_endexp),
                             ('max_in_flow',max_in_flow),
                             ('max_ex_flow',max_ex_flow),
                             ('in_flow_midvol',inflowmidvol),
                             ('ex_flow_midvol',exflowmidvol),
                             ('ti',ti),
                             ('te',te),
                             ('ttot',ttot),
                             ('ti_ttot',ti_ttot),
                             ('vt',vt),
                             ('bf',bcnt * vefactor),
                             ('ve',ve),
                             ('vmr',vmr),
                             ('tlr_insp',tlr_insp)
                             ])
    retbreath["mechanics"] = mechs
    return retbreath

def adjustforintegration(data):
    mind = min(data)
    maxd = max(data)
    
    adjustment = mind
    if (maxd<0):
        adjustment = maxd
    
    return data - adjustment

def calcptp(pressure, bcnt, vefactor, samplingfreq):
    pressure = pressure.squeeze()
    xval = np.linspace(0, len(pressure)/samplingfreq, len(pressure))
    integral = sp.integrate.simps(pressure, xval)
    ptp = integral * bcnt * vefactor

    return ptp, integral

def calculatewob(breath, bcnt, vefactor, avgvolumein, avgvolumeex, avgpoesin, avgpoesex, settings):
    WOBUNITCHANGEFACTOR = 98.0638/1000 #Multiplication factor to change cmH2O to Joule;  Pa = J / m3.
    
    if settings.processing.wob.calcwobfrom == "average":
        volin = breath["inspiration"]["volumeavg"]
        volex = breath["expiration"]["volumeavg"]
        poesin = breath["inspiration"]["poesavg"]
        poesex = breath["expiration"]["poesavg"]
    else:
        volin = breath["inspiration"]["volume"]
        volex = breath["expiration"]["volume"]
        poesin = breath["inspiration"]["poes"]
        poesex = breath["expiration"]["poes"]
        
    eilv = [volin[len(poesin)-1], poesin[len(poesin)-1]]
    eelv = [volex[len(volex)-1], poesex[len(volex)-1]]
    
    #Inspiratory elastic WOB:
    tbase = abs(eilv[0]-eelv[0])
    theight = abs(eilv[1]-eelv[1])
    wobinela = tbase * theight / 2 * WOBUNITCHANGEFACTOR
    
    #Inspiratory resistive WOB:
    slope = (poesin[len(poesin)-1]-poesin[0]) / (volin[len(volin)-1]-volin[0])
    flyin = volin * slope + poesin[0]
    levelpoesin = (poesin*-1) - (flyin*-1)
    levelpoesin[np.where(levelpoesin<0)]=0
     
    wobinres = max(abs(sp.integrate.simps(levelpoesin, volin)),0) * WOBUNITCHANGEFACTOR
    
    #Expiratory WOB:
    levelpoesex = poesex - poesex[len(poesex)-1]
    levelpoesex[np.where(levelpoesex<0)]=0
    wobex = max(abs(sp.integrate.simps(levelpoesex, volex)),0) * WOBUNITCHANGEFACTOR
    
    #Totals
    wobin = wobinela + wobinres
    wobtotal = wobin + wobex

    wob = OrderedDict([
            ('wobtotal', wobtotal * bcnt * vefactor),
            ('wob_in_total', wobin * bcnt *  vefactor),
            ('wob_ex_total', wobex * bcnt *  vefactor),
            ('wob_in_ela', wobinela * bcnt *  vefactor),
            ('wob_in_res', wobinres * bcnt *  vefactor)
            ])
    
    return wob

def calculatebreathmechsandwob(breaths, bcnt, vefactor, avgvolumein, avgvolumeex, avgpoesin, avgpoesex, settings):
    
    retbreaths = OrderedDict()
    for breathno in breaths:
        print('\n\t\t... breath #' + str(breathno) + ":", end="")
        breath = breaths[breathno]
        if breath["ignored"]:
            print(' (ignored)', end="")
            retbreaths[breathno] = breath
        else:
            breathmechswob = calculatemechanics(breath, bcnt, vefactor, avgvolumein, avgvolumeex, avgpoesin, avgpoesex, settings)
            retbreaths[breathno] = breathmechswob
    print("")
    return retbreaths

def resample(x, settings, kind='linear'):
    x = x.squeeze()
    n = settings.processing.wob.avgresamplingobs
    f = sp.interpolate.interp1d(np.linspace(0, 1, x.size), x, kind)
    return f(np.linspace(0, 1, n))

def calculateaveragebreaths(breaths, settings):
    resamplingobs = settings.processing.wob.avgresamplingobs
    
    nobreaths = 0
    for breathno in breaths:
        if not breaths[breathno]["ignored"]:
            nobreaths += 1

    volumein = np.empty([resamplingobs, nobreaths])
    volumeex = np.empty([resamplingobs, nobreaths])
    poesin = np.empty([resamplingobs, nobreaths])
    poesex = np.empty([resamplingobs, nobreaths])
        
    for breathno in breaths:
        breath = breaths[breathno]
        if not breath["ignored"]:
            nobreaths -= 1
            volumein[:,nobreaths] = resample(breath["inspiration"]["volume"], settings)
            volumeex[:,nobreaths] = resample(breath["expiration"]["volume"], settings)
            poesin[:,nobreaths] = resample(breath["inspiration"]["poes"], settings)
            poesex[:,nobreaths] = resample(breath["expiration"]["poes"], settings)            
            
    avgvolumein = np.mean(volumein, axis=1)
    avgvolumeex = np.mean(volumeex, axis=1)
    avgpoesin = np.mean(poesin, axis=1)
    avgpoesex = np.mean(poesex, axis=1)
        
    return avgvolumein, avgvolumeex, avgpoesin, avgpoesex

def calculateentropy(breath, settings):
    
    ent = import_file("ent", "entropy.py")
     
    columns = breath["entcols"]

    #If EMG columns contained in entropy columns, use the processed data (not the original input)
    if len(settings.input.data.columns_emg)>0:
        emgcolnos = settings.input.data.columns_emg
        for entcolno in range(0, len(settings.input.data.columns_entropy)):
            entc = settings.input.data.columns_entropy[entcolno]
            if entc in emgcolnos:
                columns[:,entcolno] = breath["emgcols"][:,emgcolnos.index(entc)]
    
    epoch = settings.processing.entropy.entropy_epochs
    tolerancesd = settings.processing.entropy.entropy_tolerance
    sampen = np.zeros(len(columns[1,:]))
    
    for i in range(0,columns.shape[1]):
        std_ds = np.std(columns[:,i])
        sample_entropy = ent.sample_entropy(columns[:,i], epoch, tolerancesd * std_ds)
        sampen[i] = sample_entropy[len(sample_entropy)-1]
        
    return sampen

def savepvbreaths(file, breaths, flow, volume, poes, pgas, pdi, settings, averages=False):   
    try:
        os.makedirs(pjoin(settings.output.outputfolder, "plots"))
    except FileExistsError:
        pass
    
    nobreaths = len(breaths)
    maxnocols = settings.output.diagnostics.pvcolumns
    maxnorows = settings.output.diagnostics.pvrows
    plotsperpage = maxnocols * maxnorows
    nopages = -(-nobreaths // plotsperpage)

    nocols = maxnocols #math.floor(math.sqrt(nobreaths))
    norows = maxnorows #math.ceil(nobreaths/nocols)

    maxx = -np.inf 
    minx = np.inf 
    maxy = -np.inf 
    miny = np.inf
    
    no=0
    for breathno in breaths:
        no += 1
        breath = breaths[breathno]
        if averages:
            calcvol = "volumeavg"
            calcpoes = "poesavg"
        else:
            calcvol = "volume"
            calcpoes = "poes"
             
        if not breath["ignored"]:   
            maxnewx = max(breath[calcvol])
            maxnewy = max(breath[calcpoes])
            minnewx = min(breath[calcvol])
            minnewy = min(breath[calcpoes])

            maxx = max(maxx, maxnewx, 0) 
            maxy = max(maxy, maxnewy, 0) 

            minx = min(minx, minnewx, 0)
            miny = min(miny, minnewy, 0)
            
     
    maxx = maxx*1.1
    maxy = maxy*1.1
    minx = minx*1.1
    miny = miny*1.1

    if averages:
        savefile = pjoin(settings.output.outputfolder, "plots", "All files – average Campbell.pdf")
    else:
        savefile = pjoin(settings.output.outputfolder, "plots", file + " - Campbell.pdf")

    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(savefile) as pdf:
        no=0
        tno=0
        pno = 1
        for breathno in breaths: #for no in range(1, nobreaths+1):    
            no += 1
            tno += 1
            if (no == 1):
                fig, axes = plt.subplots(nrows=norows, ncols=nocols, figsize=(21,29.7))
                if averages:
                    plt.suptitle("Averages - Campbell diagrams (page " + str(pno) + " of " + str(nopages) + ")", fontsize=48)
                else:
                    plt.suptitle(file + " - Campbell diagrams (page " + str(pno) + " of " + str(nopages) + ")", fontsize=48)

            breath = breaths[breathno]
            if breath["ignored"]:
                aline = 0.2
                afill = 0.2
            else:
                aline = 1
                afill = 0.5
            
            ax = plt.subplot(norows, nocols, no)
            ax.set_xlim([minx, maxx])
            ax.set_ylim([miny, maxy])
            ax.invert_xaxis()
            
            if averages:
                ax.set_title(breath["filename"],fontweight="bold", size=20)
            else:
                ax.set_title("Breath #" + str(breathno),fontweight="bold", size=20)
                            
            ax.set_xlabel('Inspired volume (L)', size=16)
            ax.set_ylabel(r'Oesophageal pressure (cm ' + r'$H_2O$' + ')', size=16)
            ax.grid(True)
            
            ax.plot(breath[calcvol], breath[calcpoes], '-k', linewidth=2, alpha=aline)
            
            if breath["ignored"]:
                ax.plot([minx, maxx], [miny, maxy], '-r', linewidth=1)
                ax.plot([minx, maxx], [maxy, miny], '-r', linewidth=1)
            else:
                if averages:
                    eilv = breath["eilvavg"]
                    eelv = breath["eelvavg"]
                else:
                    eilv = breath["eilv"]
                    eelv = breath["eelv"]
                    
                [lx, ly] = list(zip(eilv, eelv))
                l = mlines.Line2D(lx, ly, linewidth=2, alpha=aline)
                ax.add_line(l)
                
                wobelpolygon = Polygon([[p[0], p[1]] for p in [eelv, eilv, [eilv[0], eelv[1]]]], alpha=afill, color="#999999", fill=True )
                ax.add_patch(wobelpolygon)

            if (no == plotsperpage) or (tno == nobreaths):
                plt.figtext(0.99, 0.01, CREATED + " on " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), horizontalalignment='right')       
                pdf.savefig() 
                plt.close(fig)   
                plt.close()   
                pno +=1
                no = 0

    return []



def saverawplots(suffix, file, rows, titles, ylabels, settings, breaths=None):   
    try:
        os.makedirs(pjoin(settings.output.outputfolder, "plots"))
    except FileExistsError:
        pass
    plt.ioff()
    norows = len(rows)
    fig, axes  = plt.subplots(nrows=norows, ncols=1, sharex=True, figsize=(29.7, 21))
    fig.suptitle(file + " - " + suffix, fontsize=48)    
    
    for i in range(0, norows):
        ax = plt.subplot(norows, 1, i+1)
        if breaths:
            ax.yaxis.grid(True)
        else:
            ax.grid(True)
        ax.plot(rows[i], linewidth=1)
        ax.set_title(titles[i],fontweight="bold", size=20)
        ax.set_ylabel(ylabels[i],fontweight="bold", size=16)
        startx = 0
        yl = list(ax.get_ylim())

        if breaths:
            for breathno in breaths:
                blab = " #" + str(breathno)
                breath= breaths[breathno]
                if breath["ignored"]:
                    blab += " (ignored)"
                    poly = Rectangle([startx, yl[0]], len(breath["poes"]), yl[1]-yl[0], alpha=0.1, color="#FF0000", fill=True)
                    ax.add_patch(poly)
                
                ax.axvline(x=startx, linewidth=0.5, linestyle="--", color="#0000FF")
                ax.text(startx, yl[1]-((yl[1]-yl[0])*0.05), blab, fontsize=12)
        
                startx = startx + len(breath["poes"])

    
    ax.set_xlabel(r'$observation #$', size=16)

    savefile = pjoin(settings.output.outputfolder, "plots", file + " - " + suffix + ".pdf")
    plt.figtext(0.99, 0.01, CREATED + " on " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), horizontalalignment='right')       
    fig.savefig(savefile)
    plt.close(fig)


def formatheader(df, writer, sheetname):
    writer.sheets[sheetname].freeze_panes(1, 1)
    header_format = writer.book.add_format({'bold': True,
                                            'fg_color': '#ffcccc',
                                            'border': 1,
                                            'align': 'right'})
    for col_num, value in enumerate(df.columns.values):
        writer.sheets[sheetname].write(0, col_num, value, header_format)  
        column_len = df[value].astype(str).str.len().max()
        column_len = max(column_len, len(str(value))) + 1
        writer.sheets[sheetname].set_column(col_num, col_num, column_len)
    
def savedataaverage(totals, settings):   
    try:
        os.makedirs(pjoin(settings.output.outputfolder, "data"))
    except FileExistsError:
        pass
    
    savefile = pjoin(settings.output.outputfolder, "data", "Average breathdata.xlsx")
    
    writer = pd.ExcelWriter(savefile, engine='xlsxwriter', options={'strings_to_urls': True}) # pylint: disable=abstract-class-instantiated
    totals.to_excel(writer, sheet_name='Data', index=False)
    formatheader(totals, writer, "Data")

    version = pd.DataFrame({'Created': CREATED + " on " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'Website': 'https://github.com/emilwalsted/respmech'
                            }, index=[0]).T
    
    version.columns=['Version info']
    version.to_excel(writer, sheet_name='Version', index=False)
    formatheader(version, writer, "Version")
    
    writer.save()
    
def savedataindividual(file, breaths, settings):   
    try:
        os.makedirs(pjoin(settings.output.outputfolder, "data"))
    except FileExistsError:
        pass
    
    mechs = []
    for breathno in breaths:   
        breath= breaths[breathno]
        if not breath["ignored"]:
            dfmech = pd.DataFrame(breath["mechanics"], index=[0])
            dfwob = pd.DataFrame(breath["wob"], index=[0])
            dfmech.insert(loc=0, column="breath_no", value=breath["number"])
            dfmech = dfmech.join(dfwob, how="outer", sort=False)
            
            if len(settings.input.data.columns_emg)>0:
                dfemg = pd.DataFrame(breath["rms"]).transpose()
                cols = ["rms_col_" +  str(settings.input.data.columns_emg[x]) for x in range(0, len(settings.input.data.columns_emg))]
                cols = np.append(cols, ['RMS_max', 'RMS_mean'])
                dfemg.columns = cols
                dfmech = dfmech.join(dfemg, how="outer", sort=False)

            if len(settings.input.data.columns_entropy)>0:
                dfent = pd.DataFrame(breath["entropy"]).transpose()
                cols = ["sample_entropy_col_" +  str(settings.input.data.columns_entropy[x]) for x in range(0, len(settings.input.data.columns_entropy))]
                cols = np.append(cols, ['sample_entropy_max', 'sample_entropy_min', 'sample_entropy_mean'])
                dfent.columns = cols
                dfmech = dfmech.join(dfent, how="outer", sort=False)
                
            if len(mechs)>0:
                mechs = pd.concat([mechs, dfmech], sort=False)
            else:
                mechs = dfmech
                    
    savefile = pjoin(settings.output.outputfolder, "data", file + ".breathdata.xlsx")
    
    ret = pd.DataFrame(mechs.mean())
    ret = ret.T
    ret = ret.drop(columns="breath_no")
    ret.insert(loc=0, column="file", value=file)
    
    writer = pd.ExcelWriter(savefile, engine='xlsxwriter', options={'strings_to_urls': True}) # pylint: disable=abstract-class-instantiated
    mechs.to_excel(writer, sheet_name='Data', index=False)
    formatheader(mechs, writer, "Data")
    
    version = pd.DataFrame({'Created': CREATED + " on " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'Website': 'https://github.com/emilwalsted/respmech'
                            }, index=[0]).T
    version.columns=['Version info']
    version.to_excel(writer, sheet_name='Version', index=False)
    formatheader(version, writer, "Version")
    
    writer.save()
    
    return ret

def getprocesseddata(breaths, settings):
    
    processeddata = []
    for breathno in breaths:   
        breath= breaths[breathno]
        if settings.output.data.includeignoredbreaths or not breath["ignored"]:
            times = np.arange(0, len(breath["flow"])-1, dtype=int) / settings.input.format.samplingfrequency
            dftime = pd.DataFrame(times, columns=["Time"])
            dftime.set_index(dftime['Time'])

            bnos = np.arange(0, len(breath["flow"])-1, dtype=int) * 0 + breathno
            dfbreathno = pd.merge(dftime, pd.DataFrame(bnos, columns=["Breathno"]), how="outer", left_index=True, right_index=True)
            dfflow= pd.merge(dfbreathno, pd.DataFrame(breath["flow"], columns=["Flow"]), how="outer", left_index=True, right_index=True)
            dfvol = pd.merge(dfflow, pd.DataFrame(breath["volume"], columns=["Volume"]), how="outer", left_index=True, right_index=True)
            dfpoes = pd.merge(dfvol, pd.DataFrame(breath["poes"], columns=["Poes"]), how="outer", left_index=True, right_index=True)
            dfpgas = pd.merge(dfpoes, pd.DataFrame(breath["pgas"], columns=["Pgas"]), how="outer", left_index=True, right_index=True)
            dfpdi = pd.merge(dfpgas, pd.DataFrame(breath["pdi"], columns=["Pdi"]), how="outer", left_index=True, right_index=True)
            br = dfpdi
            #dfmech.insert(loc=0, column="time", value=breath["number"])
            
            if len(settings.input.data.columns_emg)>0:
                dfemg = pd.merge(dfpdi, pd.DataFrame(breath["emgcols"], columns=["EMG1", "EMG2", "EMG3", "EMG4", "EMG5"]), how="outer", left_index=True, right_index=True)
                br = dfemg
                
            if len(processeddata)>0:
                processeddata = pd.concat([processeddata, br], sort=False)
            else:
                processeddata = br
    
    return processeddata.dropna()

def saveprocesseddata(breaths, settings, file):   
    try:
        os.makedirs(pjoin(settings.output.outputfolder, "data"))
    except FileExistsError:
        pass
    
    savefile = pjoin(settings.output.outputfolder, "data", ntpath.basename(file) + " – Processed data.csv")
    processeddata = getprocesseddata(breaths, settings)

    processeddata.to_csv(savefile, index=False)

def savesoundemgchannel(emgchannel, filename, settings):
    from scipy.io.wavfile import write
    
    scaled = np.int16(emgchannel/np.max(np.abs(emgchannel)) * 32767)
    write(filename, settings.input.format.samplingfrequency, scaled)    

def savesoundemg(emgchannels, wavfilename, settings):
    
    for i in range(0, len(settings.input.data.columns_emg)):
        emgchannel = emgchannels[:,i]
        savesoundemgchannel(emgchannel, str.replace(wavfilename, "#$#", str(i+1)), settings)

def applysubsettings(defaultsettings, newsettings):
    retsettings = defaultsettings
    for attr, value in newsettings.__dict__.items():
        a = str.replace(attr, "'", "")
        if isinstance(value, SimpleNamespace):
            subsettings = applysubsettings(defaultsettings.__dict__[a], newsettings.__dict__[a])
            setattr(retsettings, a, subsettings) 
        else: 
            setattr(retsettings, a, value)    
    return retsettings

def applysettings(defaultsettings, usersettings):
   
    defaultset = json.loads(defaultsettings, object_hook=lambda d: SimpleNamespace(**d))   
    userset = json.loads(json.dumps(usersettings), object_hook=lambda d: SimpleNamespace(**d))
    
    settings = applysubsettings(defaultset, userset)
    return settings
    
def analyse(usersettings):
    
    settings = applysettings(defaultsettings, usersettings)
  
    print('Loading data...')
    filepath = pjoin(settings.input.inputfolder, settings.input.files)
    files = [f for f in glob.glob(filepath)]
    files.sort()

    print('Processing ' + str(len(files)) + ' file(s):')
    
    averagebreaths = {}
    averagecalcs = []
    
    for fi in files:
        file  = os.path.abspath(fi)
        filename = ntpath.basename(file)
        print('\nProcessing \'' + file + '\'')

        flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns = load(file, settings)

        timecol = np.arange(0, len(flow)-1, dtype=int) / settings.input.format.samplingfrequency
        
        if (settings.output.diagnostics.savedataviewraw):
            print('\t\tSaving raw data plots...')
            saverawplots("Flow, volume and pressures (raw data)", ntpath.basename(file), [flow, volume, poes, pgas, pdi], 
                         ['Flow', 'Volume (uncorrected)', 'Oesophageal pressure', 'Gastric pressure', 'Trans diaphragmatic pressure'],
                         [r'$L/s$', r'$L$', r'$cm H_2O$', r'$cm H_2O$', r'$cm H_2O$'],
                         settings)

        print('\t\tTrimming to whole breaths...')
        timecol, flow, volume, poes, pgas, pdi, emgcolumns = trim(timecol, flow, volume, poes, pgas, pdi, emgcolumns, settings)
        
        if len(emgcolumns) > 0:
            print('\t\tProcessing EMG')
            emgcols = np.array(emgcolumns)
            emgcolsraw = emgcols
            emglib = import_file("emglib", "emg.py")

            if settings.processing.emg.remove_ecg:
                print('\t\t...removing ECG')
                emgcolumns_ecgremoved, ecgw = emglib.remove_ecg(emgcolumns, 
                    emgcols[:,settings.processing.emg.column_detect], 
                    samplingfrequency=settings.input.format.samplingfrequency, 
                    ecgminheight=settings.processing.emg.minheight, 
                    ecgmindistance=settings.processing.emg.mindistance, 
                    ecgminwidth=settings.processing.emg.minwidth, 
                    windowsize=settings.processing.emg.windowsize)
                
                emgcols = np.array(emgcolumns_ecgremoved)

            if settings.processing.emg.remove_noise:
                print('\t\t...reducing noise')
                colheaders = ["EMG " + str(i) for i in range(1, len(emgcols[0])+1)]
                noiseprofiles = settings.processing.emg.noise_profile
                noiseprofile = []
                for nop in noiseprofiles:
                    if nop[0]==filename:
                        noiseprofile=nop[1]
                        break
                
                nrcols=[]
                for i in range(0, len(emgcols[0])):
                    nrcol = emglib.reducenoise(np.array(emgcols[:,i]), noiseprofile, settings.input.format.samplingfrequency)
                    nrcols += [nrcol]
                   
                emgcolumns_noiseremoved = np.array(nrcols).T
                emgcolumns_noiseremoved = np.pad(emgcolumns_noiseremoved, ((0, len(emgcols)-len(emgcolumns_noiseremoved)),(0,0)), 'constant') 
                emgcols = emgcolumns_noiseremoved

            emgcolumns = emgcols
            
        #Add any post processing that should be performed on the entire data file
        #here.
        
        print('\t\tDrift correcting...')
        uncorvol = volume
        zerovol = zero(volume)
        
        if settings.processing.mechanics.correctvolumedrift:
            driftvol = correctdrift(zerovol, settings)
        else:
            driftvol = zerovol
            
        if settings.processing.mechanics.correctvolumetrend:
            trendcorvol = correcttrend(ntpath.basename(file), driftvol, settings)
            volume = trendcorvol
        else:
            volume = driftvol

        print('\t\tDetecting breathing cycles...')
        if settings.processing.mechanics.separateby == "volume":
            breaths = separateintobreaths("volume", filename, timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns, settings)
        else:
            breaths = separateintobreaths("flow", filename, timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns, settings)

        if (settings.output.diagnostics.savedataviewtrimmed):
            print('\t\tSaving trimmed data plots...')
            saverawplots("Trimmed data", ntpath.basename(file), [flow, volume, poes, pgas, pdi], 
                         ['Flow', 'Volume (drift corrected)', 'Oesophageal pressure', 'Gastric pressure', 'Trans diaphragmatic pressure'],
                         [r'$L/s$', r'$L$', r'$cm H_2O$', r'$cm H_2O$', r'$cm H_2O$'],
                         settings, breaths)
                    
        if (settings.output.diagnostics.savedataviewdriftcor):
            print('\t\tSaving drift corrected data plots...')
            if settings.processing.mechanics.correctvolumetrend:
                saverawplots("Volume correction", ntpath.basename(file), [flow, uncorvol, zerovol, driftvol, trendcorvol], 
                             ['Flow', 'Uncorrected volume', 'Zeroed volume', 'Linear drift corrected volume', 'Trend adjusted volume (' + settings.processing.mechanics.volumetrendadjustmethod + ')'],
                             [r'$L/s$', r'$L$', r'$L$', r'$L$', r'$L$'],
                             settings, breaths)
            else:
                saverawplots("Volume correction", ntpath.basename(file), [flow, uncorvol, zerovol, driftvol], 
                             ['Flow', 'Uncorrected volume', 'Zeroed volume', 'Linear drift corrected volume'],
                             [r'$L/s$', r'$L$', r'$L$', r'$L$'],
                             settings, breaths)
        
        #Add any post processing that should be performed for each breath here.
        
        if len(emgcolumns) > 0:
            print('\t\tSaving EMG raw channel overview...')
            colheaders = ["EMG " + str(i) for i in range(1, len(emgcols[0])+1)]
            savefile = pjoin(settings.output.outputfolder, "plots", ntpath.basename(file) + " - EMG (raw data)" + ".pdf")
            emglib.saveemgplots(savefile,
                breaths,
                timecol,
                emgcolsraw,
                colheaders,
                [r'$mcV$', r'$mcV$', r'$mcV$', r'$mcV$', r'$mcV$'],
                "Raw EMG",
                ylim=[-0.1,0.1],
                refsig=flow,
                reflabel="Flow (for reference)"
                )

            if settings.processing.emg.save_sound:
                print('\t\tSaving EMG raw channel sound files...')
                wavfile = pjoin(settings.output.outputfolder, "plots", ntpath.basename(file) + " - EMG #$# (raw data)" + ".wav")
                savesoundemg(emgcolsraw, wavfile, settings)

            if settings.processing.emg.remove_ecg:
                print('\t\tSaving EMG with ECG removed overview...')
                emgcols = np.array(emgcolumns_ecgremoved)
                ecgw_time = np.array(ecgw) / settings.input.format.samplingfrequency
                savefile = pjoin(settings.output.outputfolder, "plots", ntpath.basename(file) + " - EMG (ECG removed)" + ".pdf")
                emglib.saveemgplots(savefile,
                    breaths,
                    timecol,
                    emgcols,
                    colheaders,
                    [r'$mcV$', r'$mcV$', r'$mcV$', r'$mcV$', r'$mcV$'],
                    "EMG (ECG removed)",
                    ylim=[-0.1,0.1],
                    ecgwindows = ecgw_time,
                    refsig=flow,
                    reflabel="Flow (for reference)"
                    )
                
                if settings.processing.emg.save_sound:
                    print('\t\tSaving EMG with ECG removed sound files...')
                    wavfile = pjoin(settings.output.outputfolder, "plots", ntpath.basename(file) + " - EMG #$# (ECG removed)" + ".wav")
                    savesoundemg(emgcols, wavfile, settings)

            if settings.processing.emg.remove_noise:
                print('\t\tSaving noise reduced EMG overview...')
                colheaders = ["EMG " + str(i) for i in range(1, len(emgcols[0])+1)]
                savefile = pjoin(settings.output.outputfolder, "plots", ntpath.basename(file) + " - EMG (ECG removed and noise reduced)" + ".pdf")
                ecgw_time = [[int(noiseprofile[0] * settings.input.format.samplingfrequency), int(noiseprofile[1] * settings.input.format.samplingfrequency)]]
                ecgw_time = np.array(ecgw_time) / settings.input.format.samplingfrequency
                emglib.saveemgplots(savefile,
                    breaths,
                    timecol,
                    emgcolumns_noiseremoved,
                    colheaders,
                    [r'$mcV$', r'$mcV$', r'$mcV$', r'$mcV$', r'$mcV$'],
                    "EMG (ECG removed and noise reduced)",
                    ylim=[-0.1,0.1],
                    ecgwindows = ecgw_time,
                    refsig=flow,
                    reflabel="Flow (for reference)"
                    )

                if settings.processing.emg.save_sound:
                    print('\t\tSaving EMG with ECG removed and noise reduced sound files...')
                    wavfile = pjoin(settings.output.outputfolder, "plots", ntpath.basename(file) + " - EMG #$# (ECG removed and noise reduced)" + ".wav")
                    savesoundemg(emgcolumns_noiseremoved, wavfile, settings)
                        
        
            
        print('\t\tCalculating mechanics, WOB, entropy, RMS...')
        vefactor = 60/(len(flow)/settings.input.format.samplingfrequency) #Calculate multiplication factor for VE for this file.
        
        bcnt = len(breaths) 
        for bc in settings.processing.mechanics.breathcounts:
            if bc[0] == filename:
                bcnt = bc[1]
                break
       
        avgvolumein, avgvolumeex, avgpoesin, avgpoesex = calculateaveragebreaths(breaths, settings)
        
        breathmechswob = calculatebreathmechsandwob(breaths, bcnt, vefactor, avgvolumein, avgvolumeex, avgpoesin, avgpoesex, settings)
  
        for breathno in breathmechswob:
            breath = breathmechswob[breathno]
            if not breath["ignored"]:
                averagebreaths[filename] = breath
                break
    
        if (settings.output.diagnostics.savepvindividualworkload):
            print('\n\t\tSaving detailed pressure/volume overview plots...')
            savepvbreaths(filename, breathmechswob, flow, volume, poes, pgas, pdi, settings, False)
         
        if (settings.output.data.savebreathbybreathdata):
            print('\t\tSaving individual breath data...')
        
            av = savedataindividual(filename, breathmechswob, settings)
               
            if len(averagecalcs) > 0:
                averagecalcs = pd.concat([averagecalcs, av])
            else:
                averagecalcs = av

        if (settings.output.data.saveprocesseddata):
            saveprocesseddata(breaths, settings, ntpath.basename(file))

    #Save data and average breath plots
    if (settings.output.diagnostics.savepvaverage):
        print('\n\tSaving average plots...')
        savepvbreaths(filename, averagebreaths, flow, volume, poes, pgas, pdi, settings, True)  
    
    
    if (settings.output.data.saveaveragedata):
        print('\n\tSaving total average data...')
        savedataaverage(averagecalcs, settings)
        
        
    print('\nFinished analysing \'' + file + '\'.\n')
    return settings.output.outputfolder

defaultsettings = """{
    "input": {
        "inputfolder": "/Users/emilnielsen/Documents/Medicin/Forskning/Code/Respiratory mechanics/test/input/MATLAB Mac",
        "files": "*.*",
        "format": {
            "samplingfrequency": null,
            "matlabfileformat": null,
            "decimalcharacter": "."
        },
        "data": {
            "column_poes": null,
            "column_pgas": null,
            "column_pdi": null,
            "column_volume": null,
            "column_flow": null,
            "columns_entropy": [],
            "columns_emg": []
        }
    },
    "processing": {
        "mechanics": {
            "breathseparationbuffer": 800,
            "inverseflow": false,
            "integratevolumefromflow": false,
            "inversevolume": false,
            "correctvolumedrift": true,
            "correctvolumetrend": false,
            "volumetrendadjustmethod": "linear",
            "volumetrendpeakminheight": 0.8,
            "volumetrendpeakmindistance": 0.4,
            "excludebreaths": [],
            "breathcounts": []
        },
        "wob": {
            "calcwobfrom": "average",
            "avgresamplingobs": 500
        },
        "emg": {
            "rms_s": 0.050,
            "remove_ecg": false,  
            "minheight": 0.0005, 
            "mindistance": 0.5, 
            "minwidth": 0.001, 
            "windowsize": 0.4, 
            "remove_noise": false, 
            "save_sound": false
        },
        "entropy": {
            "entropy_epochs": 2,
            "entropy_tolerance": 0.1
        }
    },
    "output": {
        "outputfolder": "/Users/emilnielsen/Documents/Medicin/Forskning/Code/Respiratory mechanics/test/output",
        "data": {
            "saveaveragedata": true,
            "savebreathbybreathdata": true,
            "saveprocesseddata": false,
            "includeignoredbreaths": true
        },
        "diagnostics": {
            "savepvaverage": true,
            "savepvoverview": true,
            "savepvindividualworkload": true,
            "pvcolumns": 3,
            "pvrows": 4,
            "savedataviewraw": true,
            "savedataviewtrimmed": true,
            "savedataviewdriftcor": true
        }
    }
}"""

