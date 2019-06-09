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
The latest version of this code is available at GitHub: ...

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

import os
import glob
import ntpath
import math
from os.path import join as pjoin
import numpy as np 
from shapely.geometry import Polygon
from descartes import PolygonPatch
import scipy as sp
import scipy.io as sio  
from collections import OrderedDict 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import seaborn as sns; sns.set()
plt.style.use('seaborn-white')


def load(filepath, settings):
    
    fn, fext = os.path.splitext(filepath)
    
    def loadmat(path):
        if settings["matlabfileformat"] == 2:
            flow, volume, poes, pgas, pdi, entropycolumns = loadmatmac(filepath)
        else:
            flow, volume, poes, pgas, pdi, entropycolumns = loadmatwin(filepath)               
        return flow, volume, poes, pgas, pdi, entropycolumns

    def loadmatmac(filename):
        rawdata = sio.loadmat(filename)
        data = OrderedDict(rawdata)
        cols = list(data.items())
        
        flow = cols[settings["column_flow"]-1][1]
        volume = cols[settings["column_volume"]-1][1]
        poes = cols[settings["column_poes"]-1][1]
        pgas = cols[settings["column_pgas"]-1][1]
        pdi = cols[settings["column_pdi"]-1][1]
        
        if len(settings["columns_entropy"])==0:
            entropycolumns = []
        else:
            entropycolumns = np.empty([len(cols[settings["columns_entropy"][0]][1]), len(settings["columns_entropy"])])
            for i in range(0, len(settings["columns_entropy"])):
                entropycolumns[:,i] = cols[settings["columns_entropy"][i]-1][1].squeeze()
                        
        return flow, volume, poes, pgas, pdi, entropycolumns
        
    def loadmatwin(filename):
        rawdata = sio.loadmat(filename)
        data = OrderedDict(rawdata)
        cols = list(data["data_block1"])
        
        flow = cols[settings["column_flow"]-1]
        volume = cols[settings["column_volume"]-1]
        poes = cols[settings["column_poes"]-1]
        pgas = cols[settings["column_pgas"]-1]
        pdi = cols[settings["column_pdi"]-1]
        
        if len(settings["columns_entropy"])==0:
            entropycolumns = []
        else:
            entropycolumns = np.empty([len(cols[settings["columns_entropy"][0]]), len(settings["columns_entropy"])])
            for i in range(0, len(settings["columns_entropy"])):
                entropycolumns[:,i] = cols[settings["columns_entropy"][i]-1]
        
        return flow, volume, poes, pgas, pdi, entropycolumns
    
    def loadxls(filename):
        data = pd.read_excel(filename)
        
        flow = data.iloc[:,settings["column_flow"]-1].to_numpy()
        volume = data.iloc[:,settings["column_volume"]-1].to_numpy()
        poes = data.iloc[:,settings["column_poes"]-1].to_numpy()
        pgas = data.iloc[:,settings["column_pgas"]-1].to_numpy()
        pdi = data.iloc[:,settings["column_pdi"]-1].to_numpy()
        
        if len(settings["columns_entropy"])==0:
            entropycolumns = []
        else:
            entropycolumns = data.iloc[:,np.array(settings["columns_entropy"])-1].to_numpy()
        
        return flow, volume, poes, pgas, pdi, entropycolumns

    def loadcsv(filename):
        data = pd.read_csv(filename)
        
        flow = data.iloc[:,settings["column_flow"]-1].to_numpy()
        volume = data.iloc[:,settings["column_volume"]-1].to_numpy()
        poes = data.iloc[:,settings["column_poes"]-1].to_numpy()
        pgas = data.iloc[:,settings["column_pgas"]-1].to_numpy()
        pdi = data.iloc[:,settings["column_pdi"]-1].to_numpy()
        
        if len(settings["columns_entropy"])==0:
            entropycolumns = []
        else:
            entropycolumns = data.iloc[:,np.array(settings["columns_entropy"])-1].to_numpy()
        
        return flow, volume, poes, pgas, pdi, entropycolumns
    
    def loadfext(x):
        return {
            #Add new file extensions and a handler function here.
            '.xls': loadxls,
            '.xlsx': loadxls,
            '.csv': loadcsv,
            '.mat': loadmat
        }.get(x) #, default)
    
    flow = []
    volume = []
    poes = []
    pgas = []
    pdi = []
    
    loadfunc = loadfext(fext)
    flow, volume, poes, pgas, pdi, entropycolumns = loadfunc(filepath)
      
    return flow, volume, poes, pgas, pdi, entropycolumns


def zero(indata):
    return indata-(indata[0]);

def correctdrift(volume, settings):
    xno = len(volume)-1
      
    a = ((volume[xno])-volume[0])/xno

    val = a;
    corvol=np.zeros(xno)
    for i in range(0, xno):
        corvol[i] = volume[i] + val
        val = val - a
        
    return corvol

def trim(flow, volume, poes, pgas, pdi, settings):
    startix = np.argmax(flow <= 0)
    
    posflow = np.argwhere(flow >= 0)
    endix = posflow[:,0][len(posflow[:,0])-1]
    
    return flow[startix:endix], volume[startix:endix], poes[startix:endix], pgas[startix:endix], pdi[startix:endix]

def ignorebreaths(curfile, settings):
    allignore = settings['excludebreaths']
    d = dict(allignore)
    if curfile in d:
        return d[curfile]
    else:
        return []

   
def separateintobreaths(filename, flow, volume, poes, pgas, pdi, entropycolumns, settings):
    breaths = OrderedDict()
    j = len(flow)
    bufferwidth = 800

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
        
        exstart = i;       
        while (i<j and ((flow[i]>0) or (np.mean(flow[i:min(j,i+bufferwidth)])>0))):
            i += 1
        exend = i-1
        exend = min(exend, j)
        
        exp = {'flow':flow[exstart:exend].squeeze(), 'poes':poes[exstart:exend].squeeze(),
                   'pgas': pgas[exstart:exend].squeeze(), 'pdi':pdi[exstart:exend].squeeze(), 'volume':volume[exstart:exend].squeeze()}
        
        insp = {'flow':flow[instart:inend].squeeze(), 'poes':poes[instart:inend].squeeze(), 
                    'pgas':pgas[instart:inend].squeeze(), 'pdi':pdi[instart:inend].squeeze(), 'volume':volume[instart:inend].squeeze()}
            
        if breathcnt in ib:
            ignored = True
        else:
            breathno += 1
            ignored = False
            
        entlen = exend-instart
        if len(entropycolumns) > 0:
            entcols = np.zeros([entlen, entropycolumns.shape[1]])
        else:
            entcols = []
            
        for ix in range(0, entropycolumns.shape[1]):      
            entcols[:,ix] = entropycolumns[instart:exend,ix]
        
        breath = OrderedDict([('number', breathcnt),
                             ('name','Breath #' + str(breathcnt)), 
                             ('expiration', exp),
                             ('inspiration', insp), 
                             ('flow', np.concatenate((exp["flow"], insp["flow"])).squeeze()),
                             ('volume', np.concatenate((exp["volume"],insp["volume"])).squeeze()),
                             ('poes', np.concatenate((exp["poes"], insp["poes"])).squeeze()),
                             ('pgas', np.concatenate((exp["pgas"], insp["pgas"])).squeeze()),
                             ('pdi', np.concatenate((exp["pdi"], insp["pdi"])).squeeze()),
                             ('breathcnt', breathcnt), 
                             ('ignored', ignored),
                             ('entcols', entcols),
                             ('filename', filename)])
        breaths[breathcnt] = breath
        
    return breaths

def calculatemechanics(breath, bcnt, vefactor, avgvolumein, avgvolumeex, avgpoesin, avgpoesex, settings):
    retbreath = breath
    
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
    
    midvolexp = min(exp["volume"]) + ((max(exp["volume"])-min(exp["volume"]))/2); 
    midvolexpix = np.where(exp["volume"] <= midvolexp)[0][0]
    poes_midvolexp = exp["poes"][midvolexpix]
    flow_midvolexp = -exp["flow"][midvolexpix]
    
    insp = retbreath["inspiration"]
    poes_mininsp = min(insp["poes"]);
    poes_endinsp = insp["poes"][len(insp["poes"])-1]
    pdi_maxinsp = max(insp["pdi"])
    pdi_endinsp = insp["pdi"][len(insp["pdi"])-1]
    pgas_endinsp = insp["pgas"][len(insp["pgas"])-1]

    midvolinsp = min(insp["volume"]) + ((max(insp["volume"])-min(insp["volume"]))/2)
    midvolinspix = np.where(insp["volume"] >= midvolinsp)[0][0]

    poes_midvolinsp = insp["poes"][midvolinspix]
    flow_midvolinsp = -insp["flow"][midvolinspix]
    
    vol_endinsp = insp["volume"][len(insp["volume"])-1]
    vol_endexp = exp["volume"][len(exp["volume"])-1]
    
    ti_ttot = len(insp["flow"])/len(retbreath["flow"])
    vt = max(retbreath["volume"])-min(retbreath["volume"])
    
    ve = vt * bcnt * vefactor
    vmr = (pgas_endinsp-pgas_endexp)/(poes_endinsp-poes_endexp)
    tlr_insp = abs((poes_midvolexp-poes_midvolinsp)/(flow_midvolexp-flow_midvolinsp))
    insp_pdi_rise = pdi_maxinsp - min(insp["pdi"])
    exp_pgas_rise = pgas_maxexp - min(exp["pgas"])

    poesinsp = adjustforintegration(-insp["poes"])
    ptp_oesinsp, int_oesinsp = calcptp(poesinsp, bcnt, vefactor, settings["samplingfrequency"])
    
    pdiinsp = adjustforintegration(insp["pdi"])
    ptp_pdiinsp, int_pdiinsp = calcptp(pdiinsp, bcnt, vefactor, settings["samplingfrequency"])
    
    pgasexp = adjustforintegration(exp["pgas"])
    ptp_pgasexp, int_pgasexp = calcptp(pgasexp, bcnt, vefactor, settings["samplingfrequency"])
    
    max_in_flow = min(insp["flow"]) * -1
    max_ex_flow = max(exp["flow"])
    inflowmidvol = insp["flow"][midvolinspix] * -1
    exflowmidvol = exp["flow"][midvolexpix]
    
    wob = calculatewob(breath, bcnt, vefactor, avgvolumein, avgvolumeex, avgpoesin, avgpoesex, settings)
    retbreath["wob"] = wob
    
    entropy = calculateentropy(breath, settings)
    retbreath["entropy"] = entropy.T
    
    mechs = OrderedDict([('poes_maxexp', poes_maxexp),
                             ('poes_mininsp', poes_mininsp),
                             ('poes_endinsp', poes_endinsp), 
                             ('poes_endexp',poes_endexp),
                             ('poes_midvolexp',poes_midvolexp),
                             ('poes_midvolinsp',poes_midvolinsp),
                             ('pdi_minexp',pdi_minexp),
                             ('pdi_maxinsp',pdi_maxinsp),
                             ('pdi_endinsp',pdi_endinsp),
                             ('pdi_endexp',pdi_endexp),
                             ('pgas_endinsp',pgas_endinsp),
                             ('pgas_endexp',pgas_endexp),
                             ('pgas_maxexp',pgas_maxexp),
                             ('pgas_minexp',pgas_minexp),
                             ('flow_midvolexp',flow_midvolexp),
                             ('flow_midvolinsp',flow_midvolinsp),
                             ('vol_endinsp',vol_endinsp),
                             ('vol_endexp',vol_endexp),
                             ('ti_ttot',ti_ttot),
                             ('vt',vt),
                             ('bf',bcnt * vefactor),
                             ('ve',ve),
                             ('vmr',vmr),
                             ('tlr_insp',tlr_insp),
                             ('insp_pdi_rise',insp_pdi_rise),
                             ('exp_pgas_rise',exp_pgas_rise),
                             ('int_oesinsp',int_oesinsp),
                             ('ptp_oesinsp',ptp_oesinsp),
                             ('int_pdiinsp',int_pdiinsp),
                             ('ptp_pdiinsp',ptp_pdiinsp),
                             ('int_pgasexp',int_pgasexp),
                             ('ptp_pgasexp',ptp_pgasexp),
                             ('max_in_flow',max_in_flow),
                             ('max_ex_flow',max_ex_flow),
                             ('in_flow_midvol',inflowmidvol),
                             ('ex_flow_midvol',exflowmidvol)
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
    wobunitchangefactor = 98.0638/1000 #Multiplication factor to change cmH2O to Joule;  Pa = J / m3.
    
    if settings["calcwobfromaverage"]:
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
    wobelpolygon = Polygon([[p[0], p[1]] for p in [eelv, eilv, [eilv[0], eelv[1]]]])
    wobinela = wobelpolygon.area * wobunitchangefactor

    #Inspiratory resistive WOB:
    slope = (poesin[len(poesin)-1]-poesin[0]) / (volin[len(volin)-1]-volin[0])
    flyin = volin * slope + poesin[0]
    levelpoesin = (poesin*-1) - (flyin*-1)
    levelpoesin[np.where(levelpoesin<0)]=0
     
    wobinres = max(abs(sp.integrate.simps(levelpoesin, volin)),0) * wobunitchangefactor
    
    #Expiratory WOB:
    levelpoesex = poesex - poesex[len(poesex)-1]
    levelpoesex[np.where(levelpoesex<0)]=0
    wobex = max(abs(sp.integrate.simps(levelpoesex, volex)),0) * wobunitchangefactor
    
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
        breath = breaths[breathno]
        if breath["ignored"]:
            retbreaths[breathno] = breath
        else:
            breathmechswob = calculatemechanics(breath, bcnt, vefactor, avgvolumein, avgvolumeex, avgpoesin, avgpoesex, settings)
            retbreaths[breathno] = breathmechswob
    
    return retbreaths

def resample(x, settings, kind='linear'):
    x = x.squeeze()
    n = settings["avgresamplingobs"]
    f = sp.interpolate.interp1d(np.linspace(0, 1, x.size), x, kind)
    return f(np.linspace(0, 1, n))

def calculateaveragebreaths(breaths, settings):
    resamplingobs = settings["avgresamplingobs"]
    
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
    from pyentrp import entropy as ent
    
    columns = breath["entcols"]
    
    epoch = 2
    tolerancesd = 0.1
    sampen = np.zeros(len(columns[1,:]))
    
    for i in range(0,columns.shape[1]):
        std_ds = np.std(columns[:,i])
        sample_entropy = ent.sample_entropy(columns[:,i], epoch, tolerancesd * std_ds)
        sampen[i] = sample_entropy[len(sample_entropy)-1];
        
    return sampen


def savepvbreaths(file, breaths, flow, volume, poes, pgas, pdi, settings, averages=False):   
    nobreaths = len(breaths)
    nocols = math.floor(math.sqrt(nobreaths))
    norows = math.ceil(nobreaths/nocols)

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

            maxx = max(maxx, maxnewx) 
            maxy = max(maxy, maxnewy) 

            minx = min(minx, minnewx)
            miny = min(miny, minnewy)
            
     
    maxx = maxx*1.1
    maxy = maxy*1.1
    minx = minx*1.1
    miny = miny*1.1
    
    fig, axes = plt.subplots(nrows=norows, ncols=nocols, figsize=(21,29.7))
    plt.suptitle(file + " - Campbell diagrams", fontsize=48)

    no=0
    for breathno in breaths: #for no in range(1, nobreaths+1):
        no += 1
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
            ax.set_title("Breath #" + str(no),fontweight="bold", size=20)
                         
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
            
            wobelpolygon = Polygon([[p[0], p[1]] for p in [eelv, eilv, [eilv[0], eelv[1]]]])
        
            GRAY = '#999999'
            elpatch = PolygonPatch(wobelpolygon, fc=GRAY, ec=GRAY, alpha=afill, zorder=1)
            ax.add_patch(elpatch)      
            

    if averages:
        savefile = pjoin(settings['outputfolder'], "plots", "All files – average Campbell.pdf")
    else:
        savefile = pjoin(settings['outputfolder'], "plots", file + ".Campbell.pdf")
        
    fig.savefig(savefile) 
    plt.close(fig)    
    return []



def saverawplots(suffix, file, rows, titles, ylabels, settings):   
    
    plt.ioff()
    norows = len(rows)
    fig, axes = plt.subplots(nrows=norows, ncols=1, sharex=True, figsize=(21,29.7))
    plt.suptitle(suffix, fontsize=48)    
    
    for i in range(0, norows):
        axes[i].grid(True)
        axes[i].plot(rows[i], linewidth=0.5)
        axes[i].set_title(titles[i],fontweight="bold", size=20)
        axes[i].set_ylabel(ylabels[i],fontweight="bold", size=16)
    
    axes[i].set_xlabel(r'$observation #$', size=16)

    savefile = pjoin(settings['outputfolder'], "plots", file + "." + suffix + ".pdf")
    fig.savefig(savefile)
    plt.close(fig)
  
def savedataaverage(totals, settings):   
    savefile = pjoin(settings['outputfolder'], "data", "Average breathdata.xlsx")
    totals.to_excel(savefile, sheet_name='Output')

def savedataindividual(file, breaths, settings):   
    
    mechs = []
    for breathno in breaths:   
        breath= breaths[breathno]
        if not breath["ignored"]:
            dfmech = pd.DataFrame(breath["mechanics"], index=[0])
            dfwob = pd.DataFrame(breath["wob"], index=[0])
            dfmech.insert(loc=0, column="breath_no", value=breath["number"])
            dfmech = dfmech.join(dfwob, how="outer", sort=False)
            
            if len(settings["columns_entropy"])>0:
                dfent = pd.DataFrame(breath["entropy"]).transpose()
                dfent.columns = ["sample_entropy_col_" +  str(settings['columns_entropy'][x]) for x in range(0, len(settings["columns_entropy"]))]
                dfmech = dfmech.join(dfent, how="outer", sort=False)
                
            if len(mechs)>0:
                mechs = pd.concat([mechs, dfmech], sort=False)
            else:
                mechs = dfmech
                    
    savefile = pjoin(settings['outputfolder'], "data", file + ".breathdata.xlsx")
    
    ret = pd.DataFrame(mechs.mean())
    ret = ret.T
    ret = ret.drop(columns="breath_no")
    ret.insert(loc=0, column="file", value=file)
    
    mechs.to_excel(savefile, sheet_name='Output')
    return ret
    
def analyse(settings):
    print('Loading data...')
    filepath = pjoin(settings['inputfolder'], settings['files'])
    files = [f for f in glob.glob(filepath)]
    files.sort()

    print('Processing ' + str(len(files)) + ' file(s):')
    
    averagebreaths = {}
    averagecalcs = []
    
    for fi in files:
        file  = os.path.abspath(fi)
        filename = ntpath.basename(file)
        print('\nProcessing \'' + file + '\'')

        flow, volume, poes, pgas, pdi, entropycolumns = load(file, settings)
        
        if (settings['savedataviewraw']):
            print('\t\tSaving raw data plots...')
            saverawplots("Raw data", ntpath.basename(file), [flow, volume, poes, pgas, pdi], 
                         ['Flow', 'Volume (uncorrected)', 'Oesophageal pressure', 'Gastric pressure', 'Trans diaphragmatic pressure'],
                         [r'$L/s$', r'$L$', r'$cm H_2O$', r'$cm H_2O$', r'$cm H_2O$'],
                         settings)

        print('\t\tTrimming to whole breaths...')
        flow, volume, poes, pgas, pdi = trim(flow, volume, poes, pgas, pdi, settings)
        
        if (settings['savedataviewtrimmed']):
            print('\t\tSaving trimmed data plots...')
            saverawplots("Trimmed data", ntpath.basename(file), [flow, volume, poes, pgas, pdi], 
                         ['Flow', 'Volume (drift corrected)', 'Oesophageal pressure', 'Gastric pressure', 'Trans diaphragmatic pressure'],
                         [r'$L/s$', r'$L$', r'$cm H_2O$', r'$cm H_2O$', r'$cm H_2O$'],
                         settings)
        
        print('\t\tDrift correcting...')
        uncorvol = volume
        zerovol = zero(volume)
        volume = correctdrift(zerovol, settings)
        if (settings['savedataviewtrimmed']):
            print('\t\tSaving drift corrected data plots...')
            saverawplots("Drift corrected volume", ntpath.basename(file), [flow, uncorvol, zerovol, volume], 
                         ['Flow', 'Uncorrected volume', 'Zeroed volume', 'Drift corrected volume'],
                         [r'$L/s$', r'$L$', r'$L$', r'$L$'],
                         settings)
      
        #Add any post processing that should be performed on the entire data file
        #here.
        
        print('\t\tDetecting breathing cycles...')
        breaths = separateintobreaths(filename, flow, volume, poes, pgas, pdi, entropycolumns, settings)
        
        #Add any post processing that should be performed for each breath here.
        
        print('\t\tCalculating mechanics and WOB...')
        vefactor = 60/(len(flow)/settings["samplingfrequency"]) #Calculate multiplication factor for VE for this file.
        
        bcnt = len(breaths) 
        for bc in settings["breathcounts"]:
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
    
        if (settings['savepvindividualworkload']):
            print('\t\tSaving detailed pressure/volume overview plots...')
            savepvbreaths(filename, breathmechswob, flow, volume, poes, pgas, pdi, settings, False)
         
        if (settings['savebreathbybreathdata']):
            print('\t\tSaving individual breath data...')
        
            av = savedataindividual(filename, breathmechswob, settings)
               
            if len(averagecalcs) > 0:
                averagecalcs = pd.concat([averagecalcs, av])
            else:
                averagecalcs = av

    #Save data and average breath plots
    if (settings['savepvaverage']):
        print('\n\tSaving average plots...')
        savepvbreaths(filename, averagebreaths, flow, volume, poes, pgas, pdi, settings, True)  
    
    
    if (settings['saveaveragedata']):
        print('\n\tSaving total average data...')
        savedataaverage(averagecalcs, settings)
        
        
    print('\nFinished analysing \'' + file + '\'.\n')
    return 
    
    #TODO: save file version in output