#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:11:39 2019

@author: Emil S. Walsted 
------------------------------------------------------------------------ 
EXAMPLE: Respiratory mechanics and work of breathing calculation
------------------------------------------------------------------------ 
(c) Copyright 2019 Emil Schwarz Walsted <emilwalsted@gmail.com>

This file demonstrates the use of 'respemech.py'. Copy and modify to your own 
needs.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
 
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public$ License
along with this program.  If not, see <http://www.gnu.org/licenses/>
"""

import importlib.util

#Modify to the location of your respmech.py:
spec = importlib.util.spec_from_file_location("analyse", "/Users/emilnielsen/Documents/Medicin/Forskning/Code/Respiratory mechanics/respmech.py")

m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
settings = {
    
    #General settings
    'samplingfrequency': 4000, #No. of data recordings per second
    'matlabfileformat': 2,  #Only relevant if input data are in MATLAB format. 1=MATLAB for Windows, 2=MATLAB for Mac.
    'breathseparationbuffer': 800, #Number of measurements to average over for breathing cycle detection (default=800). Will depend on your sampling frequency.
   
    #Data input/output
    'saveaveragedata': True, #False: don't save, True: save.
    'savebreathbybreathdata': True, #False: don't save, True: save.

    'inputfolder': "/Users/emilnielsen/Documents/Medicin/Forskning/Code/Respiratory mechanics/test/input/MATLAB Mac",
    'files': "*.mat", #Filename or mask (e.g. "*.csv" for all CSV files in folder)
    
    'outputfolder': "/Users/emilnielsen/Documents/Medicin/Forskning/Code/Respiratory mechanics/test/output", #Note: Output folder must have two subfolders named 'data' and 'plots', respectively.

    'column_poes': 8,     #Column number containing oesophageal pressure
    'column_pgas': 9,    #Column number containing gastric pressure
    'column_pdi': 6,     #Column number containing transdiaphragmatic pressure
    'column_volume': 23, #Column number containing (inspired) volume, BTPS
    'column_flow': 20,   #Column number containing flow
    'columns_entropy': [1,2,3,4,5],   #The data columns containing EMG signals to calculate EMG entropy from, e.g. [4,5,6,7,8]. Leave as [] to skip EMG calculation.

    
    #Exclude individual breaths from analysis, if appropriate. Takes ibput in the format [['file1.mat', [04, 07]], ['file2.mat', [01]]]
    #If no breaths should be excluded, set to []. NOTE: File name is case sensitive!
    'excludebreaths': [
                       ['S07 -  Rest.mat', [5,7]],
                       ['S07 - 000W.mat', [2]],
                       ['S07 - 020W.mat', [2,3,4,5,6,7,8,9,10]],
                       ['S07 - 040W.mat', []],
                       ['S07 - 060W.mat', [1,4,5,6,7]], 
                       ['S07 - 080W.mat', [1,2,3,4,6,7,9]],
                       ['S07 - 100W.mat', [6,7]],
                       ['S07 - 120W.mat', [1,2,3,4,5,6,7,8,9,10,11]],
                       ['S07 - 140W.mat', [3,10,11,13]],
                       ['S07 - 160W.mat', [1,3,4,10,11]],
                       ['S07 - 180W.mat', [1,6,7,8,10,12,16]],
                       ['S07 - 200W.mat', []],
                       ['S07 - 220W.mat', []]                       
                       ],
    
    #Breath count is detected automatically. However, in some cases, the breathing pattern can lead to erroneous breathing frequencies,
    #which will affect the mechanics calculations. In this case you can override the autumatically detected breath count in a specific
    #file by adding below:
    'breathcounts': [
                       ['S07 -  Rest.mat', 5],
                       ['S07 - 000W.mat', 3],
                       ['S07 - 020W.mat', 6],
                       ['S07 - 040W.mat', 8],
                       ['S07 - 060W.mat', 5],
                       ['S07 - 080W.mat', 8],
                       ['S07 - 100W.mat', 9],
                       ['S07 - 120W.mat', 11],
                       ['S07 - 140W.mat', 11],
                       ['S07 - 160W.mat', 15],
                       ['S07 - 180W.mat', 13],
                       ['S07 - 200W.mat', 19],
                       ['S07 - 220W.mat', 20]                   
                       ],
    #Calculations:
    'calcwobfromaverage': True, #False: calculates WOB for each breath, then averages. True: Averages breaths to produce an averaged Campbell diagram, from which WOB is calculated.
    'avgresamplingobs': 500, #Downsampling to # of observations for breath P/V averaging. A good default would be sampling frequency divided by 8-10. Must be lower than the lowest # of observation in any inspiration or expiration in the file.
    
    #Diagnostics plots (saved as PDF in output folder). False: don't save, True: save.
    'savepvaverage': True,
    'savepvoverview': True,
    'savepvindividualworkload': True, 
    'savedataviewraw': True, 
    'savedataviewtrimmed': True, 
    'savedataviewdriftcor': True
}


#Perform the analysis. Output will be saved to the specified output folder. 
print('\nRunning analysis...\n')
m.analyse(settings)
print('\n... Done. Find the results here:\n')
print(settings['outputfolder'])
