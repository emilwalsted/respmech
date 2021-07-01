#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:11:39 2019

@author: Emil S. Walsted 
------------------------------------------------------------------------ 
EXAMPLE: Respiratory mechanics and work of breathing calculation
------------------------------------------------------------------------ 
(c) Copyright 2019 Emil Schwarz Walsted <emilwalsted@gmail.com>

This file demonstrates the use of "respemech.py". Copy and modify to your own 
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
spec = importlib.util.spec_from_file_location("analyse", "/Users/emilnielsen/Documents/Medicin/Forskning/Code/RespMech/respmech.py")

m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
settings = {
    "input": {
        "inputfolder": "/Users/emilnielsen/Documents/Medicin/Forskning/Code/Respiratory mechanics/txtimport",
        "files": "33_Healthy.txt", #Filename or mask (e.g. "*.csv" for all CSV files in folder)
        "format": {
            #General settings
            "samplingfrequency": 2000, #No. of data recordings per second
        },
        "data": {
            "column_poes": 7,     #Column number containing oesophageal pressure
            "column_pgas": 8,    #Column number containing gastric pressure
            "column_pdi": 10,     #Column number containing transdiaphragmatic pressure
            "column_volume": 13, #Column number containing (inspired) volume, BTPS. Exclude this and specify "integratevolumefromflow: True" to instead obtain volume by integrating the flow signal.
            "column_flow": 9,   #Column number containing flow
            "columns_emg": [2,3,4,5,6],   #The data columns containing EMG signals to calculate RMS from, e.g. [4,5,6,7,8]. Leave as [] to skip EMG calculation.
            "columns_entropy": [2,3,4,5,6],   #The data columns to perform entropy calculation on, e.g. [4,5,6,7,8]. Leave as [] to skip entropy calculation.
        }
    },
    "processing": {
        "mechanics": {
            "breathseparationbuffer": 400, #Number of measurements to average over for breathing cycle detection (default=800). Will depend on your sampling frequency.
            
            #Calculations:
            "inverseflow": False, #True: For calculations, inspired flow should be negative. This setting inverses the input flow signal. Default is False.
            "integratevolumefromflow": True, #True: Creates the volume signal by integrating the (optionally reversed) flow signal. False: Volume is specified in input data.
            "inversevolume": False, #True: For calculations, inspired volume should be positive and expired should be negative. This setting inverses the volume input signal. Default is False.
            "correctvolumedrift": True, #True: Correct volume drift. False: Do not correct volume drift. Default is True
            "correctvolumetrend": False, #True: Correct volume  for trend changes. False: Do not correct. Default is False
            "volumetrendadjustmethod": "linear",
            "volumetrendpeakminheight": 0.5, #For trend adjustment: How tall (absolute value) a volume peak should at least be, to be considered a peak. Default is 0.8
            "volumetrendpeakminwidth": 0.005, #For trend adjustment: How far apart (seconds) should peaks at least be, to be considered a peak. Default is 0.01
            "volumetrendpeakmindistance": 0.25, #For trend adjustment: How far apart (seconds) should peaks at least be, to be considered a peak. Default is 0.25
            "calcwobfromaverage": True, #False: calculates WOB for each breath, then averages. True: Averages breaths to produce an averaged Campbell diagram, from which WOB is calculated.
            "avgresamplingobs": 500, #Downsampling to # of observations for breath P/V averaging. A good default would be sampling frequency divided by 8-10. Must be lower than the lowest # of observation in any inspiration or expiration in the file.
            
            #Exclude individual breaths from analysis, if appropriate. Takes input in the format [["file1.mat", [04, 07]], ["file2.mat", [01]]]
            #If no breaths should be excluded, set to []. NOTE: File name is case sensitive!
            "excludebreaths": [["33_Healthy.txt", [7]]],
            
            #Breath count is detected automatically. However, in some cases, the breathing pattern can lead to erroneous breathing frequencies,
            #which will affect the mechanics calculations. In this case you can override the autumatically detected breath count in a specific
            #file by adding below:
            "breathcounts": [],
        },
        "emg": {
            "rms_s": 0.050, #The size of the rolling window centered around the data point, to calculate EMG RMS from (in seconds). Default is 0.05s.
            "remove_ecg": True, #Perform ECG removal before calculating RMS. Default is True.
            "column_detect": 4, #Which of the EMG columns to use for ECG detection. Default is the first (0). Use the data column where the ECG is most prominent.
            "minheight": 0.0005, #(For peak detection): The minimum height (in volt) of an R wave. Default is 0.001.
            "mindistance": 0.5, #The minimum distance between R waves (in seconds). Default is 0.25
            "minwidth": 0.001, #The minimum width of R waves (in seconds). Default is 0.005
            "windowsize": 0.4, #Window size (in seconds) for averaging an ECG complex. Default is 0.8.
            "avgfitting": 5, #No. of passes to apply ECG removal. Default is 50.
            "passno": 10, #No. of passes to apply ECG removal. Default is 50.
            "remove_noise": True, #Perform EMG noise removal before calculating RMS. Default is True.
            "noise_profile": [["33_Healthy.txt", [3.6, 4]]], #Required for noise removal, for each file processed. Specifies the start- and end times (in seconds) for the noise profile (i.e. an area with no relevant EMG activity).
        },
         "entropy": {
            "entropy_epochs": 2, #Epochs used for entropy calculation. Default is 2.
            "entropy_tolerance": 0.1 #Tolerance (in standard deviations) used for entropy calculation. Default is 0.1 SD.
        },
    },
   
    "output": {
        "outputfolder": "/Users/emilnielsen/Documents/Medicin/Forskning/Code/Respiratory mechanics/txtimport/output", #Note: Output folder must have two subfolders named "data" and "plots", respectively.
        "data": {
            #Data input/output
            "saveaveragedata": True, #False: don"t save, True: save.
            "savebreathbybreathdata": True, #False: don"t save, True: save.
        },
        "diagnostics": {
            #Diagnostics plots (saved as PDF in output folder). False: don"t save, True: save.
            "savepvaverage": True,
            "savepvoverview": True,
            "savepvindividualworkload": True, 
            "savedataviewraw": True, 
            "savedataviewtrimmed": True, 
            "savedataviewdriftcor": True
        },
    }
}


#Perform the analysis. Output will be saved to the specified output folder. 
print("\nRunning analysis...\n")

outfolder = m.analyse(settings)
print("\n... Done. Find the results here:\n")
print(outfolder, "\n\n")
