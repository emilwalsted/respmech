# Respiratory mechanics, work of breathing and EMG entropy calculations
------------------------------------------------------------------------ 
_(c) Copyright 2019 Emil Schwarz Walsted (emilwalsted@gmail.com), ORCID [0000-0002-6640-7175](https://orcid.org/0000-0002-6640-7175)_

This is a port/rewrite of my MATLAB code from 2016, with some changes and additions.

The code calculates respiratory mechanics and/or in- and expiratory work of breathing and/or diaphragm EMG entropy from a time series of pressure/flow/volume recordings (e.g. obtained using LabChart).

# Usage

Start by making a copy of the example file 'example.py' and modify to reflect your setup and analysis.

## Input data

Currently, supported input formats are: MATLAB, Excel and CSV. You can extend
this yourself by modifying the 'load()' function in respmech.py.

First, you need to modify the path below to point to **respmech.py**. This is to ensure that you are in control of exactly which version of the analysis code you use.

```
#Modify to the location of your respmech.py:
spec = importlib.util.spec_from_file_location("analyse", "/Users/emilwalsted/Documents/Respiratory mechanics/respmech.py")
```

Next, specify input folder and file mask, and the output path where the results of the analysis will be saved. 

```
'inputfolder': "/Users/emilwalsted/Documents/Respiratory mechanics/test/input/MATLAB Mac",
'files': "*.mat",
    
'outputfolder': "/Users/emilnielsen/Documents/Medicin/Forskning/Code/Respiratory mechanics/test/output", 
```
_Note: The output folder must have two subfolders named 'data' and 'plots', respectively._

**Data recording requirements**: This code analyses a time series of respiratory physiological measurements. The code analyses data breat-by-breath and it is imperative that input data **starts with the last part of an expiration and end with the first bit of an inspiration**. The data recording is automatically trimmed at the beginning and end, to start at exactly the first inspiration and end at exactly the last expiration. 


![Data trimming](https://github.com/emilwalsted/respmechdocs/blob/master/img/datatrim1.png)


Breath segmenting is performed automatically by joining an inspiration with the following expiration. The _flow_ signal is used to determine the transition between inspiration and expiration. To allow for a bit of "wobbly" flow around 0 (as it often happens in quiet breathing), a buffer mechanism mitigates these artefacts. You can adjust the buffer length (i.e. in how many observations the "wobbling" could take place):

```
'breathseparationbuffer': 800,
```

**Volume drift** (often as a result of integrating volume from flow) is automatically corrected as shown here:

![Data trimming](https://github.com/emilwalsted/respmechdocs/blob/master/img/volumedrift.png)


## Output data
Output data are saved as Excel spreadsheets in the _'data'_ subfolder of the output folder. There are two options for data output: The overall averages from each file, merged together in a single spreadsheet, and the individual breath-by-breath values for each file are saved in a separate spreadsheet per input file. You can turn these outputs on/off using these settings:

```
# Data input/output
'saveaveragedata': True, #False: don't save, True: save.
'savebreathbybreathdata': True, #False: don't save, True: save.
```

### Diagnostic plots
A number of diagnostic plots allows you to inspect the basis of the calculations. In your recordings there might be breaths that you wish to exclude from analysis (e.g. IC manoeuvres or coughs). The diagnostic plots are saved in the output folder, in the _'plots'_ subfolder. The following plots are available for each input file:

* Raw data plot of Flow, Volume, Pes, Pga, and Pdi.
* Trimmed data plot of the above
* Volume drift correction plot (as shown above)
* Breath-by-breath Campbell diagrams 

Using these diagnostic plots you can then determine the breath numbers you wish to exclude (if any), and enter this in the exclusion list:

```
#Exclude individual breaths from analysis, if appropriate. Takes input in the format [['file1.mat', [04, 07]], ['file2.mat', [01]]]
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
```

---

# About this code
## Note to respiratory scientists

I created this code for the purpose of my own work and shared it hoping that 
other researchers working with respiratory physiology  might find it useful. 
If you have any suggestions that will make the code more useful to you or 
generally, please email me to discuss.

### How do I cite this code in scientific papers – and should I?
It is up to you, really. Personally I am a fan of transparency and Open Source / Open Science and I would appreciate a mention. This will also make readers of your papers aware that the code exists – if you found the code useful, maybe they will too.

Every release of the code has a DOI you can use for citation. The current release DOI is: **(TODO INSERT DOI BAGDE)**

_An example of a citation could look like this:_
```
[...] were calculated with the Python package RespMech (ES Walsted, RespmMech version 1.0, 2019, https://github.com/emilwalsted/respmech/, DOI: xxxx) [...]
```

## Note to software engineers/computer scientists

This code is far from as concise or computationally efficient as it 
could be. I have focused on some degree of readability in an attempt to enable 
respiratory scientists with basic programming skills to understand, debug and 
modify/extend the code themselves.

# License and usage
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
 
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

[Read the entire licence here.](LICENSE)
