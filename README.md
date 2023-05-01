# Respiratory mechanics, work of breathing and EMG entropy calculations
------------------------------------------------------------------------ 
_(c) Copyright 2019 Emil Schwarz Walsted (emilwalsted@gmail.com), ORCID [0000-0002-6640-7175](https://orcid.org/0000-0002-6640-7175)_

[![DOI](https://zenodo.org/badge/191052676.svg)](https://zenodo.org/badge/latestdoi/191052676)

** Current version: v1.0.0 ** 

![Data trimming](https://github.com/emilwalsted/respmechdocs/blob/master/img/githubimage.png)

# About RespMech
This is a port/rewrite of my MATLAB code from 2016, with some changes and additions. The code calculates 

* respiratory mechanics
* in- and expiratory work of breathing 
* Sample Entropy (e.g. from diaphragm EMG)

from a time series of recordings (e.g. obtained via LabChart).

# Prerequisites
The code runs on Python 3.6+. If you don't have Python installed, I would recommend that you download and install the free [Anaconda](https://www.anaconda.com/distribution/) – An all-inclusive package for scientists that contains everything you need. I prefer running/editing with Spyder (choose from the Anaconda main menu).

# Getting started
The file **respmech.py** contains the actual analysis code. Start by downloading it. You should then create another Python file for each setting/project which contains the analysis settings for that particular project (i.e. points to input files etc.). Make a copy of the template project file **example.py** and rename/modify to reflect your setup and analysis (see below).

Specifically, Input and Output location details must be provided in the settings template in order to render the code functional (see details below). Additional details (e.g. samplingfrequency; file format) are used to further tailor the code to the settings of your data acquisition file and resulting exported analysis files.

### Running the code
You can run your analysis by opening your modified **example.py** in e.g. Spyder (from within the Anaconda main menu) and selecting "Run". If you prefer running the code in a command line window, type

_Windows:_
```
python C:\path\to\my-settings-file.py
```
_Mac/Linux:_
```
python /path/to/my-settings-file.py
```

## Input and Output: Data locations and format
Currently, supported input formats are: MATLAB, Excel and CSV. You can extend this yourself by modifying the _load()_ function in **respmech.py**.

First, you need to modify the settings file path below to point to **respmech.py**. This is to ensure that you are in control of exactly which version of the analysis code you use. _Note: the examples below use Unix-style paths – if you are on Windows, use 'C:\\\\folder\\\\folder...' style. NOTE: Backslashes __must__ be double._

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
The output folder must have two subfolders named 'data' and 'plots', respectively.

_Note for MATLAB input format:_ MATLAB format files exported from Windows and Macintosh versions of LabChart have different formats. You must specify if the input files were created using a Windows or Macintosh version:
```
    'matlabfileformat': 2,  #Only relevant if input data are in MATLAB format. 1=MATLAB for Windows, 2=MATLAB for Mac.
```

## Data recording requirements: 
The respmech.py code analyses a time series of respiratory physiological measurements. Your input data do _not_ need to be a specific length of time, but as some output values are calculated as per-time (e.g. minute ventilation), you must specify the sampling frequency of the input data for the calculations to be correct:

```
'samplingfrequency': 4000, #No. of data recordings per second
```

The code analyses data breath-by-breath and it is imperative that input data **starts with the last part of an expiration and end with the first bit of an inspiration**. The data recording is automatically trimmed at the beginning and end, to start at exactly the first inspiration and end at exactly the last expiration. 


![Data trimming](https://github.com/emilwalsted/respmechdocs/blob/master/img/datatrim1.png)


Breath segmenting is performed automatically by joining an inspiration with the following expiration. The _flow_ signal is used to determine the transition between inspiration and expiration. To allow for a bit of "wobbly" flow around 0 (as it often happens in quiet breathing), a buffer mechanism mitigates these artefacts. You can adjust the buffer length (i.e. in how many observations the "wobbling" could take place - value will vary based on sampling frequency and breathing frequency):

```
'breathseparationbuffer': 800,
```
**Flow and volume adjustments**
For correct analysis the code assumes that the flow signal is negative on inspiration and positive on expiration. Should your data recording have it the other way around, you can use the setting _inverseflow_ to inverse the input signal:

```
'inverseflow': False, #True: For calculations, inspired flow should be negative. This setting inverses the input flow signal. Default is False.
```

If your input data do not contain a volume signal, RespMech.py can integrate the flow signal for you:

```
'integratevolumefromflow': True, #True: Creates the volume signal by integrating the (optionally reversed) flow signal. False: Volume is specified in input data.
```

If your input data does contain a volume signal, it must be _inspired volume_ for analysis. If your volume signal is expired volume, you can inverse this using the setting

```
'inversevolume': False, #True: For calculations, inspired volume should be positive and expired should be negative. This setting inverses the volume input signal. Default is False.
```

***Volume drift*** (often as a result of integrating volume from flow) can be automatically corrected by the respmech.py by specifying 

```
'correctvolumedrift': True, #True: Correct volume drift. False: Do not correct volume drift. Default is True
```

A flow- and volume corrections overview for each file is created in the output/plots folder, allowing you to visually verify the corrections made:

![Data trimming](https://github.com/emilwalsted/respmechdocs/blob/master/img/volumedrift1.png)

## Specifying input columns
You need to specify in your settings which columns in your data contain the required individual physiological measurements:
```
    'column_poes': 8,     #Column number containing oesophageal pressure
    'column_pgas': 9,    #Column number containing gastric pressure
    'column_pdi': 6,     #Column number containing transdiaphragmatic pressure
    'column_volume': 23, #Column number containing (inspired) volume, BTPS
    'column_flow': 20,   #Column number containing flow
    'columns_entropy': [1,2,3,4,5],   #The data columns containing EMG signals to calculate EMG entropy from, e.g. [4,5,6,7,8]. Leave as [] to skip EMG calculation.

```

## Work of breathing
Two options are available for calculating WOB: Breath-by-breath, calculation WOB from individual breath Campbell diagrams which are then averaged or, alternatively, first create an averaged pressure/volume loop of the breaths in the file, then calculate WOB from a Campbell diagram made from the averaged p/v loop. The two methods give quite similar results but in the case of irregular breaths the latter might be more robust. Also, you can specify the number of observations for resampling when averaging the loop. Unless each breath is very short or your sampling frequency very low, you can leave this at 500.

```
'calcwobfromaverage': True, #False: calculates WOB for each breath, then averages. True: Averages breaths to produce an averaged Campbell diagram, from which WOB is calculated.
'avgresamplingobs': 500, #Downsampling to # of observations for breath P/V averaging. A good default would be sampling frequency divided by 8-10. Must be lower than the lowest # of observation in any inspiration or expiration in the file.
```

## Entropy calculation
Sample Entropy<sup>[1](#sampenref1),[2](#sampenref2)</sup> here is calculated for the selected input columns/channels on a per-breath basis, and averaged as the other measurements. If you selected to calculate entropy for one or more columns, you must specify the parameters for entropy calculation in your settings:

```
'entropy_epochs': 2, #Epoch parameter (m) to use with entropy calculation. Default is 2.
'entropy_tolerance': 0.1, #Tolerance (r) parameter to use with entropy calculation. This value is multiplied with the SD of the data. Default is 0.1.
```

_<a name="sampenref1">1</a>) Lozano-García M, Leonardo, Moxham J, Rafferty F., Torres A, Jolley CJ, Jané R. Assessment of Inspiratory Muscle Activation using Surface Diaphragm Mechanomyography and Crural Diaphragm Electromyography. doi:10.1109/EMBC.2018.8513046._

_<a name="sampenref2">2</a>) Aboy M, David, Austin D, Pau. Characterization of Sample Entropy in the Context of Biomedical  Signal Analysis. 2007. doi:10.1109/IEMBS.2007.4353701._



## Output data
Output data are saved as Excel spreadsheets in the _'data'_ subfolder of the output fold    er, while diagnostic plots are saved in the _‘plots’_ subfolder of the output folder. 

### Numeric Data
There are two options for numeric data output: The overall averages from each file, merged together in a single spreadsheet, and the individual breath-by-breath values for each file are saved in a separate spreadsheet per input file. You can turn these outputs on/off using these settings and choose to output both or either option:
```
# Data input/output
'saveaveragedata': True, #False: don't save, True: save.
'savebreathbybreathdata': True, #False: don't save, True: save.
```

![Data trimming](https://github.com/emilwalsted/respmechdocs/blob/master/img/outputdata.png)

A detailed description of the output variables and how they are calculated is available here: [Output variable description (TODO)](http://TODO).


### Diagnostic plots
A number of diagnostic plots allows you to interrogate the basis and validity of the calculations. For example, your recordings might include breaths that you wish to exclude from analysis (e.g. IC manoeuvres or coughs). The diagnostic plots are saved in the output folder, in the _'plots'_ subfolder. The following plots are available for each input file:

* Raw data plot of Flow, Volume, Pes, Pga, and Pdi.
* Trimmed data plot of the above
* Volume drift correction plot (as shown above)
* Breath-by-breath Campbell diagrams 

Using these diagnostic plots you can then determine the breath numbers you wish to exclude (if any), and enter this in the exclusion list:

```
#Exclude individual breaths from analysis, if appropriate. Takes input in the format [['file1.mat', [04, 07]], ['file2.mat', [01]]]. If no breaths should be excluded, set to []. NOTE: File name is case sensitive!
'excludebreaths': [
                  ['Testsubject - Rest.mat', [5,7]],
                  ['Testsubject - 000W.mat', [2]],
                  ['Testsubject - 020W.mat', [2,3,4,10]],
                  ['Testsubject - 040W.mat', []],
                  ['Testsubject - 060W.mat', [1,4,5,6,7]], 
                  ['Testsubject - 080W.mat', [1,2,7,9]],
                  ['Testsubject - 100W.mat', [6,7]],
                  ['Testsubject - 120W.mat', [1,2,3,11]],
                  ['Testsubject - 140W.mat', [3,10,11,13]],
                  ['Testsubject - 160W.mat', [1,3,11]],
                  ['Testsubject - 180W.mat', [1,6,16]],
                  ['Testsubject - 200W.mat', []],
                  ['Testsubject - 220W.mat', []]                       
                  ],
```

This is an example of the individual breaths Campbell diagrams, including breaths excluded from analysis:

![Data trimming](https://github.com/emilwalsted/respmechdocs/blob/master/img/campbell.png)


In some instances (i.e. irregular breaths), the automatic breath count might not be accurate. In this case you can override the automatically detected breath count, to allow for correct calculation of per-time based output variables (such as minute ventilation (VE)):

```
'breathcounts': [
                ['Testsubject - Rest.mat', 5],
                ['Testsubject - 140W.mat', 11]            
                ],
```


---

# License and usage
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

[Read the entire licence here.](LICENSE)

## Note to respiratory scientists
I created this code for the purpose of my own work and shared it hoping that other researchers working with respiratory physiology  might find it useful. If you have any questions or suggestions that will make the code more useful to you or generally, please drop me an email.

### How do I cite this code in scientific papers – and should I?
It is up to you, really. Personally I am a fan of transparency and Open Source / Open Science and I would appreciate a mention. This will also make readers of your papers aware that this code exists – if you found it useful, perhaps they will too.

Every released version of the code has its own individual DOI you can use for citation. You can reference the latest version by using this DOI: [![DOI](https://zenodo.org/badge/191052676.svg)](https://zenodo.org/badge/latestdoi/191052676) or a specific version using the DOI of that version. Click the DOI button to see a list of all available versions with a DOI.

An example of a citation could look like this:

_[...] were calculated using the Python package RespMech (ES Walsted, RespMech v1.0, 2019, https://github.com/emilwalsted/respmech/, DOI: 10.5281/zenodo.3270826) [...]_

## Note to software engineers/computer scientists
This code is far from as concise or computationally efficient as it could be. This is in part because it is a port of old code I wrote just for myself back in the day. I have since come to realise that my code has 'travelled' among researchers and although this is flattering, I would prefer that everyone also got the fixes, updates and improvements, and also proper instructions for usage. This is the first step towards a more generally useful code library and I have focused on some degree of readability in an attempt to enable respiratory scientists with basic programming skills to understand, debug and modify/extend the code themselves. Future versions might be properly restructured and offered as a PIP package but for now, this is good enough for jazz.
