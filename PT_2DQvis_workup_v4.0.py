# -*- coding: utf-8 -*-
"""

Code to process raw 2DES data generated using PhaseTech 2DQvis spectrometer in COSY PewPew Lab. 

Raw data is in the form of .scan files and metadata in the form of .info files.

Generates processed files that are to be used for data analysis.

Data analysis can be performed on the processed .pickle files using 2DES_GUI.py

Author: Sankaran Ramesh
Email: rame0010@e.ntu.edu.sg, sankaranramesh17@gmail.com



This code has been modified from the code for data processing written by PhaseTech.

company: PhaseTech Spectroscopy, Inc. <http://phasetechspectroscopy.com/>

email: support@phasetechspectroscopy.com


"""

"""###################################################################################################################
# MODULE IMPORT 
###################################################################################################################"""
import sys
import time
t_start = time.time()
import os
from os import listdir
#import seaborn as sns
import json
import pickle
import csv
import pandas as pd
import numpy as np # for array and matrix calculations
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PT_data_workup_functions import mlList
from PT_data_workup_functions import backgroundCorrect
from PT_data_workup_functions import hammingWindow
from PT_data_workup_functions import rowWiseLoop
from PT_data_workup_functions import fft_deadpx_filter,interpolate_deadpx
from matplotlib import animation
from tkinter import *
from tkinter import filedialog
from tkinter.ttk import *

polyFitRange = list()
######################################################################################################################################################################################"""

    
"""###################################################################################################################
    
COMMENT 1:
    
    Description: sample code for loading, processing, plotting, and saving 2D 
    spectral data from 2DQVis spectrometers.

    This programs runs in two modes: 
    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>  AUTO MODE  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    Program will prompt the user to choose the .info file, program reads all the necessary data 
    and generates processed .pickle file.
            
    Useful when the 2D data at different waiting times is in the form of a single scan
             
    Currently configured to handle linear set of waiting time delays
             
    Required conditions for raw data: 
        1. File should have set of waiting times in increasing linear order
        2. raw data files should be named "YYYYMMDD#<filename>_T<nn>.scan"
        3. metadata file should be named "YYYYMMDD#<filename>.info"

    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>  MANUAL MODE  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    User inputs all the parameters, including folder, data set and delays 
            
    Needs to be done when the 2D data files at different waiting times are in different scans
            
    Required Steps:
        1. Specify the file details containing 2D data. This includes:
        
            fileDirectory         : String - File Directory in Memory
            str_date              : String - Date YYYYMMDD is in the default file name format <YYYYMMDD>#<NN>_T<dT2>    
            str_data              : String List - Data file number NN is in the default file name format <YYYYMMDD>#<NN>_T<dT2> (List for multiple file processing)
            str_delaydata         : String List - Delay times dT2 corresponding to file number NN is in the default file name format <YYYYMMDD>#<NN>_T<dT2> (List for multiple file processing)
            delays                : Int List - Values of dT2, in fs
            sample_name           : Name for saving output file
            binning               : detector binning of pixels (1=1024 pixels, 2=512 pixels, 3=256 pixels, 4= 128 pixels)    

###################################################################################################################"""


"""###################################################################################################################

1. USER SECTION : Edit the following lines as required:

###################################################################################################################"""

sample_name="MAPbBr3"                         # give the sample name, will be used while saving processed data

Automode=True                                 # When automode, the program automates data processing, 
                                              # only need to select the related .info file. See comment 1 for details
specDisplay = False                           # plot 2D spectrum after FT and processing
generate_datacube=True                        # generates the processed data as .pickle file

str1='_T'                                     # DONT CHANGE
str2='_t2_'                                   # DONT CHANGE
binning=1                                     # binning setting in PhaseTech software for the data    
freqRangePump = [520,660]                     # Enter the range of pump frequency you want to process - if you leave this range too broad this will finally be limited further by experimental parameters
freqRangeProbe = [500,700]                    # Enter the range of probe frequency you want to process


"""Details for 2D Plot"""
indexRangeProbe = [470,800]                   # DONT CHANGE - ONLY FOR UNCALIBRATED PLOTS px display range for the probe axis   
indexRangePump = [0,1000]                     # DONT CHANGE - ONLY FOR UNCALIBRATED PLOTS px display range for the pump axis
symmetricContours = True                      # plots with contours and a colormap symmetric around 0
nContours = 10                                # number of contours to plot
manualContourRange =True*[-0.25, 0.25]
showProjections = True                        # adds projections along each frequency axis

"""PIXEL - WAVELENGTH CALIBRATION"""

""" for calib_method = 0 (Manual px-nm calibration of detector) 

    - OLD PARAMETERS - TO BE USED TO AS REQUIRED """

"""Parameters for PT sample data"""
# calibFreqs =  [1980, 1960]                                                    # in wavemnumbers
# calibPixels = [50, 70]
# freqRangePump = [1929, 2021]                                                  # display range for the pump axis
# freqRangeProbe = [1929, 2021]                                                 # display range for the probe axis

"""Parameters for 20210928   mono: 620 nm"""
# calibFreqs =  [1e7/519.5,1e7/567.8,1e7/638.4] #[1e7/519.5,1e7/567.8]638.4
# calibPixels = [330,417,545]#[330, 417] 545
# freqRangePump = [505,607]                                         # display range for the pump axis - for diagMethod 1
# freqRangeProbe = [505,607]                                        # display range for the probe axis - for diagMethod 1

"""Parameters for 20211227   mono: 700 nm"""
calibFreqs =  [566.5,655.6] #[1e7/519.5,1e7/567.8]638.4
calibPixels = [270,431]#[330, 417] 545

""" END OF OLD PARAMETERS """       
######################################################################################################################################################################################"""


"""###################################################################################################################

2. ADVANCED USER SECTION

    Expand the comments below for further details
###################################################################################################################"""

"""############################################################################

COMMENT 2:
    
    Data Saving Options
    
    it is possible to use either pickle or json 
    
    pickle (PREFERRED OPTION FOR DATA ANALYSIS USING 2DES_GUI) : 
        
    saves a dictionary as a non-human readable pickle binary, but handles
    numpy ndarrays (how we store matrices) with no extra hassle and easily 
    reloads (unpickle) into python. 
    The pickle module is not secure. Only unpickle data you trust.
    see: https://docs.python.org/3/library/pickle.html for more information
    
    json: 
    
    saves a dictionary as a human readable json file, but numpy ndarrays
    must be converted to nested lists before saving. when reloading into python,
    you would need to use something like:
        
    np.asarray(dictionaryname["arrayname"])
    
    to restore the numpy ndarray for use in python

############################################################################"""
    
start = 0                                       # starting index for files to average

"""TIME DOMAIN PROCESSING OPTIONS"""
fftLength = 0                                   # specify FFT length? if '0', fftLength will be auto determined.
apodizeData = True                              # Apply a Hamming filter to data 
fftAxis = 0

"""Units for Frequency Axis"""
calibUnits = 'wn'                               # options : 'wn' for wavenumber (cm-1) or 'nm' for nanometer
calibUnits = 'nm'                                           

"""contour plot colormap options"""
plotColorMap = 'bwr'                               # default color map for the contour plot
symmetricContoursColorMap = 'bwr'                  # color map for the contour plot with symmetricContours = True
colorbar = True                                    # display a colorbar
swapAxes = False                                   # places prob freq on vertical axis

"""Manually control axes aspect ratio for 2D plot"""
manualAxisAspect = 'equal'                         # options: False, 'auto', 'equal', or a value

"""Data Saving Options"""
flag_save2D = False
flag_save3D = True
save_format_2D = 'csv'                             # options: 'json' 'pickle' 'csv'
save_format_3D = 'pickle'                          # options: 'json' 'pickle' (pickle works for sure!)

"""FIGURE SAVING"""
flag_save_fig = False                              # turns on or off saving the figure as an image

"""BACKGROUND CORRECTION"""                   # DONT CHANGE
bkgdCorrect = True
#polyFitRange = [3:6 124:126] MATLAB SYNTAX
polyFitRange += mlList(1, 20)
polyFitRange += mlList(100, 127)
polyFitOrder = 0

######################################################################################################################################################################################"""

"""###################################################################################################################

3. METADATA READING SECTION : Edit the following section ONLY IF YOU ARE NOT USING AUTOMODE:
    
                         Follow the examples commented below IN THE IF LOOP to use the code in MANUAL MODE

###################################################################################################################"""


if(not Automode):
    
    fileDirectory=os.getcwd()    
    
    
    
    timestep=4
    last_delay=1000
    rotating_frame=18000            #in cm-1
    timestep*=1e-15
    last_delay*=1e-15
    rotating_frame=1E7/(rotating_frame)
    mono_center_wl=700
    
    
    str_date="20211227" 
    
    
    
    str_data=["MAPbBr3_2Dscan_cWL580nm_GVD-7.6k01","MAPbBr3_2Dscan_cWL580nm_GVD-7.6k01","MAPbBr3_2Dscan_cWL580nm_GVD-7.6k01",
              "MAPbBr3_2Dscan_cWL580nm_GVD-7.6k01","MAPbBr3_2Dscan_cWL580nm_GVD-7.6k01","MAPbBr3_2Dscan_cWL580nm_GVD-7.6k01",
              "MAPbBr3_2Dscan_cWL580nm_GVD-7.6k01","MAPbBr3_2Dscan_cWL580nm_GVD-7.6k01","MAPbBr3_2Dscan_cWL580nm_GVD-7.6k01",
              "MAPbBr3_2Dscan_cWL580nm_GVD-7.6k01","MAPbBr3_2Dscan_cWL580nm_GVD-7.6k01","MAPbBr3_2Dscan_cWL580nm_GVD-7.6k01"
              ]#,"24","24","24"]
    # str_data=["09","09","09","09","09","09","09","09","09","09","09","09","09","09","09","09","09","09","09","09","09","09","09","09","09","09","09","09","09","09","09"]                                                   
    # data_no="23"
    # for i in range(31):
    #     str_data.append(data_no)
    # list1=["64","63","62","61","60","59","58"]
    # str_data.extend(list1)
    
    
    
    delays=[-100.000000,0.000000,100.000000,200.000000,300.000000,400.000000,500.000000,600.000000,700.000000,800.000000,900.000000,1000.000000]#,110,120,130]
    # delays=[-10.000000,0.000000,10.000000,21.000000,33.100000,46.410000,61.051000,77.156100,94.871710,114.358881,135.794769,
    #         159.374246,185.311671,213.842838,245.227121,279.749834,317.724817,359.497299,405.447028,455.991731,511.590904,
    #         572.749995,640.024994,714.027494,795.430243,884.973268,983.470594,1091.817654,1210.999419,1342.099361,1486.309297,
    #         1644.940227,1819.434250,2011.377675,2222.515442,2454.766986,2710.243685,2991.268053,3300.394859,3640.434344,
    #         4014.477779,4425.925557,4878.518112,5376.369924,5924.006916,6526.407608,7189.048369,7917.953205,8719.748526,
    #         9601.723378,10000,12000,14000,17000,20000]



    str_delaydata=["01","02","03","04","05","06","07","08","09","10", "11","12"]     
    # str_delaydata=["01","02","03","04","05","06","07","08","09"]
    # for i in range(len(str_data)-9):
    #     str_delaydata.append(str(i+10))
    # list1=["01","01","01","01","01","01","111"]
    # str_delaydata.extend(list1)
    
    """for calib_method = 1
    
    for 150 g/mm grating: fitting a parabola Ax^2+Bx+C
        
        px:nm A ~ -1.32e-5
        px:nm B ~ 0.563
        px:nm C ~ (monochromator center wavelength - 284.5)   >>>>  calibrated based on existing data 
    
    """
    
    pixels=np.arange(0,1024,1)
    calib_probeaxis=(-1.32e-5*(pixels**2))+(0.563*pixels)+(mono_center_wl-284.5)


else:

    filepath = os.getcwd()
    window=Tk()                                                                     # prompt user to select the .info file for the dataset that needs to be extracted
    window.withdraw()
    window.fname=filedialog.askopenfilename(initialdir=filepath,title="open .info file", filetypes=[(".INFO files",'*.INFO')])            
    with open(window.fname) as f:
        metadata = f.readlines()
    
    found=False
    for lines in metadata:                                                          # reads the .info file and extracts all experiment related parameters automatically
        if 'Waiting Time Step Mode' in lines:                                       # code checks the type of delay list first, linear, list or exponential
            stepmode=lines.partition('Waiting Time Step Mode\t')[2]
            stepmode=(stepmode[:-1])
            found=True
            
    if found==False:
        print("Error! The .info file you selected is not in the right format. Please check the file")
        sys.exit()
            
    for lines in metadata:                                                          # reads the .info file and extracts all experiment related parameters automatically    

        if 'Step Size' in lines:
            timestep=lines.partition('Step Size (fs)\t')[2]                         # partitions the string after the given substring, 
            timestep=float(timestep[:-2])                                           # for more: https://docs.python.org/dev/library/stdtypes.html#str.partition
        if 'Final Delay' in lines:
            last_delay=lines.partition('Final Delay (fs)\t')[2]
            last_delay=float(last_delay[:-2])
        if 'Rotating Frame (Scanned)' in lines:
            rot_frame_scanned=lines.partition('Rotating Frame (Scanned)\t')[2]
            rot_frame_scanned=float(rot_frame_scanned[:-2])
        if 'Rotating Frame (Fixed)' in lines:
            rot_frame_fixed=lines.partition('Rotating Frame (Fixed)\t')[2]
            rot_frame_fixed=float(rot_frame_fixed[:-2])                         
        if 'MONO1 Wavelength' in lines:
            mono_center_wl=lines.partition('MONO1 Wavelength\t')[2]
            mono_center_wl=float(mono_center_wl[:-2])

        if stepmode=='Linear':
            if 'Waiting Time Linear First' in lines:
                wt0=lines.partition('Waiting Time Linear First\t')[2]
                wt0=float(wt0[:-2])
            if 'Waiting Time Linear Last' in lines:
                wtl=lines.partition('Waiting Time Linear Last\t')[2]
                wtl=float(wtl[:-2])
            if 'Waiting Time Linear Step' in lines:
                wtstep=lines.partition('Waiting Time Linear Step\t')[2]
                wtstep=float(wtstep[:-2])

        if stepmode=='List':
            if 'Waiting Time Delay List' in lines:
                wtlist=lines.partition('Waiting Time Delay List\t')[2]
                wtlist=(wtlist[:-2])
                wt0=float(wtlist.partition(',')[0])
                wt1=float(wtlist.partition(',')[2].partition(',')[0])
                wtstep=wt1-wt0
                wtl=wt0+(wtlist.count(',')*wtstep)
    
    timestep*=1e-15
    last_delay*=1e-15
    rotating_frame=np.abs(rot_frame_fixed-rot_frame_scanned)
    rotating_frame=1E7/(rotating_frame)    
    delays=np.arange((wt0),(wtl+wtstep),(wtstep))                                   # generates linear set of delays 
    
    # print(window.fname)
    
    filename=window.fname.partition('#')[2]
    filename=filename.partition('.info')[0]
    str_data=[filename]*np.size(delays)                                             # generates a list of string
    str_date=window.fname.partition('#')[0][-8:]
    fileDirectory=window.fname.partition('#')[0][:-9]
    str_delaydata=[]
    for i in range(1,np.size(delays)+1):
        new=f'{i:02d}'
        str_delaydata.append(str(new))                                              # generates a list of string of numbers according
                                                                                    # to waiting times in datafile: 01, 02, 03 ...

    """for calib_method = 1
    
    for 150 g/mm grating: fitting a parabola Ax^2+Bx+C
        
        px:nm A ~ -1.32e-5
        px:nm B ~ 0.563
        px:nm C ~ (monochromator center wavelength - 284.5)   >>>>  calibrated based on existing data 
    
    """
    
    pixels=np.arange(0,1024,1)
    calib_probeaxis=(-1.32e-5*(pixels**2))+(0.563*pixels)+(mono_center_wl-284.5)


######################################################################################################################################################################################"""

"""###################################################################################################################

4. ESSENTIAL FUNCTIONS - DONT CHANGE

    File Import, Data Processing (FT) and Axis Calibration
###################################################################################################################"""


def fileimport(fileDirectory,dataSet,numScans=0,start=0):
    # DETERMINING WHICH FILES TO IMPORT
    fileDirectory = os.path.abspath(fileDirectory)
#    print("Directory: ",fileDirectory)
    fileList = listdir(fileDirectory) # list of all files in the directory
    filePrefix = "#".join(dataSet) + '#'
    fileExtension = ".scan"
    filteredList = [i for i in fileList if filePrefix in i]
    filteredList = [i for i in filteredList if fileExtension in i]
    totalScans = len(filteredList)
    
#    print("\tFile prefix:",filePrefix)
#    print("\tFile extension:",fileExtension)
    print("\t", totalScans, "files found")
    # LOAD FILES, STORE IN MEMORY, AND DETERMINE THEIR SIZES
    if totalScans != 0:
        temp = np.loadtxt(os.path.join(fileDirectory,filteredList[0]))
        [nT1,nPixels] = temp.shape
        nT1 = nT1 - 1 # time points (ending row is not a time point)
        nPixels = nPixels - 1 # number of pixels (column 0 is time, not a pixel)
    else:
        print("Fatal Error. No data files found matching found matching prefix",
              filePrefix,"in directory",fileDirectory)
    
    # DETERMINING THE NUMBER OF FILES TO AVERAGE
    A = numScans
    if A == 0:
        A = totalScans - start
    if A > totalScans: # don't allow more scans than the total number of scans
        A = totalScans
    if A < 0: # negative numbers allow for averaging from end
        if start == 0:
            start = totalScans + A
        else:
            start = start + A
        if start < 0:
            start = 0
            A = -totalScans
        A = -A
    numScans = A
    scans2Avg = np.arange(numScans) # 0:numScans - 1
    
    # initialize data matrices
    FTdata = np.zeros([nT1,nPixels,numScans])
    
    # Data loading
#    print("### Loading files ###")
    for i in scans2Avg:
        dataTemp = np.loadtxt(os.path.join(fileDirectory,filteredList[start + i]))
        # print("\tLoading file", i + 1, "of", numScans)
        FTdata[...,i] = dataTemp[:-1,1:]
    
    where_are_infs = np.isinf(FTdata)
    FTdata[where_are_infs] = 0            
                    
    where_are_NaNs = np.isnan(FTdata)
    FTdata[where_are_NaNs] = 0  

    return FTdata,nPixels,nT1,filePrefix,numScans         


def dataprocess(FTdata,fftLength=0):
    
    data = FTdata

    # Background correction
    if bkgdCorrect:
#        print("#### Background Correction ####")
        data = rowWiseLoop(data,backgroundCorrect,polyFitRange = polyFitRange,
                              polyFitOrder = polyFitOrder)
    
    # Apodization
    if apodizeData:
#        print("#### Apodization ####")
#        print("\tCalculating windowing function")
        H = hammingWindow(nT1, data)
#        print("\tApodizing data")
        data = np.multiply(H,data)
    
    # Spectral averaging
#    print("### Spectral averaging ###")
    if len(data.shape) == 3:
        avgTF = np.mean(data[...,np.arange(numScans)], axis = 2)
        try:
            avgTFStd = np.std(data[...,np.arange(numScans)], axis = 2)
        except:
            print("\tException: unable to take standard deviation of spectral matrix")
    elif len(data.shape) == 2:
        print("\tData has only 2 dimensions. Not averaging data.")
    else:
        print("\tUnexpected shape of data:",data.shape)
        print("\tUnable to take average")
    
    # timearr=np.linspace(0,nT1,nT1)
    # pixarr=np.linspace(0,nPixels,nPixels)
    # plt.contour(pixarr,timearr,avgTF,levels=1)
    
    # Fourier transform
    # calculate the FFT length, if requested
    if fftLength == 0:
        fftLength = 2**(np.ceil(np.log2(nT1))+1)
    axisList = ['column','row','page']
    avgTF[0,...] *= 0.5 # see Hamm and Zanni section 9.5.3
#    print("### Fourier Transform ###")
#    print("\tFFT Length:",fftLength)
#    print("\tZero-padding by an additional factor of 2")
#    print("\tTaking FFT along axis", fftAxis,
#          "- each vector is a matrix", axisList[fftAxis])
    # See Hamm & Zanni section 9.5.4 regarding FFT Length
    FF = np.real(np.fft.rfft(avgTF, n = int(2*fftLength), axis = fftAxis))
    FF = FF[:-1,...]
    # disregarding the last element - see:
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.rfft.html
    return FF,fftLength,data,avgTF


def ax_calibration(calibrate,nPixels,fftLength,diagMethod,diagSlope,diagIntercept,calib_method,calibUnits,calibFreqs,calib_pumpaxis,calib_probeaxis,freqRangePump,freqRangeProbe):
    # FREQUENCY AXES #
    # calculate the probe and pump indices
    probeIdx = np.arange(nPixels)
    pumpIdx = np.arange(fftLength)
    probeFreqs = probeIdx # useful if not calibrating data
    pumpFreqs = pumpIdx # useful if not calibrating data
    
    if diagMethod == 0:        
        diagonal = diagSlope*(nPixels-probeFreqs) + diagIntercept
        diagIdx = np.arange(len(diagonal))
    
    
    if diagMethod == 1:            
        diagonal = probeFreqs
        diagIdx = np.arange(len(diagonal))
        
    cropUncalibratedPlot = False # displays a cropped version of the calibration plot
    
    if calibrate:
        print("### Calibrating pump and probe axes ###")
        calibFreqs = np.array(calibFreqs)                                           
        # nm2cm1 = 1E7 # conversion between nm and cm-1
        if calib_method == 0:
            calibParams = np.polyfit(calibPixels,calibFreqs,len(calibPixels)-1)         # CalibParams fits probe pixels to wavenumber
            probeFreqs = np.polyval(calibParams,probeFreqs)                             # convert probe pixels axis to wavenumber

        if calib_method == 1:
            probeFreqs=calib_probeaxis
    
        if diagMethod == 0:
            pumpFreqs = nPixels - (1/diagSlope)*(pumpFreqs-diagIntercept)           # convert to corresponding probe pixel number 
            pumpFreqs = np.polyval(calibParams,pumpFreqs)                           # convert pump axis to wavenumber 
    
        if diagMethod == 1:
            pumpFreqs=1e7/calib_pumpaxis
            freqRangePump=np.array(freqRangePump)
            freqRangeProbe=np.array(freqRangeProbe)


        diagonal = probeFreqs
        freqRangePump.sort()
        freqRangeProbe.sort()
        
        # Find the appropriate indices
        pumpIdx = ((pumpFreqs >= freqRangePump[0])*
                   (pumpFreqs<=freqRangePump[1]))
        probeIdx = ((probeFreqs >= freqRangeProbe[0])*
                    (probeFreqs<=freqRangeProbe[1]))
        diagIdx = ((diagonal >= freqRangePump[0])*
                   (diagonal <= freqRangePump[1]))
        
    elif not calibrate and cropUncalibratedPlot:
        pumpIdx = ((pumpFreqs >= indexRangePump[0])*
                   (pumpFreqs<=indexRangePump[1]))
        probeIdx = ((probeFreqs >= indexRangeProbe[0])*
                    (probeFreqs<= indexRangeProbe[1]))
        diagIdx = ((diagonal >= indexRangePump[0])*
                   (diagonal <= indexRangePump[1]))
    else:
        pumpIdx = np.arange(0,len(pumpFreqs))
        probeIdx = np.arange(0,len(probeFreqs))

    # if calibUnits=='nm':
        # probeFreqs=(1E7)/probeFreqs
        # pumpFreqs=(1E7)/pumpFreqs
        # diagonal=(1E7)/diagonal
        
    if calibUnits=='wn':
        probeFreqs=(1E7)/probeFreqs
        pumpFreqs=(1E7)/pumpFreqs
        diagonal=(1E7)/diagonal
        
    return pumpFreqs,probeFreqs,pumpIdx,probeIdx,diagIdx,diagonal
######################################################################################################################################################################################"""

"""###################################################################################################################
5. DIAGONAL CONSTRUCTION AND CALIBRATION

    Expand the comments below for further details
###################################################################################################################"""

"""############################################################################

COMMENT 3:

    diagMethod = int(0) : input diagSlope and diagIntercept value
    diagmethod = int(1) : fourier transform + rotating frame  

############################################################################"""

"""DIAGONAL SETTINGS"""

plotDiagonal = True                                         # plots a diagonal line (probe freq = pump freq)

diagMethod = int(1)                                         # manual (0) or auto (1) calibration

diagSlope = -2.842293605340221#((540-600)/(130-70))#.61     # only for diagMethod = int(0)
diagIntercept =1987.6633303002698#670#220                   # only for diagMethod = int(0)

"""Pump index range in usually shorter than (WL probe expt) or equal to(TOPAS probe expt) probe index range"""

nPixels=int(1024/binning)                                   # Calibration of pump axis after FT, only for diagMethod = int(1)
nT1=int(last_delay/timestep)
if fftLength == 0:
    fftLength = 2**(np.ceil(np.log2(nT1))+1)
    
calib_pumpaxis = np.fft.fftfreq(int(fftLength), d=timestep)     
calib_pumpaxis=np.sort(calib_pumpaxis)
calib_pumpaxis=calib_pumpaxis[(int((fftLength/2))):]
freq_p=np.zeros(int(fftLength))

flag=np.abs((calib_pumpaxis[0]-calib_pumpaxis[1])/2)

for i in range(int(fftLength)):
    if (np.mod(i,2)==0):
        freq_p[i]=calib_pumpaxis[int(i/2)]
    else:
        freq_p[i]=freq_p[i-1]+flag

calib_pumpaxis=freq_p                           
rotating_frame=(3e8/(rotating_frame*1e-9))
calib_pumpaxis+=rotating_frame
calib_pumpaxis=1e9*3e8/calib_pumpaxis
calib_pumpaxis=(1E7)/calib_pumpaxis
       
"""CALIBRATION SETTINGS"""
calibrate = True                                                # frequency calibration of axis
calib_method=int(1)                                             # manual (0) or auto (1) calibration
cropUncalibratedPlot = False                                    # displays a cropped version of the calibration plot
# cropUncalibratedPlot = True 
dead_px=False                                                   # DONT CHANGE                    

pumpFreqs,probeFreqs,pumpIdx,probeIdx,diagIdx,diagonal=ax_calibration(calibrate,nPixels,fftLength,diagMethod,diagSlope,diagIntercept,calib_method,calibUnits,calibFreqs,calib_pumpaxis,calib_probeaxis,freqRangePump,freqRangeProbe)

    
######################################################################################################################################################################################"""

"""###################################################################################################################

6. 2D DATA PROCESSING

###################################################################################################################"""


for ii in range(len(delays)):

    """############################################################################
    # LOADING FILE DETAILS FOR EXPORT
    ############################################################################"""
    
    str_2DdataSet=[str_data[ii],str_delaydata[ii]]           #in case the file names are different
    str_file=str1.join(str_2DdataSet)
    dataSet = [str_date, str_file]                           # prefix and experiment number from QuickControl
    numScans = 0                                             # number of scan files to average. if '0' all scans are averaged
    
    str_fig=[sample_name,str(delays[ii])]
    FigTitle=str2.join(str_fig)+'fs'
    figure_savepath = fileDirectory                          # File directory to save figures as image, if you have selected the option in Line 214
    figure_filename = FigTitle
    data_savepath = fileDirectory
    data_filename = FigTitle                                 # 'json' or 'pickle' extension will be appended to the filename

    # data_filename = '20200312_15' # 'json' or 'pickle' extension will be appended to the filename
    
    """############################################################################
    # DATA PROCESSING
    ############################################################################"""
    
    FTdata,nPixels,nT1,filePrefix,numScans=fileimport(fileDirectory,dataSet,numScans)
    FF,fftLength,data,avgTF=dataprocess(FTdata)
        
    """############################################################################
    # SPECTRUM PLOTTING - you may set specDisplay = False if you dont want to see 2D map now
    ############################################################################"""

    if (ii == 0):

        xStr = "probe"
        yStr = "pump"
        
    x = probeFreqs
    y = pumpFreqs
    z = FF
    
    sumX = np.sum(z, axis = 0)                          # Sum of data along the (probe) x-axis
    sumY = np.sum(z, axis = 1)                          # Sum of data along the (pump)  y-axis
    sumAbsX = np.sum(np.abs(z), axis = 0)
    sumAbsY = np.sum(np.abs(z), axis = 1)

    z=FF
    
    xInd = probeIdx
    yInd = pumpIdx
    
    if not calibrate and (ii == 0):
        xStr += " (pixels)"
        yStr += " (freq index)"
    elif (ii == 0):
        xStr += " freq / " + calibUnits
        yStr += " freq / " + calibUnits
    
    if swapAxes:
        x, y = y, x
        z = z.T
        xInd, yInd = yInd, xInd
        xStr, yStr = yStr, xStr
        diagonal, probeFreqs = probeFreqs, diagonal

    """Contour settings"""
    
    if not symmetricContours:
        if manualContourRange:
            lowZ = np.min(manualContourRange)
            highZ = np.max(manualContourRange)
        else:
            lowZ = np.min(FF)
            highZ = np.max(FF)
        mapString = plotColorMap
    else:
        if manualContourRange:
            highZ = np.min(np.abs(manualContourRange))
            if np.abs(manualContourRange[0]) != np.abs(manualContourRange[1]):
                print("\tWarning: symmetric z contours requested, but manual "
                      "bounds supplied not symmetric.")
                print("\tUsing the bound provided"
                      f" with the smallest absolute value ({highZ}) as max.")
        else:
            highZ = np.max(np.abs(FF))
        lowZ = -highZ
        mapString = symmetricContoursColorMap
    cints = np.linspace(lowZ-0.05*np.abs(lowZ),
                        highZ+0.05*np.abs(highZ),nContours)        

    Vm=max(abs(lowZ),abs(highZ))
    x = x[xInd]
    y = y[yInd]
    z = z[yInd,:][:,xInd]
        
    # titlestring = (filePrefix,FigTitle);
    # titlestring = ' '.join(titlestring)
    titlestring = FigTitle                   

    if (ii == 0):        
        if (generate_datacube):
            z_3D=np.zeros((len(y),len(x),len(delays)))      # To generate datacube 
        
    else:
        z_3D[:,:,ii]=z

    
    if specDisplay:
        print('### Plotting ###')
        fig = plt.figure(figsize=[7,7],dpi= 150)
        """Contour Plot"""

        if manualAxisAspect:
            axisAspect = manualAxisAspect                   # alternate: 'auto'
        elif calibrate and not manualAxisAspect:
            axisAspect = 'equal'
        elif not calibrate and not manualAxisAspect:
            axisAspect = 'auto'
    
        if not showProjections:
            contourPlotF = plt.contourf(x, y, z, cints, alpha = 0.75, 
                                        cmap = mapString)
            contourPlot = plt.contour(x, y, z, cints, colors = 'black', 
                                      linewidths=0.5)
            ax = fig.gca()
            if plotDiagonal:
                break
                diag = plt.plot(probeFreqs, diagonal, color = 'black', 
                                linewidth = 0.5, linestyle = 'dashed')
            ax.set_xlim(np.min(x), np.max(x))
            ax.set_ylim(np.min(y), np.max(y))
        
        else:
            fig.subplots_adjust(top = 0.9, left = 0.25)
            ax = fig.add_subplot(111)
        
            contourPlotF = plt.contourf(x, y, z, cints,levels=50, vmin=-Vm/1.5, vmax=Vm/1.5, alpha = 1, 
                                        cmap=mapString)
            # contourPlotF = plt.contourf(x, y, z, cints, levels=20, alpha = 1, 
            #                             cmap=mapString)
    
            contourPlot = plt.contour(x, y, z, cints, colors='black', 
                                      linewidths=0.5)
            if plotDiagonal:
                diag = plt.plot(probeFreqs, diagonal, color = 'black', 
                                linewidth = 0.5, linestyle = 'dashed')
            ax.set_xlim(np.min(x), np.max(x))
            ax.set_ylim(np.min(y), np.max(y))
        
                # break
            sumX = np.sum(z, axis = 0)
            sumY = np.sum(z, axis = 1)
            sumAbsX = np.sum(np.abs(z), axis = 0)
            sumAbsY = np.sum(np.abs(z), axis = 1)
            # for i in range(len(x)):
            #     z[:,i]=z[:,i]/sumAbsY
                
            # plt.plot(y,sumY)

            if swapAxes:
                sumX = sumAbsX
            else:
                sumY = sumAbsY
      
            """Projections"""
            
            div = make_axes_locatable(ax)
            axSumX = div.append_axes("top", 0.75, pad = 0., sharex = ax)
            axSumY = div.append_axes("right", 0.75, pad = 0., sharey = ax) 
            axSumX.plot(x, sumX, linewidth = 0.75, color = "black", alpha = 0.75)
            axSumY.plot(sumY, y, linewidth = 0.75, color = "black", alpha=0.75)
            axSumY.tick_params(bottom = True, top = False, left = False, right = False,
                                    labelbottom = False, labelleft = False)
            axSumX.tick_params(bottom = False, top = False, left = True, right = False,
                                    labelbottom = False, labelleft = False)
            axSumY.spines['right'].set_visible(False)
            axSumY.spines['top'].set_visible(False)
            axSumX.spines['right'].set_visible(False)
            axSumX.spines['top'].set_visible(False)
            axSumX.set_title(titlestring, fontsize = 12)
            
        if colorbar:
            if not showProjections:
                div = make_axes_locatable(ax)
            axColorbar = div.append_axes("right", 0.05, pad = 0.05)
            axColorbar.tick_params(bottom = False, top = False, left = False, right = True,
                                    labelbottom = False, labelright = True, 
                                    labelleft = False)
            cbar = fig.colorbar(contourPlotF, cax = axColorbar)
            cbar.ax.tick_params(labelsize = 8)
            cbarInts = np.round(cints, 1)
            cbarTicks = [cbarInts[1], 0, cbarInts[-2]]
            cbar.set_ticks(cbarTicks)
        ax.set_aspect(axisAspect)
        ax.set_xlabel(xStr)
        ax.set_ylabel(yStr)
        if not calibrate and swapAxes:
            ax.invert_yaxis()
        elif not calibrate and not swapAxes:
            ax.invert_xaxis()
        if flag_save_fig:
            fn = os.path.join(os.path.abspath(figure_savepath),figure_filename)
            plt.savefig(fn)
            print("### Saving Figure ###")
            print(f"\tFigure saved as {fn}")
    

    
    
    """Data Saving - For each T2 delay"""
    
    print("### Saving Data ###")
    scans2Avg = np.arange(numScans)
    
    if flag_save2D:
        fn = os.path.join(os.path.abspath(data_savepath), 
                               data_filename + "." + save_format_2D)
        print(f"\tSaving data to {fn}")
    if flag_save2D and save_format_2D.lower() == 'json':
        s2a = scans2Avg.tolist()
        # CP = calibParams.tolist()
        w1 = pumpFreqs.tolist()
        w3 = probeFreqs.tolist()
        TFSpectrum = avgTF.tolist()
        FFSpectrum = FF.tolist()
        d = {
             "experiment info": {"datestring": dataSet[0], 
                                 "exptNum": dataSet[1], "scans": s2a},
             # "proc opts": {"calibrated": calibrate, "calibParams": CP, 
             #               "bkgdCorrected": bkgdCorrect, "apodized": apodizeData},
             "proc opts": {"calibrated": calibrate, 
                           "bkgdCorrected": bkgdCorrect, "apodized": apodizeData},
    
             "data": {"probeFreqs": w3, "pumpFreqs": w1, 
                      "TFSpectrum": TFSpectrum, "FFSpectrum": FFSpectrum}
             }
        with open(fn, 'w') as hand:
            json.dump(d, hand, separators=(',', ':'), indent = 4)
    elif flag_save2D and save_format_2D.lower() == 'pickle':
        d = {
             "experiment info": {"datestring": dataSet[0], 
                                 "exptNum": dataSet[1], "scans": scans2Avg},
             "proc opts": {"calibrated": calibrate, "calibParams": calibParams, 
                           "bkgdCorrected": bkgdCorrect, "apodized": apodizeData},
             "data": {"probeFreqs": probeFreqs, "pumpFreqs": pumpFreqs, 
                      "TFSpectrum": avgTF, "FFSpectrum": FF}
         }
        with open(fn, 'wb') as hand:
            pickle.dump(d, hand, protocol=pickle.HIGHEST_PROTOCOL)
    elif flag_save2D and save_format_2D.lower() == 'csv':
        dataf=pd.DataFrame(data=z,columns=x,index=y)
        dataf.to_csv(fn) 
######################################################################################################################################################################################"""

"""###################################################################################################################

7. POST PROCESSING: SAVING DATACUBE

###################################################################################################################"""

if flag_save3D:
    fn = os.path.join(os.path.abspath(data_savepath),str_date+sample_name+ "_datacube." + save_format_3D)
    print(f"\tSaving data to {fn}")
    
if flag_save3D and save_format_3D.lower() == 'json':
    s2a = scans2Avg.tolist()
    CP = calibParams.tolist()
    w1 = pumpFreqs.tolist()
    t2 = delays.tolist()
    w3 = probeFreqs.tolist()
    FFSpectrum = z_3D.tolist()
    d = {
         "experiment info": {"datestring": dataSet[0], 
                             "exptNum": dataSet[1], "scans": s2a},
         "proc opts": {"calibrated": calibrate, "calibParams": CP, 
                        "bkgdCorrected": bkgdCorrect, "apodized": apodizeData},
         "proc opts": {"calibrated": calibrate, 
                       "bkgdCorrected": bkgdCorrect, "apodized": apodizeData},
         "data": {"probeFreqs": w3, "waiting times": t2, "pumpFreqs": w1, 
                  "3DFFSpectrum": FFSpectrum}
         }
    with open(fn, 'w') as hand:
        json.dump(d, hand, separators=(',', ':'), indent = 4)
        
elif flag_save3D and save_format_3D.lower() == 'pickle':
    d = {
         "experiment info": {"datestring": dataSet[0], 
                             "exptNum": dataSet[1], "scans": scans2Avg},
         "proc opts": {"calibrated": calibrate, 
                       "bkgdCorrected": bkgdCorrect, "apodized": apodizeData},
         "data": {"probeFreqs": x, "waiting times": delays, "pumpFreqs": y, 
                  "3DFFSpectrum": z_3D,"Diagonal": diagonal}
     }
    with open(fn, 'wb') as hand:
        pickle.dump(d, hand, protocol=pickle.HIGHEST_PROTOCOL)    
######################################################################################################################################################################################"""

"""###################################################################################################################

8. BETA: GENERATING GIF

###################################################################################################################"""        

"""
# def time_evolution(pumpscale,probescale,spectrum):
fig= plt.figure()
# diag=diagonal[diagIdx]
ax=plt.axes()
plt.xlabel('Probe Freq (wavenumber)')
plt.ylabel('Pump Freq (wavenumber)')
# diag = plt.plot(x, x, color = 'black', linewidth = 0.5, linestyle = 'dashed')
# line= plt.contourf(x,y,z_3D[:,:,0],cints,levels=200, vmin=-Vm, vmax=Vm,cmap='bwr')
# contourPlot = plt.contour(x, y, z_3D[:,:,0], cints, colors='black', linewidths=0.5)

# line= ax.contourf(probescale,pumpscale,spectrum[:,:,0])
# def init():
#     line= plt.contourf(probescale,pumpscale,spectrum[:,:,0])       
#     return line
def animate(i):   
    ax.clear()    
    diag = plt.plot(x, x, color = 'black', linewidth = 0.5, linestyle = 'dashed')
    line= plt.contourf(x,y,z_3D[:,:,i],cints,levels=20, vmin=-Vm, vmax=Vm,cmap='bwr')
    contourPlot = plt.contour(x,y,z_3D[:,:,i],cints,colors='black',linewidths=0.5)

    return line,contourPlot,diag

ani = animation.FuncAnimation(fig, animate,14, interval=5, blit=False)

#     #for gif
ani.save('2Dvideo' +'.gif', dpi=80, writer='pillow', fps = 1)
plt.show()

# time_evolution(y,x,z_3D)
"""       
######################################################################################################################################################################################"""

t_end = time.time()
total = t_end - t_start
print('Total elapsed time:', total, 's')