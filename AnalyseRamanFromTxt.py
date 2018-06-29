# -*- coding: utf-8 -*-
"""
Script for plotting Spectral data from input files with assumed structure: | wavelength[nm] | CCD counts |



Created on Wed Feb  1 12:28:51 2017

@author: Jan Overbeck
"""


"""Enter path of python script folder"""




import os
os.chdir(r"C:\Users\ovj\Documents\PythonScripts\Data Analysis")
import numpy as np
import matplotlib.pyplot as plt
import RamanPlottingFunctions as RamPlot #current WD correctly set? os.getcwd()
import RamanFittingFunctions as RamFit #current WD correctly set? os.getcwd()

#==============================================================================
# Specifications of path, select file and fill in measurement conditions
#==============================================================================


"""Specify path here!"""

path = r'C:\Users\ovj\Desktop\test\input'
#needs to be a raw string!

"""Specify excitation wavelength here!"""
lambdaex = 532.2 # excitation wavelength in nm: 532.2 vs. 632.8

#==============================================================================
# Add Headers to Files
#==============================================================================
outfolder = path.split("input")[0] + "head"
os.mkdir(outfolder)
for file in os.listdir(path):
     filepath=os.path.join(path,file) 
     RamPlot.makeRamanHeaderFromFilename(filepath,outfolder)
plotpath = outfolder

#==============================================================================
# Plot specific file
#==============================================================================
"""Optional: look at all files found in directory"""
os.listdir(path)


"""Specify which file to plot"""
filename=os.listdir(plotpath)[2]# Choose file from directory
filepath=os.path.join(plotpath,filename)
RamPlot.plotRamanSpectra(filepath,lambdaex,"show","relcm","relcm")#plotRamanSpectra(filepath, lambdaex=532.2, show save="0", Xin="nm", Xout="relcm", plot=True)

#==============================================================================
# Plot All Files:
#==============================================================================

for file in os.listdir(plotpath):
#    if filename.find("head") == -1:
        
     filepath=os.path.join(plotpath,file) 
     RamPlot.plotRamanSpectra(filepath,lambdaex,"both","relcm","relcm",False)
     
#==============================================================================
# Analysis:
#==============================================================================

"""Fit and plot specific file"""
Fitresults = ()
filename=os.listdir(path)[2]# Choose file from directory
filepath=os.path.join(path,filename)
Fitresults = RamFit.fitLorentzPeak(filepath,lambdaex,2650,100,"nm","show")







"""Fit and save all files in directory"""

for file in os.listdir(path):
        filepath=os.path.join(path,file)
        try:
            Fitresults = RamFit.fitLorentzPeak(filepath,lambdaex,2650,100,"relcm","save")
        except :
            pass
        savepath = os.path.join(path,file.split(".txt")[0]+"_fitted.txt")
      
        with open(savepath, 'w+') as f:
            for t in Fitresults[0]:
                f.write(' \t'.join(str(s) for s in t) + '\n')
            f.write("\t \t \t \t \t "+ str(Fitresults[2]) + '\n')
            for t in Fitresults[1]:
                f.write(' \t'.join(str(s) for s in t) + '\n')
                
                
#Old version:
#==============================================================================
#         f = open(savepath, 'w+')
#         for t in Fitresults[0]:
#             f.write(' \t'.join(str(s) for s in t) + '\n')
#         f.write("\t \t \t \t \t "+ str(Fitresults[2]) + '\n')
#         for t in Fitresults[1]:
#             f.write(' \t'.join(str(s) for s in t) + '\n')
#         f.close()
#==============================================================================
        
    
