# -*- coding: utf-8 -*-
"""
Script for plotting Spectral data from input files with assumed structure: | wavelength[nm] | CCD counts |



Created on Wed Feb  1 12:28:51 2017

@author: Jan Overbeck
"""
import os
import numpy
import matplotlib.pyplot as plt

#==============================================================================
# Specifications of path, select file and fill in measurement conditions
#==============================================================================

"""Specify path here!"""

path = r'\\PHYS-MEKHONG.physik.unibas.ch\meso-share\Overbeck\B_Data_Analysis\A_Strained_Graphene\20170131_Raman_BJ161005_1355_postZardoSetup\RawSpectra'
#needs to be a raw string!

"""Optional: look at all files found in directory"""
#os.listdir(path)

"""Specify which file to plot"""
filename=os.listdir(path)[0]# Choose file from directory

"""Specify excitation wavelength here!"""
lambdaex = 632.8 # excitation wavelength in nm: 532.2 vs. 632.8

#==============================================================================
# Basic Datahandling
#==============================================================================
    
filepath=os.path.join(path,filename) 
a = numpy.loadtxt(filepath,skiprows=0) # No need to skip rows without header
x_nm = a[:,0]
y = a[:,1]
x_relcm = 1E7*((1/lambdaex) - (1/x_nm)) # Convert input in nm to rel. cm-1

#==============================================================================
# Option, in case input file is in rel.cm-1
#==============================================================================

#x_relcm = a[:,0]
#y = a[:,1]
#x_nm = 1/(1/lambdaex-(x_relcm*1E-7)) # Convert input in rel. cm-1 to nm

#==============================================================================
# Plot in nm 
#==============================================================================

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_title("Plot nm")
ax1.set_xlabel("nm")
ax1.set_ylabel("CCD Counts")
ax1.plot(x_nm, y)

#==============================================================================
# Plot in rel. cm-1 
#==============================================================================

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.set_title("Plot rel. cm-1")
ax2.set_xlabel("rel. cm-1")
ax2.set_ylabel("CCD Counts")
ax2.plot(x_relcm, y)
