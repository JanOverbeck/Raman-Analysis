# -*- coding: utf-8 -*-
"""
Script for plotting Spectral data from input files with assumed structure: | wavelength[nm] | CCD counts |



Created on Wed Feb  1 12:28:51 2017

@author: Jan Overbeck
"""


"""Enter path of python script folder"""


import os
os.chdir(r"C:\Users\ovj\Desktop\ToDo")
import numpy as np
import matplotlib.pyplot as plt
import RamanPlottingFunctions as RamPlot #current WD correctly set? os.getcwd()
import RamanFittingFunctions as RamFit #current WD correctly set? os.getcwd()
import csv
import re

#==============================================================================
# Specifications of path, select file and fill in measurement conditions
#==============================================================================


"""Specify path here!"""

dir = r'C:\Users\ovj\Desktop\ToDo'
#needs to be a raw string!


"""Optional: look at all files foun.d in directory"""
files = os.listdir(dir)


"""Specify which file to analyse"""
fname=os.listdir(dir)[5]# Choose file from directory
filepath=os.path.join(dir, fname)                    
                        
#%%
#==============================================================================
# Load data
#==============================================================================

with open(filepath, 'r') as f:
    infodata = f.read()


results = re.split(r"All Results Table:", infodata)[-1]  # Just use the results table at the end
results = re.split(r"\n",results)   # split into lines


#%%
#==============================================================================
# reshape data
#==============================================================================
       
MeasNames = [] # List of Measurement Names
data = []  # I know this should be properly initialized, but finding the dimensions does not seem clear
for i,line in enumerate(results):
    if len(line) != 0:  #remove empty lines
        (name,y0,x0,w,A) = re.split(r"\t",results[i]) # split into names and data
        if A == "A":
            header = [name,y0,x0,w,A]
            continue
            
            print("hallo")
        MeasNames.append(name)
        data = np.append(data, [float(y0),float(x0),float(w),float(A)])

type(A) == str

data = data.reshape((len(MeasNames),4))

yoffset = data[:,0]
xcentre = data[:,1]
wlor = data[:,2]

fwhm = np.sqrt(2*np.log(2))*wlor
intens = data[:,3]
#%%
#==============================================================================
# define variable parameter along measurement axis
#==============================================================================

measparam = np.arange(4,4+len(data)*10,10) #central pixel



#%%
#==============================================================================
# Plot data
#==============================================================================

plt.close("all")
fig1 = plt.figure('Fig. 1')
ax1 = fig1.add_subplot(111)
#ax1.set_title("Convolution - full range")
ax1.set_xlabel("Pixel")
ax1.set_ylabel("Intensity (arb.u.)")
'''Plot full Range'''
#ax1.plot(theta, intens2(theta, *popt),'g-', label='cos^2 fit: theta0=%5.3f, sigma=%5.3f, A=%5.3f, I0=%5.3f' % param)
#'''Plot Meas-Data XY-Coordinates (Extended X-Range)'''
#ax1.plot(fxdata2,fydata2, 'bx',label='test-data')

ax1.plot(measparam,intens, 'b-',label='Intensity')
ax1.plot(measparam,xcentre, 'r-',label='xc')
ax1.plot(measparam,fwhm, 'k-',label='FWHM*')

#ax1.axvline(0)

#Add Legend
plt.legend()
plt.show()


