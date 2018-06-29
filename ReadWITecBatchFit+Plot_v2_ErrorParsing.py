# -*- coding: utf-8 -*-
"""
Script for plotting Spectral data from input files with assumed structure: | wavelength[nm] | CCD counts |



Created on Wed Feb  1 12:28:51 2017

@author: Jan Overbeck


%reset does magic



"""


"""Enter path of python script folder"""


import os
import numpy as np
import matplotlib.pyplot as plt
import RamanPlottingFunctions as RamPlot #current WD correctly set? os.getcwd()
import RamanFittingFunctions as RamFit #current WD correctly set? os.getcwd()
import csv
import re
import codecs
from scipy.optimize import curve_fit # for fitting at end

#==============================================================================
# Specifications of path, select file and fill in measurement conditions
#==============================================================================


"""Specify path here!"""

dir = r'C:\Users\ovj\Desktop\ToDo\180611_9AGNRonAu788_Vac_Cos4\in' ### Only works for one file in directory
#needs to be a raw string!


"""Optional: look at all files foun.d in directory"""
files = os.listdir(dir)

"""Search For Batch Fitting File"""

k = None
for i,f in enumerate(files):
    if re.search('Batch Fitting',f,1) != None and re.search('_out',f,1) == None:
        k = i
    
    
    
"""Specify which file to analyse"""
fname=os.listdir(dir)[k]# Choose file from directory
filepath=os.path.join(dir, fname)                    
                        
#%%
#==============================================================================
# Load data
#==============================================================================

with open(filepath, 'r') as f:
    infodata = f.read()


#%%

#==============================================================================
# Restructure Data in Fit-Datasets
#==============================================================================

Alldata = re.split(r"\n\n\n", infodata)  # Split into substrings at every double paragraph
DataIndFit = Alldata[1:-1] # Throw away first (fit description) & Last (all-results table)


#Debug                  
results = re.split(r"\n",DataIndFit[0])   # split into lines     



#%%
#==============================================================================
# reshape Individual Fit-Dataset
#==============================================================================
       
MeasNames = [] # List of Measurement Names
PolData = [] # Polarization extracted from measurement Name
OutData = []  # I know this should be properly initialized, but finding the dimensions does not seem clear
#%%
for m,IndFit in enumerate(DataIndFit):
    results = re.split(r"\n", IndFit)   # split into lines
    data = []
    for i,line in enumerate(results):
    
        #    if len(line) != 0:  #remove empty lines
    
    #        if A == "A":
    #            header = [name,y0,x0,w,A]
    #            continue
    #            
    #            print("hallo")
        if i == 0:        
            MeasNames.append(line)
            PolData.append(re.search("_Polin(...)",line).group(1))
        if i > 12 and re.search('\tyes\t',line) != None:
            (name,value,error,vary,unit) = re.split(r"\t",results[i]) # split into names and data
            if m == 0:
                data = np.append(data, (name,unit,value,name+"err",unit,error)) # create Structure
            else:
                data = np.append(data, (value,error)) # create Structure
    if m == 0:
        data = data.reshape(8,3)
    else:
        data = data.reshape(8,1)
    data = data.transpose()
    OutData=np.append(OutData,data)
    OutData=OutData.reshape(len(OutData)/8,8)
    header = OutData[0:2,:]



#%%  Save

outpath=os.path.join(dir, re.split('.txt',fname)[0]+'_out.txt') 
   

with codecs.open(outpath, 'w+', "utf-8-sig") as outfile: # "r+" mode means reading & updating, without deleting everything in it (trucating). "w+" will truncate. x mode means: create new file and open it for writing, raises error if file exists
        for i,line in enumerate(OutData):
            if i >1:
                outfile.write(PolData[i-2])
            else:
                outfile.write('Polin')                
            for k in line:
                outfile.write('\t' + k)

            outfile.write('\n')
         

#%%
#==============================================================================
# Plot data
#==============================================================================

plt.close("all")
fig1 = plt.figure('Fig. 1')
ax1 = fig1.add_subplot(111)
#ax1.set_title("Convolution - full range")
ax1.set_xlabel("Input Polarization Angle (°)")
ax1.set_ylabel("Intensity (arb.u.)")
ax1.plot(PolData,OutData[2:,6], 'b-',label='Intensity')
#ax1.plot(np.arange(len(OutData[2:,6])),OutData[2:,6], 'b-',label='Intensity')


#ax1.plot(measparam,xcentre, 'r-',label='xc')
#ax1.plot(measparam,fwhm, 'k-',label='FWHM*')

#ax1.axvline(0)

#Add Legend
plt.legend()
plt.show()




'''fitting'''


def cos2(theta, theta0=0, I0=0, A=1):
    return I0 + A * np.cos((theta/360)*(2*np.pi)-(theta0/360)*(2*np.pi))**2          
        

popt, pcov = curve_fit(cos2,np.asarray(PolData,dtype=np.float32),np.asarray(OutData[2:,6],dtype=np.float32),p0=(0,1000,2000))#, sigma = errdata) # fit with real data         



#redefine fit-parameters
#param = (popt[0], np.mod(popt[1],2*np.pi)*360/(2*np.pi), popt[2], popt[3])


fig3 = plt.figure('Fitted')
ax3 = fig3.add_subplot(111)
ax3.set_title("Quickfit")
ax3.set_xlabel("Theta (°)")
ax3.set_ylabel("Intensity (arb.u.)")
'''Plot full Range'''
ax3.plot(np.asarray(PolData,dtype=np.float32), cos2(np.asarray(PolData,dtype=np.float32), *popt),'g-', label='cos^2 fit: theta0=%5.3f' % popt[0])
'''Plot Meas-Data XY-Coordinates (Extended X-Range)'''
#ax1.plot(fxdata2,fydata2, 'bx',label='test-data')
ax3.plot(np.asarray(PolData,dtype=np.float32),np.asarray(OutData[2:,6],dtype=np.float32), 'bx',label='data')
#ax3.axvline(0)

#Add Legend
plt.legend()
plt.show()