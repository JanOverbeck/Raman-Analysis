# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 16:27:30 2018

@author: ovj

%reset does magic


"""

"""Analysis BRNC Raman - Map Export"""


import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import re
import codecs
from scipy.optimize import curve_fit # for fitting at end
import RamanFittingFunctions as RamFit #current WD correctly set? os.getcwd()

#==============================================================================
# Specifications of path, select file and fill in measurement conditions
#==============================================================================


"""Specify path here!"""

dir = r'C:\Users\ovj\Desktop\ToDo\280612_Raman_BRNC' ### Only works for one file in directory
#needs to be a raw string!


"""Optional: look at all files foun.d in directory"""
files = os.listdir(dir)




#"""Search For ***  File"""
#
#k = None
#for i,f in enumerate(files):
#    if re.search('***',f,1) != None and re.search('_out',f,1) == None:
#        k = i
#    
    
    
"""Specify which file to analyse"""
fname=os.listdir(dir)[19]# Choose file from directory
filepath=os.path.join(dir, fname)           

#%%

data = np.loadtxt(filepath, dtype='float', delimiter=',')

xval = data[:,0]
yval = data[:,1]
nm = data[0,2::2]
relcm = RamFit.nm2relcm(632.817,nm)
cts =data[:,3::2]


#%%
#plot first spectrum
plt.plot(relcm,cts[14,:])

#subset: 0,1,2,3,4,5,9,10,11,15,19,21,27,32,34

#%%
#Average

#subset = cts[(0,1,2,3,4,5,9,10,11,15,19,21,27,32,34),:]
#subAve = subset.mean(0)
#plt.plot(relcm,subAve)


aveSpec = cts.mean(0)
plt.plot(relcm,aveSpec)

#%%
#Export
outdata = np.column_stack((relcm,aveSpec))
#plt.plot(outdata[:,0],outdata[:,1])

np.savetxt(filepath.split('.txt')[0]+'_out.txt', outdata, delimiter='\t')


