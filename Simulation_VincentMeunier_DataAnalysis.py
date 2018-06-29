# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 23:44:31 2018

@author: ovj
"""


import os
import numpy as np
import matplotlib.pyplot as plt

files = os.listdir()


def conv(x):
    return x.replace(',', '.').encode()


dataRBLM = np.genfromtxt((conv(x) for x in open(files[8])), delimiter='\t')
dataLCM = np.genfromtxt((conv(x) for x in open(files[9])), delimiter='\t')


wavelengths = np.array([488,532,785]) # nm
energies = 1240/wavelengths

for e in energies:
    i = np.argmin(abs(dataRBLM[:,0]-e))
    j = np.argmin(abs(dataLCM[:,0]-e))
    print(round(e,3), 'eV', dataRBLM[i,1], "cts")
    print(round(e,3), 'eV', dataRBLM[i,1]/dataLCM[j,1], "cts")
    
plt.plot(data[:,0],data[:,1])    

plt.axvline(x= 1.58, color = 'darkred')
plt.axvline(x= 2.33, color = 'green')
plt.axvline(x= 2.54, color = 'skyblue')

plt.axvline(x= 1.16, color = 'black')#1064 Laser
plt.axvline(x= 1.96, color = 'red')#633 Laser

           

plt.plot(dataLCM[:,0],dataLCM[:,1])    

plt.axvline(x= 1.58, color = 'darkred')
plt.axvline(x= 2.33, color = 'g')
plt.axvline(x= 2.54, color = 'b')

plt.axvline(x= 1.16, color = 'k')#1064 Laser
plt.axvline(x= 1.96, color = 'r')#633 Laser
