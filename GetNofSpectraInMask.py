# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:08:05 2017

@author: ovj


%reset does magic


"""

import numpy as np
import re
import os
#import codecs
#import copy

Wdir = r"C:\Users\ovj\Desktop"
dir = Wdir #+ '\input'
listFiles = os.listdir(dir)

"""
Select file:
"""
fname = listFiles[0]



filepath=os.path.join(dir, fname) 
#==============================================================================
# Open file & extract header (line by line)
#==============================================================================
with open(filepath, 'r') as f:
    header = []
    #test = f.read()
    for line in f:
       header.append(line.split("\n")[0]) # remove trailing newline
       if line.find("[Data]") != -1:   # Specify keyword before actual data
           break
#==============================================================================
# Read data into numpy array
#==============================================================================
#data = np.loadtxt(filepath, skiprows=len(header))
data = np.genfromtxt(filepath, dtype='float', skip_header=len(header))

#==============================================================================
# Do something useful
#==============================================================================


nspectra = np.sum(data)
print(nspectra)
