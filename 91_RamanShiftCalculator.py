# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 18:49:04 2016

@author: Jan
"""

def convertRelToNM(excitation, shift):
    shifted=1/((1/excitation)-(shift/1E7))
    print("Shifted WL in nm:")
    return shifted

    
def convertWLToRelcm(excitation, scatterWL):
    shift=1E7*((1/excitation)-(1/scatterWL))
    print("Raman shift in rel. 1/cm")
    return shift
    
def convertRelToDE(excitation, shift):
    shifted=1/((1/excitation)-(shift/1E7)) 
    delE=(1240/excitation)-(1240/shifted)
    print("Energy-Shift between WL:")
    return delE    


#%%

#==============================================================================
# Rel. cm-1  ---> nm
#==============================================================================
convertRelToNM(488,100)



#==============================================================================
# nm  ---> Rel. cm-1
#==============================================================================
convertWLToRelcm(532,640)



#==============================================================================
# # Rel. cm-1  ---> eV
#==============================================================================
convertRelToDE(488,100)
