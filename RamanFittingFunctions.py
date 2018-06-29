# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:04:57 2017

@author: Jan Overbeck

%reset does magic


"""
 
from scipy.optimize import leastsq # Levenberg-Marquadt Algorithm #
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import scipy.constants


#########################################################################
########################### DEFINING FUNCTIONS ##########################

def lorentzian(x,p):
    # p = parameters [hwhm, peak center, intensity] or [gamma,x0,prefactor], cp. https://en.wikipedia.org/wiki/Cauchy_distribution
    numerator =  (p[0]**2 )
    denominator = ( x - (p[1]) )**2 + p[0]**2
    y = p[2]*(numerator/denominator)
    return y

def residuals(p,y,x):
    err = y - lorentzian(x,p)
    return err

def relcm2nm(lambdaex,relcm):
    nm = 1/(1/lambdaex-(relcm*1E-7)) # Convert input in rel. cm-1 to nm
    return nm

def nm2relcm(lambdaex,nm):
    relcm = 1E7*((1/lambdaex) - (1/nm)) # Convert input in nm to rel. cm-1
    return relcm

def nm2Ephoton(nm):
    Ephoton = 10**9*(scipy.constants.c*scipy.constants.h)/(scipy.constants.e*nm) # energy in eV
    return Ephoton

def fitLorentzPeak(filepath,lambdaex=532,peakexpect=2680, approxheight = 100, Xin="nm", plot="show", fixbgd= 160):
    """fitLorentzPeak(filepath,lambdaex=532,peakexpect=2680, approxheight = 100, Xin="nm", plot="show", fixbgd= 160)
    Input: path to file, excitation wavelength[nm], expected peak position[rel. cm-1], x-axis input units, plot: "show" or "save" or nothing, fix background counts for substraction.
    Output allways in rel.cm-1"""
    
    fname = filepath.split("\\")[-1] #Get the last part of the filepath as filename.
    header = [["Wavelength","Intensity", "Raman shift", "Intensity - Bgrd.", "Bgrd.", "Lorentz Fit"],["nm","CCD counts","rel. cm-1","CCD counts","CCD counts","CCD counts"]]
    outdata = [] # will get the above structure
    
    with open(filepath, 'r') as csvfile:  # because I've used with, I don't need to close it afterwards...
        dialect = csv.Sniffer().sniff(csvfile.read(1024))
        delim = dialect.delimiter
    
    a = np.loadtxt(filepath,skiprows=0, delimiter=delim ) # No need to skip rows without header   

    if Xin=="nm":
        x_nm = a[:,0]
        y = a[:,1]
        x_relcm = 1E7*((1/lambdaex) - (1/x_nm)) # Convert input in nm to rel. cm-1
        x=x_relcm
        xrange=abs(max(x_relcm))-abs(min(x_relcm))


        
    #==============================================================================
    # Option, in case input file is in rel.cm-1
    #==============================================================================
    elif Xin=="relcm":
        x_relcm = a[:,0]
        y = a[:,1]
        x_nm = 1/(1/lambdaex-(x_relcm*1E-7)) # Convert input in rel. cm-1 to nm
        x = x_relcm
        xrange=abs(max(x_relcm))-abs(min(x_relcm))
        
    
    outdata = np.c_[x_nm, y, x_relcm] # Arrange outputdata 

        
    """
    #==============================================================================
    # Analysis in rel. cm-1
    #==============================================================================
    """




    #########################################################################
    ######################## BACKGROUND SUBTRACTION with Polyfit#############
    
    # defining the 'background' part of the spectrum between ind_bg_low = (x > min(x)) & (x < 540.0)
    ind_bg_low = (x > 50) & (x < peakexpect-200)
    ind_bg_high = (x > peakexpect+200) & (x < max(x))

    x_bg = np.concatenate((x[ind_bg_low],x[ind_bg_high]))
    y_bg = np.concatenate((y[ind_bg_low],y[ind_bg_high]))
    #plt.plot(x_bg,y_bg)

    # fitting the background to a line # 
    m, c = np.polyfit(x_bg, y_bg, 1)

    # removing fitted background # 
    background = m*x + c
    #plt.plot(x,background)    
    y_bg_corr = y - background
    #plt.plot(x,y_bg_corr)
    
    outdata = np.c_[outdata, y_bg_corr, background]    
    
    #########################################################################
    ######################## BACKGROUND SUBTRACTION with FIXED Value ########
        
    #    background = 0*x + fixbgd # create array of size x with all values=fixbgd
    #    y_bg_corr = y - background 
    #    #plt.plot(x,y_bg_corr)    
    
    #########################################################################
    ############################# FITTING DATA ## ###########################

    # initial values #  
    p = [5.0,peakexpect,approxheight]  # [hwhm, peak center, intensity] #

    # optimization # 
    pbest = leastsq(residuals,p,args=(y_bg_corr,x),full_output=1)
    best_parameters = pbest[0]

    # fit to data 
    fit = lorentzian(x,best_parameters)
    print("Fitted a peak at %d rel. cm-1." %best_parameters[1])
    
    outdata = np.c_[outdata, fit]  
    

    
    #########################################################################
    ############################# Plot FIT and DATA ## ###########################
    
    if plot =="show":
        fig1 = plt.figure(fname+"_rel. cm-1") # a new figure window
        ax1 = fig1.add_subplot(111) # ax1 is an Axes element ("plotting Window"). Specify (nrows, ncols, axnum)
        ax1.set_title(fname+" [rel. cm-1]")
        ax1.set_xlabel("rel. cm-1")
        ax1.set_ylabel("CCD Counts")
        ax1.plot(x,fit)
        ax1.plot(x,y_bg_corr)
        ax1.legend(['Centre: %d rel. cm-1'%best_parameters[1], 'Bgrd. substr.'])
        fig1
        
    if plot =="save":
        fig1 = plt.figure(fname+"_rel. cm-1") # a new figure window
        ax1 = fig1.add_subplot(111) # ax1 is an Axes element ("plotting Window"). Specify (nrows, ncols, axnum)
        ax1.set_title(fname+" [rel. cm-1]")
        ax1.set_xlabel("rel. cm-1")
        ax1.set_ylabel("CCD Counts")
        ax1.plot(x,fit)
        ax1.plot(x,y_bg_corr)
        ax1.legend(['Centre: %d rel. cm-1'%best_parameters[1], 'Bgrd. substr.'])
        fig1
        
        savepath = filepath.split(fname)[0]+"Plots\\" # create subdirectory
        if not os.path.exists(savepath):
            os.makedirs(savepath)
            
        fig1.savefig(savepath+fname+"_relcm"+".png")
        
    
    
    #==============================================================================
    # Optional editing of graph:
    #==============================================================================
        
    #ax1.set_xlim(x_nm[0],x_nm[-1])
    #ax1.legend(['Excitation at %d'%lambdaex, 'Annotation for next plot'])
    #fig1  # this is required to re-display the figure (in case you execute this line by line)
    
#    outdata.append("\n "+filename)
#    outdata.append(polrot)
#    for q in best_parameters:
#        outdata.append(q)
#    loopindex +=1
    
    return (header,outdata,best_parameters)
    
