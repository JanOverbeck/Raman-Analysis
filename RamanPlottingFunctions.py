# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 14:35:31 2017

@author: Jan Overbeck

%reset does magic



"""
import os
import numpy
import matplotlib.pyplot as plt
import csv

def makeRamanHeaderFromFilename(filepath,outfolder):
    
    fname = filepath.split("\\")[-1]                #get input filename as last element behind a backslash.
    outname = fname.split(".txt")[0]+"_head.txt"
    outpath = outfolder + "\\" + outname

                           
    Wavelength = "Wavelength: " + fname[fname.find("WL")+2:fname.find("WL")+5]
    Power = "nm, Power: " + fname[fname.find("Power")+5:fname.find("Power")+10]
    Tint = ", T\-(int): " + fname[fname.find("Tint")+4:fname.find("Tint")+7]
    Pol1 = ", Pol\-(in): " + fname[fname.find("Pol")+3:fname.find("Pol")+9]
    Pol2 = ", Pol\-(out): " + fname[fname.find("Pol")+10:fname.find("Pol")+16]
    


    with open(outpath, 'wb') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(["Raman Shift \t" + "CCD Counts"])
        writer.writerow(["rel. cm\+(-1) \t" + "Arb. U"])
        writer.writerow(["\t" + Wavelength + Power + Tint + Pol1 + Pol2 ])
        

        with open(filepath, 'r') as incsv:
            reader = csv.reader(incsv)
            writer.writerows(row for row in reader)
            help(csv.writer)




def plotRamanSpectra(filepath, lambdaex=532.2, save="0", Xin="nm", Xout="relcm", plot=True):    
    """ plotRamanSpectra(filepath, lambdaex=532.2, save="0", Xin="nm", Xout="relcm", plot=True) 
        save -allows you to save all the files as "png" or "pdf", everything else is not saved and only shown. 
        Xin  -specifies which is the input format
        Xout -specifies whether the figure is displayed with relcm or nm x-axis.
        plot -can be set "False" to prevent plotting
        """    

    
    fname = filepath.split("\\")[-1] #Get the last part of the filepath as filename.
    
    #==============================================================================
    # Basic Datahandling
    #==============================================================================
        
    with open(filepath, 'r') as csvfile:  # because I've used with, I don't need to close it afterwards...
        dialect = csv.Sniffer().sniff(csvfile.read(1024))
        delim = dialect.delimiter
   
    try:
        a = numpy.loadtxt(filepath,skiprows=5,delimiter=delim) # No need to skip rows without header    
    except:
        print("Couldn't open file. Strange format?")

    
    if Xin=="nm":
        x_nm = a[:,0]
        y = a[:,1]
        x_relcm = 1E7*((1/lambdaex) - (1/x_nm)) # Convert input in nm to rel. cm-1
        
    #==============================================================================
    # Option, in case input file is in rel.cm-1
    #==============================================================================
    elif Xin=="relcm":
        x_relcm = a[:,0]
        y = a[:,1]
        x_nm = 1/(1/lambdaex-(x_relcm*1E-7)) # Convert input in rel. cm-1 to nm
    
    

    
    #==============================================================================
    # Plot in nm 
    #==============================================================================
    if Xout == "nm":
        fig = plt.figure(fname+"_nm") # a new figure window
        ax1 = fig.add_subplot(111) # ax1 is an Axes element ("plotting Window"). Specify (nrows, ncols, axnum)
        ax1.set_title("Plot nm")
        ax1.set_xlabel("nm")
        ax1.set_ylabel("CCD Counts")
        ax1.plot(x_nm, y)
    
    #==============================================================================
    # Optional editing of graph:
    #==============================================================================
        
    #ax1.set_xlim(x_nm[0],x_nm[-1])
    #ax1.legend(['Excitation at %d'%lambdaex, 'Annotation for next plot'])
    #fig1  # this is required to re-display the figure (in case you execute this line by line)
    
    #==============================================================================
    # Plot in rel. cm-1 
    #==============================================================================
    elif Xout == "relcm":
        fig = plt.figure(fname+"_relcm")
        ax2 = fig.add_subplot(111)
        ax2.set_title("Plot rel. cm-1")
        ax2.set_xlabel("rel. cm-1")
        ax2.set_ylabel("CCD Counts")
        ax2.plot(x_relcm, y)
    
    
    #==============================================================================
    # Saving
    #==============================================================================    
    
    pardir = os.path.abspath(os.path.join(filepath.split(fname)[0], os.pardir))
    savepath = pardir + "\\Plots\\" # define savedir
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        
    if save=="png":
        fig.savefig(savepath+fname+"_relcm"+".png") # create subdirectory and save file
        
    if save=="pdf":
        fig.savefig(savepath+fname+"_relcm"+".pdf") # create subdirectory and save file
        
    if save=="both":
        fig.savefig(savepath+fname+"_relcm"+".png") # create subdirectory and save file
        fig.savefig(savepath+fname+"_relcm"+".pdf") # create subdirectory and save file
    
    if not(plot): # option to prevent plotting    
        plt.close(fig)

#==============================================================================
# Functions for Fitting
#==============================================================================

def lorentzian(x,p):
    # p = parameters [hwhm, peak center, intensity] or [gamma,x0,prefactor], cp. https://en.wikipedia.org/wiki/Cauchy_distribution
    numerator =  (p[0]**2 )
    denominator = ( x - (p[1]) )**2 + p[0]**2
    y = p[2]*(numerator/denominator)
    return y

def residuals(p,y,x):
    err = y - lorentzian(x,p)
    return err
    
