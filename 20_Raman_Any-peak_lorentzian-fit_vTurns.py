# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 13:49:57 2016

@author: Jan Overbeck
"""
#########################################################################
####################### IMPORTING REQUIRED MODULES ######################
import os
import numpy
import pylab
from scipy.optimize import leastsq # Levenberg-Marquadt Algorithm #
import matplotlib.pyplot as plt
import sys


"""
This Script assumes, that you ran the "MakeHeader_Raman-Export-Tables_" routine (seperate script)
"""



#########################################################################
############################# LOADING DATA ##############################
path = os.getcwd()
# or
path = r'\\PHYS-MEKHONG.physik.unibas.ch\meso-share\Overbeck\B_Data_Analysis\A_Strained_Graphene\20161006_BJ_F_160817_1702_Raman_Pol\ExportTables\Copy-with-header'
#filenames = os.listdir(path)


#==============================================================================
# Specifications of measurement
#==============================================================================

lambdaex = 532.2 # excitation wavelength in nm
peakexpect = 3230 # expected 2D-peak position in rel. cm-1
approxheight = 200 # specify approximate height as a starting parameter
peakname = "2Ddash"
peakwl = 1/(1/lambdaex-(peakexpect*1E-7)) # expected 2D-peak position in nm
fixbgd = 310


#==============================================================================
# preview file No.1 in rel. cm-1
#==============================================================================
filename=os.listdir(path)[1]    
filepath=os.path.join(path,filename)
a = numpy.loadtxt(filepath,skiprows=3)
x = a[:,0]
y = a[:,1]

xrelcm = 1E7*((1/lambdaex) - (1/x))

plt.plot(xrelcm,y)

#zoom
plt.plot(xrelcm,y)
plt.axis([peakexpect-100,peakexpect+100,-500,max(y)])
plt(show)
    
"""
#==============================================================================
# Analysis in rel. cm-1
#==============================================================================
"""

outdata = []
#loopindex = 0
for filename in os.listdir(path):
    filepath=os.path.join(path,filename)
    a = numpy.loadtxt(filepath,skiprows=3) #There's 2 or three lines of  header, worst case we loose the first point of the spectrum.
    header=[]
    with open(filepath,'r') as f:  #read first 3 lines as header
        header.append(f.readline())
        header.append(f.readline())
        header.append(f.readline())
    polrot = ''                    #RoutineForPolarizationDetection
    if header[2].find("deg") >0:
        polrot = header[2].replace("deg","").replace("\n","").replace("\t","")
    
                
    x_nm = a[:,0]
    y = a[:,1]
    x = 1E7*((1/lambdaex) - (1/x_nm))
    xrange=abs(max(x))-abs(min(x))
    
    plt.plot(x,y)

##plotting 2D-detail in nm
#plt.plot(x,y)
#plt.axis([610,630,0,max(y)])
#plt.show()

##plotting 2D-detail in rel. cm-1
#plt.plot(xrelcm,y)
#plt.axis([2550,2750,0,max(y)])
#plt.show()

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
    
#########################################################################
######################## BACKGROUND SUBTRACTION with Polyfit#############

## defining the 'background' part of the spectrum between #
##ind_bg_low = (x > min(x)) & (x < 540.0)
#    ind_bg_low = (x > 50) & (x < peakexpect-200)
##ind_bg_high = (x > 590.0) & (x < max(x))
#    ind_bg_high = (x > peakexpect+200) & (x < max(x))
#
#    x_bg = numpy.concatenate((x[ind_bg_low],x[ind_bg_high]))
#    y_bg = numpy.concatenate((y[ind_bg_low],y[ind_bg_high]))
##plt.plot(x_bg,y_bg)
#
## fitting the background to a line # 
#    m, c = numpy.polyfit(x_bg, y_bg, 1)
#
## removing fitted background # 
#    background = m*x + c
##plt.plot(x,background)    
#    y_bg_corr = y - background
##plt.plot(x,y_bg_corr)
    
#########################################################################
######################## BACKGROUND SUBTRACTION with FIXED Value ########
    
    background = 0*x + fixbgd # create array of size x with all values=fixbgd
    y_bg_corr = y - background 
#plt.plot(x,y_bg_corr)    

#########################################################################
############################# FITTING DATA ## ###########################

# initial values #
    p = [5.0,peakexpect,approxheight]  # [hwhm, peak center, intensity] #

# optimization # 
    pbest = leastsq(residuals,p,args=(y_bg_corr,x),full_output=1)
    best_parameters = pbest[0]

# fit to data #
    fit = lorentzian(x,best_parameters)
    outdata.append("\n "+filename)
    outdata.append(polrot)
    for q in best_parameters:
        outdata.append(q)
#    loopindex +=1
    print("Fitted a peak at %d rel. cm-1." %best_parameters[1])
#########################################################################
############################# Exporting Data ############################
   
    
#fitdata = outdata[1::4]
#peakc = []
#filenumber = 0
#for index in fitdata:
#    peakc.append(fitdata[filenumber][1])
#    filenumber +=1
#plt.plot(outdata[1::4]) #plot peakwidth
#plt.plot(outdata[2::4]) #plot peakcentre
#plt.plot(outdata[3::4]) #plot peakintensity

#test = str(outdata)
#==============================================================================
# Cleanup outdata to writedata
#==============================================================================
outheader = "Filename \t Pol.Setting \t hwhm \t peak center \t intensity \n \t ° \t rel. cm-1 \t rel. cm-1 \t rel. cm-1 \n \t %s \t %s \t %s \t CCD counts" %(peakname,peakname,peakname)
writedata = outheader + str(outdata).replace("[","").replace("]","").replace("'","").replace("\\n","\n").replace(",","\t")

savepath = os.path.join(os.path.dirname(path),'best-fit-data_%s_relcm.txt' % peakname)   
file = open(savepath, 'w+')
file.write(writedata)
file.close()      
    
    

"""
#==============================================================================
# Analysis in nm-wavelength
#==============================================================================
"""

outdata = []
#loopindex = 0
for filename in os.listdir(path):
    filepath=os.path.join(path,filename)
    a = numpy.loadtxt(filepath,skiprows=3) #There's 2 or three lines of  header, worst case we loose the first point of the spectrum.
    header=[]
    with open(filepath,'r') as f:  #read first 3 lines as header
        header.append(f.readline())
        header.append(f.readline())
        header.append(f.readline())
    polrot = ' ,' #comma later gets replaced by tab
    if header[2].find("deg") >0:
        polrot = header[2].replace("deg","").replace("\n","").replace("\t","")
                           
    x = a[:,0]
    y = a[:,1]
    xrange=abs(max(x))-abs(min(x))
  
    xrelcm = 1E7*((1/lambdaex) - (1/x))

    plt.plot(x,y)

##plotting 2D-detail in nm
#plt.plot(x,y)
#plt.axis([610,630,0,max(y)])
#plt.show()

##plotting 2D-detail in rel. cm-1
#plt.plot(xrelcm,y)
#plt.axis([2550,2750,0,max(y)])
#plt.show()

#########################################################################
########################### DEFINING FUNCTIONS ##########################

#       Origin: y = y0 + (2*A/PI)*(w      / (4*(x-xc)^2 + w^2))
#       Here: y =           p[2] *(p[0]^2 / ((x-p[1])^2 + p[0]^2)

    def lorentzian(x,p):
        # p = parameters [hwhm, peak center, intensity] or [gamma,x0,prefactor], cp. https://en.wikipedia.org/wiki/Cauchy_distribution
#        numerator =  (p[0]**2 )  # why square??
        numerator =  (p[0])  # 
        denominator = ( x - (p[1]) )**2 + p[0]**2
        y = p[2]*(numerator/denominator)
        return y

    def residuals(p,y,x):
        err = y - lorentzian(x,p)
        return err
    
#########################################################################
######################## BACKGROUND SUBTRACTION with Polyfit ############

## defining the 'background' part of the spectrum between #
##ind_bg_low = (x > min(x)) & (x < 540.0)
#    ind_bg_low = (x > 535) & (x < peakwl-10)
##ind_bg_high = (x > 590.0) & (x < max(x))
#    ind_bg_high = (x > peakwl+10) & (x < max(x))
#
#    x_bg = numpy.concatenate((x[ind_bg_low],x[ind_bg_high]))
#    y_bg = numpy.concatenate((y[ind_bg_low],y[ind_bg_high]))
##plt.plot(x_bg,y_bg)
#
## fitting the background to a line # 
#    m, c = numpy.polyfit(x_bg, y_bg, 1)
#
## removing fitted background # 
#    background = m*x + c
#    y_bg_corr = y - background
##plt.plot(x,y_bg_corr)
    
#########################################################################
######################## BACKGROUND SUBTRACTION with FIXED Value ########
    
    background = 0*x + fixbgd # create array of size x with all values=fixbgd
    y_bg_corr = y - background 
#plt.plot(x,y_bg_corr)    

#########################################################################
############################# FITTING DATA ## ###########################

# initial values #
    p = [0.5,peakwl,approxheight]  # [hwhm, peak center, intensity] #
    init_parameters = [0.5,peakwl,approxheight]
# optimization # 
    pbest = leastsq(residuals,p,args=(y_bg_corr,x),full_output=1)
    best_parameters = pbest[0]

# fit to data #
    fit = lorentzian(x,best_parameters)
    outdata.append("\n"+filename)
    outdata.append(polrot)
    for q in best_parameters:
        outdata.append(q)
#    loopindex +=1
    print("Fitted a peak at %d nm." %best_parameters[1])
#########################################################################
############################# Exporting Data ############################
   
    
#fitdata = outdata[1::4]
#peakc = []
#filenumber = 0
#for index in fitdata:
#    peakc.append(fitdata[filenumber][1])
#    filenumber +=1
#plt.plot(outdata[1::4]) #plot peakwidth
#plt.plot(outdata[2::4]) #plot peakcentre
#plt.plot(outdata[3::4]) #plot peakintensity

#test = str(outdata)

#==============================================================================
# Cleanup outdata to writedata
#==============================================================================
outheader = "Filename \t Pol.Setting \t hwhm \t peak center \t intensity \n \t ° \t nm \t nm \t nm \n \t %s \t %s \t %s \t CCD counts" %(peakname,peakname,peakname)
writedata = outheader + str(outdata).replace("[","").replace("]","").replace("'","").replace("\\n","\n").replace(",","\t")

savepath = os.path.join(os.path.dirname(path),'best-fit-data_%s_nm.txt' % peakname)   
file = open(savepath, 'w+')
file.write(writedata)
file.close()    

"""
##########################################################################
############################### PLOTTING #################################
"""

plt.plot(x,y_bg_corr,'go-')
plt.plot(x,fit,'r-',lw=2)
plt.axis([min(x)-0.05*xrange, max(x)+0.05*xrange, -500, int(max(y)*1.1)])
plt.xlabel(r'$\omega$ (cm$^{-1}$)', fontsize=18)
plt.ylabel('Intensity (a.u.)', fontsize=18)
plt.show()
#==============================================================================
# Zoom in
#==============================================================================
plt.plot(x,y_bg_corr,'wo')
plt.plot(x,fit,'r-',lw=2)
plt.axis([int(best_parameters[1]-10*best_parameters[0]), int(best_parameters[1]+10*best_parameters[0]), -500, int(best_parameters[2]*1.2)])
plt.xlabel(r'$\omega$ (cm$^{-1}$)', fontsize=18)
plt.ylabel('Intensity (a.u.)', fontsize=18)
plt.show()


#==============================================================================
# Plot specific graph
#==============================================================================

index=65
spec_graph = outdata[index].replace("\n","")
print(spec_graph)
filepath=os.path.join(path,spec_graph)
a = numpy.loadtxt(filepath,skiprows=3) #There's 2 or three lines of  header, worst case we loose the first point of the spectrum.
x = a[:,0]
y = a[:,1]


plt.plot(x,y-background)
plt.plot(x,lorentzian(x,outdata[index+2:index+5]))
#plt.plot(x,lorentzian(x,init_parameters))
plt.show()