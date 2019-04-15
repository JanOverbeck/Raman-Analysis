# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 08:36:18 2018

@author: ovj
"""



#import sympy as sp
import numpy as np
#from scipy import exp, sin, pi
import matplotlib.pyplot as plt
#from scipy.integrate import quad, odeint
from scipy.optimize import curve_fit
import io_functions_ovj as iovj # input-output-function library jan overbeck
import plt_functions_ovj as povj # plotting function library jan overbeck

import os
import re


#==============================================================================
# Try fitting cos2 to actual data
#==============================================================================


#load data from file:
#cd C:\Users\ovj\Desktop
#cd ToDo
#a=os.listdir()

#%%
'''
#==============================================================================
# Import Data & prep data-structure
#==============================================================================
'''

"""Specify path here!"""

dir = r'C:\Users\ovj\Documents\_Polybox_Local\Projects_OVJ\__ToDo__\190227_9-AGNR_S455_T-RPads2' ### Only works for one file in directory
#needs to be a raw string!



"""old"""
#"""Optional: look at all files found in directory"""
#files = os.listdir(dir)
#
#"""Search For Batch Fitting_output File"""
#
#k = None
#for i,f in enumerate(files):
#    if re.search('Batch Fitting',f,1) != None and re.search('_out',f,1) != None:
#        k = i
#       
#filepath=os.path.join(dir, files[k])  

filepath=iovj.file2analyse(dir, r'Batch Fitting.*_out')# in regular expressions, "." means any character and "*" means N-times the previous character.....

rawdata = np.loadtxt(filepath,skiprows=2)
RawFig, AxRaw = povj.makeParameterFig(rawdata[:,0], rawdata[:,7], "Rawdata plot", "G-peak amplitude", "Polin [°]", "Intensity [arb.u.]")

#%%
#Select data boundaries + optional shift


start = np.where(60==rawdata)[0][0]
stop =  np.where(240==rawdata)[0][0]
data = rawdata[start:stop+1,(0,7,8)] # cutting away anything beyond 360° + 1 step

    
#==============================================================================
"""Option: Shift curve around zero for further analysis --> all angles relative to maximum"""                
#==============================================================================
shift_to_maximum = True

maximum = 0  # This is also used as initial parameter for the fits

if shift_to_maximum:
    data[:,0] = np.arange(maximum-90, maximum+91, 10)               

ShiftedFig, AxShift = povj.makeParameterFig(data[:,0], data[:,1], "Shifted data plot", "G-peak amplitude", "Polin [°]", "Intensity [arb.u.]")


#%%

xdata=data[:-1, 0]*(2*np.pi/360)  # convert to rad
ydata=data[:-1,1] # peak intensity
ydata[0]=(data[0,1]+data[-1,1])*0.5  # average first and last point
errydata=data[:-1,2] # intensity error
errydata[0]=(data[0,2]+data[-1,2])*0.5  # average first and last point
          
# Case separation for 180 / 360 degree data

if abs(xdata[0]-xdata[-1]) > 180:
    n=3
    k=2
else:
    n=7
    k=1
    
#==============================================================================
#    extend the data to n-times it range, to minimize boundary effects in convolution       
#==============================================================================
     
nxdata=[]
for i in range(n):
    nxdata = np.append(nxdata, (i)*k*np.pi+xdata)
nxdata = nxdata - 3*np.pi
nxdata = np.append(nxdata,-nxdata[0])
          
nydata=[]
for i in range(n):
    nydata = np.append(nydata, ydata)
nydata = np.append(nydata,nydata[0])

errdata=[]
for i in range(n):
    errdata = np.append(errdata, errydata)
errdata = np.append(errdata,errdata[0])
#errdata=np.zeros(len(nydata))
errdata[0:6]=5000
errdata[-6:]=5000

'''define theta in terms of degree'''
nPiDeg=3
theta = np.arange(-180*nPiDeg,180*nPiDeg+1,1)*2*np.pi/360
                 
                 
#testplot
#plt.polar(xdata,ydata, 'bx',label='data')
#plt.errorbar(nxdata,nydata,errdata)#), 'bx',label='data')

NdataFig, AxN = povj.makeParameterFig(nxdata, nydata, "Ndata plot", "G-peak amplitude", "Polin [rad]", "Intensity [arb.u.]")

         
#%%

#==============================================================================
# Define functions           
#==============================================================================
def cos4(theta, theta0=0, I0=0, A=1):
    return I0 + A * np.cos(theta-theta0)**4.0    

def cos2(theta, theta0=0, I0=0, A=1):
    return I0 + A * np.cos(theta-theta0)**2          
        
def gauss(theta,theta0=0,sigma=1):
    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(theta-theta0)**2/(2*sigma**2))

def FWHM(sigma):
    return sigma*np.sqrt(np.log(2)*2)

#==============================================================================
# Actual Problem: I(theta) = I0 + A/(sigma*sqrt(2*pi)) * Int_-pi^pi exp(-(phi-theta0)^2/2*sigma^2) * cos^4(phi-theta) dphi
#==============================================================================
#%%
def intens2(theta, theta0, sigma, A, I0=0):
    return (max(theta)-min(theta))/len(theta) * np.convolve(gauss(theta, 0, sigma), cos2(theta, theta0, I0, A) , 'same') #normalized to steps of theta
    
def intens2I0(theta, theta0, sigma, A, I0=0):
    return (max(theta)-min(theta))/len(theta) * np.convolve(gauss(theta, 0, sigma), cos2(theta, theta0, 0, A) , 'same') #normalized to steps of theta
    
#define function for fitting to the data

def intens4(theta, theta0, sigma, A, I0=0):
    return (max(theta)-min(theta))/len(theta) * np.convolve(gauss(theta, 0, sigma), cos4(theta, theta0, I0, A) , 'same') #normalized to steps of theta

def intens4I0(theta, theta0, sigma, A, I0=0):
    return (max(theta)-min(theta))/len(theta) * np.convolve(gauss(theta, 0, sigma), cos4(theta, theta0, 0, A) , 'same') #normalized to steps of theta

#Example Parameters
I0 = 0  # offset
A = 1   #Amplitude
theta0= 2*np.pi/360*60
sigma= 2*np.pi/360*10

#plot smooth convoluted function
#plt.plot(theta, intens4(theta, 0, 2*np.pi/360*10, 1700, 0))
#plt.plot(theta, intens2(theta, 0, 2*np.pi/360*10, 2400, 200))
#plot convoluted function on nxdata
#plt.plot(nxdata, intens4(nxdata, 0, 2*np.pi/360*10, 1700))


'''Plot Meas-Data in Polar Coordinates (0-2Pi)''' 
                                       
                                       
#plt.polar(xdata, ydata, 'bx', label='data')                                       
                                       
#PolarP, AxPolar = povj.makePolFig(xdata, ydata, "Polar Plot", "data")



#%%
'''Plot Meas-Data XY-Coordinates (Extended X-Range)'''

#plt.plot(xdata,ydata, 'bx',label='data')


#%%
'''
#==============================================================================
# Fit convoluted function to meas data with cos4
#==============================================================================
'''

'''fitting'''
#popt, pcov = curve_fit(intens4, nxdata, nydata, p0=(maximum, 2*np.pi/360*10, 2000, 200), method='lm', sigma = errdata)

'''fitting with fixed I0'''
popt, pcov = curve_fit(intens4I0, nxdata, nydata, p0=(maximum, 2*np.pi/360*10, 2000, 0), method='lm', sigma = errdata)
         
#redefine fit-parameters
param = (popt[0]*360/(2*np.pi), np.mod(popt[1],2*np.pi)*360/(2*np.pi), abs(popt[2]), popt[3]) # theta0 | sigma | amplitude A | offset I0

#%%
fig1 = plt.figure('Convolution - full range')
ax1 = fig1.add_subplot(111)
ax1.set_title("Convolution - full range")
ax1.set_xlabel("Theta (°)")
ax1.set_ylabel("Intensity (arb.u.)")
'''Plot full Range'''
plt.plot(theta, intens4(theta, *popt),'g-', label='cos^4 fit: theta0=%5.3f, sigma=%5.3f, A=%5.3f, I0=%5.3f' % param)
'''Plot Meas-Data XY-Coordinates (Extended X-Range)'''
plt.plot(nxdata,nydata, 'bx', label='data')
#Add Legend
plt.legend()

'''Plot 0-2Pi'''

fig2 = plt.figure('Convolution - 0-2Pi')
ax2 = fig2.add_subplot(111)
ax2.set_title("Convolution - 0-2Pi")
ax2.set_xlabel("Theta (°)")
ax2.set_ylabel("Intensity (arb.u.)")
ax2.plot(theta, intens4(theta, *popt),'g-', label='cos^4 fit: theta0=%5.3f, sigma=%5.3f, A=%5.3f, I0=%5.3f' % param)
ax2.set_xlim([0,2*np.pi])

'''Plot Meas-Data XY-Coordinates (0-2Pi X-Range)'''
ax2.plot(xdata,ydata, 'bx', label='data')
#Add Legend
plt.legend()


#%%
'''
#==============================================================================
# Fit convoluted function to meas data with cos2
#==============================================================================
'''

'''fitting'''
#popt = curve_fit(intens2,fxdata2,fydata2,p0=(0,2*np.pi/360*10,1,0), sigma = errdata) # fit with fake test-data
#popt, pcov = curve_fit(intens2, nxdata, nydata, p0=(maximum, 2*np.pi/360*20, 2000, 10000), sigma = errdata) # fit with real data         


'''fitting with fixed I0'''
#popt = curve_fit(intens2,fxdata2,fydata2,p0=(0,2*np.pi/360*10,1,0), sigma = errdata) # fit with fake test-data
popt, pcov = curve_fit(intens2I0, nxdata, nydata, p0=(maximum, 2*np.pi/360*20, 2000, 0), sigma = errdata) # fit with real data      

                      
#redefine fit-parameters
param = (popt[0]*360/(2*np.pi), np.mod(popt[1],2*np.pi)*360/(2*np.pi), popt[2], popt[3])

#%%

fig1 = plt.figure('Convolution - full range')
ax1 = fig1.add_subplot(111)
ax1.set_title("Convolution - full range")
ax1.set_xlabel("Theta (°)")
ax1.set_ylabel("Intensity (arb.u.)")
'''Plot full Range'''
ax1.plot(theta, intens2I0(theta, *popt),'g-', label='cos^2 fit: theta0=%5.3f, sigma=%5.3f, A=%5.3f, I0=%5.3f' % param)
'''Plot Meas-Data XY-Coordinates (Extended X-Range)'''
#ax1.plot(fxdata2,fydata2, 'bx',label='test-data')
ax1.plot(nxdata,nydata, 'bx',label='data')
ax1.axvline(0)
leg = plt.legend()
leg.draggable(state = True)
##Add Legend

#
#'''Plot 0-2Pi'''
#
#fig2 = plt.figure('Convolution - 0-2Pi')
#ax2 = fig2.add_subplot(111)
#ax2.set_title("Convolution - 0-2Pi")
#ax2.set_xlabel("Theta (°)")
#ax2.set_ylabel("Intensity (arb.u.)")
#ax2.plot(theta, intens2(theta, *popt),'g-', label='cos^2 fit: theta0=%5.3f, sigma=%5.3f, A=%5.3f, I0=%5.3f' % param)
#ax2.set_xlim([0,2*np.pi])
#
#'''Plot Meas-Data XY-Coordinates (0-2Pi X-Range)'''
##ax2.plot(fxdata2,fydata2, 'bx',label='test-data')
#ax2.plot(nxdata,nydata, 'bx',label='data')
##Add Legend
#plt.legend()
#
#
#'''Plot 0-2Pi'''
#
#fig2 = plt.figure('Convolution - 0-2Pi')
#ax2 = fig2.add_subplot(111)
#ax2.set_title("Convolution - 0-2Pi")
#ax2.set_xlabel("Theta (°)")
#ax2.set_ylabel("Intensity (arb.u.)")
#ax2.plot(theta, intens2(theta, *popt),'g-', label='cos^2 fit: theta0=%5.3f, sigma=%5.3f, A=%5.3f, I0=%5.3f' % param)
#ax2.set_xlim([0,2*np.pi])
#
#'''Plot Meas-Data XY-Coordinates (0-2Pi X-Range)'''
##ax2.plot(fxdata2,fydata2, 'bx',label='test-data')
#ax2.plot(nxdata,nydata, 'bx',label='data')
##Add Legend
#plt.legend()
#
#plt.polar(theta[round(len(theta)/3):round(2*len(theta)/3)],intens2(theta, *popt)[round(len(theta)/3):round(2*len(theta)/3)],label='cos^2 fit: theta0=%5.3f, sigma=%5.3f, A=%5.3f, I0=%5.3f' % param)
#
#plt.polar(xdata,ydata, 'bx',label='data')

#%%


#plt.close("all")

#FWHM(param[1])

##%%
#
#'''
##==============================================================================
## create fake data for testing
##==============================================================================
#'''
#
#
#noise = np.random.normal(0,1,len(theta))
#
#fdata4 = (max(theta)-min(theta))/len(theta) * np.convolve(gauss(theta,0,2*np.pi/360*30),cos4(theta),'same')
#fdata4 = fdata4 + np.random.normal(0,0.01,len(fdata4))
#
#fdata2 = (max(theta)-min(theta))/len(theta) * np.convolve(gauss(theta,0,2*np.pi/360*30),cos2(theta),'same')
#fdata2 = fdata2 + np.random.normal(0,0.01,len(fdata2))
#
#
#fig3 = plt.figure('fake data')
#ax3 = plt.subplot(111)
#ax3.plot(theta,fdata2)
#
#def nearest(array,value):
#    idx = np.abs(array-value).argmin()
#    return (idx, array[idx])
#
#fydata4 = []
#fxdata4 = []
#for i,x in enumerate(nxdata):
#    (ind,val) = nearest(theta,x)
#    fxdata4.append(theta[ind])
#    fydata4.append(fdata4[ind])
#
#plt.plot(fxdata4,fydata4)
#
#
#fydata2 = []
#fxdata2 = []
#for i,x in enumerate(nxdata):
#    (ind,val) = nearest(theta,x)
#    fxdata2.append(theta[ind])
#    fydata2.append(fdata2[ind])
#
#plt.plot(fxdata2,fydata2)
#
#plt.axvline(0)
