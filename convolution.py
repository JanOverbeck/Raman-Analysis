# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 08:36:18 2018

@author: ovj
"""


#==============================================================================
# Analytical integration with python
#==============================================================================
import sympy as sp
import numpy as np
from scipy import exp, sin, pi
import matplotlib.pyplot as plt
from scipy.integrate import quad, odeint
from scipy.optimize import curve_fit
import os


''' Symbolic Integration Example'''

x = sp.Symbol('x')
help(sp.integrate)
sp.integrate(3.0*x**2 + 1, x)

#==============================================================================
# Example Analytically
#==============================================================================

sp.integrate(sp.exp(-x)*sp.sin(3.0*x),(x)) #,0,2.0*sp.pi))

#==============================================================================
# with borders
#==============================================================================
sp.integrate(sp.exp(-x)*sp.sin(3.0*x),(x,0,2*sp.pi))


''' Numeric Integration Example'''
#==============================================================================
# Do it numerically
#==============================================================================

def f(x):
    return 3.0*x**2 + 1

f(2)
f(1.8)

i,err = quad(f,0,2)

print(i)
print(err)

#==============================================================================
# Example Numerically
#==============================================================================


def g(x):
    return exp(-1.0*x)*sin(3.0*x)  
#==============================================================================
# Testing
#==============================================================================
g(1) 
#==============================================================================
# Numerical best practice incl. plotting
#==============================================================================

a = 0
b = 10

t = np.arange(a,b,0.01)
y = g(t)

plt.plot(t,y)

#==============================================================================
# Solve
#==============================================================================
 
i2,err2 = quad(g,0, 2*pi)

print(i2)
print(err2)

'''Fitting Example'''
#==============================================================================
# Fitting Example
#==============================================================================


#from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

#Define the data to be fit with some noise:


xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
np.random.seed(1729)
y_noise = 0.2 * np.random.normal(size=xdata.size)
ydata = y + y_noise
plt.plot(xdata, ydata, 'b-', label='data')

#Fit for the parameters a, b, c of the function func:


popt, pcov = curve_fit(func, xdata, ydata)

plt.plot(xdata, func(xdata, *popt), 'r-',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

#Constrain the optimization to the region of 0 <= a <= 3, 0 <= b <= 1 and 0 <= c <= 0.5:


popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
plt.plot(xdata, func(xdata, *popt), 'g--',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()



'''
#============================================================================================================================================================
# Relevant Stuff
#============================================================================================================================================================
'''

#==============================================================================
# Try fitting cos2 to actual data
#==============================================================================


#load data from file:
cd C:\Users\ovj\Desktop
ls
cd ToDo
a=os.listdir()

'''
#==============================================================================
# Import Data & Prep data-structure
#==============================================================================
'''

data = np.loadtxt(a[5],skiprows=1)
xdata=data[:-2,0]*(2*np.pi/360)
ydata=data[:-2,1]


#interpolated x-range for plotting the fitted function smoothly        
stepsPerUnit = 10  
xinterp=np.arange(0,2*np.pi,1/stepsPerUnit)
          
#==============================================================================
#    extend the data to n-times it range, to minimize boundary effects in convolution       
#==============================================================================
n=3
          
nxdata=[]
for i in range(n):
    nxdata = np.append(nxdata, (i)*2*np.pi+xdata)
nxdata = nxdata - 3*np.pi
nxdata = np.append(nxdata,-nxdata[0])
          
nydata=[]
for i in range(n):
    nydata = np.append(nydata, ydata)
nydata = np.append(nydata,nydata[0])

errdata = np.ones(len(nxdata))
errdata[0:6]=5000
errdata[-6:]=5000


#testplot
#plt.polar(xdata,ydata, 'bx',label='data')
#plt.errorbar(nxdata,nydata,errdata)#), 'bx',label='data')

          


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

#Example Parameters
I0 = 0  # offset
A = 1   #Amplitude
RangePi= 3*np.pi
theta = np.arange(-RangePi,+RangePi,1/stepsPerUnit)
theta0= 2*np.pi/360*60
theta0= 2*np.pi/360*0
sigma= 2*np.pi/360*10


         
#==============================================================================
# Test Integration, fitting & plotting
#==============================================================================

#==============================================================================
# test functions for convolution
#==============================================================================
x = np.arange(-3*np.pi,3*np.pi,6*np.pi/3000)
'''step'''
step = np.zeros(1000)
step[500:] = 1
plt.plot(x,step)
'''gradien'''
grad = np.zeros(1000)    
grad[400:600] = np.arange(0,1,0.005)
grad[600:]=0    
plt.plot(x,grad)
'''box'''
box = np.zeros(1000)
box[400:410] = 1
plt.plot(x,box) 

'''convolute'''
conv = np.convolve(test,d(I0,A,x,0),'same')
'''plot'''
plt.plot(np.arange(0,len(box),1),box)
plt.plot(x,test)
plt.plot(x,d(I0,A,x,0))
plt.plot(x,conv)

#==============================================================================
# Test integration
#==============================================================================

# integral over gaussian
i,err = quad(gauss,-2*np.pi,2*np.pi)
print(i)


# integral over cos4
j,err = quad(cos4,-0.5*np.pi,0.5*np.pi)
print(j)

#fit
popt, pcov = curve_fit(cos2,xdata,ydata )

# redefine parameters
param = (np.mod(popt[0],2*np.pi)*360/(2*np.pi), popt[1], popt[2])

#plot
plt.plot(xinterp, cos2(xinterp, *popt),'g-', label='cos^2 fit: theta0=%5.3f, I0=%5.3f, A=%5.3f' % param)
plt.legend()

#==============================================================================
# Plot for testing
#==============================================================================
plt.axvline(-np.pi) # make vertical line to indicate stuff

           
plt.polar(theta, cos4(theta,0,I0,A))
plt.plot(theta, cos4(theta,0,I0,A))
plt.plot(theta, gauss(theta,0,2*np.pi/360*10))





#==============================================================================
# convolution gauss*step 
#==============================================================================
#define step
step = np.zeros(len(theta))
step[round(len(step)/2):]=1
     
#convolute     
conv = np.convolve(gauss(theta),step, 'same')

#plot
plt.plot(theta, step)
plt.plot(theta, gauss(theta))
plt.plot(theta, conv)

#==============================================================================
# convolution cos4 * gauss
#==============================================================================
#define width of gaussian
sigmaDeg = [1,10,30]

for i in sigmaDeg:
    sigmaNew = 2*np.pi/360*sigmaDeg
    #convolute
    conv = 1/stepsPerUnit*np.convolve(gauss(theta,0,sigmaNew),cos4(theta),'same')
    #plot
    plt.plot(theta, conv)
    
    
plt.plot(theta, cos4(theta))
plt.plot(theta, gauss(theta,0,sigmaNew))
plt.plot(theta, conv)



'''
#==============================================================================
# create fake data for testing
#==============================================================================
'''


noise = np.random.normal(0,1,len(theta))

fdata4 = (max(theta)-min(theta))/len(theta) * np.convolve(gauss(theta,0,2*np.pi/360*30),cos4(theta),'same')
fdata4 = fdata4 + np.random.normal(0,0.01,len(fdata4))

fdata2 = (max(theta)-min(theta))/len(theta) * np.convolve(gauss(theta,0,2*np.pi/360*30),cos2(theta),'same')
fdata2 = fdata2 + np.random.normal(0,0.01,len(fdata2))


fig3 = plt.figure('fake data')
ax3 = plt.subplot(111)
ax3.plot(theta,fdata2)

def nearest(array,value):
    idx = np.abs(array-value).argmin()
    return (idx, array[idx])

fydata4 = []
fxdata4 = []
for i,x in enumerate(nxdata):
    (ind,val) = nearest(theta,x)
    fxdata4.append(theta[ind])
    fydata4.append(fdata4[ind])

plt.plot(fxdata4,fydata4)


fydata2 = []
fxdata2 = []
for i,x in enumerate(nxdata):
    (ind,val) = nearest(theta,x)
    fxdata2.append(theta[ind])
    fydata2.append(fdata2[ind])

plt.plot(fxdata2,fydata2)

plt.axvline(0)









'''
#==============================================================================
# Actual Problem: I(theta) = I0 + A/(sigma*sqrt(2*pi)) * Int_-pi^pi exp(-(phi-theta0)^2/2*sigma^2) * cos^4(phi-theta) dphi
#==============================================================================
'''

'''redefint theta in terms of degree'''
nPiDeg=3
theta = np.arange(-180*nPiDeg,180*nPiDeg+1,1)*2*np.pi/360

#define function for fitting to the data

def intens4(theta, theta0, sigma, A, I0=0):
    return (max(theta)-min(theta))/len(theta) * np.convolve(gauss(theta,0,sigma),cos4(theta,theta0,I0,A) , 'same') #normalized to steps of theta

def intens2(theta, theta0, sigma, A, I0=0):
    return (max(theta)-min(theta))/len(theta) * np.convolve(gauss(theta,0,sigma),cos2(theta,theta0,I0,A) , 'same') #normalized to steps of theta



#plot smooth convoluted function
plt.plot(theta, intens4(theta, 0, 2*np.pi/360*10, 1700, 0))
plt.plot(theta, intens2(theta, 0, 2*np.pi/360*10, 2400, 200))
#plot convoluted function on nxdata
plt.plot(nxdata, intens4(nxdata, 0, 2*np.pi/360*10, 1700))


'''Plot Meas-Data in Polar Coordinates (0-2Pi)''' 

plt.polar(xdata,ydata, 'bx',label='data')


'''Plot Meas-Data XY-Coordinates (Extended X-Range)'''

#plt.plot(xdata,ydata, 'bx',label='data')
plt.plot(nxdata,nydata, 'x',label='data')


'''
#==============================================================================
# Fit convoluted function to meas data with cos4
#==============================================================================
'''

'''fitting'''
popt, pcov = curve_fit(intens4,fxdata,fydata,p0=(0,2*np.pi/360*10,2000,200), sigma = errdata)
         
#redefine fit-parameters
param = (popt[0], np.mod(popt[1],2*np.pi)*360/(2*np.pi), popt[2])


fig1 = plt.figure('Convolution - full range')
ax1 = fig1.add_subplot(111)
ax1.set_title("Convolution - full range")
ax1.set_xlabel("Theta (°)")
ax1.set_ylabel("Intensity (arb.u.)")
'''Plot full Range'''
plt.plot(theta, intens4(theta, *popt),'g-', label='cos^4 fit: theta0=%5.3f, sigma=%5.3f, A=%5.3f' % param)
'''Plot Meas-Data XY-Coordinates (Extended X-Range)'''
plt.plot(nxdata,nydata, 'bx',label='data')
#Add Legend
plt.legend()

'''Plot 0-2Pi'''

fig2 = plt.figure('Convolution - 0-2Pi')
ax2 = fig2.add_subplot(111)
ax2.set_title("Convolution - 0-2Pi")
ax2.set_xlabel("Theta (°)")
ax2.set_ylabel("Intensity (arb.u.)")
ax2.plot(theta, intens4(theta, *popt),'g-', label='cos^4 fit: theta0=%5.3f, sigma=%5.3f, A=%5.3f' % param)
ax2.set_xlim([0,2*np.pi])

'''Plot Meas-Data XY-Coordinates (0-2Pi X-Range)'''
ax2.plot(xdata,ydata, 'bx',label='data')
#Add Legend
plt.legend()

'''
#==============================================================================
# Fit convoluted function to meas data with cos2
#==============================================================================
'''

'''fitting'''
#popt = curve_fit(intens2,fxdata2,fydata2,p0=(0,2*np.pi/360*10,1,0), sigma = errdata) # fit with fake test-data
popt, pcov = curve_fit(intens2,nxdata,nydata,p0=(0,2*np.pi/360*10,2000,200), sigma = errdata) # fit with real data         



#redefine fit-parameters
param = (popt[0], np.mod(popt[1],2*np.pi)*360/(2*np.pi), popt[2], popt[3])


fig1 = plt.figure('Convolution - full range')
ax1 = fig1.add_subplot(111)
ax1.set_title("Convolution - full range")
ax1.set_xlabel("Theta (°)")
ax1.set_ylabel("Intensity (arb.u.)")
'''Plot full Range'''
ax1.plot(theta, intens2(theta, *popt),'g-', label='cos^2 fit: theta0=%5.3f, sigma=%5.3f, A=%5.3f, I0=%5.3f' % param)
'''Plot Meas-Data XY-Coordinates (Extended X-Range)'''
#ax1.plot(fxdata2,fydata2, 'bx',label='test-data')
ax1.plot(nxdata,nydata, 'bx',label='data')
ax1.axvline(0)

#Add Legend
plt.legend()

'''Plot 0-2Pi'''

fig2 = plt.figure('Convolution - 0-2Pi')
ax2 = fig2.add_subplot(111)
ax2.set_title("Convolution - 0-2Pi")
ax2.set_xlabel("Theta (°)")
ax2.set_ylabel("Intensity (arb.u.)")
ax2.plot(theta, intens2(theta, *popt),'g-', label='cos^2 fit: theta0=%5.3f, sigma=%5.3f, A=%5.3f, I0=%5.3f' % param)
ax2.set_xlim([0,2*np.pi])

'''Plot Meas-Data XY-Coordinates (0-2Pi X-Range)'''
#ax2.plot(fxdata2,fydata2, 'bx',label='test-data')
ax2.plot(nxdata,nydata, 'bx',label='data')
#Add Legend
plt.legend()


'''Plot 0-2Pi'''

fig2 = plt.figure('Convolution - 0-2Pi')
ax2 = fig2.add_subplot(111)
ax2.set_title("Convolution - 0-2Pi")
ax2.set_xlabel("Theta (°)")
ax2.set_ylabel("Intensity (arb.u.)")
ax2.plot(theta, intens2(theta, *popt),'g-', label='cos^2 fit: theta0=%5.3f, sigma=%5.3f, A=%5.3f, I0=%5.3f' % param)
ax2.set_xlim([0,2*np.pi])

'''Plot Meas-Data XY-Coordinates (0-2Pi X-Range)'''
#ax2.plot(fxdata2,fydata2, 'bx',label='test-data')
ax2.plot(nxdata,nydata, 'bx',label='data')
#Add Legend
plt.legend()

plt.polar(theta[round(len(theta)/3):round(2*len(theta)/3)],intens2(theta, *popt)[round(len(theta)/3):round(2*len(theta)/3)])

plt.polar(xdata,ydata, 'bx',label='data')

plt.close("all")

