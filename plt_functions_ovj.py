# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:50:30 2018

@author: ovj
"""



import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const


#%%
#==============================================================================
# Plotting functions
#==============================================================================

#maybe, it makes more sense to implement this via a function, rather than a new class?

def makeRamFig(xdata, ydata, figname=None, label=None):
    """use as fig, ax = makeRamFig(xdata, ydata, figname, data-label)"""
    fig = plt.figure(figname)
    global ax1
    ax1 = fig.add_subplot(111)
    ax1.plot(xdata, ydata, label=label)
    ax1.set_ylabel("Raman intensity [arb.u.]", fontsize=12)
    ax1.set_xlabel("Raman shift [cm-1]", fontsize=12)
    if label:
        leg = ax1.legend()
        leg.draggable(state = True)
    
    ax1.tick_params(axis='both', which='major', labelsize=10)
    fig.tight_layout()
    return fig, ax1


def makePolFig(theta, Idata, figname=None, label=None):
    fig = plt.figure(figname)
    global ax1
    ax1 = fig.add_subplot(111, projection='polar')
    ax1.plot(theta, Idata, label=label)
#    ax1.set_ylabel("Raman intensity [arb.u.]", fontsize=12)
    ax1.set_xlabel("Pol [°]", fontsize=12)
    if label:
        leg = ax1.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fancybox=False, shadow=False, ncol=1)
        leg.draggable(state = True)
        
    ax1.tick_params(axis='both', which='major', labelsize=10)
    fig.tight_layout()
    return fig, ax1


def makeParameterFig(xdata, ydata, figname=None, label=None, xlabel="X", ylabel="Y"):
    """use as fig, ax = makeParameterFig(xdata, ydata, figname, data-label, xlabel, ylabel)"""
    fig = plt.figure(figname)
    global ax1
    ax1 = fig.add_subplot(111)
    ax1.plot(xdata, ydata, label=label)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.set_xlabel(xlabel, fontsize=12)
    if label:
        leg = ax1.legend()
        leg.draggable(state = True)
    
    ax1.tick_params(axis='both', which='major', labelsize=10)
    fig.tight_layout()
    return fig, ax1


#==============================================================================
# example 2 via function
#==============================================================================
#plt.rcParams['savefig.facecolor'] = "0.8"
#
#def example_plot(ax, fontsize=12):
#     ax.plot([1, 2])
#     ax.locator_params(nbins=3)
#     ax.set_xlabel('x-label', fontsize=fontsize)
#     ax.set_ylabel('y-label', fontsize=fontsize)
#     ax.set_title('Title', fontsize=fontsize)
#
#plt.close('all')
#fig, ax = plt.subplots()
#example_plot(ax, fontsize=24)

# We change the fontsize of minor ticks label 
#plot.tick_params(axis='both', which='major', labelsize=10)
#plot.tick_params(axis='both', which='minor', labelsize=8)

##https://matplotlib.org/users/tight_layout_guide.html





#class RamFig(object):
#    def __init__(self, xdata, ydata, name=None, label=None):
#        self.name = name
#        self.label = label
#        rfig = plt.figure(name)
#        global ax1
#        ax1 = rfig.add_subplot(111)
#        ax1.plot(xdata, ydata, label=label)
#        ax1.set_ylabel("Raman intensity")
#        ax1.set_xlabel("Raman shift [cm-1]")
#        if label:
#            ax1.legend()
#               
#    def __str__(self):
#        return 'RamFig - Name: %s, Label: %s' %(self.name, self.label)
#    
##    def addline(self, xdata, ydata):
#        
#        
##==============================================================================
## Testing        
##==============================================================================
#    
#if __name__ == '__main__': # checks, if this file is run by itself (namespace = __main__) or imported as a module
#    f = RamFig(np.arange(10), np.arange(10)**2, name="Poly2Fig", label="Parabola")
#    g = RamFig(np.arange(10), np.arange(10)**3, name="Poly3Fig", label="Poly3")
#    #    print(f)
#
if __name__ == '__main__':
    # checks, if this file is run by itself (namespace = __main__) or imported as a module
    plt.close('all')
    fig1, ax = makeRamFig(np.arange(10), np.arange(10)**2, figname="Poly2Fig", label="Parabola")
    fig2, ax = makePolFig(np.arange(0, 360, 5)*2*np.pi/360, np.cos(np.arange(0, 360, 5)*2*np.pi/360)**2, "Cosine Square", label="Cos2")
    fig3, ax = makeParameterFig(np.arange(0, 360, 5)*2*np.pi/360, np.cos(np.arange(0, 360, 5)*2*np.pi/360)**2, "Cosine Square XY", label="Cos2", xlabel="Polin [°]", ylabel="Intensity [arb.u.]")


    
#%% Example

#class RFig(plt.figure):
#    def __init__(self, name):
#        self.name = name
#        plt.figure(name)
#        
        
#%% Example

#
#import matplotlib.pyplot as plt
#from matplotlib.figure import Figure
#
#
#class RamanFigure(Figure):
#    def __init__(self, *args, figtitle2='Additional Title', **kwargs):
#        """
#        custom kwarg figtitle2 is an additional figure title
#        """
#        super().__init__(*args, **kwargs)
#        self.text(0.5, 0.95, figtitle2, ha='center')
#        global ax1
#        ax1 = self.add_subplot(111)
#        ax1.set_ylabel("Raman intensity [arb.u.]")
#        ax1.set_xlabel("Raman shift [cm$^{-1}$]")
#        
##class RamSubplot(plt.axes):
##    def __init__(self, *args, figtitle2='Additional Title', **kwargs):
##        """
##        custom kwarg figtitle2 is an additional figure title
##        """
##        super().__init__(*args, **kwargs)
##        self.text(0.5, 0.95, figtitle2, ha='center')
##        global ax1
##        ax1 = self.add_subplot(111)
##        ax1.set_ylabel("Raman intensity [arb.u.]")
##        ax1.set_xlabel("Raman shift [cm$^{-1}$]")        
##        
##==============================================================================
## Testing
##==============================================================================
#if __name__ == '__main__': # checks, if this file is run by itself (namespace = __main__) or imported as a module
#
#                                                                   
#    fig = plt.figure("DefinedNameFig", FigureClass=RamanFigure)
#    ax1.plot([1,2,3])
#    fig.clf()
#    ax2 = fig.add_subplot(212)
#    ax2.plot([1, 2, 3])
#    
#    plt.show()
#    
#    
#    # does this also work like this?
#    f3, ax3 = plt.subplots(nrows=2, ncols=1, num="Fig 3", FigureClass=RamanFigure)
#    ax3
#    
#    
#    

def adjustColourScale(s_low, s_high, nlevels = '256', cmap = 'viridis'):
    """Adjust a colorscale according to max and min values."""
    levels = MaxNLocator(nbins=nlevels).tick_values(s_low, s_high)  # for lin-scale
    cmap = plt.get_cmap(cmap)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
    return cmap, norm



#%%

#==============================================================================
# Maths Functions
#==============================================================================

def lorentz(x, x0, gamma):
    return 1 / (np.pi*gamma*(1.0 + ((x-x0)/gamma)**2))


#%%

#==============================================================================
# Raman Functions
#==============================================================================

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

def convertWLToeV(WL):
    """WL in nM"""
    # E = hc/lambda
    return 1E9* const.speed_of_light * const.h / (const.electron_volt * WL)

def converteVToWL(eV):
    """WL in nM"""
    # E = hc/lambda
    return 1E9* const.speed_of_light * const.h / (const.electron_volt * eV)