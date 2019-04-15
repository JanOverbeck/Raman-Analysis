# -*- coding: utf-8 -*-
"""
Script for plotting Spectral data from input files with assumed structure: | wavelength[nm] | CCD counts |



Created on Wed Feb  1 12:28:51 2017

@author: Jan Overbeck


%reset does magic


"""


"""Enter path of python script folder"""


import os
import re

#os.chdir(r"C:\Users\ovj\Desktop\ToDo")



#%%
"""Input Needed!"""

dir = r'C:\Users\ovj\Desktop\ToDo\K11_4mW' #needs to be a raw string!





unifier = "Information.txt" # Unifying part of File Name (selects files to open)

searchitem = "Laser Power [mW]:"#  Integration Time [s]:" # Parameter to search and retrieve

                               
                               #Laser Power [mW]:
                          
#%%                        
                          

#Look at all files foun.d in directory
files = os.listdir(dir)             
                          
# extract unit and description from searchitem                       
try:
    unit = re.search("\[(\D+)\]",searchitem).group(1)
except:
    unit = []
    
try:
    searchname = re.sub(".\[.+$", "",searchitem)
except:
    searchname = []
    
rsearchitem = re.escape(searchitem) # turn it into rawstring

#%%


parselist = []
outdata = ["Filename \t num \t" + searchname, "\t\t"+unit]



for i,f in enumerate(files):
    if re.search(unifier,f,1) != None:
        parselist.append(f)     
        
        
for k in parselist:      
    """Specify which file to analyse"""
    filepath=os.path.join(dir, k)    
#%% Extract data from filename           
    num = re.split("1s_",k)[1][0:5] #   

    try:
        polin = re.search("Polin(\d\d\d)",k).group(1)
    except:
        polin = ""



                
#%%
#==============================================================================
# Load data
#==============================================================================

    with open(filepath, 'r') as f:
        data = f.read()
        val = re.search(rsearchitem+".(\d.+)",data).group(1)

    outdata.append(k + "\t" + num + "\t" + val)
#    outdata.append(k + "\t" + num + "\t" + val + "\t" + polin)    

"""alternatives"""

#==============================================================================
#      with open(filepath, 'r') as f:
#          infodata = f.readlines()
# 
# #==============================================================================
# #  search for searchitem
# #==============================================================================
# 
#     for l, line in enumerate(infodata):
#         if re.search(rsearchitem,line) != None:
#             outdata.append(k + "\t" + num + "\t" + re.split(r":",line)[1])    
#==============================================================================
            


#==============================================================================
#         print(re.sub(r'Laser Power \[mW\]:\t',"",infodata[12])) # alternative via substitution of string with empty string
#==============================================================================



#%%    

#==============================================================================
# save
#==============================================================================
    
with open(os.path.join(dir, "Textfiles_Parsed_" + searchname + ".txt"), 'w+') as outfile:
    for item in outdata:
        outfile.write(item + "\n")

            
            
            
            