# -*- coding: utf-8 -*-
"""
Created on Sun May 14 11:59:30 2017

@author: Jan


%reset does magic


"""



import numpy as np
import re
import os
import codecs
import copy

#==============================================================================
# Define Function for searching
#==============================================================================
def getInfoAfterSearchString(search,StringList):
    info = None
    for s in StringList:
        try:
            info = re.split(search, s)[1].strip()
        except:
            continue
    try: 
        info = info.split(r':', maxsplit = 1)[1].strip()
    except:
        info = info
    return info    

""" Input Path here!!!"""

#dir = os.getcwd()

Wdir = r"C:\Users\ovj\Desktop"
dir = Wdir + '\input'

listFiles = os.listdir(dir)
#==============================================================================
# Seperate Files into Info- and Spectra Files
#==============================================================================
ListInfoFiles = []
ListSpecFiles = []
for file in listFiles:
    if re.search(r'Information.txt', file) != None:
        ListInfoFiles.append(file)
    elif re.search(r'Spec.Data', file) != None:
        ListSpecFiles.append(file)
        
        
#==============================================================================
# Filter Image / Image Scan / Fit Data        
#==============================================================================
# from Info-Files
newlist=[]
for info in ListInfoFiles:
    filepath=os.path.join(dir, info) 
    with open(filepath, 'r') as f:
        infodata = f.read()
    if re.search(r"UHTS", infodata) != None:
        newlist.append(info)

ListInfoFiles=newlist

# from Spec-Files
newlist=[]
for spec in ListSpecFiles:
    if re.search(r"Spec.Data 1_F.txt", spec) == None and re.search(r'Parameter', spec) == None: # remove maps & fits
        newlist.append(spec)

ListSpecFiles=newlist
del newlist        

#==============================================================================
# Create a Dictionary to assign each Spec-File an Information File.
#==============================================================================
D = dict() # Dictionary: Key - SpecNo from ListSpecFiles, Value - InfoNo from ListInfoFiles
SpecID = None
InfoID = None
#file = ListSpecFiles[0]# for debugging
SpecNo=0
for spec in ListSpecFiles:
    SpecID = re.split(r'_Spec.Data', spec)[0][-3:]
    #SpecCheck = re.split(r'_Spec.Data', file)[0][-20:]
    InfoNo=0
    for info in ListInfoFiles:
        InfoID = re.split(r' Information', info)[0][-3:]
        if SpecID == InfoID:
            D[SpecNo] = InfoNo
        InfoNo +=1
    SpecNo +=1
    
#for x in D.keys():
#    print(ListSpecFiles[x] , "-->" , ListInfoFiles[D[x]] , "\n")
        
   
#   
##==============================================================================
## Check consistency of input files!
##==============================================================================
#
#if len(ListInfoFiles) != len(ListSpecFiles):
#    print("Not the same number of Spectra & Information Files")
#
#fileNo = 0    
#for file in ListSpecFiles:
#     SpecNo = re.split(r'_Spec.Data', file)[0][-3:]
#     InfoNo = re.split(r' Information', ListInfoFiles[fileNo])[0][-3:] 
#     if SpecNo != InfoNo:
#         print("Infodata doesn't match spectrum for No. " + SpecNo)
#     fileNo +=1
#     
#print(str(fileNo) + ' Spec-files found')         
         
         
#==============================================================================
# Loop:
#==============================================================================
       
fileNo = 0
#file = ListInfoFiles[0]  # debug
#for file in ListInfoFiles:
for x in D.keys():
    print("Opening: " + ListInfoFiles[D[x]])
#    filepath=os.path.join(dir, file) 
    filepath=os.path.join(dir, ListInfoFiles[D[x]]) 
    with open(filepath, 'r') as f:
        infodata = f.read()
        InfoNo = re.split(r' Information', file)[0][-3:]
        infoList = re.split(r'\n', infodata)
    
##==============================================================================
## Open File        
##==============================================================================
#filepath=os.path.join(dir + '/input',ListInfoFiles[1]  ) 
#with open(filepath, 'r') as f:
#    infodata = f.read()
#  
#infoList = re.split(r'\n', infodata)
   

    

#==============================================================================
# Split Information into Groups (for further data Handling)
#==============================================================================
    infodata = re.split(r'\n{2,}', infodata)
    i=0
    for info in infodata:
        if re.search(r'General:', info) != None:
            infoGeneral = re.split(r'\n', infodata[i])
        if re.search(r'UHTS', info) != None:
            infoSpec = re.split(r'\n', infodata[i])
        if re.search(r'Camera Serial Nr', info) != None:
            infoCam = re.split(r'\n', infodata[i])
        if re.search(r'Accumulations', info) != None:
            infoTint = re.split(r'\n', infodata[i])
        if re.search(r'Sample Location', info) != None:
            infoLocation = re.split(r'\n', infodata[i])
        i+=1
    
#==============================================================================
# Get relevant Info 
#==============================================================================
        
    Date = getInfoAfterSearchString('Date',infoGeneral)
    Timestamp = Date + ", " + getInfoAfterSearchString('Time',infoGeneral)
    
    LaserWL = getInfoAfterSearchString('Excitation Wavelength',infoSpec)+'nm'
    LaserPow = getInfoAfterSearchString('Power',infoSpec)+'mW'
    
    Grating = getInfoAfterSearchString('Grating:',infoSpec)
    Grating = Grating[0:Grating.find("g/mm")+4] # skip the blazing
    
    
    SpecCenter = getInfoAfterSearchString('Spectral Center', infoSpec)+'rel. cm-1'
    
    try:
        Tint = getInfoAfterSearchString('Integration Time',infoTint) 
        Tint = str(round(float(Tint))) + ' s' # round to full seconds
    except:
        print(infodata)        
        
        
    Accum = getInfoAfterSearchString('Accumul',infoTint)
    if Accum != "1":
        Tint = Accum +'x ' + Tint
        
    CCDAmp = getInfoAfterSearchString('Amplifier',infoCam)
    CCDPreAmpGain = getInfoAfterSearchString('Preamp',infoCam)
    EMCCDGain = getInfoAfterSearchString('EMCCD',infoCam)
    CCDMode = getInfoAfterSearchString('Mode',infoCam)  
    
    if CCDAmp == "Conventional":
        CCDinfo = CCDAmp + ", " + CCDMode + ", PreAmp Gain: " + CCDPreAmpGain
    else:
        CCDinfo = "EMCCD, Gain: " + EMCCDGain + ", "  + CCDMode + ", PreAmp Gain: " + CCDPreAmpGain
        
        
#==============================================================================
# Get Additional Info from Filename 
#==============================================================================
    #fname = ListSpecFiles[fileNo]
    #outname = fname.split(".txt")[0]+"_head.txt"
    #outpath = outfolder + "\\" + outname

                           
    #Wavelength = "Wavelength: " + fname[fname.find("WL")+2:fname.find("WL")+5]
    #Power = "nm, Power: " + fname[fname.find("Power")+5:fname.find("Power")+10]
    #Tint = ", T\-(int): " + fname[fname.find("Tint")+4:fname.find("Tint")+7]
    #Polin = ", Pol\-(in): " + fname[fname.find("Pol")+3:fname.find("Pol")+9]
    #Polin = fname[fname.find("Polin")+5:fname.find("Polin")+8] # searches for "Polin", --> returns position of first char, +5(len("Polin")...
    #Polout =fname[fname.find("Polout")+6:fname.find("Polout")+9]
    #Turns =fname[fname.find("Turn")+4:fname.find("Turn")+7]

    for field in re.split('_',ListSpecFiles[fileNo]):
        if field.find("Polin") != -1:
            Polin = field[field.find("Polin")+5:]
            break
        else:
            Polin = "?"
            
    for field in re.split('_',ListSpecFiles[fileNo]):
        if field.find("Polout") != -1:
            Polout = field[field.find("Polout")+6:]
            break
        else:
            Polout = "?"
          
    for field in re.split('_',ListSpecFiles[fileNo]):
        if field.find("Turn") != -1:
            Turns = field[field.find("Turn")+4:]
            cleanTurns = ""    
            for char in Turns:
                try:
                    int(char)
                    cleanTurns = cleanTurns + char
                except:
                    pass
            break
        else:
            cleanTurns = "?" 

#==============================================================================
# ToDo: Make exception for case of Turns not specified    
#==============================================================================
    



#==============================================================================
#   ToDo: Entferne "Polin" aus Feld und nehme rest!
#==============================================================================


#==============================================================================
# Open Specfile & read data
#==============================================================================
       
    specfilepath=os.path.join(dir, ListSpecFiles[x]) 
    with open(specfilepath, 'r') as spec:
        data = spec.read()
        
#==============================================================================
# Open Outfile & Write data
#==============================================================================
#    Descr=dict()
#    Descr[' Turns: '] = cleanTurns
#    Descr[', Pol\-(in): '] = Polin
#    Descr[', Pol\-(out):'] = Polout
#    Descr[', t\-(int)='] = Tint
#    Descr[', WL='] = str(round(float(LaserWL[0:-2]))) + "nm"
#    Descr[', P\-(Laser)='] = LaserPow
#    Descr[', Grating: '] = Grating
#
#    filedescr = ""
#    for y in Descr.keys():
##        if y != None and Descr[y] != None:
#            filedescr = filedescr + y + Descr[y]
        
    filedescr = "Raman Shift\tIntensity" + " Polin " + Polin + "\n rel. cm-1\tArb. Units \n x\t" + cleanTurns + " Turns" + ", Pol\-(in): " + Polin + ", Pol\-(out):" + Polout + ', t\-(int)=' + Tint + ', WL=' + str(round(float(LaserWL[0:-2]))) + "nm" + ", P\-(Laser)=" + LaserPow + ", Grating: " + Grating

    header = filedescr + "\n\t\t\n"    
#    header = "Raman Shift\tIntensity \n rel. cm-1\tArb. Units \n x\t" + filedescr + "\n\t\t\n"
    savepath = os.path.join(os.path.dirname(dir), 'output' , ListSpecFiles[fileNo]) 
    os.makedirs(os.path.join(os.path.dirname(dir), 'output'), exist_ok=True)
    writedata = header + data      
#    with open(savepath, 'w+') as outfile: # "r+" mode means reading & updating, without deleting everything in it (trucating). "w+" will truncate. x mode means: create new file and open it for writing, raises error if file exists
#        outfile.write(writedata)
    with codecs.open(savepath, 'w+', "utf-8-sig") as outfile: # "r+" mode means reading & updating, without deleting everything in it (trucating). "w+" will truncate. x mode means: create new file and open it for writing, raises error if file exists
        outfile.write(writedata)
    
    fileNo +=1

    