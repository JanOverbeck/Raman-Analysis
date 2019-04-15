# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 20:48:14 2018


#==============================================================================
# Useful functions for data handling
#==============================================================================

@author: ovj
"""

import re
import os
import codecs
import numpy as np
import pandas as pd
import glob


def file2analyse(directory, identifier, exclude="SuperUnluckyFilename"):
    """
    use as: filepath = file2analyse(directory, identifier)
    Looks for files in directory and selects the file containing a unique "identifier" in the filename.
    If exactly one match is found its full path is returned.
    If more than one file matching the identifier is returned, an error message is displayed."""
    lf = glob.glob(directory + "\\*.*")
    files = []
    for file in lf:
        if re.search(identifier, file) and not re.search(exclude, file):
            files.append(file)
    if len(files) == 1:
        print("found: %s"%os.path.basename(files[0]))
        return files[0]
    elif len(files) > 1:
        print("More than one file matches identifier:")
        for file in files:
            print(os.path.basename(file))
        raise FileNotFoundError
    elif len(files) == 0:
        print("No file found. Directory contains the following files:")
        if len(lf) < 50:
            for file in lf:
                print(os.path.basename(file))
        else:
            print("Many (>50). Look for yourself.")       
        raise FileNotFoundError
        
if __name__ == '__main__':
    # checks, if this file is run by itself (namespace = __main__) or imported as a module
    print("testing")
    filepath = file2analyse(dir, 'avg_0eV.txt')
    



def getheader(path, end_header="[Data]"):
    """Reads (up to 1000) lines from file, searching for end_header. Returns header, headerlen."""
    with open(path, 'r') as f:
        headerlen = 0
        header = ""
        while True and headerlen < 1000:
            line = f.readline()
            headerlen +=1
            header = header + line
            if line.startswith(end_header):
                break
    return header, headerlen



def getInfoAfterSearchString(searchString, stringList, delim = ""):
    """Search for a string s in a stringList e.g.[search1 : info1, search2 : info2] and return info after this (and optional delimiter, :)."""
    info = 'unknown'
    for s in stringList:
        try:
            info = re.split(searchString, s)[1].strip()
        except:
            continue
    try: 
        info = info.split(delim, maxsplit = 1)[1].strip()
    except:
        info = info
    return info 




def write2dat(savepath, data, header = "", separator = "[Data]"):
    """Write data and optional header to file. Make directories if necessary.
    For data with d>2 dimensions supply data as a list of strings and sperator as a list of seperators between data matrices."""

    os.makedirs(os.path.dirname(savepath), exist_ok=True) # make target directory if it doesn't exist.
    filename = os.path.basename(savepath)

    if type(data) == str:
        
        writedata = header + "\n" + separator + "\n" + data   
        
        with codecs.open(savepath, 'w+', "utf-8-sig") as outfile: # "r+" mode means reading & updating, without deleting everything in it (trucating). "w+" will truncate. x mode means: create new file and open it for writing, raises error if file exists
            outfile.write(writedata)
            
            print("Saved data to file: %s" %filename)
    
    elif type(data) == list:
        print("wow - data is a list. Still need to implement this.")
        
    elif type(data) == pd.core.frame.DataFrame or type(data) == pd.core.series.Series:
        with codecs.open(savepath, 'w+', "utf-8-sig") as outfile: # "r+" mode means reading & updating, without deleting everything in it (trucating). "w+" will truncate. x mode means: create new file and open it for writing, raises error if file exists
            outfile.write(header + "\n" + separator + "\n") 
            
        data.to_csv(savepath, sep='\t', float_format='%g', mode='a')
        print("Saved DataFrame to file: %s" %filename)
        
    elif type(data) == np.ndarray:
        np.savetxt(savepath, data, header=header + "\n" + separator, delimiter="\t", comments="")
        print("Saved numpy array to file: %s" %filename)
        
    else:
        print("wrong type for data")
        

def param2header(oldheader, parameters):
    """takes an old header, splits it before '[Data]'-Statement and appends all keys and parameters from supplied dictionary of parameters. This is version 0.1 and works with data format version 1.0."""
    version = "0.1"
                                              
    try:
        header = oldheader.split("[Data]")[0]
    except:
        raise
    
    header += "\nAnalysis Parameters:\t"
            
    for p, val in parameters.items():
        header = header + '\t' + p + '\t' + str(val) + ';'
    
    p2h_version = '\t' + 'param2header version' +  '\t' + version + ';'
    data_format_version = '\tDataFormatVersion\t1.0;'
    
    header += p2h_version
    header += data_format_version
    
    
#    header += "\n[Data]\n"
    
    return header    


def header2paramParser(header, parameters):
    """Parses header for known parameters format and appends found parameters to existing dictionary of parameters, overwriting existing values."""
    
    p2h_version = "param2header version	0.1"
    hlines = header.split("\n")
    
    for line in hlines:
        if re.search(p2h_version, line):
            print("Parameters found in header.")
            desc, infos = line.split(":")
            infos = infos.split(";")
    
            for info in infos:
                info = info.strip()  #strip whitespaces and tabs at start and end
                try:
                    p, val = info.split("\t")
                except:
                    print("Can't split " + info)
                    
                try:
                    parameters[str(p)] = float(val)
                except:
                    parameters[str(p)] = val
    return parameters    


def specHeader(append_header=""):
    """Returns header for plotting a spectrum + append_header"""
    colnames = "Raman shift\tRaman intensity"
    units = "cm-1\tarb.u."
    return colnames + "\n" + units + "\n" + append_header