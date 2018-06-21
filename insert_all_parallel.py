#!/bin/env python
# insert_all_parallel.py
# Created by AV Sivaprasad on Jun 08, 2018
# Last modified by AV Sivaprasad on Jun 18,2018
# ----------------------------------------------------------
# This script reads in the NetCDF datasets in the specified directory and
# adds them to the 'wofs_confidence' database.
# Pre-requisite: Add the following definitions into the database once
# 1. Product definitions
# 2. Dataset definitions
#
# Run this program from any directory.
# ----------------------------------------------------------
import sys
import os
from subprocess import Popen, PIPE
import re
import multiprocessing
import time
from multiprocessing import Pool

path = '/g/data/u46/wofs/confidence_albers/MrVBF/tiles'
cores = 16 # Number of workers spawned for each iteration. A value of 16 is the optimum on one node. 
        # Not sure how it works when using 32 or more cores on more than one node.
j = 0   # Count of the sets of np processes. i.e. 1600 datasets will use ~100 sets
limit = 0
# Input file is the list of all datasets. It is mandatory.
try:
    inp_list = sys.argv
    inp = inp_list[1] 
    if (inp is '1'):   path = '/g/data/u46/wofs/confidence_albers/MrVBF/tiles'
    elif (inp is '2'): path = '/g/data/u46/wofs/confidence_albers/modis/tiles' 
    elif (inp is '3'): path = '/g/data/u46/wofs/confidence_albers/urbanAreas/tiles'
    cores = int(inp_list[2]) 
except:
    pass
try:
    match = re.match( r'(/$)', path, re.M|re.I)
    if(match): pass
    else:
        path += "/"
except: # It will never get here
    print("Usage: ./insert_all_parallel.py Path")
    print("Default Path: /g/data/u46/wofs/confidence_albers/MrVBF/tiles/")
    path = '/g/data/u46/wofs/confidence_albers/MrVBF/tiles/' # Use this as default
try:
    limit = int(sys.argv[3])
except:
    pass
#    limit = 10000 # Used only for debugging. Must specify a valid index.
#-------------------------------------------------------------------------------
# Function to be spawned as workers
#-------------------------------------------------------------------------------
def f(item):
    global j
    match = re.match( r'(.*\.nc$)', item, re.M|re.I) # Take only the NetCDF files
    if(match):
        j += 1
        dataset = path + match.group()
        echo_line = "{}. Adding {}:".format(j,dataset)
        print(echo_line)
        process = Popen(["datacube", "-E", "confidence", "dataset", "add", dataset], stdout=PIPE)
        time.sleep(15) # make this wait long enough to finish the datacube processing.
    else:
        dataset = item
        echo_line = "{}. ******* Not Adding {}:".format(j,dataset)
        print(echo_line)

#-------------------------------------------------------------------------------
# Main Function
#-------------------------------------------------------------------------------
def main():
    global limit,cores
    pool = Pool(processes=cores)              # start $np worker processes. It is the optimum
    files = os.listdir(path)
    print(len(files))
    if not limit: limit = len(files)
    print("Path: {}; Cores: {}; Limit: {}".format(path,cores,limit))
    pool.map(f, files[:limit]) # Send $cores files each time until the set limit         
    print("Finished !")    
    
if __name__ == '__main__':
    main()
