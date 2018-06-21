#!/bin/env python
# insert_all_serial.py
# Created by AV Sivaprasad on Jun 08, 2018
# Last modified by AV Sivaprasad on Jun 18,2018
# ----------------------------------------------------------
# This script reads in the NetCDF datasets in the specified directory and
# adds them to the 'wofs_confidence' database.
# Pre-requisite: Add the following definitions into the database once
# 1. Product definitions
# 2. Dataset definitions
# ----------------------------------------------------------
import sys
import os
from subprocess import Popen, PIPE
import re
# Input file is the list of all datasets. It is mandatory.
try:
	path = sys.argv[1]
	match = re.match( r'(/$)', path, re.M|re.I)
	if(match): pass
	else:
		path += "/"
except:
	print("Usage: /g/data/u46/users/sa9525/avs/WOfS/insert_all.py Dataset_Dir")
	print("Example Dir: /g/data/u46/wofs/confidence_albers/MrVBF/tiles/")
	sys.exit()
try:
	limit = int(sys.argv[2])
except:
	limit = 100000
print("Dir: {}; Limit: {}".format(path,limit))
files = os.listdir(path)
j=0

# We are loading only the NetCDF files
for item in files:
	match = re.match( r'(.*\.nc$)', item, re.M|re.I)
	if(match):
		j += 1
		dataset = path + match.group()
		echo_line = "{}. Adding {}:\n".format(j,dataset)
		print(echo_line)
		process = Popen(["datacube", "-E", "confidence", "dataset", "add", dataset], stdout=PIPE)
		(output, err) = process.communicate()
		exit_code = process.wait()
		output = output.decode("utf-8")
		print(output)
		if(j > limit): sys.exit()


