#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
## STANDARD MODULES
import os
import sys
import subprocess
import string
import time
import signal
from threading import Thread
import datetime
import numpy as np
import random
import math
import logging
import io

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections
import csv

## ASTRO MODULES
from astropy.io import ascii 

## MODULES
from sclassifier import __version__, __date__
from sclassifier import logger
from sclassifier.data_loader import DataLoader
from sclassifier.utils import Utils
from sclassifier.feature_selector import FeatSelector


import matplotlib.pyplot as plt

#### GET SCRIPT ARGS ####
def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

###########################
##     ARGS
###########################
def get_args():
	"""This function parses and return arguments passed in"""
	parser = argparse.ArgumentParser(description="Parse args.")

	# - Input options
	parser.add_argument('-inputfiles','--inputfiles', dest='inputfiles', required=True, type=str, help='Input feature data table filenames, separated by commas') 
	
	parser.add_argument('--allow_novars', dest='allow_novars', action='store_true',help='Allow merging of files with no features (e.g. only sname & id)')	
	parser.set_defaults(allow_zerovars=False)

	# - Output options
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='featdata_merged.dat', help='Output filename (.dat) with selected feature data') 

	args = parser.parse_args()	

	return args



##############
##   MAIN   ##
##############
def main():
	"""Main function"""

	#===========================
	#==   PARSE ARGS
	#===========================
	logger.info("Get script args ...")
	try:
		args= get_args()
	except Exception as ex:
		logger.error("Failed to get and parse options (err=%s)",str(ex))
		return 1

	# - Input filelist
	if args.inputfiles=="":
		logger.error("Empty input file list!")
		return 1
	inputfiles= [str(x.strip()) for x in args.inputfiles.split(',')]

	allow_novars= args.allow_novars

	# - Output options
	outfile= args.outfile

	#===========================
	#==   READ FEATURE DATA
	#===========================
	dlist= []
	nvars_tot= 0

	for i in range(len(inputfiles)):
		inputfile= inputfiles[i]
		colprefix= "featset" + str(i+1) + "_"
		d= Utils.read_feature_data_dict(inputfile, colprefix=colprefix, allow_novars=allow_novars)
		if not d or d is None:
			logger.error("Failed to read data from file %s!" % (inputfile))
			return 1

		nentries= len(d.keys())
		firstitem= next(iter(d.items()))
		nvars= len(firstitem[1].keys()) - 2
		nvars_tot+= nvars
		logger.info("Data file %s has #%d entries (#%d vars) ..." % (inputfile, nentries, nvars))

		dlist.append(d)
		
		#for key, value in d.items():
		#	print("key")
		#	print(key)
		#	print("value")
		#	print(value)

	logger.info("Merged set is expected to have %d vars ..." % (nvars_tot))

	#===========================
	#==   MERGE FEATURES
	#===========================
	# - Merge features
	logger.info("Merging feature data for all input files ...")

	dmerged= collections.OrderedDict()

	for d in dlist:
		for key, value in d.items():
			if key not in dmerged:
				dmerged[key]= collections.OrderedDict({})
			dmerged[key].update(value)
			dmerged[key].move_to_end("id")

	
	# - Remove rows with less number of entries
	logger.info("Removing rows with number of vars !=%d ..." % (nvars_tot))

	par_dict_list= []
	for key, value in dmerged.items():
		nvars= len(value.keys())-2
		if nvars!=nvars_tot:
			logger.info("Removing entry (%s) as number of vars (%d) is !=%d ..." % (key, nvars, nvars_tot))
			#del dmerged[key]
			continue
		par_dict_list.append(value)

	#print("dmerged")
	#for key, value in dmerged.items():
	#		print("key")
	#		print(key)
	#		print("value")
	#		print(value)

	#print("par_dict_list")
	#print(par_dict_list)

	#===========================
	#==   SAVE
	#===========================
	# - Write CSV
	outfile_noext= os.path.splitext(outfile)[0]
	outfile_csv= outfile_noext + '.csv'
	logger.info("Saving merged feature data to file %s ..." % (outfile_csv))

	parnames = par_dict_list[0].keys()
	#print("parnames")
	#print(parnames)
		
	with open(outfile_csv, 'w') as fp:
		fp.write("# ")
		dict_writer = csv.DictWriter(fp, parnames)
		dict_writer.writeheader()
		dict_writer.writerows(par_dict_list)

	# - Read CSV
	logger.info("Reading CSV written table %s ..." % (outfile_csv))
	row_start= 0
	table= ascii.read(outfile_csv, data_start=row_start)
	colnames= table.colnames
	colnames_mod= np.copy(colnames)
	colnames_mod[0]= '# ' + colnames[0]
		
	# - Write ASCII (replace quotes that astropy puts in colnames after #)
	logger.info("Writing ascii table to file %s ..." % (outfile))
	buf= io.StringIO()
	ascii.write(table, buf, format="basic", names=colnames_mod, overwrite=True)
	with open(outfile, 'w') as f:
		f.write(buf.getvalue().replace('"',""))

	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())




