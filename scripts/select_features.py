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

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections

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
	parser.add_argument('-inputfile','--inputfile', dest='inputfile', required=True, type=str, help='Input feature data table filename') 
	
	# - Pre-processing options
	parser.add_argument('--normalize', dest='normalize', action='store_true',help='Normalize feature data in range [0,1] before applying models (default=false)')	
	parser.set_defaults(normalize=False)

	# - Feature selection model options
	parser.add_argument('-classifier','--classifier', dest='classifier', required=False, type=str, default='DecisionTreeClassifier', help='Classifier to be used.') 
	parser.add_argument('-scoring','--scoring', dest='scoring', required=False, type=str, default='f1_weighted', help='Classifier scoring to be used. Valid values: {f1_weighted,accuracy}') 
	parser.add_argument('-cv_nsplits','--cv_nsplits', dest='cv_nsplits', required=False, type=int, default=5, help='Number of dataset split for cross-validation') 	
	parser.add_argument('-nfeat_min','--nfeat_min', dest='nfeat_min', required=False, type=int, default=2, help='Min number of features to be scanned') 	
	parser.add_argument('-nfeat_max','--nfeat_max', dest='nfeat_max', required=False, type=int, default=-1, help='Max number of features to be scanned (-1=all)') 	
	parser.add_argument('--autoselect', dest='autoselect', action='store_true',help='Select number of features automatically (default=false)')	
	parser.set_defaults(autoselect=False)

	# - Feature selection run options
	parser.add_argument('--colselect', dest='colselect', action='store_true',help='If true, just extract selected column ids, if false run feature selection (default=false)')	
	parser.set_defaults(colselect=False)
	parser.add_argument('-selcols','--selcols', dest='selcols', required=False, type=str, default='', help='Data column ids to be selected from input data, separated by commas') 
	
	# - Output options
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='featdata_sel.dat', help='Output filename (.dat) with selected feature data') 

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
	inputfile= args.inputfile

	# - Data pre-processing
	normalize= args.normalize

	# - Model options
	classifier= args.classifier
	scoring= args.scoring
	cv_nsplits= args.cv_nsplits
	nfeat_min= args.nfeat_min
	nfeat_max= args.nfeat_max
	autoselect= args.autoselect

	# - Run options
	colselect= args.colselect
	selcols= []
	if colselect:
		if args.selcols=="":
			logger.error("No selected column ids given (mandatory when colselect option is chosen)!")
			return 1
		selcols= [int(x.strip()) for x in args.selcols.split(',')]

	# - Output options
	outfile= args.outfile

	#===========================
	#==   READ FEATURE DATA
	#===========================
	ret= Utils.read_feature_data(inputfile)
	if not ret:
		logger.error("Failed to read data from file %s!" % (inputfile))
		return 1

	data= ret[0]
	snames= ret[1]
	classids= ret[2]

	#===========================
	#==   SELECT FEATURES
	#===========================
	logger.info("Running feature selector on input feature data ...")
	fsel= FeatSelector()
	fsel.normalize= normalize
	fsel.classifier= classifier
	fsel.scoring= scoring
	fsel.outfile= outfile
	fsel.nfeat_min= nfeat_min
	fsel.nfeat_max= nfeat_max
	fsel.auto_selection= autoselect

	if colselect:
		status= fsel.select(data, selcols, classids, snames)
	else:
		status= fsel.run(data, classids, snames)
	
	if status<0:
		logger.error("Feature selector failed!")
		return 1
	


	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

