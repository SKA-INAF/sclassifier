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
from sclassifier_vae import __version__, __date__
from sclassifier_vae import logger
from sclassifier_vae.data_loader import DataLoader
from sclassifier_vae.utils import Utils
from sclassifier_vae.feature_selector import FeatSelector


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

	# - Model options
	parser.add_argument('-classifier','--classifier', dest='classifier', required=False, type=str, default='DecisionTreeClassifier', help='Classifier to be used.') 
	parser.add_argument('-scoring','--scoring', dest='scoring', required=False, type=str, default='f1_weighted', help='Classifier scoring to be used. Valid values: {f1_weighted,accuracy}') 
	parser.add_argument('-cv_nsplits','--cv_nsplits', dest='cv_nsplits', required=False, type=int, default=5, help='Number of dataset split for cross-validation') 	
	
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
