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
from sclassifier_vae.classifier import SClassifier


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
	parser.add_argument('-modelfile', '--modelfile', dest='modelfile', required=False, type=str, default='', action='store',help='Classifier model filename (.sav)')
	parser.add_argument('--predict', dest='predict', action='store_true',help='Predict model on input data (default=false)')	
	parser.set_defaults(predict=False)

	# - Decision Tree options
	parser.add_argument('-max_depth','--max_depth', dest='max_depth', required=False, type=int, default=3, help='Max depth for decision tree or random forest') 
	

	# - Output options
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='classified_data.dat', help='Output filename (.dat) with classified data') 

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
	modelfile= args.modelfile
	predict= args.predict

	# - Decision Tree options
	max_depth= args.max_depth
	
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
	#==   CLASSIFY DATA
	#===========================
	logger.info("Running classifier on input feature data ...")
	sclass= SClassifier()
	sclass.normalize= normalize
	sclass.classifier= classifier
	sclass.outfile= outfile
	sclass.max_depth= max_depth

	if predict:
		status= sclass.run_predict(data, classids, snames, modelfile)
	else:
		status= sclass.run_train(data, classids, snames, modelfile)
	
	if status<0:
		logger.error("Classifier run failed!")
		return 1
	

	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

