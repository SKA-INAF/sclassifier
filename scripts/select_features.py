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
	parser.add_argument('-scalerfile', '--scalerfile', dest='scalerfile', required=False, type=str, default='', action='store',help='Load and use data transform stored in this file (.sav)')
	
	# - Feature selection options
	parser.add_argument('-classifier','--classifier', dest='classifier', required=False, type=str, default='DecisionTreeClassifier', help='Classifier to be used {LGBMClassifier,DecisionTreeClassifier,RandomForestClassifier,GradientBoostingClassifier,MLPClassifier,SVC,QuadraticDiscriminantAnalysis}.') 
	parser.add_argument('-scoring','--scoring', dest='scoring', required=False, type=str, default='f1_weighted', help='Classifier scoring to be used. Valid values: {f1_weighted,accuracy}') 
	parser.add_argument('-cv_nsplits','--cv_nsplits', dest='cv_nsplits', required=False, type=int, default=5, help='Number of dataset split for cross-validation') 	
	parser.add_argument('-nfeat_min','--nfeat_min', dest='nfeat_min', required=False, type=int, default=2, help='Min number of features to be scanned') 	
	parser.add_argument('-nfeat_max','--nfeat_max', dest='nfeat_max', required=False, type=int, default=-1, help='Max number of features to be scanned (-1=all)') 	
	parser.add_argument('--autoselect', dest='autoselect', action='store_true',help='Select number of features automatically (default=false)')	
	parser.set_defaults(autoselect=False)
	
	parser.add_argument('--binary_class', dest='binary_class', action='store_true',help='Perform a binary classification {0=EGAL,1=GAL} (default=multiclass)')	
	parser.set_defaults(binary_class=False)
	parser.add_argument('--balance_classes', dest='balance_classes', action='store_true',help='Apply class weights to balance classes (default=false)')	
	parser.set_defaults(balance_classes=False)

	# - Tree options
	parser.add_argument('-max_depth','--max_depth', dest='max_depth', required=False, type=int, default=None, help='Max depth for decision tree, random forest and LGBM')
	parser.add_argument('-min_samples_split','--min_samples_split', dest='min_samples_split', required=False, type=int, default=2, help='Minimum number of samples required to split an internal node')
	parser.add_argument('-min_samples_leaf','--min_samples_leaf', dest='min_samples_leaf', required=False, type=int, default=1, help='Minimum number of samples required to be at a leaf node')
	parser.add_argument('-n_estimators','--n_estimators', dest='n_estimators', required=False, type=int, default=100, help='Number of boosted or forest trees to fit') 
	parser.add_argument('-num_leaves','--num_leaves', dest='num_leaves', required=False, type=int, default=31, help='Max number of leaves in one tree for LGBM classifier') 
	parser.add_argument('-learning_rate','--learning_rate', dest='learning_rate', required=False, type=float, default=0.1, help='Learning rate for LGBM classifier and others (TBD)') 
	parser.add_argument('-niters','--niters', dest='niters', required=False, type=int, default=100, help='Number of boosting iterations for LGBM classifier and others (TBD)') 
	

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
	scalerfile= args.scalerfile

	# - Model options
	classifier= args.classifier
	scoring= args.scoring
	cv_nsplits= args.cv_nsplits
	nfeat_min= args.nfeat_min
	nfeat_max= args.nfeat_max
	autoselect= args.autoselect

	multiclass= True
	if args.binary_class:
		multiclass= False

	balance_classes= args.balance_classes

	# - Tree options
	max_depth= args.max_depth
	min_samples_split= args.min_samples_split
	min_samples_leaf= args.min_samples_leaf
	n_estimators= args.n_estimators
	num_leaves= args.num_leaves
	learning_rate= args.learning_rate
	niters= args.niters

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
	fsel= FeatSelector(multiclass=multiclass)
	fsel.normalize= normalize
	fsel.classifier= classifier
	fsel.scoring= scoring
	fsel.outfile= outfile
	fsel.nfeat_min= nfeat_min
	fsel.nfeat_max= nfeat_max
	fsel.auto_selection= autoselect

	fsel.max_depth= max_depth
	fsel.min_samples_split= min_samples_split
	fsel.min_samples_leaf= min_samples_leaf
	fsel.n_estimators= n_estimators
	fsel.num_leaves= num_leaves
	fsel.learning_rate= learning_rate
	fsel.niters= niters
	fsel.balance_classes= balance_classes

	if colselect:
		status= fsel.select(data, selcols, classids, snames, scalerfile)
	else:
		status= fsel.run(data, classids, snames, scalerfile)
	
	if status<0:
		logger.error("Feature selector failed!")
		return 1
	


	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

