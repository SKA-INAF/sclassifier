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
import ast

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections

## MODULES
from sclassifier import __version__, __date__
from sclassifier import logger
from sclassifier.data_loader import DataLoader
from sclassifier.utils import Utils
from sclassifier.outlier_finder import OutlierFinder

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
	parser.add_argument('-datalist_key','--datalist_key', dest='datalist_key', required=False, type=str, default="data", help='Dictionary key name to be read in input datalist (default=data)') 
	
	# - Pre-processing options
	parser.add_argument('--normalize', dest='normalize', action='store_true',help='Normalize feature data in range [0,1] before applying models (default=false)')	
	parser.set_defaults(normalize=False)
	parser.add_argument('-scalerfile', '--scalerfile', dest='scalerfile', required=False, type=str, default='', action='store',help='Load and use data transform stored in this file (.sav)')
	
	# - Model options
	parser.add_argument('-modelfile', '--modelfile', dest='modelfile', required=False, type=str, default='', action='store',help='Classifier model filename (.sav)')
	parser.add_argument('--predict', dest='predict', action='store_true',help='Predict model on input data (default=false)')	
	parser.set_defaults(predict=False)
	parser.add_argument('--classid_label_map', dest='classid_label_map', required=False, type=str, default='', help='Class ID label dictionary')
	parser.add_argument('-n_estimators','--n_estimators', dest='n_estimators', required=False, type=int, default=100, help='Number of forest trees to fit') 
	parser.add_argument('-max_features','--max_features', dest='max_features', required=False, type=int, default=1, help='Number of max features used in each forest tree (default=1)')
	parser.add_argument('-max_samples','--max_samples', dest='max_samples', required=False, type=float, default=-1, help='Number of max samples used in each forest tree. -1 means auto options, e.g. 256 entries, otherwise it is the fraction of total available entries (default=-1)') 	
	parser.add_argument('-contamination','--contamination', dest='contamination', required=False, type=float, default=None, help='Fraction of outliers expected [0,0.5]. If None set it to auto (default=None)')
	parser.add_argument('-anomaly_thr','--anomaly_thr', dest='anomaly_thr', required=False, type=float, default=0.9, help='Threshold in anomaly score above which observation is set as outlier (default=0.9)') 
	
	parser.add_argument('--run_scan', dest='run_scan', action='store_true',help='Run parameter optimization scan before run (default=false)')	
	parser.set_defaults(run_scan=False)
	
	parser.add_argument('--scan_nestimators', dest='scan_nestimators', action='store_true',help='Scan n_estimators parameter (default=false)')	
	parser.set_defaults(scan_nestimators=False)
	
	parser.add_argument('--scan_maxfeatures', dest='scan_maxfeatures', action='store_true',help='Scan max_features parameter (default=false)')	
	parser.set_defaults(scan_maxfeatures=False)
	
	parser.add_argument('--scan_maxsamples', dest='scan_maxsamples', action='store_true',help='Scan max_samples parameter (default=false)')	
	parser.set_defaults(scan_maxsamples=False)
	
	parser.add_argument('--scan_contamination', dest='scan_contamination', action='store_true',help='Scan contamination parameter (default=false)')	
	parser.set_defaults(scan_contamination=False)
	
	parser.add_argument('--random_state', dest='random_state', required=False, type=int, default=None, help='Model random state (default=None)')
		
	# - Output options
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='outlier_data.dat', help='Output filename (.dat) with classified data') 
	parser.add_argument('-outfile_json','--outfile_json', dest='outfile_json', required=False, type=str, default='outlier_data.json', help='Output filename (.json) with classified data') 

	parser.add_argument('--no_save_ascii', dest='no_save_ascii', action='store_true',help='Do not save output data to ascii (default=false)')	
	parser.set_defaults(no_save_ascii=False)
	parser.add_argument('--no_save_json', dest='no_save_json', action='store_true',help='Do not save output data to json (default=false)')	
	parser.set_defaults(no_save_json=False)
	parser.add_argument('--no_save_model', dest='no_save_model', action='store_true',help='Do not save model (default=false)')	
	parser.set_defaults(no_save_model=False)
	
	parser.add_argument('--save_labels_in_ascii', dest='save_labels_in_ascii', action='store_true',help='Save class labels to ascii (default=save classids)')
	parser.set_defaults(save_labels_in_ascii=False)
	parser.add_argument('--no_save_features', dest='no_save_features', action='store_true',help='Save features in output file (default=yes)')
	parser.set_defaults(no_save_features=False)

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
	datalist_key=args.datalist_key

	# - Data pre-processing
	normalize= args.normalize
	scalerfile= args.scalerfile

	# - Model options
	modelfile= args.modelfile
	predict= args.predict
	n_estimators= args.n_estimators
	contamination= args.contamination
	if contamination is None:
		contamination= 'auto'
	anomaly_thr= args.anomaly_thr	
	max_features= args.max_features

	max_samples= "auto"
	if args.max_samples>0:
		max_samples= args.max_samples
		
	run_scan= args.run_scan
	scan_nestimators= args.scan_nestimators
	scan_maxfeatures= args.scan_maxfeatures
	scan_maxsamples= args.scan_maxsamples
	scan_contamination= args.scan_contamination
	random_state= args.random_state
	
	classid_label_map= {}
	if args.classid_label_map!="":
		try:
			classid_label_map= ast.literal_eval(args.classid_label_map)
		except Exception as e:
			logger.error("Failed to convert classid label map string to dict (err=%s)!" % (str(e)))
			return -1	
		
		print("== classid_label_map ==")
		print(classid_label_map)
	
	# - Output options
	outfile= args.outfile
	outfile_json= args.outfile_json
	no_save_ascii= args.no_save_ascii
	no_save_json= args.no_save_json
	no_save_model= args.no_save_model
	save_labels_in_ascii= args.save_labels_in_ascii
	no_save_features= args.no_save_features

	#===========================
	#==   READ FEATURE DATA
	#===========================
	#ret= Utils.read_feature_data(inputfile)
	#if not ret:
	#	logger.error("Failed to read data from file %s!" % (inputfile))
	#	return 1

	#data= ret[0]
	#snames= ret[1]
	#classids= ret[2]

	#===========================
	#==   DETECT OUTLIERS
	#===========================
	logger.info("Detecting outliers on input feature data ...")
	ofinder= OutlierFinder()
	ofinder.normalize= normalize
	ofinder.n_estimators= n_estimators
	ofinder.max_samples= max_samples
	ofinder.max_features= max_features
	ofinder.contamination= contamination
	ofinder.run_scan= run_scan
	ofinder.scan_nestimators= scan_nestimators
	ofinder.scan_maxfeatures= scan_maxfeatures
	ofinder.scan_maxsamples= scan_maxsamples
	ofinder.scan_contamination= scan_contamination
	ofinder.anomaly_thr= anomaly_thr
	
	ofinder.classid_label_map= classid_label_map
	ofinder.random_state= random_state
	ofinder.predict= predict
	
	ofinder.outfile= outfile
	ofinder.outfile_json= outfile_json
	ofinder.save_ascii= False if no_save_ascii else True
	ofinder.save_json= False if no_save_json else True
	ofinder.save_model= False if no_save_model else True
	ofinder.save_features= False if no_save_features else True
	ofinder.save_labels_in_ascii= save_labels_in_ascii
	
	#status= ofinder.run(
	#	data, classids, snames, 
	#	modelfile, scalerfile
	#)
	
	status= ofinder.run(
		datafile=inputfile, 
		modelfile=modelfile, 
		scalerfile=scalerfile, 
		datalist_key=datalist_key
	)
		
	if status<0:
		logger.error("Outlier search run failed!")
		return 1
	

	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

