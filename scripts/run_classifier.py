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
import json
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
from sclassifier.classifier import SClassifier


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
	parser.add_argument('-inputfile_cv','--inputfile_cv', dest='inputfile_cv', type=str, default='', help='Input feature validation data table filename') 
	
	# - Pre-processing options
	parser.add_argument('--normalize', dest='normalize', action='store_true',help='Normalize feature data in range [0,1] before applying models (default=false)')	
	parser.set_defaults(normalize=False)
	parser.add_argument('-scalerfile', '--scalerfile', dest='scalerfile', required=False, type=str, default='', action='store',help='Load and use data transform stored in this file (.sav)')
	
	# - Model options
	parser.add_argument('-classifier','--classifier', dest='classifier', required=False, type=str, default='DecisionTreeClassifier', help='Classifier to be used {DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, MLPClassifier, SVC, QuadraticDiscriminantAnalysis, LinearClassifier, LGBMClassifier}.') 
	parser.add_argument('-modelfile', '--modelfile', dest='modelfile', required=False, type=str, default='', action='store',help='Classifier model filename (.sav)')
	parser.add_argument('--predict', dest='predict', action='store_true',help='Predict model on input data (default=false)')	
	parser.set_defaults(predict=False)
	parser.add_argument('--binary_class', dest='binary_class', action='store_true',help='Perform a binary classification {0=EGAL,1=GAL} (default=multiclass)')	
	parser.set_defaults(binary_class=False)
	parser.add_argument('--balance_classes', dest='balance_classes', action='store_true',help='Apply class weights to balance classes (default=false)')	
	parser.set_defaults(balance_classes=False)
	
	# - Override class target configuration
	parser.add_argument('--classid_remap', dest='classid_remap', required=False, type=str, default='', help='Class ID remap dictionary')
	parser.add_argument('--target_label_map', dest='target_label_map', required=False, type=str, default='', help='Target label dictionary')	
	parser.add_argument('--classid_label_map', dest='classid_label_map', required=False, type=str, default='', help='Class ID label dictionary')
	parser.add_argument('--target_names', dest='target_names', required=False, type=str, default='', help='Target names')
	parser.add_argument('-nclasses', '--nclasses', dest='nclasses', required=False, type=int, default=0, action='store', help='Number of classes (default=-1)')
	parser.add_argument('--objids_excluded_in_train', dest='objids_excluded_in_train', required=False, type=str, default='-1,0', help='Source ids not included for training as considered unknown classes')

	# - Tree options
	parser.add_argument('-max_depth','--max_depth', dest='max_depth', required=False, type=int, default=None, help='Max depth for decision tree, random forest and LGBM')
	parser.add_argument('-min_samples_split','--min_samples_split', dest='min_samples_split', required=False, type=int, default=2, help='Minimum number of samples required to split an internal node')
	parser.add_argument('-min_samples_leaf','--min_samples_leaf', dest='min_samples_leaf', required=False, type=int, default=1, help='Minimum number of samples required to be at a leaf node')
	parser.add_argument('-n_estimators','--n_estimators', dest='n_estimators', required=False, type=int, default=100, help='Number of boosted or forest trees to fit') 
	parser.add_argument('-num_leaves','--num_leaves', dest='num_leaves', required=False, type=int, default=31, help='Max number of leaves in one tree for LGBM classifier') 
	parser.add_argument('-learning_rate','--learning_rate', dest='learning_rate', required=False, type=float, default=0.1, help='Learning rate for LGBM classifier and others (TBD)') 
	parser.add_argument('-niters','--niters', dest='niters', required=False, type=int, default=100, help='Number of boosting iterations for LGBM classifier and others (TBD)') 
	parser.add_argument('--importance_type', dest='importance_type', required=False, type=str, default='split', help='LGBM importance_type parameter {"split","gain"}')
	parser.add_argument('--feature_fraction', dest='feature_fraction', required=False, type=float, default=1.0, help='LGBM feature_fraction parameter (default=1.0)')
	parser.add_argument('--max_bin', dest='max_bin', required=False, type=int, default=255, help='LGBM max_bin parameter (default=1.0)')
	
	parser.add_argument('--early_stop_round', dest='early_stop_round', required=False, type=int, default=10, help='LGBM early_stop_round if val data is available (default=10)')
	parser.add_argument('--scan_test_size', dest='scan_test_size', required=False, type=float, default=0.3, help='LGBM scan fold test size. (default=0.2)')
	parser.add_argument('-nsplits','--nsplits', dest='nsplits', required=False, type=int, default=5, help='Number of Optuna fold splits (default=5)')
	parser.add_argument('--optimize_f1score', dest='optimize_f1score', action='store_true',help='Optimize F1-score (maximize) in LGBM scan, otherwise val loss (minimize) (default=false)')	
	parser.set_defaults(optimize_f1score=False)
	parser.add_argument('--split_samples_in_scan', dest='split_samples_in_scan', action='store_true',help='Use train sample split to run scan. If false use train/val data. NB: in this case you must pass val data (default=false)')	
	parser.set_defaults(split_samples_in_scan=False)
	
	parser.add_argument('--scan_min_data_in_leaf', dest='scan_min_data_in_leaf', action='store_true',help='Enable scan of min_data_in_leaf LGBM par (default=false)')	
	parser.set_defaults(scan_min_data_in_leaf=False)
	parser.add_argument('--scan_lr', dest='scan_lr', action='store_true',help='Enable scan of learning_rate LGBM par (default=false)')	
	parser.set_defaults(scan_lr=False)
	parser.add_argument('--scan_nestimators', dest='scan_nestimators', action='store_true',help='Enable scan of n_estimators LGBM par (default=false)')	
	parser.set_defaults(scan_nestimators=False)
	parser.add_argument('--scan_maxdepth', dest='scan_maxdepth', action='store_true', help='Enable scan of max_depth LGBM par (default=false)')	
	parser.set_defaults(scan_maxdepth=False)
	parser.add_argument('--scan_featfract', dest='scan_featfract', action='store_true', help='Enable scan of feature_fraction LGBM par (default=false)')	
	parser.set_defaults(scan_featfract=False)
	parser.add_argument('--scan_num_leaves', dest='scan_num_leaves', action='store_true', help='Enable scan of num_leaves LGBM par (default=false)')	
	parser.set_defaults(scan_num_leaves=False)
	parser.add_argument('--scan_maxbin', dest='scan_maxbin', action='store_true', help='Enable scan of max_bin LGBM par (default=false)')	
	parser.set_defaults(scan_maxbin=False)
	
	# - Linear classifier custom options
	parser.add_argument('-tol','--tol', dest='tol', required=False, type=float, default=None, help='Linear classifier tol parameter')
	parser.add_argument('-verbosity','--verbosity', dest='verbosity', required=False, type=int, default=None, help='Linear classifier verbosity parameter')
	
	
	# - Outlier detection
	parser.add_argument('--find_outliers', dest='find_outliers', action='store_true',help='Find outliers in data (only in prediction step) (default=false)')	
	parser.set_defaults(find_outliers=False)
	parser.add_argument('-modelfile_outlier', '--modelfile_outlier', dest='modelfile_outlier', required=False, type=str, default='', action='store',help='Outlier model filename (.sav)')
	parser.add_argument('-anomaly_thr','--anomaly_thr', dest='anomaly_thr', required=False, type=float, default=0.9, help='Threshold in anomaly score above which observation is set as outlier (default=0.9)') 
	parser.add_argument('--save_outlier', dest='save_outlier', action='store_true',help='Save outlier results to file (default=false)')	
	parser.set_defaults(save_outlier=False)
	parser.add_argument('-outfile_outlier','--outfile_outlier', dest='outfile_outlier', required=False, type=str, default='outlier_data.dat', help='Output filename (.dat) with outlier data') 
	
	# - Run option
	parser.add_argument('--run_scan', dest='run_scan', action='store_true',help='Run LGBM scan (default=false)')	
	parser.set_defaults(run_scan=False)
	parser.add_argument('-ntrials','--ntrials', dest='ntrials', required=False, type=int, default=1, help='Number of Optuna study trials (default=1)') 
	
	# - Output options
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='classified_data.dat', help='Output filename (.dat) with classified data') 

	# - Draw options
	parser.add_argument('--feature_names', dest='feature_names', required=False, type=str, default='', help='Feature names shown in plots')

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
	inputfile_cv= args.inputfile_cv

	# - Data pre-processing
	normalize= args.normalize
	scalerfile= args.scalerfile

	# - Model options
	classifier= args.classifier
	modelfile= args.modelfile
	predict= args.predict
	multiclass= True
	if args.binary_class:
		multiclass= False

	balance_classes= args.balance_classes
	
	# - Override target/class id configuration
	classid_remap= {}
	if args.classid_remap!="":
		try:
			classid_remap= ast.literal_eval(args.classid_remap)
		except Exception as e:
			logger.error("Failed to convert class id remap string to dict (err=%s)!" % (str(e)))
			return -1	
		
		print("== classid_remap ==")
		print(classid_remap)

	target_label_map= {}
	if args.target_label_map!="":
		try:
			target_label_map= ast.literal_eval(args.target_label_map)
		except Exception as e:
			logger.error("Failed to convert target label map string to dict (err=%s)!" % (str(e)))
			return -1	
		
		print("== target_label_map ==")
		print(target_label_map)

	classid_label_map= {}
	if args.classid_label_map!="":
		try:
			classid_label_map= ast.literal_eval(args.classid_label_map)
		except Exception as e:
			logger.error("Failed to convert classid label map string to dict (err=%s)!" % (str(e)))
			return -1	
		
		print("== classid_label_map ==")
		print(classid_label_map)
		
	target_names= []
	if args.target_names:
		target_names= [str(x) for x in args.target_names.split(',')]
		
	nclasses= args.nclasses
	
	objids_excluded_in_train= [int(x) for x in args.objids_excluded_in_train.split(',')]

	# - Tree options
	max_depth= args.max_depth
	min_samples_split= args.min_samples_split
	min_samples_leaf= args.min_samples_leaf
	n_estimators= args.n_estimators
	num_leaves= args.num_leaves
	learning_rate= args.learning_rate
	niters= args.niters
	importance_type= args.importance_type
	feature_fraction= args.feature_fraction
	max_bin= args.max_bin
	early_stop_round= args.early_stop_round
	nsplits= args.nsplits
	optimize_f1score= args.optimize_f1score
	split_samples_in_scan= args.split_samples_in_scan
	scan_test_size= args.scan_test_size
	if args.scan_test_size>1:
		scan_test_size= int(args.scan_test_size)
		
	scan_min_data_in_leaf= args.scan_min_data_in_leaf
	scan_lr= args.scan_lr
	scan_nestimators= args.scan_nestimators
	scan_maxdepth= args.scan_maxdepth
	scan_featfract= args.scan_featfract
	scan_num_leaves= args.scan_num_leaves
	scan_maxbin= args.scan_maxbin
	
	# - Linear classifier options
	tol= args.tol
	verbosity= args.verbosity

	# - Outlier search options
	find_outliers= args.find_outliers
	modelfile_outlier= args.modelfile_outlier
	anomaly_thr= args.anomaly_thr
	save_outlier= args.save_outlier
	outfile_outlier= args.outfile_outlier

	# - Run options
	run_scan= args.run_scan
	ntrials= args.ntrials
	
	# - Output options
	outfile= args.outfile
	
	# - Draw options
	feature_names= ''
	if args.feature_names!='':
		feature_names= [str(x) for x in args.feature_names.split(',')]
	

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

	#====================================
	#==   READ FEATURE VALIDATION DATA
	#====================================
	data_cv= None
	snames_cv= []
	classids_cv= []

	if inputfile_cv!="":
		ret_cv= Utils.read_feature_data(inputfile_cv)
		if not ret_cv:
			logger.error("Failed to read validation data from file %s!" % (inputfile_cv))
			return 1

		data_cv= ret_cv[0]
		snames_cv= ret_cv[1]
		classids_cv= ret_cv[2]

	#===========================
	#==   CLASSIFY DATA
	#===========================
	logger.info("Running classifier on input feature data ...")
	sclass= SClassifier(multiclass=multiclass)
	sclass.normalize= normalize
	sclass.classifier= classifier
	sclass.outfile= outfile
	sclass.max_depth= max_depth
	sclass.min_samples_split= min_samples_split
	sclass.min_samples_leaf= min_samples_leaf
	sclass.n_estimators= n_estimators
	sclass.num_leaves= num_leaves
	sclass.learning_rate= learning_rate
	sclass.max_bin= max_bin
	sclass.niters= niters
	sclass.balance_classes= balance_classes
	sclass.importance_type= importance_type
	sclass.feature_fraction= feature_fraction
	sclass.early_stop_round= early_stop_round
	sclass.scan_test_size= scan_test_size
	sclass.nsplits= nsplits
	sclass.optimize_f1score= optimize_f1score
	sclass.split_samples_in_scan= split_samples_in_scan
	sclass.scan_min_data_in_leaf= scan_min_data_in_leaf
	sclass.scan_lr= scan_lr
	sclass.scan_nestimators= scan_nestimators
	sclass.scan_maxdepth= scan_maxdepth
	sclass.scan_featfract= scan_featfract
	sclass.scan_num_leaves= scan_num_leaves
	sclass.scan_maxbin= scan_maxbin
	
	sclass.tol= tol
	sclass.verbosity= verbosity

	sclass.find_outliers = find_outliers 
	sclass.outlier_modelfile = modelfile_outlier
	sclass.outlier_thr = anomaly_thr
	sclass.save_outlier= save_outlier
	sclass.outlier_outfile = outfile_outlier
		
	# - Override class target configuration?
	if classid_remap:
		sclass.set_classid_remap(classid_remap)

	if target_label_map:
		sclass.target_label_map= target_label_map

	if classid_label_map:
		sclass.set_classid_label_map(classid_label_map)	
		
	if nclasses>0:
		sclass.nclasses= nclasses
	
	if target_names:
		sclass.target_names= target_names
			
	sclass.excluded_objids_train = objids_excluded_in_train	
	
	sclass.feature_names= feature_names
		
	# - Run predict/train
	if predict:
		status= sclass.run_predict(
			data, classids, snames, 
			modelfile, scalerfile
		)
	else:
		if run_scan:
			status= sclass.run_lgbm_scan(
				data, classids, snames,
				scalerfile,
				data_cv, classids_cv, snames_cv,
				n_trials=ntrials
			)

		else:
			status= sclass.run_train(
				data, classids, snames, 
				modelfile, scalerfile,
				data_cv, classids_cv, snames_cv,
			)
	
	if status<0:
		logger.error("Classifier run failed!")
		return 1
	

	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

