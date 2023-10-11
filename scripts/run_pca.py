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
from sclassifier.utils import Utils
from sclassifier.data_loader import DataLoader
from sclassifier.clustering import Clusterer

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
	parser.add_argument('--normalize', dest='normalize', action='store_true',help='Normalize feature data before PCA (default=false)')	
	parser.set_defaults(normalize=False)
	parser.add_argument('-norm_transf', '--norm_transf', dest='norm_transf', required=False, type=str, default='minmax', action='store',help='Normalization transf to be applied: {"minmax","robust"} (default=minmax)')
	
	parser.add_argument('-scalerfile', '--scalerfile', dest='scalerfile', required=False, type=str, default='', action='store',help='Load and use data transform stored in this file (.sav)')
	parser.add_argument('-modelfile', '--modelfile', dest='modelfile', required=False, type=str, default='', action='store',help='PCA model filename (.h5)')
	
	parser.add_argument('-pca_ncomps', '--pca_ncomps', dest='pca_ncomps', required=False, type=int, default=-1, action='store',help='Number of PCA components to be used (-1=retain all cumulating a variance above threshold) (default=-1)')
	parser.add_argument('-pca_varthr', '--pca_varthr', dest='pca_varthr', required=False, type=float, default=0.9, action='store',help='Cumulative variance threshold used to retain PCA components (default=0.9)')

	parser.add_argument('--classid_label_map', dest='classid_label_map', required=False, type=str, default='', help='Class ID label dictionary')
	
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
	norm_transf= args.norm_transf
	
	classid_label_map= {}
	if args.classid_label_map!="":
		try:
			classid_label_map= ast.literal_eval(args.classid_label_map)
		except Exception as e:
			logger.error("Failed to convert classid label map string to dict (err=%s)!" % (str(e)))
			return -1	
		
		print("== classid_label_map ==")
		print(classid_label_map)

	# - PCA model options
	modelfile= args.modelfile
	pca_ncomps= args.pca_ncomps
	pca_varthr= args.pca_varthr
	
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

	#==============================
	#==   RUN PCA
	#==============================
	logger.info("Running PCA on input feature data ...")
	clust= Clusterer()
	clust.normalize= normalize
	clust.norm_transf= norm_transf
	clust.pca_ncomps= pca_ncomps
	clust.pca_varthr= pca_varthr
	clust.classid_label_map= classid_label_map
	
	status= 0
	if clust.run_pca(data, class_ids=classids, snames=snames, modelfile=modelfile, scalerfile=scalerfile)<0:
		logger.error("PCA run failed!")
		return 1
	
	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

