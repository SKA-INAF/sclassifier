#!/usr/bin/env python

from __future__ import print_function

##################################################
###    SET SEED FOR REPRODUCIBILITY (DEBUG)
##################################################
#from numpy.random import seed
#seed(1)
#import tensorflow
#tensorflow.random.set_seed(2)

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
	parser.add_argument('--normalize', dest='normalize', action='store_true',help='Normalize feature data in range [0,1] before clustering (default=false)')	
	parser.set_defaults(normalize=False)	
	parser.add_argument('-scalerfile', '--scalerfile', dest='scalerfile', required=False, type=str, default='', action='store',help='Load and use data transform stored in this file (.sav)')
	
	parser.add_argument('--reduce_dim', dest='reduce_dim', action='store_true',help='Reduce feature data dimensionality before applying the clustering (default=false)')	
	parser.set_defaults(reduce_dim=False)
	parser.add_argument('-reduce_dim_method', '--reduce_dim_method', dest='reduce_dim_method', default='pca', required=False, type=str, action='store',help='Dimensionality reduction method {pca} (default=pca)')
	parser.add_argument('-pca_ncomps', '--pca_ncomps', dest='pca_ncomps', required=False, type=int, default=-1, action='store',help='Number of PCA components to be used (-1=retain all cumulating a variance above threshold) (default=-1)')
	parser.add_argument('-pca_varthr', '--pca_varthr', dest='pca_varthr', required=False, type=float, default=0.9, action='store',help='Cumulative variance threshold used to retain PCA components (default=0.9)')

	# - Clustering options
	parser.add_argument('-min_cluster_size', '--min_cluster_size', dest='min_cluster_size', required=False, type=int, default=5, action='store',help='Minimum cluster size for HDBSCAN clustering (default=5)')
	parser.add_argument('-min_samples', '--min_samples', dest='min_samples', required=False, type=int, default=None, action='store',help='Minimum cluster sample parameter for HDBSCAN clustering. Typically equal to min_cluster_size (default=None')	
	parser.add_argument('-modelfile_clust', '--modelfile_clust', dest='modelfile_clust', required=False, type=str, action='store',help='Clustering model filename (.h5)')
	parser.add_argument('--predict_clust', dest='predict_clust', action='store_true',help='Only predict clustering according to current clustering model (default=false)')	
	parser.set_defaults(predict_clust=False)

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
	reduce_dim= args.reduce_dim
	reduce_dim_method= args.reduce_dim_method
	pca_ncomps= args.pca_ncomps
	pca_varthr= args.pca_varthr
	
	# - Clustering options
	min_cluster_size= args.min_cluster_size
	min_samples= args.min_samples	
	modelfile_clust= args.modelfile_clust
	predict_clust= args.predict_clust

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
	#==   RUN CLUSTERING
	#==============================
	logger.info("Running HDBSCAN classifier prediction on input feature data ...")
	clust_class= Clusterer()
	clust_class.min_cluster_size= min_cluster_size
	clust_class.min_samples= min_samples
	clust_class.normalize= normalize
	clust_class.reduce_dim= reduce_dim
	clust_class.reduce_dim_method= reduce_dim_method
	clust_class.pca_ncomps= pca_ncomps
	clust_class.pca_varthr= pca_varthr

	status= 0
	if predict_clust:
		if clust_class.run_predict(data, class_ids=classids, snames=snames, modelfile=modelfile_clust, scalerfile=scalerfile)<0:
			logger.error("Clustering predict failed!")
			return 1
	else:
		if clust_class.run_clustering(data, class_ids=classids, snames=snames, modelfile=modelfile_clust, scalerfile=scalerfile)<0:
			logger.error("Clustering run failed!")
			return 1

	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

