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
	#parser.add_argument('--normalize', dest='normalize', action='store_true',help='Normalize feature data in range [0,1] before UMAP (default=false)')	
	#parser.set_defaults(normalize=False)	
	
	# - UMAP classifier options
	parser.add_argument('-latentdim_umap', '--latentdim_umap', dest='latentdim_umap', required=False, type=int, default=2, action='store',help='Encoded data dim in UMAP (default=2)')
	parser.add_argument('-mindist_umap', '--mindist_umap', dest='mindist_umap', required=False, type=float, default=0.1, action='store',help='Min dist UMAP par (default=0.1)')
	parser.add_argument('-nneighbors_umap', '--nneighbors_umap', dest='nneighbors_umap', required=False, type=int, default=15, action='store',help='N neighbors UMAP par (default=15)')
	parser.add_argument('-outfile_umap_unsupervised', '--outfile_umap_unsupervised', dest='outfile_umap_unsupervised', required=False, type=str, default='latent_data_umap_unsupervised.dat', action='store',help='Name of UMAP encoded data output file')
	parser.add_argument('-outfile_umap_supervised', '--outfile_umap_supervised', dest='outfile_umap_supervised', required=False, type=str, default='latent_data_umap_supervised.dat', action='store',help='Name of UMAP output file with encoded data produced using supervised method')
	parser.add_argument('-outfile_umap_preclassified', '--outfile_umap_preclassified', dest='outfile_umap_preclassified', required=False, type=str, default='latent_data_umap_preclass.dat', action='store',help='Name of UMAP output file with encoded data produced from pre-classified data')

	parser.add_argument('-modelfile_umap', '--modelfile_umap', dest='modelfile_umap', required=False, type=str, action='store',help='UMAP model filename (.h5)')

	parser.add_argument('--predict', dest='predict', action='store_true',help='Only predict data according to loaded UMAP model (default=false)')	
	parser.set_defaults(predict=False)

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
	#normalize= args.normalize
	
	# - UMAP options
	latentdim_umap= args.latentdim_umap
	mindist_umap= args.mindist_umap
	nneighbors_umap= args.nneighbors_umap
	outfile_umap_unsupervised= args.outfile_umap_unsupervised
	outfile_umap_supervised= args.outfile_umap_supervised
	outfile_umap_preclassified= args.outfile_umap_preclassified

	modelfile_umap= args.modelfile_umap
	predict= args.predict

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
	#==   RUN UMAP
	#==============================
	logger.info("Running UMAP classifier training on input feature data ...")
	umap_class= FeatExtractorUMAP()

	umap_class.set_encoded_data_unsupervised_outfile(outfile_umap_unsupervised)
	umap_class.set_encoded_data_supervised_outfile(outfile_umap_supervised)
	umap_class.set_encoded_data_preclassified_outfile(outfile_umap_preclassified)
	umap_class.set_encoded_data_dim(latentdim_umap)
	umap_class.set_min_dist(mindist_umap)
	umap_class.set_n_neighbors(nneighbors_umap)

	status= 0
	if predict:
		if umap_class.run_predict(data, class_ids=classids, snames=snames, modelfile=modelfile_umap)<0:
			logger.error("UMAP prediction failed!")
			return 1
	else:
		if umap_class.run_train(data, class_ids=classids, snames=snames)<0:
			logger.error("UMAP training failed!")
			return 1


	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

