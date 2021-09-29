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
from sclassifier_vae import __version__, __date__
from sclassifier_vae import logger
from sclassifier_vae.data_loader import DataLoader
from sclassifier_vae.classifier import VAEClassifier
from sclassifier_vae.classifier_umap import UMAPClassifier

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
	parser.add_argument('-datalist','--datalist', dest='datalist', required=True, type=str, help='Input data json filelist') 
	
	# - Data pre-processing options
	parser.add_argument('-nx', '--nx', dest='nx', required=False, type=int, default=128, action='store',help='Image resize width in pixels (default=128)')
	parser.add_argument('-ny', '--ny', dest='ny', required=False, type=int, default=128, action='store',help='Image resize height in pixels (default=128)')	

	# - Autoencoder model options
	parser.add_argument('-modelfile', '--modelfile', dest='modelfile', required=True, type=str, action='store',help='Autoencoder model filename (.h5)')

	# - UMAP classifier options
	parser.add_argument('--run_umap', dest='run_umap', action='store_true',help='Run UMAP of VAE latent vector')	
	parser.set_defaults(run_umap=False)
	parser.add_argument('-modelfile_umap', '--modelfile_umap', dest='modelfile_umap', required=True, type=str, action='store',help='UMAP model filename (.h5)')
	parser.add_argument('-outfile_umap_unsupervised', '--outfile_umap_unsupervised', dest='outfile_umap_unsupervised', required=False, type=str, default='latent_data_umap_unsupervised.dat', action='store',help='Name of UMAP encoded data output file')
	
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
	datalist= args.datalist

	# - Data process options	
	nx= args.nx
	ny= args.ny
	
	# - Autoencoder options
	modelfile= args.modelfile
	
	# - UMAP options
	run_umap= args.run_umap
  modelfile_umap= args.modelfile_umap
	outfile_umap_unsupervised= args.outfile_umap_unsupervised
		
	#===========================
	#==   READ DATALIST
	#===========================
	# - Create data loader
	dl= DataLoader(filename=datalist)

	# - Read datalist	
	logger.info("Reading datalist %s ..." % datalist)
	if dl.read_datalist()<0:
		logger.error("Failed to read input datalist!")
		return 1
	

	#===============================
	#==   RUN AUTOENCODER PREDICT
	#===============================
	logger.info("Running autoencoder classifier predict ...")
	vae_class= VAEClassifier(dl)
	vae_class.set_image_size(nx, ny)
	
	if vae_class.predict_model(modelfile)<0:
		logger.error("VAE predict failed!")
		return 1


	#===========================
	#==   RUN UMAP PREDICT
	#===========================
	if run_umap:
		# - Retrieve VAE encoded data
		logger.info("Retrieve latent data from VAE ...")
		snames= vae_class.source_names
		classids= vae_class.source_ids
		vae_data= vae_class.encoded_data

		# - Run UMAP
		logger.info("Running UMAP classifier prediction on VAE latent data ...")
		umap_class= UMAPClassifier()
		umap_class.set_encoded_data_unsupervised_outfile(outfile_umap_unsupervised)
		
		if umap_class.run_predict(vae_data, class_ids=classids, snames=snames, modelfile=modelfile_umap)<0:
			logger.error("UMAP prediction failed!")
			return 1
	

	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

