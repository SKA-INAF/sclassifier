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
	parser.add_argument('-modelfile_encoder', '--modelfile_encoder', dest='modelfile_encoder', required=True, type=str, action='store',help='Encoder model architecture filename (.json)')
	parser.add_argument('-weightfile_encoder', '--weightfile_encoder', dest='weightfile_encoder', required=True, type=str, action='store',help='Encoder model weights filename (.h5)')
	parser.add_argument('-modelfile_decoder', '--modelfile_decoder', dest='modelfile_decoder', required=True, type=str, action='store',help='Decoder model architecture filename (.json)')
	parser.add_argument('-weightfile_decoder', '--weightfile_decoder', dest='weightfile_decoder', required=True, type=str, action='store',help='Decoder model weights filename (.h5)')

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
	modelfile_encoder= args.modelfile_encoder
	modelfile_decoder= args.modelfile_decoder
	weightfile_encoder= args.weightfile_encoder
	weightfile_decoder= args.weightfile_decoder

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
	#==   RUN AUTOENCODER RECO
	#===============================
	logger.info("Running autoencoder classifier reconstruction ...")
	vae_class= VAEClassifier(dl)
	vae_class.set_image_size(nx, ny)
	
	if vae_class.reconstruct_data(modelfile_encoder, weightfile_encoder, modelfile_decoder, weightfile_decoder)<0:
		logger.error("Autoencoder reconstruction failed!")
		return 1

	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

