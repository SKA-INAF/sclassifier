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
	
	# - Network training options
	parser.add_argument('-nepochs', '--nepochs', dest='nepochs', required=False, type=int, default=100, action='store',help='Number of epochs used in network training (default=100)')	
	parser.add_argument('-optimizer', '--optimizer', dest='optimizer', required=False, type=str, default='rmsprop', action='store',help='Optimizer used (default=rmsprop)')
	parser.add_argument('-learning_rate', '--learning_rate', dest='learning_rate', required=False, type=float, default=1.e-4, action='store',help='Learning rate (default=1.e-4)')
	parser.add_argument('-batch_size', '--batch_size', dest='batch_size', required=False, type=int, default=32, action='store',help='Batch size used in training (default=32)')
	
	# - Network architecture options
	parser.add_argument('--add_maxpooling_layer', dest='add_maxpooling_layer', action='store_true',help='Add max pooling layer after conv layers ')	
	parser.set_defaults(add_maxpooling_layer=False)	
	parser.add_argument('-nfilters_cnn', '--nfilters_cnn', dest='nfilters_cnn', required=False, type=str, default='32,64,128', action='store',help='Number of convolution filters per each layer')
	parser.add_argument('-kernsizes_cnn', '--kernsizes_cnn', dest='kernsizes_cnn', required=False, type=str, default='3,5,7', action='store',help='Convolution filter kernel sizes per each layer')
	parser.add_argument('-strides_cnn', '--strides_cnn', dest='strides_cnn', required=False, type=str, default='2,2,2', action='store',help='Convolution strides per each layer')
	
	parser.add_argument('-intermediate_layer_size', '--intermediate_layer_size', dest='intermediate_layer_size', required=False, type=int, default=512, action='store',help='Intermediate dense layer size used in shallow network (default=512)')
	parser.add_argument('-n_intermediate_layers', '--n_intermediate_layers', dest='n_intermediate_layers', required=False, type=int, default=1, action='store',help='Number of intermediate dense layers used in shallow network (default=1)')
	parser.add_argument('-intermediate_layer_size_factor', '--intermediate_layer_size_factor', dest='intermediate_layer_size_factor', required=False, type=float, default=1, action='store',help='Reduction factor used to compute number of neurons in dense layers (default=1)')
	
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

	# - NN architecture
	add_maxpooling_layer= args.add_maxpooling_layer
	nfilters_cnn= [int(x.strip()) for x in args.nfilters_cnn.split(',')]
	kernsizes_cnn= [int(x.strip()) for x in args.kernsizes_cnn.split(',')]	
	strides_cnn= [int(x.strip()) for x in args.strides_cnn.split(',')]

	print("nfilters_cnn")
	print(nfilters_cnn)
	print("kernsizes_cnn")
	print(kernsizes_cnn)
	print("strides_cnn")
	print(strides_cnn)

	intermediate_layer_size= args.intermediate_layer_size
	n_intermediate_layers= args.n_intermediate_layers
	intermediate_layer_size_factor= args.intermediate_layer_size_factor

	# - Train options
	optimizer= args.optimizer
	learning_rate= args.learning_rate
	batch_size= args.batch_size
	nepochs= args.nepochs
	

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
	

	#===========================
	#==   TRAIN NN
	#===========================
	logger.info("Running VAE classifier training ...")
	nn= VAEClassifier(dl)

	nn.set_optimizer(optimizer)
	nn.set_learning_rate(learning_rate)	
	nn.set_batch_size(batch_size)
	nn.set_nepochs(nepochs)

	nn.add_max_pooling= add_maxpooling_layer
	nn.nfilters_cnn= nfilters_cnn
	nn.kernsizes_cnn= kernsizes_cnn
	nn.strides_cnn= strides_cnn
	nn.set_intermediate_layer_size(intermediate_layer_size)
	nn.set_n_intermediate_layers(n_intermediate_layers)
	nn.set_intermediate_layer_size_factor(intermediate_layer_size_factor)

	if nn.train_model()<0:
		logger.error("NN training failed!")
		return 1


	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

