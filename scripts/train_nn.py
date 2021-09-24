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
	parser.add_argument('--augment', dest='augment', action='store_true',help='Augment images')	
	parser.set_defaults(augment=False)

	# - Network training options
	parser.add_argument('-nepochs', '--nepochs', dest='nepochs', required=False, type=int, default=100, action='store',help='Number of epochs used in network training (default=100)')	
	parser.add_argument('-optimizer', '--optimizer', dest='optimizer', required=False, type=str, default='rmsprop', action='store',help='Optimizer used (default=rmsprop)')
	parser.add_argument('-learning_rate', '--learning_rate', dest='learning_rate', required=False, type=float, default=None, action='store',help='Learning rate. If None, use default for the selected optimizer (default=None)')
	parser.add_argument('-batch_size', '--batch_size', dest='batch_size', required=False, type=int, default=32, action='store',help='Batch size used in training (default=32)')
	
	# - Network architecture options
	parser.add_argument('--add_maxpooling_layer', dest='add_maxpooling_layer', action='store_true',help='Add max pooling layer after conv layers ')	
	parser.set_defaults(add_maxpooling_layer=False)	
	parser.add_argument('--add_batchnorm_layer', dest='add_batchnorm_layer', action='store_true',help='Add batch normalization layer after conv layers ')	
	parser.set_defaults(add_batchnorm_layer=False)	
	parser.add_argument('--add_dense_layer', dest='add_dense_layer', action='store_true',help='Add dense layers in encoder after flattening layers ')	
	parser.set_defaults(add_dense_layer=False)

	parser.add_argument('-nfilters_cnn', '--nfilters_cnn', dest='nfilters_cnn', required=False, type=str, default='32,64,128', action='store',help='Number of convolution filters per each layer')
	parser.add_argument('-kernsizes_cnn', '--kernsizes_cnn', dest='kernsizes_cnn', required=False, type=str, default='3,5,7', action='store',help='Convolution filter kernel sizes per each layer')
	parser.add_argument('-strides_cnn', '--strides_cnn', dest='strides_cnn', required=False, type=str, default='2,2,2', action='store',help='Convolution strides per each layer')
	
	parser.add_argument('-dense_layer_sizes', '--dense_layer_sizes', dest='dense_layer_sizes', required=False, type=str, default='16', action='store',help='Dense layer sizes used (default=16)')
	parser.add_argument('-dense_layer_activation', '--dense_layer_activation', dest='dense_layer_activation', required=False, type=str, default='relu', action='store',help='Dense layer activation used {relu,softmax} (default=relu)')
	parser.add_argument('-decoder_output_layer_activation', '--decoder_output_layer_activation', dest='decoder_output_layer_activation', required=False, type=str, default='sigmoid', action='store',help='Output decoder layer activation used {sigmoid,softmax} (default=sigmoid)')

	parser.add_argument('--mse_loss', dest='mse_loss', action='store_true',help='use MSE loss and not crossentropy as recontruction loss')	
	parser.set_defaults(mse_loss=False)	
	parser.add_argument('-rec_loss_weight', '--rec_loss_weight', dest='rec_loss_weight', required=False, type=float, default=0.5, action='store',help='Reconstruction loss weight (default=0.5)')
	parser.add_argument('-kl_loss_weight', '--kl_loss_weight', dest='kl_loss_weight', required=False, type=float, default=0.5, action='store',help='KL loss weight (default=0.5)')
	
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
	augment= args.augment

	# - NN architecture
	add_maxpooling_layer= args.add_maxpooling_layer
	add_batchnorm_layer= args.add_batchnorm_layer
	add_dense_layer= args.add_dense_layer	
	nfilters_cnn= [int(x.strip()) for x in args.nfilters_cnn.split(',')]
	kernsizes_cnn= [int(x.strip()) for x in args.kernsizes_cnn.split(',')]	
	strides_cnn= [int(x.strip()) for x in args.strides_cnn.split(',')]
	dense_layer_sizes= [int(x.strip()) for x in args.dense_layer_sizes.split(',')]
	dense_layer_activation= args.dense_layer_activation
	decoder_output_layer_activation= args.decoder_output_layer_activation
	
	print("nfilters_cnn")
	print(nfilters_cnn)
	print("kernsizes_cnn")
	print(kernsizes_cnn)
	print("strides_cnn")
	print(strides_cnn)
	print("dense_layer_sizes")
	print(dense_layer_sizes)

	
	# - Train options
	optimizer= args.optimizer
	learning_rate= args.learning_rate
	batch_size= args.batch_size
	nepochs= args.nepochs
	mse_loss= args.mse_loss
	rec_loss_weight= args.rec_loss_weight
	kl_loss_weight= args.kl_loss_weight


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

	nn.set_image_size(nx, ny)
	nn.augmentation= augment
	nn.batch_size= batch_size
	nn.nepochs= nepochs
	nn.set_optimizer(optimizer, learning_rate)
	
	nn.add_max_pooling= add_maxpooling_layer
	nn.add_batchnorm= add_batchnorm_layer
	nn.add_dense= add_dense_layer
	nn.nfilters_cnn= nfilters_cnn
	nn.kernsizes_cnn= kernsizes_cnn
	nn.strides_cnn= strides_cnn
	nn.dense_layer_sizes= dense_layer_sizes
	nn.dense_layer_activation= dense_layer_activation

	nn.use_mse_loss= mse_loss
	nn.rec_loss_weight= rec_loss_weight
	nn.kl_loss_weight= kl_loss_weight

	if nn.train_model()<0:
		logger.error("NN training failed!")
		return 1


	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

