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
from sclassifier_vae.data_provider import DataProvider
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
	parser.add_argument('-filelists','--filelists', dest='filelists', required=True, nargs='+', type=str, default=[], help='List of image filelists') 
	parser.add_argument('-catalog_file','--catalog_file', dest='catalog_file', required=False, type=str, default='', help='Caesar source catalog ascii file') 

	# - Data process options
	parser.add_argument('--crop_img', dest='crop_img', action='store_true',help='Crop input images')	
	parser.set_defaults(crop_img=False)	
	parser.add_argument('-nx', '--nx', dest='nx', required=False, type=int, default=51, action='store',help='Image crop width in pixels (default=51)')
	parser.add_argument('-ny', '--ny', dest='ny', required=False, type=int, default=51, action='store',help='Image crop height in pixels (default=51)')	

	parser.add_argument('--normalize_img', dest='normalize_img', action='store_true',help='Normalize input images in range [0,1]')	
	parser.set_defaults(normalize_inputs=False)
	parser.add_argument('-normdatamin', '--normdatamin', dest='normdatamin', required=False, type=float, default=-0.0100, action='store',help='Normalization min used to scale data in [0,1] range (default=-100 mJy/beam)')	
	parser.add_argument('-normdatamax', '--normdatamax', dest='normdatamax', required=False, type=float, default=10, action='store',help='Normalization max used to scale data in [0,1] range (default=10 Jy/beam)')
	
	parser.add_argument('--normalize_img_to_first_chan', dest='normalize_img_to_first_chan', action='store_true',help='Normalize input images to first channel')	
	parser.set_defaults(normalize_img_to_first_chan=False)
	parser.add_argument('--normalize_img_to_chanmax', dest='normalize_img_to_chanmax', action='store_true',help='Normalize input images to channel maximum')	
	parser.set_defaults(normalize_img_to_chanmax=False)

	parser.add_argument('--apply_weights', dest='apply_weights', action='store_true',help='Apply weights to input image channels')	
	parser.set_defaults(apply_weights=False)	
	parser.add_argument('-img_weights','--img_weights', dest='img_weights', required=False, nargs='+', type=float, default=[], help='List of image weights (must have same size of input filelists)') 

	# - NN architecture	
	parser.add_argument('-nnarcfile', '--nnarcfile', dest='nnarcfile', required=False, type=str,action='store',help='Name of file with NN architecture')	
	
	
	# - Network training options
	parser.add_argument('-nepochs', '--nepochs', dest='nepochs', required=False, type=int, default=100, action='store',help='Number of epochs used in network training (default=100)')	
	parser.add_argument('-optimizer', '--optimizer', dest='optimizer', required=False, type=str, default='rmsprop', action='store',help='Optimizer used (default=rmsprop)')
	parser.add_argument('-learning_rate', '--learning_rate', dest='learning_rate', required=False, type=float, default=1.e-4, action='store',help='Learning rate (default=1.e-4)')
	parser.add_argument('-batch_size', '--batch_size', dest='batch_size', required=False, type=int, default=32, action='store',help='Batch size used in training (default=32)')
	parser.add_argument('-intermediate_layer_size', '--intermediate_layer_size', dest='intermediate_layer_size', required=False, type=int, default=512, action='store',help='Intermediate dense layer size used in shallow network (default=512)')
	parser.add_argument('-n_intermediate_layers', '--n_intermediate_layers', dest='n_intermediate_layers', required=False, type=int, default=1, action='store',help='Number of intermediate dense layers used in shallow network (default=1)')
	parser.add_argument('-intermediate_layer_size_factor', '--intermediate_layer_size_factor', dest='intermediate_layer_size_factor', required=False, type=float, default=1, action='store',help='Reduction factor used to compute number of neurons in dense layers (default=1)')
	
	
	# - Output options
	#parser.add_argument('-outfile_loss', '--outfile_loss', dest='outfile_loss', required=False, type=str, default='nn_loss.png', action='store',help='Name of NN loss plot file (default=nn_loss.png)')
	#parser.add_argument('-outfile_accuracy', '--outfile_accuracy', dest='outfile_accuracy', required=False, type=str, default='nn_accuracy.png', action='store',help='Name of NN accuracy plot file (default=nn_accuracy.png)')
	#parser.add_argument('-outfile_model', '--outfile_model', dest='outfile_model', required=False, type=str, default='nn_model.png', action='store',help='Name of NN model plot file (default=nn_model.png)')
	#parser.add_argument('-outfile_nnout_train', '--outfile_nnout_train', dest='outfile_nnout_train', required=False, type=str, default='train_nnout.dat', action='store',help='Name of output file with NN output for train data (default=train_nnout.dat)')
	#parser.add_argument('-outfile_nnout_test', '--outfile_nnout_test', dest='outfile_nnout_test', required=False, type=str, default='test_nnout.dat', action='store',help='Name of output file with NN output for test data (default=test_nnout.dat)')
	#parser.add_argument('-outfile_nnout_metrics', '--outfile_nnout_metrics', dest='outfile_nnout_metrics', required=False, type=str, default='nnout_metrics.dat', action='store',help='Name of output file with NN train metrics (default=nnout_metrics.dat)')


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
	filelists= args.filelists
	catalog_file= args.catalog_file
	print(filelists)

	# - Data process options	
	crop_img= args.crop_img
	nx= args.nx
	ny= args.ny
	normalize_img= args.normalize_img
	normdatamin= args.normdatamin
	normdatamax= args.normdatamax
	normalize_img_to_first_chan= args.normalize_img_to_first_chan
	normalize_img_to_chanmax= args.normalize_img_to_chanmax
	apply_weights= args.apply_weights
	img_weights= args.img_weights

	# - NN architecture
	nnarcfile= args.nnarcfile
	intermediate_layer_size= args.intermediate_layer_size
	n_intermediate_layers= args.n_intermediate_layers
	intermediate_layer_size_factor= args.intermediate_layer_size_factor

	# - Train options
	optimizer= args.optimizer
	learning_rate= args.learning_rate
	batch_size= args.batch_size
	nepochs= args.nepochs
	
	
	# - Output file
	#outfile_loss= args.outfile_loss
	#outfile_accuracy= args.outfile_accuracy
	#outfile_model= args.outfile_model
	#outfile_nnout_train= args.outfile_nnout_train
	#outfile_nnout_test= args.outfile_nnout_test
	#outfile_nnout_metrics= args.outfile_nnout_metrics
	
	#===========================
	#==   CHECK ARGS
	#===========================
	if apply_weights and len(img_weights)!=len(filelists):
		logger.error("Input image weights has size different from input filelists!")
		return 1

	#===========================
	#==   READ DATA
	#===========================
	# - Create data provider
	dp= DataProvider(filelists=filelists)

	# - Set options
	dp.set_catalog_filename(catalog_file)
	dp.enable_inputs_normalization(normalize_img)
	dp.set_input_data_norm_range(normdatamin,normdatamax)
	dp.enable_inputs_normalization_to_first_channel(normalize_img_to_first_chan)
	dp.enable_inputs_normalization_to_chanmax(normalize_img_to_chanmax)
	dp.enable_img_crop(crop_img)
	dp.set_img_crop_size(nx,ny)
	dp.enable_img_weights(apply_weights)
	dp.set_img_weights(img_weights)


	# - Read data	
	logger.info("Running data provider to read image data ...")
	status= dp.read_data()
	if status<0:
		logger.error("Failed to read input image data!")
		return 1
	

	#===========================
	#==   TRAIN NN
	#===========================
	logger.info("Running VAE classifier training ...")
	nn= VAEClassifier(dp)

	nn.set_optimizer(optimizer)
	nn.set_learning_rate(learning_rate)	
	nn.set_batch_size(batch_size)
	nn.set_nepochs(nepochs)

	nn.set_intermediate_layer_size(intermediate_layer_size)
	nn.set_n_intermediate_layers(n_intermediate_layers)
	nn.set_intermediate_layer_size_factor(intermediate_layer_size_factor)

	#nn.set_outfile_loss(outfile_loss)
	#nn.set_outfile_accuracy(outfile_accuracy)	
	#nn.set_outfile_model(outfile_model)
	#nn.set_outfile_nnout_train(outfile_nnout_train)
	#nn.set_outfile_nnout_test(outfile_nnout_test)
	#nn.set_outfile_nnout_metrics(outfile_nnout_metrics)

	status= nn.train_model()
	if status<0:
		logger.error("NN training failed!")
		return 1


	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

