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
from sclassifier.data_loader import DataLoader
from sclassifier.feature_extractor_simclr import FeatExtractorSimCLR
from sclassifier.feature_extractor_umap import FeatExtractorUMAP
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
	parser.add_argument('-datalist','--datalist', dest='datalist', required=True, type=str, help='Input data json filelist') 
	parser.add_argument('-datalist_cv','--datalist_cv', dest='datalist_cv', required=False, default="", type=str, help='Input data json filelist for validation') 
	
	# - Data pre-processing options
	parser.add_argument('-nx', '--nx', dest='nx', required=False, type=int, default=64, action='store',help='Image resize width in pixels (default=64)')
	parser.add_argument('-ny', '--ny', dest='ny', required=False, type=int, default=64, action='store',help='Image resize height in pixels (default=64)')
	
	parser.add_argument('--normalize', dest='normalize', action='store_true',help='Normalize input images in range [0,1]')	
	parser.set_defaults(normalize=False)
	parser.add_argument('--scale_to_abs_max', dest='scale_to_abs_max', action='store_true',help='In normalization, if scale_to_max is active, scale to global max across all channels')	
	parser.set_defaults(scale_to_abs_max=False)
	parser.add_argument('--scale_to_max', dest='scale_to_max', action='store_true',help='In normalization, scale to max not to min-max range')	
	parser.set_defaults(scale_to_max=False)

	parser.add_argument('--log_transform', dest='log_transform', action='store_true',help='Apply log transform to images')	
	parser.set_defaults(log_transform=False)

	parser.add_argument('--scale', dest='scale', action='store_true',help='Apply scale factors to images')	
	parser.set_defaults(scale=False)
	parser.add_argument('-scale_factors', '--scale_factors', dest='scale_factors', required=False, type=str, default='', action='store',help='Image scale factors separated by commas (default=empty)')

	parser.add_argument('--standardize', dest='standardize', action='store_true',help='Apply standardization to images')	
	parser.set_defaults(standardize=False)
	parser.add_argument('-img_means', '--img_means', dest='img_means', required=False, type=str, default='', action='store',help='Image means (separated by commas) to be used in standardization (default=empty)')
	parser.add_argument('-img_sigmas', '--img_sigmas', dest='img_sigmas', required=False, type=str, default='', action='store',help='Image sigmas (separated by commas) to be used in standardization (default=empty)')

	parser.add_argument('--chan_divide', dest='chan_divide', action='store_true',help='Apply channel division to images')	
	parser.set_defaults(chan_divide=False)
	parser.add_argument('-chan_mins', '--chan_mins', dest='chan_mins', required=False, type=str, default='', action='store',help='Image channel means (separated by commas) to be used in chan divide (default=empty)')

	parser.add_argument('--erode', dest='erode', action='store_true',help='Apply erosion to image sourve mask')	
	parser.set_defaults(erode=False)	
	parser.add_argument('-erode_kernel', '--erode_kernel', dest='erode_kernel', required=False, type=int, default=5, action='store',help='Erosion kernel size in pixels (default=5)')	

	# - Network training options
	parser.add_argument('-nepochs', '--nepochs', dest='nepochs', required=False, type=int, default=100, action='store',help='Number of epochs used in network training (default=100)')	
	parser.add_argument('-optimizer', '--optimizer', dest='optimizer', required=False, type=str, default='rmsprop', action='store',help='Optimizer used (default=rmsprop)')
	parser.add_argument('-learning_rate', '--learning_rate', dest='learning_rate', required=False, type=float, default=None, action='store',help='Learning rate. If None, use default for the selected optimizer (default=None)')
	parser.add_argument('-batch_size', '--batch_size', dest='batch_size', required=False, type=int, default=32, action='store',help='Batch size used in training (default=32)')
	parser.add_argument('-weight_seed', '--weight_seed', dest='weight_seed', required=False, type=int, default=None, action='store',help='Weight seed to set reproducible training (default=None)')
	parser.add_argument('--reproducible', dest='reproducible', action='store_true',help='Fix seed and make model reproducible from run to run')	
	parser.set_defaults(reproducible=False)
	parser.add_argument('-validation_steps', '--validation_steps', dest='validation_steps', required=False, type=int, default=10, action='store',help='Number of validation steps used in each epoch (default=10)')

	# - Network architecture options
	parser.add_argument('-weightfile', '--weightfile', dest='weightfile', required=False, type=str, default="", action='store',help='Weight file (hd5) to be loaded (default=no)')	
	parser.add_argument('-modelfile', '--modelfile', dest='modelfile', required=False, type=str, default="", action='store',help='Model architecture file (json) to be loaded (default=no)')
	parser.add_argument('-weightfile_encoder', '--weightfile_encoder', dest='weightfile_encoder', required=False, type=str, default="", action='store',help='Encoder weight file (hd5) to be loaded (default=no)')	
	parser.add_argument('-modelfile_encoder', '--modelfile_encoder', dest='modelfile_encoder', required=False, type=str, default="", action='store',help='Encoder model architecture file (json) to be loaded (default=no)')
	parser.add_argument('-latentdim', '--latentdim', dest='latentdim', required=False, type=int, default=2, action='store',help='Dimension of latent vector (default=2)')	
	parser.add_argument('--add_maxpooling_layer', dest='add_maxpooling_layer', action='store_true',help='Add max pooling layer after conv layers ')	
	parser.set_defaults(add_maxpooling_layer=False)	
	parser.add_argument('--add_batchnorm_layer', dest='add_batchnorm_layer', action='store_true',help='Add batch normalization layer after conv layers ')	
	parser.set_defaults(add_batchnorm_layer=False)	
	parser.add_argument('--add_leakyrelu', dest='add_leakyrelu', action='store_true',help='Add LeakyRELU after batch norm layers ')	
	parser.set_defaults(add_leakyrelu=False)
	parser.add_argument('--add_dense_layer', dest='add_dense_layer', action='store_true',help='Add dense layers in encoder after flattening layers ')	
	parser.set_defaults(add_dense_layer=False)
	parser.add_argument('--add_dropout_layer', dest='add_dropout_layer', action='store_true',help='Add dropout layers before dense layers')	
	parser.set_defaults(add_dropout_layer=False)
	parser.add_argument('-dropout_rate', '--dropout_rate', dest='dropout_rate', required=False, type=float, default=0.5, action='store',help='Dropout rate (default=0.5)')


	parser.add_argument('-nfilters_cnn', '--nfilters_cnn', dest='nfilters_cnn', required=False, type=str, default='32,64,128', action='store',help='Number of convolution filters per each layer')
	parser.add_argument('-kernsizes_cnn', '--kernsizes_cnn', dest='kernsizes_cnn', required=False, type=str, default='3,5,7', action='store',help='Convolution filter kernel sizes per each layer')
	parser.add_argument('-strides_cnn', '--strides_cnn', dest='strides_cnn', required=False, type=str, default='2,2,2', action='store',help='Convolution strides per each layer')
	
	parser.add_argument('-dense_layer_sizes', '--dense_layer_sizes', dest='dense_layer_sizes', required=False, type=str, default='16', action='store',help='Dense layer sizes used (default=16)')
	parser.add_argument('-dense_layer_activation', '--dense_layer_activation', dest='dense_layer_activation', required=False, type=str, default='relu', action='store',help='Dense layer activation used {relu,softmax} (default=relu)')

	# - Run options
	parser.add_argument('--predict', dest='predict', action='store_true',help='Predict model on input data (default=false)')	
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
	datalist= args.datalist
	datalist_cv= args.datalist_cv
	
	# - Data process options	
	nx= args.nx
	ny= args.ny
	scale= args.scale
	scale_factors= []
	if args.scale_factors!="":
		scale_factors= [float(x.strip()) for x in args.scale_factors.split(',')]

	normalize= args.normalize
	scale_to_abs_max= args.scale_to_abs_max
	scale_to_max= args.scale_to_max
	log_transform= args.log_transform
	standardize= args.standardize
	img_means= []
	img_sigmas= []
	if args.img_means!="":
		img_means= [float(x.strip()) for x in args.img_means.split(',')]
	if args.img_sigmas!="":
		img_sigmas= [float(x.strip()) for x in args.img_sigmas.split(',')]

	chan_divide= args.chan_divide
	chan_mins= []
	if args.chan_mins!="":
		chan_mins= [float(x.strip()) for x in args.chan_mins.split(',')]
	erode= args.erode	
	erode_kernel= args.erode_kernel

	# - Model architecture
	modelfile= args.modelfile
	weightfile= args.weightfile
	modelfile_encoder= args.modelfile_encoder
	weightfile_encoder= args.weightfile_encoder
	latentdim= args.latentdim
	add_maxpooling_layer= args.add_maxpooling_layer
	add_batchnorm_layer= args.add_batchnorm_layer
	add_leakyrelu= args.add_leakyrelu
	add_dense_layer= args.add_dense_layer	
	nfilters_cnn= [int(x.strip()) for x in args.nfilters_cnn.split(',')]
	kernsizes_cnn= [int(x.strip()) for x in args.kernsizes_cnn.split(',')]	
	strides_cnn= [int(x.strip()) for x in args.strides_cnn.split(',')]
	dense_layer_sizes= [int(x.strip()) for x in args.dense_layer_sizes.split(',')]
	dense_layer_activation= args.dense_layer_activation
	add_dropout_layer= args.add_dropout_layer
	dropout_rate= args.dropout_rate
	
	# - Train options
	optimizer= args.optimizer
	learning_rate= args.learning_rate
	batch_size= args.batch_size
	nepochs= args.nepochs
	weight_seed= args.weight_seed
	reproducible= args.reproducible
	validation_steps= args.validation_steps

	# - Run options
	predict= args.predict

	#===========================
	#==   READ DATALIST
	#===========================
	# - Create data loader
	dl= DataLoader(filename=datalist, augmenter_choice='simclr')

	# - Read datalist	
	logger.info("Reading datalist %s ..." % datalist)
	if dl.read_datalist()<0:
		logger.error("Failed to read input datalist!")
		return 1
	
	# - Create data loader for validation
	dl_cv= None
	if datalist_cv!="":
		logger.info("Reading datalist_cv %s ..." % (datalist_cv))
		dl_cv= DataLoader(filename=datalist_cv, augmenter_choice='simclr')
		if dl_cv.read_datalist()<0:
			logger.error("Failed to read input datalist for validation!")
			return 1

	#===========================
	#==   BUILD MODEL
	#===========================
	simclr= FeatExtractorSimCLR(dl)

	simclr.modelfile= modelfile
	simclr.weightfile= weightfile
	simclr.modelfile_encoder= modelfile_encoder
	simclr.weightfile_encoder= weightfile_encoder
	simclr.set_image_size(nx, ny)
	simclr.latent_dim= latentdim

	simclr.normalize= normalize	
	simclr.scale_to_abs_max= scale_to_abs_max
	simclr.scale_to_max= scale_to_max
	simclr.log_transform_img= log_transform
	simclr.scale_img= scale
	simclr.scale_img_factors= scale_factors
	simclr.standardize_img= standardize
	simclr.img_means= img_means
	simclr.img_sigmas= img_sigmas
	simclr.chan_divide= chan_divide
	simclrchan_mins= chan_mins
	simclr.erode= erode
	simclr.erode_kernel= erode_kernel

	simclr.batch_size= batch_size
	simclr.nepochs= nepochs
	simclr.validation_steps= validation_steps
	simclr.set_optimizer(optimizer, learning_rate)
	if reproducible:
		simclr.set_reproducible_model()
	
	simclr.add_max_pooling= add_maxpooling_layer
	simclr.add_batchnorm= add_batchnorm_layer
	simclr.add_leakyrelu= add_leakyrelu
	simclr.add_dense= add_dense_layer
	simclr.nfilters_cnn= nfilters_cnn
	simclr.kernsizes_cnn= kernsizes_cnn
	simclr.strides_cnn= strides_cnn
	simclr.dense_layer_sizes= dense_layer_sizes
	simclr.dense_layer_activation= dense_layer_activation
	simclr.add_dropout_layer= add_dropout_layer
	simclr.dropout_rate= dropout_rate

	# - Run train/predict
	simclr.dl_cv= dl_cv

	if predict:
		status= simclr.run_predict(modelfile, weightfile)
	else:
		status= simclr.run_train(modelfile, weightfile, modelfile_encoder, weightfile_encoder)
	
	if status<0:
		logger.error("SimCLR run failed!")
		return 1
	
	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())
