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
	parser.add_argument('--augment', dest='augment', action='store_true',help='Augment images')	
	parser.set_defaults(augment=False)

	# - Network training options
	parser.add_argument('-latentdim', '--latentdim', dest='latentdim', required=False, type=int, default=2, action='store',help='Dimension of latent vector (default=2)')	
	parser.add_argument('-nepochs', '--nepochs', dest='nepochs', required=False, type=int, default=100, action='store',help='Number of epochs used in network training (default=100)')	
	parser.add_argument('-optimizer', '--optimizer', dest='optimizer', required=False, type=str, default='rmsprop', action='store',help='Optimizer used (default=rmsprop)')
	parser.add_argument('-learning_rate', '--learning_rate', dest='learning_rate', required=False, type=float, default=None, action='store',help='Learning rate. If None, use default for the selected optimizer (default=None)')
	parser.add_argument('-batch_size', '--batch_size', dest='batch_size', required=False, type=int, default=32, action='store',help='Batch size used in training (default=32)')
	parser.add_argument('-weight_seed', '--weight_seed', dest='weight_seed', required=False, type=int, default=None, action='store',help='Weight seed to set reproducible training (default=None)')
	parser.add_argument('--reproducible', dest='reproducible', action='store_true',help='Fix seed and make model reproducible from run to run')	
	parser.set_defaults(reproducible=False)

		

	# - Network architecture options
	parser.add_argument('--use_vae', dest='use_vae', action='store_true',help='Use variational autoencoders')	
	parser.set_defaults(use_vae=False)
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
	
	# - UMAP classifier options
	parser.add_argument('--run_umap', dest='run_umap', action='store_true',help='Run UMAP of VAE latent vector')	
	parser.set_defaults(run_umap=False)	
	parser.add_argument('-latentdim_umap', '--latentdim_umap', dest='latentdim_umap', required=False, type=int, default=2, action='store',help='Encoded data dim in UMAP (default=2)')
	parser.add_argument('-mindist_umap', '--mindist_umap', dest='mindist_umap', required=False, type=float, default=0.1, action='store',help='Min dist UMAP par (default=0.1)')
	parser.add_argument('-nneighbors_umap', '--nneighbors_umap', dest='nneighbors_umap', required=False, type=int, default=15, action='store',help='N neighbors UMAP par (default=15)')
	parser.add_argument('-outfile_umap_unsupervised', '--outfile_umap_unsupervised', dest='outfile_umap_unsupervised', required=False, type=str, default='latent_data_umap_unsupervised.dat', action='store',help='Name of UMAP encoded data output file')
	parser.add_argument('-outfile_umap_supervised', '--outfile_umap_supervised', dest='outfile_umap_supervised', required=False, type=str, default='latent_data_umap_supervised.dat', action='store',help='Name of UMAP output file with encoded data produced using supervised method')
	parser.add_argument('-outfile_umap_preclassified', '--outfile_umap_preclassified', dest='outfile_umap_preclassified', required=False, type=str, default='latent_data_umap_preclass.dat', action='store',help='Name of UMAP output file with encoded data produced from pre-classified data')


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
	use_vae= args.use_vae
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
	latentdim= args.latentdim
	optimizer= args.optimizer
	learning_rate= args.learning_rate
	batch_size= args.batch_size
	nepochs= args.nepochs
	mse_loss= args.mse_loss
	rec_loss_weight= args.rec_loss_weight
	kl_loss_weight= args.kl_loss_weight
	weight_seed= args.weight_seed
	reproducible= args.reproducible

	# - UMAP options
	run_umap= args.run_umap
	latentdim_umap= args.latentdim_umap
	mindist_umap= args.mindist_umap
	nneighbors_umap= args.nneighbors_umap
	outfile_umap_unsupervised= args.outfile_umap_unsupervised
	outfile_umap_supervised= args.outfile_umap_supervised
	outfile_umap_preclassified= args.outfile_umap_preclassified
		
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
	#==   TRAIN VAE
	#===========================
	logger.info("Running VAE classifier training ...")
	vae_class= VAEClassifier(dl)

	vae_class.use_vae= use_vae
	vae_class.latent_dim= latentdim
	vae_class.set_image_size(nx, ny)
	vae_class.augmentation= augment
	vae_class.batch_size= batch_size
	vae_class.nepochs= nepochs
	vae_class.set_optimizer(optimizer, learning_rate)
	if reproducible:
		vae_class.set_reproducible_model()
	
	vae_class.add_max_pooling= add_maxpooling_layer
	vae_class.add_batchnorm= add_batchnorm_layer
	vae_class.add_dense= add_dense_layer
	vae_class.nfilters_cnn= nfilters_cnn
	vae_class.kernsizes_cnn= kernsizes_cnn
	vae_class.strides_cnn= strides_cnn
	vae_class.dense_layer_sizes= dense_layer_sizes
	vae_class.dense_layer_activation= dense_layer_activation

	vae_class.use_mse_loss= mse_loss
	vae_class.rec_loss_weight= rec_loss_weight
	vae_class.kl_loss_weight= kl_loss_weight
	vae_class.weight_seed= weight_seed

	if vae_class.train_model()<0:
		logger.error("VAE training failed!")
		return 1


	#===========================
	#==   TRAIN UMAP
	#===========================
	if run_umap:
		# - Retrieve VAE encoded data
		logger.info("Retrieve latent data from VAE ...")
		snames= vae_class.source_names
		classids= vae_class.source_ids
		vae_data= vae_class.encoded_data

		# - Run UMAP	
		logger.info("Running UMAP classifier training on VAE latent data ...")
		umap_class= UMAPClassifier()

		umap_class.set_encoded_data_unsupervised_outfile(outfile_umap_unsupervised)
		umap_class.set_encoded_data_supervised_outfile(outfile_umap_supervised)
		umap_class.set_encoded_data_preclassified_outfile(outfile_umap_preclassified)
		umap_class.set_encoded_data_dim(latentdim_umap)
		umap_class.set_min_dist(mindist_umap)
		umap_class.set_n_neighbors(nneighbors_umap)

		if umap_class.run_train(vae_data, class_ids=classids, snames=snames)<0:
			logger.error("UMAP training failed!")
			return 1
	

	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

