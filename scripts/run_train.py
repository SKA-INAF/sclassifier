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
from sclassifier_vae.feature_extractor_ae import FeatExtractorAE
from sclassifier_vae.feature_extractor_umap import FeatExtractorUMAP
from sclassifier_vae.clustering import Clusterer

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
	parser.add_argument('-nx', '--nx', dest='nx', required=False, type=int, default=64, action='store',help='Image resize width in pixels (default=64)')
	parser.add_argument('-ny', '--ny', dest='ny', required=False, type=int, default=64, action='store',help='Image resize height in pixels (default=64)')	
	parser.add_argument('--augment', dest='augment', action='store_true',help='Augment images')	
	parser.set_defaults(augment=False)
	parser.add_argument('-augment_scale_factor', '--augment_scale_factor', dest='augment_scale_factor', required=False, type=int, default=1, action='store',help='Number of times images are augmented. E.g. if 2, nsteps_per_epoch=2*nsamples/batch_size (default=1)')
	
	parser.add_argument('--normalize', dest='normalize', action='store_true',help='Normalize input images in range [0,1]')	
	parser.set_defaults(normalize=False)

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
	parser.add_argument('-latentdim', '--latentdim', dest='latentdim', required=False, type=int, default=2, action='store',help='Dimension of latent vector (default=2)')	
	parser.add_argument('-nepochs', '--nepochs', dest='nepochs', required=False, type=int, default=100, action='store',help='Number of epochs used in network training (default=100)')	
	parser.add_argument('-optimizer', '--optimizer', dest='optimizer', required=False, type=str, default='rmsprop', action='store',help='Optimizer used (default=rmsprop)')
	parser.add_argument('-learning_rate', '--learning_rate', dest='learning_rate', required=False, type=float, default=None, action='store',help='Learning rate. If None, use default for the selected optimizer (default=None)')
	parser.add_argument('-batch_size', '--batch_size', dest='batch_size', required=False, type=int, default=32, action='store',help='Batch size used in training (default=32)')
	parser.add_argument('-weight_seed', '--weight_seed', dest='weight_seed', required=False, type=int, default=None, action='store',help='Weight seed to set reproducible training (default=None)')
	parser.add_argument('--reproducible', dest='reproducible', action='store_true',help='Fix seed and make model reproducible from run to run')	
	parser.set_defaults(reproducible=False)
	parser.add_argument('-validation_steps', '--validation_steps', dest='validation_steps', required=False, type=int, default=10, action='store',help='Number of validation steps used in each epoch (default=10)')

	# - Network architecture options
	parser.add_argument('--use_vae', dest='use_vae', action='store_true',help='Use variational autoencoders')	
	parser.set_defaults(use_vae=False)
	parser.add_argument('--add_maxpooling_layer', dest='add_maxpooling_layer', action='store_true',help='Add max pooling layer after conv layers ')	
	parser.set_defaults(add_maxpooling_layer=False)	
	parser.add_argument('--add_batchnorm_layer', dest='add_batchnorm_layer', action='store_true',help='Add batch normalization layer after conv layers ')	
	parser.set_defaults(add_batchnorm_layer=False)	
	parser.add_argument('--add_leakyrelu', dest='add_leakyrelu', action='store_true',help='Add LeakyRELU after batch norm layers ')	
	parser.set_defaults(add_leakyrelu=False)

	parser.add_argument('--add_dense_layer', dest='add_dense_layer', action='store_true',help='Add dense layers in encoder after flattening layers ')	
	parser.set_defaults(add_dense_layer=False)

	parser.add_argument('-nfilters_cnn', '--nfilters_cnn', dest='nfilters_cnn', required=False, type=str, default='32,64,128', action='store',help='Number of convolution filters per each layer')
	parser.add_argument('-kernsizes_cnn', '--kernsizes_cnn', dest='kernsizes_cnn', required=False, type=str, default='3,5,7', action='store',help='Convolution filter kernel sizes per each layer')
	parser.add_argument('-strides_cnn', '--strides_cnn', dest='strides_cnn', required=False, type=str, default='2,2,2', action='store',help='Convolution strides per each layer')
	
	parser.add_argument('-dense_layer_sizes', '--dense_layer_sizes', dest='dense_layer_sizes', required=False, type=str, default='16', action='store',help='Dense layer sizes used (default=16)')
	parser.add_argument('-dense_layer_activation', '--dense_layer_activation', dest='dense_layer_activation', required=False, type=str, default='relu', action='store',help='Dense layer activation used {relu,softmax} (default=relu)')
	parser.add_argument('-decoder_output_layer_activation', '--decoder_output_layer_activation', dest='decoder_output_layer_activation', required=False, type=str, default='sigmoid', action='store',help='Output decoder layer activation used {sigmoid,softmax} (default=sigmoid)')

	parser.add_argument('--mse_loss', dest='mse_loss', action='store_true',help='Compute and include MSE reco loss in total loss')
	parser.add_argument('--no-mse_loss', dest='mse_loss', action='store_false',help='Skip MSE calculation and exclude MSE reco loss from total loss')
	parser.set_defaults(mse_loss=True)
	
	parser.add_argument('--ssim_loss', dest='ssim_loss', action='store_true',help='Compute and include SSIM reco loss in total loss')
	parser.add_argument('--no-ssim_loss', dest='ssim_loss', action='store_false',help='Skip SSIM calculation and exclude SSIM reco loss from total loss')
	parser.set_defaults(ssim_loss=False)

	parser.add_argument('--kl_loss', dest='kl_loss', action='store_true',help='Compute and include KL reco loss in total loss (effective only for VAE model)')
	parser.add_argument('--no-kl_loss', dest='kl_loss', action='store_false',help='Skip KL calculation and exclude KL reco loss from total loss')
	parser.set_defaults(kl_loss=False)	

	parser.add_argument('-mse_loss_weight', '--mse_loss_weight', dest='mse_loss_weight', required=False, type=float, default=1.0, action='store',help='Reconstruction loss weight (default=1.0)')
	parser.add_argument('-kl_loss_weight', '--kl_loss_weight', dest='kl_loss_weight', required=False, type=float, default=1.0, action='store',help='KL loss weight (default=1.0)')
	parser.add_argument('-ssim_loss_weight', '--ssim_loss_weight', dest='ssim_loss_weight', required=False, type=float, default=1.0, action='store',help='SSIM loss weight (default=1.0)')
	parser.add_argument('-ssim_win_size', '--ssim_win_size', dest='ssim_win_size', required=False, type=int, default=3, action='store',help='SSIM filter window size in pixels (default=3)')
	
	# - UMAP classifier options
	parser.add_argument('--run_umap', dest='run_umap', action='store_true',help='Run UMAP of VAE latent vector')	
	parser.set_defaults(run_umap=False)	
	parser.add_argument('-latentdim_umap', '--latentdim_umap', dest='latentdim_umap', required=False, type=int, default=2, action='store',help='Encoded data dim in UMAP (default=2)')
	parser.add_argument('-mindist_umap', '--mindist_umap', dest='mindist_umap', required=False, type=float, default=0.1, action='store',help='Min dist UMAP par (default=0.1)')
	parser.add_argument('-nneighbors_umap', '--nneighbors_umap', dest='nneighbors_umap', required=False, type=int, default=15, action='store',help='N neighbors UMAP par (default=15)')
	parser.add_argument('-outfile_umap_unsupervised', '--outfile_umap_unsupervised', dest='outfile_umap_unsupervised', required=False, type=str, default='latent_data_umap_unsupervised.dat', action='store',help='Name of UMAP encoded data output file')
	parser.add_argument('-outfile_umap_supervised', '--outfile_umap_supervised', dest='outfile_umap_supervised', required=False, type=str, default='latent_data_umap_supervised.dat', action='store',help='Name of UMAP output file with encoded data produced using supervised method')
	parser.add_argument('-outfile_umap_preclassified', '--outfile_umap_preclassified', dest='outfile_umap_preclassified', required=False, type=str, default='latent_data_umap_preclass.dat', action='store',help='Name of UMAP output file with encoded data produced from pre-classified data')

	# - Clustering options
	parser.add_argument('--run_clustering', dest='run_clustering', action='store_true',help='Run clustering on autoencoder latent vector')	
	parser.set_defaults(run_clustering=False)
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
	datalist= args.datalist

	# - Data process options	
	nx= args.nx
	ny= args.ny
	augment= args.augment
	augment_scale_factor= args.augment_scale_factor
	scale= args.scale
	scale_factors= []
	if args.scale_factors!="":
		scale_factors= [float(x.strip()) for x in args.scale_factors.split(',')]

	normalize= args.normalize
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

	# - NN architecture
	use_vae= args.use_vae
	add_maxpooling_layer= args.add_maxpooling_layer
	add_batchnorm_layer= args.add_batchnorm_layer
	add_leakyrelu= args.add_leakyrelu
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
	kl_loss= args.kl_loss
	ssim_loss= args.ssim_loss
	mse_loss_weight= args.mse_loss_weight
	kl_loss_weight= args.kl_loss_weight
	ssim_loss_weight= args.ssim_loss_weight
	ssim_win_size= args.ssim_win_size
	weight_seed= args.weight_seed
	reproducible= args.reproducible
	validation_steps= args.validation_steps

	# - UMAP options
	run_umap= args.run_umap
	latentdim_umap= args.latentdim_umap
	mindist_umap= args.mindist_umap
	nneighbors_umap= args.nneighbors_umap
	outfile_umap_unsupervised= args.outfile_umap_unsupervised
	outfile_umap_supervised= args.outfile_umap_supervised
	outfile_umap_preclassified= args.outfile_umap_preclassified
		
	# - Clustering options
	run_clustering= args.run_clustering
	min_cluster_size= args.min_cluster_size
	min_samples= args.min_samples	
	modelfile_clust= args.modelfile_clust
	predict_clust= args.predict_clust

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
	vae_class= FeatExtractorAE(dl)

	vae_class.use_vae= use_vae
	vae_class.latent_dim= latentdim
	vae_class.set_image_size(nx, ny)
	vae_class.augmentation= augment
	vae_class.augment_scale_factor= augment_scale_factor
	vae_class.normalize= normalize	
	vae_class.log_transform_img= log_transform
	vae_class.scale_img= scale
	vae_class.scale_img_factors= scale_factors
	vae_class.standardize_img= standardize
	vae_class.img_means= img_means
	vae_class.img_sigmas= img_sigmas
	vae_class.chan_divide= chan_divide
	vae_class.chan_mins= chan_mins
	vae_class.erode= erode
	vae_class.erode_kernel= erode_kernel

	vae_class.batch_size= batch_size
	vae_class.nepochs= nepochs
	vae_class.validation_steps= validation_steps
	vae_class.set_optimizer(optimizer, learning_rate)
	if reproducible:
		vae_class.set_reproducible_model()
	
	vae_class.add_max_pooling= add_maxpooling_layer
	vae_class.add_batchnorm= add_batchnorm_layer
	vae_class.add_leakyrelu= add_leakyrelu
	vae_class.add_dense= add_dense_layer
	vae_class.nfilters_cnn= nfilters_cnn
	vae_class.kernsizes_cnn= kernsizes_cnn
	vae_class.strides_cnn= strides_cnn
	vae_class.dense_layer_sizes= dense_layer_sizes
	vae_class.dense_layer_activation= dense_layer_activation

	vae_class.use_mse_loss= mse_loss
	vae_class.use_kl_loss= kl_loss
	vae_class.use_ssim_loss= ssim_loss
	vae_class.mse_loss_weight= mse_loss_weight
	vae_class.kl_loss_weight= kl_loss_weight
	vae_class.ssim_loss_weight= ssim_loss_weight
	vae_class.ssim_win_size= ssim_win_size
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
		umap_class= FeatExtractorUMAP()

		umap_class.set_encoded_data_unsupervised_outfile(outfile_umap_unsupervised)
		umap_class.set_encoded_data_supervised_outfile(outfile_umap_supervised)
		umap_class.set_encoded_data_preclassified_outfile(outfile_umap_preclassified)
		umap_class.set_encoded_data_dim(latentdim_umap)
		umap_class.set_min_dist(mindist_umap)
		umap_class.set_n_neighbors(nneighbors_umap)

		if umap_class.run_train(vae_data, class_ids=classids, snames=snames)<0:
			logger.error("UMAP training failed!")
			return 1

	#==============================
	#==   RUN CLUSTERING
	#==============================
	if run_clustering:
		# - Retrieve VAE encoded data
		logger.info("Retrieve latent data from VAE ...")
		snames= vae_class.source_names
		classids= vae_class.source_ids
		vae_data= vae_class.encoded_data

		# - Run HDBSCAN clustering
		logger.info("Running HDBSCAN classifier prediction on autoencoder latent data ...")
		clust_class= Clusterer()
		clust_class.min_cluster_size= min_cluster_size
		clust_class.min_samples= min_samples
	
		status= 0
		if predict_clust:
			if clust_class.run_predict(vae_data, class_ids=classids, snames=snames, modelfile=modelfile_clust)<0:
				logger.error("Clustering predict failed!")
				return 1
		else:
			if clust_class.run_clustering(vae_data, class_ids=classids, snames=snames, modelfile=modelfile_clust)<0:
				logger.error("Clustering run failed!")
				return 1
	

	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

