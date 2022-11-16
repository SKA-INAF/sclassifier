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
from sclassifier.feature_extractor_ae import FeatExtractorAE
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

	# - Autoencoder model options
	parser.add_argument('-modelfile_encoder', '--modelfile_encoder', dest='modelfile_encoder', required=True, type=str, action='store',help='Encoder model architecture filename (.json)')
	parser.add_argument('-weightfile_encoder', '--weightfile_encoder', dest='weightfile_encoder', required=True, type=str, action='store',help='Encoder model weights filename (.h5)')
	#parser.add_argument('--add_channorm_layer', dest='add_channorm_layer', action='store_true',help='Add norm layer before encoder input and denorm layer before decoder output')	
	#parser.set_defaults(add_channorm_layer=False)

	# - UMAP classifier options
	parser.add_argument('--run_umap', dest='run_umap', action='store_true',help='Run UMAP on autoencoder latent vector')	
	parser.set_defaults(run_umap=False)
	parser.add_argument('-modelfile_umap', '--modelfile_umap', dest='modelfile_umap', required=False, type=str, action='store',help='UMAP model filename (.h5)')
	parser.add_argument('-outfile_umap_unsupervised', '--outfile_umap_unsupervised', dest='outfile_umap_unsupervised', required=False, type=str, default='latent_data_umap_unsupervised.dat', action='store',help='Name of UMAP encoded data output file')

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

	normalize= args.normalize
	scale_to_abs_max= args.scale_to_abs_max
	scale_to_max= args.scale_to_max
	log_transform= args.log_transform
	scale= args.scale
	scale_factors= []
	if args.scale_factors!="":
		scale_factors= [float(x.strip()) for x in args.scale_factors.split(',')]
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
	
	# - Autoencoder options
	modelfile_encoder= args.modelfile_encoder
	weightfile_encoder= args.weightfile_encoder
	#add_channorm_layer= args.add_channorm_layer

	# - UMAP options
	run_umap= args.run_umap
	modelfile_umap= args.modelfile_umap
	outfile_umap_unsupervised= args.outfile_umap_unsupervised
		
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
	
	#===============================
	#==   RUN AUTOENCODER PREDICT
	#===============================
	vae_class= FeatExtractorAE(dl)
	vae_class.set_image_size(nx, ny)
	vae_class.normalize= normalize
	vae_class.scale_to_abs_max= scale_to_abs_max
	vae_class.scale_to_max= scale_to_max
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
	#vae_class.add_channorm_layer= add_channorm_layer
	
	logger.info("Running autoencoder predict ...")
	if vae_class.predict_model(modelfile_encoder, weightfile_encoder)<0:
		logger.error("VAE predict failed!")
		return 1

	#===========================
	#==   RUN UMAP PREDICT
	#===========================
	if run_umap:
		# - Retrieve VAE encoded data
		logger.info("Retrieve latent data from autoencoder ...")
		snames= vae_class.source_names
		classids= vae_class.source_ids
		vae_data= vae_class.encoded_data

		# - Run UMAP
		logger.info("Running UMAP classifier prediction on autoencoder latent data ...")
		umap_class= FeatExtractorUMAP()
		umap_class.set_encoded_data_unsupervised_outfile(outfile_umap_unsupervised)
		
		if umap_class.run_predict(vae_data, class_ids=classids, snames=snames, modelfile=modelfile_umap)<0:
			logger.error("UMAP prediction failed!")
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

