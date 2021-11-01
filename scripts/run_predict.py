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

## CLUSTERING
import hdbscan

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
	parser.add_argument('--run_umap', dest='run_umap', action='store_true',help='Run UMAP on autoencoder latent vector')	
	parser.set_defaults(run_umap=False)
	parser.add_argument('-modelfile_umap', '--modelfile_umap', dest='modelfile_umap', required=True, type=str, action='store',help='UMAP model filename (.h5)')
	parser.add_argument('-outfile_umap_unsupervised', '--outfile_umap_unsupervised', dest='outfile_umap_unsupervised', required=False, type=str, default='latent_data_umap_unsupervised.dat', action='store',help='Name of UMAP encoded data output file')

	# - Clustering options
	parser.add_argument('--run_clustering', dest='run_clustering', action='store_true',help='Run clustering on autoencoder latent vector')	
	parser.set_defaults(run_clustering=False)
	parser.add_argument('-min_cluster_size', '--min_cluster_size', dest='min_cluster_size', required=False, type=int, default=5, action='store',help='Minimum cluster size for HDBSCAN clustering (default=5)')
	parser.add_argument('-min_samples', '--min_samples', dest='min_samples', required=False, type=int, default=None, action='store',help='Minimum cluster sample parameter for HDBSCAN clustering. Typically equal to min_cluster_size (default=None')	
	parser.add_argument('-modelfile_clust', '--modelfile_clust', dest='modelfile_clust', required=True, type=str, action='store',help='Clustering model filename (.h5)')
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
	
	# - Autoencoder options
	modelfile= args.modelfile
	
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
		logger.info("Retrieve latent data from autoencoder ...")
		snames= vae_class.source_names
		classids= vae_class.source_ids
		vae_data= vae_class.encoded_data

		# - Run UMAP
		logger.info("Running UMAP classifier prediction on autoencoder latent data ...")
		umap_class= UMAPClassifier()
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

