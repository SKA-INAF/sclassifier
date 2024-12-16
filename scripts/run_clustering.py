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
import ast

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections

## MODULES
from sclassifier import __version__, __date__
from sclassifier import logger
from sclassifier.utils import Utils
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
	parser.add_argument('-inputfile','--inputfile', dest='inputfile', required=True, type=str, help='Input feature data table filename') 
	parser.add_argument('-datalist_key','--datalist_key', dest='datalist_key', required=False, type=str, default="data", help='Dictionary key name to be read in input datalist (default=data)') 
	parser.add_argument('-selcols','--selcols', dest='selcols', required=False, type=str, default='', help='Data column ids to be selected from input data, separated by commas') 

	# - Pre-processing options
	parser.add_argument('--normalize', dest='normalize', action='store_true',help='Normalize feature data in range [0,1] before clustering (default=false)')	
	parser.set_defaults(normalize=False)	
	parser.add_argument('-scalerfile', '--scalerfile', dest='scalerfile', required=False, type=str, default='', action='store',help='Load and use data transform stored in this file (.sav)')
	
	parser.add_argument('--reduce_dim', dest='reduce_dim', action='store_true',help='Reduce feature data dimensionality before applying the clustering (default=false)')	
	parser.set_defaults(reduce_dim=False)
	parser.add_argument('-reduce_dim_method', '--reduce_dim_method', dest='reduce_dim_method', default='pca', required=False, type=str, action='store',help='Dimensionality reduction method {pca} (default=pca)')
	parser.add_argument('-pca_ncomps', '--pca_ncomps', dest='pca_ncomps', required=False, type=int, default=-1, action='store',help='Number of PCA components to be used (-1=retain all cumulating a variance above threshold) (default=-1)')
	parser.add_argument('-pca_varthr', '--pca_varthr', dest='pca_varthr', required=False, type=float, default=0.9, action='store',help='Cumulative variance threshold used to retain PCA components (default=0.9)')
	
	parser.add_argument('--classid_label_map', dest='classid_label_map', required=False, type=str, default='', help='Class ID label dictionary')
	parser.add_argument('--objids_excluded_in_train', dest='objids_excluded_in_train', required=False, type=str, default='-1,0', help='Source ids not included for training as considered unknown classes')

	# - Clustering options
	parser.add_argument('-min_cluster_size', '--min_cluster_size', dest='min_cluster_size', required=False, type=int, default=5, action='store',help='Minimum cluster size for HDBSCAN clustering (default=5)')
	parser.add_argument('-min_samples', '--min_samples', dest='min_samples', required=False, type=int, default=None, action='store',help='Minimum cluster sample parameter for HDBSCAN clustering. Typically equal to min_cluster_size (default=None')
	parser.add_argument('-cluster_selection_epsilon', '--cluster_selection_epsilon', dest='cluster_selection_epsilon', required=False, type=float, default=0.0, action='store',help='A distance threshold. Clusters below this value will be merged. (default=0')
	parser.add_argument('-modelfile_clust', '--modelfile_clust', dest='modelfile_clust', required=False, type=str, action='store',help='Clustering model filename (.h5)')
	parser.add_argument('--predict_clust', dest='predict_clust', action='store_true',help='Only predict clustering according to current clustering model (default=false)')	
	parser.set_defaults(predict_clust=False)

	# - Save options
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='clustered_data.dat', help='Output filename (.dat) with clustered data') 
	parser.add_argument('-outfile_json','--outfile_json', dest='outfile_json', required=False, type=str, default='clustered_data.json', help='Output filename (.json) with clustered data') 
	parser.add_argument('--no_save_ascii', dest='no_save_ascii', action='store_true',help='Do not save output data to ascii (default=false)')	
	parser.set_defaults(no_save_ascii=False)
	parser.add_argument('--no_save_json', dest='no_save_json', action='store_true',help='Do not save output data to json (default=false)')	
	parser.set_defaults(no_save_json=False)
	parser.add_argument('--no_save_model', dest='no_save_model', action='store_true',help='Do not save model (default=false)')	
	parser.set_defaults(no_save_model=False)
	
	parser.add_argument('--save_labels_in_ascii', dest='save_labels_in_ascii', action='store_true',help='Save class labels to ascii (default=save classids)')
	parser.set_defaults(save_labels_in_ascii=False)
	parser.add_argument('--no_save_features', dest='no_save_features', action='store_true',help='Save features in output file (default=yes)')
	parser.set_defaults(no_save_features=False)

	# - Draw options
	parser.add_argument('--draw', dest='draw', action='store_true',help='Draw plots (default=false)')	
	parser.set_defaults(draw=False)

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
	inputfile= args.inputfile
	datalist_key=args.datalist_key
	selcols= []
	if args.selcols!="":
		selcols= [int(x.strip()) for x in args.selcols.split(',')]
		

	# - Data pre-processing
	normalize= args.normalize
	scalerfile= args.scalerfile
	reduce_dim= args.reduce_dim
	reduce_dim_method= args.reduce_dim_method
	pca_ncomps= args.pca_ncomps
	pca_varthr= args.pca_varthr
	
	classid_label_map= {}
	if args.classid_label_map!="":
		try:
			classid_label_map= ast.literal_eval(args.classid_label_map)
		except Exception as e:
			logger.error("Failed to convert classid label map string to dict (err=%s)!" % (str(e)))
			return -1	
		
		print("== classid_label_map ==")
		print(classid_label_map)

	objids_excluded_in_train= [int(x) for x in args.objids_excluded_in_train.split(',')]	
	
	# - Clustering options
	min_cluster_size= args.min_cluster_size
	min_samples= args.min_samples	
	cluster_selection_epsilon= args.cluster_selection_epsilon
	modelfile_clust= args.modelfile_clust
	predict_clust= args.predict_clust

	# - Plot
	draw= args.draw

	# - Output options
	outfile= args.outfile
	outfile_json= args.outfile_json
	no_save_ascii= args.no_save_ascii
	no_save_json= args.no_save_json
	no_save_model= args.no_save_model
	save_labels_in_ascii= args.save_labels_in_ascii
	no_save_features= args.no_save_features
	
	#===========================
	#==   READ FEATURE DATA
	#===========================
	#ret= Utils.read_feature_data(inputfile)
	#if not ret:
	#	logger.error("Failed to read data from file %s!" % (inputfile))
	#	return 1

	#data= ret[0]
	#snames= ret[1]
	#classids= ret[2]

	#==============================
	#==   RUN CLUSTERING
	#==============================
	logger.info("Running HDBSCAN classifier prediction on input feature data ...")
	clust_class= Clusterer()
	clust_class.selcols= selcols
	clust_class.min_cluster_size= min_cluster_size
	clust_class.min_samples= min_samples
	clust_class.cluster_selection_epsilon= cluster_selection_epsilon
	clust_class.normalize= normalize
	clust_class.reduce_dim= reduce_dim
	clust_class.reduce_dim_method= reduce_dim_method
	clust_class.pca_ncomps= pca_ncomps
	clust_class.pca_varthr= pca_varthr
	clust_class.draw= draw
	clust_class.classid_label_map= classid_label_map
	clust_class.excluded_objids_train = objids_excluded_in_train

	clust_class.outfile= outfile
	clust_class.outfile_json= outfile_json
	clust_class.save_ascii= False if no_save_ascii else True
	clust_class.save_json= False if no_save_json else True
	clust_class.save_model= False if no_save_model else True
	clust_class.save_features= False if no_save_features else True
	clust_class.save_labels_in_ascii= save_labels_in_ascii
	
	status= 0
	if predict_clust:
		status= clust_class.run_predict_from_file(
			datafile=inputfile, 
			modelfile=modelfile_clust, 
			scalerfile=scalerfile, 
			datalist_key=datalist_key
		)
	else:
		status= clust_class.run_clustering_from_file(
			datafile=inputfile, 
			modelfile=modelfile_clust, 
			scalerfile=scalerfile,
			datalist_key=datalist_key
		)
			
	if status<0:
		logger.error("Clustering run failed!")
		return 1

	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

