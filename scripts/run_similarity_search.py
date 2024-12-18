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
import io

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections
import csv
import json
import pickle

## MODULES
from sclassifier import __version__, __date__
from sclassifier import logger
from sclassifier.utils import Utils
from sclassifier.utils import NoIndent, MyEncoder
from sclassifier.faiss_utils import get_top_k_similar_within_data, get_top_k_similar
from sclassifier.tf_utils import extract_tf_features_from_img

import matplotlib.pyplot as plt

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
	parser.add_argument('-datafile','--datafile', dest='datafile', required=True, type=str, help='Path to feature data file (.json)')
	parser.add_argument('-datalist_key','--datalist_key', dest='datalist_key', required=False, type=str, default="data", help='Dictionary key name to be read in input datalist (default=data)') 
	parser.add_argument('-selcols','--selcols', dest='selcols', required=False, type=str, default='', help='Data column ids to be selected from input data, separated by commas') 

	parser.add_argument('-img','--img', dest='img', required=False, default="", type=str, help='Path to image file. If provided the program will find the images in the datafile that are most similar to this image (.fits/.png/.jpg)') 
	
	# - Image representation & processing options (to extract image representation)
	parser.add_argument('-model','--model', dest='model', required=True, type=str, help='Path to model architecture (.h5)') 
	parser.add_argument('-model_weights','--model_weights', dest='model_weights', required=True, type=str, help='Path to model weights (.h5)') 
	parser.add_argument('--imgsize', default=224, type=int, help='Image resize size in pixels (default=224)')
	parser.add_argument('--zscale', dest='zscale', action='store_true',help='Apply zscale transform (default=false)')	
	parser.set_defaults(zscale=False)
	parser.add_argument('--zscale_contrast', default=0.25, type=float, help='ZScale transform contrast (default=0.25)')
	
	# - Similarity search options
	parser.add_argument('-k', '--k', dest='k', required=False, type=int, default=10, action='store',help='Number of neighbors in similarity search (default=10)')
	parser.add_argument('-score_thr', '--score_thr', dest='score_thr', required=False, type=float, default=0.6, action='store',help='Similarity threshold below which neighbors are not include in graph (default=0.0)')
	parser.add_argument('-large_data_thr', '--large_data_thr', dest='large_data_thr', required=False, type=int, default=1000000, action='store',help='Number of entries in data above which an approximate search algorithm is used (default=1000000)')
	parser.add_argument('-nlist', '--nlist', dest='nlist', required=False, type=int, default=100, action='store',help='The number of clusters (inverted lists) for the IVFPQ index (default=100)')
	parser.add_argument('-M', '--M', dest='M', required=False, type=int, default=8, action='store',help='The number of sub-quantizers in Product Quantization. (default=8)')
	parser.add_argument('-nprobe', '--nprobe', dest='nprobe', required=False, type=int, default=10, action='store',help='The number of clusters to visit during search. Larger nprobe = better recall but slower (default=10)')
	
	# - Output options
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='featdata_simsearch.json', help='Output filename (.json) of feature data with similarity search info') 
	
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
	if args.datafile=="":
		logger.error("Empty input data filename!")
		return 1
		
	# - Check if image is given
	run_simsearch_with_img= False
	if args.img!="":
		run_simsearch_with_img= True
		
	selcols= []
	if args.selcols!="":
		selcols= [int(x.strip()) for x in args.selcols.split(',')]
	
	#===========================
	#==   READ FEATURE DATA
	#===========================
	# - Read file
	logger.info("Read image dataset %s ..." % (args.datafile))
	datadict= Utils.read_json_datadict(args.datafile)
	if datadict is None:
		logger.error("Failed to read data file %s!" % (args.datafile))
		return 1
		
	# - Check if datalist is empty and has feature data
	if args.datalist_key not in datadict:
		logger.error("No key %s found in read datadict!" % (args.datalist_key))
		return 1
		
	datalist= datadict[args.datalist_key]
		
	if not datalist:
		logger.error("Read datalist is empty!")
		return 1
			
	# - Read feature data from datalist
	featdata= []
	snames= []
	data_labels= []
	data_classids= []
	
	for idx, item in enumerate(datalist):
		sname= item['sname']
		classid= item['id']
		label= item['label']
		if 'feats' not in item:
			logger.error("Missing feats data in entry %d, exit!" % (idx))
			return 1
			
		feats= item['feats']
		
		# - Add entries to list
		snames.append(sname)
		data_labels.append(label)
		data_classids.append(classid)
		featdata.append(feats)

	# - Select columns?
	if selcols:
		data= Utils.get_selected_data_cols(np.array(featdata), selcols)
	else:
		data= np.array(featdata)
	
	N, Nfeat = data.shape
	
	logger.info("%d data read ..." % (N))

	#=================================
	#==   EXTRACT FEATURES FROM IMG?
	#=================================
	if run_simsearch_with_img:
		logger.info("Extracting features from file %s ..." % (args.img))
		data_vector= extract_tf_features_from_img(
			filename=args.img, 
			modelfile=args.model,
			weightfile=args.model_weights, 
			imgsize=args.imgsize, 
			zscale=args.zscale, 
			contrast=args.zscale_contrast
		)
		if data_vector is None:
			logger.error("Failed to extract data representation from image %s!" % (args.img))
			return 1

	#===========================
	#==   SIMILARITY SEARCH
	#===========================
	# - Compute indices & scores of top similar data
	if run_simsearch_with_img:
		logger.info("Compute indices & scores of top similar data between dataset and image ...")
		nn_indices, nn_scores= get_top_k_similar(
			data, 
			data_vector, 
			k= args.k,
			threshold= args.score_thr,
			large_data_thr= args.large_data_thr,
			nlist= args.nlist,
			M= args.M,
			nprobe= args.nprobe
		)
	else:
		logger.info("Compute indices & scores of top similar data within dataset ...")
		nn_indices, nn_scores= get_top_k_similar_within_data(
			data,
			k= args.k,
			threshold= args.score_thr,
			large_data_thr= args.large_data_thr,
			nlist= args.nlist,
			M= args.M,
			nprobe= args.nprobe
		)
	
	#===========================
	#==   SAVE OUTPUTS
	#===========================
	# - Prepare output data
	logger.info("Prepare output data ...")
	outdata= {args.datalist_key: []}
	
	if run_simsearch_with_img:
		for i in range(len(nn_indices)):
			index= nn_indices[i]
			score= nn_scores[i]
			feats= list(data[index])
			feats= [float(item) for item in feats]
			
			# - Set output data
			d= datalist[index]
			d['feats']= NoIndent(feats)
			d['nn_index']= int(index)
			d['nn_score']= float(score)
			outdata[args.datalist_key].append(d)
			
	else:
		for i in range(N):
			feats= list(data[i])
			feats= [float(item) for item in feats]
			indices= list(nn_indices[i])
			indices= [int(item) for item in indices]
			scores= list(nn_scores[i])
			scores= [float(item) for item in scores]
		
			# - Set output data
			d= datalist[i]
			d['feats']= NoIndent(feats)
			d['nn_indices']= NoIndent(indices)
			d['nn_scores']= NoIndent(scores)
			outdata[args.datalist_key].append(d)

	
	# - Write selected datalist
	logger.info("Write output data to file %s ..." % (args.outfile))
	
	with open(args.outfile, 'w') as fp:
		json.dump(outdata, fp, cls=MyEncoder, indent=2)
	
	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

