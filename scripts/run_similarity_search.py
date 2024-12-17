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

## ASTRO/IMG PROCESSING MODULES
#from astropy.io import ascii
#from astropy.io import fits
#from astropy.stats import sigma_clipped_stats
#from astropy.stats import sigma_clip
#from astropy.visualization import ZScaleInterval
#from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

## ADDON MODULES
import faiss
#import networkx as nx

## MODULES
from sclassifier import __version__, __date__
from sclassifier import logger
from sclassifier.utils import Utils
from sclassifier.utils import NoIndent, MyEncoder

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

###################################
##   EXTRACT TOP-K SIMILAR DATA  ##
###################################
def get_top_k_similar_within_data(
	data: np.ndarray,
	k: int = 10,
	threshold: float = 0.8,
	large_data_thr: int = 1000000,  
	nlist: int = 100,
	M: int = 8,
	nprobe: int = 10
):
	""" Return top_k similar entries above a threshold. If data size larger than large_data_thr an approximate computation is used. """
	
	N, Nfeat = data.shape
	if N>large_data_thr:
		return get_approx_top_k_similar_within_data(data, k, threshold, nlist, M, nprobe)
	else:
		return get_exact_top_k_similar_within_data(data, k, threshold)
	

def get_exact_top_k_similar_within_data(
	data: np.ndarray, 
	k: int = 10, 
	threshold: float = 0.0
):
	"""
	For each observation in 'data', find the top-k most similar observations 
	(rows) in 'data' itself based on cosine similarity that are above a configurable threshold.

	Parameters
	----------
	data : np.ndarray
		Shape (N, Nfeat). The dataset of N observations, each with Nfeat features.
	k : int, optional
		The number of neighbors to retrieve for each observation, excluding itself.
	threshold : float, optional
		The cosine similarity threshold for including neighbors.

	Returns
	-------
	neighbors_indices : np.ndarray
		Shape (N, k). neighbors_indices[i, :] gives the row indices of the top-k neighbors for the i-th row in the original data, excluding the row itself.
	neighbors_scores : np.ndarray
		Shape (N, k). neighbors_scores[i, :] gives the cosine similarity scores for those top-k neighbors.
		
	neighbors_indices_list : list of np.ndarray
		List of length N. neighbors_indices_list[i] is an array of row indices for neighbors of row i that pass the threshold, truncated to length <= k.
	neighbors_scores_list : list of np.ndarray
		List of length N. neighbors_scores_list[i] is an array of corresponding cosine similarity scores for those neighbors.
	"""
	
	N, Nfeat = data.shape

	# Convert data to float32 if not already, for Faiss compatibility
	if data.dtype != np.float32:
		data = data.astype(np.float32)

	# 1) Normalize each row of 'data' so that each row has L2 norm = 1
	norms = np.linalg.norm(data, axis=1, keepdims=True)
	data_norm = data / norms

	# 2) Create a Faiss index for Inner Product (which matches cosine similarity on normalized vectors)
	index = faiss.IndexFlatIP(Nfeat)

	# 3) Add normalized data to the index
	index.add(data_norm)

	# 4) Search the entire dataset against itself
	#    We'll do a top- (k+1) search because the self-match (row i matching row i)
	#    will always have similarity = 1.0 and we want to exclude that.
	distances, indices = index.search(data_norm, k + 1)
	# distances: shape (N, k+1)
	# indices:   shape (N, k+1)
	
	neighbors_indices_list = []
	neighbors_scores_list = []
    
	for i in range(N):
		# distances[i], indices[i] hold the top-(k+1) neighbors for row i, 
		# sorted descending by similarity
		row_indices = indices[i]
		row_scores  = distances[i]

		# Exclude the row itself (self-match), and filter by threshold
		mask = (row_indices != i) & (row_scores > threshold)

		valid_indices = row_indices[mask]
		valid_scores  = row_scores[mask]

		# Now keep only the top k from these valid neighbors
		# (They are already sorted by similarity in descending order)
		top_k_indices = valid_indices[:k]
		top_k_scores  = valid_scores[:k]

		neighbors_indices_list.append(top_k_indices)
		neighbors_scores_list.append(top_k_scores)

	return neighbors_indices_list, neighbors_scores_list
    

# - Use this version for very large data
def get_approx_top_k_similar_within_data(
	data: np.ndarray,
	k: int = 10,
	threshold: float = 0.8,
	nlist: int = 100,
	M: int = 8,
	nprobe: int = 10
):
	"""
	For each row in 'data', return up to k most similar observations 
	whose cosine similarity is above 'threshold', using a Faiss IndexIVFPQ 
	for approximate nearest neighbor search (Inner Product metric).

	Parameters
	----------
	data : np.ndarray
		Shape (N, D). The dataset of N observations, each D-dimensional.
	k : int, optional
		The maximum number of neighbors to retrieve for each observation.
	threshold : float, optional
		Cosine similarity threshold for including neighbors.
	nlist : int, optional
		The number of clusters (inverted lists) for the IVFPQ index.
	M : int, optional
		The number of sub-quantizers in Product Quantization.
	nprobe : int, optional
		The number of clusters to visit during search. Larger nprobe = better recall but slower.

	Returns
	-------
	neighbors_indices_list : list of np.ndarray
		List of length N. neighbors_indices_list[i] is an array of row indices for neighbors of row i passing the threshold, up to k results.
	neighbors_scores_list : list of np.ndarray
		List of length N. neighbors_scores_list[i] is an array of the corresponding cosine similarities for those neighbors.
	"""

	N, D = data.shape

	# Ensure float32 for Faiss
	if data.dtype != np.float32:
		data = data.astype(np.float32)

	# 1) Normalize data so that inner product = cosine similarity
	norms = np.linalg.norm(data, axis=1, keepdims=True)
	data_norm = data / norms

	# 2) Create an IVF quantizer for the IndexIVFPQ
	#    We'll use an IndexFlatIP quantizer since we want to approximate an inner-product space
	quantizer = faiss.IndexFlatIP(D)

	# 3) Construct the IndexIVFPQ for Inner Product
	#    M=8 means we split the vectors into 8 sub-quantizers. Each sub-vector is quantized to 8 bits by default.
	#    If we do not specify metric_type explicitly, we must set it after creation for IP usage.
	index = faiss.IndexIVFPQ(quantizer, D, nlist, M, 8)
	index.metric_type = faiss.METRIC_INNER_PRODUCT

	# 4) Train the IVF-PQ index on the data
	#    For large data, you might want to sample a subset to speed up training
	index.train(data_norm)

	# 5) Add the data to the index
	index.add(data_norm)

	# 6) Set the number of lists to probe. Higher nprobe => better recall but slower search
	index.nprobe = nprobe

	# 7) We'll do a top-(k+1) search for each row as a query 
	#    Because each row will find itself as the top match with similarity=1
	#    shape: (N, k+1)
	distances, indices = index.search(data_norm, k + 1)

	neighbors_indices_list = []
	neighbors_scores_list = []

	for i in range(N):
		row_indices = indices[i]
		row_scores = distances[i]

		# Exclude the row itself and filter by threshold
		mask = (row_indices != i) & (row_scores > threshold)
		valid_indices = row_indices[mask]
		valid_scores = row_scores[mask]

		# Keep only top k among the valid neighbors
		# (They should already be in descending order of similarity)
		top_k_indices = valid_indices[:k]
		top_k_scores = valid_scores[:k]

		neighbors_indices_list.append(top_k_indices)
		neighbors_scores_list.append(top_k_scores)

	return neighbors_indices_list, neighbors_scores_list
    
    
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
		
	selcols= []
	if args.selcols!="":
		selcols= [int(x.strip()) for x in args.selcols.split(',')]
		
	# - Similarity search options
	#k= args.k + 1  # adding +1 to search for exactly k neighbors in addition to the self similarity
	
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

	#===========================
	#==   SIMILARITY SEARCH
	#===========================
	# - Compute indices & scores of top similar data
	logger.info("Compute indices & scores of top similar data ...")
	
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
	
	for i in range(N):
		feats= list(data[i])
		feats= [float(item) for item in feats]
		indices= list(nn_indices[i]) 
		scores= list(nn_scores[i])
		
		# - Set output data
		d= datalist[i]
		d['feats']= NoIndent(feats)
		d['nn_indices']= NoIndent(indices)
		d['nn_scores']= NoIndent(scores)
		outdata[args.datalist_key].append(d)

	# - Write selected datalist
	logger.info("Write output data to file %s ..." % (outfile))
	
	with open(outfile, 'w') as fp:
		json.dump(outdata, fp, cls=MyEncoder, indent=2)
	
	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

