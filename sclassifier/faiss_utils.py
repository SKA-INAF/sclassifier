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

## ADDON MODULES
import faiss
#import networkx as nx

## SCLASSIFIER MODULES
from .utils import Utils
from .utils import NoIndent, MyEncoder

##############################
##     GLOBAL VARS
##############################
from sclassifier import logger


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
		logger.info("Using approximate similarity search with IndexIVFPQ faiss index ...")
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
	

