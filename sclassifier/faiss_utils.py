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
	""" Return top_k similar entries above a threshold within a data array. If data size larger than large_data_thr an approximate computation is used. """
	
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

	# Replace NaNs and Infs in data with 0
	data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

	# 1) Normalize data so that inner product = cosine similarity
	# Compute row-wise norms
	# Avoid division by zero by clipping very small norms
	norms = np.linalg.norm(data, axis=1, keepdims=True)
	norms = np.clip(norms, a_min=1e-12, a_max=None)

	# Normalize safely
	data_norm = data / norms

	# Ensure no NaN or Inf remain
	if not np.all(np.isfinite(data_norm)):
		print("Non-finite values found in normalized data, setting them to 0 ...")
		#raise ValueError("Non-finite values found in normalized data")
		data_norm = np.nan_to_num(data_norm, nan=0.0, posinf=0.0, neginf=0.0)

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
	



####################################################
##  FIND NN BETWEEN DATA COLLECTION AND VECTOR
####################################################
def get_top_k_similar(
	data: np.ndarray, 
	data_vector: np.ndarray, 
	k: int = 5, 
	threshold: float = 0.8,
	large_data_thr: int = 1000000,
	nlist: int = 256,
	M: int = 8,
	nprobe: int = 10
):
	""" Return top_k similar entries above a threshold between a data vector and a data array. If data size larger than large_data_thr an approximate computation is used."""
	
	N, Nfeat = data.shape
	if N>large_data_thr:
		logger.info("Using approximate similarity search with IndexIVFPQ faiss index ...")
		return get_approx_top_k_similar(data, data_vector, k, threshold, nlist, M, nprobe)
	else:
		return get_exact_top_k_similar(data, data_vector, k, threshold)
	
def get_exact_top_k_similar(
	data: np.ndarray, 
	data_vector: np.ndarray, 
	k: int = 5, 
	threshold: float = 0.8
):
	"""
	Extract "at least top-k" observations from 'data' that are most similar (cosine similarity) to 'data_vector', while filtering by a configurable threshold.
    
	If more than k neighbors exceed the threshold, we return all of them. 
	If fewer than k neighbors exceed the threshold, we return that smaller subset.

	Parameters
	----------
	data : np.ndarray
		Shape (N, Nfeat). The dataset of N observations, each with Nfeat features.
	data_vector : np.ndarray
		Shape (1, Nfeat). A single query vector.
	k : int, optional
		The minimum number of neighbors to retrieve (if that many pass threshold).
	threshold : float, optional
		The cosine similarity threshold for including neighbors.

	Returns
	-------
	selected_indices : np.ndarray
		1D array of row indices from 'data' that pass the threshold. Sorted in descending order of similarity.
	selected_scores : np.ndarray
		1D array of corresponding cosine similarity scores, same length as selected_indices.
	"""
    
	# Sanity check for matching number of features
	if data.shape[1] != data_vector.shape[1]:
		logger.error("Input data has %d features, but data_vector has %d features. They must match!" % (data.shape[1], data_vector.shape[1]))
		return None
           
	# Ensure float32 for Faiss
	if data.dtype != np.float32:
		data = data.astype(np.float32)
	if data_vector.dtype != np.float32:
		data_vector = data_vector.astype(np.float32)

	N, Nfeat = data.shape
    
	# 1) Normalize data rows => inner product = cosine similarity
	data_norms = np.linalg.norm(data, axis=1, keepdims=True)
	data_norm = data / data_norms

	# 2) Normalize the query vector
	query_norm = np.linalg.norm(data_vector, axis=1, keepdims=True)
	query_vector_norm = data_vector / query_norm

	# 3) Build a Faiss index for inner product (exact search)
	index = faiss.IndexFlatIP(Nfeat)
	index.add(data_norm)  # add the normalized data

	# 4) Perform a top-N search with the single query vector
	#    This returns similarity scores for all N items in the dataset
	#distances, indices = index.search(query_vector_norm, N)
	# distances.shape: (1, N)
	# indices.shape:   (1, N)
	distances, indices = index.search(query_vector_norm, k)
	# distances.shape: (1, k)
	# indices.shape:   (1, k)	
	
	# Both sorted by similarity descending

	#all_scores = distances[0]  # shape (N,)
	#all_indices = indices[0]   # shape (N,)
	all_scores = distances[0]  # shape (k,)
	all_indices = indices[0]   # shape (k,)

	# 5) Filter by threshold
	mask = (all_scores >= threshold)
	filtered_indices = all_indices[mask]
	filtered_scores  = all_scores[mask]

	# The results are already sorted in descending order of similarity.
	# If the threshold is very low, you might get many neighbors above threshold.
	# If the threshold is very high, you might get fewer than k neighbors.

	# 6) "At least top-k": 
	#    If you have more than k neighbors above threshold, keep them all.
	#    If fewer than k pass threshold, just return those few.
	#    So effectively we just return 'filtered_indices' as is.

	return filtered_indices, filtered_scores


def get_approx_top_k_similar(
	data: np.ndarray, 
	data_vector: np.ndarray, 
	k: int = 5, 
	threshold: float = 0.8,
	nlist: int = 256,
	M: int = 8,
	nprobe: int = 10
):
	"""
	Extract "at least top-k" observations from 'data' that are most similar (cosine similarity)
	to 'data_vector', while filtering by a configurable threshold. Uses Faiss IndexIVFPQ
	for large datasets (approximate nearest neighbor).

	If more than k neighbors exceed the threshold, we return all of them.
	If fewer than k neighbors exceed the threshold, we return that smaller subset.

	Parameters
	----------
	data : np.ndarray
		Shape (N, Nfeat). The dataset of N observations, each with Nfeat features.
	data_vector : np.ndarray
		Shape (1, Nfeat). A single query vector.
	k : int, optional
		The minimum number of neighbors to retrieve (if that many pass threshold).
	threshold : float, optional
		The cosine similarity threshold for including neighbors.
	nlist : int, optional
		Number of clusters (inverted lists) for IVF.
	M : int, optional
		Number of sub-quantizers for product quantization.
	nprobe : int, optional
		Number of clusters to visit during search. Larger nprobe = better recall, slower search.

	Returns
	-------
	selected_indices : np.ndarray
		1D array of row indices from 'data' that pass the threshold, sorted in descending order of similarity.
	selected_scores : np.ndarray
		1D array of corresponding cosine similarity scores, same length as selected_indices.
	"""

	# Check feature dimension consistency
	if data.shape[1] != data_vector.shape[1]:
		logger.error("Input data has %d features, but data_vector has %d features. They must match!" % (data.shape[1], data_vector.shape[1]))
		return None

	# 1) Ensure float32 for Faiss
	if data.dtype != np.float32:
		data = data.astype(np.float32)
	if data_vector.dtype != np.float32:
		data_vector = data_vector.astype(np.float32)

	N, D = data.shape

	# 2) Normalize data => inner product = cosine similarity
	data_norms = np.linalg.norm(data, axis=1, keepdims=True)
	data_norm = data / data_norms

	# 3) Normalize the query vector
	query_norms = np.linalg.norm(data_vector, axis=1, keepdims=True)
	query_vector_norm = data_vector / query_norms

	# 4) Build an IVF-PQ index for approximate nearest neighbor search
	#    - The quantizer is a flat IP index
	#    - nlist is the number of clusters
	#    - M is the number of sub-quantizers for product quantization
	quantizer = faiss.IndexFlatIP(D)
	index = faiss.IndexIVFPQ(quantizer, D, nlist, M, 8)
	index.metric_type = faiss.METRIC_INNER_PRODUCT

	# 5) Train the index on data_norm
	#    For very large data, you can sample a subset to speed up training
	index.train(data_norm)

	# 6) Add the normalized data to the index
	index.add(data_norm)

	# 7) Set the number of clusters to search (nprobe)
	index.nprobe = nprobe

	# 8) We want a top-N search so we can filter by threshold.
	#    If N is extremely large, searching top-N might be prohibitive. 
	#    A common approach is to search the top-X (X << N), then filter.
	#    For demonstration, we'll search top-N. Tweak as needed.
	##distances, indices = index.search(query_vector_norm, N)
	# shapes: (1, N)
	distances, indices = index.search(query_vector_norm, k)
	# shapes: (1, k)

	#all_scores = distances[0]  # shape (N,)
	#all_indices = indices[0]   # shape (N,)
	all_scores = distances[0]  # shape (k,)
	all_indices = indices[0]   # shape (k,)

	# 9) Filter by threshold
	mask = all_scores >= threshold
	filtered_indices = all_indices[mask]
	filtered_scores  = all_scores[mask]

	# Results are in descending order of similarity.
	# "At least top-k": if more neighbors exceed threshold, return them all.
	# If fewer than k pass threshold, just return those.
	
	return filtered_indices, filtered_scores
	
