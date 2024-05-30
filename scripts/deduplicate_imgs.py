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

## ASTRO/IMG PROCESSING MODULES
from astropy.io import ascii
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

## ADDON MODULES
import faiss
import networkx as nx

## MODULES
from sclassifier import __version__, __date__
from sclassifier import logger
from sclassifier.data_loader import DataLoader
from sclassifier.utils import Utils

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
	parser.add_argument('-inputfile','--inputfile', dest='inputfile', required=True, type=str, help='Input feature data table filename') 
	
	# - Pre-processing options
	parser.add_argument('--normalize', dest='normalize', action='store_true',help='Normalize feature data in range [0,1] before applying models (default=false)')	
	parser.set_defaults(normalize=False)
	parser.add_argument('-norm_min', '--norm_min', dest='norm_min', required=False, type=float, default=0., action='store',help='Normalization min value (default=0)')
	parser.add_argument('-norm_max', '--norm_max', dest='norm_max', required=False, type=float, default=1., action='store',help='Normalization max value (default=1)')
	parser.add_argument('-scalerfile', '--scalerfile', dest='scalerfile', required=False, type=str, default='', action='store',help='Load and use data transform stored in this file (.sav)')
	
	# - Similarity search options
	parser.add_argument('-k', '--k', dest='k', required=False, type=int, default=64, action='store',help='Number of neighbors in similarity search (default=64)')
	parser.add_argument('-score_thr', '--score_thr', dest='score_thr', required=False, type=float, default=0.6, action='store',help='Similarity threshold below which neighbors are not include in graph (default=0.6)')
	
	
	# - Output options
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='featdata_dedupl.dat', help='Output filename (.dat) with duplicated features removed') 

	args = parser.parse_args()	

	return args


def transform_data(self, x, norm_min=0, norm_max=1, data_scaler=None, outfile_scaler='datascaler.sav'):
	""" Transform input data here or using a loaded scaler """

	# - Print input data min/max
	x_min= x.min(axis=0)
	x_max= x.max(axis=0)

	print("== INPUT DATA MIN/MAX ==")
	print(x_min)
	print(x_max)

	if data_scaler is None:
		# - Define and run scaler
		logger.info("Define and running data scaler ...")
		data_scaler= MinMaxScaler(feature_range=(norm_min, norm_max))
		x_transf= data_scaler.fit_transform(x)

		print("== TRANSFORM DATA MIN/MAX ==")
		print(data_scaler.data_min_)
		print(data_scaler.data_max_)

		# - Save scaler to file
		logger.info("Saving data scaler to file %s ..." % (outfile_scaler))
		pickle.dump(data_scaler, open(outfile_scaler, 'wb'))
			
	else:
		# - Transform data
		logger.info("Transforming input data using loaded scaler ...")
		x_transf = data_scaler.transform(x)

	# - Print transformed data min/max
	print("== TRANSFORMED DATA MIN/MAX ==")
	x_transf_min= x_transf.min(axis=0)
	x_transf_max= x_transf.max(axis=0)
	print(x_transf_min)
	print(x_transf_max)
	
	return x_transf


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
	if args.inputfile=="":
		logger.error("Empty input file list!")
		return 1
		
	inputfile= args.inputfile
	
	# - Data pre-processing
	normalize= args.normalize
	norm_min= args.norm_min
	norm_max= args.norm_max
	scalerfile= args.scalerfile
	
	# - Similarity search options
	k= args.k + 1  # adding +1 to search for exactly k neibhbors in addition to the self similarity
	score_thr= args.score_thr
	
	# - Output options
	outfile= args.outfile

	#===========================
	#==   READ FEATURE DATA
	#===========================
	ret= Utils.read_feature_data(inputfile)
	if not ret:
		logger.error("Failed to read data from file %s!" % (inputfile))
		return 1

	data= ret[0]
	snames= ret[1]
	classids= ret[2]
	
	#===========================
	#==   NORMALIZE FEATURES
	#===========================
	# - Normalize feature data in range?
	if normalize:
		logger.info("Normalizing feature data ...")
		data_norm= transform_data(data, norm_min, norm_max)
		if data_norm is None:
			logger.error("Data transformation failed!")
			return 1
		data= data_norm

	#===========================
	#==   DEDUPLICATE FEATURES
	#===========================
	# - L2 normalize feature data
	logger.info("L2 normalize feature data ...")
	nrows= data.shape[0]
	nfeats= data.shape[1]
	faiss.normalize_L2(data)
	
	# - Build index for cosine similarity
	logger.info("Building index for cosine similarity search ...")
	index= faiss.IndexFlatIP(nfeats)
	index.add(data)
	logger.info("Created index (ntotal=%d) ..." % (index.ntotal))
	
	# - Run similarity search
	logger.info("Running similarity search (k=%d) ..." % (k))
	D, I = index.search(data, k)

	# - Create graph of similarities
	logger.info("Creating similarity graph ...")
	G = nx.Graph()
	n, d = I.shape

	for i in range(n):
		for j in range(d):
			nn_index= I[i,j]
			score= D[i,j]
			if nn_index!=-1 and score>score_thr:
				G.add_edge(i, nn_index, weight=score)

	n_nodes= G.number_of_nodes()
	n_edges= G.number_of_edges()
	n_cc= nx.number_connected_components(G)
	logger.info("#nodes=%d, #edges=%d, #connected_components=%d" % (n_nodes, n_edges, n_cc))

	# - Loop through connected graph components and retain only the centroid of subgraphs
	logger.info("Looping through %d connected graph components and retain only the centroid of subgraphs ..." % (n_cc))
	cc= nx.connected_components(G)
	subgraphs = [G.subgraph(c).copy() for c in cc]
	
	indices_sel= []
	for subgraph in subgraphs:
		barycenter_node= nx.barycenter(subgraph, weight="weight")
		barycenter_first_node= barycenter_node[0]
		indices_sel.append(indices_sel)
	
	# - Sort selected indices
	logger.info("#%d/%d feature rows selected, sorting them ..." % (len(indices_sel), nrows))
	indices_sel_sorted= sorted(indices_sel) 
	
	# - Extract selected feature data
	logger.info("Extract selected feature data ...")
	data_sel= data[indices_sel_sorted, :]
	snames_sel= np.array(snames)[indices_sel_sorted]
	classids_sel= np.array(classids)[indices_sel_sorted]
	
	# - Wrte selected feature data table
	logger.info("Writing selected feature data to file %s ..." % (outfile))
	outdata= np.concatenate(
		(snames_sel, data_sel, classids_sel),
		axis=1
	)
	
	znames_counter= list(range(1,nfeats+1))
	znames= '{}{}'.format('z',' z'.join(str(item) for item in znames_counter))
	head= '{} {} {}'.format("# sname",znames,"id")
	
	Utils.write_ascii(outdata, outfile, head)
	
	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())





