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
from astropy.io import ascii
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip
from astropy.visualization import ZScaleInterval
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
	parser.add_argument('-inputfile_embeddings','--inputfile_embeddings', dest='inputfile_embeddings', required=True, type=str, help='Path to feature data table file (.dat/.txt)') 
	parser.add_argument('-inputfile_datalist','--inputfile_datalist', dest='inputfile_datalist', required=True, type=str, help='Path to file with image datalist (.json)') 
	
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
	parser.add_argument('-outfile_embeddings','--outfile_embeddings', dest='outfile_embeddings', required=False, type=str, default='featdata_dedupl.dat', help='Output filename (.dat) of feature data with duplicated images removed') 
	parser.add_argument('-outfile_datalist','--outfile_datalist', dest='outfile_datalist', required=False, type=str, default='filelist_dedupl.json', help='Output datalist filename (.json) with duplicated image entries removed') 
	
	# - Draw options
	parser.add_argument('--draw', dest='draw', action='store_true',help='Draw similar images in connected component graph (default=false)')	
	parser.set_defaults(draw=False)
	parser.add_argument('-nimgs_draw', '--nimgs_draw', dest='nimgs_draw', required=False, type=int, default=3, action='store',help='Number of similar images nxn to draw (default=3)')
	

	args = parser.parse_args()	

	return args


def transform_data(x, norm_min=0, norm_max=1, data_scaler=None, outfile_scaler='datascaler.sav'):
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


def sigma_clipping(data, sigma_low, sigma_up, sigma_bkg=3):
	""" Clip input data """

	# - Get 1D data
	cond= np.logical_and(data!=0, np.isfinite(data))
	data_1d= data[cond]
	
	# - Subtract background
	bkgval, _, _ = sigma_clipped_stats(data_1d, sigma=sigma_bkg)
	data_bkgsub= data - bkgval
	data= data_bkgsub

	# - Clip all pixels that are below sigma clip
	cond= np.logical_and(data!=0, np.isfinite(data))
	data_1d= data[cond]
	res= sigma_clip(data_1d, sigma_lower=sigma_low, sigma_upper=sigma_up, masked=True, return_bounds=True)
	thr_low= float(res[1])
	thr_up= float(res[2])
	print("thr_low=%f, thr_up=%f" % (thr_low, thr_up))

	data_clipped= np.copy(data)
	data_clipped[data_clipped<thr_low]= thr_low
	data_clipped[data_clipped>thr_up]= thr_up
	
	return data_clipped

def zscale_stretch(data, contrast):
	""" Apply zscale stretch """	
	
	transform = ZScaleInterval(contrast=contrast)
	data_stretched = transform(data)
	
	return data_stretched

def transform_img(data, contrast, clip_data, sigma_low, sigma_up, sigma_bkg=3):
	""" Transform input data """

	data_transf= np.copy(data)

	# - Clip data?
	if clip_data:
		logger.info("Applying sigma clipping ...")
		data_clipped= sigma_clipping(data_transf, sigma_low, sigma_up, sigma_bkg)
		data_transf= data_clipped

	# - Apply zscale stretch
	logger.info("Applying zscale stretch ...")
	data_stretched= zscale_stretch(data_transf, contrast=contrast)
	data_transf= data_stretched 
	
	return data_transf
	
def read_img(filename):
	""" Read fits image """

	# - Check filename
	if filename=="":
		return None
		
	# - Read fits image
	data, _, _= Utils.read_fits(filename)
	if data is None:
		return None
		
	# - Apply transform
	data_transf= transform_img(data, 
		contrast=0.25, 
		clip_data=False, 
		sigma_low=5, sigma_up=30, sigma_bkg=3
	)

	return data_transf

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
	if args.inputfile_embeddings=="":
		logger.error("Empty embeddings input file!")
		return 1
		
	if args.inputfile_datalist=="":
		logger.error("Empty datalist input file!")
		return 1
		
	inputfile_embeddings= args.inputfile_embeddings
	inputfile_datalist= args.inputfile_datalist
	
	# - Data pre-processing
	normalize= args.normalize
	norm_min= args.norm_min
	norm_max= args.norm_max
	scalerfile= args.scalerfile
	
	# - Similarity search options
	k= args.k + 1  # adding +1 to search for exactly k neibhbors in addition to the self similarity
	score_thr= args.score_thr
	
	# - Output options
	outfile_embeddings= args.outfile_embeddings
	outfile_datalist= args.outfile_datalist
	
	# - Draw options
	draw= args.draw
	nimgs_draw= args.nimgs_draw

	#===========================
	#==   READ DATALIST
	#===========================
	logger.info("Read image dataset filelist %s ..." % (inputfile_datalist))
	fp= open(inputfile_datalist, "r")
	datalist= json.load(fp)["data"]
	nfiles= len(datalist)
	
	#===========================
	#==   READ FEATURE DATA
	#===========================
	logger.info("Read feature data from file %s ..." % (inputfile_embeddings))
	ret= Utils.read_feature_data(inputfile_embeddings)
	if not ret:
		logger.error("Failed to read data from file %s!" % (inputfile_embeddings))
		return 1

	data= ret[0]
	snames= ret[1]
	classids= ret[2]
	
	# - Check for mismatch between feature table and datalist
	nrows= data.shape[0]
	if nrows!=nfiles:
		logger.warning("Mismatch found between number of entries in datalist (%d) and in feature table (%d), exit!" % (nfiles, nrows))
		return 1
	
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
		print("Img no. %d: nneigh=%s, scores=%s" % (i, str(I[i,:]), str(D[i,:])))
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
	logger.info("Finding connected components and subgraphs ...")
	cc= nx.connected_components(G)
	subgraphs = [G.subgraph(c).copy() for c in cc]
	
	logger.info("Looping through %d connected graph components and retain only the centroid of subgraphs ..." % (n_cc))
	indices_sel= []
	for subgraph in subgraphs:
		# - Add connected componet barycenter to selected list
		barycenter_node= nx.barycenter(subgraph, weight="weight")
		barycenter_first_node= barycenter_node[0]
		indices_sel.append(barycenter_first_node)
		
		# - Draw images inside the connected group for testing
		if draw:
			nn_indices= [node for node in subgraph if node!=barycenter_first_node]
			
			fig, axs = plt.subplots(nimgs_draw, nimgs_draw, figsize=(15, 15))
			
			for i in range(nimgs_draw):
				for j in range(nimgs_draw):
					gindex= i*nimgs_draw + j
					if i==0 and j==0:
						filename_img= datalist[barycenter_first_node]["img"]
					else:
						if gindex<len(nn_indices):
							nn_index= nn_indices[gindex]
							filename_img= datalist[nn_index]["img"]
						else:
							filename_img= ""
							
					# - Read image
					if filename_img!="":
						imgdata= read_img(filename_img)
						if imgdata is not None:
							axs[i][j].imshow(imgdata, origin='lower', cmap='inferno')
							
			plt.show()						
							
	
	# - Sort selected indices
	logger.info("#%d/%d feature rows selected, sorting them ..." % (len(indices_sel), nrows))
	indices_sel_sorted= sorted(indices_sel) 
	
	# - Extract selected feature data
	logger.info("Extract selected feature data ...")
	data_sel= data[indices_sel_sorted, :]
	snames_sel= np.array(snames)[indices_sel_sorted]
	classids_sel= np.array(classids)[indices_sel_sorted]
	datalist_sel= list(np.array(datalist)[indices_sel_sorted])
	
	# - Write selected feature data table
	logger.info("Writing selected feature data to file %s ..." % (outfile_embeddings))
	outdata= np.concatenate(
		(snames_sel, data_sel, classids_sel),
		axis=1
	)
	
	znames_counter= list(range(1,nfeats+1))
	znames= '{}{}'.format('z',' z'.join(str(item) for item in znames_counter))
	head= '{} {} {}'.format("# sname",znames,"id")
	
	Utils.write_ascii(outdata, outfile_embeddings, head)
	
	# - Write selected datalist
	logger.info("Write selected datalist to file %s ..." % (outfile_datalist))
	
	outdata_datalist= {"data": datalist_sel}
	with open(outfile_datalist, 'w') as fp:
		json.dump(outdata_datalist, fp)	
	
	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())





