









#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
## CUSTOM MODULES
import sys
sys.path.append('/home/riggi/Software/Sources/ssl-sky-surveys')
from utils.load_trained_model import load_model_from_checkpoint
import h5py

## STANDARD MODULES
import os
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

## PYTORCH
import torch
#import torchvision.transforms as T
from PIL import Image

## TRANSFORMERS
from transformers import AutoProcessor, AutoModel

## ASTRO/IMG PROCESSING MODULES
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip
from astropy.visualization import ZScaleInterval

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
	parser.add_argument('-inputfile','--inputfile', dest='inputfile', required=True, type=str, help='Path to file with image datalist (.json)') 
	parser.add_argument('-nmax','--nmax', dest='nmax', required=False, type=int, default=-1, help='Max number of entries processed in filelist (-1=all)') 
	
	# - Model option
	parser.add_argument('-model_checkpoint','--model_checkpoint', dest='model_checkpoint', required=True, type=str, help='Model checkpoint (ex. pretrained_paper_model.pth.tar)') 
	
	# - Run options
	parser.add_argument('-device','--device', dest='device', required=False, type=str, default="cuda", help='Device where to run inference. Default is cuda, if not found use cpu.') 
	
	# - Outfile option
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='featdata.dat', help='Output filename (.dat) of feature data') 
	
	args = parser.parse_args()	

	return args
	
	
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
	#print("thr_low=%f, thr_up=%f" % (thr_low, thr_up))

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
	
	# - Replace nan values with min
	cond= np.logical_and(data_transf!=0, np.isfinite(data_transf))
	data_1d= data_transf[cond]
	if data_1d.size == 0:
		print("WARN: Input data are all zeros/nan, return None!")
		return None
		
	data_min= np.min(data_1d)
	data_transf[~np.isfinite(data_transf)]= data_min 

	# - Clip data?
	if clip_data:
		data_clipped= sigma_clipping(data_transf, sigma_low, sigma_up, sigma_bkg)
		data_transf= data_clipped

	# - Apply zscale stretch
	data_stretched= zscale_stretch(data_transf, contrast=contrast)
	data_transf= data_stretched 
	
	# - Convert to uint8
	#data_transf= (data_transf*255.).astype(np.uint8)

	# - Convert to 5 channels (see original paper: https://iopscience.iop.org/article/10.3847/2041-8213/abf2c7#apjlabf2c7app1)
	data_cube= np.zeros((data_transf.shape[0], data_transf.shape[1], 5), dtype=data_transf.dtype)
	data_cube[:,:,0]= data_transf
	data_cube[:,:,1]= data_transf
	data_cube[:,:,2]= data_transf
	data_cube[:,:,3]= data_transf
	data_cube[:,:,4]= data_transf
	
	# - Import image in pytorch
	#PIL_image = Image.fromarray(data_cube)
	
	# - Resize
	transform = T.Compose(
  	[
  		T.ToTensor(),	
  	  T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
  #	  ###T.Normalize(mean=[0.5], std=[0.5])
  #	  T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
  	]
	)
	
	#img= transform(PIL_image)[:3].unsqueeze(0)
	
	return data_transf
	
def read_img(filename):
	""" Read fits image """

	# - Check filename
	if filename=="":
		return None
		
	# - Read fits image
	data= fits.open(filename)[0].data
	if data is None:
		return None
		
	# - Apply stretch transform
	data_transf= transform_img(data, 
		contrast=0.25, 
		clip_data=False, 
		sigma_low=5, sigma_up=30, sigma_bkg=3
	)
	if data_transf is None:
		return None
	
	# - Convert to PIL image RGB
	#img= Image.fromarray(data_transf).convert("RGB")
	
	#return img
	
	return data_transf
	
def write_ascii(data, filename, header=''):
	""" Write data to ascii file """

	# - Skip if data is empty
	if data.size<=0:
		print("WARN: Empty data given, no file will be written!")
		return

	# - Open file and write header
	fout = open(filename, 'wt')
	if header:
		fout.write(header)
		fout.write('\n')	
		fout.flush()	
		
	# - Write data to file
	nrows= data.shape[0]
	ncols= data.shape[1]
	for i in range(nrows):
		fields= '  '.join(map(str, data[i,:]))
		fout.write(fields)
		fout.write('\n')	
		fout.flush()	

	fout.close()

##############
##   MAIN   ##
##############
def main():
	"""Main function"""

	#===========================
	#==   PARSE ARGS
	#===========================
	print("INFO: Get script args ...")
	try:
		args= get_args()
	except Exception as ex:
		print("ERROR: Failed to get and parse options (err=%s)",str(ex))
		return 1
		
	# - Input filelist
	if args.inputfile=="":
		print("ERROR: Empty datalist input file!")
		return 1	
	
	inputfile= args.inputfile
	nmax= args.nmax
	model_checkpoint= args.model_checkpoint
	outfile= args.outfile
	
	#device = 'cuda' if torch.cuda.is_available() else 'cpu'
	device= args.device
	if "cuda" in device:
		if not torch.cuda.is_available():
			print("WARN: cuda not available, using cpu...")
			device= "cpu"
	
	print('device:',device)

		
	#===========================
	#==   READ DATALIST
	#===========================
	print("INFO: Read image dataset filelist %s ..." % (inputfile))
	fp= open(inputfile, "r")
	datalist= json.load(fp)["data"]
	nfiles= len(datalist)
	
	#===========================
	#==   MODEL PREDICTION
	#===========================
	# - Load model
	print("INFO: Loading mode from checkpoint %s ..." % (model_checkpoint))
	model = load_model_from_checkpoint(model_checkpoint)
	
	
	# - Loop over images and get representation
	feature_list= []
	snames= []
	class_ids= []
	
	for i in range(nfiles):
		if nmax!=-1 and i>=nmax:
			print("INFO: Max number of entries processed (n=%d), exit." % (nmax))
			break
			
		if i%1000==0:
			print("%d/%d images processed ..." % (i+1, nfiles))
	
		# - Read image
		filename= datalist[i]["filepaths"][0]
		sname= datalist[i]["sname"]
		class_id= datalist[i]["id"]
		
		print("INFO: Reading image %s ..." % (filename))
		img= read_img(filename)
		if img is None:
			print("WARN: Read/processed image %s is None, skip to next!" % (filename))
			continue
			
		img.to(device)
			
		# - Extract image features
		with torch.no_grad():
			features= model(img)
			
		features_numpy= features.cpu().numpy()[0]
		
		if i==0:
			print("features.shape")
			print(features.shape)
			print("features_numpy.shape")
			print(features_numpy.shape)
		
		# - Append to main list
		feature_list.append(features_numpy)
		snames.append(sname)
		class_ids.append(class_id)
		
	# - Write selected feature data table
	print("INFO: Writing selected feature data to file %s ..." % (outfile))
	
	N= len(feature_list)
	nfeats= feature_list[0].shape[0]
	print("INFO: N=%d, nfeats=%d" % (N, nfeats))
	
	featdata_arr= np.array(feature_list)
	snames_arr= np.array(snames).reshape(N,1)
	classids_arr= np.array(class_ids).reshape(N,1)
	
	outdata= np.concatenate(
		(snames_arr, featdata_arr, classids_arr),
		axis=1
	)
	
	znames_counter= list(range(1,nfeats+1))
	znames= '{}{}'.format('z',' z'.join(str(item) for item in znames_counter))
	head= '{} {} {}'.format("# sname",znames,"id")
	
	write_ascii(outdata, outfile, head)
	
	return 0
	
###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())
