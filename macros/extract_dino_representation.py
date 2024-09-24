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

## PYTORCH
import torch
import torchvision.transforms as T
from PIL import Image

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
	
	# - Data options
	parser.add_argument('--imgsize', default=224, type=int, help='Image resize size in pixels')
	parser.add_argument('--clip_data', dest='clip_data', action='store_true',help='Apply sigma clipping transform (default=false)')	
	parser.set_defaults(clip_data=False)
	parser.add_argument('--zscale', dest='zscale', action='store_true',help='Apply zscale transform (default=false)')	
	parser.set_defaults(zscale=False)
	parser.add_argument('--norm_min', default=0., type=float, help='Norm min (default=0)')
	parser.add_argument('--norm_max', default=1., type=float, help='Norm max (default=1)')
	parser.add_argument('--to_uint8', dest='to_uint8', action='store_true',help='Convert to uint8 (default=false)')	
	parser.set_defaults(to_uint8=False)
	parser.add_argument('--set_zero_to_min', dest='shift_zero_to_min', action='store_true',help='Set blank pixels to min>0 (default=false)')	
	parser.set_defaults(set_zero_to_min=False)
	parser.add_argument('--in_chans', default = 1, type = int, help = 'Length of subset of dataset to use.')
	
	# - Model option
	parser.add_argument('-model','--model', dest='model', required=False, type=str, default="dinov2_vits14", help='DINO v2 pretrained model {dinov2_vits14,dinov2_vits14_reg,dinov2_vitl14,dinov2_vitl14_reg}') 
	
	# - Run options
	parser.add_argument('-device','--device', dest='device', required=False, type=str, default="cuda", help='Device where to run inference. Default is cuda, if not found use cpu.') 
	
	# - Outfile option
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='featdata.dat', help='Output filename (.dat) of feature data') 
	
	args = parser.parse_args()	

	return args
	

	
def get_clipped_data(self, data, sigma_low=5, sigma_up=30):
	""" Apply sigma clipping to input data and return transformed data """

	# - Find NaNs pixels
	cond= np.logical_and(data!=0, np.isfinite(data))
	data_1d= data[cond]

	# - Clip all pixels that are below sigma clip
	res= sigma_clip(data_1d, sigma_lower=sigma_low, sigma_upper=sigma_up, masked=True, return_bounds=True)
	thr_low= res[1]
	thr_up= res[2]

	data_clipped= np.copy(data)
	data_clipped[data_clipped<thr_low]= thr_low
	data_clipped[data_clipped>thr_up]= thr_up

	# - Set NaNs to 0
	data_clipped[~cond]= 0

	return data_clipped


def get_zscaled_data(self, data, contrast=0.25):
	""" Apply sigma clipping to input data and return transformed data """

	# - Find NaNs pixels
	cond= np.logical_and(data!=0, np.isfinite(data))

	# - Apply zscale transform
	transform= ZScaleInterval(contrast=contrast)
	data_transf= transform(data)	

	# - Set NaNs to 0
	data_transf[~cond]= 0

	return data_transf


def transform_img(data, args):
	""" Transform input data """
	
	data_transf= np.copy(data)

	# - Set NANs to image min
	if args.set_zero_to_min:
		cond= np.logical_and(data_transf!=0, np.isfinite(data_transf))
	else:
		cond= np.isfinite(data_transf)
				
	data_1d= data_transf[cond]
	if data_1d.size==0:
		print("WARN: Input data are all zeros/nan, return None!")
		return None
			
	data_min= np.min(data_1d)
	data_transf[~cond]= data_min

	print("== DATA MIN/MAX ==")
	print(data_transf.min())
	print(data_transf.max())

	# - Clip data?
	if args.clip_data:
		data_clipped= get_clipped_data(data_transf, sigma_low=5, sigma_up=30)
		data_transf= data_clipped

	# - Apply zscale stretch
	if args.zscale:
		print("Apply zscale stretch ...")
		data_stretched= get_zscaled_data(data_transf, contrast=0.25)
		data_transf= data_stretched

	# - Convert to uint8
	#data_transf= (data_transf*255.).astype(np.uint8)
		
	# - Normalize to range
	data_min= data_transf.min()
	data_max= data_transf.max()
	norm_min= args.norm_min
	norm_max= args.norm_max
	if norm_min==data_min and norm_max==data_max:
		print("INFO: Data already normalized in range (%f,%f)" % (norm_min, norm_max))
	else:
		data_norm= (data_transf-data_min)/(data_max-data_min) * (norm_max-norm_min) + norm_min
		data_transf= data_norm
			
	print("== DATA MIN/MAX (AFTER TRANSF) ==")
	print(data_transf.min())
	print(data_transf.max())
	
	# - Convert to uint8
	if args.to_uint8:
		data_transf= data_transf.astype(np.uint8)
	
	return data_transf


def read_img(filename, args):
	""" Read fits image """

	# - Check filename
	if filename=="":
		return None
	
	file_ext= os.path.splitext(filename)[1]
		
	# - Read fits image
	if file_ext=='.fits':
		data= fits.open(filename)[0].data
	else:
		image= Image.open(filename)
		data= np.asarray(image)
	
	if data is None:
		return None
		
	# - Apply transform to numpy array
	data_transf= transform_img(data, args)
	if data_transf is None:
		return None
	data= data_transf
	
	# - Convert numpy to PIL image
	image = Image.fromarray(data)
	
	# - Convert to RGB image
	if args.in_chans==3:
		image= image.convert("RGB")

	print("--> image.shape")
	print(np.asarray(image).shape)	
		
	# - Apply other transforms (e.g. resize, model-specific transforms)
	transform = T.Compose(
  	[
  	  T.Resize(args.imgsize, interpolation=T.InterpolationMode.BICUBIC),
  	  T.ToTensor(),
  	  #T.Normalize(mean=[0.5], std=[0.5])
  	  T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
  	]
	)
	
	image_tensor= transform(image)[:3].unsqueeze(0)
	
	return image_tensor
	
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
	model= args.model
	outfile= args.outfile
	
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
	print("INFO: Loading model %s ..." % (model))
	model= torch.hub.load('facebookresearch/dinov2', model)
	model.to(device)

	# - Loop over images and get representation
	feature_list= []
	snames= []
	class_ids= []
	
	for i in range(nfiles):
		if i%1000==0:
			print("%d/%d images processed ..." % (i+1, nfiles))
		#if i>100:
		#	break
	
		# - Read image
		filename= datalist[i]["filepaths"][0]
		sname= datalist[i]["sname"]
		class_id= datalist[i]["id"]
		
		print("INFO: Reading image %s ..." % (filename))
		image_tensor= read_img(filename, args)
		if image_tensor is None:
			print("WARN: Read/processed image %s is None, skip to next!" % (filename))
			continue
		
		# - Extract model prediction
		with torch.no_grad():
			features = model(image_tensor.to(device))[0]
			##features = model(image_tensor.to("cuda"))[0]
	
		features_numpy= features.cpu().numpy()
		
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
