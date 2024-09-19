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
	
	# - Data options
	parser.add_argument('-nmax','--nmax', dest='nmax', required=False, type=int, default=-1, help='Max number of entries processed in filelist (-1=all)') 
	parser.add_argument('--imgsize', default=256, type=int, help='Image resize size in pixels (default=256)')
	parser.add_argument('--reset_meanstd', dest='reset_meanstd', action='store_true',help='Apply zscale transform (default=false)')	
	parser.set_defaults(reset_meanstd=False)
	
	parser.add_argument('--zscale', dest='zscale', action='store_true',help='Apply zscale transform (default=false)')	
	parser.set_defaults(zscale=False)
	parser.add_argument('--norm_min', default=0., type=float, help='Norm min (default=0)')
	parser.add_argument('--norm_max', default=1., type=float, help='Norm max (default=1)')
	parser.add_argument('--to_uint8', dest='to_uint8', action='store_true',help='Convert to uint8 (default=false)')	
	parser.set_defaults(to_uint8=False)
	parser.add_argument('--in_chans', default = 1, type = int, help = 'Length of subset of dataset to use.')
	parser.add_argument('--set_zero_to_min', dest='shift_zero_to_min', action='store_true',help='Set blank pixels to min>0 (default=false)')	
	parser.set_defaults(set_zero_to_min=False)
	parser.add_argument('--center_crop', dest='center_crop', action='store_true', help='Center crop image to fixed desired size in pixel, specified in crop_size option (default=no)')	
	parser.set_defaults(center_crop=False)
	parser.add_argument('-crop_size', '--crop_size', dest='crop_size', required=False, type=int, default=224, action='store',help='Crop size in pixels (default=224)')
	
	# - Model option
	parser.add_argument('-model','--model', dest='model', required=False, type=str, default="google/siglip-so400m-patch14-384", help='SigLIP pretrained model {google/siglip-so400m-patch14-384,google/siglip-so400m-patch14-224,google/siglip-base-patch16-256,google/siglip-large-patch16-256}') 
	
	# - Run options
	parser.add_argument('-device','--device', dest='device', required=False, type=str, default="cuda", help='Device where to run inference. Default is cuda, if not found use cpu.') 
	
	# - Outfile option
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='featdata.dat', help='Output filename (.dat) of feature data') 
	
	args = parser.parse_args()	

	return args
	
	
def sigma_clipping(data, sigma_low, sigma_up):
	""" Clip input data """

	# - Get 1D data
	cond= np.logical_and(data!=0, np.isfinite(data))
	data_1d= data[cond]
	
	# - Subtract background
	#bkgval, _, _ = sigma_clipped_stats(data_1d, sigma=sigma_bkg)
	#data_bkgsub= data - bkgval
	#data= data_bkgsub

	# - Clip all pixels that are below sigma clip
	#cond= np.logical_and(data!=0, np.isfinite(data))
	#data_1d= data[cond]
	#res= sigma_clip(data_1d, sigma_lower=sigma_low, sigma_upper=sigma_up, masked=True, return_bounds=True)
	#thr_low= float(res[1])
	#thr_up= float(res[2])
	##print("thr_low=%f, thr_up=%f" % (thr_low, thr_up))

	#data_clipped= np.copy(data)
	#data_clipped[data_clipped<thr_low]= thr_low
	#data_clipped[data_clipped>thr_up]= thr_up
	
	
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
	
	


def zscale_stretch(data, contrast):
	""" Apply zscale stretch """	
	
	transform = ZScaleInterval(contrast=contrast)
	data_stretched = transform(data)
	
	return data_stretched

def transform_img(data, norm_range, apply_zscale, contrast, clip_data, sigma_low, sigma_up, to_uint8, set_zero_to_min=False):
	""" Transform input data """

	data_transf= np.copy(data)
	
	# - Replace nan values with min
	if set_zero_to_min:
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
	if clip_data:
		data_clipped= sigma_clipping(data_transf, sigma_low, sigma_up, sigma_bkg)
		data_transf= data_clipped

	# - Apply zscale stretch
	if apply_zscale:
		data_stretched= zscale_stretch(data_transf, contrast=contrast)
		data_transf= data_stretched 
		
	# - Normalize to range
	data_min= data_transf.min()
	data_max= data_transf.max()
	norm_min= norm_range[0]
	norm_max= norm_range[1]
	if norm_min==data_min and norm_max==data_max:
		print("INFO: Data already normalized in range (%f,%f)" % (norm_min, norm_max))
	else:
		data_norm= (data_transf-data_min)/(data_max-data_min) * (norm_max-norm_min) + norm_min
		data_transf= data_norm
			
	print("== DATA MIN/MAX (AFTER TRANSF) ==")
	print(data_transf.min())
	print(data_transf.max())
	
	# - Convert to uint8
	if to_uint8:
		data_transf= data_transf.astype(np.uint8)	
		#data_transf= (data_transf*255.).astype(np.uint8)

	return data_transf
	
def read_img(filename, args):
	""" Read fits image """

	# - Check filename
	if filename=="":
		return None
		
	# - Read fits image
	data= fits.open(filename)[0].data
	if data is None:
		return None
				
	# - Apply stretch transform
	data_transf= transform_img(
		data, 
		norm_range=(args.norm_min, args.norm_max),
		apply_zscale=args.zscale, contrast=0.25, 
		clip_data=False, sigma_low=5, sigma_up=30,
		to_uint8=args.to_uint8,
		set_zero_to_min=args.set_zero_to_min
	)
	if data_transf is None:
		return None
	
	# - Convert to PIL image RGB?
	if args.in_chans==3:
		img= Image.fromarray(data_transf).convert("RGB")
	else:
		img= Image.fromarray(data_transf)
	
	return img
	
	
	
	
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
	model_id= args.model
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
	print("INFO: Loading model %s ..." % (model_id))
	
	model = AutoModel.from_pretrained(model_id).to(device)
	
	# - Load model processor
	print("INFO: Loading model processor ...")
	processor = AutoProcessor.from_pretrained(model_id)
	image_processor= processor.image_processor
	
	print("image_processor")
	print(image_processor)
	print("image_processor.size")
	print(image_processor.size)
	
	print("== ORIGINAL PROCESSOR OPTIONS ==")
	imgsize_orig= (image_processor["size"]["height"], image_processor["size"]["width"])
	imgmean_orig= image_processor["image_mean"]
	imgstd_orig= image_processor["image_std"]
	print("imgsize_orig")
	print(imgsize_orig)
	print("imgmean_orig")
	print(imgmean_orig)
	print("imgstd_orig")
	print(imgstd_orig)
	
	# - Update processor options
	processor.image_processor["size"]["height"]= args.imgsize
	processor.image_processor["size"]["width"]= args.imgsize
	
	if args.reset_meanstd:
		processor.image_processor["image_mean"]= [0.,0.,0.]
		processor.image_processor["image_std"]= [1.,1.,1.]
		
	print("== FINAL PROCESSOR OPTIONS ==")
	imgsize_final= (image_processor["size"]["height"], image_processor["size"]["width"])
	imgmean_final= image_processor["image_mean"]
	imgstd_final= image_processor["image_std"]
	print("imgsize_final")
	print(imgsize_final)
	print("imgmean_final")
	print(imgmean_final)
	print("imgstd_final")
	print(imgstd_final)	
	
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
		img= read_img(filename, args)
		if img is None:
			print("WARN: Read/processed image %s is None, skip to next!" % (filename))
			continue
			
		# - Apply model pre-processing
		inputs = processor(images=img, return_tensors="pt").to(device)
		
		# - Extract image features
		with torch.no_grad():
			features= model.get_image_features(**inputs)
    
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
