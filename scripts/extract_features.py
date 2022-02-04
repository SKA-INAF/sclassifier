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

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections

## MODULES
from sclassifier_vae import __version__, __date__
from sclassifier_vae import logger
from sclassifier_vae.data_loader import DataLoader
from sclassifier_vae.utils import Utils
from sclassifier_vae.feature_extractor import FeatExtractor


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
	parser.add_argument('-datalist','--datalist', dest='datalist', required=True, type=str, help='Input data json filelist') 
	parser.add_argument('-nmax', '--nmax', dest='nmax', required=False, type=int, default=-1, action='store',help='Max number of images to be read (-1=all) (default=-1)')
	
	# - Data pre-processing options
	parser.add_argument('-nx', '--nx', dest='nx', required=False, type=int, default=64, action='store',help='Image resize width in pixels (default=64)')
	parser.add_argument('-ny', '--ny', dest='ny', required=False, type=int, default=64, action='store',help='Image resize height in pixels (default=64)')	
	
	parser.add_argument('--normalize', dest='normalize', action='store_true',help='Normalize input images in range [0,1]')	
	parser.set_defaults(normalize=False)
	parser.add_argument('--scale_to_abs_max', dest='scale_to_abs_max', action='store_true',help='In normalization, if scale_to_max is active, scale to global max across all channels')	
	parser.set_defaults(scale_to_abs_max=False)
	parser.add_argument('--scale_to_max', dest='scale_to_max', action='store_true',help='In normalization, scale to max not to min-max range')	
	parser.set_defaults(scale_to_max=False)

	parser.add_argument('--log_transform', dest='log_transform', action='store_true',help='Apply log transform to images')	
	parser.set_defaults(log_transform=False)

	parser.add_argument('--scale', dest='scale', action='store_true',help='Apply scale factors to images')	
	parser.set_defaults(scale=False)

	parser.add_argument('-scale_factors', '--scale_factors', dest='scale_factors', required=False, type=str, default='', action='store',help='Image scale factors separated by commas (default=empty)')

	parser.add_argument('--standardize', dest='standardize', action='store_true',help='Apply standardization to images')	
	parser.set_defaults(standardize=False)
	parser.add_argument('-img_means', '--img_means', dest='img_means', required=False, type=str, default='', action='store',help='Image means (separated by commas) to be used in standardization (default=empty)')
	parser.add_argument('-img_sigmas', '--img_sigmas', dest='img_sigmas', required=False, type=str, default='', action='store',help='Image sigmas (separated by commas) to be used in standardization (default=empty)')

	parser.add_argument('--chan_divide', dest='chan_divide', action='store_true',help='Apply channel division to images')	
	parser.set_defaults(chan_divide=False)
	parser.add_argument('-chan_mins', '--chan_mins', dest='chan_mins', required=False, type=str, default='', action='store',help='Image channel means (separated by commas) to be used in chan divide (default=empty)')

	parser.add_argument('--erode', dest='erode', action='store_true',help='Apply erosion to image sourve mask')	
	parser.set_defaults(erode=False)	
	parser.add_argument('-erode_kernel', '--erode_kernel', dest='erode_kernel', required=False, type=int, default=5, action='store',help='Erosion kernel size in pixels (default=5)')	

	
	parser.add_argument('--augment', dest='augment', action='store_true',help='Augment images')	
	parser.set_defaults(augment=False)
	
	parser.add_argument('--shuffle', dest='shuffle', action='store_true',help='Shuffle images')	
	parser.set_defaults(shuffle=False)

	parser.add_argument('--resize', dest='resize', action='store_true',help='Resize images')	
	parser.set_defaults(resize=False)

	parser.add_argument('--save_plots', dest='save_plots', action='store_true',help='Save image plots')	
	parser.set_defaults(save_plots=False)
	
	parser.add_argument('-ssim_winsize','--ssim_winsize', dest='ssim_winsize', required=False, type=int, default=3, help='Filter window size to be used in similarity image calculation (default=3)') 
	parser.add_argument('-ssim_thr','--ssim_thr', dest='ssim_thr', required=False, type=float, default=0., help='ssim threshold used to compute trimmed parameters (default=0.)') 
	parser.add_argument('-fthr_zeros', '--fthr_zeros', dest='fthr_zeros', required=False, type=float, default=0.1, action='store',help='Max fraction of zeros above which channel is bad (default=0.1)')
	parser.add_argument('--weight_colmap_with_ssim', dest='weight_colmap_with_ssim', action='store_true',help='Weight color index map with ssim')	
	parser.set_defaults(weight_colmap_with_ssim=False)

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
	datalist= args.datalist
	nmax= args.nmax

	# - Data process options	
	nx= args.nx
	ny= args.ny
	normalize= args.normalize
	scale_to_abs_max= args.scale_to_abs_max
	scale_to_max= args.scale_to_max
	log_transform= args.log_transform
	standardize= args.standardize
	img_means= []
	img_sigmas= []
	if args.img_means!="":
		img_means= [float(x.strip()) for x in args.img_means.split(',')]
	if args.img_sigmas!="":
		img_sigmas= [float(x.strip()) for x in args.img_sigmas.split(',')]

	chan_divide= args.chan_divide
	chan_mins= []
	if args.chan_mins!="":
		chan_mins= [float(x.strip()) for x in args.chan_mins.split(',')]
	erode= args.erode	
	erode_kernel= args.erode_kernel
	resize= args.resize
	augment= args.augment
	shuffle= args.shuffle
	save_plots= args.save_plots
	scale= args.scale
	scale_factors= []
	if args.scale_factors!="":
		scale_factors= [float(x.strip()) for x in args.scale_factors.split(',')]
	
	ssim_winsize= args.ssim_winsize
	ssim_thr= args.ssim_thr
	fthr_zeros= args.fthr_zeros
	weight_colmap_with_ssim= args.weight_colmap_with_ssim

	#===========================
	#==   READ DATA
	#===========================
	# - Create data loader
	dl= DataLoader(filename=datalist)

	# - Read datalist	
	logger.info("Reading datalist %s ..." % datalist)
	if dl.read_datalist()<0:
		logger.error("Failed to read input datalist!")
		return 1
	
	#===========================
	#==   EXTRACT FEATURE DATA
	#===========================
	fe= FeatExtractor(dl)
	fe.set_image_size(nx, ny)
	fe.scale_img= scale
	fe.scale_img_factors= scale_factors
	fe.nmaximgs= nmax
	fe.normalize_img= normalize
	fe.scale_to_abs_max= scale_to_abs_max 
	fe.scale_to_max= scale_to_max
	fe.shuffle_data= shuffle
	fe.log_transform_img= log_transform
	fe.standardize_img= standardize
	fe.img_means= img_means
	fe.img_sigmas= img_sigmas
	fe.chan_divide= chan_divide
	fe.chan_mins= chan_mins
	fe.erode= erode
	fe.erode_kernel= erode_kernel
	fe.scale_img= scale
	fe.scale_img_factors= scale_factors
	fe.save_imgs= save_plots
	fe.winsize= ssim_winsize
	fe.ssim_thr= ssim_thr
	fe.fthr_zeros= fthr_zeros

	logger.info("Running feature extractor ...")
	if fe.run()<0:
		logger.error("Failed to run feat extractor (see logs)!")
		return 1

	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

