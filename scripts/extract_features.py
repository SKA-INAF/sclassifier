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

	parser.add_argument('--log_transform', dest='log_transform', action='store_true',help='Apply log transform to images')	
	parser.set_defaults(log_transform=False)

	parser.add_argument('--scale', dest='scale', action='store_true',help='Apply scale factors to images')	
	parser.set_defaults(scale=False)

	parser.add_argument('-scale_factors', '--scale_factors', dest='scale_factors', required=False, type=str, default='', action='store',help='Image scale factors separated by commas (default=empty)')
	
	parser.add_argument('--augment', dest='augment', action='store_true',help='Augment images')	
	parser.set_defaults(augment=False)
	
	parser.add_argument('--shuffle', dest='shuffle', action='store_true',help='Shuffle images')	
	parser.set_defaults(shuffle=False)

	parser.add_argument('--resize', dest='resize', action='store_true',help='Resize images')	
	parser.set_defaults(resize=False)

	parser.add_argument('--save_plots', dest='save_plots', action='store_true',help='Save image plots')	
	parser.set_defaults(save_plots=False)
	
	parser.add_argument('-ssim_winsize','--ssim_winsize', dest='ssim_winsize', required=False, type=int, default=3, help='Filter window size to be used in similarity image calculation (default=3)') 
	parser.add_argument('-ssim_thr','--ssim_thr', dest='ssim_thr', required=False, type=float, default=0.5, help='ssim threshold used to compute trimmed parameters (default=0.1)') 

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
	log_transform= args.log_transform
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
	fe.shuffle_data= shuffle
	fe.log_transform_img= log_transform
	fe.scale_img= scale
	fe.scale_img_factors= scale_factors
	fe.save_imgs= save_plots
	fe.winsize= ssim_winsize
	fe.ssim_thr= ssim_thr

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

