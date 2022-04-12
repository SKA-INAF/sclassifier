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
import re
import shutil
import glob
import json

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections
from collections import defaultdict

## ASTRO MODULES
from astropy.io import fits
from astropy.wcs import WCS
from astropy.io import ascii
from astropy.table import Column
import regions

## SCUTOUT MODULES
import scutout
from scutout.config import Config

## MONTAGE MODULES
from montage_wrapper.commands import mImgtbl

## PLOT MODULES
import matplotlib.pyplot as plt


## MODULES
from sclassifier_vae import __version__, __date__
from sclassifier_vae import logger
from sclassifier_vae.data_loader import DataLoader
from sclassifier_vae.utils import Utils
from sclassifier_vae.classifier import SClassifier
from sclassifier_vae.cutout_maker import SCutoutMaker
from sclassifier_vae.feature_extractor_mom import FeatExtractorMom
from sclassifier_vae.data_checker import DataChecker
from sclassifier_vae.data_aereco_checker import DataAERecoChecker
from sclassifier_vae.feature_merger import FeatMerger
from sclassifier_vae.feature_selector import FeatSelector
from sclassifier_vae.pipeline import Pipeline
from sclassifier_vae.pipeline import procId, MASTER, nproc, comm


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

	# - Input image options
	parser.add_argument('-img','--img', dest='img', required=False, type=str, help='Input 2D radio image filename (.fits)') 
	
	# - Region options
	parser.add_argument('-region','--region', dest='region', required=True, type=str, help='Input DS9 region filename with sources to be classified (.reg)') 
	parser.add_argument('--filter_regions_by_tags', dest='filter_regions_by_tags', action='store_true')	
	parser.set_defaults(filter_regions_by_tags=False)
	parser.add_argument('-tags','--tags', dest='tags', required=False, type=str, help='List of region tags to be used for region selection.') 
	
	# - Source cutout options
	parser.add_argument('-scutout_config','--scutout_config', dest='scutout_config', required=True, type=str, help='scutout configuration filename (.ini)') 
	parser.add_argument('-surveys','--surveys', dest='surveys', required=False, type=str, help='List of surveys to be used for cutouts, separated by comma. First survey is radio.') 
	parser.add_argument('-surveys_radio','--surveys_radio', dest='surveys_radio', required=False, type=str, help='List of radio surveys to be used for cutouts and spectral index, separated by comma.') 
	
	# - Autoencoder model options
	parser.add_argument('--run_aereco', dest='run_aereco', action='store_true',help='Run AE reconstruction metrics (default=false)')	
	parser.set_defaults(run_aereco=False)
	parser.add_argument('-nx', '--nx', dest='nx', required=False, type=int, default=64, action='store',help='Image resize width in pixels (default=64)')
	parser.add_argument('-ny', '--ny', dest='ny', required=False, type=int, default=64, action='store',help='Image resize height in pixels (default=64)')
	parser.add_argument('-modelfile_encoder', '--modelfile_encoder', dest='modelfile_encoder', required=False, type=str, default='', action='store',help='Encoder model architecture filename (.json)')
	parser.add_argument('-weightfile_encoder', '--weightfile_encoder', dest='weightfile_encoder', required=False, type=str, default='', action='store',help='Encoder model weights filename (.h5)')
	parser.add_argument('-modelfile_decoder', '--modelfile_decoder', dest='modelfile_decoder', required=False, type=str, default='', action='store',help='Decoder model architecture filename (.json)')
	parser.add_argument('-weightfile_decoder', '--weightfile_decoder', dest='weightfile_decoder', required=False, type=str, default='', action='store',help='Decoder model weights filename (.h5)')
	parser.add_argument('-aereco_thr', '--aereco_thr', dest='aereco_thr', required=False, type=float, default=0.5, action='store',help='AE reco threshold below which data is considered bad (default=0.5)')

	# - Color/moment options
	parser.add_argument('-refch','--refch', dest='refch', required=False, type=int, default=0, help='Reference channel id (default=0)') 
	parser.add_argument('--shrink_mask', dest='shrink_mask', action='store_true')	
	parser.set_defaults(shrink_mask=False)
	parser.add_argument('-kernsizes_shrink','--kernsizes_shrink', dest='kernsizes_shrink', required=False, type=str, default='', help='Mask erosion kernel sizes in pixels used for mask shrinking (default=empty)') 
	parser.add_argument('--grow_mask', dest='grow_mask', action='store_true')	
	parser.set_defaults(grow_mask=False)
	parser.add_argument('-kernsizes_grow','--kernsizes_grow', dest='kernsizes_grow', required=False, type=str, default='', help='Mask dilation kernel sizes in pixels used for mask growing (default=empty)') 
	#parser.add_argument('--subtract_bkg', dest='subtract_bkg', action='store_true')	
	#parser.set_defaults(subtract_bkg=False)
	parser.add_argument('-seed_thr','--seed_thr', dest='seed_thr', required=False, type=float, default=4, help='Seed threshold (default=4)')
	parser.add_argument('-merge_thr','--merge_thr', dest='merge_thr', required=False, type=float, default=2.5, help='Merge threshold (default=2.5)')

	# - Spectral index options
	parser.add_argument('--add_spectral_index', dest='add_spectral_index', action='store_true', help='Run radio multi-freq cutouts, compute spectral index and add it to features')	
	parser.set_defaults(add_spectral_index=False)
	parser.add_argument('-img_group_1','--img_group_1', dest='img_group_1', required=False, type=str, help='Indexes of images in group 1 when computing the spectral index, separated by commas') 
	parser.add_argument('-img_group_2','--img_group_2', dest='img_group_2', required=False, type=str, help='Indexes of images in group 2 when computing the spectral index, separated by commas') 
	parser.add_argument('-alpha_rcoeff_thr', '--alpha_rcoeff_thr', dest='alpha_rcoeff_thr', required=False, type=float, default=0.9, action='store', help='Correlation coefficient threshold used in spectral index T-T plot fit (default=0.9)')

	# - Model options
	parser.add_argument('-modelfile', '--modelfile', dest='modelfile', required=False, type=str, default='', action='store',help='Classifier model filename (.sav)')
	parser.add_argument('--binary_class', dest='binary_class', action='store_true',help='Perform a binary classification {0=EGAL,1=GAL} (default=multiclass)')	
	parser.set_defaults(binary_class=False)
	parser.add_argument('--normalize_feat', dest='normalize_feat', action='store_true',help='Normalize feature data in range [0,1] before applying models (default=false)')	
	parser.set_defaults(normalize_feat=False)
	parser.add_argument('-scalerfile', '--scalerfile', dest='scalerfile', required=False, type=str, default='', action='store',help='Load and use data transform stored in this file (.sav)')
	parser.add_argument('--save_class_labels', dest='save_class_labels', action='store_true',help='Save class labels instead of classid in classification data output (default=false)')	
	parser.set_defaults(save_class_labels=False)

	# - Outlier detection
	parser.add_argument('--find_outliers', dest='find_outliers', action='store_true',help='Find outliers in data (only in prediction step) (default=false)')	
	parser.set_defaults(find_outliers=False)
	parser.add_argument('-modelfile_outlier', '--modelfile_outlier', dest='modelfile_outlier', required=False, type=str, default='', action='store',help='Outlier model filename (.sav)')
	parser.add_argument('-anomaly_thr','--anomaly_thr', dest='anomaly_thr', required=False, type=float, default=0.9, help='Threshold in anomaly score above which observation is set as outlier (default=0.9)') 

	# - Quality data options
	parser.add_argument('-negative_pix_fract_thr','--negative_pix_fract_thr', dest='negative_pix_fract_thr', required=False, type=float, default=0.9, help='Threshold in negative pixel value fraction above which data images are set as bad (default=0.9)') 
	parser.add_argument('-bad_pix_fract_thr','--bad_pix_fract_thr', dest='bad_pix_fract_thr', required=False, type=float, default=0.05, help='Threshold in bad (NAN/0) pixel values above which data images are set as bad (default=0.9)') 

	# - Run options	
	parser.add_argument('-jobdir','--jobdir', dest='jobdir', required=False, type=str, default='', help='Job directory. Set to PWD if empty') 

	# - Output options
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='classified_data.dat', help='Output filename (.dat) with classified data') 
	parser.add_argument('--save_outlier', dest='save_outlier', action='store_true',help='Save outlier results to file (default=false)')	
	parser.set_defaults(save_outlier=False)
	parser.add_argument('-outfile_outlier','--outfile_outlier', dest='outfile_outlier', required=False, type=str, default='outlier_data.dat', help='Output filename (.dat) with outlier data') 
	parser.add_argument('--save_spectral_index', dest='save_spectral_index', action='store_true', help='Save spectral index data to file (default=false)')	
	parser.set_defaults(save_spectral_index=False)

	args = parser.parse_args()	

	return args




##############
##   MAIN   ##
##############
def main():
	"""Main function"""

	#===========================
	#==   PARSE ARGS
	#==     (ALL PROCS)
	#===========================
	if procId==MASTER:
		logger.info("[PROC %d] Parsing script args ..." % (procId))
	try:
		args= get_args()
	except Exception as ex:
		logger.error("[PROC %d] Failed to get and parse options (err=%s)" % (procId, str(ex)))
		return 1

	imgfile= args.img
	regionfile= args.region
	configfile= args.scutout_config 

	surveys= []
	if args.surveys!="":
		surveys= [str(x.strip()) for x in args.surveys.split(',')]

	surveys_radio= []
	if args.surveys_radio!="":
		surveys_radio= [str(x.strip()) for x in args.surveys_radio.split(',')]


	if imgfile=="" and not surveys:
		logger.error("[PROC %d] No image passed, surveys option cannot be empty!" % (procId))
		return 1

	filter_regions_by_tags= args.filter_regions_by_tags
	tags= []
	if args.tags!="":
		tags= [str(x.strip()) for x in args.tags.split(',')]

	jobdir= os.getcwd()
	if args.jobdir!="":
		if not os.path.exists(args.jobdir):
			logger.error("[PROC %d] Given job dir %s does not exist!" % (procId, args.jobdir))
			return 1
		jobdir= args.jobdir

	# - Classifier options
	normalize_feat= args.normalize_feat
	scalerfile= args.scalerfile
	binary_class= args.binary_class
	modelfile= args.modelfile
	save_class_labels= args.save_class_labels

	# - Autoencoder options
	run_aereco= args.run_aereco
	nx= args.nx
	ny= args.ny
	modelfile_encoder= args.modelfile_encoder
	modelfile_decoder= args.modelfile_decoder
	weightfile_encoder= args.weightfile_encoder
	weightfile_decoder= args.weightfile_decoder
	aereco_thr= args.aereco_thr
	empty_filenames= (
		(modelfile_encoder=="" or modelfile_decoder=="") or
		(weightfile_encoder=="" or weightfile_decoder=="")
	)

	if run_aereco and empty_filenames:
		logger.error("[PROC %d] Empty AE model/weight filename given!" % (procId))
		return 1

	# - Outlier search options
	find_outliers= args.find_outliers
	modelfile_outlier= args.modelfile_outlier
	anomaly_thr= args.anomaly_thr
	save_outlier= args.save_outlier
	outfile_outlier= args.outfile_outlier

	# - Color index
	refch= args.refch
	shrink_mask= args.shrink_mask
	kernsizes_shrink= args.kernsizes_shrink
	grow_mask= args.grow_mask
	kernsizes_grow= args.kernsizes_grow
	seed_thr= args.seed_thr
	merge_thr= args.merge_thr

	# - Spectral index
	add_spectral_index= args.add_spectral_index
	img_group_1= []
	img_group_2= []
	if args.img_group_1!="":
		img_group_1= [int(x.strip()) for x in args.img_group_1.split(',')]
	if args.img_group_2!="":
		img_group_2= [int(x.strip()) for x in args.img_group_2.split(',')]

	alpha_rcoeff_thr= args.alpha_rcoeff_thr
	save_spectral_index= args.save_spectral_index

	if add_spectral_index:
		if not img_group_1 or not img_group_2:
			logger.error("Group image indices for spectral index calculation not given in input or empty!")
			return 1
		if len(img_group_1)!=len(img_group_2):
			logger.error("Given group image indices for spectral index calculation do not have the same length!")
			return 1

	# - Quality data options
	negative_pix_fract_thr= args.negative_pix_fract_thr
	bad_pix_fract_thr= args.bad_pix_fract_thr

	#==================================
	#==   RUN
	#==================================
	pipeline= Pipeline()
	pipeline.jobdir= jobdir
	pipeline.filter_regions_by_tags= filter_regions_by_tags
	pipeline.tags= tags
	pipeline.configfile= configfile
	pipeline.surveys= surveys
	pipeline.surveys_radio= surveys_radio
	pipeline.normalize_feat= normalize_feat
	pipeline.scalerfile= scalerfile
	pipeline.modelfile= modelfile
	pipeline.binary_class= binary_class
	pipeline.save_class_labels= save_class_labels

	pipeline.find_outliers = find_outliers 
	pipeline.modelfile_outlier = modelfile_outlier
	pipeline.outlier_thr = anomaly_thr
	pipeline.save_outlier= save_outlier
	pipeline.outfile_outlier = outfile_outlier

	pipeline.run_aereco= run_aereco
	pipeline.modelfile_encoder= modelfile_encoder
	pipeline.modelfile_decoder= modelfile_decoder
	pipeline.weightfile_encoder= weightfile_encoder
	pipeline.weightfile_decoder= weightfile_decoder
	pipeline.resize_img= True
	pipeline.nx= nx
	pipeline.ny= ny
	pipeline.normalize_img= True
	pipeline.scale_img_to_abs_max= False
	pipeline.scale_img_to_max= False
	pipeline.log_transform_img= False
	pipeline.scale_img= False
	pipeline.scale_img_factors= []
	pipeline.standardize_img= False
	pipeline.img_means= []
	pipeline.img_sigmas= []
	pipeline.img_chan_divide= False
	pipeline.img_chan_mins= []
	pipeline.img_erode= False
	pipeline.img_erode_kernel= 9
	pipeline.add_channorm_layer= False
	pipeline.winsize= 3

	pipeline.refch= refch
	pipeline.shrink_mask= shrink_mask
	pipeline.kernsizes_shrink= kernsizes_shrink
	pipeline.grow_mask= grow_mask
	pipeline.kernsizes_grow= kernsizes_grow
	pipeline.seed_thr= seed_thr
	pipeline.merge_thr= merge_thr

	pipeline.add_spectral_index= add_spectral_index
	pipeline.img_group_1= img_group_1
	pipeline.img_group_2= img_group_2
	pipeline.alpha_rcoeff_thr= alpha_rcoeff_thr
	pipeline.save_spectral_index_data= save_spectral_index

	pipeline.negative_pix_fract_thr= negative_pix_fract_thr
	pipeline.bad_pix_fract_thr= bad_pix_fract_thr

	logger.info("[PROC %d] Running source classification pipeline ..." % (procId))
	status= pipeline.run(
		imgfile, regionfile
	)

	if status<0:
		logger.error("Source classification pipeline run failed (see logs)!")
		return 1

	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

