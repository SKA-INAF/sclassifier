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
	
	
	# - Autoencoder model options
	parser.add_argument('--check_aereco', dest='check_aereco', action='store_true',help='Check AE reconstruction metrics (default=false)')	
	parser.set_defaults(check_aereco=False)
	parser.add_argument('-nx', '--nx', dest='nx', required=False, type=int, default=64, action='store',help='Image resize width in pixels (default=64)')
	parser.add_argument('-ny', '--ny', dest='ny', required=False, type=int, default=64, action='store',help='Image resize height in pixels (default=64)')
	parser.add_argument('-modelfile_encoder', '--modelfile_encoder', dest='modelfile_encoder', required=False, type=str, default='', action='store',help='Encoder model architecture filename (.json)')
	parser.add_argument('-weightfile_encoder', '--weightfile_encoder', dest='weightfile_encoder', required=False, type=str, default='', action='store',help='Encoder model weights filename (.h5)')
	parser.add_argument('-modelfile_decoder', '--modelfile_decoder', dest='modelfile_decoder', required=False, type=str, default='', action='store',help='Decoder model architecture filename (.json)')
	parser.add_argument('-weightfile_decoder', '--weightfile_decoder', dest='weightfile_decoder', required=False, type=str, default='', action='store',help='Decoder model weights filename (.h5)')
	parser.add_argument('-aereco_thr', '--aereco_thr', dest='aereco_thr', required=False, type=float, default=0.5, action='store',help='AE reco threshold below which data is considered bad (default=0.5)')

	# - Model options
	parser.add_argument('-modelfile', '--modelfile', dest='modelfile', required=False, type=str, default='', action='store',help='Classifier model filename (.sav)')
	parser.add_argument('--binary_class', dest='binary_class', action='store_true',help='Perform a binary classification {0=EGAL,1=GAL} (default=multiclass)')	
	parser.set_defaults(binary_class=False)
	parser.add_argument('--normalize_feat', dest='normalize_feat', action='store_true',help='Normalize feature data in range [0,1] before applying models (default=false)')	
	parser.set_defaults(normalize_feat=False)
	parser.add_argument('-scalerfile', '--scalerfile', dest='scalerfile', required=False, type=str, default='', action='store',help='Load and use data transform stored in this file (.sav)')
	
	# - Output options
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='classified_data.dat', help='Output filename (.dat) with classified data') 
	parser.add_argument('-jobdir','--jobdir', dest='jobdir', required=False, type=str, default='', help='Job directory. Set to PWD if empty') 
	
	args = parser.parse_args()	

	return args


#===========================
#==   READ REGIONS
#===========================
class_labels= ["UNKNOWN","PN","HII","PULSAR","YSO","STAR","GALAXY","QSO"]
class_label_id_map= {
	"UNKNOWN": 0,
	"STAR": 1,
	"GALAXY": 2,
	"PN": 3,
	"HII": 6,
	"PULSAR": 23,
	"YSO": 24,
	"QSO": 6000
}



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
	normalize_feat= args.normalize
	scalerfile= args.scalerfile
	binary_class= args.binary_class
	modelfile= args.modelfile
	

	# - Autoencoder options
	check_aereco= args.check_aereco
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

	if check_aereco and empty_filenames:
		logger.error("[PROC %d] Empty AE model/weight filename given!" % (procId))
		return 1

	#==================================
	#==   RUN
	#==================================
	pipeline= Pipeline()
	pipeline.jobdir= jobdir
	pipeline.filter_regions_by_tags= filter_regions_by_tags
	pipeline.tags= tags
	pipeline.configfile= configfile
	pipeline.surveys= surveys
	pipeline.normalize_feat= normalize_feat
	pipeline.scalerfile= scalerfile
	pipeline.modelfile= modelfile
	pipeline.binary_class= binary_class
	
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

