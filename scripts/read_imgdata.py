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
from sclassencoders import __version__, __date__
from sclassencoders import logger
from sclassencoders.data_provider import DataProvider


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
	parser.add_argument('-filelists','--filelists', dest='filelists', required=True, nargs='+', type=str, default=[], help='List of image filelists') 

	# - Data process options
	parser.add_argument('--normalize_inputs', dest='normalize_inputs', action='store_true',help='Normalize input data before training')	
	parser.set_defaults(normalize_inputs=False)
	parser.add_argument('-normdatamin', '--normdatamin', dest='normdatamin', required=False, type=float, default=-0.0100, action='store',help='Normalization min used to scale data in (0,1) range (default=-100 mJy/beam)')	
	parser.add_argument('-normdatamax', '--normdatamax', dest='normdatamax', required=False, type=float, default=10, action='store',help='Normalization max used to scale data in (0,1) range (default=10 Jy/beam)')
	

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
	filelists= args.filelists
	print(filelists)

	# - Data process options
	normalize_inputs= args.normalize_inputs
	normdatamin= args.normdatamin
	normdatamax= args.normdatamax
	

	#===========================
	#==   CHECK ARGS
	#===========================
	# ...

	#===========================
	#==   READ DATA
	#===========================
	# - Create data provider
	dp= DataProvider(filelists=filelists)

	# - Set options
	dp.enable_inputs_normalization(normalize_inputs)
	dp.set_input_data_norm_range(normdatamin,normdatamax)

	# - Read data	
	logger.info("Running data provider to read image data ...")
	status= dp.read_data()
	if status<0:
		logger.error("Failed to read input image data!")
		return 1
	
	
	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

