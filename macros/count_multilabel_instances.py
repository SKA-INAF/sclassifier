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
import fnmatch
import glob

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections
import json

#### GET SCRIPT ARGS ####
def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

## LOGGER
import logging
import logging.config
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)-15s %(levelname)s - %(message)s",datefmt='%Y-%m-%d %H:%M:%S')
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

###########################
##     ARGS
###########################
def get_args():
	"""This function parses and return arguments passed in"""
	parser = argparse.ArgumentParser(description="Parse args.")

	parser.add_argument('-inputfile','--inputfile', dest='inputfile', required=False, type=str,default='', help='Input file (json)') 
	
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
		
	inputfile= args.inputfile
	
	#===========================
	#==   COUNT CLASSES
	#===========================
	count_dict= {
		"BACKGROUND": 0,
		"COMPACT": 0,
		"RADIO-GALAXY": 0,
		"EXTENDED": 0,
		"DIFFUSE": 0,
		"DIFFUSE-LARGE": 0,
		"FILAMENT": 0,
		"ARTEFACT": 0,
		"RING": 0,
		"ARC": 0,
		"BORDER": 0,
		"MOSAICING": 0,
		"PECULIAR": 0,
		"DUBIOUS": 0,
	}
	
	# - Read data list
	fp= open(inputfile, "r")
	datalist= json.load(fp)["data"]

	# - Count objects
	for item in datalist:
		labels= item["label"]
		for key in count_dict:
			if key in labels:
				count_dict[key]+= 1
		

	print("== COUNTS ==")
	print("#images=%d" % (len(datalist)))
	print(count_dict)

	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

