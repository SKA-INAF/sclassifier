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
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='filelist_remapped.json', help='Output file') 
	
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
	outfile= args.outfile
	
	#===========================
	#==   COUNT CLASSES
	#===========================
	count_dict= {
		"BACKGROUND": 0,
		"RADIO-GALAXY": 0,
		"EXTENDED": 0,
		"DIFFUSE": 0,
		"DIFFUSE-LARGE": 0,
		"ARTEFACT": 0,
	}
	
	label_remap= {
		"BACKGROUND": "BACKGROUND",
		"COMPACT": "BACKGROUND",
		"RADIO-GALAXY": "RADIO-GALAXY",
		"EXTENDED": "EXTENDED",
		"DIFFUSE": "DIFFUSE",
		"DIFFUSE-LARGE": "DIFFUSE-LARGE",
		"FILAMENT": "DIFFUSE-LARGE",
		"ARTEFACT": "ARTEFACT",
		"RING": "NONE",
		"ARC": "NONE",
		"BORDER": "NONE",
		"MOSAICING": "NONE",
		"PECULIAR": "NONE",
		"DUBIOUS": "NONE",
	}
	
	classid_remap= {
		1: 0, # BACKGROUND
		2: 0, # COMPACT
		3: 1,  # RADIO-GALAXY
		4: 2,  # EXTENDED
		5: 3,  # DIFFUSE
		6: 4,  # DIFFUSE-LARGE
		7: 4,  # FILAMENT
		8: 5,  # ARTEFACT
		9: -1, # RING
		10: -1, # ARC
		11: -1, # BORDER
		12: -1, # MOSAICING
		13: -1, # PECULIAR
		14: -1, # DUBIOUS
	}
	
	# - Read data list
	fp= open(inputfile, "r")
	datalist= json.load(fp)["data"]

	# - Remap objects
	for k in range(len(datalist)):
		item= datalist[k]
		labels= item["label"]
		classids= item["id"]
		labels_new= []
		classids_new= []
		for i in range(len(labels)):
			label= labels[i]
			classid= classids[i]
			label_new= label_remap[label]
			classid_new= classid_remap[classid]
			if classid_new==-1: # skip classid=-1
				continue
			
			if classid_new not in classids_new:
				classids_new.append(classid_new)
			if label_new not in labels_new:
				labels_new.append(label_new)
		
		# - Update labels & ids
		datalist[k]["label"]= labels_new
		datalist[k]["id"]= classids_new
		
				
	# - Count objects
	for item in datalist:
		labels= item["label"]
		for key in count_dict:
			if key in labels:
				count_dict[key]+= 1
		
	print("== COUNTS ==")
	print("#images=%d" % (len(datalist)))
	print(count_dict)

	# - Save updated datalist
	print("INFO: Saving filelist with annotation info to file %s ..." % (outfile))
	outdata= {"data": datalist}
	with open(outfile, 'w') as fp:
		json.dump(outdata, fp)
		

	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

