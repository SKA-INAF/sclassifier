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
	
	# - Data pre-processing options
	parser.add_argument('-nx', '--nx', dest='nx', required=False, type=int, default=128, action='store',help='Image resize width in pixels (default=128)')
	parser.add_argument('-ny', '--ny', dest='ny', required=False, type=int, default=128, action='store',help='Image resize height in pixels (default=128)')	
	
	parser.add_argument('--normalize', dest='normalize', action='store_true',help='Normalize input images in range [0,1]')	
	parser.set_defaults(normalize=False)
	
	parser.add_argument('--augment', dest='augment', action='store_true',help='Augment images')	
	parser.set_defaults(augment=False)
	
	parser.add_argument('--shuffle', dest='shuffle', action='store_true',help='Shuffle images')	
	parser.set_defaults(shuffle=False)

	parser.add_argument('--resize', dest='resize', action='store_true',help='Resize images')	
	parser.set_defaults(resize=False)

	parser.add_argument('--draw', dest='draw', action='store_true',help='Draw images')	
	parser.set_defaults(draw=False)
	

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

	# - Data process options	
	nx= args.nx
	ny= args.ny
	normalize= args.normalize
	resize= args.resize
	augment= args.augment
	shuffle= args.shuffle
	draw= args.draw

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


	# - Read data	
	logger.info("Running data loader ...")
	data_generator= dl.data_generator(
		batch_size=1, 
		shuffle=shuffle,
		resize=resize, nx=nx, ny=ny, 	
		normalize=normalize, 
		augment=augment
	)	

	img_counter= 0

	while True:
		try:
			data, _= next(data_generator)
			img_counter+= 1

			logger.info("Reading image no. %d" % img_counter)
			#print("data shape")
			#print(data.shape)

			nchannels= data.shape[3]
			
			# - Check for NANs
			has_naninf= np.any(~np.isfinite(data))
			if has_naninf:
				logger.error("Image %d has some nan/inf, check!" % img_counter)
				break

			# - Draw data
			if draw:
				logger.info("Drawing data ...")
				fig = plt.figure(figsize=(20, 10))
				for i in range(nchannels):
					#logger.info("Reading nchan %d ..." % i+1)
					plt.subplot(1, nchannels, i+1)
					plt.imshow(data[0,:,:,i], origin='lower')
			
				plt.tight_layout()
				plt.show()

		except (GeneratorExit, KeyboardInterrupt):
			logger.info("Stop loop (keyboard interrupt) ...")
			break
		except Exception as e:
			logger.warn("Stop loop (exception catched %s) ..." % str(e))
			break
	
	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

