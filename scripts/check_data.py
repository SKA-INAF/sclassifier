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

	parser.add_argument('--draw', dest='draw', action='store_true',help='Draw images')	
	parser.set_defaults(draw=False)
	
	parser.add_argument('--dump_stats', dest='dump_stats', action='store_true',help='Dump image stats')	
	parser.set_defaults(dump_stats=False)

	parser.add_argument('--exit_on_fault', dest='exit_on_fault', action='store_true',help='Exit on fault')	
	parser.set_defaults(exit_on_fault=False)
	
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
	draw= args.draw
	dump_stats= args.dump_stats
	scale= args.scale
	scale_factors= []
	if args.scale_factors!="":
		scale_factors= [float(x.strip()) for x in args.scale_factors.split(',')]
	outfile_stats= "stats_info.dat"
	exit_on_fault= args.exit_on_fault
	
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
	
	source_labels= dl.snames
	nsamples= len(source_labels)
	if nmax>0 and nmax<nsamples:
		nsamples= nmax

	logger.info("#%d samples to be read ..." % nsamples)


	# - Read data	
	logger.info("Running data loader ...")
	data_generator= dl.data_generator(
		batch_size=1, 
		shuffle=shuffle,
		resize=resize, nx=nx, ny=ny, 	
		normalize=normalize, 
		augment=augment,
		log_transform=log_transform,
		scale=scale, scale_factors=scale_factors,
		retsdata=True
	)	

	img_counter= 0
	img_stats_all= []
	
	while True:
		try:
			data, sdata= next(data_generator)
			img_counter+= 1

			sname= sdata.sname
			label= sdata.label
			classid= sdata.id

			logger.info("Reading image no. %d (name=%s, label=%s) ..." % (img_counter, sname, label))
			#print("data shape")
			#print(data.shape)

			nchannels= data.shape[3]
			
			# - Check for NANs
			has_naninf= np.any(~np.isfinite(data))
			if has_naninf:
				logger.error("Image %d (name=%s) has some nan/inf, check!" % (img_counter, source_labels[img_counter-1]))
				if exit_on_fault:
					return 1
				else:
					break

			# - Check if channels have elements all equal
			for i in range(nchannels):
				data_min= np.min(data[0,:,:,i])
				data_max= np.max(data[0,:,:,i])
				same_values= (data_min==data_max)
				if same_values:
					logger.error("Image %d chan %d (name=%s) has all elements equal to %f, check!" % (img_counter, i+1, source_labels[img_counter-1],data_min))
					if exit_on_fault:
						return 1
					else:
						break
						
				
			# - Check correct norm
			if normalize:
				data_min= np.min(data[0,:,:,:])
				data_max= np.max(data[0,:,:,:])
				correct_norm= (data_min==0 and data_max==1)
				if not correct_norm:
					logger.error("Image %d chan %d (name=%s) has invalid norm (%f,%f), check!" % (img_counter, i+1, source_labels[img_counter-1],data_min,data_max))
					if exit_on_fault:
						return 1
					else:
						break

			# - Dump image stats
			if dump_stats:
				img_stats= [sname]
				
				for i in range(nchannels):
					data_masked= np.ma.masked_equal(data[0,:,:,i], 0.0, copy=False)
					data_min= data_masked.min()
					data_max= data_masked.max()
					data_mean= data_masked.mean() 
					data_std= data_masked.std()
					img_stats.append(data_min)
					img_stats.append(data_max)
					img_stats.append(data_mean)
					img_stats.append(data_std)

				img_stats.append(classid)
				img_stats_all.append(img_stats)

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

			# - Stop generator
			if img_counter>=nsamples:
				logger.info("Sample size (%d) reached, stop generation..." % nsamples)
				break

		except (GeneratorExit, KeyboardInterrupt):
			logger.info("Stop loop (keyboard interrupt) ...")
			break
		except Exception as e:
			logger.warn("Stop loop (exception catched %s) ..." % str(e))
			break

	# - Dump img stats
	if dump_stats:
		logger.info("Dumping img stats info to file %s ..." % (outfile_stats))

		head= "# sname "
		for i in range(nchannels):
			ch= i+1
			s= 'min_ch{i} max_ch{i} mean_ch{i} std_ch{i} '.format(i=ch)
			head= head + s
		head= head + "id"
		logger.info("Stats file head: %s" % (head))
		
		# - Dump to file
		Utils.write_ascii(np.array(img_stats_all), outfile_stats, head)	

	
	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

