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

## PLOT MODULES
import matplotlib.pyplot as plt

## MODULES
from sclassifier import __version__, __date__
from sclassifier import logger
from sclassifier.utils import Utils

#===========================
#==   IMPORT MPI
#===========================
MASTER=0
try:
	from mpi4py import MPI as MPI
	comm= MPI.COMM_WORLD
	nproc= comm.Get_size()
	procId= comm.Get_rank()
except Exception as e:
	logger.warn("Failed to import mpi4py module (err=%s), cannot run in parallel ..." % str(e))
	MPI= None
	comm= None
	nproc= 1
	procId= 0

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
	parser.add_argument('-img','--img', dest='img', required=True, type=str, help='Input 2D radio image filename (.fits)')

	# - Distributed processing options
	#parser.add_argument('--split_tiles', dest='split_tiles', action='store_true',help='Split input image into tiles and run classifier on them (default=false)')	
	#parser.set_defaults(split_tiles=False)
	parser.add_argument('-cutout_size','--cutout_size', dest='cutout_size', required=False, type=int, default=256, help='Cutout size (default=256)') 
	parser.add_argument('-grid_step','--grid_step', dest='grid_step', required=False, type=float, default=0.5, help='Grid step fraction with respect to cutout size (default=0.5)') 

	parser.add_argument('--skip_nan', dest='skip_nan', action='store_true',help='Skip cutouts with NAN/bad pixel fraction above threshold (default=false)')	
	parser.set_defaults(skip_nan=False)
	parser.add_argument('-fthr_bad','--fthr_bad', dest='fthr_bad', required=False, type=float, default=0.1, help='Bad pixel fraction threshold above which cutout is skipped (default=0.1)') 


	# - Save options
	parser.add_argument('--save', dest='save', action='store_true',help='Save extracted cutouts (default=false)')	
	parser.set_defaults(save=False)
	parser.add_argument('--add_coords_in_outfile', dest='add_coords_in_outfile', action='store_true',help='Add cutout center coordinates in image output file (default=false)')	
	parser.set_defaults(add_coords_in_outfile=False)

	args = parser.parse_args()	

	return args

#=========================
#==   READ IMG
#=========================
def read_img(filename):
	""" Read input image """

	try:
		data, header, wcs= Utils.read_fits(filename, strip_deg_axis=True)
			
	except Exception as e:
		logger.error("[PROC %d] Failed to read input image %s (err=%s)!" % (procId, filename, str(e)))
		return None

	return data, header, wcs

def write_fits_cutout(cutout, counter, prefix, addcoords=False, xc=None, yc=None):
	""" Write cutout file """

	# - Set outfilename
	digits= '000000'
	if counter>=10 and counter<100:
		digits= '00000'
	elif counter>=100 and counter<1000:
		digits= '0000'
	elif counter>=1000 and counter<10000:
		digits= '000'
	elif counter>=10000 and counter<100000:
		digits= '00'
	elif counter>=100000 and counter<1000000:
		digits= '0'
	elif counter>=1000000 and counter<10000000:
		digits= ''
						
	if MPI is None or nproc==1:
		outfile_cutout= prefix + '_cutout' + digits + str(counter) + '.fits'
		if addcoords:
			outfile_cutout= prefix + '_cutout' + digits + str(counter) + '_x' + str(xc) + '_y' + str(yc) + '.fits'
	else:
		outfile_cutout= prefix + '_proc' + str(procId) + '_cutout' + digits + str(counter) + '.fits'
		if addcoords:
			outfile_cutout= prefix + '_proc' + str(procId) + '_cutout' + digits + str(counter) + '_x' + str(xc) + '_y' + str(yc) + '.fits'
	
	# - Save cutout
	logger.info("[PROC %d] Saving cutout to file %s ..." % (procId, outfile_cutout))
	hdu= fits.PrimaryHDU(cutout.data, header=cutout.wcs.to_header())
	hdul = fits.HDUList([hdu])
	hdul.writeto(outfile_cutout, overwrite=True)




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
	cutout_size= args.cutout_size
	grid_step_fract= args.grid_step
	grid_step= int(grid_step_fract*cutout_size)
	fthr_bad= args.fthr_bad
	skip_nan= args.skip_nan
	save= args.save
	add_coords_in_outfile= args.add_coords_in_outfile

	#===========================
	#==   READ IMAGE
	#==     (ALL PROCS)
	#===========================
	# - Read image
	if procId==MASTER:
		logger.info("[PROC %d] Reading input image %s ..." % (procId, imgfile))
	res= read_img(imgfile)

	if res is None:
		logger.error("[PROC %d] Failed to read image %s!" % (procId, imgfile))
		return 1

	data= res[0]
	header= res[1]
	wcs= res[2]

	data_shape= data.shape
	nx= data_shape[1]
	ny= data_shape[0]

	# - Create a 2D grid
	if procId==MASTER:
		logger.info("[PROC %d] Creating a 2D grid from img size (nx,ny)=(%d,%d) (cutout_size=%d, grid_step=%d) ..." % (procId, nx, ny, cutout_size, grid_step))
	grid_coords= Utils.extract_2d_grid(nx, ny, cutout_size, grid_step)

	#if procId==MASTER:
	#	print("grid_coords")
	#	print(grid_coords)
	#	print(type(grid_coords))
	#	print(grid_coords.shape)

	coords_list= list(grid_coords)

	# - Assign tiles to each MPI proc
	Ncoords= len(coords_list)
	coord_index_start_proc= (np.floor(Ncoords * procId / nproc)).astype(int)
	ncoords_proc= (np.floor(Ncoords * (procId + 1) / nproc) - np.floor(Ncoords * procId / nproc)).astype(int)

	logger.info("[PROC %d] N=%d, index_start=%d, length=%d" % (procId, Ncoords, coord_index_start_proc, ncoords_proc))

	# - Set number of digits for output filename
	filename_base= os.path.basename(imgfile)
	filename_base_noext= os.path.splitext(filename_base)[0]
	outfilename_prefix= filename_base_noext

	#==================================
	#==   RUN
	#==     (ALL PROCS)
	#==================================
	# - Process tiles
	counter= 0
	for i in range(ncoords_proc):
		index= coord_index_start_proc + i
		coord= coords_list[index]
		x= coord[0]
		y= coord[1]

		# - Extract cutout		
		logger.debug("[PROC %d] Creating cutout around coord %d (%d,%d), cutout_size=%d" % (procId, index, x, y, cutout_size))
		cutout= Utils.extract_fits_cutout(data, x, y, cutout_size, wcs)
		#cutout= Utils.extract_fits_cutout(data, x, y, cutout_size)
		if cutout is None:
			logger.warn("[PROC %d] Failed to create cutout around coord %d (%d,%d)!" % (procId, index, x, y))
			continue

		cutout_data= cutout.data
		nrows= cutout_data.shape[0]
		ncols= cutout_data.shape[1]
		npixels= nrows*ncols
		
		# - Check cutout data integrity
		cond= np.logical_and(cutout_data!=0, np.isfinite(cutout_data))
		npixels_bad= np.count_nonzero(~cond)
		f_bad= float(npixels_bad)/float(npixels)
		
		if skip_nan and f_bad>fthr_bad:
			logger.info("[PROC %d] Skipping extracted cutout around coord %d (%d,%d) as f_bad=%f>thr=%f ..." % (procId, index, x, y, f_bad, fthr_bad))
			continue

		if npixels_bad>0:
			logger.warn("[PROC %d] Processed cutout around coord %d (%d,%d) has %d/%d=%f bad pixels ..." % (procId, index, x, y, npixels_bad, npixels, f_bad))
	
		counter+= 1

		# - Save cutout
		if save:
			write_fits_cutout(
				cutout, 
				counter, 
				outfilename_prefix,
				add_coords_in_outfile, x, y
			)

		# - Process cutout image (e.g. apply model)
		# ...
		# ...
		
	if procId==MASTER:
		logger.info("[PROC %d] End processing" % (procId))

	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

