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
import collections
import csv
import pickle
import glob

## ASTRO MODULES
from astropy.io import fits
from astropy.wcs import WCS
import regions

## IMAGE PROC MODULES
import cv2

##############################
##     GLOBAL VARS
##############################
from sclassifier_vae import logger
from sclassifier_vae.utils import Utils
import scutout
from scutout.cutout_extractor import CutoutHelper

##################################
##     SCutoutMaker CLASS
##################################
class SCutoutMaker(object):
	""" Source cutout maker class """
	
	def __init__(self, config, workdir=""):
		""" Return a SCutoutMaker object. cutout is scutout.Config() """

		# - Set scutout config
		self.config= config
		self.region_sky= None
		self.nsurveys= len(config.surveys)

		# - Set source options
		self.ra= -1
		self.dec= -1
		self.radius= 0
		self.sname= ""

		# - Dilate mask options
		self.dilatemask= False
		self.kernsize= 5 
		self.maskval= 0

		# - Output options
		self.databasedir= "cutouts"
		self.databasedir_mask= "cutouts_masked"
		self.topdir= os.getcwd()
		if workdir!="":
			self.topdir= workdir
		#self.jobdir= self.topdir
		self.datadir= os.path.join(self.topdir, self.databasedir)
		self.datadir_mask= os.path.join(self.topdir, self.databasedir_mask)

	def make_cutout(self, coord, radius, sname, region_sky):
		""" Run source cutout maker """

		#===========================
		#==   SET SOURCE PARS
		#===========================
		# - Check and set source cutout pars
		if len(coord)!=2:
			logger.error("Empty source position given!")
			return -1

		if radius<=0:
			logger.error("Radius must be >0")
			return -1

		if sname=="":
			logger.error("Source name must not be empty string!")
			return -1

		if region_sky is None:
			logger.error("None region given!")
			return -1

		if self.nsurveys<=0:
			logger.error("No surveys present in config!")
			return -1

		self.ra= coord[0]
		self.dec= coord[1]
		self.radius= radius
		self.sname= sname
		self.region_sky= region_sky
	
		# - Update job dir in config
		#self.jobdir= os.path.join(self.topdir, sname)
		
		if not os.path.exists(self.datadir):
			logger.info("Creating cutout data dir %s ..." % (self.datadir))
			Utils.mkdir(self.datadir, delete_if_exists=False)

		if not os.path.exists(self.datadir_mask):
			logger.info("Creating cutout masked data dir %s ..." % (self.datadir_mask))
			Utils.mkdir(self.datadir_mask, delete_if_exists=False)

		self.config.workdir= self.datadir		
	
		#===========================
		#==   RUN CUTOUT SEARCH
		#===========================
		logger.info("Run cutout search for source %s ..." % (self.sname))
		try:
			ch= CutoutHelper(self.config, self.ra, self.dec, self.sname, self.radius)
			if ch.run()<0:
				errmsg= 'Failed to extract cutout for source ' + self.sname + '!'
				logger.warn(errmsg)
				return -1

		except Exception as e:
			logger.error('Exception (%s) occurred when extracting cutout for source %s!' % (str(e), self.sname))
			return -1

		#===========================
		#==   MASKED CUTOUT DATA
		#===========================
		logger.info("Computing masked cutouts for source %s ..." % (self.sname))
		if self.make_masked_cutouts(self.region_sky, self.dilatemask, self.kernsize, self.maskval)<0:
			logger.error("Failed to create masked cutouts for source %s!" % (self.sname))
			return -1

		
		return 0


	def make_masked_cutouts(self, region_sky, dilatemask=False, kernsize=5, maskval=0):
		""" Produce masked cutouts """

		# - Find cutout files produced
		logger.info("Searching for produced cutouts for source %s ..." % (self.sname))
		cutout_dir= os.path.join(self.datadir, self.sname)
		file_pattern= os.path.join(cutout_dir, "*.fits")
		files= glob.glob(file_pattern)

		nfiles= len(files)
		if nfiles==0 or nfiles!=self.nsurveys:
			logger.warn("Number of cutout files produced (%d) different wrt expected (%d)!" % (nfiles, self.nsurveys))
			return -1

		# - Create directory for masked cutouts
		masked_cutout_dir= os.path.join(self.datadir_mask, self.sname)
		if not os.path.exists(masked_cutout_dir):
			logger.info("Creating cutout masked data dir %s ..." % (masked_cutout_dir))
			Utils.mkdir(masked_cutout_dir, delete_if_exists=False)

		# - Retrieve FITS header & wcs	
		logger.info("Retrieving cutout FITS header & WCS for source %s ..." % (self.sname))
		try:
			header= fits.getheader(files[0])
			data_shape= fits.getdata(files[0]).shape
			wcs= WCS(header)
		except Exception as e:
			logger.error("Failed to retrieve file %s header/WCS for source %s (err=%s)!" % (files[0], self.sname, str(e)))
			return -1

		# - Convert region to pixel coords
		logger.info("Converting sky region for source %s to pixel coordinates ..." % (self.sname))
		try:
			region= region_sky.to_pixel(wcs)
		except Exception as e:
			logger.error("Failed to convert sky region for source %s to pixel coordinates (err=%s)!" % (self.sname, str(e)))
			return -1
			
		# - Compute mask
		logger.info("Computing mask for source %s ..." % (self.sname))
		try:
			mask= region.to_mask(mode='center')
		except Exception as e:
			logger.error("Failed to get mask from region for source %s (err=%s)!" % (self.sname, str(e)))
			return -1

		if mask is None:
			logger.warn("mask obtained from region for source %s is None!" % (self.sname))
			return -1

		# - Compute image mask 
		logger.info("Computing image mask for source %s ..." % (self.sname))
		maskimg= mask.to_image(data_shape)
		if maskimg is None:
			logger.error("maskimg is None for source %s, this shoudn't occur at this stage!" % (self.sname))
			return -1

		maskimg[maskimg!=0]= 1
		maskimg= maskimg.astype(np.uint8)

		# - Dilate image mask to enlarge area around source
		if dilatemask:
			logger.info("Dilating image mask to enlarge area around source %s ..." % (self.sname))
			structel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernsize,kernsize))
			maskimg_dil = cv2.dilate(maskimg, structel, iterations = 1)
			maskimg= maskimg_dil
		
		# - Loop over files and create masked cutouts
		for i in range(nfiles):
			filename= files[i]
			filename_base= os.path.basename(filename)
			filename_base_noext= os.path.splitext(filename_base)[0]
			filename_mask= os.path.join(masked_cutout_dir, filename_base_noext + '_masked.fits')
			
			logger.info("Creating masked cutout file %s from file %s ..." % (filename_mask, filename_base))
			try:
				header= fits.getheader(filename)
				data= fits.getdata(filename)
				data[maskimg==0]= maskval
	
				hdu_out= fits.PrimaryHDU(data, header)
				hdul = fits.HDUList([hdu_out])
				hdul.writeto(filename_mask, overwrite=True)	
			
			except Exception as e:	
				logger.error("Failed to create masked file %s for source %s!" % (filename_mask, self.sname))
				return -1	
		
		return 0

