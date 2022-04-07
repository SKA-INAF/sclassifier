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


##############################
##   SProcessor CLASS
##############################
class SProcessor(object):
	""" Class to process source """

	def __init__(self):
		""" Return a SProcessor object """

	

##############################
##     Pipeline CLASS
##############################
class Pipeline(object):
	""" Pipeline class """
	
	def __init__(self):
		""" Return a Pipeline object """

		# - Job dir
		self.jobdir= os.getcwd()	

		# - DS9 region options
		self.regionfile= ""
		self.filter_regions_by_tags= False
		self.tags= []
		
		# - Input image data
		self.imgfile= ""
		self.imgfile_fullpath= ""
		self.img_metadata= ""

		# - scutout info
		self.configfile= ""
		self.config= None
		self.surveys= []
		self.nsurveys= 0

		# - Source catalog info
		self.nsources= 0
		self.nsources_proc= 0
		self.snames_proc= []
		self.slabels_proc= []
		self.regions_proc= []
		self.centroids_proc= []
		self.radii_proc= []
		self.sname_label_map= {}

	#=========================
	#==   READ IMG
	#=========================
	def read_img(self):
		""" Read input image and generate Montage metadata """

		# - Read FITS (ALL PROC)
		logger.info("[PROC %d] Reading input image %s ..." % (procId, self.imgfile_fullpath))
		try:
			data, header, wcs= Utils.read_fits(self.imgfile_fullpath)
			
		except Exception as e:
			logger.error("[PROC %d] Failed to read input image %s (err=%s)!" % (procId, self.imgfile_fullpath, str(e)))
			return -1

			data= ret[0]
			header= ret[1]
			wcs= ret[2]
		
		# - Write input image Montage metadata (PROC 0)
		status= 0
		
		if procId==MASTER:
			self.img_metadata= os.path.join(self.jobdir, "metadata.tbl")
			status= Utils.write_montage_fits_metadata(inputfile=self.imgfile_fullpath, metadata_file=self.img_metadata, jobdir=self.jobdir)
		
		else: # OTHER PROCS
			status= -1
			
		if comm is not None:
			status= comm.bcast(status, root=MASTER)

		if status<0:
			logger.error("[PROC %d] Failed to generate Montage metadata for input image %s, exit!" % (procId, self.imgfile_fullpath))
			return -1

		return 0

	#=========================
	#==   READ REGIONS
	#=========================
	def read_regions(self):
		""" Read regions """

		# - Read regions
		logger.info("[PROC %d] Reading DS9 region file %s ..." % (procId, self.regionfile))
		ret= Utils.read_regions([self.regionfile])
		if ret is None:
			logger.error("[PROC %d] Failed to read regions (check format)!" % (procId))
			return -1
	
		regs= ret[0]
		snames= ret[1]
		slabels= ret[2]

		# - Select region by tag
		regs_sel= regs
		snames_sel= snames
		slabels_sel= slabels
		if self.filter_regions_by_tags and self.tags:
			logger.info("[PROC %d] Selecting DS9 region with desired tags ..." % (procId))
			regs_sel, snames_sel, slabels_sel= Utils.select_regions(regs, tags)
		
		if not regs_sel:
			logger.warn("[PROC %d] No region left for processing (check input region file)!" % (procId))
			return -1

		self.sname_label_map= {}
		for i in range(len(snames_sel)):
			sname= snames_sel[i]
			slabel= slabels_sel[i]
			self.sname_label_map[sname]= slabel

		print("sname_label_map")
		print(self.sname_label_map)

		# - Compute centroids & radius
		centroids, radii= Utils.compute_region_info(regs_sel)

		# - Assign sources to each processor
		self.nsources= len(regs_sel)
		source_indices= list(range(0, self.nsources))
		source_indices_split= np.array_split(source_indices, nproc)
		source_indices_proc= list(source_indices_split[procId])
		self.nsources_proc= len(source_indices_proc)
		imin= source_indices_proc[0]
		imax= source_indices_proc[self.nsources_proc-1]
	
		self.snames_proc= snames_sel[imin:imax+1]
		self.slabels_proc= slabels_sel[imin:imax+1]
		self.regions_proc= regs_sel[imin:imax+1]
		self.centroids_proc= centroids[imin:imax+1]
		self.radii_proc= radii[imin:imax+1]
		logger.info("[PROC %d] #%d sources assigned to this processor ..." % (procId, self.nsources_proc))
	
		print("snames_proc %d" % (procId))
		print(self.snames_proc)
	
		return 0	
		

	
	#=========================
	#==   MAKE SCUTOUTS
	#=========================
	#def make_scutouts(self):
	def make_scutouts(self, config, datadir, datadir_mask, nbands, datalist_file, datalist_mask_file):	
		""" Run scutout and produce source cutout data """

		# - Prepare dir
		#datadir= os.path.join(self.jobdir, "cutouts")
		#datadir_mask= os.path.join(self.jobdir, "cutouts_masked")

		mkdir_status= -1
		
		if procId==MASTER:
			if not os.path.exists(datadir):
				logger.info("[PROC %d] Creating cutout data dir %s ..." % (procId, datadir))
				Utils.mkdir(datadir, delete_if_exists=False)

			if not os.path.exists(datadir_mask):
				logger.info("[PROC %d] Creating cutout masked data dir %s ..." % (procId, datadir_mask))
				Utils.mkdir(datadir_mask, delete_if_exists=False)

			mkdir_status= 0

		if comm is not None:
			mkdir_status= comm.bcast(mkdir_status, root=MASTER)

		if mkdir_status<0:
			logger.error("[PROC %d] Failed to create cutout data directory, exit!" % (procId))
			return -1

		# - Make cutouts
		logger.info("[PROC %d] Making cutouts for #%d sources ..." % (procId, self.nsources_proc))
		cm= SCutoutMaker(config)
		cm.datadir= datadir
		cm.datadir_mask= datadir_mask

		for i in range(self.nsources_proc):
			sname= self.snames_proc[i]
			centroid= self.centroids_proc[i]
			radius= self.radii_proc[i]
			region= self.regions_proc[i]

			if cm.make_cutout(centroid, radius, sname, region)<0:
				logger.warn("[PROC %d] Failed to make cutout of source %s, skip to next ..." % (procId, sname))
				continue

		# - Remove source cutout directories if having less than desired survey files
		#   NB: Only PROC 0
		if comm is not None:
			comm.Barrier()

		if procId==MASTER:
			logger.info("[PROC %d] Ensuring that cutout directories contain exactly #%d survey files ..." % (procId, nbands))
			Utils.clear_cutout_dirs(datadir, datadir_mask, nbands)

		#self.datadir= datadir
		#self.datadir_mask= datadir_mask

		# - Make json data lists
		#   NB: Only PROC 0
		if procId==MASTER:
			mkdatalist_status= 0

			# - Create data filelists for cutouts
			#datalist_file= os.path.join(self.jobdir, "datalist.json")
			logger.info("[PROC %d] Creating cutout data list file %s ..." % (procId, datalist_file))
			Utils.make_datalists(datadir, self.sname_label_map, datalist_file)

			# - Create data filelists for masked cutouts
			#datalist_mask_file= os.path.join(self.jobdir, "datalist_masked.json")
			logger.info("[PROC %d] Creating masked cutout data list file %s ..." % (procId, datalist_mask_file))
			Utils.make_datalists(datadir_mask, self.sname_label_map, datalist_mask_file)

			# - Check datalist number of entries
			logger.info("[PROC %d] Checking cutout data list number of entries ..." % (procId))
			try:
				with open(datalist_file) as fp:
					datadict= json.load(fp)
					n= len(datadict["data"])
				
				with open(datalist_mask_file) as fp:
					datadict= json.load(fp)
					n_masked= len(datadict["data"])

				logger.info("[PROC %d] Cutout filelists have sizes (%d,%d) ..." % (procId, n, n_masked))

				if n!=n_masked:
					logger.error("[PROC %d] Data lists for cutouts and masked cutouts differ in size (%d!=%d)!" % (procId, n, n_masked))
					mkdatalist_status= -1
			
			except Exception as e:
				logger.error("[PROC %d] Exception occurred when checking cutout datalist sizes!" % (procId))
				mkdatalist_status= -1

			# - Set data loader
			logger.info("[PROC %d] Reading datalist %s ..." % (procId, datalist_file))
			dl= DataLoader(filename=datalist_file)
			if dl.read_datalist()<0:
				logger.error("Failed to read cutout datalist!")
				mkdatalist_status= -1

			# - Set masked data loader
			logger.info("[PROC %d] Reading masked datalist %s ..." % (procId, datalist_mask_file))
			dl_mask= DataLoader(filename=datalist_mask_file)
			if dl_mask.read_datalist()<0:
				logger.error("Failed to read masked cutout datalist!")
				mkdatalist_status= -1

		else:
			mkdatalist_status= 0
		
		if comm is not None:
			mkdatalist_status= comm.bcast(mkdatalist_status, root=MASTER)

		if mkdatalist_status<0:
			logger.error("[PROC %d] Error on creating cutout data lists, exit!" % (procId))
			return -1

		return 0

	

	#=========================
	#==   RUN
	#=========================
	def run(self, imgfile, regionfile):
		""" Run pipeline """
	
		#==================================
		#==   CHECK INPUTS (ALL PROC)
		#==================================
		# - Check inputs
		if imgfile=="":
			logger.error("Empty input image file given!")
			return -1

		if regionfile=="":
			logger.error("Empty input DS9 region file given!")
			return -1

		# - Check job dir exists
		if not os.path.exists(self.jobdir):
			logger.error("[PROC %d] Given job dir %s does not exist!" % (procId, jobdir))
			return -1

		# - Check configfile has been set (by default it is empty)
		if not os.path.exists(self.configfile):
			logger.error("[PROC %d] Given job dir %s does not exist!" % (procId, jobdir))
			return -1

		# - Check surveys
		if not self.surveys:
			logger.warn("Survey list is empty, please double check ...")

		# - Set vars
		self.imgfile= imgfile
		self.imgfile_fullpath= os.path.abspath(imgfile)
		self.regionfile= regionfile
		self.img_metadata= os.path.join(jobdir, "metadata.tbl")
	
		#==================================
		#==   READ IMAGE DATA   
		#==     (READ - ALL PROC)
		#==     (METADATA GEN - PROC 0) 
		#==================================
		# - Read & generate image metadata
		logger.info("[PROC %d] Reading input image %s and generate metadata ..." % (procId, self.imgfile))
		if self.read_img()<0:
			logger.error("[PROC %d] Failed to read input image %s and/or generate metadata!" % (procId, self.imgfile))
			return -1

	
		#=============================
		#==   READ SCUTOUT CONFIG
		#==      (ALL PROCS)
		#=============================
		# - Create scutout config class (radio+IR)
		logger.info("[PROC %d] Creating scutout config class from template config file %s ..." % (procId, self.configfile))
		add_survey= True
		
		config= Utils.make_scutout_config(
			self.configfile, 
			self.surveys, 
			self.jobdir, 
			add_survey, 
			self.img_metadata
		)

		if config is None:
			logger.error("[PROC %d] Failed to create scutout config!" % (procId))
			return -1

		self.config= config
		self.nsurveys= len(config.surveys)

		#===========================
		#==   READ REGIONS
		#==     (ALL PROCS)
		#===========================
		# - Read DS9 regions and assign sources to each processor
		if self.read_regions()<0:
			logger.error("[PROC %d] Failed to read input DS9 region!" % (procId))
			return -1

		#=============================
		#==   MAKE CUTOUTS
		#== (DISTRIBUTE AMONG PROCS)
		#=============================
		# - Create radio+IR cutouts
		datadir= os.path.join(self.jobdir, "cutouts")
		datadir_mask= os.path.join(self.jobdir, "cutouts_masked")
		datalist_file= os.path.join(self.jobdir, "datalist.json")
		datalist_mask_file= os.path.join(self.jobdir, "datalist_masked.json")

		if self.make_scutouts(self.config, datadir, datadir_mask, self.nsurveys, datalist_file, datalist_mask_file)<0:
			logger.error("PROC %d] Failed to create source cutouts!" % (procId))
			return -1

		
		

		return 0

