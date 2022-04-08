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
from sclassifier_vae.utils import g_class_labels, g_class_label_id_map
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
		self.datadir= ""
		self.datadir_mask= ""
		self.datalist_file= ""
		self.datalist_mask_file= ""
		self.datadict= {}
		self.datadict_mask= {}

		self.datadir_radio= ""
		self.datadir_radio_mask= ""
		self.datalist_radio_file= ""
		self.datalist_radio_mask_file= ""
		self.datadict_radio= {}
		self.datadict_radio_mask= {}

		# - scutout info
		self.jobdir_scutout= os.path.join(self.jobdir, "scutout")
		self.jobdir_scutout_multiband= os.path.join(self.jobdir_scutout, "multiband")
		self.jobdir_scutout_radio= os.path.join(self.jobdir_scutout, "radio")
		self.configfile= ""
		self.config= None
		self.config_radio= None
		self.surveys= []
		self.surveys_radio= []
		self.nsurveys= 0
		self.nsurveys_radio= 0

		# - Source catalog info
		self.nsources= 0
		self.nsources_proc= 0
		self.snames_proc= []
		self.slabels_proc= []
		self.regions_proc= []
		self.centroids_proc= []
		self.radii_proc= []
		self.sname_label_map= {}
		self.datalist_proc= []
		self.datalist_mask_proc= []

		self.datalist_radio_proc= []
		self.datalist_radio_mask_proc= []

		# - Color feature extraction options
		self.jobdir_sfeat= os.path.join(self.jobdir, "sfeat")
		self.refch= 0
		self.shrink_masks= False
		self.grow_masks= False
		# 9,10,11,12,13,14,15,16,17,18,19,20,21,22,73,74,75,76,77,78,79,80,81,82
		#self.selfeatcols_5bands= [0,1,2,3,14,15,16,18,20,23]
		self.selfeatcols_5bands= [9,10,11,12,73,74,75,77,79,82]	

		# SELCOLS="13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,     117,118,119,120,121,122,  123,124,125,126,  127,128,129,  130,131,132,133,134,135,136,137"
		#self.selfeatcols_7bands= [0,1,2,3,4,5,  27,28,29,31,  33,35,36,  40,41,42,45,47]
		self.selfeatcols_7bands= [13,14,15,16,17,18,117,118,119,121,123,125,126,130,131,132,135,137]
		self.selfeatcols= []
		self.feat_colors= None
		self.feat_colors_snames= []
		self.feat_colors_classids= []

		# - Classification options
		self.jobdir_sclass= os.path.join(self.jobdir, "sclass")
		self.binary_class= False
		self.modelfile= ""
		self.normalize_feat= False
		self.scalerfile= ""
		self.save_class_labels= False

		# - Output data
		self.outfile_sclass= "classified_data.dat"
		self.outfile_sclass_metrics= "classification_metrics.dat"
		self.outfile_sclass_cm= "confusion_matrix.dat"
		self.outfile_sclass_cm_norm= "confusion_matrix_norm.dat"

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
			status= Utils.write_montage_fits_metadata(inputfile=self.imgfile_fullpath, metadata_file=self.img_metadata, jobdir=self.jobdir_scutout)
		
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
			regs_sel, snames_sel, slabels_sel= Utils.select_regions(regs, self.tags)
		
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
	def make_scutouts(self, config, datadir, datadir_mask, nbands, datalist_file, datalist_mask_file):	
		""" Run scutout and produce source cutout data """

		# - Prepare dir
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

	
	#=============================
	#==   EXTRACT COLOR FEATURES
	#=============================
	def extract_color_features(self):
		""" Extract color features """

		# - Select color features
		if self.nsurveys==5:
			selcols= self.selfeatcols_5bands
		elif nsurveys==7:
			selcols= self.selfeatcols_7bands
		else:
			selcols= []


		# - Create feat extractor obj
		#   NB: All PROC
		fem= FeatExtractorMom()
		fem.refch= self.refch
		fem.draw= False	
		fem.shrink_masks= self.shrink_masks
		fem.grow_masks= self.grow_masks
		fem.subtract_bkg= True
		fem.subtract_bkg_only_refch= False
		fem.ssim_winsize= 3
		fem.save_ssim_pars= True
		fem.save= False
		fem.select_feat= True
		fem.selfeatids= selcols
			
		logger.info("[PROC %d] Extracting color features from cutout data (nsources=%d) ..." % (procId, len(self.datalist_proc)))
		if fem.run_from_datalist(self.datalist_proc, self.datalist_mask_proc)<0:
			logger.error("[PROC %d] Failed to extract color features (see logs)!" % (procId))
			return -1

		param_dict_list= fem.par_dict_list

		# - Merge parameters found by each proc
		if comm is None:
			colfeat_dict_list= param_dict_list
		else:
			logger.info("[PROC %d] Gathering color features ... " % (procId))
			colfeat_dict_list= comm.gather(param_dict_list, root=MASTER)
		
			if procId==MASTER:
				print("colfeat_dict_list")
				print(colfeat_dict_list)
	
				# - Set col feat data
				self.feat_colors= []
				self.feat_colors_snames= []
				self.feat_colors_classids= []

				for dictlist in colfeat_dict_list:
					for d in dictlist:
						keys= list(d.keys())
						nvars= len(keys)-2
						featvars= []
						print("keys")
						print(keys)
						print("nvars")
						print(nvars)

						for i in range(nvars):
							var_index= i+1 # exclude sname
							varname= keys[var_index]
							var= d[varname]
							featvars.append(var)
							print("Adding feat %s ..." % (varname))

						sname= d["sname"]
						classid= d["id"]
						self.feat_colors_snames.append(sname)
						self.feat_colors_classids.append(classid)
						self.feat_colors.append(featvars)
					
				self.feat_colors= np.array(self.feat_colors)

				print("snames")
				print(self.feat_colors_snames)
				print("classids")
				print(self.feat_colors_classids)
				print("feat colors")
				print(self.feat_colors)

		return 0

	#===========================
	#==   CLASSIFY SOURCES
	#===========================
	def classify_sources(self):
		""" Run source classification """

		# - Run source classification
		if procId==MASTER:
			sclass_status= 0
			
			# - Define sclassifier class
			multiclass= True
			if self.binary_class:
				multiclass= False

			sclass= SClassifier(multiclass=multiclass)
			sclass.normalize= self.normalize_feat
			sclass.outfile= self.outfile_sclass
			sclass.outfile_metrics= self.outfile_sclass_metrics
			sclass.outfile_cm= self.outfile_sclass_cm
			sclass.outfile_cm_norm= self.outfile_sclass_cm_norm
			sclass.save_labels= self.save_class_labels
	
			# - Run classification
			sclass_status= sclass.run_predict(
				data=self.feat_colors, class_ids=self.feat_colors_classids, snames=self.feat_colors_snames,
				modelfile=self.modelfile, 
				scalerfile=self.scalerfile
			)
	
			if sclass_status<0:		
				logger.error("[PROC %d] Failed to run classifier on data %s!" % (procId, featfile_allfeat))

		else:
			sclass_status= 0

		if comm is not None:
			sclass_status= comm.bcast(sclass_status, root=MASTER)

		if sclass_status<0:
			logger.error("[PROC %d] Failed to run classifier on data %s, exit!" % (procId, featfile_allfeat))
			return -1

		return 0


	#=========================
	#==   PREPARE JOB DIRS
	#=========================
	def set_job_dirs(self):
		""" Set and create job dirs """

		# - Set job directories & filenames
		self.jobdir_scutout= os.path.join(self.jobdir, "scutout")
		self.jobdir_scutout_multiband= os.path.join(self.jobdir_scutout, "multiband")
		self.jobdir_scutout_radio= os.path.join(self.jobdir_scutout, "radio")
		self.jobdir_sfeat= os.path.join(self.jobdir, "sfeat")
		self.jobdir_sclass= os.path.join(self.jobdir, "sclass")
		
		#self.img_metadata= os.path.join(self.jobdir, "metadata.tbl")
		#self.datadir= os.path.join(self.jobdir, "cutouts")
		#self.datadir_mask= os.path.join(self.jobdir, "cutouts_masked")
		#self.datalist_file= os.path.join(self.jobdir, "datalist.json")
		#self.datalist_mask_file= os.path.join(self.jobdir, "datalist_masked.json")

		self.img_metadata= os.path.join(self.jobdir_scutout, "metadata.tbl")

		self.datadir= os.path.join(self.jobdir_scutout_multiband, "cutouts")
		self.datadir_mask= os.path.join(self.jobdir_scutout_multiband, "cutouts_masked")
		self.datalist_file= os.path.join(self.jobdir_scutout_multiband, "datalist.json")
		self.datalist_mask_file= os.path.join(self.jobdir_scutout_multiband, "datalist_masked.json")

		self.datadir_radio= os.path.join(self.jobdir_scutout_radio, "cutouts")
		self.datadir_radio_mask= os.path.join(self.jobdir_scutout_radio, "cutouts_masked")
		self.datalist_radio_file= os.path.join(self.jobdir_scutout_radio, "datalist.json")
		self.datalist_radio_mask_file= os.path.join(self.jobdir_scutout_radio, "datalist_masked.json")


		self.outfile_sclass= os.path.join(self.jobdir_sclass, "classified_data.dat")
		self.outfile_sclass_metrics= os.path.join(self.jobdir_sclass, "classification_metrics.dat")
		self.outfile_sclass_cm= os.path.join(self.jobdir_sclass, "confusion_matrix.dat")
		self.outfile_sclass_cm_norm= os.path.join(self.jobdir_sclass, "confusion_matrix_norm.dat")

		# - Create directories
		#   NB: Done by PROC 0
		mkdir_status= -1
		
		if procId==MASTER:

			# - Create scutout dir
			mkdir_scutout_status= 0

			if not os.path.exists(self.jobdir_scutout):
				logger.info("[PROC %d] Creating scutout dir %s ..." % (procId, self.jobdir_scutout))
				mkdir_scutout_status= Utils.mkdir(self.jobdir_scutout, delete_if_exists=False)
				
			mkdir_scutout_subdir1_status= 0
			if not os.path.exists(self.jobdir_scutout_multiband):
				logger.info("[PROC %d] Creating scutout dir %s ..." % (procId, self.jobdir_scutout_multiband))
				mkdir_scutout_subdir1_status= Utils.mkdir(self.jobdir_scutout_multiband, delete_if_exists=False)

			mkdir_scutout_subdir2_status= 0
			if not os.path.exists(self.jobdir_scutout_radio):
				logger.info("[PROC %d] Creating scutout dir %s ..." % (procId, self.jobdir_scutout_radio))
				mkdir_scutout_subdir2_status= Utils.mkdir(self.jobdir_scutout_radio, delete_if_exists=False)

			# - Create sfeat dir
			mkdir_sfeat_status= 0
			if not os.path.exists(self.jobdir_sfeat):
				logger.info("[PROC %d] Creating sfeat dir %s ..." % (procId, self.jobdir_sfeat))
				mkdir_sfeat_status= Utils.mkdir(self.jobdir_sfeat, delete_if_exists=False)

			# - Create sclass dir
			mkdir_sclass_status= 0
			if not os.path.exists(self.jobdir_sclass):
				logger.info("[PROC %d] Creating sclass dir %s ..." % (procId, self.jobdir_sfeat))
				mkdir_sclass_status= Utils.mkdir(self.jobdir_sclass, delete_if_exists=False)

			# - Check status
			mkdir_status= 0
			if mkdir_scutout_status<0 or mkdir_scutout_subdir1_status<0 or mkdir_scutout_subdir2_status<0 or mkdir_sfeat_status<0 or mkdir_sclass_status<0:
				mkdir_status= -1

		if comm is not None:
			mkdir_status= comm.bcast(mkdir_status, root=MASTER)

		if mkdir_status<0:
			logger.error("[PROC %d] Failed to create job directories, exit!" % (procId))
			return -1

		return 0

	#=========================
	#==   DISTRIBUTE SOURCE
	#=========================
	def distribute_sources(self):
		""" Distribute sources to each proc """

		# - Read multi-band cutout data list dict and partition source list across processors
		logger.info("[PROC %d] Reading multi-band cutout data list and assign sources to processor ..." % (procId))
		with open(self.datalist_file) as fp:
			self.datadict= json.load(fp)

		with open(self.datalist_mask_file) as fp:
			self.datadict_mask= json.load(fp)
		
		self.nsources= len(self.datadict["data"])
		source_indices= list(range(0, self.nsources))
		source_indices_split= np.array_split(source_indices, nproc)
		source_indices_proc= list(source_indices_split[procId])
		self.nsources_proc= len(source_indices_proc)
		imin= source_indices_proc[0]
		imax= source_indices_proc[self.nsources_proc-1]
	
		logger.info("[PROC %d] #%d sources (multi-band) assigned to this processor ..." % (procId, self.nsources_proc))

		self.datalist_proc= self.datadict["data"][imin:imax+1]
		self.datalist_mask_proc= self.datadict_mask["data"][imin:imax+1]

		# - Read radio cutout data and partition source list across processors
		logger.info("[PROC %d] Reading multi-radio cutout data list and assign sources to processor ..." % (procId))
		with open(self.datalist_radio_file) as fp:
			self.datadict_radio= json.load(fp)

		with open(self.datalist_radio_mask_file) as fp:
			self.datadict_radio_mask= json.load(fp)

		self.datalist_radio_proc= self.datadict_radio["data"][imin:imax+1]
		self.datalist_radio_mask_proc= self.datadict_radio_mask["data"][imin:imax+1]

	
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

		#==================================
		#==   SET JOB DIRS (PROC 0)
		#==================================
		logger.info("[PROC %d] Set and create job directories ..." % (procId))
		if self.set_job_dirs()<0:
			logger.error("[PROC %d] Failed to set and create job dirs!" % (procId))
			return -1

		#==================================
		#==   READ IMAGE DATA   
		#==     (READ - ALL PROC)
		#==     (METADATA GEN - PROC 0) 
		#==================================
		# - Read & generate image metadata
		logger.info("[PROC %d] Reading input image %s and generate metadata ..." % (procId, self.imgfile_fullpath))

		os.chdir(self.jobdir_scutout)

		if self.read_img()<0:
			logger.error("[PROC %d] Failed to read input image %s and/or generate metadata!" % (procId, self.imgfile_fullpath))
			return -1

		#=============================
		#==   READ SCUTOUT CONFIG
		#==      (ALL PROCS)
		#=============================
		# - Create scutout config class (radio+IR)
		logger.info("[PROC %d] Creating scutout config class from template config file %s ..." % (procId, self.configfile))
		add_survey= True
		os.chdir(self.jobdir_scutout_multiband)
		
		config= Utils.make_scutout_config(
			self.configfile, 
			self.surveys, 
			self.jobdir_scutout_multiband, 
			add_survey, 
			self.img_metadata
		)

		if config is None:
			logger.error("[PROC %d] Failed to create scutout config!" % (procId))
			return -1

		self.config= config
		self.nsurveys= len(config.surveys)

		# - Create scutout config class (radio multi)
		os.chdir(self.jobdir_scutout_radio)
		config_radio= Utils.make_scutout_config(
			self.configfile, 
			self.surveys_radio, 
			self.jobdir_scutout_radio, 
			add_survey, 
			self.img_metadata
		)

		if config_radio is None:
			logger.error("[PROC %d] Failed to create scutout radio config!" % (procId))
			return -1

		self.config_radio= config_radio
		self.nsurveys_radio= len(config_radio.surveys)

		#===========================
		#==   READ REGIONS
		#==     (ALL PROCS)
		#===========================
		# - Read DS9 regions and assign sources to each processor
		os.chdir(self.jobdir)

		if self.read_regions()<0:
			logger.error("[PROC %d] Failed to read input DS9 region %s!" % (procId, self.regionfile))
			return -1

		#=============================
		#==   MAKE CUTOUTS
		#== (DISTRIBUTE AMONG PROCS)
		#=============================
		# - Create radio+IR cutouts
		logger.info("[PROC %d] Creating radio-IR cutouts ..." % (procId))
		os.chdir(self.jobdir_scutout_multiband)

		if self.make_scutouts(self.config, self.datadir, self.datadir_mask, self.nsurveys, self.datalist_file, self.datalist_mask_file)<0:
			logger.error("[PROC %d] Failed to create multi-band source cutouts!" % (procId))
			return -1
		
		# - Create radio multi cutouts
		logger.info("[PROC %d] Creating multi-frequency radio cutouts ..." % (procId))
		os.chdir(self.jobdir_scutout_radio)

		if self.make_scutouts(self.config_radio, self.datadir_radio, self.datadir_radio_mask, self.nsurveys_radio, self.datalist_radio_file, self.datalist_radio_mask_file)<0:
			logger.error("[PROC %d] Failed to create radio source cutouts!" % (procId))
			return -1

		# - Distribute sources among proc
		self.distribute_sources()

		#=============================
		#==   EXTRACT FEATURES
		#== (DISTRIBUTE AMONG PROCS)
		#=============================
		os.chdir(self.jobdir_sfeat)

		# - Extract color features
		logger.info("[PROC %d] Extracting color features ..." % (procId))

		if self.extract_color_features()<0:
			logger.error("[PROC %d] Failed to extract color features ..." % (procId))
			return -1
	
		# - Extract spectral index features
		# ...
		# ...

		# - Concatenate features
		# ...
		# ...

		#=============================
		#==   CLASSIFY SOURCES
		#== (PROC 0)
		#=============================
		os.chdir(self.jobdir_sclass)

		logger.info("[PROC %d] Run source classification ..." % (procId))
		
		if self.classify_sources()<0:	
			logger.error("[PROC %d] Failed to run source classification ..." % (procId))
			return -1



		return 0

