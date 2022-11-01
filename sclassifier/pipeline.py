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
from collections import defaultdict, OrderedDict

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
from sclassifier import __version__, __date__
from sclassifier import logger
from sclassifier.data_loader import DataLoader
from sclassifier.utils import Utils
from sclassifier.utils import g_class_labels, g_class_label_id_map
from sclassifier.classifier import SClassifier
from sclassifier.cutout_maker import SCutoutMaker
from sclassifier.feature_extractor_mom import FeatExtractorMom
from sclassifier.data_checker import DataChecker
from sclassifier.data_aereco_checker import DataAERecoChecker
from sclassifier.feature_merger import FeatMerger
from sclassifier.feature_selector import FeatSelector
from sclassifier.feature_extractor_ae import FeatExtractorAE
from sclassifier.spectral_index_tt import SpectralIndexTTCalculator

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
		self.shrink_kernels= []
		self.grow_masks= False
		self.grow_kernels= []
		self.subtract_bkg= True
		self.save_ssim_pars= True
		self.seed_thr= 4
		self.merge_thr= 2.5

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
		self.feat_colors_dict= None

		# - Classification options
		self.jobdir_sclass= os.path.join(self.jobdir, "sclass")
		self.binary_class= False
		self.modelfile= ""
		self.normalize_feat= False
		self.scalerfile= ""
		self.save_class_labels= False

		# - Outlier detection options
		self.find_outliers= False
		self.modelfile_outlier= ""
		self.anomaly_thr= 0.7
		self.max_samples= "auto"
		self.max_features= 1
		self.save_outlier= False
		self.outfile_outlier= "outlier_data.dat"

		# - Autoencoder reco options
		self.run_aereco= False
		self.modelfile_encoder= ""
		self.modelfile_decoder= ""
		self.weightfile_encoder= ""
		self.weightfile_decoder= ""
		self.resize_img= True
		self.nx= 64
		self.ny= 64
		self.normalize_img= True
		self.scale_img_to_abs_max= False
		self.scale_img_to_max= False
		self.log_transform_img= False
		self.scale_img= False
		self.scale_img_factors= []
		self.standardize_img= False
		self.img_means= []
		self.img_sigmas= []
		self.img_chan_divide= False
		self.img_chan_mins= []
		self.img_erode= False
		self.img_erode_kernel= 9
		self.add_channorm_layer= False
		self.winsize= 3

		# - Radio spectral index calculation
		self.add_spectral_index= False
		self.alpha_img_freqs= []
		self.alpha_img_group_1= []
		self.alpha_img_group_2= []
		self.alpha_rcoeff_thr= 0.9
		self.save_spectral_index_data= False
		self.feat_alpha= None
		self.feat_alpha_snames= []
		self.feat_alpha_classids= []
		self.feat_alpha_dict= None

		# - Merged feature data
		self.feat_all= None
		self.feat_all_snames= []
		self.feat_all_classids= []
		self.feat_all_dict= None

		# - Data quality options
		self.negative_pix_fract_thr= 0.9
		self.bad_pix_fract_thr= 0.05

		# - Output data
		self.outfile_sclass= "classified_data.dat"
		self.outfile_sclass_metrics= "classification_metrics.dat"
		self.outfile_sclass_cm= "confusion_matrix.dat"
		self.outfile_sclass_cm_norm= "confusion_matrix_norm.dat"
		self.outfile_aerecometrics= "aereco_metrics.dat"
		self.outfile_feat_alpha= "features_alpha.csv"
		self.outfile_feat_merged= "features_merged.csv"

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
			#logger.info("[PROC %d] Reading datalist %s ..." % (procId, datalist_file))
			#dl= DataLoader(filename=datalist_file)
			#if dl.read_datalist()<0:
			#	logger.error("Failed to read cutout datalist!")
			#	mkdatalist_status= -1

			# - Set masked data loader
			#logger.info("[PROC %d] Reading masked datalist %s ..." % (procId, datalist_mask_file))
			#dl_mask= DataLoader(filename=datalist_mask_file)
			#if dl_mask.read_datalist()<0:
			#	logger.error("Failed to read masked cutout datalist!")
			#	mkdatalist_status= -1

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
		fem.erode_kernels = self.shrink_kernels
		fem.grow_masks= self.grow_masks
		fem.dilate_kernels = self.grow_kernels
		fem.seed_thr= self.seed_thr
		fem.merge_thr= self.merge_thr
		fem.subtract_bkg= self.subtract_bkg
		fem.subtract_bkg_only_refch= False
		fem.ssim_winsize= 3
		fem.save_ssim_pars= self.save_ssim_pars
		fem.save= False
		fem.select_feat= True
		fem.selfeatids= selcols
		fem.negative_pix_fract_thr= self.negative_pix_fract_thr
		fem.bad_pix_fract_thr= self.bad_pix_fract_thr
			
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
			self.feat_colors_dict= OrderedDict()

			for dictlist in colfeat_dict_list:
				for d in dictlist:
					keys= list(d.keys())
					nvars= len(keys)-2
					featvars= []
					print("keys")
					print(keys)
					print("nvars")
					print(nvars)

					sname= d["sname"]
					classid= d["id"]

					if sname in self.feat_colors_dict:
						logger.warn("[PROC %d] Source %s is already present in data dict, overwriting it ..." % (procId, sname))
					self.feat_colors_dict[sname]= OrderedDict()
					self.feat_colors_dict[sname]["sname"]= sname	
						
					for i in range(nvars):
						var_index= i+1 # exclude sname
						varname= keys[var_index]
						var= d[varname]
						featvars.append(var)
						print("Adding feat %s ..." % (varname))
						self.feat_colors_dict[sname][varname]= var

					self.feat_colors_dict[sname]["id"]= classid
					self.feat_colors_snames.append(sname)
					self.feat_colors_classids.append(classid)
					self.feat_colors.append(featvars)
					
			self.feat_colors= np.array(self.feat_colors)

			print("feat_colors_snames")
			print(self.feat_colors_snames)
			print("feat_colors_classids")
			print(self.feat_colors_classids)
			print("feat colors")
			print(self.feat_colors)	
			print("feat_colors_dict")
			print(self.feat_colors_dict)

		return 0

	#========================================
	#==   COMPUTE SPECTRAL INDEX
	#========================================
	def compute_spectral_index(self):
		""" Extract spectral index measurement """

		# - Create spectral index calculator obj
		#   NB: All PROC
		sic= SpectralIndexTTCalculator()
		sic.img_freqs= self.alpha_img_freqs
		sic.alpha_rcoeff_thr= self.alpha_rcoeff_thr
		sic.negative_pix_fract_thr= self.negative_pix_fract_thr
		sic.bad_pix_fract_thr= self.bad_pix_fract_thr
		sic.save= self.save_spectral_index_data
		sic.outfile= self.outfile_feat_alpha

		print("self.alpha_img_group_1")
		print(self.alpha_img_group_1)
		print("self.alpha_img_group_2")
		print(self.alpha_img_group_2)
			
		logger.info("[PROC %d] Measuring spectral index from cutout data (nsources=%d) ..." % (procId, len(self.datalist_radio_mask_proc)))
		if sic.run_from_datalist(self.datalist_radio_mask_proc, self.alpha_img_group_1, self.alpha_img_group_2)<0:
			logger.error("[PROC %d] Failed to measure spectral index (see logs)!" % (procId))
			return -1

		param_dict_list= sic.par_dict_list

		# - Merge parameters found by each proc
		if comm is None:
			alphafeat_dict_list= param_dict_list
		else:
			logger.info("[PROC %d] Gathering spectral index features ... " % (procId))
			alphafeat_dict_list= comm.gather(param_dict_list, root=MASTER)
		
		if procId==MASTER:
			print("alphafeat_dict_list")
			print(alphafeat_dict_list)
	
			# - Set col feat data
			self.feat_alpha= []
			self.feat_alpha_snames= []
			self.feat_alpha_classids= []
			self.feat_alpha_dict= OrderedDict()

			for dictlist in alphafeat_dict_list:
				for d in dictlist:
					keys= list(d.keys())
					nvars= len(keys)-2
					featvars= []

					sname= d["sname"]
					classid= d["id"]
				
					if sname in self.feat_alpha_dict:
						logger.warn("[PROC %d] Source %s is already present in data dict, overwriting it ..." % (procId, sname))
					self.feat_alpha_dict[sname]= OrderedDict()
					self.feat_alpha_dict[sname]["sname"]= sname	

					for i in range(nvars):
						var_index= i+1 # exclude sname
						varname= keys[var_index]
						var= d[varname]
						featvars.append(var)
						print("Adding feat %s ..." % (varname))
						self.feat_alpha_dict[sname][varname]= var

					self.feat_alpha_dict[sname]["id"]= classid
						
					self.feat_alpha_snames.append(sname)
					self.feat_alpha_classids.append(classid)
					self.feat_alpha.append(featvars)
					
			self.feat_alpha= np.array(self.feat_alpha)

			print("feat_alpha_snames")
			print(self.feat_alpha_snames)
			print("feat_alpha_classids")
			print(self.feat_alpha_classids)
			print("feat_alpha")
			print(self.feat_alpha)
			print("feat_alpha_dict")
			print(self.feat_alpha_dict)

		return 0

	#========================================
	#==   MERGE FEATURES
	#========================================
	def merge_features(self):
		""" Merge feature data """
			
		# - Merge features if PROC 0
		if procId==MASTER:
			
			mergefeat_status= 0

			# - Set feature data dict to be merged
			fm= FeatMerger()
		
			mergeable_dicts= []
			if self.feat_colors_dict and self.feat_colors_dict is not None:
				logger.info("[PROC %d] Adding color features for merging ..." % (procId))
				mergeable_dicts.append(self.feat_colors_dict)

			if self.feat_alpha_dict and self.feat_alpha_dict is not None:
				logger.info("[PROC %d] Adding spectral index features for merging ..." % (procId))
				mergeable_dicts.append(self.feat_alpha_dict)

			# - Merge features
			if mergeable_dicts:
				mergefeat_status= fm.run_from_dictlist(mergeable_dicts, outfile=self.outfile_feat_merged)
			else:
				logger.error("[PROC %d] No feature data to be merged!" % (procId))
				mergefeat_status= -1
				
			# - Set merged feat data
			if mergefeat_status==0:
				allfeat_dict_list= fm.par_dict_list

				self.feat_all= []
				self.feat_all_snames= []
				self.feat_all_classids= []
				self.feat_all_dict= OrderedDict()

				for d in allfeat_dict_list:
					keys= list(d.keys())
					nvars= len(keys)-2
					featvars= []

					sname= d["sname"]
					classid= d["id"]
				
					if sname in self.feat_all_dict:
						logger.warn("[PROC %d] Source %s is already present in data dict, overwriting it ..." % (procId, sname))
					self.feat_all_dict[sname]= OrderedDict()
					self.feat_all_dict[sname]["sname"]= sname	

					for i in range(nvars):
						var_index= i+1 # exclude sname
						varname= keys[var_index]
						var= d[varname]
						featvars.append(var)
						print("Adding feat %s ..." % (varname))
						self.feat_all_dict[sname][varname]= var

					self.feat_all_dict[sname]["id"]= classid
						
					self.feat_all_snames.append(sname)
					self.feat_all_classids.append(classid)
					self.feat_all.append(featvars)
					
				self.feat_all= np.array(self.feat_all)

				print("snames")
				print(self.feat_all_snames)
				print("classids")
				print(self.feat_all_classids)
				print("feat all")
				print(self.feat_all)


		else:
			mergefeat_status= 0
		
		if comm is not None:
			mergefeat_status= comm.bcast(mergefeat_status, root=MASTER)

		if mergefeat_status<0:
			logger.error("[PROC %d] Error on merging feature data, exit!" % (procId))
			return -1

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

			sclass.find_outliers= self.find_outliers
			sclass.outlier_modelfile= self.modelfile_outlier
			sclass.outlier_thr= self.anomaly_thr
			sclass.outlier_max_samples= self.max_samples
			sclass.outlier_max_features= self.max_features
			sclass.save_outlier= self.save_outlier
			sclass.outlier_outfile= self.outfile_outlier
	
			# - Run classification
			sclass_status= sclass.run_predict(
				#data=self.feat_colors, class_ids=self.feat_colors_classids, snames=self.feat_colors_snames,
				data=self.feat_all, class_ids=self.feat_all_classids, snames=self.feat_all_snames,
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

	#=====================================
	#==   AUTOENCODER RECONSTRUCTION
	#=====================================
	def run_ae_reconstruction(self, datalist):
		""" Run AE reconstruction """

		if procId==MASTER:
			aereco_status= 0

			# - Create data loader
			dl= DataLoader(filename=datalist)

			# - Read datalist	
			logger.info("[PROC %d] Reading datalist %s ..." % (procId, datalist))
			dataread_status= dl.read_datalist()

			if dataread_status<0:
				logger.error("[PROC %d] Failed to read input datalist %s" % (procId, datalist))
				aereco_status= -1

			else:
				# - Run AE reco
				logger.info("[PROC %d] Running autoencoder classifier reconstruction ..." % (procId))
				ae= FeatExtractorAE(dl)
				ae.resize= self.resize_img
				ae.set_image_size(self.nx, self.ny)
				ae.normalize= self.normalize_img
				ae.scale_to_abs_max= self.scale_img_to_abs_max
				ae.scale_to_max= self.scale_img_to_max
				ae.log_transform_img= self.log_transform_img
				ae.scale_img= self.scale_img
				ae.scale_img_factors= self.scale_img_factors
				ae.standardize_img= self.standardize_img
				ae.img_means= self.img_means
				ae.img_sigmas= self.img_sigmas
				ae.chan_divide= self.img_chan_divide
				ae.chan_mins= self.img_chan_mins
				ae.erode= self.img_erode
				ae.erode_kernel= self.img_erode_kernel
				ae.add_channorm_layer= self.add_channorm_layer

				aereco_status= ae.reconstruct_data(
					self.modelfile_encoder, self.weightfile_encoder, 
					self.modelfile_decoder, self.weightfile_decoder,
					winsize= self.winsize,
					outfile_metrics=self.outfile_aerecometrics,
					save_imgs= False
				)

		else:
			aereco_status= 0

		if comm is not None:
			aereco_status= comm.bcast(aereco_status, root=MASTER)

		if aereco_status<0:
			logger.error("[PROC %d] Failed to run autoencoder reconstruction on data %s, exit!" % (procId, datalist))
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

		self.outfile_aerecometrics= os.path.join(self.jobdir_sclass, "aereco_metrics.dat")
		self.outfile_feat_alpha= os.path.join(self.jobdir_sclass, "features_alpha.csv")
		self.outfile_feat_merged= os.path.join(self.jobdir_sclass, "features_merged.dat")

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
		if self.add_spectral_index:
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

		logger.info("[PROC %d] #surveys=%d" % (procId, self.nsurveys))


		# - Create scutout config class (radio multi)
		if self.add_spectral_index:
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

			logger.info("[PROC %d] #surveys_radio=%d" % (procId, self.nsurveys_radio))

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
		
		# - Create radio multi cutouts?
		if self.add_spectral_index:
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
		if self.add_spectral_index:
			logger.info("[PROC %d] Computing spectral index ..." % (procId))

			if self.compute_spectral_index()<0:
				logger.error("[PROC %d] Failed to compute spectral index ..." % (procId))
				return -1

		# - Run AE reconstruction
		#   NB: Done by PROC 0
		if self.run_aereco:
			if self.run_ae_reconstruction(self.datalist_mask_file)<0:
				logger.error("[PROC %d] Failed to run AE reconstruction on data %s ..." % (procId, self.datalist_mask_file))
				return -1

		# - Concatenate features
		#   NB: Done by PROC 0
		logger.info("[PROC %d] Concatenating all extracted features ..." % (procId))
		if self.merge_features()<0:
			logger.error("[PROC %d] Failed to concatenate available features extracted ..." % (procId))
			return -1

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

