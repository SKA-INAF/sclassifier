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
import pickle

## SCLASSIFIER MODULES
from .utils import Utils
from .data_loader import DataLoader

##############################
##     GLOBAL VARS
##############################
from sclassifier_vae import logger

##############################
##     DataAERecoChecker CLASS
##############################
class DataAERecoChecker(object):
	""" Class to store data AE reconstruction checker """
	
	def __init__(self):
		""" Return a DataAERecoChecker object """

		# - Data generator options
		self.datalist= ""
		self.dl= None
		self.nsamples= -1
		self.nsamples_max= -1

		# - Data pre-processing options
		self.shuffle= False
		self.resize= False
		self.nx= 64
		self.ny= 64	
		self.normalize= False
		self.scale_to_abs_max= False
		self.scale_to_max= False
		self.augment= False
		self.log_transform= False
		self.scale= False
		self.scale_factors= []
		self.standardize= False
		self.img_means= []
		self.img_sigmas= []
		self.chan_divide= False
		self.chan_mins= []
		self.erode= False
		self.erode_kernel= False
		self.refch= 0

		# - AE options
		self.encoder_model= ""
		self.encoder_weights= ""
		self.decoder_model= "" 
		self.decoder_weights= ""
		self.winsize= 3
		self.save_imgs= False

		# - Quality cuts
		self.metric_name= "ssim_mean_ch"
		self.reco_thr= 0.5
		
		# - Output data
		self.nvars_out= 0
		self.param_dict_list= []
		self.output= "reco_metrics.dat"

	#############################
	##     READ DATA
	#############################
	def __read_data(self, datalist):
		""" Set data generator from loader """

		#===========================
		#==   READ DATA
		#===========================
		# - Create data loader
		self.datalist= datalist
		self.dl= DataLoader(filename=datalist)

		# - Read datalist	
		logger.info("Reading datalist %s ..." % (datalist))
		if self.dl.read_datalist()<0:
			logger.error("Failed to read input datalist %s!" % (datalist))
			return -1

		# - Set max source to be read
		source_labels= self.dl.snames
		self.nsamples= len(source_labels)
		#if self.nsamples_max<0 or self.nsamples_max>=self.nsamples:
		#	self.nsamples_max= self.nsamples

		#logger.info("#%d/%d samples to be read ..." % (self.nsamples_max, self.nsamples))
		logger.info("#%d samples to be read ..." % (self.nsamples))

		#===========================
		#==   SET GENERATOR
		#===========================
		# - Set data generator
		#logger.info("Running data loader ...")
		#self.data_generator= self.dl.data_generator(
		#	batch_size=1, 
		#	shuffle=self.shuffle,
		#	resize=self.resize, nx=self.nx, ny=self.ny, 	
		#	normalize=self.normalize, scale_to_abs_max=self.scale_to_abs_max, scale_to_max=self.scale_to_max,
		#	augment=self.augment,
		#	log_transform=self.log_transform,
		#	scale=self.scale, scale_factors=self.scale_factors,
		#	standardize=self.standardize, means=self.img_means, sigmas=self.img_sigmas,
		#	chan_divide=self.chan_divide, chan_mins=self.chan_mins,
		#	erode=self.erode, erode_kernel=self.erode_kernel,
		#	retsdata=True
		#)	

		

		return 0

	#############################
	##     COMPUTE PARS
	#############################
	def __fill_metric_data(self):
		""" Compute source image quality pars """

		# - Read metric data and apply thresholds
		colprefix= ""
		d= Utils.read_feature_data_dict(self.output, colprefix=colprefix)
		if not d or d is None:
			logger.error("Failed to read reco metrics file %s!" % (self.output))
			return -1

		nentries= len(d.keys())
		firstitem= next(iter(d.items()))
		nvars= len(firstitem[1].keys()) - 2
		nmetrics_per_band= 5
		nbands= int(float(nvars)/float(nmetrics_per_band))
		
		# - Init data dict
		self.param_dict_list= []
		
		
		for i in range(nentries):
			param_dict= collections.OrderedDict()
			param_dict["sname"]= sname

			is_bad_reco= False
			self.nvars_out= 0

			for j in range(nbands):
				varname= self.metric_name + str(j+1)
				metric= param_dict[varname]
				param_dict[varname]= metric
				self.nvars_out+= 1
				
				if metric<self.reco_thr:
					is_bad_reco= True
				
			param_dict["isBadAEReco"]= int(is_bad_reco)
			self.nvars_out+= 1

			param_dict["id"]= classid

			self.param_dict_list.append(param_dict)

		return 0

	#############################
	##     RUN AE RECO
	#############################
	def __run_aereco(self):
		""" Run AE reconstruction """

		# - Set FeatExtractorAE class
		ae= FeatExtractorAE(self.dl)
		ae.set_image_size(self.nx, self.ny)
		ae.normalize= self.normalize
		ae.scale_to_abs_max= self.scale_to_abs_max
		ae.scale_to_max= self.scale_to_max
		ae.log_transform_img= self.log_transform
		ae.scale_img= self.scale
		ae.scale_img_factors= self.scale_factors
		ae.standardize_img= self.standardize
		ae.img_means= self.img_means
		ae.img_sigmas= self.img_sigmas
		ae.chan_divide= self.chan_divide
		ae.chan_mins= self.chan_mins
		ae.erode= self.erode
		ae.erode_kernel= self.erode_kernel
		ae.add_channorm_layer= self.add_channorm_layer

		# - Run AE reco
		status= ae.reconstruct_data(
			self.modelfile_encoder, self.weightfile_encoder, 
			self.modelfile_decoder, self.weightfile_decoder,
			winsize= self.winsize,
			outfile_metrics=self.output,
			save_imgs= self.save_imgs
		)

		if status<0:
			logger.error("AE reconstruction failed (see logs)!")
			return -1

		return 0

	#############################
	##     SAVE
	#############################
	def __save(self):
		""" Save data to file """

		if self.param_dict_list:
			logger.info("Saving parameter file %s ..." % (self.outfile))
			parnames = self.param_dict_list[0].keys()
			
			with open(self.outfile, 'w') as fp:
				fp.write("# ")
				dict_writer = csv.DictWriter(fp, parnames)
				dict_writer.writeheader()
				dict_writer.writerows(self.par_dict_list)
		else:
			logger.warn("Parameter dict list is empty, no files will be written!")
			return -1

		return 0

	#############################
	##     RUN
	#############################
	def run(self, datalist):
		""" Run data checker """

		# - Init 
		self.param_dict_list= []

		# - Read data
		logger.info("Read data list %s ..." % (self.datalist))
		self.__read_data(datalist)

		# - Run AE reco
		logger.info("Running autoencoder reconstruction ...")
		if self.__run_aereco()<0:
			logger.error("AE reconstruction failed!")
			return -1
		
		# - Select AE reco data
		logger.info("Reading and thresholding AE reco metrics ...")
		if self.__fill_metric_data()<0:
			logger.error("Failed to read and threshold AE reco metrics!")
			return -1			

		# - Save output data
		logger.info("Saving output to file ...")
		if self.__save()<0:
			logger.warn("Failed to save output data to file %s!" % (self.outfile))
			return -1

		return 0


