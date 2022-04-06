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
##     DataChecker CLASS
##############################
class DataChecker(object):
	""" Class to store data checker """
	
	def __init__(self):
		""" Return a DataChecker object """

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

		# - Quality cuts
		self.negative_pix_fract_thr= 0.9
		self.bad_pix_fract_thr= 0.05

		# - Output data
		self.nvars_out= 0
		self.param_dict_list= []
		self.output= "datacheck.dat"

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
		if self.nsamples_max<0 or self.nsamples_max>=self.nsamples:
			self.nsamples_max= self.nsamples

		logger.info("#%d/%d samples to be read ..." % (self.nsamples_max, self.nsamples))

		#===========================
		#==   SET GENERATOR
		#===========================
		# - Set data generator
		logger.info("Running data loader ...")
		self.data_generator= self.dl.data_generator(
			batch_size=1, 
			shuffle=self.shuffle,
			resize=self.resize, nx=self.nx, ny=self.ny, 	
			normalize=self.normalize, scale_to_abs_max=self.scale_to_abs_max, scale_to_max=self.scale_to_max,
			augment=self.augment,
			log_transform=self.log_transform,
			scale=self.scale, scale_factors=self.scale_factors,
			standardize=self.standardize, means=self.img_means, sigmas=self.img_sigmas,
			chan_divide=self.chan_divide, chan_mins=self.chan_mins,
			erode=self.erode, erode_kernel=self.erode_kernel,
			retsdata=True
		)	

		return 0

	#############################
	##     COMPUTE PARS
	#############################
	def __compute_pars(self, data, sname, classid):
		""" Compute source image quality pars """

		# - Init data dict
		param_dict= collections.OrderedDict()
		param_dict["sname"]= sname

		# - Find ref channel mask
		nchannels= data.shape[3]
		cond= np.logical_and(data[0,:,:,self.refch]!=0, np.isfinite(data[0,:,:,self.refch]))

		is_bad_data= False
		self.nvars_out= 0

		for i in range(nchannels):
			data_2d= data[0,:,:,i]
			data_1d= data_2d[cond] # pixel in ref band mask
			n= data_1d.size
			n_bad= np.count_nonzero(np.logical_or(~np.isfinite(data_1d), data_1d==0))
			n_neg= np.count_nonzero(data_1d<0)
			f_bad= float(n_bad)/float(n)
			f_negative= float(n_neg)/float(n)
			same_values= int(data_min==data_max)

			is_bad_ch_data= (
				f_negative>=self.negative_pix_fract_thr or
				f_bad>=self.bad_pix_fract_thr or
				same_values==1
			)
			if is_bad_ch_data:
				is_bad_data= True
	
			# - Fill dict
			par_name= "equalPixValues_ch" + str(i+1)
			param_dict[par_name]= same_values
			self.nvars_out+= 1

			par_name= "badPixFract_ch" + str(i+1)
			param_dict[par_name]= f_bad
			self.nvars_out+= 1

			par_name= "negativePixFract_ch" + str(i+1)		
			param_dict[par_name]= f_negative
			self.nvars_out+= 1
	
			par_name= "isBad_ch" + str(i+1)
			param_dict[par_name]= int(is_bad_ch_data)
			self.nvars_out+= 1

		param_dict["isBadData"]= int(is_bad_data)
		self.nvars_out+= 1

		param_dict["id"]= classid

		return param_dict

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

		# - Set data generator
		logger.info("Read data list %s ..." % (self.datalist))
		self.__read_data(datalist)

		# - Loop over source images and compute quality flags
		while True:
			try:
				# - Read data
				data, sdata= next(self.data_generator)
				sname= sdata.sname
				label= sdata.label
				classid= sdata.id
				img_counter+= 1
				logger.info("Processing image no. %d (name=%s, label=%s) ..." % (img_counter, sname, label))

				# - Compute data parameters				
				param_dict= self.__compute_pars(data, sname, classid)
				if param_dict is None or not param_dict:
					logger.warn("Feature dict for source %d (name=%s, label=%s) is empty or None, skip it ..." % (img_counter, sname, label))
				else:
					self.param_dict_list.append(par_dict)

				# - Stop generator
				if img_counter>=self.nsamples_max:
					logger.info("Sample size (%d) reached, stop generation..." % self.nsamples_max)
					break

			except (GeneratorExit, KeyboardInterrupt):
				logger.info("Stop loop (keyboard interrupt) ...")
				break
			except Exception as e:
				logger.warn("Stop loop (exception catched %s) ..." % str(e))
				break

		# - Save output data
		if self.__save()<0:
			logger.warn("Failed to save output data to file %s!" % (self.outfile))
			return -1

		return 0
