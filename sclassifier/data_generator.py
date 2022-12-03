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
from collections import Counter
import json

## KERAS MODULES
from tensorflow.keras.utils import to_categorical

## ASTROPY MODULES 
from astropy.io import ascii
from astropy.stats import sigma_clipped_stats

from sclassifier.data_loader import SourceData

##############################
##     GLOBAL VARS
##############################
from sclassifier import logger


##############################
##     DATA GENERATOR
##############################
class DataGenerator(object):

	""" Read data from disk and provide it to the network

			Arguments:
				- datalist: Filelist (json) with input data
				
	"""
	
	def __init__(self, filename, preprocessor=None):
		""" Return a DataLoader object """

		# - Input data
		self.datalistfile= filename
		self.datalist= {}
		self.datasize= 0
		self.classids= []
		self.classfract_map= {}
		self.labels= []
		self.snames= []
		self.nchannels= 0
		
		# - Pre-processor
		self.preprocessor= preprocessor


	#############################
	##     READ DATALIST
	#############################
	def read_datalist(self):
		""" Read json filelist """

		# - Read data list
		self.datalist= {}
		try:
			with open(self.datalistfile) as fp:
				self.datalist= json.load(fp)
		except Exception as e:
			logger.error("Failed to read data filelist %s!" % self.datalistfile)
			return -1

		# - Check number of channels per image
		nchannels_set= set([len(item["filepaths"]) for item in self.datalist["data"]])
		if len(nchannels_set)!=1:
			logger.warn("Number of channels in each object instance is different (len(nchannels_set)=%d!=1)!" % (len(nchannels_set)))
			print(nchannels_set)
			return -1
		
		self.nchannels= list(nchannels_set)[0]

		# - Inspect data (store number of instances per class, etc)
		self.datasize= len(self.datalist["data"])
		self.labels= [item["label"] for item in self.datalist["data"]]
		self.snames= [item["sname"] for item in self.datalist["data"]]
		self.classids= 	[item["id"] for item in self.datalist["data"]]
		self.classfract_map= dict(Counter(self.classids).items())

		logger.info("#%d objects in dataset" % self.datasize)

		return 0

	#############################
	##     READ IMAGE DATA
	#############################
	def read_data(self, index, read_crop=False, crop_size=32, crop_range=None):	
		""" Read data at given index """

		# - Check index
		if index<0 or index>=self.datasize:
			logger.error("Invalid index %d given!" % index)
			return None

		# - Read source filelist
		logger.debug("Reading source image data %d ..." % index)
		d= self.datalist["data"][index]
		sdata= SourceData()
		if sdata.set_from_dict(d)<0:
			logger.error("Failed to set source image data %d!" % index)
			return None

		# - Read source image data
		status= 0
		if read_crop:
			if crop_range is None:
				status= sdata.read_random_img_crops(crop_size)
			else:
				ixmin= crop_range[0]
				ixmax= crop_range[1]
				iymin= crop_range[2]
				iymax= crop_range[3]
				status= sdata.read_img_crops(ixmin, ixmax, iymin, iymax)
		else:
			status= sdata.read_imgs()
				
		if status<0:
			logger.error("Failed to read source images %d!" % index)
			return None

		if sdata.img_cube is None:
			logger.error("Source image %d read data cube is None!" % index)
			return None

		# - Apply pre-processing?
		if self.preprocessor is not None:
			logger.info("Apply pre-processing ...")
			data_proc= self.preprocessor(sdata.img_cube)
			if data_proc is None:
				logger.error("Failed to pre-process source image data %d!" % index)
				return None
			sdata.img_cube= data_proc

		# - Check data cube integrity
		logger.info("Check bad pixels ...")
		has_bad_pixs= sdata.has_bad_pixels(check_fract=False, thr=0)
		if has_bad_pixs:
			logger.warn("Source image data %d has bad pixels!" % index)	
			return None

		logger.info("Returning sdata ...")

		return sdata


	

	#####################################
	##     GENERATE CNN TRAIN DATA
	#####################################
	def generate_cnn_data(self, batch_size=32, shuffle=True, read_crop=False, crop_size=32, classtarget_map={}, nclasses=7):
		""" Generator function for CNN classification task """

		nb= 0
		data_index= -1
		data_indexes= np.arange(0, self.datasize)
		target_ids= []

		logger.info("Starting data generator ...")

		while True:
			try:

				if nb==0:
					logger.debug("Starting new batch ...")

				# - Generate random data index and read data at this index
				data_index = (data_index + 1) % self.datasize
				if shuffle:
					data_index= np.random.choice(data_indexes)

				logger.debug("Reading data at index %d (batch %d/%d) ..." % (data_index, nb, batch_size))
				
				sdata= self.read_data(data_index, read_crop, crop_size)
				if sdata is None:
					logger.warn("Failed to read source data at index %d, skip to next ..." % data_index)
					continue

				data_shape= sdata.img_cube.shape
				inputs_shape= (batch_size,) + data_shape
				logger.debug("Data %d shape=(%d,%d,%d)" % (data_index, data_shape[0], data_shape[1], data_shape[2]))
				
				# - Set class targets
				class_id= sdata.id
				target_id= class_id
				if classtarget_map:
					target_id= classtarget_map[class_id]
				
				# - Initialize return data
				if nb==0:
					inputs= np.zeros(inputs_shape, dtype=np.float32)
					target_ids= []
				
				# - Update inputs
				inputs[nb]= sdata.img_cube
				target_ids.append(target_id)
				nb+= 1

				# - Return data if number of batch is reached and restart the batch
				if nb>=batch_size:
					logger.debug("Batch size (%d) reached, yielding generated data of size (%d,%d,%d,%d) ..." % (nb,inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3]))
					output_targets= to_categorical(np.array(target_ids), num_classes=nclasses)
					yield inputs, output_targets

					nb= 0

			except (GeneratorExit):
				logger.info("Data generator complete execution ...")
				raise
			except (KeyboardInterrupt):
				logger.warn("Keyboard exception catched while generating data...")
				raise
			except Exception as e:
				logger.warn("Exception catched while generating data (err=%s) ..." % str(e))
				raise

	#####################################
	##     GENERATE CAE TRAIN DATA
	#####################################
	def generate_cae_data(self, batch_size=32, shuffle=True, read_crop=False, crop_size=32):
		""" Generator function for CAE task """
	
		nb= 0
		data_index= -1
		data_indexes= np.arange(0,self.datasize)

		logger.info("Starting CAE data generator ...")

		while True:
			try:

				if nb==0:
					logger.debug("Starting new batch ...")

				# - Generate random data index and read data at this index
				data_index = (data_index + 1) % self.datasize
				if shuffle:
					data_index= np.random.choice(data_indexes)

				sdata= self.read_data(data_index, read_crop, crop_size)
				if sdata is None:
					logger.warn("Failed to read source data at index %d, skip to next ..." % data_index)
					continue

				data_shape= sdata.img_cube.shape
				inputs_shape= (batch_size,) + data_shape
				
				# - Initialize return data
				if nb==0:
					inputs= np.zeros(inputs_shape, dtype=np.float32)
				
				# - Update inputs
				inputs[nb]= sdata.img_cube
				nb+= 1

				# - Return data if number of batch is reached and restart the batch
				if nb>=batch_size:
					yield inputs, inputs
					nb= 0

			except (GeneratorExit):
				logger.info("Data generator complete execution ...")
				raise
			except (KeyboardInterrupt):
				logger.warn("Keyboard exception catched while generating data...")
				raise
			except Exception as e:
				logger.warn("Exception catched while generating data (err=%s) ..." % str(e))
				raise
			


	#####################################
	##     GENERATE SIMCLR TRAIN DATA
	#####################################
	def generate_simclr_data(self, batch_size=32, shuffle=True, read_crop=False, crop_size=32):
		""" Generator function for SimCLR task """
	
		nb= 0
		data_index= -1
		data_indexes= np.arange(0,self.datasize)

		logger.info("Starting data generator ...")

		while True:
			try:

				if nb==0:
					logger.debug("Starting new batch ...")

				# - Generate random data index and pairs of augmented data for SimCLR
				data_index = (data_index + 1) % self.datasize
				if shuffle:
					data_index= np.random.choice(data_indexes)

				if read_crop:# NB: must read the same crop range for the data pair
					sdata_1= self.read_data(data_index, read_crop=True, crop_size=crop_size, crop_range=None)
					crop_range= (sdata_1.ixmin, sdata_1.ixmax, sdata_1.iymin, sdata_1.iymax)
					sdata_2= self.read_data(data_index, read_crop=True, crop_size=crop_size, crop_range=crop_range)
				else:
					sdata_1= self.read_data(data_index)
					sdata_2= self.read_data(data_index)
				
				if sdata_1 is None or sdata_2 is None:
					logger.warn("Failed to read source data pair at index %d!" % (data_index))
					return None

				data_shape= sdata_1.img_cube.shape
				inputs_shape= (batch_size,) + data_shape
				
				# - Initialize return data
				if nb==0:
					# - The ref implementation (https://github.com/mwdhont/SimCLRv1-keras-tensorflow/blob/master/DataGeneratorSimCLR.py)
					#   uses a dimension (2*batch, 1, ny, nx, nchan), so that returned inputs is a list of len(2*batch) and item passed to encoder has shape (1,ny,nx,nchan) (NB: batch size=1)
					inputs_simclr_shape= (2*batch_size, 1) + data_shape # original ref
					inputs_simclr= np.empty(inputs_simclr_shape, dtype=np.float32)
					labels_ab_aa = np.zeros((batch_size, 2 * batch_size))
					labels_ba_bb = np.zeros((batch_size, 2 * batch_size))

				# - Update inputs
				# - The ref implementation (https://github.com/mwdhont/SimCLRv1-keras-tensorflow/blob/master/DataGeneratorSimCLR.py)
				#   shuffles the position of augmented image pair
				inputs_simclr[nb]= sdata_1.img_cube
				inputs_simclr[nb + batch_size]= sdata_2.img_cube
				labels_ab_aa[nb, nb] = 1
				labels_ba_bb[nb, nb] = 1

				nb+= 1

				# - Return data if number of batch is reached and restart the batch
				if nb>=batch_size:
					y= np.concatenate([labels_ab_aa, labels_ba_bb], 1)
					yield list(inputs_simclr), y # original implementation: returns a list (len=2xbatch_size) of arrays of shape (1, ny, nx, nchan). Each Input layer takes one list entry as input.

					nb= 0

			except (GeneratorExit):
				logger.info("Data generator complete execution ...")
				raise
			except (KeyboardInterrupt):
				logger.warn("Keyboard exception catched while generating data...")
				raise
			except Exception as e:
				logger.warn("Exception catched while generating data (err=%s) ..." % str(e))
				raise
			

	#####################################
	##     GENERATE TRAIN DATA
	#####################################
	def generate_data(self, batch_size=32, shuffle=True, read_crop=False, crop_size=32):
		""" Generator function reading nsamples images from disk and returning to caller """
	
		nb= 0
		data_index= -1
		data_indexes= np.arange(0,self.datasize)
		
		logger.info("Starting data generator ...")

		while True:
			try:

				if nb==0:
					logger.debug("Starting new batch ...")

				# - Generate random data index and read data at this index
				data_index = (data_index + 1) % self.datasize
				if shuffle:
					data_index= np.random.choice(data_indexes)

				logger.debug("Reading data at index %d (batch %d/%d) ..." % (data_index,nb, batch_size))
				
				sdata= self.read_data(data_index, read_crop, crop_size)
				if sdata is None:
					logger.warn("Failed to read source data at index %d, skip to next ..." % data_index)
					continue

				data_shape= sdata.img_cube.shape
				inputs_shape= (batch_size,) + data_shape

				# - Initialize return data
				if nb==0:
					inputs= np.zeros(inputs_shape, dtype=np.float32)
					
				# - Update inputs
				inputs[nb]= sdata.img_cube
				nb+= 1

				# - Return data if number of batch is reached and restart the batch
				if nb>=batch_size:
					logger.info("Returning generator data ...")
					yield inputs, sdata
					
					nb= 0

			except (GeneratorExit):
				logger.info("Data generator complete execution ...")
				raise
			except (KeyboardInterrupt):
				logger.warn("Keyboard exception catched while generating data...")
				raise
			except Exception as e:
				logger.warn("Exception catched while generating data (err=%s) ..." % str(e))
				raise
			


