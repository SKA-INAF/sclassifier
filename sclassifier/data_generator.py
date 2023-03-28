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

## SKLEARN MODULES
from sklearn.preprocessing import MultiLabelBinarizer

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
	##     DISABLE AUGMENTATION
	#############################
	def disable_augmentation(self):
		""" Disable augmentation """

		if self.preprocessor is None:
			logger.warn("Pre-processor is None, nothing will be done...")
			return -1

		self.preprocessor.disable_augmentation()		

		return 0

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
			logger.error("Invalid index %d given!" % (index))
			return None

		# - Read source filelist
		logger.debug("Reading source image data at index %d ..." % (index))
		d= self.datalist["data"][index]
		sdata= SourceData()
		if sdata.set_from_dict(d)<0:
			logger.error("Failed to set source image data at index %d!" % (index))
			return None

		sname= sdata.sname
		label= sdata.label
		classid= sdata.id

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
			logger.error("Failed to read source image at index %d (sname=%s, label=%s, classid=%d)!" % (index, sname, label, classid))
			return None

		if sdata.img_cube is None:
			logger.error("Source image data cube at index %d (sname=%s, label=%s, classid=%d) is None!" % (index, sname, label, classid))
			return None

		# - Apply pre-processing?
		if self.preprocessor is not None:
			logger.debug("Apply pre-processing ...")
			data_proc= self.preprocessor(sdata.img_cube)
			if data_proc is None:
				logger.error("Failed to pre-process source image data at index %d (sname=%s, label=%s, classid=%d)!" % (index, sname, label, classid))
				return None
			sdata.img_cube= data_proc

		# - Check data cube integrity
		logger.debug("Check bad pixels ...")
		has_bad_pixs= sdata.has_bad_pixels(check_fract=False, thr=0)
		if has_bad_pixs:
			logger.warn("Source image data at index %d (sname=%s, label=%s, classid=%d) has bad pixels!" % (index, sname, label, classid))	
			return None

		return sdata


	

	#####################################
	##     GENERATE CNN TRAIN DATA
	#####################################
	def generate_cnn_data(self, batch_size=32, shuffle=True, read_crop=False, crop_size=32, classtarget_map={}, nclasses=7, balance_classes=False, class_probs={}):
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
				#   NB: If id & label are a list treat as multilabel problem
				class_id= sdata.id
				class_name= sdata.label
				target_id= class_id
				multilabel= (isinstance(class_id, list)) and (isinstance(class_name, list))
				if classtarget_map:
					if multilabel:
						target_id= [classtarget_map[item] for item in class_id]
					else:
						target_id= classtarget_map[class_id]

				# - Apply class rebalancing?
				if balance_classes and class_probs:
					accept= True

					if multilabel:
						for item in class_name:
							prob= class_probs[item]
							r= random.uniform(0, 1)
							if r<prob:
								accept= False
								break
					else:
						prob= class_probs[class_name]
						r= random.uniform(0, 1)
						accept= r<prob
						
					if not accept:
						continue
					
				# - Initialize return data
				if nb==0:
					inputs= np.zeros(inputs_shape, dtype=np.float32)
					target_ids= []
					class_ids= []
				
				# - Update inputs
				inputs[nb]= sdata.img_cube
				target_ids.append(target_id)
				class_ids.append(class_id)
				nb+= 1

				# - Return data if number of batch is reached and restart the batch
				if nb>=batch_size:
					# - Compute class abundances in batch
					if not multilabel:
						class_counts= np.bincount(class_ids)
						class_fracts= class_counts/len(class_ids)
						class_counts_str= ' '.join([str(x) for x in class_counts])
						class_fracts_str= ' '.join([str(x) for x in class_fracts])
						logger.debug("Class counts/fract in batch: counts=[%s], fract=[%s]" % (class_counts_str, class_fracts_str))

					# - Return data
					logger.debug("Batch size (%d) reached, yielding generated data of size (%d,%d,%d,%d) ..." % (nb,inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3]))
					if multilabel:
						mlb = MultiLabelBinarizer(classes=np.arange(0,nclasses))
						output_targets= mlb.fit_transform(target_ids).astype('float32')
					else:
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
	def generate_cae_data(self, batch_size=32, shuffle=True, read_crop=False, crop_size=32, balance_classes=False, class_probs={}):
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

				#print("datasize")
				#print(self.datasize)
				#print("batch_size")
				#print(batch_size)
				#print("data_shape")
				#print(data_shape)
				#print("inputs_shape")
				#print(inputs_shape)
				
				# - Apply class rebalancing?
				class_id= sdata.id
				class_name= sdata.label
				if balance_classes and class_probs:					
					#prob= class_probs[class_id]
					prob= class_probs[class_name]
					r= random.uniform(0, 1)
					accept= r<prob
					if not accept:
						continue

				# - Initialize return data
				if nb==0:
					inputs= np.zeros(inputs_shape, dtype=np.float32)
					class_ids= []
				
				# - Update inputs
				try:
					inputs[nb]= sdata.img_cube
				except Exception as e:
					logger.error("Exception occurred while filling input data (nb=%d), exit generator!" % (nb))
					print("datasize")
					print(self.datasize)
					print("data_index")
					print(data_index)
					print("batch_size")
					print(batch_size)
					print("data_shape")
					print(data_shape)
					print("inputs_shape")
					print(inputs_shape)
					break

				class_ids.append(class_id)
				nb+= 1

				# - Return data if number of batch is reached and restart the batch
				if nb>=batch_size:
					# - Compute class abundances in batch
					class_counts= np.bincount(class_ids)
					class_fracts= class_counts/len(class_ids)
					class_counts_str= ' '.join([str(x) for x in class_counts])
					class_fracts_str= ' '.join([str(x) for x in class_fracts])
					logger.debug("Class counts/fract in batch: counts=[%s], fract=[%s]" % (class_counts_str, class_fracts_str))

					# - Return data
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
	def generate_simclr_data(self, batch_size=32, shuffle=True, read_crop=False, crop_size=32, balance_classes=False, class_probs={}):
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

				# - Apply class rebalancing?
				class_id= sdata_1.id
				class_name= sdata_1.label

				if balance_classes and class_probs:					
					##prob= class_probs[class_id]
					prob= class_probs[class_name]
					r= random.uniform(0, 1)
					accept= r<prob
					if not accept:
						continue
				
				# - Initialize return data
				if nb==0:
					# - The ref implementation (https://github.com/mwdhont/SimCLRv1-keras-tensorflow/blob/master/DataGeneratorSimCLR.py)
					#   uses a dimension (2*batch, 1, ny, nx, nchan), so that returned inputs is a list of len(2*batch) and item passed to encoder has shape (1,ny,nx,nchan) (NB: batch size=1)
					inputs_simclr_shape= (2*batch_size, 1) + data_shape # original ref
					inputs_simclr= np.empty(inputs_simclr_shape, dtype=np.float32)
					labels_ab_aa = np.zeros((batch_size, 2 * batch_size))
					labels_ba_bb = np.zeros((batch_size, 2 * batch_size))
					class_ids= []

				# - Update inputs
				# - The ref implementation (https://github.com/mwdhont/SimCLRv1-keras-tensorflow/blob/master/DataGeneratorSimCLR.py)
				#   shuffles the position of augmented image pair
				inputs_simclr[nb]= sdata_1.img_cube
				inputs_simclr[nb + batch_size]= sdata_2.img_cube
				labels_ab_aa[nb, nb] = 1
				labels_ba_bb[nb, nb] = 1

				class_ids.append(class_id)
				nb+= 1

				# - Return data if number of batch is reached and restart the batch
				if nb>=batch_size:
					# - Compute class abundances in batch
					class_counts= np.bincount(class_ids)
					class_fracts= class_counts/len(class_ids)
					class_counts_str= ' '.join([str(x) for x in class_counts])
					class_fracts_str= ' '.join([str(x) for x in class_fracts])
					logger.debug("Class counts/fract in batch: counts=[%s], fract=[%s]" % (class_counts_str, class_fracts_str))

					# - Return data
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


	def generate_simclr_data_v2(self, batch_size=32, shuffle=True, read_crop=False, crop_size=32, balance_classes=False, class_probs={}):
		""" Generator function for SimCLR task (version 2) """
	
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
				inputs_shape= (2*batch_size,) + data_shape

				# - Apply class rebalancing?
				class_id= sdata_1.id
				class_name= sdata_1.label

				if balance_classes and class_probs:					
					prob= class_probs[class_name]
					r= random.uniform(0, 1)
					accept= r<prob
					if not accept:
						continue
				
				# - Initialize return data
				if nb==0:
					# - The ref implementation (https://github.com/garder14/simclr-tensorflow2/blob/main/datasets.py)
					#   uses a dimension (2*batch, ny, nx, nchan)
					inputs_simclr= np.empty(inputs_shape, dtype=np.float32)
					class_ids= []

				# - Update inputs
				inputs_simclr[nb]= sdata_1.img_cube
				inputs_simclr[nb + 1]= sdata_2.img_cube

				class_ids.append(class_id)
				nb+= 2

				# - Return data if number of batch is reached and restart the batch
				if nb>=batch_size:
					# - Compute class abundances in batch
					class_counts= np.bincount(class_ids)
					class_fracts= class_counts/len(class_ids)
					class_counts_str= ' '.join([str(x) for x in class_counts])
					class_fracts_str= ' '.join([str(x) for x in class_fracts])
					logger.debug("Class counts/fract in batch: counts=[%s], fract=[%s]" % (class_counts_str, class_fracts_str))

					# - Return data
					yield inputs_simclr

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
	##     GENERATE BYOL TRAIN DATA
	#####################################
	def generate_byol_data(self, batch_size=32, shuffle=True, read_crop=False, crop_size=32, balance_classes=False, class_probs={}):
		""" Generator function for BYOL task """
	
		nb= 0
		data_index= -1
		data_indexes= np.arange(0,self.datasize)

		logger.info("Starting data generator ...")

		while True:
			try:

				if nb==0:
					logger.debug("Starting new batch ...")

				# - Generate random data index and pairs of augmented data
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

				# - Apply class rebalancing?
				class_id= sdata_1.id
				class_name= sdata_1.label

				if balance_classes and class_probs:
					prob= class_probs[class_name]
					r= random.uniform(0, 1)
					accept= r<prob
					if not accept:
						continue
				
				# - Initialize return data
				if nb==0:
					inputs_1= np.zeros(inputs_shape, dtype=np.float32)
					inputs_2= np.zeros(inputs_shape, dtype=np.float32)
					class_ids= []

				# - Update inputs
				inputs_1[nb]= sdata_1.img_cube
				inputs_2[nb]= sdata_2.img_cube
				class_ids.append(class_id)
				nb+= 1

				# - Return data if number of batch is reached and restart the batch
				if nb>=batch_size:
					# - Compute class abundances in batch
					class_counts= np.bincount(class_ids)
					class_fracts= class_counts/len(class_ids)
					class_counts_str= ' '.join([str(x) for x in class_counts])
					class_fracts_str= ' '.join([str(x) for x in class_fracts])
					logger.debug("Class counts/fract in batch: counts=[%s], fract=[%s]" % (class_counts_str, class_fracts_str))

					# - Return data
					yield inputs_1, inputs_2
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
	def generate_data(self, batch_size=32, shuffle=True, read_crop=False, crop_size=32, balance_classes=False, class_probs={}):
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

				# - Apply class rebalancing?
				class_id= sdata.id
				class_name= sdata.label
				if balance_classes and class_probs:					
					prob= class_probs[class_name]
					r= random.uniform(0, 1)
					accept= r<prob
					if not accept:
						continue

				# - Initialize return data
				if nb==0:
					inputs= np.zeros(inputs_shape, dtype=np.float32)
					class_ids= []
					
				# - Update inputs
				inputs[nb]= sdata.img_cube
				class_ids.append(class_id)
				nb+= 1

				# - Return data if number of batch is reached and restart the batch
				if nb>=batch_size:
					# - Compute class abundances in batch
					class_counts= np.bincount(class_ids)
					class_fracts= class_counts/len(class_ids)
					class_counts_str= ' '.join([str(x) for x in class_counts])
					class_fracts_str= ' '.join([str(x) for x in class_fracts])
					logger.debug("Class counts/fract in batch: counts=[%s], fract=[%s]" % (class_counts_str, class_fracts_str))

					# - Return data
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
			


