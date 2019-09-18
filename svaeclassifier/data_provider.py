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

## ADDON ML MODULES
from sklearn.model_selection import train_test_split

## PACKAGE MODULES
from .utils import Utils

##############################
##     GLOBAL VARS
##############################
logger = logging.getLogger(__name__)


##############################
##     CLASS DEFINITIONS
##############################
class DataProvider(object):
	""" Class to read train data from disk and provide to network

			Arguments:
				- datadir: Data root directory where to search for images
				- fileext: Image file extension to be searched
	"""
	
	
	def __init__(self,filelists=[]):
		""" Return a DataProvider object """

		# - Input data
		self.filelists= filelists	
		self.nx= 0
		self.ny= 0	
		self.nimgs= 0
		self.crop_img= False
		self.nx_crop= 0
		self.ny_crop= 0
		
		# - Input data normalization
		self.input_data= None
		self.normalize_to_first_chan= False	
		self.apply_weights= False
		self.img_weights= []
		self.normalize_inputs= True
		self.normmin= 0.001
		self.normmax= 10
		self.nBadPixThr= 0.6
		self.badPixReplaceVal= 0 #-999
		

	#################################
	##     SETTERS/GETTERS
	#################################
	def enable_inputs_normalization_to_first_channel(self,choice):
		""" Turn on/off inputs normalization to first channel"""
		self.normalize_to_first_chan= choice

	def enable_inputs_normalization(self,choice):
		""" Turn on/off inputs normalization """
		self.normalize_inputs= choice

	def set_input_data_norm_range(self,datamin,datamax):
		""" Set input data normalization range """
		self.normmin= datamin
		self.normmax= datamax

	def get_img_size(self):
		""" Return the train image size """
		return self.nx, self.ny

	def enable_img_crop(self,choice):
		""" Turn on/off input image crop """
		self.crop_img= choice

	def set_img_crop_size(self,nx,ny):
		""" Set input data normalization range """
		self.nx_crop= nx
		self.ny_crop= ny

	def enable_img_weights(self,choice):
		""" Turn on/off apply of input image weights """
		self.apply_weights= choice

	def set_img_weights(self,w):
		""" Set input image weights """
		self.img_weights= w

	def get_data(self):
		""" Return read data """
		return self.input_data	

	#############################
	##     READ INPUT DATA
	#############################
	def read_data(self):	
		""" Read data from disk recursively """
			
		# - Check data filelists
		if not self.filelists:
			logger.error("Empty filelists given!")
			return -1

		nfilelists= len(self.filelists)
		
		# - Check weights size
		if self.apply_weights and len(self.img_weights)!=nfilelists:
			logger.error("Image weights size is different from number of channels given!")
			return -1
		
		# - Read filelists
		filelist_counter= 0
		imgfilenames= []

		for filelist in self.filelists:
			filelist_counter+= 1
			imgfilenames.append([])

			try:
				filenames= Utils.read_ascii(filelist,['#'])
			except IOError:
				errmsg= 'Cannot read file: ' + filelist
				logger.error(errmsg)
				return -1

			# - Check lists have the same number of files
			if filelist_counter==1:
				self.nimgs= len(filenames)
			else:
				if len(filenames)!=self.nimgs:
					logger.error("Given filelists have a different number of file entries (%s!=%s)!" % (len(filenames),self.nimgs))
					return -1

			
			# - Read image files in list	
			for item in filenames:
				
				filename= item[0]
				#logger.info("Reading file %s ..." % filename) 

				imgfilenames[filelist_counter-1].append(filename)

		# - Reorder list of files by channel
		imgfilenames= map(list, zip(*imgfilenames))

		# - Loop over image files and read them
		imgcounter= 0
		input_data_list= []

		for i in range(len(imgfilenames)):
		
			imgdata_stack= []
			isGoodImage= True

			for j in range(len(imgfilenames[i])):
				imgcounter+= 1
				filename= imgfilenames[i][j]
				logger.info("Reading file %s ..." % filename) 
				data= None
				try:
					data, header= Utils.read_fits(filename)
				except Exception as ex:
					errmsg= 'Failed to read image data (err=' + str(ex) + ')'
					logger.warn(errmsg)
					isGoodImage= False
					break
	
				imgsize= np.shape(data)
				nx= imgsize[1]
				ny= imgsize[0]
				nchannels= len(imgfilenames[i])
				logger.info("Image no. %d (chan=%d) has size (%d,%d)" % (i+1,j+1,nx,ny) )	

				# - Extract crop img data
				data_crop= data
				if self.crop_img:
					if self.nx_crop<=0 or self.nx_crop>nx or self.ny_crop<=0 or self.ny_crop>ny:
						errmsg= 'Requested crop size is zero or exceeding image size!'
						logger.warn(errmsg)
						isGoodImage= False
						break

					if self.nx_crop!=nx and self.ny_crop!=ny:
						x0= np.ceil(nx/2.)
						y0= np.ceil(ny/2.)
						data_crop= Utils.crop_img(data,x0,y0,self.nx_crop,self.ny_crop)
						imgsize= np.shape(data_crop)
						nx= imgsize[1]
						ny= imgsize[0]

					logger.info("Cropped image no. %d (chan=%d) has size (%d,%d)" % (i+1,j+1,nx,ny) )	

				# - Check image size is equal for all files
				if imgcounter==1:
					self.nx= nx
					self.ny= ny	
				else:
					if (nx!=self.nx or ny!=self.ny):
						errmsg= 'Image no. ' + str(imgcounter) + ' has different size (nx=' + str(self.nx) + ',ny=' + str(self.ny) + ') wrt previous images!'
						logger.error(errmsg)
						return -1

				#	- Set image data as a tensor of size [Nsamples,Nx,Ny,Nchan] Nchan=1 and create stack
				data_crop.reshape(imgsize[0],imgsize[1],1)

				# - Check image value integrity
				npixels= data_crop.size
				npixels_nan= np.count_nonzero(np.isnan(data_crop)) 
				npixels_inf= np.count_nonzero(np.isinf(data_crop))
				badPixFraction= (npixels_nan+npixels_inf)/float(npixels)
				if badPixFraction>self.nBadPixThr:
					logger.warn("Cropped image no. %d (chan=%d) has too many bad pixels (badPixFract=%f), skip it" % (i+1,j+1,badPixFraction) )	
					isGoodImage= False
					break

				# - Append image channel data to stack
				imgdata_stack.append(data_crop)
				logger.info("Cropped image no. %d (chan=%d) : min/max=%f/%f" % (i+1,j+1,np.min(data_crop),np.max(data_crop)))
			

			# - Skip image if marked as bad
			if not isGoodImage:
				logger.warn("Skipping image no. %d as marked as bad..." % (i+1) )
				continue	

			# - Apply weights to images
			if self.apply_weights:
				for index in range(0,len(imgdata_stack)):
					print('DEBUG: Chan %d weight=%f' % (index,self.img_weights[index]))
					imgdata_stack[index]*= self.img_weights[index]
					logger.info("Cropped image no. %d (chan=%d) (AFTER WEIGHTS): min/max=%f/%f" % (i+1,index+1,np.min(imgdata_stack[index]),np.max(imgdata_stack[index])))
			

			# - Normalize data to first channel?	
			if self.normalize_to_first_chan and len(imgdata_stack)>1:
				for index in range(0,len(imgdata_stack)):
					if index>0:	
						imgdata_stack[index]= np.divide(imgdata_stack[index],imgdata_stack[0])
					logger.info("Cropped image no. %d (chan=%d) (AFTER NORM): min/max=%f/%f" % (i+1,index+1,np.min(imgdata_stack[index]),np.max(imgdata_stack[index])))
			
				
			# - Replace NaNs & inf with safe value	
			badPixSafeVal= self.badPixReplaceVal
			if self.normalize_inputs:
				badPixSafeVal= self.normmin
				
			for index in range(0,len(imgdata_stack)):
				#np.nan_to_num(imgdata_stack[index])
				(imgdata_stack[index])[~np.isfinite( (imgdata_stack[index]) )]= badPixSafeVal
				logger.info("Cropped image no. %d (chan=%d) (AFTER NORM & WEIGHTS & SANITIZE): min/max=%f/%f" % (i+1,index+1,np.min(imgdata_stack[index]),np.max(imgdata_stack[index])))
			
			
			#	- Set image data as a tensor of size [Nsamples,Nx,Ny,Nchan]
			imgdata_cube= np.dstack(imgdata_stack)
			input_data_list.append(imgdata_cube)
			logger.info("Input data (BEFORE NORMALIZATION): min/max=%s/%s" % (str(np.min(imgdata_cube)),str(np.max(imgdata_cube))))
			
			
		#- Convert list to array
		self.input_data= np.array(input_data_list)
		self.input_data= self.input_data.astype('float32')

		logger.info("Shape of input data")
		print(np.shape(self.input_data))

		# - Normalize to [0,1]
		if self.normalize_inputs:
			logger.info("Input data (BEFORE NORMALIZATION): min/max=%s/%s" % (str(np.min(self.input_data)),str(np.max(self.input_data))))
			self.input_data= (self.input_data - self.normmin)/(self.normmax-self.normmin)
			logger.info("Input data (AFTER NORMALIZATION): min/max=%s/%s" % (str(np.min(self.input_data)),str(np.max(self.input_data))))
				
				
		return 0

	
		

