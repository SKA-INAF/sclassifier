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
		
		# - Input data normalization
		self.input_data= None
		self.normalize_inputs= True
		self.normmin= 0.001
		self.normmax= 10
		

	#################################
	##     SETTERS/GETTERS
	#################################
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
		imgfilenames= []
		
		# - Read filelists
		filelist_counter= 0

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

			for j in range(len(imgfilenames[i])):
				imgcounter+= 1
				filename= imgfilenames[i][j]
				logger.info("Reading file %s ..." % filename) 
				data= None
				try:
					data, header= Utils.read_fits(filename)
				except Exception as ex:
					errmsg= 'Failed to read bkg image data (err=' + str(ex) + ')'
					logger.error(errmsg)
					return -1
	
				imgsize= np.shape(data)
				nx= imgsize[1]
				ny= imgsize[0]
				nchannels= len(imgfilenames[i])
				logger.info("Image no. %d (chan=%d) has size (%d,%d)" % (i+1,j+1,nx,ny) )	

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
				data.reshape(imgsize[0],imgsize[1],1)
				imgdata_stack.append(data)
				
			#	- Set image data as a tensor of size [Nsamples,Nx,Ny,Nchan]
			imgdata_cube= np.dstack(imgdata_stack)
			input_data_list.append(imgdata_cube)
			
		#- Convert list to array
		self.input_data= np.array(input_data_list)
		self.input_data= self.input_data.astype('float32')

		logger.info("Shape of input data")
		print(np.shape(self.input_data))

		# - Normalize to [0,1]
		#if self.normalize_inputs:
		#	logger.debug("inputs_source (BEFORE NORMALIZATION): min/max=%s/%s" % (str(np.min(self.inputs_source)),str(np.max(self.inputs_source))))
		#	self.inputs_source= (self.inputs_source - self.normmin)/(self.normmax-self.normmin)
		#	logger.debug("inputs_source (AFTER NORMALIZATION): min/max=%s/%s" % (str(np.min(self.inputs_source)),str(np.max(self.inputs_source))))
				
				
		return 0

	
		

