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

## ASTROPY MODULES 
from astropy.io import ascii

## ADDON ML MODULES
from sklearn.model_selection import train_test_split
import imgaug
from imgaug import augmenters as iaa

## OPENCV
import cv2

## PACKAGE MODULES
from .utils import Utils

##############################
##     GLOBAL VARS
##############################
from sclassifier_vae import logger


##############################
##     SOURCE DATA CLASS
##############################
class SourceData(object):
	""" Source data class """

	def __init__(self):
		""" Create a source data object """
	
		# - Init vars
		self.filepaths= []
		self.sname= "XXX"
		self.label= "UNKNOWN"
		self.id= -1
		self.f_badpix_thr= 0.3
		self.img_data= []
		self.img_data_mask= []
		self.img_heads= []
		self.img_cube= None
		self.img_cube_mask= None
		self.nx= 0
		self.ny= 0
		self.nchannels= 0

		
	def set_from_dict(self, d):
		""" Set source data from input dictionary """ 
		try:
			self.filepaths= d["filepaths"]
			self.sname= d["sname"]
			self.label= d["label"]
			self.id= d["id"]
		except:
			logger.warn("Failed to read values from given dictionary, check keys!")
			return -1

		return 0

	def read_imgs(self):
		""" Read image data from paths """

		# - Check data filelists
		if not self.filepaths:
			logger.error("Empty filelists given!")
			return -1

		# - Read images
		nimgs= len(self.filepaths)
		self.nchannels= nimgs
		#print("filepaths")
		#print(self.filepaths)

		for filename in self.filepaths:
			# - Read image
			logger.debug("Reading file %s ..." % filename) 
			data= None
			try:
				data, header= Utils.read_fits(filename)
			except Exception as e:
				logger.error("Failed to read image data from file %s (err=%s)!" % (filename,str(e)))
				return -1

			# - Compute data mask
			#   NB: =1 good values, =0 bad (pix=0 or pix=inf or pix=nan)
			data_mask= np.logical_and(data!=0,np.isfinite(data)).astype(np.uint8)
		
			# - Check image integrity
			has_bad_pixs= self.has_bad_pixel(data, check_fract=False, thr=0)
			if has_bad_pixs:
				logger.warn("Image %s has too many bad pixels (f=%f>%f)!" % (filename,f_badpix,self.f_badpix_thr) )	
				return -1

			# - Append image channel data to list
			self.img_data.append(data)
			self.img_heads.append(header)
			self.img_data_mask.append(data_mask)
		
		# - Check image sizes
		if not self.check_img_sizes():
			logger.error("Image channels for source %s do not have the same size, check your dataset!" % self.sname)
			return -1

		# - Set data cube
		self.img_cube= np.dstack(self.img_data)
		self.img_cube= self.img_cube.astype(np.float32) # convert otherwise skimage resize fails for 1d image
		self.img_cube_mask= np.dstack(self.img_data_mask)
		self.nx= self.img_cube.shape[1]
		self.ny= self.img_cube.shape[0]
		self.nchannels= self.img_cube.shape[-1]

		return 0

	def check_img_sizes(self):
		""" Check if images have the same size """
		
		# - Return false if no images are stored
		if not self.img_data:
			return False

		# - Compare image sizes across different channels
		same_size= True
		nx_tmp= 0
		ny_tmp= 0
		for i in range(len(self.img_data)):
			imgsize= np.shape(self.img_data)
			nx= imgsize[1]
			ny= imgsize[0]
			if i==0:
				nx_tmp= nx
				ny_tmp= ny	
			else:
				if (nx!=nx_tmp or ny!=ny_tmp):
					logger.debug("Image %s has different size (%d,%d) wrt to previous images (%d,%d)!" % (self.filepaths[i],nx,ny,nx_tmp,ny_tmp))
					same_size= False

		return same_size


	def has_bad_pixel(self, data, check_fract=True, thr=0.1):
		""" Check image data values """ 
		
		npixels= data.size
		npixels_nan= np.count_nonzero(np.isnan(data)) 
		npixels_inf= np.count_nonzero(np.isinf(data))
		n_badpix= npixels_nan + npixels_inf
		f_badpix= n_badpix/float(npixels)
		if check_fract:
			if f_badpix>thr:
				logger.warn("Image has too many bad pixels (f=%f>%f)!" % (f_badpix,thr) )	
				return True
		else:
			if n_badpix>thr:
				logger.warn("Image has too many bad pixels (n=%f>%f)!" % (n_badpix,thr) )	
				return True

		return False


	def has_bad_pixel_cube(self, datacube, check_fract=True, thr=0.1):
		""" Check image data cube values """ 
		
		if datacube.ndim!=3:
			logger.warn("Given data cube has not 3 dimensions!")
			return False
		
		nchannels= datacube.shape[2]
		status= 0
		for i in range(nchannels):
			data= datacube[:,:,i]
			check= self.check_img_data(data, check_fract, thr) 
			if not check:
				logger.warn("Channel %d in cube has bad pixels ..." % i+1)
				status= False	

		return status

	
	def resize_imgs(self, nx, ny, preserve_range=True):
		""" Resize images to the same size """

		# - Return if data cube is None
		if self.img_cube is None:
			logger.error("Image data cube is None!")
			return -1

		# - Check if resizing is needed
		is_same_size= (nx==self.nx) and (ny==self.ny)
		if is_same_size:
			logger.info("Images have already the desired size (%d,%d), nothing to be done..." % (nx,ny))
			return 0

		# - Resize data cube
		try:
			data_resized= Utils.resize_img(self.img_cube, (ny, nx, self.nchannels), preserve_range=True)
		except Exception as e:
			logger.warn("Failed to resize data to size (%d,%d) (err=%s)!" % (nx,ny,str(e)))
			return -1

		# - Resize data cube mask
		try:
			data_mask_resized= Utils.resize_img(self.img_cube_mask, (ny, nx, self.nchannels), preserve_range=True)
		except Exception as e:
			logger.warn("Failed to resize data mask to size (%d,%d) (err=%s)!" % (nx,ny,str(e)))
			return -1

		# - Check data cube integrity
		has_bad_pixs= self.has_bad_pixel(data_resized, check_fract=False, thr=0)
		if has_bad_pixs:
			logger.warn("Resized data cube has bad pixels!")	
			return -1

		has_bad_pixs= self.has_bad_pixel(data_mask_resized, check_fract=False, thr=0)
		if has_bad_pixs:
			logger.warn("Resized data cube mask has bad pixels!")	
			return -1
		
		# - Update data cube
		self.img_cube= data_resized
		self.img_cube_mask= data_mask_resized
		self.nx= self.img_cube.shape[1]
		self.ny= self.img_cube.shape[0]
		self.nchannels= self.img_cube.shape[-1]
		#print("Image cube size after resizing")
		#print(self.img_cube.shape)

		return 0	
		
	
	def scale_imgs(self, scale_factors):
		""" Rescale image pixels with given weights """

		# - Return if data cube is None
		if self.img_cube is None:
			logger.error("Image data cube is None!")
			return -1

		# - Check size of scale factors
		nchannels= self.img_cube.shape[2]
		nscales= len(scale_factors)
		if nscales<=0 or nscales!=nchannels:
			logger.error("Empty scale factors or size different from data channels!")
			return -1

		# - Apply scale factors
		self.img_cube*= scale_factors

		return 0


	def standardize_imgs(self, means, sigmas=[]):
		""" Rescale image pixels with given weights """

		# - Return if data cube is None
		if self.img_cube is None:
			logger.error("Image data cube is None!")
			return -1

		# - Check size of means
		nchannels= self.img_cube.shape[2]
		nmeans= len(means)
		if nmeans<=0 or nmeans!=nchannels:
			logger.error("Empty mean coefficient or size different from data channels!")
			return -1

		# - Check size of sigmas
		sigma_scaling= False
		if sigmas:
			nsigmas= len(sigmas)
			if nsigmas<=0 or nsigmas!=nchannels:
				logger.error("Empty sigma coefficient or size different from data channels!")
				return -1
			sigma_scaling= True

		# - Subtract mean.
		#   NB: Set previously masked pixels to 0
		#print("== img_cube shape ==")
		#print(self.img_cube.shape)
		#print("== means ==")
		#print(means)
		#print("== sigmas ==")
		#print(sigmas)
		#print("== pixels (before standardization) ==")
		#pix_x= 30
		#pix_y= 30
		#for i in range(self.img_cube.shape[-1]):
		#	print("--> ch%d" % (i+1))
		#	print(self.img_cube[pix_y,pix_x,i])

		if sigma_scaling:
			data_norm= (self.img_cube-means)/sigmas
		else:
			data_norm= (self.img_cube-means)
		data_norm[self.img_cube==0]= 0

		self.img_cube= data_norm

		#print("== pixels (after standardization) ==")
		#for i in range(self.img_cube.shape[-1]):
		#	print("--> ch%d" % (i+1))
		#	print(self.img_cube[pix_y,pix_x,i])

		return 0
		

	def log_transform_imgs(self):
		""" Apply log transform to images """

		# - Return if data cube is None
		if self.img_cube is None:
			logger.error("Image data cube is None!")
			return -1

		# - Find min & max across all channels
		#   NB: Excluding masked pixels (=0)
		data_masked= np.ma.masked_equal(self.img_cube, 0.0, copy=False)
		data_min= data_masked.min()
		data_max= data_masked.max()

		# - Log transform
		#   NB: Set previously masked pixels to 0
		data_transf= np.log10(self.img_cube+1-data_min)
		data_transf[self.img_cube==0]= 0

		# - Check data cube integrity
		has_bad_pixs= self.has_bad_pixel(data_transf, check_fract=False, thr=0)
		if has_bad_pixs:
			logger.warn("Log-transformed data cube has bad pixels!")	
			return -1

		# - Normalize in range [0,1].
		#   NB: Set previously masked pixels to 0
		data_masked= np.ma.masked_equal(data_transf, 0.0, copy=False)
		data_min= data_masked.min()
		data_max= data_masked.max()
		data_norm= (data_transf-data_min)/(data_max-data_min)
		data_norm[self.img_cube==0]= 0

		# - Check data cube integrity
		has_bad_pixs= self.has_bad_pixel(data_norm, check_fract=False, thr=0)
		if has_bad_pixs:
			logger.warn("Log-transformed and normalized data cube has bad pixels!")	
			return -1

		# - Update data cube
		self.img_cube= data_norm
	
		return 0

	def erode_imgs(self, kernsize=5):
		""" Erode images """

		# - Return if data cube is None
		if self.img_cube is None:
			logger.error("Image data cube is None!")
			return -1

		# - Define erosion operation
		structel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernsize,kernsize))
		#structel= cv2.getStructuringElement(cv2.MORPH_RECTANGLE, (kernsize,kernsize))

		# - Create erosion masks and apply to input data
		for i in range(self.img_cube.shape[-1]):
			mask= np.logical_and(self.img_cube[:,:,i]!=0,np.isfinite(self.img_cube[:,:,i])).astype(np.uint8)
			mask= mask.astype(np.uint8)
			#mask[mask!=0]= 1
			#print(mask.min())
			#print(mask.max())
			mask_eroded = cv2.erode(mask, structel, iterations = 1)
			
			img_eroded= self.img_cube[:,:,i]
			img_eroded[mask_eroded==0]= 0
			self.img_cube[:,:,i]= img_eroded
			
		# - Update mask
		self.img_cube_mask= np.logical_and(self.img_cube!=0,np.isfinite(self.img_cube)).astype(np.uint8)

		#print("self.img_cube_mask.shape")
		#print(self.img_cube_mask.shape)
		
		return 0


	def divide_imgs(self, chref=0, logtransf=False, make_positive=True, chan_mins=[], strip_chref=True, trim=True, trim_min=-6, trim_max=6):
		""" Normalize images by dividing for a given channel id """

		# - Return if data cube is None
		if self.img_cube is None:
			logger.error("Image data cube is None!")
			return -1

		# - Make positive?
		data_norm= np.copy(self.img_cube)
		if make_positive and len(chan_mins)==self.img_cube.shape[-1]:
			data_norm-= chan_mins

		# - Divide by reference channel
		data_denom= np.copy(data_norm[:,:,chref])
		data_denom[data_denom==0]= 1
		for i in range(data_norm.shape[-1]):
			data_norm[:,:,i]/= data_denom
		data_norm[self.img_cube==0]= 0

		#data_min= data_norm.min()
		#data_max= data_norm.max()

		#print("== data min/max ==")
		#print(data_min)
		#print(data_max)

		if logtransf:
			data_norm[data_norm<=0]= 1
			data_norm_lg= np.log10(data_norm)
			data_norm= data_norm_lg

			data_norm[self.img_cube==0]= 0

			#data_min= data_norm.min()
			#data_max= data_norm.max()
			#print("== lg data min/max ==")
			#print(data_min)
			#print(data_max)

			if trim:
				data_norm[data_norm>trim_max]= trim_max
				data_norm[data_norm<trim_min]= trim_min	
					
		# - Check data cube integrity
		has_bad_pixs= self.has_bad_pixel(data_norm, check_fract=False, thr=0)
		if has_bad_pixs:
			logger.warn("Normalized data cube has bad pixels!")	
			return -1

		# - Update data cube 
		if strip_chref:
			self.img_cube= np.delete(data_norm, chref, axis=2)
			self.nchannels= self.img_cube.shape[-1]
		else:
			self.img_cube= data_norm

		return 0


	def normalize_imgs(self, scale_to_abs_max=False, scale_to_max=False, refch=-1):
		""" Normalize images in range [0,1] """

		# - Return if data cube is None
		if self.img_cube is None:
			logger.error("Image data cube is None!")
			return -1

		# - Find min & max across all channels
		#   NB: Excluding masked pixels (=0)
		data_masked= np.ma.masked_equal(self.img_cube, 0.0, copy=False)
		data_min= data_masked.min()
		data_max= data_masked.max()

		#data_masked_ch= np.ma.masked_equal(self.img_cube[:,:,refch], 0.0, copy=False)	
		#data_min_ch= data_masked_ch.min()
		#data_max_ch= data_masked_ch.max()

		data_mins= []
		data_maxs= []
		for i in range(self.img_cube.shape[-1]):
			data_masked_ch= np.ma.masked_equal(self.img_cube[:,:,i], 0.0, copy=False)
			data_min_ch= data_masked_ch.min()
			data_max_ch= data_masked_ch.max()
			data_mins.append(data_min_ch)
			data_maxs.append(data_max_ch)

		####### DEBUG ###########
		print("== data min/max ==")
		print(data_min)
		print(data_max)

		print("== data mins/maxs ==")
		print(data_mins)
		print(data_maxs)

		print("== pixels (before norm) ==")
		pix_x= 31
		pix_y= 31
		for i in range(self.img_cube.shape[-1]):
			print("--> ch%d" % (i+1))
			print(self.img_cube[pix_y,pix_x,i])
		###########################


		# - Normalize in range [0,1] or to max.
		#   NB: Set previously masked pixels to 0
		if scale_to_max:
			if scale_to_abs_max:
				scale_factor= data_max
			else:
				if refch==-1:
					scale_factor= data_maxs
				else:
					scale_factor= data_maxs[refch]
			data_norm= self.img_cube/scale_factor
		else:
			if scale_to_abs_max:
				data_norm= (self.img_cube-data_min)/(data_max-data_min)
			else:
				diffs= [x - y for x, y in zip(data_maxs, data_mins)] 
				data_norm= (self.img_cube-data_mins)/diffs			

		data_norm[self.img_cube==0]= 0

		# - Check data cube integrity
		has_bad_pixs= self.has_bad_pixel(data_norm, check_fract=False, thr=0)
		if has_bad_pixs:
			logger.warn("Normalized data cube has bad pixels!")	
			return -1

		# - Update data cube
		self.img_cube= data_norm


		##### DEBUG ############
		data_masked= np.ma.masked_equal(self.img_cube, 0.0, copy=False)
		data_min= data_masked.min()
		data_max= data_masked.max()

		data_mins= []
		data_maxs= []
		for i in range(self.img_cube.shape[-1]):
			data_masked_ch= np.ma.masked_equal(self.img_cube[:,:,i], 0.0, copy=False)
			data_min_ch= data_masked_ch.min()
			data_max_ch= data_masked_ch.max()
			data_mins.append(data_min_ch)
			data_maxs.append(data_max_ch)

		print("== data min/max (after norm) ==")
		print(data_min)
		print(data_max)

		print("== data mins/maxs (after norm) ==")
		print(data_mins)
		print(data_maxs)

		print("== pixels (after norm) ==")
		for i in range(self.img_cube.shape[-1]):
			print("--> ch%d" % (i+1))
			print(self.img_cube[pix_y,pix_x,i])
	
		##########################

		return 0

	def augment_imgs(self, augmenter):
		""" Augment images """

		# - Return if data cube is None
		if self.img_cube is None:
			logger.error("Image data cube is None!")
			return -1

		# Make augmenters deterministic to apply similarly to images and masks
		augmenter_det = augmenter.to_deterministic()

		# - Augment data cube
		try:
			#data_aug= augmenter(images=self.img_cube)
			data_aug= augmenter_det.augment_image(self.img_cube)
		except Exception as e:
			logger.error("Failed to augment data (err=%s)!" % str(e))
			return -1

		# - Apply same augmentation to mask
		#def activator(images, augmenter, parents, default):
		#	return False if augmenter.name in ["blur", "dropout"] else default

		try:
			#data_mask_aug= augmenter(images=self.img_cube_mask)
			#data_mask_aug= augmenter_det.augment_image(self.img_cube_mask, hooks=imgaug.HooksImages(activator=activator))
			data_mask_aug= augmenter_det.augment_image(self.img_cube_mask)
		except Exception as e:
			logger.error("Failed to augment data mask (err=%s)!" % str(e))
			return -1

		# - Check data cube integrity
		has_bad_pixs= self.has_bad_pixel(data_aug, check_fract=False, thr=0)
		if has_bad_pixs:
			logger.warn("Augmented data cube has bad pixels!")	
			return -1

		has_bad_pixs= self.has_bad_pixel(data_mask_aug, check_fract=False, thr=0)
		if has_bad_pixs:
			logger.warn("Augmented data cube mask has bad pixels!")	
			return -1

		# - Update image cube mask
		self.img_cube= data_aug
		self.img_cube_mask= data_mask_aug

		return 0

##############################
##     DATA LOADER
##############################
class DataLoader(object):

	""" Read data from disk and provide it to the network

			Arguments:
				- datalist: Filelist (json) with input data
				
	"""
	
	def __init__(self, filename):
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

		# - Define augmenters
		naugmenters_applied= 2
		self.augmenter= iaa.SomeOf((0,naugmenters_applied),
			[
  			iaa.Fliplr(1.0),
    		iaa.Flipud(1.0),
    		iaa.Affine(rotate=(-90, 90), mode='constant', cval=0.0),
				#iaa.Affine(scale=(0.5, 1.5), mode='constant', cval=0.0),
				iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, mode='constant', cval=0.0)
			],
			random_order=True
		)

	#############################
	##     READ INPUT DATA
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

	def read_data(self, index, resize=True, nx=64, ny=64, normalize=True, scale_to_abs_max=False, scale_to_max=False, augment=False, log_transform=False, scale=False, scale_factors=[], standardize=False, means=[], sigmas=[], chan_divide=False,chan_mins=[], erode=False, erode_kernel=5):	
		""" Read data at given index """

		# - Check index
		if index<0 or index>=self.datasize:
			logger.error("Invalid index %d given!" % index)
			return None

		# - Read source image data
		logger.debug("Reading source image data %d ..." % index)
		d= self.datalist["data"][index]
		sdata= SourceData()
		if sdata.set_from_dict(d)<0:
			logger.error("Failed to set source image data %d!" % index)
			return None

		if sdata.read_imgs()<0:
			logger.error("Failed to read source images %d!" % index)
			return None

		# - Rescale image data?
		#if scale and scale_factors:
		#	logger.debug("Rescaling source image data %d ..." % index)
		#	if sdata.scale_imgs(scale_factors)<0:
		#		logger.error("Failed to re-scale source image %d!" % index)
		#		return None

		# - Standardize image data?
		#if standardize and means:
		#	logger.debug("Standardizing source image data %d ..." % index)
		#	if sdata.standardize_imgs(means, sigmas)<0:
		#		logger.error("Failed to standardize source image %d!" % index)
		#		return None

		# - Erode image?
		if erode:
			logger.debug("Eroding source image data %d ..." % index)
			if sdata.erode_imgs(erode_kernel)<0:
				logger.error("Failed to erode source image %d!" % index)
				return None
			
		# - Run augmentation?
		if augment:
			logger.debug("Augmenting source image data %d ..." % index)
			if sdata.augment_imgs(self.augmenter)<0:
				logger.error("Failed to augment source image %d!" % index)
				return None

		# - Resize image?
		if resize:
			logger.debug("Resizing source image data %d ..." % index)
			if sdata.resize_imgs(nx, ny, preserve_range=True)<0:
				logger.error("Failed to resize source image %d to size (%d,%d)!" % (index,nx,ny))
				return None

		# - Rescale image data?
		if scale and scale_factors:
			logger.debug("Rescaling source image data %d ..." % index)
			if sdata.scale_imgs(scale_factors)<0:
				logger.error("Failed to re-scale source image %d!" % index)
				return None

		# - Standardize image data?
		if standardize and means:
			logger.debug("Standardizing source image data %d ..." % index)
			if sdata.standardize_imgs(means, sigmas)<0:
				logger.error("Failed to standardize source image %d!" % index)
				return None			

		# - Normalize image?
		if normalize:
			if sdata.normalize_imgs(scale_to_abs_max, scale_to_max)<0:
				logger.error("Failed to normalize source image %d!" % index)
				return None

		# - Channel division?
		if chan_divide:
			if sdata.divide_imgs(chref=0, logtransf=log_transform, make_positive=True, chan_mins=chan_mins)<0:
				logger.error("Failed to chan divide source image %d!" % index)
				return None

		# - Log-tranform image?
		#if log_transform:
		#	if sdata.log_transform_imgs()<0:
		#		logger.error("Failed to log-transform source image %d!" % index)
		#		return None

		return sdata

	###################################
	##     GENERATE DATA FOR TRAINING
	###################################
	def data_generator(self, batch_size=32, shuffle=True, resize=True, nx=64, ny=64, normalize=True, scale_to_abs_max=False, scale_to_max=False, augment=False, log_transform=False, scale=False, scale_factors=[], standardize=False, means=[], sigmas=[], chan_divide=False, chan_mins=[], erode=False, erode_kernel=5, retsdata=False):	
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
				
				sdata= self.read_data(
					data_index, 
					resize=resize, nx=nx, ny=ny,
					normalize=normalize, scale_to_abs_max=scale_to_abs_max, scale_to_max=scale_to_max, 
					augment=augment,
					log_transform=log_transform,
					scale=scale, scale_factors=scale_factors,
					standardize=standardize, means=means, sigmas=sigmas,
					chan_divide=chan_divide, chan_mins=chan_mins,
					erode=erode, erode_kernel=erode_kernel
				)
				if sdata is None:
					logger.warn("Failed to read source data at index %d, skip to next ..." % data_index)
					continue

				if sdata.img_cube is None:
					logger.warn("Failed to read source data cube at index %d, skip to next ..." % data_index)
					continue

				data_shape= sdata.img_cube.shape
				inputs_shape= (batch_size,) + data_shape
				#print("Generating batch %d/%d ..." % (nb, batch_size))
				#print(inputs_shape)
				logger.debug("Data %d shape=(%d,%d,%d)" % (data_index,data_shape[0],data_shape[1],data_shape[2]))
				

				# - Initialize return data
				if nb==0:
					inputs= np.zeros(inputs_shape, dtype=np.float32)
				
				# - Update inputs
				inputs[nb]= sdata.img_cube
				nb+= 1


				##### DEBUG ############
				data_masked= np.ma.masked_equal(sdata.img_cube, 0.0, copy=False)
				data_min= data_masked.min()
				data_max= data_masked.max()

				data_mins= []
				data_maxs= []
				for i in range(sdata.img_cube.shape[-1]):
					data_masked_ch= np.ma.masked_equal(sdata.img_cube[:,:,i], 0.0, copy=False)
					data_min_ch= data_masked_ch.min()
					data_max_ch= data_masked_ch.max()
					data_mins.append(data_min_ch)
					data_maxs.append(data_max_ch)

				print("== data min/max (after norm) ==")
				print(data_min)
				print(data_max)

				print("== data mins/maxs (after norm) ==")
				print(data_mins)
				print(data_maxs)

				print("== pixels (after norm) ==")
				for i in range(sdata.img_cube.shape[-1]):
					print("--> ch%d" % (i+1))
					print(sdata.img_cube[pix_y,pix_x,i])
	
				##########################

				# - Return data if number of batch is reached and restart the batch
				if nb>=batch_size:
					#print("inputs.shape")
					#print(inputs.shape)
					logger.debug("Batch size (%d) reached, yielding generated data of size (%d,%d,%d,%d) ..." % (nb,inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3]))
					if retsdata:
						yield inputs, sdata
					else:
						yield inputs, inputs
					nb= 0

			except (GeneratorExit, KeyboardInterrupt):
				logger.warn("Generator or keyboard exception catched while generating data...")
				raise
			except Exception as e:
				logger.warn("Exception catched while generating data (err=%s) ..." % str(e))
				raise
			

