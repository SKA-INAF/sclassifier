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

## ADDON ML MODULES
from sklearn.model_selection import train_test_split
import imgaug
from imgaug import augmenters as iaa

## OPENCV
import cv2
cv2.setNumThreads(1) # workaround to avoid potential conflicts between TF and OpenCV multithreading (parallel_impl.cpp (240) WorkerThread 18: Can't spawn new thread: res = 11)

## PACKAGE MODULES
from .utils import Utils

##############################
##     GLOBAL VARS
##############################
from sclassifier import logger


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
				data, header, wcs= Utils.read_fits(filename)
			except Exception as e:
				logger.error("Failed to read image data from file %s (err=%s)!" % (filename,str(e)))
				return -1

			# - Compute data mask
			#   NB: =1 good values, =0 bad (pix=0 or pix=inf or pix=nan)
			data_mask= np.logical_and(data!=0,np.isfinite(data)).astype(np.uint8)
		
			# - Check image integrity
			has_bad_pixs= self.has_bad_pixel(data, check_fract=False, thr=0)
			if has_bad_pixs:
				logger.warn("Image %s has too many bad pixels!" % (filename))
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
			logger.debug("Images have already the desired size (%d,%d), nothing to be done..." % (nx,ny))
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
		
	def log_transform_imgs(self, skip_chref=True, chref=0):
		""" Apply log transform to images """
	
		# - Return if data cube is None
		if self.img_cube is None:
			logger.error("Image data cube is None!")
			return -1

		# - Apply log
		cond= np.logical_and(self.img_cube>0, np.isfinite(self.img_cube))
		data_transf= np.log10(self.img_cube, where=cond)

		# - Set to zero neg or nan pixels
		data_transf[~cond]= 0

		# - Check data cube integrity
		has_bad_pixs= self.has_bad_pixel(data_transf, check_fract=False, thr=0)
		if has_bad_pixs:
			logger.warn("Log-transformed data cube has bad pixels!")	
			return -1

		# - Update data cube
		if skip_chref:
			data_ref= np.copy(self.img_cube[:,:,chref])
			self.img_cube= data_transf
			self.img_cube[:,:,chref]= data_ref
		else:
			self.img_cube= data_transf

		return 0

	def log_transform_imgs_old(self):
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


	def mask_borders(self, mask_fract=0.7):
		""" Mask input data at borders """
			
		# - Return if data cube is None
		if self.img_cube is None:
			logger.error("Image data cube is None!")
			return -1

		# - Mask all channels at border
		logger.info("Masking all channels at border (fract=%f) ..." % (mask_fract))
		for i in range(self.img_cube.shape[-1]):
			#data= self.img_cube[:,:,i]
			data_shape= self.img_cube[:,:,i].shape
			data= np.zeros(data_shape, dtype=self.img_cube[:,:,i].dtype)
			xc= int(data_shape[1]/2)
			yc= int(data_shape[0]/2)
			dy= int(data_shape[0]*mask_fract/2.)
			dx= int(data_shape[1]*mask_fract/2.)
			xmin= xc - dx
			xmax= xc + dx
			ymin= yc - dy
			ymax= yc + dy
			logger.info("Masking chan %d (%d,%d) in range x[%d,%d] y[%d,%d]" % (i, data_shape[0], data_shape[1], xmin, xmax, ymin, ymax))
			data[ymin:ymax, xmin:xmax]= self.img_cube[ymin:ymax, xmin:xmax, i]
			self.img_cube[i]= data
	
		return 0


	def __subtract_bkg_and_clip(self, data, sigma_bkg=3, sigma_clip=1, use_mask=True, mask_fract=0.7):
		""" Subtract background and clip below a given sigma in input data """

		# - Return if input data is None
		if data is None:
			logger.error("Input data is None!")
			return None
		
		cond= np.logical_and(data!=0, np.isfinite(data))
		
		# - Mask region at image center (where source is supposed to be)?
		bkgdata= np.copy(data) 
		if use_mask:
			data_shape= data.shape
			xc= int(data_shape[1]/2)
			yc= int(data_shape[0]/2)
			dy= int(data_shape[0]*mask_fract/2.)
			dx= int(data_shape[1]*mask_fract/2.)
			xmin= xc - dx
			xmax= xc + dx
			ymin= yc - dy
			ymax= yc + dy
			logger.info("Masking data (%d,%d) in range x[%d,%d] y[%d,%d]" % (data_shape[0], data_shape[1], xmin, xmax, ymin, ymax))
			bkgdata[ymin:ymax, xmin:xmax]= 0
	
		# - Retrieve chref data
		#cond= np.logical_and(data_ref!=0, np.isfinite(data_ref))
		#data_ref_1d= data_ref[cond]
		#logger.info("--> data ref min/max (before bkgsub)=%s/%s" % (str(data_ref_1d.min()), str(data_ref_1d.max())))

		# - Compute and subtract mean bkg from data
		logger.info("Subtracting bkg ...")
		cond_bkg= np.logical_and(bkgdata!=0, np.isfinite(bkgdata))
		bkgdata_1d= bkgdata[cond_bkg]
		logger.info("--> bkgdata min/max=%s/%s" % (str(bkgdata_1d.min()), str(bkgdata_1d.max())))

		bkgval, _, _ = sigma_clipped_stats(bkgdata_1d, sigma=sigma_bkg)

		data_bkgsub= np.copy(data)
		data_bkgsub-= bkgval
		data_bkgsub[~cond]= 0
		cond_bkgsub= np.logical_and(data_bkgsub!=0, np.isfinite(data_bkgsub))
		data_bkgsub_1d= data_bkgsub[cond_bkgsub]

		logger.info("--> data min/max (after bkgsub)=%s/%s (bkg=%s)" % (str(data_bkgsub_1d.min()), str(data_bkgsub_1d.max()), str(bkgval)))

		# - Set to zero all pixels in reference channel that are below sigma clip
		#clipval= 0
		logger.info("Clipping all pixels in reference channel that are below %f sigma ..." % (sigma_clip))
		clipmean, _, _ = sigma_clipped_stats(data_bkgsub_1d, sigma=sigma_clip)
		data_clipped= np.copy(data_bkgsub)
		#data_clipped[data_clipped<clipmean]= clipval
		data_clipped[data_clipped<clipmean]= clipmean
		data_clipped[~cond]= 0
		cond_clipped= np.logical_and(data_clipped!=0, np.isfinite(data_clipped))
		data_clipped_1d= data_clipped[cond_clipped]

		logger.info("--> data min/max (after sigmaclip)=%s/%s (clipmean=%s)" % (str(data_clipped_1d.min()), str(data_clipped_1d.max()), str(clipmean)))

		return data_clipped 


	def subtract_bkg_and_clip(self, limit_to_chref=True, chref=0, sigma_bkg=3, sigma_clip=1, use_mask=True, mask_fract=0.7):
		""" Subtract background from reference channel and clip below a given sigma """
		
		# - Return if data cube is None
		if self.img_cube is None:
			logger.error("Image data cube is None!")
			return -1

		# - Loop over channels and get bgsub and clipped data
		for i in range(self.img_cube.shape[-1]):
			if limit_to_chref and i!=chref:
				continue
			
			data_clipped= self.__subtract_bkg_and_clip(self.img_cube[:,:,i], sigma_bkg, sigma_clip, use_mask, mask_fract)
			self.img_cube[:,:,i]= data_clipped
		
		# - Retrieve chref data
		#data_ref= np.copy(self.img_cube[:,:,chref])
		#cond= np.logical_and(data_ref!=0, np.isfinite(data_ref))
		#data_ref_1d= data_ref[cond]
		#logger.info("--> data ref min/max (before bkgsub)=%s/%s" % (str(data_ref_1d.min()), str(data_ref_1d.max())))

		# - Subtract mean bkg in reference channel
		#logger.info("Subtracting bkg in reference channel ...")
		#bkgval, _, _ = sigma_clipped_stats(data_ref_1d, sigma=sigma_bkg)

		#data_bkgsub= np.copy(data_ref)
		#data_bkgsub-= bkgval
		#data_bkgsub[~cond]= 0
		#cond_bkgsub= np.logical_and(data_bkgsub!=0, np.isfinite(data_bkgsub))
		#data_bkgsub_1d= data_bkgsub[cond_bkgsub]

		#logger.info("--> data ref min/max (after bkgsub)=%s/%s (bkg=%s)" % (str(data_bkgsub_1d.min()), str(data_bkgsub_1d.max()), str(bkgval)))

		# - Set to zero all pixels in reference channel that are below sigma clip
		#clipval= 0
		#logger.info("Clipping all pixels in reference channel that are below %f sigma ..." % (sigma_clip))
		#clipmean, _, _ = sigma_clipped_stats(data_bkgsub_1d, sigma=sigma_clip)
		#data_clipped= np.copy(data_bkgsub)
		###data_clipped[data_clipped<clipmean]= clipval
		#data_clipped[data_clipped<clipmean]= clipmean
		#data_clipped[~cond]= 0
		#cond_clipped= np.logical_and(data_clipped!=0, np.isfinite(data_clipped))
		#data_clipped_1d= data_clipped[cond_clipped]

		#logger.info("--> data ref min/max (after sigmaclip)=%s/%s (clipmean=%s)" % (str(data_clipped_1d.min()), str(data_clipped_1d.max()), str(clipmean)))

		# - Update data cube
		#self.img_cube[:,:,chref]= data_clipped

		return 0 


	def divide_imgs(self, chref=0, logtransf=False, strip_chref=False, trim=False, trim_min=-6, trim_max=6):
		""" Normalize images by dividing for a given channel id """

		# - Return if data cube is None
		if self.img_cube is None:
			logger.error("Image data cube is None!")
			return -1

		# - Init ref channel
		data_ref= np.copy(self.img_cube[:,:,chref])
		cond= np.logical_and(data_ref!=0, np.isfinite(data_ref))

		# - Divide other channels by reference channel
		data_norm= np.copy(self.img_cube)
		data_denom= np.copy(data_ref)
		data_denom[data_denom==0]= 1

		for i in range(data_norm.shape[-1]):
			if i==chref:
				data_norm[:,:,i]= np.copy(data_ref)
			else:
				logger.info("Divide channel %d by reference channel %d ..." % (i, chref))
				dn= data_norm[:,:,i]/data_denom
				dn[~cond]= 0 # set ratio to zero if ref pixel flux was zero or nan
				data_norm[:,:,i]= dn

		data_norm[self.img_cube==0]= 0

		# - Apply log transform to ratio channels?
		if logtransf:
			logger.info("Applying log-transform to channel ratios ...")
			data_transf= np.copy(data_norm)
			data_transf[data_transf<=0]= 1
			data_transf_lg= np.log10(data_transf)
			data_transf= data_transf_lg
			data_transf[self.img_cube==0]= 0

			if trim:
				data_transf[data_transf>trim_max]= trim_max
				data_transf[data_transf<trim_min]= trim_min

			data_transf[:,:,chref]= data_norm[:,:,chref]
			data_norm= data_transf

		# - Check data cube integrity
		has_bad_pixs= self.has_bad_pixel(data_norm, check_fract=False, thr=0)
		if has_bad_pixs:
			logger.warn("Normalized data cube after chan division has bad pixels!")	
			return -1

		# - Update data cube 
		if strip_chref:
			self.img_cube= np.delete(data_norm, chref, axis=2)
			self.nchannels= self.img_cube.shape[-1]
		else:
			self.img_cube= data_norm

		return 0



	def divide_imgs_old(self, chref=0, logtransf=False, make_positive=True, chan_mins=[], strip_chref=True, trim=True, trim_min=-6, trim_max=6):
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

		if logtransf:
			data_norm[data_norm<=0]= 1
			data_norm_lg= np.log10(data_norm)
			data_norm= data_norm_lg

			data_norm[self.img_cube==0]= 0

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


	def fix_negative_imgs(self):
		""" Subtract image min if image has all negative pixels (e.g. max is negative) """
			
		# - Return if data cube is None
		if self.img_cube is None:
			logger.error("Image data cube is None!")
			return -1

		# - Find image min/max	
		#   NB: Excluding masked pixels (=0)
		img_cube_mod= np.copy(self.img_cube)
		
		for i in range(self.img_cube.shape[-1]):
			data_masked_ch= np.ma.masked_equal(self.img_cube[:,:,i], 0.0, copy=False)
			data_min_ch= data_masked_ch.min()
			data_max_ch= data_masked_ch.max()

			if data_max_ch>0:
				continue

			img_mod= self.img_cube[:,:,i] - data_min_ch
			img_mod[self.img_cube[:,:,i]==0]= 0
			img_cube_mod[:,:,i]= img_mod
			
			
		# - Update data cube
		self.img_cube= img_cube_mod

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
		#print("== data min/max ==")
		#print(data_min)
		#print(data_max)

		#print("== data mins/maxs ==")
		#print(data_mins)
		#print(data_maxs)

		#print("== pixels (before norm) ==")
		#pix_x= 31
		#pix_y= 31
		#for i in range(self.img_cube.shape[-1]):
		#	print("--> ch%d" % (i+1))
		#	print(self.img_cube[pix_y,pix_x,i])
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
		#data_masked= np.ma.masked_equal(self.img_cube, 0.0, copy=False)
		#data_min= data_masked.min()
		#data_max= data_masked.max()

		#data_mins= []
		#data_maxs= []
		#for i in range(self.img_cube.shape[-1]):
		#	data_masked_ch= np.ma.masked_equal(self.img_cube[:,:,i], 0.0, copy=False)
		#	data_min_ch= data_masked_ch.min()
		#	data_max_ch= data_masked_ch.max()
		#	data_mins.append(data_min_ch)
		#	data_maxs.append(data_max_ch)

		#print("== data min/max (after norm) ==")
		#print(data_min)
		#print(data_max)

		#print("== data mins/maxs (after norm) ==")
		#print(data_mins)
		#print(data_maxs)

		#print("== pixels (after norm) ==")
		#for i in range(self.img_cube.shape[-1]):
		#	print("--> ch%d" % (i+1))
		#	print(self.img_cube[pix_y,pix_x,i])
	
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
	
	def __init__(self, filename, augmenter_choice="cae"):
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

		# - Options
		self.fix_negative_imgs= True

		# - Define and set augmenter to be used
		self.__set_augmenters(augmenter_choice)


	#############################
	##     DEFINE AUGMENTERS
	#############################
	def __set_augmenters(self, choice='cae'):
		""" Define and set augmenters """

		# - Define augmenter for CAE
		#naugmenters_cae= 2
		#augmenter_cae= iaa.SomeOf((0,naugmenters_cae),
		#	[
  	#		iaa.Fliplr(1.0),
    #		iaa.Flipud(1.0),
    #		iaa.Affine(rotate=(-90, 90), mode='constant', cval=0.0),
		#		#iaa.Affine(scale=(0.5, 1.5), mode='constant', cval=0.0),
		#		iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, mode='constant', cval=0.0)
		#	],
		#	random_order=True
		#)

		augmenter_cae= iaa.Sequential(
			[
				iaa.OneOf([iaa.Fliplr(1.0), iaa.Flipud(1.0), iaa.Noop()]),
  			iaa.Affine(rotate=(-90, 90), mode='constant', cval=0.0),
				iaa.Sometimes(0.5, iaa.Affine(scale=(0.5, 1.0), mode='constant', cval=0.0))
			]
		)

		# - Define augmenter for CNN
		augmenter_cnn= iaa.Sequential(
			[
				iaa.OneOf([iaa.Fliplr(1.0), iaa.Flipud(1.0), iaa.Noop()]),
  			iaa.Affine(rotate=(-90, 90), mode='constant', cval=0.0),
				iaa.Sometimes(0.5, iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, mode='constant', cval=0.0))
			]
		)

		# - Define augmenter for SimCLR
		naugmenters_simclr= 2
		augmenter_simclr= iaa.Sequential(
			[
  			iaa.OneOf([iaa.Fliplr(1.0), iaa.Flipud(1.0), iaa.Affine(rotate=(-90, 90), mode='constant', cval=0.0)]),
				iaa.SomeOf(naugmenters_simclr,
						[
							iaa.Affine(scale=(0.5, 1.0), mode='constant', cval=0.0),
							iaa.GaussianBlur(sigma=(0.1, 2.0)),
							iaa.AdditiveGaussianNoise(scale=(0, 0.1))
						],
						random_order=True
				)
			]
		)

		# - Apply (flip + rotate) always + scale (50%) + blur (50%) + noise (50%)
		augmenter_simclr2= iaa.Sequential(
			[
				iaa.OneOf([iaa.Fliplr(1.0), iaa.Flipud(1.0)]),
  			iaa.Affine(rotate=(-90, 90), mode='constant', cval=0.0),
				iaa.Sometimes(0.5, iaa.Affine(scale=(0.5, 1.0), mode='constant', cval=0.0)),
				iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.1, 2.0))),
				iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.1)))
			]
		)

		# - Apply flip (66%) + rotate (always) + scale/blur/noise (75%)
		augmenter_simclr3= iaa.Sequential(
			[
				iaa.OneOf([iaa.Fliplr(1.0), iaa.Flipud(1.0), iaa.Noop()]),
  			iaa.Affine(rotate=(-90, 90), mode='constant', cval=0.0),
				iaa.OneOf(
					[
						iaa.Affine(scale=(0.5, 1.0), mode='constant', cval=0.0),
						iaa.GaussianBlur(sigma=(0.1, 2.0)),
						iaa.AdditiveGaussianNoise(scale=(0, 0.1)),
						iaa.Noop()
					]
				)
			]
		)

		# - Apply flip (66%) + rotate (always) + scale/blur/noise (50%)
		augmenter_simclr4= iaa.Sequential(
			[
				iaa.OneOf([iaa.Fliplr(1.0), iaa.Flipud(1.0), iaa.Noop()]),
  			iaa.Affine(rotate=(-90, 90), mode='constant', cval=0.0),
				iaa.Sometimes(0.5, 
					iaa.OneOf(
						[
							iaa.Affine(scale=(0.5, 1.0), mode='constant', cval=0.0),
							iaa.GaussianBlur(sigma=(0.1, 2.0)),
							iaa.AdditiveGaussianNoise(scale=(0, 0.1)),
						]
					),
					iaa.Noop()
				)
			]
		)
	
		# - Set augmenter chosen
		if choice=='cae':
			self.augmenter= augmenter_cae
		elif choice=='cnn':
			self.augmenter= augmenter_cnn
		elif choice=='simclr':
			self.augmenter= augmenter_simclr4
		else:
			logger.warn("Unknown choice (%s), setting CAE augmenter..." % (choice))
			self.augmenter= augmenter_cae


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

	def read_data(self, index, resize=True, nx=64, ny=64, normalize=True, scale_to_abs_max=False, scale_to_max=False, augment=False, log_transform=False, scale=False, scale_factors=[], standardize=False, means=[], sigmas=[], chan_divide=False, chan_mins=[], erode=False, erode_kernel=5, subtract_bkg_and_clip=False):	
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

		# - Subtract bkg from ref channel?
		#   NB: Prefer to do it before image augmentation and resize
		if subtract_bkg_and_clip:
			if sdata.subtract_bkg_and_clip(limit_to_chref=True, chref=0, sigma_bkg=3, sigma_clip=1)<0:
				logger.error("Failed to chan divide source image %d!" % index)
				return None

		# - Log-tranform image?
		if log_transform:
			if sdata.log_transform_imgs()<0:
				logger.error("Failed to log-transform source image %d!" % index)
				return None

		# - Channel division?
		#   NB: Prefer to do it before image augmentation and resize
		if chan_divide:
			if sdata.divide_imgs(chref=0, logtransf=log_transform, strip_chref=False, trim=False, trim_min=-6, trim_max=6)<0:
				logger.error("Failed to chan divide source image %d!" % index)
				return None

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

		# - Fix negative images?
		if self.fix_negative_imgs:
			logger.debug("Fixing negative images (if any) in source data %d ..." % index)
			if sdata.fix_negative_imgs()<0:
				logger.error("Failed to fix negative images in source data %d!" % index)
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

		# - Mask image borders?
		if sdata.mask_borders(mask_fract=0.7)<0:
			logger.error("Failed to mask source image borders %d!" % index)
			return None

		
		return sdata


	###################################
	##     GENERATE SIMCLR DATA
	###################################
	def __generate_simclr_data(self, data_index, resize=True, nx=64, ny=64, normalize=True, scale_to_abs_max=False, scale_to_max=False, log_transform=False, scale=False, scale_factors=[], standardize=False, means=[], sigmas=[], chan_divide=False, chan_mins=[], erode=False, erode_kernel=5, subtract_bkg_and_clip=False):
		""" Generate augmented data pair for SimCLR """

		# - Read data at index and create pair item no. #1
		sdata_1= self.read_data(
			data_index, 
			resize=resize, nx=nx, ny=ny,
			normalize=normalize, scale_to_abs_max=scale_to_abs_max, scale_to_max=scale_to_max, 
			augment=True,
			log_transform=log_transform,
			scale=scale, scale_factors=scale_factors,
			standardize=standardize, means=means, sigmas=sigmas,
			chan_divide=chan_divide, chan_mins=chan_mins,
			erode=erode, erode_kernel=erode_kernel,
			subtract_bkg_and_clip=subtract_bkg_and_clip
		)
		if sdata_1 is None:
			logger.warn("Failed to read source data at index %d (pair item #1)!" % (data_index))
			return None

		if sdata_1.img_cube is None:
			logger.warn("Failed to read source data cube at index %d (pair item #1)!" % (data_index))
			return None

		# - Read data at index and create pair item no. #2
		sdata_2= self.read_data(
			data_index, 
			resize=resize, nx=nx, ny=ny,
			normalize=normalize, scale_to_abs_max=scale_to_abs_max, scale_to_max=scale_to_max, 
			augment=True,
			log_transform=log_transform,
			scale=scale, scale_factors=scale_factors,
			standardize=standardize, means=means, sigmas=sigmas,
			chan_divide=chan_divide, chan_mins=chan_mins,
			erode=erode, erode_kernel=erode_kernel,
			subtract_bkg_and_clip=subtract_bkg_and_clip
		)
		if sdata_2 is None:
			logger.warn("Failed to read source data at index %d (pair item #2)!" % (data_index))
			return None

		if sdata_2.img_cube is None:
			logger.warn("Failed to read source data cube at index %d (pair item #2)!" % (data_index))
			return None

		return sdata_1, sdata_2

	###################################
	##     GENERATE DATA FOR TRAINING
	###################################
	def data_generator(self, batch_size=32, shuffle=True, resize=True, nx=64, ny=64, normalize=True, scale_to_abs_max=False, scale_to_max=False, augment=False, log_transform=False, scale=False, scale_factors=[], standardize=False, means=[], sigmas=[], chan_divide=False, chan_mins=[], erode=False, erode_kernel=5, outdata_choice='cae', classtarget_map={}, nclasses=7, subtract_bkg_and_clip=False):
		""" Generator function reading nsamples images from disk and returning to caller """
	
		nb= 0
		data_index= -1
		data_indexes= np.arange(0,self.datasize)
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
					erode=erode, erode_kernel=erode_kernel,
					subtract_bkg_and_clip=subtract_bkg_and_clip
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
				logger.debug("Data %d shape=(%d,%d,%d)" % (data_index, data_shape[0], data_shape[1], data_shape[2]))
				
				# - Set class targets
				class_id= sdata.id
				target_id= class_id
				if classtarget_map:
					target_id= classtarget_map[class_id]
				
				#print("--> class_id")
				#print(class_id)
				#print("--> target_id")
				#print(target_id)

				# - Generate pairs of augmented data for SimCLR
				if outdata_choice=='simclr':
					data_pair= self.__generate_simclr_data(
						data_index,
						resize=resize, nx=nx, ny=ny, 
						normalize=normalize, scale_to_abs_max=scale_to_abs_max, scale_to_max=scale_to_max, 
						log_transform=log_transform, 
						scale=scale, scale_factors=scale_factors, 
						standardize=standardize, means=means, sigmas=sigmas,
						chan_divide=chan_divide, chan_mins=chan_mins,
						erode=erode, erode_kernel=erode_kernel,
						subtract_bkg_and_clip=subtract_bkg_and_clip
					)
					if data_pair is None:
						logger.warn("Failed to read source data cube at index %d and generate data pairs, skip to next ..." % data_index)
						continue
					sdata_1= data_pair[0]
					sdata_2= data_pair[1]

				# - Initialize return data
				if nb==0:
					inputs= np.zeros(inputs_shape, dtype=np.float32)
					if outdata_choice=='simclr':
						# - The ref implementation (https://github.com/mwdhont/SimCLRv1-keras-tensorflow/blob/master/DataGeneratorSimCLR.py)
						#   uses a dimension (2*batch, 1, ny, nx, nchan), so that returned inputs is a list of len(2*batch) and item passed to encoder has shape (1,ny,nx,nchan) (NB: batch size=1)
						inputs_simclr_shape= (2*batch_size, 1) + data_shape # original ref
						inputs_simclr= np.empty(inputs_simclr_shape, dtype=np.float32)
						labels_ab_aa = np.zeros((batch_size, 2 * batch_size))
						labels_ba_bb = np.zeros((batch_size, 2 * batch_size))

					target_ids= []
				
				# - Update inputs
				inputs[nb]= sdata.img_cube
				if outdata_choice=='simclr':
					# - The ref implementation (https://github.com/mwdhont/SimCLRv1-keras-tensorflow/blob/master/DataGeneratorSimCLR.py)
					#   shuffles the position of augmented image pair
					inputs_simclr[nb]= sdata_1.img_cube
					inputs_simclr[nb + batch_size]= sdata_2.img_cube
					labels_ab_aa[nb, nb] = 1
					labels_ba_bb[nb, nb] = 1

				target_ids.append(target_id)
				nb+= 1


				##### DEBUG ############
				#pix_x= 31
				#pix_y= 31

				#data_masked= np.ma.masked_equal(sdata.img_cube, 0.0, copy=False)
				#data_min= data_masked.min()
				#data_max= data_masked.max()

				#data_mins= []
				#data_maxs= []
				#for i in range(sdata.img_cube.shape[-1]):
				#	data_masked_ch= np.ma.masked_equal(sdata.img_cube[:,:,i], 0.0, copy=False)
				#	data_min_ch= data_masked_ch.min()
				#	data_max_ch= data_masked_ch.max()
				#	data_mins.append(data_min_ch)
				#	data_maxs.append(data_max_ch)

				#print("== GENERATOR data min/max (after norm) ==")
				#print(data_min)
				#print(data_max)

				#print("== GENERATOR data mins/maxs (after norm) ==")
				#print(data_mins)
				#print(data_maxs)

				#print("== GENERATOR pixels (after norm) ==")
				#for i in range(sdata.img_cube.shape[-1]):
				#	print("--> ch%d" % (i+1))
				#	print(sdata.img_cube[pix_y,pix_x,i])
				##########################

				# - Return data if number of batch is reached and restart the batch
				if nb>=batch_size:
					#print("inputs.shape")
					#print(inputs.shape)

					logger.debug("Batch size (%d) reached, yielding generated data of size (%d,%d,%d,%d) ..." % (nb,inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3]))

					if outdata_choice=='sdata':
						yield inputs, sdata
					elif outdata_choice=='inputs':
						yield inputs
					elif outdata_choice=='cae':
						yield inputs, inputs
					elif outdata_choice=='cnn':
						output_targets= to_categorical(np.array(target_ids), num_classes=nclasses)
						#print(output_targets)
						#print(output_targets.shape)
						yield inputs, output_targets
					elif outdata_choice=='simclr':
						#y = tf.concat([labels_ab_aa, labels_ba_bb], 1)
						y= np.concatenate([labels_ab_aa, labels_ba_bb], 1)
						#print("== inputs_simclr shape ==")
        		#print(inputs_simclr.shape)
						#print("== y shape ==") 
						#print(y.shape)
        		#print("")
						yield list(inputs_simclr), y # original implementation: returns a list (len=2xbatch_size) of arrays of shape (1, ny, nx, nchan). Each Input layer takes one list entryas input.
						#yield inputs_simclr
					else:
						logger.warn("Unknown outdata_choice (%s), returning inputs ..." % (outdata_choice))
						yield inputs


					#if retsdata:
					#	yield inputs, sdata
					#else:
					#	if ret_classtargets:
					#		output_targets= to_categorical(np.array(target_ids), num_classes=nclasses)
					#		yield inputs, output_targets
					#	else:
					#		yield inputs, inputs

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
			

