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
from astropy.stats import sigma_clipped_stats

## SKIMAGE
from skimage.util import img_as_float64

## IMG AUG
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
##     PREPROCESSOR CLASS
##############################
class DataPreprocessor(object):
	""" Data pre-processor class """

	def __init__(self, stages):
		""" Create a data pre-processor object """
	
		# - stages is a list of pre-processing instances (e.g. MinMaxNormalizer, etc).
		#   NB: First element is the first stage to be applied to data.
		self.fcns= [] # list of pre-processing functions
		for stage in stages: 
			self.fcns.append(stage.__call__)

		# - Reverse list as fcn compose take functions in the opposite order
		self.fcns.reverse()
		#print(self.fcns)

		# - Create pipeline
		self.pipeline= Utils.compose_fcns(*self.fcns)

	def __call__(self, data):
		""" Apply sequence of pre-processing steps """
		return self.pipeline(data)

	def disable_augmentation(self):
		""" Disable augmentation pre-processing (if existing) """

		# - Create a list without augmenter method
		recreate_pipeline= False
		fcns_new= []
		for i in range(len(self.fcns)):
			is_augmenter= isinstance(self.fcns[i].__self__, Augmenter)
			if is_augmenter:
				recreate_pipeline= True
			else:
				fcns_new.append(self.fcns[i])

		# - Recreate pipeline?
		if recreate_pipeline:
			logger.info("Recreating pipeline with these pre-processing stages ...")
			print(fcns_new)
			self.fcns= fcns_new
			self.pipeline= Utils.compose_fcns(*self.fcns)
		

##############################
##     MinMaxNormalizer
##############################
class MinMaxNormalizer(object):
	""" Normalize each image channel to range  """

	def __init__(self, norm_min=0, norm_max=1, **kwparams):
		""" Create a data pre-processor object """
			
		# - Set parameters
		self.norm_min= norm_min
		self.norm_max= norm_max


	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Normalize data
		data_norm= np.copy(data)

		for i in range(data.shape[-1]):
			data_ch= data[:,:,i]
			cond= np.logical_and(data_ch!=0, np.isfinite(data_ch))
			data_ch_1d= data_ch[cond]
			data_ch_min= data_ch_1d.min()
			data_ch_max= data_ch_1d.max()
			data_ch_norm= (data_ch-data_ch_min)/(data_ch_max-data_ch_min) * (self.norm_max-self.norm_min) + self.norm_min
			data_ch_norm[~cond]= 0 # Restore 0 and nans set in original data
			data_norm[:,:,i]= data_ch_norm

		return data_norm

##############################
##   AbsMinMaxNormalizer
##############################
class AbsMinMaxNormalizer(object):
	""" Normalize each image channel to range using absolute min/max among all channels and not per-channel """

	def __init__(self, norm_min=0, norm_max=1, **kwparams):
		""" Create a data pre-processor object """
			
		# - Set parameters
		self.norm_min= norm_min
		self.norm_max= norm_max

	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Find absolute min & max across all channels
		#   NB: Excluding masked pixels (=0, & NANs)
		cond= np.logical_and(data!=0, np.isfinite(data))
		data_masked= np.ma.masked_where(~cond, data, copy=False)
		data_min= data_masked.min()
		data_max= data_masked.max()

		# - Normalize data
		data_norm= (data-data_min)/(data_max-data_min) * (self.norm_max-self.norm_min) + self.norm_min
		data_norm[~cond]= 0 # Restore 0 and nans set in original data
		
		return data_norm



##############################
##   MaxScaler
##############################
class MaxScaler(object):
	""" Divide each image channel by their maximum value """

	def __init__(self, **kwparams):
		""" Create a data pre-processor object """

	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Find max for each channel
		#   NB: Excluding masked pixels (=0, & NANs)
		cond= np.logical_and(data!=0, np.isfinite(data))
		data_masked= np.ma.masked_where(~cond, data, copy=False)
		data_max= data_masked.max(axis=(0,1)).data

		# - Scale data
		data_scaled= data/data_max
		data_scaled[~cond]= 0 # Restore 0 and nans set in original data
		
		return data_scaled


##############################
##   AbsMaxScaler
##############################
class AbsMaxScaler(object):
	""" Divide each image channel by their absolute maximum value """

	def __init__(self, **kwparams):
		""" Create a data pre-processor object """

	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Find absolute max
		#   NB: Excluding masked pixels (=0, & NANs)
		cond= np.logical_and(data!=0, np.isfinite(data))
		data_masked= np.ma.masked_where(~cond, data, copy=False)
		data_max= data_masked.max()

		# - Scale data
		data_scaled= data/data_max
		data_scaled[~cond]= 0 # Restore 0 and nans set in original data
		
		return data_scaled


##############################
##   ChanMaxScaler
##############################
class ChanMaxScaler(object):
	""" Divide each image channel by selected channel maximum value """

	def __init__(self, chref=0, use_mask_box=False, mask_fract=0.5, **kwparams):
		""" Create a data pre-processor object """

		self.chref= chref
		self.use_mask_box= use_mask_box
		self.mask_fract= mask_fract

	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		cond= np.logical_and(data!=0, np.isfinite(data))

		# - Find selected channel max
		#   NB: Excluding masked pixels (=0, & NANs)
		data_ch= data[:,:,self.chref]
		if self.use_mask_box:
			data_shape= data_ch.shape
			xc= int(data_shape[1]/2)
			yc= int(data_shape[0]/2)
			dy= int(data_shape[0]*self.mask_fract/2.)
			dx= int(data_shape[1]*self.mask_fract/2.)
			xmin= xc - dx
			xmax= xc + dx
			ymin= yc - dy
			ymax= yc + dy
			logger.debug("Using box x[%d,%d] y[%d,%d] to compute chan max ..." % (xmin,xmax,ymin,ymax))
			data_ch= data[ymin:ymax, xmin:xmax, self.chref]
		
		cond_ch= np.logical_and(data_ch!=0, np.isfinite(data_ch))		
		data_masked= np.ma.masked_where(~cond_ch, data_ch, copy=False)
		data_max= data_masked.max()
		logger.debug("Chan %d max: %s" % (self.chref, str(data_max)))
	
		# - Check that channels are not entirely negatives
		for i in range(data.shape[-1]):
			data_ch= data[:,:,i]
			if self.use_mask_box:
				data_ch= data[ymin:ymax, xmin:xmax,i]
			cond_ch= np.logical_and(data_ch!=0, np.isfinite(data_ch))
			data_ch_1d= data_ch[cond_ch]
			data_ch_max= data_ch_1d.max()
			if data_ch_max<=0 or not np.isfinite(data_ch_max):
				logger.warn("Chan %d max is <=0 or not finite, returning None!" % (i))
				return None

		# - Scale data
		data_scaled= data/data_max
		data_scaled[~cond]= 0 # Restore 0 and nans set in original data

		return data_scaled

##############################
##   MinShifter
##############################
class MinShifter(object):
	""" Shift data to min, e.g. subtract min from each pixel """

	def __init__(self, **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		self.chid= -1 # do for all channels, otherwise on just selected channel
		if 'chid' in kwparams:	
			self.chid= kwparams['chid']
		
	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Loop over channels and shift
		data_shifted= np.copy(data)

		for i in range(data.shape[-1]):
			if self.chid!=-1 and i!=self.chid:
				continue
			data_ch= data[:,:,i]
			cond= np.logical_and(data_ch!=0, np.isfinite(data_ch))
			data_ch_1d= data_ch[cond]
			data_ch_min= data_ch_1d.min()
			data_ch_shifted= (data_ch-data_ch_min)
			data_ch_shifted[~cond]= 0 # Set 0 and nans in original data to min
			data_shifted[:,:,i]= data_ch_shifted

		return data_shifted


##############################
##   Shifter
##############################
class Shifter(object):
	""" Shift data to input value """

	def __init__(self, offsets, **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		self.offsets= offsets
		
		
	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Check size of offsets
		nchannels= data.shape[2]
		noffsets= len(self.offsets)
		if noffsets<=0 or noffsets!=nchannels:
			logger.error("Empty offsets or size different from data channels!")
			return None

		# - Shift data
		cond= np.logical_and(data!=0, np.isfinite(data))
		data_shifted= (data-self.offsets)
		data_shifted[~cond]= 0

		return data_shifted


##############################
##   Standardizer
##############################
class Standardizer(object):
	""" Standardize data according to given means and sigmas """

	def __init__(self, means, sigmas, **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		self.means= means
		self.sigmas= sigmas

	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Check size of means/sigmas
		nchannels= data.shape[2]
		nmeans= len(self.means)
		if nmeans<=0 or nmeans!=nchannels:
			logger.error("Empty means or size different from data channels!")
			return None
		nsigmas= len(self.sigmas)
		if nsigmas<=0 or nsigmas!=nchannels:
			logger.error("Empty sigmas or size different from data channels!")
			return None

		# - Transform data
		cond= np.logical_and(data!=0, np.isfinite(data))
		data_norm= (data-self.means)/self.sigmas
		data_norm[~cond]= 0

		return data_norm

##############################
##   NegativeDataFixer
##############################
class NegativeDataFixer(object):
	""" Shift data to min for entirely negative channels """

	def __init__(self, **kwparams):
		""" Create a data pre-processor object """

	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Find negative channels
		data_shifted= np.copy(data)

		for i in range(data.shape[-1]):
			data_ch= data[:,:,i]
			cond= np.logical_and(data_ch!=0, np.isfinite(data_ch))
			data_ch_1d= data_ch[cond]
			data_ch_min= data_ch_1d.min()
			data_ch_max= data_ch_1d.max()

			if data_ch_max>0:
				continue

			data_ch_shifted= (data_ch-data_ch_min)
			data_ch_shifted[~cond]= 0 # Set 0 and nans in original data to min
			data_shifted[:,:,i]= data_ch_shifted
			

		return data_shifted

		
##############################
##   Scaler
##############################
class Scaler(object):
	""" Scale data by a factor """

	def __init__(self, scale_factors, **kwparams):
		""" Create a data pre-processor object """
	
		# - Set parameters
		self.scale_factors= self.scale_factors
		

	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Check size of scale factors
		nchannels= data.shape[2]
		nscales= len(self.scale_factors)
		if nscales<=0 or nscales!=nchannels:
			logger.error("Empty scale factors or size different from data channels!")
			return None

		# - Apply scale factors
		data_scaled= data*self.scale_factors

		return data_scaled


##############################
##   LogStretcher
##############################
class LogStretcher(object):
	""" Apply log transform to data """

	def __init__(self, chid=-1, minmaxnorm=False, data_norm_min=-6, data_norm_max=6, clip_neg=False, **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		self.chid= chid # do for all channels, otherwise skip selected channel
		self.minmaxnorm= minmaxnorm
		self.data_norm_min= data_norm_min
		self.data_norm_max= data_norm_max
		self.clip_neg= clip_neg

	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Loop over channel and convert to lg
		data_transf= np.copy(data)

		for i in range(data.shape[-1]):
			# - Exclude channel?
			if self.chid!=-1 and i==self.chid:
				continue

			data_ch= data[:,:,i]
			cond_ch= np.logical_and(data_ch>0, np.isfinite(data_ch))

			# - Check that there are pixel >0 for log transform
			data_ch_1d= data_ch[cond_ch]
			if data_ch_1d.size<=0:
				logger.warn("All pixels in channel %d are negative and cannot be log transformed, returning None!" % (i))
				return None

			# - Apply log
			data_ch_lg= np.log10(data_ch, where=cond_ch)
			data_ch_lg_1d= data_ch_lg[cond_ch]
			data_ch_lg_min= data_ch_lg_1d.min()
			data_ch_lg[~cond_ch]= 0
			##data_ch_lg[~cond_ch]= data_ch_lg_min

			# - Apply min/max norm data using input parameters
			if self.minmaxnorm:
				data_ch_lg_norm= (data_ch_lg-self.data_norm_min)/(self.data_norm_max-self.data_norm_min)
				if self.self.clip_neg:
					data_ch_lg_norm[data_ch_lg_norm<0]= 0
				data_ch_lg= data_ch_lg_norm
				data_ch_lg[~cond_ch]= 0
				
			# - Set in cube
			data_transf[:,:,i]= data_ch_lg


		return data_transf

##############################
##   BorderMasker
##############################
class BorderMasker(object):
	""" Mask input data at borders """

	def __init__(self, mask_fract=0.7, **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		self.mask_fract= mask_fract

	def __call__(self, data):
		""" Apply transformation and return transformed data """
			
		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Mask all channels at border
		logger.debug("Masking all channels at border (fract=%f) ..." % (self.mask_fract))
		data_masked= np.copy(data)

		for i in range(data.shape[-1]):
			data_ch= data[:,:,i]
			cond= np.logical_and(data_ch!=0, np.isfinite(data_ch))
			data_shape= data_ch.shape
			data_ch_1d= data_ch[cond]
			data_min= data_ch_1d.min()
			mask= np.zeros(data_shape)
			xc= int(data_shape[1]/2)
			yc= int(data_shape[0]/2)
			dy= int(data_shape[0]*self.mask_fract/2.)
			dx= int(data_shape[1]*self.mask_fract/2.)
			xmin= xc - dx
			xmax= xc + dx
			ymin= yc - dy
			ymax= yc + dy
			logger.debug("Masking chan %d (%d,%d) in range x[%d,%d] y[%d,%d]" % (i, data_shape[0], data_shape[1], xmin, xmax, ymin, ymax))
			mask[ymin:ymax, xmin:xmax]= 1
			#data_ch[mask==0]= 0
			data_ch[mask==0]= data_min
			data_masked[:,:,i]= data_ch
	
		return data_masked

##############################
##   BkgSubtractor
##############################
class BkgSubtractor(object):
	""" Subtract background from input data """

	def __init__(self, sigma=3, use_mask_box=False, mask_fract=0.7, chid=-1, **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		self.sigma= sigma
		self.use_mask_box= use_mask_box
		self.mask_fract= mask_fract
		self.chid= chid # -1=do for all channels, otherwise subtract only from selected channel

	def __subtract_bkg(self, data):
		""" Subtract background from channel input """

		cond= np.logical_and(data!=0, np.isfinite(data))
		
		# - Mask region at image center (where source is supposed to be)?
		bkgdata= np.copy(data) 
		if self.use_mask_box:
			data_shape= data.shape
			xc= int(data_shape[1]/2)
			yc= int(data_shape[0]/2)
			dy= int(data_shape[0]*self.mask_fract/2.)
			dx= int(data_shape[1]*self.mask_fract/2.)
			xmin= xc - dx
			xmax= xc + dx
			ymin= yc - dy
			ymax= yc + dy
			logger.debug("Masking data (%d,%d) in range x[%d,%d] y[%d,%d]" % (data_shape[0], data_shape[1], xmin, xmax, ymin, ymax))
			bkgdata[ymin:ymax, xmin:xmax]= 0
	
		# - Compute and subtract mean bkg from data
		logger.debug("Subtracting bkg ...")
		cond_bkg= np.logical_and(bkgdata!=0, np.isfinite(bkgdata))
		bkgdata_1d= bkgdata[cond_bkg]
		logger.debug("--> bkgdata min/max=%s/%s" % (str(bkgdata_1d.min()), str(bkgdata_1d.max())))

		bkgval, _, _ = sigma_clipped_stats(bkgdata_1d, sigma=self.sigma)

		data_bkgsub= data - bkgval
		data_bkgsub[~cond]= 0
		cond_bkgsub= np.logical_and(data_bkgsub!=0, np.isfinite(data_bkgsub))
		data_bkgsub_1d= data_bkgsub[cond_bkgsub]

		logger.debug("--> data min/max (after bkgsub)=%s/%s (bkg=%s)" % (str(data_bkgsub_1d.min()), str(data_bkgsub_1d.max()), str(bkgval)))

		return data_bkgsub


	def __call__(self, data):
		""" Apply transformation and return transformed data """
			
		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Loop over channels and get bgsub data
		data_bkgsub= np.copy(data)

		for i in range(data.shape[-1]):
			if self.chid!=-1 and i!=self.chid:
				continue	
			data_ch_bkgsub= self.__subtract_bkg(data[:,:,i])
			data_bkgsub[:,:,i]= data_ch_bkgsub

		return data_bkgsub


##############################
##   SigmaClipper
##############################
class SigmaClipper(object):
	""" Clip all pixels below input sigma """

	def __init__(self, sigma=1.0, chid=-1, **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		self.sigma= sigma
		self.chid= chid # -1=do for all channels, otherwise clip only selected channel

	def __clip(self, data):
		""" Clip channel input """

		cond= np.logical_and(data!=0, np.isfinite(data))
		data_1d= data[cond]

		# - Clip all pixels that are below sigma clip
		logger.debug("Clipping all pixels below %f sigma ..." % (self.sigma))
		clipmean, _, _ = sigma_clipped_stats(data_1d, sigma=self.sigma)

		data_clipped= np.copy(data)
		data_clipped[data_clipped<clipmean]= clipmean
		data_clipped[~cond]= 0
		cond_clipped= np.logical_and(data_clipped!=0, np.isfinite(data_clipped))
		data_clipped_1d= data_clipped[cond_clipped]

		logger.debug("--> data min/max (after sigmaclip)=%s/%s (clipmean=%s)" % (str(data_clipped_1d.min()), str(data_clipped_1d.max()), str(clipmean)))

		return data_clipped 
		

	def __call__(self, data):
		""" Apply transformation and return transformed data """
			
		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Loop over channels and get bgsub data
		data_clipped= np.copy(data)

		for i in range(data.shape[-1]):
			if self.chid!=-1 and i!=self.chid:
				continue	
			data_ch_clipped= self.__clip(data[:,:,i])
			data_clipped[:,:,i]= data_ch_clipped

		return data_clipped



##############################
##   Resizer
##############################
class Resizer(object):
	""" Resize image to desired size """

	def __init__(self, resize_size, preserve_range=True, upscale=False, downscale_with_antialiasing=False, set_pad_val_to_min=True, **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		self.resize_size= resize_size
		self.preserve_range= preserve_range
		self.upscale= upscale # Upscale images to resize size when original image size is smaller than desired size. If false, pad to reach desired size
		self.downscale_with_antialiasing=downscale_with_antialiasing  # Use antialiasing when down-scaling an image
		self.set_pad_val_to_min= set_pad_val_to_min
		
	def __call__(self, data):
		""" Apply transformation and return transformed data """
			
		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Check if resizing is needed
		data_shape= data.shape
		nx= data_shape[1]
		ny= data_shape[0]
		nchannels= data_shape[2]
		is_same_size= (nx==self.resize_size) and (ny==self.resize_size)
		if is_same_size:
			logger.debug("Images have already the desired size (%d,%d), nothing to be done..." % (ny, nx))
			return data

		# - Select resizing options
		max_dim= self.resize_size
		min_dim= self.resize_size
		if not self.upscale:
			min_dim=None

		downscaling= (nx>self.resize_size) and (ny>self.resize_size)
		antialiasing= False
		if downscaling and self.downscale_with_antialiasing:
			antialiasing= True

		interp_order= 1 # 1=bilinear, 2=biquadratic, 3=bicubic, 4=biquartic, 5=biquintic

		# - Resize data
		try:
			#data_resized= Utils.resize_img(data, (self.ny, self.nx, nchannels), preserve_range=self.preserve_range)
			
			try: # work for skimage<=0.15.0
				ret= Utils.resize_img_v2(data, 
					min_dim=min_dim, max_dim=max_dim, min_scale=None, mode="square", 
					order=interp_order, anti_aliasing=antialiasing, 
					preserve_range=self.preserve_range
				)
			except:
				ret= Utils.resize_img_v2(img_as_float64(data), 
					min_dim=min_dim, max_dim=max_dim, min_scale=None, mode="square", 
					order=interp_order, anti_aliasing=antialiasing, 
					preserve_range=self.preserve_range
				)

			data_resized= ret[0]
			#window= ret[1]
			#scale= ret[2] 
			#padding= ret[3] 
			#crop= ret[4]

		except Exception as e:
			logger.warn("Failed to resize data to size (%d,%d) (err=%s)!" % (self.resize_size, self.resize_size, str(e)))
			return None

		if data_resized is None:
			logger.warn("Resized data is None, failed to resize to size (%d,%d) (see logs)!" % (self.resize_size, self.resize_size))

		if self.set_pad_val_to_min:
			for i in range(data_resized.shape[-1]):
				data_ch= data_resized[:,:,i]
				cond_ch= np.logical_and(data_ch!=0, np.isfinite(data_ch))
				data_ch_1d= data_ch[cond_ch]
				data_min= data_ch_1d.min()
				data_ch[~cond_ch]= data_min
				data_resized[:,:,i]= data_ch

		return data_resized



##############################
##   ChanDivider
##############################
class ChanDivider(object):
	""" Divide channel by reference channel """

	def __init__(self, chref=0, logtransf=False, strip_chref=False, trim=False, trim_min=-6, trim_max=6, **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		self.chref= chref
		self.logtransf= logtransf
		self.strip_chref= strip_chref
		self.trim= trim
		self.trim_min= trim_min
		self.trim_max= trim_max
		
	def __call__(self, data):
		""" Apply transformation and return transformed data """
			
		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Init ref channel
		cond= np.logical_and(data!=0, np.isfinite(data)) 
		data_ref= np.copy(data[:,:,self.chref])
		cond_ref= np.logical_and(data_ref!=0, np.isfinite(data_ref))

		# - Divide other channels by reference channel
		data_norm= np.copy(data)
		data_denom= np.copy(data_ref)
		data_denom[data_denom==0]= 1

		for i in range(data_norm.shape[-1]):
			if i==self.chref:
				data_norm[:,:,i]= np.copy(data_ref)
			else:
				logger.debug("Divide channel %d by reference channel %d ..." % (i, self.chref))
				dn= data_norm[:,:,i]/data_denom
				dn[~cond_ref]= 0 # set ratio to zero if ref pixel flux was zero or nan
				data_norm[:,:,i]= dn

		data_norm[~cond]= 0

		# - Apply log transform to ratio channels?
		if self.logtransf:
			logger.debug("Applying log-transform to channel ratios ...")
			data_transf= np.copy(data_norm)
			data_transf[data_transf<=0]= 1
			data_transf_lg= np.log10(data_transf)
			data_transf= data_transf_lg
			data_transf[~cond]= 0

			if self.trim:
				data_transf[data_transf>self.trim_max]= self.trim_max
				data_transf[data_transf<self.trim_min]= self.trim_min

			data_transf[:,:,self.chref]= data_norm[:,:,self.chref]
			data_norm= data_transf

		# - Strip ref channel 
		if self.strip_chref:
			data_norm_striprefch= np.delete(data_norm, chref, axis=2)
			data_norm= data_norm_striprefch
			
		return data_norm



##############################
##   Augmenter
##############################
class Augmenter(object):
	""" Perform image augmentation according to given model """

	def __init__(self, augmenter_choice="cae", augmenter=None, **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		if augmenter is None:
			self.__set_augmenters(augmenter_choice)
		else:
			self.augmenter= augmenter

	######################################
	##     DEFINE PREDEFINED AUGMENTERS
	######################################
	def __set_augmenters(self, choice='cae'):
		""" Define and set augmenters """

		# - Define augmenter for Conv Autoencoder
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
  			iaa.Affine(rotate=(-90, 90), mode='constant', cval=0.0)
				#iaa.Sometimes(0.5, iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, mode='constant', cval=0.0))
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

		
	def __call__(self, data):
		""" Apply transformation and return transformed data """
			
		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Make augmenters deterministic to apply similarly to images and masks
		##augmenter_det = self.augmenter.to_deterministic()

		# - Augment data cube
		try:
			data_aug= self.augmenter.augment_image(data)
		except Exception as e:
			logger.error("Failed to augment data (err=%s)!" % str(e))
			return None

		return data_aug



##############################
##   MaskShrinker
##############################
class MaskShrinker(object):
	""" Shrink input data mask using an erosion operation """

	def __init__(self, kernsize, **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		self.kernsize= kernsize

	def __call__(self, data):
		""" Apply transformation and return transformed data """
			
		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Define erosion operation
		structel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernsize, self.kernsize))
		#structel= cv2.getStructuringElement(cv2.MORPH_RECTANGLE, (self.kernsize, self.kernsize))

		# - Create erosion masks and apply to input data
		data_shrinked= np.copy(data)

		for i in range(data.shape[-1]):
			mask= np.logical_and(data[:,:,i]!=0, np.isfinite(data[:,:,i])).astype(np.uint8)
			mask= mask.astype(np.uint8)
			mask_eroded = cv2.erode(mask, structel, iterations = 1)
			
			img_eroded= data[:,:,i]
			img_eroded[mask_eroded==0]= 0
			data_shrinked[:,:,i]= img_eroded
			
		return data_shrinked
		

