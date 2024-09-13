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
np.bool = np.bool_  # This is to fix numpy deprecation (imgaug require older numpy version relying on np.numpy statement)

import random
import math
import logging
from collections import Counter
import json

## ASTROPY MODULES 
from astropy.io import ascii
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip
from astropy.visualization import ZScaleInterval, MinMaxInterval, PercentileInterval, HistEqStretch

## SKIMAGE
import skimage
from skimage.util import img_as_float64
from skimage.exposure import adjust_sigmoid, rescale_intensity, equalize_hist, equalize_adapthist

## SCIPY
import scipy

## IMG AUG
import imgaug
from imgaug import augmenters as iaa
from imgaug import parameters as iap

## OPENCV
import cv2
cv2.setNumThreads(1) # workaround to avoid potential conflicts between TF and OpenCV multithreading (parallel_impl.cpp (240) WorkerThread 18: Can't spawn new thread: res = 11)


## PACKAGE MODULES
from .utils import Utils

##############################
##     GLOBAL VARS
##############################
from sclassifier import logger


## TENSORFLOW MODULES
import tensorflow as tf
#try:
#	import tensorflow_models as tfm
#except:
#	logger.warn("Cannot import module tensorflow_models, not a problem if you don't use ColorJitterer pre-processor...")


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
		#self.pipeline= Utils.compose_fcns(*self.fcns)
		self.pipeline= Utils.compose_fcns_v2(*self.fcns)

	#def __call__(self, data):
	#	""" Apply sequence of pre-processing steps """
	#	return self.pipeline(data)
		
	def __call__(self, data, **kwargs):
		""" Apply sequence of pre-processing steps """
		return self.pipeline(data, **kwargs)

	def disable_augmentation(self):
		""" Disable augmentation pre-processing (if existing) """

		# - Create a list without augmenter method
		recreate_pipeline= False
		fcns_new= []
		for i in range(len(self.fcns)):
			#is_augmenter= isinstance(self.fcns[i].__self__, Augmenter)
			is_augmenter= isinstance(self.fcns[i].__self__, Augmenter) or isinstance(self.fcns[i].__self__, Augmenters)
			if is_augmenter:
				recreate_pipeline= True
			else:
				fcns_new.append(self.fcns[i])

		# - Recreate pipeline?
		if recreate_pipeline:
			logger.info("Recreating pipeline with these pre-processing stages ...")
			print(fcns_new)
			self.fcns= fcns_new
			#self.pipeline= Utils.compose_fcns(*self.fcns)
			self.pipeline= Utils.compose_fcns_v2(*self.fcns)
		

#######################################
##     Custom Augmenters for imgaug
#######################################
class ZScaleAugmenter(iaa.meta.Augmenter):
	""" Apply ZScale transform to image as augmentation step """
	
	def __init__(self, 
		contrast=0.25, 
		random_contrast=False, 
		random_contrast_per_ch=False, contrast_min=0.1, contrast_max=0.7, 
		seed=None, name=None, random_state="deprecated", deterministic="deprecated"
	):
		""" Build class """

		# - Set parent class parameters
		super(ZScaleAugmenter, self).__init__(
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic
		)

		# - Set class parameters 
		if contrast<=0 or contrast>1:
			raise Exception("Expected contrast to be [0,1], got %f!" % (contrast))

		self.contrast= contrast
		self.random_contrast= random_contrast
		self.random_contrast_per_ch= random_contrast_per_ch
		self.contrast_min= contrast_min
		self.contrast_max= contrast_max
		self.seed= seed

	def get_parameters(self):
		""" Get class parameters """
		return [self.contrast, self.random_contrast, self.random_contrast_per_ch, self.contrast_min, self.contrast_max]

	def _augment_batch_(self, batch, random_state, parents, hooks):
		""" Augment batch of images """
	
		# - Check input batch
		if batch.images is None:
			return batch

		images = batch.images
		nb_images = len(images)
		contrasts= []

		# - Set random seed if given
		if self.seed is not None:
			np.random.seed(self.seed)

		# - Loop over image batch
		for i in range(nb_images):
			image= images[i]
			nb_channels = image.shape[2]

			# - Set zscale contrasts (fixed or random)
			if not contrasts:
				if self.random_contrast:
					if self.random_contrast_per_ch:
						for k in range(nb_channels):
							contrast_rand= np.random.uniform(low=self.contrast_min, high=self.contrast_max)
							contrasts.append(contrast_rand)
					else:
						contrast_rand= np.random.uniform(low=self.contrast_min, high=self.contrast_max)
						contrasts= [contrast_rand]*nb_channels
				else:
					contrasts= [self.contrast]*nb_channels

			# - Apply zscale stretch
			logger.debug("Applying zscale transform to batch %d with contrasts %s " % (i+1, str(contrasts)))

			batch.images[i] = self.__get_zscale_image(image, contrasts)
			if batch.images[i] is None:
				raise Exception("ZScale augmented image at batch %d is None!" % (i+1))

		return batch
	
	
	def __get_zscale_image(self, data, contrasts=[]):
		""" Apply z-scale transform to single image (W,H,Nch) """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		cond= np.logical_and(data!=0, np.isfinite(data))

		# - Check contrast dim vs nchans
		nchans= data.shape[-1]
	
		if len(contrasts)<nchans:
			logger.error("Invalid contrasts given (contrast list size=%d < nchans=%d)" % (len(self.contrasts), nchans))
			return None
		
		# - Transform each channel
		data_stretched= np.copy(data)

		for i in range(data.shape[-1]):
			data_ch= data_stretched[:,:,i]
			transform= ZScaleInterval(contrast=contrasts[i]) # able to handle NANs
			data_transf= transform(data_ch)
			data_stretched[:,:,i]= data_transf

		# - Scale data
		data_stretched[~cond]= 0 # Restore 0 and nans set in original data

		return data_stretched
		



class PercentileThrAugmenter(iaa.meta.Augmenter):
	""" Sigma threshold as augmentation step """
	
	def __init__(self, 
		percentile=50, 
		random_percentile=False, 
		random_percentile_per_ch=False, percentile_min=50, percentile_max=60, 
		seed=None, name=None, random_state="deprecated", deterministic="deprecated"
	):
		""" Build class """

		# - Set parent class parameters
		super(PercentileThrAugmenter, self).__init__(
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic
		)

		self.percentile= percentile
		self.random_percentile= random_percentile
		self.random_percentile_per_ch= random_percentile_per_ch
		self.percentile_min= percentile_min
		self.percentile_max= percentile_max
		self.seed= seed

	def get_parameters(self):
		""" Get class parameters """
		return [self.percentile, self.random_percentile, self.random_percentile_per_ch, self.percentile_min, self.percentile_max]

	def _augment_batch_(self, batch, random_state, parents, hooks):
		""" Augment batch of images """
	
		# - Check input batch
		if batch.images is None:
			return batch

		images = batch.images
		nb_images = len(images)
		percentiles= []

		# - Set random seed if given
		if self.seed is not None:
			np.random.seed(self.seed)

		# - Loop over image batch
		for i in range(nb_images):
			image= images[i]
			nb_channels = image.shape[2]

			# - Set percentiles (fixed or random)
			if not percentiles:
				if self.random_percentile:
					if self.random_percentile_per_ch:
						for k in range(nb_channels):
							percentile_rand= np.random.uniform(low=self.percentile_min, high=self.percentile_max)
							percentiles.append(percentile_rand)
					else:
						percentile_rand= np.random.uniform(low=self.percentile_min, high=self.percentile_max)
						percentiles= [percentile_rand]*nb_channels
				else:
					percentiles= [self.percentile]*nb_channels

			# - Apply percentile filtering
			logger.debug("Applying percentile thresholding to batch %d with contrasts %s " % (i+1, str(percentiles)))

			batch.images[i] = self.__get_percentile_thresholded_image(image, percentiles)
			if batch.images[i] is None:
				raise Exception("Percentile-thresholded augmented image at batch %d is None!" % (i+1))

		return batch
	
	
	def __get_percentile_thresholded_image(self, data, percentiles=[]):
		""" Apply percentile thresholding to single image (W,H,Nch) """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		cond= np.logical_and(data!=0, np.isfinite(data))

		# - Check percentiles dim vs nchans
		nchans= data.shape[-1]
	
		if len(percentiles)<nchans:
			logger.error("Invalid percentiles given (percentile list size=%d < nchans=%d)" % (len(self.percentiles), nchans))
			return None
		
		# - Threshold each channel
		data_thresholded= np.copy(data)

		for i in range(data.shape[-1]):
			percentile= percentiles[i]
			data_ch= data_thresholded[:,:,i]
			cond_ch= np.logical_and(data_ch!=0, np.isfinite(data_ch))
			data_ch_1d= data_ch[cond_ch]
			
			#p= np.percentile(data, percentile)
			p= np.percentile(data_ch_1d, percentile)

			data_ch[data_ch<p]= 0
			data_ch[~cond_ch]= 0
	
			data_thresholded[:,:,i]= data_ch

		# - Scale data
		data_thresholded[~cond]= 0 # Restore 0 and nans set in original data

		return data_thresholded



class SigmoidStretchAugmenter(iaa.meta.Augmenter):
	""" Apply sigmoid contrast adjustment as augmentation step """
	
	def __init__(self, 
		cutoff=0.5, 
		gain=10,
		random_gain=False, 
		random_gain_per_ch=False, gain_min=10, gain_max=20, 
		seed=None, name=None, random_state="deprecated", deterministic="deprecated"
	):
		""" Build class """

		# - Set parent class parameters
		super(SigmoidStretchAugmenter, self).__init__(
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic
		)

		self.cutoff= cutoff
		self.gain= gain
		self.random_gain= random_gain
		self.random_gain_per_ch= random_gain_per_ch
		self.gain_min= gain_min
		self.gain_max= gain_max
		self.seed= seed

	def get_parameters(self):
		""" Get class parameters """
		return [self.cutoff, self.gain, self.random_gain, self.random_gain_per_ch, self.gain_min, self.gain_max]

	def _augment_batch_(self, batch, random_state, parents, hooks):
		""" Augment batch of images """
	
		# - Check input batch
		if batch.images is None:
			return batch

		images = batch.images
		nb_images = len(images)
		gains= []

		# - Set random seed if given
		if self.seed is not None:
			np.random.seed(self.seed)

		# - Loop over image batch
		for i in range(nb_images):
			image= images[i]
			nb_channels = image.shape[2]

			# - Set gains (fixed or random)
			if not gains:
				if self.random_gain:
					if self.random_gain_per_ch:
						for k in range(nb_channels):
							gain_rand= np.random.uniform(low=self.gain_min, high=self.gain_max)
							gains.append(gain_rand)
					else:
						gain_rand= np.random.uniform(low=self.gain_min, high=self.gain_max)
						gains= [gain_rand]*nb_channels
				else:
					gains= [self.gain]*nb_channels

			# - Apply sigmoid contrast transform
			logger.debug("Applying sigmoid stretch to batch %d with gain %s " % (i+1, str(gains)))

			batch.images[i] = self.__get_transformed_image(image, gains)
			if batch.images[i] is None:
				raise Exception("Sigmoid stretch augmented image at batch %d is None!" % (i+1))

		return batch
	
	
	def __get_transformed_image(self, data, gains=[]):
		""" Apply sigmoid contrast stretch to single image (W,H,Nch) """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		cond= np.logical_and(data!=0, np.isfinite(data))

		# - Check gains dim vs nchans
		nchans= data.shape[-1]
	
		if len(gains)<nchans:
			logger.error("Invalid gains given (gain list size=%d < nchans=%d)" % (len(self.gains), nchans))
			return None
		
		# - Threshold each channel
		data_transformed= np.copy(data)

		for i in range(data.shape[-1]):
			data_ch= data[:,:,i]
			if data_ch.min()<0:
				data_norm= rescale_intensity(data_ch, out_range=(0.,1.))
				data_ch= data_norm
			data_transformed[:,:,i]= adjust_sigmoid(data_ch, cutoff=self.cutoff, gain=gains[i], inv=False)
	
		# - Scale data
		data_transformed[~cond]= 0 # Restore 0 and nans set in original data

		return data_transformed



class RandomCropResizeAugmenter(iaa.meta.Augmenter):
	""" Apply random crop and resize to image as augmentation step """
	
	def __init__(self, 
		crop_fract_min=0.7, crop_fract_max=1.0, 
		seed=None, name=None, random_state="deprecated", deterministic="deprecated"
	):
		""" Build class """

		# - Set parent class parameters
		super(RandomCropResizeAugmenter, self).__init__(
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic
		)

		# - Set class parameters 
		if crop_fract_min<=0 or crop_fract_min>=crop_fract_max:
			raise Exception("Expected crop_fract_min to be >0 and <crop_frac_max, got %f!" % (crop_fract_min))
		if crop_fract_max>1 or crop_fract_max<=crop_fract_min:
			raise Exception("Expected crop_fract_max to be <=1 and >crop_frac_min, got %f!" % (crop_fract_max))

		self.crop_fract_min= crop_fract_min
		self.crop_fract_max= crop_fract_max
		self.seed= seed

	def get_parameters(self):
		""" Get class parameters """
		return [self.crop_fract_min, self.crop_fract_max]

	def _augment_batch_(self, batch, random_state, parents, hooks):
		""" Augment batch of images """
	
		# - Check input batch
		if batch.images is None:
			return batch

		images = batch.images
		nb_images = len(images)
		contrasts= []

		# - Set random seed if given
		if self.seed is not None:
			np.random.seed(self.seed)

		# - Loop over image batch
		for i in range(nb_images):
			image= images[i]
			nb_channels = image.shape[2]
			ny= image.shape[0]
			nx= image.shape[1]

			# - Set crop size
			crop_fract_rand= np.random.uniform(low=self.crop_fract_min, high=self.crop_fract_max)
			crop_x= math.ceil(crop_fract_rand*nx)
			crop_y= math.ceil(crop_fract_rand*ny)
			
			# - Define augmentation
			crop_aug= iaa.KeepSizeByResize(
				iaa.CropToFixedSize(crop_y, crop_x, position='uniform')
			)

			# - Crop & resize
			logger.debug("Applying crop to batch %d with crop_fract %s " % (i+1, str(crop_fract_rand)))

			batch.images[i] = crop_aug.augment_image(image)
			if batch.images[i] is None:
				raise Exception("Crop augmented image at batch %d is None!" % (i+1))

		return batch
	
	
	
class ColorJitterAugmenter(iaa.meta.Augmenter):
	""" Apply color jitter transform to image as augmentation step """
	
	def __init__(self, 
		brightness=0.8,	
		contrast=0.8,
		saturation=0.8,
		hue=0.2,
		strength=1.0,
		seed=None, name=None, random_state="deprecated", deterministic="deprecated"
	):
		""" Build class """

		# - Set parent class parameters
		super(ColorJitterAugmenter, self).__init__(
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic
		)

		# - Set class parameters
		if brightness<=0:
			raise Exception("Expected brightness to be >0, got %f!" % (brightness))
		if contrast<=0:
			raise Exception("Expected contrast to be >0, got %f!" % (contrast))
		if saturation<=0:
			raise Exception("Expected saturation to be >0, got %f!" % (saturation))
		if hue<=0:
			raise Exception("Expected hue to be >0, got %f!" % (hue))		

		self.brightness= brightness
		self.contrast= contrast
		self.saturation= saturation
		self.hue= hue
		self.strength= strength
		self.seed= seed
		
		# - Define normalizer
		self.normalizer= MinMaxNormalizer(norm_min=0.0, norm_max=1.0)

	def get_parameters(self):
		""" Get class parameters """
		return [self.brightness, self.contrast, self.saturation, self.hue, self.strength]

	def _augment_batch_(self, batch, random_state, parents, hooks):
		""" Augment batch of images """
	
		# - Check input batch
		if batch.images is None:
			return batch

		images = batch.images
		nb_images = len(images)
		
		# - Set random seed if given
		if self.seed is not None:
			np.random.seed(self.seed)

		# - Loop over image batch
		for i in range(nb_images):
			image= images[i]
			nb_channels = image.shape[2]
			
			# - Apply color jitter
			#batch.images[i] = self.__get_transformed_image(image)
			batch.images[i] = self.__get_transformed_image_v2(image)
			
			if batch.images[i] is None:
				raise Exception("Color jitter augmented image at batch %d is None!" % (i+1))

		return batch
	
	def __get_transformed_image(self, image):
		""" Return color jittered image """
		
		# - Generate random brightness/contrast/saturation
		brightness_rand= np.random.uniform( max(0, 1 - self.brightness*self.strength), 1 + self.brightness*self.strength)
		contrast_rand= np.random.uniform( max(0, 1 - self.contrast*self.strength), 1 + self.contrast*self.strength)
		saturation_rand= np.random.uniform( max(0, 1 - self.saturation*self.strength), 1 + self.saturation*self.strength)
		
		# - Apply transform
		logger.debug("Applying color jitter transform with pars (%f, %f, %f) " % (brightness_rand, contrast_rand, saturation_rand))
		restore_original_range= True
		jitterer= ColorJitterer(brightness_rand, contrast_rand, saturation_rand, restore_original_range)	
		image_transf= jitterer(image)
		
		return image_transf
		
	
	def __get_transformed_image_v2(self, image):
		""" Return color jittered image """
		
		# - Check number of channels
		#   NB: saturation & hue applicable only to RGB
		nchans= image.shape[-1]
		
		# - Need a tensor, normalized in range [0,1]
		image_norm= self.normalizer(image)
		x= tf.convert_to_tensor(image_norm)

		# - Apply transforms
		x = tf.image.random_brightness(x, max_delta=self.brightness*self.strength)
		x = tf.image.random_contrast(x, lower=1-self.contrast*self.strength, upper=1+self.contrast*self.strength)
		if nchans==3:
			x = tf.image.random_saturation(x, lower=1-self.saturation*self.strength, upper=1+self.saturation*self.strength)
			x = tf.image.random_hue(x, max_delta=0.2*self.strength)
		x = tf.clip_by_value(x, 0, 1)

		return x.numpy()
		


class SourceRemoverAugmenter(iaa.meta.Augmenter):
	""" Apply source remover transform to image as augmentation step """
	
	def __init__(self, 
		npix_upper_thr_min=200,
		npix_upper_thr_max=600,
		seed=None, name=None, random_state="deprecated", deterministic="deprecated"
	):
		""" Build class """

		# - Set parent class parameters
		super(SourceRemoverAugmenter, self).__init__(
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic
		)

		# - Set class parameters
		if npix_upper_thr_min<=0 or npix_upper_thr_max<=0:
			raise Exception("Expected npix_upper_thr to be >0, got %f/%f!" % (npix_upper_thr_min, npix_upper_thr_max))
		if npix_upper_thr_min>=npix_upper_thr_max:
			raise Exception("Expected npix_upper_thr_min to be <npix_upper_thr_max, got %f/%f!" % (npix_upper_thr_min, npix_upper_thr_max))	
		
		self.npix_upper_thr_min= npix_upper_thr_min
		self.npix_upper_thr_max= npix_upper_thr_max
		self.seed= seed
		self.seed_thr= 4.0
		self.niters= 2
		
		
	def get_parameters(self):
		""" Get class parameters """
		return [self.npix_upper_thr_min, self.npix_upper_thr_max]

	def _augment_batch_(self, batch, random_state, parents, hooks):
		""" Augment batch of images """
	
		# - Check input batch
		if batch.images is None:
			return batch

		images = batch.images
		nb_images = len(images)
		
		# - Set random seed if given
		if self.seed is not None:
			np.random.seed(self.seed)

		# - Loop over image batch
		for i in range(nb_images):
			image= images[i]
			nb_channels = image.shape[2]
			
			# - Generate random npix_min_thr
			npix_max_thr_rand= np.random.uniform(self.npix_upper_thr_min, self.npix_upper_thr_max)
			
			# - Apply source remover
			sremover= SourceRemover(niters=self.niters, seed_thr=self.seed_thr, npix_max_thr=npix_max_thr_rand)
			batch.images[i] = sremover(image)
			
			if batch.images[i] is None:
				raise Exception("Source remover augmented image at batch %d is None!" % (i+1))

		return batch
	
	
	
##############################
##     MinMaxNormalizer
##############################
class MinMaxNormalizer(object):
	""" Normalize each image channel to range  """

	def __init__(self, norm_min=0., norm_max=1., exclude_zeros=True, **kwparams):
		""" Create a data pre-processor object """
			
		# - Set parameters
		self.norm_min= norm_min
		self.norm_max= norm_max
		self.exclude_zeros= exclude_zeros


	def __call__(self, data, **kwargs):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Normalize data
		data_norm= np.copy(data)

		for i in range(data.shape[-1]):
			data_ch= data[:,:,i]
			if self.exclude_zeros:
				cond= np.logical_and(data_ch!=0, np.isfinite(data_ch))
			else:
				cond= np.isfinite(data_ch)
			data_ch_1d= data_ch[cond]
			if data_ch_1d.size==0:
				logger.warn("Size of data_ch%d is zero, returning None!" % (i))
				return None

			data_ch_min= data_ch_1d.min()
			data_ch_max= data_ch_1d.max()
			
			#logger.info("data_ch_min=%f, data_ch_max=%f" % (data_ch_min, data_ch_max))
			
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

	def __call__(self, data, **kwargs):
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

	def __call__(self, data, **kwargs):
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

	def __init__(self, use_mask_box=False, mask_fract=0.5, **kwparams):
		""" Create a data pre-processor object """

		self.use_mask_box= use_mask_box
		self.mask_fract= mask_fract

	def __call__(self, data, **kwargs):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Find absolute max
		#   NB: Excluding masked pixels (=0, & NANs)
		cond= np.logical_and(data!=0, np.isfinite(data))

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
			border_mask= np.zeros(data.shape)
			border_mask[ymin:ymax, xmin:xmax, :]= 1
			cond_max= np.logical_and(cond, border_mask==1)
		else:
			cond_max= cond

		data_masked= np.ma.masked_where(~cond_max, data, copy=False)
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

	def __call__(self, data, **kwargs):
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
		
	def __call__(self, data, **kwargs):
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
		
		
	def __call__(self, data, **kwargs):
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

	def __call__(self, data, **kwargs):
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

	def __call__(self, data, **kwargs):
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
		self.scale_factors= scale_factors
		

	def __call__(self, data, **kwargs):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Check NANs/zero values
		cond= np.logical_and(data!=0, np.isfinite(data))

		# - Check size of scale factors
		nchannels= data.shape[2]
		nscales= len(self.scale_factors)
		if nscales<=0 or nscales!=nchannels:
			logger.error("Empty scale factors or size different from data channels!")
			return None

		# - Apply scale factors
		data_scaled= data*self.scale_factors
		
		# - restore NANs values
		data_scaled[~cond]= 0
		

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

	def __call__(self, data, **kwargs):
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
			badpix_cond= np.logical_or(data_ch==0, ~np.isfinite(data_ch))
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
			##data_ch_lg[~cond_ch]= 0
			data_ch_lg[~cond_ch]= data_ch_lg_min

			# - Apply min/max norm data using input parameters
			if self.minmaxnorm:
				data_ch_lg_norm= (data_ch_lg-self.data_norm_min)/(self.data_norm_max-self.data_norm_min)
				if self.clip_neg:
					data_ch_lg_norm[data_ch_lg_norm<0]= 0
				data_ch_lg= data_ch_lg_norm
				#data_ch_lg[~cond_ch]= 0
				data_ch_lg[badpix_cond]= 0
				
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

	def __call__(self, data, **kwargs):
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
			data_ch[mask==0]= 0
			##data_ch[mask==0]= data_min
			data_masked[:,:,i]= data_ch
	
		return data_masked
		
		
##############################
##     BBoxResizer
##############################
class BBoxResizer(object):
	""" Extract image mask bounding box and eventually resize it to desired size """

	def __init__(self, resize=False, resize_size=64, **kwparams):
		""" Create a data pre-processor object """
			
		# - Set parameters
		self.resize= resize
		self.resize_size= resize_size
		self.downscale_with_antialiasing= False
		

	def __call__(self, data, **kwargs):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		data_shape= data.shape
		nx= data_shape[1]
		ny= data_shape[0]

		# - Create binary mask
		binary_map= np.ones_like(data).astype('uint32')
		binary_map[data==0]= 0
		
		# - Extract components
		label_map= skimage.measure.label(binary_map)
		regprops= skimage.measure.regionprops(label_map, data)
		nregs= len(regprops)
		if not regprops:
			logger.error("No components extracted from image when 1 is expected!")
			return None
		if nregs>1:
			logger.warn("#%d components extracted from image when 1 is expected!" % (nregs))
			return None
			
		regprop= regprops[0]
		
		# - Retrieve image
		try:
			data_bbox= regprop.image_intensity
		except:
			data_bbox= regprop.intensity_image
		
		# - Resize image to desired max size
		if self.resize:
			logger.debug("Resizing image to size %d ..." % (self.resize_size))
			
			max_dim= self.resize_size
			min_dim= self.resize_size
			interp_order= 1 # 1=bilinear, 2=biquadratic, 3=bicubic, 4=biquartic, 5=biquintic
			downscaling= (nx>self.resize_size) and (ny>self.resize_size)
			antialiasing= False
			if downscaling and self.downscale_with_antialiasing:
				antialiasing= True
		
			try:
				try: # work for skimage<=0.15.0
					ret= Utils.resize_img_v2(data_bbox, 
						min_dim=min_dim, max_dim=max_dim, min_scale=None, mode="square", 
						order=interp_order, 
						#anti_aliasing=antialiasing, 
						preserve_range=True
					)
				except:
					ret= Utils.resize_img_v2(img_as_float64(data_bbox), 
						min_dim=min_dim, max_dim=max_dim, min_scale=None, mode="square", 
						order=interp_order, 
						#anti_aliasing=antialiasing, 
						preserve_range=True
					)

				data_bbox= ret[0]
		

			except Exception as e:
				logger.warn("Failed to resize data to size (%d,%d) (err=%s)!" % (self.resize_size, self.resize_size, str(e)))
				return None

			if data_bbox is None:
				logger.warn("Resized data is None, failed to resize to size (%d,%d) (see logs)!" % (self.resize_size, self.resize_size))
				return None
		
		
		#return binary_map
		return data_bbox

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


	def __call__(self, data, **kwargs):
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
##   SigmaClipShifter
##############################
class SigmaClipShifter(object):
	""" Shift all pixels to new zero value equal to mean+(sigma*std) and clip values below this zero """

	def __init__(self, sigma=1.0, chid=-1, **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		self.sigma= sigma
		self.chid= chid # -1=do for all channels, otherwise clip only selected channel

	def __clip(self, data):
		""" Clip channel input """

		cond= np.logical_and(data!=0, np.isfinite(data))
		data_1d= data[cond]

		# - Clip all pixels that are below sigma clip (considered noise)
		#   NB: Following Galvin et al, PASA 131, 1 (2019)
		logger.debug("Clipping all pixels below (mean + %f x stddev) ..." % (self.sigma))
		clipmean, median, stddev = sigma_clipped_stats(data_1d, sigma=self.sigma)

		newzero= clipmean + self.sigma*stddev

		data_clipped= np.copy(data)
		#data_clipped[data_clipped<clipmean]= clipmean #### CHECK!!! PROBABLY WRONG!!!
		data_clipped-= newzero
		data_clipped[data_clipped<0]= 0
		data_clipped[~cond]= 0
		cond_clipped= np.logical_and(data_clipped!=0, np.isfinite(data_clipped))
		data_clipped_1d= data_clipped[cond_clipped]

		logger.debug("--> data min/max (after sigmaclip)=%s/%s (clipmean=%s)" % (str(data_clipped_1d.min()), str(data_clipped_1d.max()), str(clipmean)))

		return data_clipped 
		

	def __call__(self, data, **kwargs):
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
##   SigmaClipper
##############################
class SigmaClipper(object):
	""" Clip all pixels below zlow=mean-(sigma_low*std) and above zhigh=mean + (sigma_up*std) """

	def __init__(self, sigma_low=10.0, sigma_up=10.0, chid=-1, **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		self.sigma_low= sigma_low
		self.sigma_up= sigma_up
		self.chid= chid # -1=do for all channels, otherwise clip only selected channel

	def __clip(self, data):
		""" Clip channel input """

		cond= np.logical_and(data!=0, np.isfinite(data))
		data_1d= data[cond]

		# - Clip all pixels that are below sigma clip
		logger.debug("Clipping all pixel values <(mean - %f x stddev) and >(mean + %f x stddev) ..." % (self.sigma_low, self.sigma_up))
		res= sigma_clip(data_1d, sigma_lower=self.sigma_low, sigma_upper=self.sigma_up, masked=True, return_bounds=True)
		thr_low= res[1]
		thr_up= res[2]

		data_clipped= np.copy(data)
		data_clipped[data_clipped<thr_low]= thr_low
		data_clipped[data_clipped>thr_up]= thr_up
		data_clipped[~cond]= 0
		
		return data_clipped
		

	def __call__(self, data, **kwargs):
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
		
		
#################################
##   PercentileThresholder
#################################
class PercentileThresholder(object):
	""" Put to zero all pixels below specified percentile """

	def __init__(self, percthr=50, **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		self.percthr= percthr
	
	def __call__(self, data, **kwargs):
		""" Apply transformation and return transformed data """
			
		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Threshold each channel
		cond= np.logical_and(data!=0, np.isfinite(data))
		data_thresholded= np.copy(data)

		for i in range(data.shape[-1]):
			data_ch= data_thresholded[:,:,i]
			cond_ch= np.logical_and(data_ch!=0, np.isfinite(data_ch))
			data_ch_1d= data_ch[cond_ch]
			
			p= np.percentile(data_ch_1d, self.percthr)

			data_ch[data_ch<p]= 0
			data_ch[~cond_ch]= 0
	
			data_thresholded[:,:,i]= data_ch
				
		# - Restore 0 and nans set in original data
		data_thresholded[~cond]= 0 
		
		
		return data_thresholded

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
		
	def __call__(self, data, **kwargs):
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
		
	def __call__(self, data, **kwargs):
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
##   ZScaleTransformer
##############################
class ZScaleTransformer(object):
	""" Apply zscale transformation to each channel """

	def __init__(self, contrasts=[0.25,0.25,0.25], **kwparams):
		""" Create a data pre-processor object """

		self.contrasts= contrasts
		
	def __call__(self, data, **kwargs):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		cond= np.logical_and(data!=0, np.isfinite(data))

		# - Check constrast dim vs nchans
		nchans= data.shape[-1]
	
		if len(self.contrasts)<nchans:
			logger.error("Invalid constrasts given (constrast list size=%d < nchans=%d)" % (len(self.contrasts), nchans))
			return None
		
		# - Transform each channel
		data_stretched= np.copy(data)

		for i in range(data.shape[-1]):
			data_ch= data_stretched[:,:,i]
			if self.contrasts[i]<=0:
				data_transf= data_ch
			else:
				transform= ZScaleInterval(contrast=self.contrasts[i]) # able to handle NANs
				data_transf= transform(data_ch)
			data_stretched[:,:,i]= data_transf

		# - Scale data
		data_stretched[~cond]= 0 # Restore 0 and nans set in original data

		return data_stretched


##############################
##   HistEqualizer
##############################
class HistEqualizer(object):
	""" Apply histogram equalization to each channel """

	def __init__(self, adaptive=False, clip_limit=0.03, **kwparams):
		""" Create a data pre-processor object """

		self.clip_limit= clip_limit
		self.adaptive= adaptive

	def __call__(self, data, **kwargs):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		cond= np.logical_and(data!=0, np.isfinite(data))
		nchans= data.shape[-1]
	
		# - Transform each channel
		data_stretched= np.copy(data)

		for i in range(data.shape[-1]):
			if self.adaptive:
				data_stretched[:,:,i]= equalize_adapthist(data[:,:,i], clip_limit=self.clip_limit)
			else:
				data_stretched[:,:,i]= equalize_hist(data[:,:,i])
				#interval = MinMaxInterval()
				#transform= HistEqStretch( interval(data[:,:,i]) ) # astropy implementation
				#data_stretched[:,:,i]= transform( interval(data[:,:,i]) )
				
		# - Scale data
		data_stretched[~cond]= 0 # Restore 0 and nans set in original data

		return data_stretched


##############################
##   SourceRemover
##############################
class SourceRemover(object):
	""" Remove sources from each channel using an iterative flood-fill algorith """

	def __init__(self, niters=2, seed_thr=4., npix_max_thr=100, **kwparams):
		""" Create a data pre-processor object """

		self.niters= niters
		self.dsigma= 0.5
		self.seed_thr= seed_thr
		self.merge_thr= 2.5
		self.sigma_clip= 3
		self.npix_min_thr= 5
		self.npix_max_thr= npix_max_thr
		self.bkgbox_thickness= 10
		self.grow_source_mask= True
		self.grow_size= 5

	def __call__(self, data, **kwargs):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		cond= np.logical_and(data!=0, np.isfinite(data))
		nchans= data.shape[-1]
	
		# - Transform each channel
		data_transf= np.copy(data)

		for i in range(data.shape[-1]):
			data_nosource= Utils.get_source_subtracted_map(
				data[:,:,i],
				niters=self.niters, dsigma=self.dsigma,
				seed_thr=self.seed_thr, merge_thr=self.merge_thr, sigma_clip=self.sigma_clip, 
				npix_min_thr=self.npix_min_thr, npix_max_thr=self.npix_max_thr,
				bkgbox_thickness=self.bkgbox_thickness, 
				grow_source_mask=self.grow_source_mask, grow_size=self.grow_size,
				draw=False
			)
			data_transf[:,:,i]= data_nosource

		# - Restore 0 and nans set in original data
		data_transf[~cond]= 0 

		return data_transf

##############################
##   Chan3Trasformer
##############################
class Chan3Trasformer(object):
	""" Create 3 channels with a different transform per each channel: 
				- ch1: sigmaclip(0,20) + zscale(0.25)
				- ch2: sigmaclip(1,20) + zscale(0.25)
				- ch3: histeq
	"""

	def __init__(self, sigma_clip_baseline=0, sigma_clip_low=1, sigma_clip_up=20, zscale_contrast=0.25, **kwparams):
		""" Create a data pre-processor object """

		self.sigma_clip_baseline= sigma_clip_baseline
		self.sigma_clip_low= sigma_clip_low
		self.sigma_clip_up= sigma_clip_up
		self.zscale_contrast= zscale_contrast
		
	def __call__(self, data, **kwargs):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		cond= np.logical_and(data!=0, np.isfinite(data))
		nchans= data.shape[-1]
	
		# - Apply ChanResizer to create 3 channels
		cr= ChanResizer(nchans=3)
		data_cube= cr(data)

		# - Define transforms
		sclipper= SigmaClipper(sigma_low=self.sigma_clip_baseline, sigma_up=self.sigma_clip_up, chid=-1)
		sclipper2= SigmaClipper(sigma_low=self.sigma_clip_low, sigma_up=self.sigma_clip_up, chid=-1)
		zscale= ZScaleTransformer(contrasts=[self.zscale_contrast])
		histEq= HistEqualizer(adaptive=False)
		#sremover= SourceRemover(niters=self.niters, seed_thr=self.seed_thr, npix_max_thr=self.npix_max_thr)

		# - Create channel 1: sigmaclip(0,20) + zscale(0.25)	
		data_transf_ch1= zscale(sclipper(np.expand_dims(data_cube[:,:,0], axis=-1)))
		data_cube[:,:,0]= data_transf_ch1[:,:,0]

		# - Create channel 2: sigmaclip(1,20) + zscale(0.25)
		data_transf_ch2= zscale(sclipper2(np.expand_dims(data_cube[:,:,1], axis=-1)))
		data_cube[:,:,1]= data_transf_ch2[:,:,0]

		# - Create channel 3: histeq
		#data_transf_ch3= histEq(zscale(np.expand_dims(data_cube[:,:,2], axis=-1)))
		#data_transf_ch3= histEq(sremover(np.expand_dims(data_cube[:,:,2], axis=-1)))
		#data_transf_ch3= histEq(sremover(sclipper(np.expand_dims(data_cube[:,:,2], axis=-1))))
		data_transf_ch3= histEq( np.expand_dims(data_cube[:,:,2], axis=-1) )
		data_cube[:,:,2]= data_transf_ch3[:,:,0]
		
		return data_cube

##############################
##   ChanResizer
##############################
class ChanResizer(object):
	""" ChanResizer modifies the number of channels until reaching desidered value. Replicate last channel when expanding. """

	def __init__(self, nchans, **kwparams):
		""" Create a data pre-processor object """

		self.nchans= nchans
		self.nchans_max= 1000
		

	def __call__(self, data, **kwargs):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Check number of given chans		
		if self.nchans>self.nchans_max or self.nchans<=0:
			logger.error("Invalid channel specified or too many channels desired (%d) (hint: the maximum is %d)!" % (self.nchans, self.nchans_max))
			return None

		cond= np.logical_and(data!=0, np.isfinite(data))

		# - Set nchan curr 
		ndim_curr= data.ndim
		if ndim_curr==2:
			nchans_curr= 1
		else:
			nchans_curr= data.shape[-1]
		
		if self.nchans==nchans_curr:
			logger.debug("Desired number of channels equal to current, nothing to be done...")
			return data
		
		expanding= self.nchans>nchans_curr

		# - Expand array first?
		#   NB: If 2D first create an extra dimension
		if ndim_curr==2:
			data= np.expand_dims(data, axis=-1)

		# - Copy last channel in new ones
		data_resized= np.zeros((data.shape[0], data.shape[1], self.nchans))

		if expanding:
			for i in range(self.nchans):
				if i<nchans_curr:
					data_resized[:,:,i]= data[:,:,i]
				else:
					data_resized[:,:,i]= data[:,:,nchans_curr-1]	
		else:
			for i in range(self.nchans):
				data_resized[:,:,i]= data[:,:,i]

		return data_resized



##############################
##     ColorJitterer
##############################
class ColorJitterer(object):
	""" Apply color jitter to input image  """

	def __init__(self, brightness=0, contrast=0, saturation=0, restore_original_range=False, **kwparams):
		""" Create a data pre-processor object """
			
		# - Set parameters
		self.brightness= brightness
		self.contrast= contrast
		self.saturation= saturation
		self.restore_original_range= restore_original_range
		
	def __call__(self, data, **kwargs):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None
			
		# - Check parameters
		if self.brightness<0:
			logger.error("Brightness parameter must be >=0!")
			return None
			
		if self.contrast<0:
			logger.error("Contrast parameter must be >=0!")
			return None	
			
		if self.saturation<0:
			logger.error("Contrast parameter must be >=0!")
			return None		
			
		# - Retrieve original number of channels & normalization
		datatype_orig= data.dtype
		norm_min_orig= data.min()
		norm_max_orig= data.max()
		ndim_orig= data.ndim
		if ndim_orig==2:
			nchans_orig= 1
		else:
			nchans_orig= data.shape[-1]
			
		logger.debug("Data original range: [%f, %f]" % (norm_min_orig, norm_max_orig))
		logger.debug("ndim_orig=%d, nchans_orig=%d" % (ndim_orig, nchans_orig))
			
		# - Define transformation
		normalizer= MinMaxNormalizer(norm_min=0., norm_max=255.)
		denormalizer= MinMaxNormalizer(norm_min=norm_min_orig, norm_max=norm_max_orig, exclude_zeros=False)
		chresizer= ChanResizer(3)
		chresizer_inv= ChanResizer(nchans_orig)
		
		# - Create 3 channels if only one channel is given
		data_transf= data
		if ndim_orig!=3 or nchans_orig!=3:
			logger.debug("Reshaping data to 3 channels ...")
			data_transf= chresizer(data)
			
		# - Convert to range [0,255]
		if norm_min_orig!=0 or norm_max_orig!=255:
			logger.debug("Converting to range [0,255] ...")
			data_transf= normalizer(data_transf)
		
		# - Convert to uint8
		data_transf= data_transf.astype('uint8')
		
		logger.debug("Data range: [%f, %f]" % (data_transf.min(), data_transf.max()))
		logger.debug("ndim=%d, nchans=%d, dtype=%s" % (data_transf.ndim, data_transf.shape[-1], str(data_transf.dtype)))
			
		# - Apply color jitter
		data_transf= self.__get_color_jittered_img(data_transf)
		
		logger.debug("Data range (after jitter): [%f, %f]" % (data_transf.min(), data_transf.max()))
		logger.debug("ndim=%d, nchans=%d, dtype=%s" % (data_transf.ndim, data_transf.shape[-1], str(data_transf.dtype)))
		
		if self.restore_original_range:
			# - Convert back to original type	
			data_transf= data_transf.astype(datatype_orig)
		
			# - Denormalize data (revert to original range)
			logger.debug("Denormalize data (revert to original range) ...")			
			data_transf= denormalizer(data_transf)
			
			logger.debug("Data range (after restore range): [%f, %f]" % (data_transf.min(), data_transf.max()))
		
			# - Convert back to original dimension
			ndim= data_transf.ndim
			nchans= data_transf.shape[-1]
			if ndim!=ndim_orig or nchans!=nchans_orig:
				logger.debug("Converting back to original dimension & type ...")
				if ndim_orig==2:
					data_transf= data_transf[:,:,0]
				else:
					data_transf= chresizer_inv(data_transf)
				
		return data_transf
		
	def __get_color_jittered_img(self, image):
		""" Return color jitter image, using TFM """
		
		image_transf= tfm.vision.augment.brightness(image, self.brightness)
		image_transf= tfm.vision.augment.contrast(image_transf, self.contrast)
		image_transf= self.__saturation(image_transf, self.saturation)
		
		return image_transf.numpy()
		
	def __saturation(self, image, saturation):
		""" Return saturated image """
		return tfm.vision.augment.blend(tf.repeat(tf.image.rgb_to_grayscale(image), 3, axis=-1), image, saturation)	
	

##############################
##     MedianFilterer
##############################
class MedianFilterer(object):
	""" Apply median filter to input image  """

	def __init__(self, size=3, **kwparams):
		""" Create a data pre-processor object """
			
		# - Set parameters
		self.size= size
		
	def __call__(self, data, **kwargs):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None
			
		# - Apply image filter
		#data_transf= scipy.ndimage.median_filter(data, size=self.size, mode='nearest')
		data_transf= scipy.ndimage.median_filter(data, size=self.size, mode='reflect')
			
		return data_transf
			
##############################
##   MaskShrinker
##############################
class MaskShrinker(object):
	""" Shrink input data mask using an erosion operation """

	def __init__(self, kernsize, **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		self.kernsize= kernsize

	def __call__(self, data, **kwargs):
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
				#iaa.Sometimes(0.5, iaa.Affine(scale=(0.5, 1.0), mode='constant', cval=0.0))
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

		# - Apply flip (66%) + rotate (always) + stretch zscale/sigmoid (always) + scale/blur (50%) + thresholding (50%)
		zscaleStretch_aug= ZScaleAugmenter(contrast=0.25, random_contrast=True, random_contrast_per_ch=False, contrast_min=0.1, contrast_max=0.5)
		zscaleStretch2_aug= ZScaleAugmenter(contrast=0.25, random_contrast=True, random_contrast_per_ch=False, contrast_min=0.25, contrast_max=0.5)
		sigmoidStretch_aug= SigmoidStretchAugmenter(cutoff=0.5, gain=10, random_gain=True, random_gain_per_ch=False, gain_min=10, gain_max=30)
		percThr_aug= PercentileThrAugmenter(percentile=50, random_percentile=True, random_percentile_per_ch=False, percentile_min=40, percentile_max=60)
		scale_aug= iaa.Affine(scale=(0.5, 1.0), mode='constant', cval=0.0)
		blur_aug= iaa.GaussianBlur(sigma=(1.0, 3.0))
		noise_aug= iaa.AdditiveGaussianNoise(scale=(0, 0.1))
		crop_aug= RandomCropResizeAugmenter(crop_fract_min=0.7, crop_fract_max=1.0)
		crop_aug_v2= RandomCropResizeAugmenter(crop_fract_min=0.8, crop_fract_max=1.0)
		
		# - See https://arxiv.org/pdf/2002.05709.pdf (pag. 12) and https://arxiv.org/pdf/2305.16127.pdf (pag 4)
		colorJitter_aug= ColorJitterAugmenter(
			brightness=0.8, 
			contrast=0.8, 
			saturation=0.8,
			hue=0.2,
			strength=0.5
		)
		
		# - Define compact source remover augmenter
		sremover_aug= SourceRemoverAugmenter(npix_upper_thr_min=100, npix_upper_thr_max=500)

		augmenter_simclr4= iaa.Sequential(
			[
				iaa.OneOf([iaa.Fliplr(1.0), iaa.Flipud(1.0), iaa.Noop()]),
  			iaa.Affine(rotate=(-90, 90), mode='constant', cval=0.0),
				iaa.OneOf([zscaleStretch_aug, sigmoidStretch_aug]),
				iaa.Sometimes(0.5, 
					iaa.OneOf([scale_aug, blur_aug]),
					iaa.Noop()
				),
				iaa.Sometimes(0.5,
					percThr_aug,
					iaa.Noop()
				)
			]
		)

		augmenter_simclr5= iaa.Sequential(
			[
				iaa.OneOf([iaa.Fliplr(1.0), iaa.Flipud(1.0), iaa.Noop()]),
  			iaa.Affine(rotate=(-90, 90), mode='constant', cval=0.0),
				zscaleStretch_aug,
				iaa.Sometimes(0.5, 
					iaa.OneOf([scale_aug, blur_aug]),
					iaa.Noop()
				)
			]
		)

		augmenter_simclr6= iaa.Sequential(
			[
				iaa.Sometimes(0.1, blur_aug),
				crop_aug,
				zscaleStretch_aug,	
				iaa.OneOf([iaa.Fliplr(1.0), iaa.Flipud(1.0), iaa.Noop()]),
  			iaa.Affine(rotate=(-90, 90), mode='constant', cval=0.0)
			]
		)
		
		augmenter_simclr7= iaa.Sequential(
			[
				iaa.Sometimes(0.1, blur_aug),
				zscaleStretch2_aug,
				iaa.OneOf([iaa.Fliplr(1.0), iaa.Flipud(1.0), iaa.Noop()]),
  			iaa.OneOf([iaa.Rot90((1,3)), iaa.Noop()]), # rotate by 90, 180 or 270 or do nothing
  			iaa.Sometimes(0.5, percThr_aug) 			
			]
		)
		
		augmenter_simclr8= iaa.Sequential(
			[
				iaa.Sometimes(0.1, blur_aug),
				iaa.Sometimes(0.8, colorJitter_aug),
				iaa.OneOf([iaa.Fliplr(1.0), iaa.Flipud(1.0), iaa.Noop()]),
  			iaa.OneOf([iaa.Rot90((1,3)), iaa.Noop()]), # rotate by 90, 180 or 270 or do nothing	
			]
		)
		
		augmenter_simclr9= iaa.Sequential(
			[
				crop_aug_v2,
				iaa.Sometimes(0.8, colorJitter_aug),
				iaa.OneOf([iaa.Fliplr(1.0), iaa.Flipud(1.0), iaa.Noop()]),
  			iaa.OneOf([iaa.Rot90((1,3)), iaa.Noop()]), # rotate by 90, 180 or 270 or do nothing		
			]
		)
		
		augmenter_simclr10= iaa.Sequential(
			[
				crop_aug_v2,
				iaa.Sometimes(0.8, colorJitter_aug),
				iaa.OneOf([iaa.Fliplr(1.0), iaa.Flipud(1.0), iaa.Noop()]),
  			iaa.OneOf([iaa.Rot90((1,3)), iaa.Noop()]), # rotate by 90, 180 or 270 or do nothing
  			iaa.Sometimes(0.1, blur_aug),
  			iaa.Sometimes(0.5, percThr_aug)
			]
		)
		
		# - Define augmenter for BYOL
		augmenter_byol= iaa.Sequential(
			[
				iaa.Sometimes(0.1, blur_aug),
				crop_aug,
				zscaleStretch_aug,	
				iaa.OneOf([iaa.Fliplr(1.0), iaa.Flipud(1.0), iaa.Noop()]),
  			iaa.Affine(rotate=(-90, 90), mode='constant', cval=0.0)
			]
		)

	
		# - Set augmenter chosen
		if choice=='cae':
			self.augmenter= augmenter_cae
		elif choice=='cnn':
			self.augmenter= augmenter_cnn
		elif choice=='simclr':
			self.augmenter= augmenter_simclr6
		elif choice=='simclr_v7':
			self.augmenter= augmenter_simclr7
		elif choice=='simclr_v8':
			self.augmenter= augmenter_simclr8
		elif choice=='simclr_v9':
			self.augmenter= augmenter_simclr9
		elif choice=='simclr_v10':
			self.augmenter= augmenter_simclr10	
		elif choice=='byol':
			self.augmenter= augmenter_byol
		else:
			logger.warn("Unknown choice (%s), setting CAE augmenter..." % (choice))
			self.augmenter= augmenter_cae

		
	def __call__(self, data, **kwargs):
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
##   Augmenters
##############################
class Augmenters(object):
	""" Perform image augmentation per image basis according to given model """

	def __init__(self, augmenter_choices=["cae"], augmenters=[], **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		if not augmenters:
			self.augmenters= []
			for choice in augmenter_choices:
				augmenter= self.__get_augmenter(choice)
				self.augmenters.append(augmenter)
		else:
			self.augmenters= augmenters

	######################################
	##     DEFINE PREDEFINED AUGMENTERS
	######################################
	def __get_augmenter(self, choice):
		""" Define and set augmenters """

		# - Define augmenter for Conv Autoencoder
		augmenter_cae= iaa.Sequential(
			[
				iaa.OneOf([iaa.Fliplr(1.0), iaa.Flipud(1.0), iaa.Noop()]),
  			iaa.Affine(rotate=(-90, 90), mode='constant', cval=0.0),
				#iaa.Sometimes(0.5, iaa.Affine(scale=(0.5, 1.0), mode='constant', cval=0.0))
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

		# - Apply flip (66%) + rotate (always) + stretch zscale/sigmoid (always) + scale/blur (50%) + thresholding (50%)
		zscaleStretch_aug= ZScaleAugmenter(contrast=0.25, random_contrast=True, random_contrast_per_ch=False, contrast_min=0.1, contrast_max=0.5)
		zscaleStretch2_aug= ZScaleAugmenter(contrast=0.25, random_contrast=True, random_contrast_per_ch=False, contrast_min=0.25, contrast_max=0.5)
		sigmoidStretch_aug= SigmoidStretchAugmenter(cutoff=0.5, gain=10, random_gain=True, random_gain_per_ch=False, gain_min=10, gain_max=30)
		percThr_aug= PercentileThrAugmenter(percentile=50, random_percentile=True, random_percentile_per_ch=False, percentile_min=40, percentile_max=60)
		scale_aug= iaa.Affine(scale=(0.5, 1.0), mode='constant', cval=0.0)
		blur_aug= iaa.GaussianBlur(sigma=(1.0, 3.0))
		noise_aug= iaa.AdditiveGaussianNoise(scale=(0, 0.1))
		crop_aug= RandomCropResizeAugmenter(crop_fract_min=0.7, crop_fract_max=1.0)
		crop_aug_v2= RandomCropResizeAugmenter(crop_fract_min=0.8, crop_fract_max=1.0)
		
		# - See https://arxiv.org/pdf/2002.05709.pdf (pag. 12) and https://arxiv.org/pdf/2305.16127.pdf (pag 4)
		colorJitter_aug= ColorJitterAugmenter(
			brightness=0.8, 
			contrast=0.8, 
			saturation=0.8,
			hue=0.2,
			strength=0.5
		)
		
		augmenter_simclr9= iaa.Sequential(
			[
				crop_aug_v2,
				iaa.Sometimes(0.8, colorJitter_aug),
				iaa.OneOf([iaa.Fliplr(1.0), iaa.Flipud(1.0), iaa.Noop()]),
  			iaa.OneOf([iaa.Rot90((1,3)), iaa.Noop()]), # rotate by 90, 180 or 270 or do nothing		
			]
		)
		
		augmenter_simclr10= iaa.Sequential(
			[
				crop_aug_v2,
				iaa.Sometimes(0.8, colorJitter_aug),
				iaa.OneOf([iaa.Fliplr(1.0), iaa.Flipud(1.0), iaa.Noop()]),
  			iaa.OneOf([iaa.Rot90((1,3)), iaa.Noop()]), # rotate by 90, 180 or 270 or do nothing
  			iaa.Sometimes(0.1, blur_aug),
  			iaa.Sometimes(0.5, percThr_aug)
			]
		)
		
		# - Return augmenter chosen
		if choice=='cae':
			augmenter= augmenter_cae
		elif choice=='cnn':
			augmenter= augmenter_cnn
		elif choice=='simclr_v9':
			augmenter= augmenter_simclr9
		elif choice=='simclr_v10':
			augmenter= augmenter_simclr10	
		else:
			logger.warn("Unknown choice (%s), setting CAE augmenter..." % (choice))
			augmenter= augmenter_cae
			
		return augmenter

		
	def __call__(self, data, , **kwargs):
		""" Apply transformation and return transformed data """
			
		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None
			
		# - Retrieve augmenter index from kwargs
		if 'augmenter_index' in kwargs:
			augmenter_index= kwargs.get('augmenter_index')
		else:
			logger.warn("augmenter_index key not found in argument, setting it to 0 (first augmenter), check if this is the expected behavior!")
			augmenter_index= 0

		# - Make augmenters deterministic to apply similarly to images and masks
		##augmenter_det = self.augmenters[augmenter_index].to_deterministic()

		# - Augment data cube
		try:
			data_aug= self.augmenters[augmenter_index].augment_image(data)
		except Exception as e:
			logger.error("Failed to augment data (err=%s)!" % str(e))
			return None

		return data_aug




