#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################

## STANDARD MODULES
import sys
import numpy as np
import os
import re
import json
from collections import defaultdict
import operator as op
import copy

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections
from collections import Counter
import csv

## ASTROPY MODULES
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
import regions
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.units as u
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.stats import sigma_clipped_stats

## SCIKIT
import skimage
from skimage import util
from skimage.metrics import structural_similarity
from skimage.measure import moments_central, moments_normalized, moments_hu, moments, regionprops
from skimage.feature import peak_local_max
import mahotas
from scipy.ndimage.morphology import distance_transform_edt
from shapely.geometry import Polygon
from shapely.geometry import Point

## OPENCV
import cv2
import imutils

## GRAPHICS MODULES
#import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches

## MODULES
from sclassifier_vae import logger


#####################################
##     SData
#####################################
class SData(object):
	""" SData class """

	def __init__(self):
		""" Create a SData object """
	
		# - Init vars
		self.filepaths= []
		self.sname= "XXX"
		self.label= "UNKNOWN"
		self.id= -1
		self.f_badpix_thr= 0.3
		self.img_data= []
		self.img_data_mask= []
		self.img_heads= []
		self.nx= 0
		self.ny= 0
		self.nchannels= 0
		self.refch= 0

		# - Mask options
		self.kernsize= 5

		# - Enclosing circle pars
		self.centroids= []
		self.radii= []

		# - Bkg vars
		self.bkg_levels= []
		self.bkg_rms= []

		# - Moment pars
		self.center_of_masses= []
		
		self.moments= []
		self.moments_c= []
		self.moments_hu= []
		self.moments_zern= []

		# - Flux pars
		self.fluxes= []
		self.speaks= []
		self.smasks= []
		self.sfluxes= []
		self.sious= []
		self.speaks_dists= []

		# - SSIM options & pars
		self.save_ssim_pars= True
		self.winsize= 3
		self.ssim_maps= []
		self.ssim_avg= []

		# - Quality cuts
		self.negative_pix_fract_thr= 0.9
		self.bad_pix_fract_thr= 0.05

		# - Draw options
		self.draw= False

		# - Output options
		self.param_dict= collections.OrderedDict()


	def __read_fits(self, filename):
		""" Read FITS image and return data """

		# - Open file
		try:
			hdu= fits.open(filename,memmap=False)
		except Exception as ex:
			errmsg= 'Cannot read image file: ' + filename
			#cls._logger.error(errmsg)
			logger.error(errmsg)
			raise IOError(errmsg)

		# - Read data
		data= hdu[0].data
		data_size= np.shape(data)
		nchan= len(data.shape)
		if nchan==4:
			output_data= data[0,0,:,:]
		elif nchan==2:
			output_data= data	
		else:
			errmsg= 'Invalid/unsupported number of channels found in file ' + filename + ' (nchan=' + str(nchan) + ')!'
			#cls._logger.error(errmsg)
			logger.error(errmsg)
			hdu.close()
			raise IOError(errmsg)

		# - Read metadata
		header= hdu[0].header

		# - Close file
		hdu.close()

		return output_data, header
		
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

		for filename in self.filepaths:
			# - Read image
			logger.debug("Reading file %s ..." % filename) 
			data= None
			try:
				data, header= self.__read_fits(filename)
			except Exception as e:
				logger.error("Failed to read image data from file %s (err=%s)!" % (filename,str(e)))
				return -1

			# - Compute data mask
			#   NB: =1 good values, =0 bad (pix=0 or pix=inf or pix=nan)
			data_mask= np.logical_and(data!=0, np.isfinite(data)).astype(np.uint8)
		
			# - Check image integrity
			#has_bad_pixs= self.__has_bad_pixels(data, check_fract=False, thr=0)
			#if has_bad_pixs:
			#	logger.warn("Image %s has too many bad pixels (f=%f>%f)!" % (filename,f_badpix,self.f_badpix_thr) )	
			#	return -1

			# - Append image channel data to list
			self.img_data.append(data)
			self.img_heads.append(header)
			self.img_data_mask.append(data_mask)
		
		# - Check image sizes
		if not self.check_img_sizes():
			logger.error("Image channels for source %s do not have the same size, check your dataset!" % self.sname)
			return -1

		# - Set data shapes
		self.nx= self.img_data[0].shape[1]
		self.ny= self.img_data[0].shape[0]
		self.nchannels= len(self.img_data)

		return 0

	def apply_masks(self, masks):
		""" Set and apply masks """
		self.img_data_mask= masks

		if self.draw:
			fig, axs = plt.subplots(2, self.nchannels)

		for i in range(self.nchannels):			
			self.img_data[i][masks[i]==0]= 0

			if self.draw:
				axs[0, i].imshow(self.img_data[i])
				axs[1, i].imshow(self.img_data_mask[i])
				
		if self.draw:
			plt.show()
	

	def has_good_data(self, check_mask=False, check_bad=True, check_neg=True, check_same=True):
		""" Check data integrity (nan, negative, etc) """
			
		is_good= True		
		for i in range(self.nchannels):
			
			data= self.img_data[i]
			if check_mask:
				data= self.img_data_mask[i]

			# - Check for bad pixels
			if check_bad:
				has_bad_pixs= self.__has_bad_pixels(data, check_fract=True, thr=self.bad_pix_fract_thr)
				if has_bad_pixs:
					is_good= False
					break

			# - Check for negative pixels
			if check_neg:
				has_neg_pixs= self.__has_neg_pixels(data, check_fract=True, thr=self.negative_pix_fract_thr)
				if has_neg_pixs:
					is_good= False
					break
		
			# - Check for equal pixels
			if check_same:
				has_same_vals= self.__has_equal_pixels(data)
				if has_same_vals:
					is_good= False
					break

		return is_good


	def __has_bad_pixels(self, data, check_fract=True, thr=0.1):
		""" Check image data NAN values """ 
		
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

	def __has_neg_pixels(self, data, check_fract=True, thr=0.9):
		""" Check image data negative values """ 
		
		npixels= data.size
		n_neg= np.count_nonzero(data<0) 
		f_neg= n_neg/float(npixels)
		if check_fract:
			if f_neg>thr:
				logger.warn("Image has too many negative pixels (f=%f>%f)!" % (f_neg, thr) )	
				return True
		else:
			if n_neg>thr:
				logger.warn("Image has too many bad pixels (n=%f>%f)!" % (n_neg, thr) )	
				return True

		return False

	def __has_equal_pixels(self, data):
		""" Check image data same values """ 

		data_min= np.nanmin(data)
		data_max= np.nanmax(data)
		return (data_min==data_max)

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

	#####################################
	##     FLUX-BKG METHODS
	#####################################
	def subtract_bkg(self, bkgs, subtract_only_refch=False):
		""" Subtract image background """

		if len(bkgs)!=self.nchannels:
			logger.error("Number of input bkgs != nchannels, cannot subtract bkg!")	
			return -1

		# - Subtract bkg
		if subtract_only_refch:
			data= self.img_data[self.refch]
			mask= self.img_data_mask[self.refch]
			bkg= bkgs[self.refch]
			self.img_data[self.refch]-= bkg 
			self.img_data[self.refch][mask==0]= 0

		else:
			for i in range(self.nchannels):
				data= self.img_data[i]
				mask= self.img_data_mask[i]
				bkg= bkgs[i]
				self.img_data[i]-= bkg 
				self.img_data[i][mask==0]= 0

		# - Draw data & masks?
		if self.draw:
			fig, axs = plt.subplots(2, self.nchannels)
			for i in range(self.nchannels):			
				axs[0, i].imshow(self.img_data[i])
				axs[1, i].imshow(self.img_data_mask[i])
				
			plt.show()
	
		return 0

	def compute_bkg(self, masks, sigma_clip=3):
		""" Compute image background """

		# - Init bkg
		self.bkg_levels= [0]*self.nchannels
		self.bkg_rms= [0]*self.nchannels
		
		if len(masks)!=self.nchannels:
			logger.error("Number of input masks != nchannels, cannot compute bkg!")	
			return -1

		# - Compute bkg levels & rms
		logger.info("Computing image clipped stats of non-masked pixels ...")

		for i in range(self.nchannels):
			data= self.img_data[i]
			mask= masks[i]
			cond= np.logical_and(np.logical_and(data!=0, np.isfinite(data)), mask==0)
			data_1d= data[cond]
			print("--> data_1d.shape")
			print(data_1d.shape)
			mean, median, stddev= sigma_clipped_stats(data_1d, sigma=sigma_clip)
			self.bkg_levels[i]= median
			self.bkg_rms[i]= stddev

		return 0

	def compute_fluxes(self, subtract_bkg=False, subtract_only_refch=False):
		""" Compute flux """

		# - Compute flux (excluding masked pixels)
		self.fluxes= [0]*self.nchannels
		
		for i in range(self.nchannels):
			data= self.img_data[i]
			cond= np.logical_and(data!=0, np.isfinite(data))
			Stot= np.nansum(data[cond])
			bkg= self.bkg_levels[i]
			npix= np.count_nonzero(data[cond])
			S= Stot
			if subtract_bkg:
				if subtract_only_refch :
					if i==self.refch:
						S= Stot-npix*bkg
				else:
					S= Stot-npix*bkg

				if S<0:
					S= Stot

			logger.info("ch%d: Stot=%f, bkg=%f, S=%f" % (i+1, Stot, bkg, S))
			
			self.fluxes[i]= S

	#####################################
	##     COMPUTE CONTOUR PARS
	#####################################
	def __compute_contour_pars(self):
		""" Extract contour from mask """

		# - Init contour pars
		self.centroids= []
		self.radii= []
		self.moments= []
		self.center_of_masses= []

		# - Compute contour and pars
		try:
			for i in range(self.nchannels):
				# - Compute moments & center of masses
				data= self.img_data[i]
				M= moments(data)
				cm= M[1, 0] / M[0, 0], M[0, 1] / M[0, 0]
				self.center_of_masses.append(cm)
				self.moments.append(M)

				# - Compute contour and enclosing circle
				mask= self.img_data_mask[i]
				contours= cv2.findContours(np.copy(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				contours= imutils.grab_contours(contours)

				if len(contours)>0:
					(xc,yc), radius= cv2.minEnclosingCircle(contours[0])
					self.centroids.append((yc,xc))
					self.radii.append(radius)
				else:
					logger.error("No contours found!")
					return -1
	
		except Exception as e:		
			logger.error("Failed to compute contours (err=%s)!" % (str(e)))
			return -1	

		return 0

	#####################################
	##     COMPUTE MOMENTS
	#####################################
	
	def compute_img_moments(self):
		""" Compute image moments """

		# - Init pars
		self.moments_c= []
		self.moments_hu= []
		self.moments_zern= []

		# - Compute raw moments and centroids
		if self.__compute_contour_pars()<0:
			logger.error("Failed to compute contour pars!")
			return -1

		# - Compute moments (central, Hu, Zernike) of intensity images	
		#   NB: use centroid from refch for all channels
		centroid= self.centroids[self.refch]
		#centroid= self.center_of_masses[self.refch]

		for i in range(self.nchannels):
			data= self.img_data[i]
			radius= self.radii[i]
			ret= self.__compute_moments(data, centroid, radius)
			if ret is None:
				logger.error("Failed to compute moments for image %s (id=%s, ch=%d)!" % (self.sname, self.label, i+1))
				return -1

			self.moments_c.append(ret[0])
			self.moments_hu.append(ret[1])
			self.moments_zern.append(ret[2])

		return 0


	def __compute_moments(self, data, centroid, radius):
		""" Compute moments"""

		# - Compute central moments
		mom_c= moments_central(data, center=centroid, order=3)

		# - Compute normalized moments
		mom_norm= moments_normalized(mom_c, 3)

		# - Compute Hu moments
		mom_hu= moments_hu(mom_norm)

		# - Flatten moments
		mom_c= mom_c.flatten()

		# - Compute Zernike moments
		#   NB: mahotas takes only positive pixels and rescale image by sum(pix) internally
		poldeg= 4
		nmom_zernike= 9
		mom_zernike= [-999]*nmom_zernike
		try:
			mom_zernike = mahotas.features.zernike_moments(data, radius, degree=poldeg, cm=centroid)
			##mom_zernike = mahotas.features.zernike_moments(mask, radius, degree=poldeg, cm=centroid)
		except Exception as e:
			logger.warn("Failed to compute Zernike moments (err=%s)!" % (str(e)))

		#print("--> mom_zernike")
		#print(mom_zernike)
		
		return (mom_c, mom_hu, mom_zernike)

	#####################################
	##     MASK OPS
	#####################################
	def __grow_mask(self, mask, distance=1):
		""" Grow mask by given pixel distance """
		distances, nearest_label_coords = distance_transform_edt(
			mask == 0, return_indices=True
		)
    
		mask_expanded = np.zeros_like(mask)
		dilate_mask = distances <= distance

		# build the coordinates to find nearest labels,
		# in contrast to [1] this implementation supports label arrays
		# of any dimension
		masked_nearest_label_coords = [
			dimension_indices[dilate_mask]
			for dimension_indices in nearest_label_coords
		]
		nearest_labels = mask[tuple(masked_nearest_label_coords)]
		mask_expanded[dilate_mask] = nearest_labels
  
		return mask_expanded

	def grow_masks(self, kernsizes=[]):
		""" Grow masks """

		# - Set dilation kernel sizes
		if not kernsizes or len(kernsizes)!=self.nchannels:
			logger.info("kernsizes not specified, setting kernsize=%d for all channels ..." % (self.kernsize))
			kernsizes= [self.kernsize]*self.nchannels

		# - Grow masks
		if self.draw:
			fig, axs = plt.subplots(4, self.nchannels)

		try:
			counter= 0
			for i in range(self.nchannels):
				data= np.copy(self.img_data[i])
				mask= np.copy(self.img_data_mask[i])

				# - Grow if kernsize is >0
				if kernsizes[i]>0:
					mask_expanded = self.__grow_mask(mask, kernsizes[i])
					self.img_data_mask[i]= mask_expanded
					data_expanded= self.img_data[i]
					data_expanded[mask_expanded==0]= 0
					self.img_data[i]= data_expanded

				if self.draw:
					axs[0, i].imshow(data)
					counter+= 1
					axs[1, i].imshow(mask)
					counter+= 1
					axs[2, i].imshow(self.img_data[i])
					counter+= 1
					axs[3, i].imshow(self.img_data_mask[i])
					counter+= 1					

		except Exception as e:		
			logger.error("Failed to expand masks (err=%s)!" % (str(e)))
			return -1	

		if self.draw:
			plt.show()

		return 0


	def shrink_masks(self, kernsizes=[]):
		""" Shrink masks """

		# - Set erosion kernel sizes
		if not kernsizes or len(kernsizes)!=self.nchannels:
			logger.info("kernsizes not specified, setting kernsize=%d for all channels ..." % (self.kernsize))
			kernsizes= [self.kernsize]*self.nchannels

		print("--> kernsizes")
		print(kernsizes)

		# - Erode masks
		if self.draw:
			fig, axs = plt.subplots(4, self.nchannels)

		try:
			counter= 0
			for i in range(self.nchannels):
				data= np.copy(self.img_data[i])
				mask= np.copy(self.img_data_mask[i])
				
				# - Do erosion if kernsize is >0
				if kernsizes[i]>0:
					structel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernsizes[i],kernsizes[i]))					
					mask_eroded = cv2.erode(mask, structel, iterations = 1)
					self.img_data_mask[i]= mask_eroded
					data_eroded= self.img_data[i]
					data_eroded[mask_eroded==0]= 0
					self.img_data[i]= data_eroded

				if self.draw:
					axs[0, i].imshow(data)
					counter+= 1
					axs[1, i].imshow(mask)
					counter+= 1
					axs[2, i].imshow(self.img_data[i])
					counter+= 1
					axs[3, i].imshow(self.img_data_mask[i])
					counter+= 1					

		except Exception as e:		
			logger.error("Failed to shrink masks (err=%s)!" % (str(e)))
			return -1	

		if self.draw:
			plt.show()

		return 0

	#####################################
	##     COMPUTE SSIM
	#####################################
	def compute_ssim_pars(self, winsize=3):
		""" Compute SSIM params """

		# - Loop over images and compute params
		index= 0
		for i in range(self.nchannels-1):
			
			img_i= self.img_data[i]
			cond_i= np.logical_and(img_i!=0, np.isfinite(img_i))

			img_max_i= np.nanmax(img_i[cond_i])
			img_min_i= np.nanmin(img_i[cond_i])
			
			img_norm_i= (img_i-img_min_i)/(img_max_i-img_min_i)
			img_norm_i[~cond_i]= 0

			# - Compute SSIM maps
			for j in range(i+1,self.nchannels):
				img_j= self.img_data[j]
				cond_j= np.logical_and(img_j!=0, np.isfinite(img_j))
				img_max_j= np.nanmax(img_j[cond_j])
				img_min_j= np.nanmin(img_j[cond_j])
				
				img_norm_j= (img_j-img_min_j)/(img_max_j-img_min_j)
				img_norm_j[~cond_j]= 0

				cond= np.logical_and(cond_i, cond_j)
				
				# - Compute SSIM moments
				#   NB: Need to normalize images to max otherwise the returned values are always ~1.
				logger.info("Computing SSIM for image %s (id=%s, ch=%d-%d) ..." % (self.sname, self.label, i+1, j+1))
				_, ssim_2d= structural_similarity(img_norm_i, img_norm_j, full=True, win_size=winsize, data_range=1)

				ssim_2d[ssim_2d<0]= 0
				ssim_2d[~cond]= 0
				self.ssim_maps.append(ssim_2d)

				ssim_1d= ssim_2d[cond]

				#if self.draw:
				#	plt.subplot(1, 3, 1)
				#	plt.imshow(img_norm_i, origin='lower')
				#	plt.colorbar()

				#	plt.subplot(1, 3, 2)
				#	plt.imshow(img_norm_j, origin='lower')
				#	plt.colorbar()
					
				#	plt.subplot(1, 3, 3)
				#	plt.imshow(ssim_2d, origin='lower')
				#	plt.colorbar()

				#	plt.show()

				if ssim_1d.size>0:
					ssim_mean= np.nanmean(ssim_1d)
					ssim_median= np.nanmedian(ssim_1d)
					ssim_avg= ssim_median
					self.ssim_avg.append(ssim_avg)
					
					logger.info("Image %s (chan=%d-%d): <SSIM>=%f" % (self.sname, i+1, j+1, ssim_avg))

				else:
					logger.warn("Image %s (chan=%d-%d): SSIM array is empty, setting estimators to -999..." % (self.sname, i+1, j+1))
					self.ssim_avg.append(-999)
					

		return 0


	#####################################
	##     FIND SOURCES
	#####################################
	def find_sources(self, seed_thr=4, merge_thr=3, dist_thr=-1, subtract_bkg=False, subtract_only_refch=False):
		""" Find sources in all channels """

		# - Extract sources from each channel and compute integrated flux
		self.speaks= [None]*self.nchannels
		self.smasks= [None]*self.nchannels
		self.sfluxes= [None]*self.nchannels

		for i in range(self.nchannels):
			data= self.img_data[i]
			bkg= self.bkg_levels[i]
			rms= self.bkg_rms[i]
			ret= self.__extract_sources(data, bkg, rms, seed_thr=seed_thr, merge_thr=merge_thr, dist_thr=dist_thr)
			if ret is None:
				continue					

			peak= ret[0]
			bmap= ret[1]
			regprop= ret[2]

			if regprop is None:
				continue

			self.speaks[i]= peak
			self.smasks[i]= bmap
			Stot= np.nansum(data[bmap==1])
			npix= np.count_nonzero(bmap)
			
			S= Stot
			if subtract_bkg:
				if subtract_only_refch :
					if i==self.refch:
						S= Stot-npix*bkg
				else:
					S= Stot-npix*bkg

				if S<0:
					S= Stot

			self.sfluxes[i]= S


		# - Compute IOU and other pars
		for i in range(self.nchannels-1):
			smask_i= self.smasks[i]
			speak_i= self.speaks[i]

			for j in range(i+1,self.nchannels):
				smask_j= self.smasks[j]
				speak_j= self.speaks[j]
				iou= 0
				speak_dist= 0
				has_smasks= (smask_i is not None) and (smask_j is not None)
				has_speaks= (speak_i is not None) and (speak_j is not None)
				
				if has_smasks:
					iou= self.__compute_iou(smask_i, smask_j)
				if has_speaks:
					speak_dist= np.sqrt( (speak_i[1]-speak_j[1])**2 + (speak_i[0]-speak_j[0])**2 )
				
				self.sious.append(iou)
				self.speaks_dists.append(speak_dist)
				
		return 0


	def __extract_sources(self, data, bkg, rms, mask=None, seed_thr=4, merge_thr=3, dist_thr=-1):
		""" Find sources in channel data """
	
		# - Compute image center
		data_shape= data.shape
		y_c= data_shape[0]/2.;
		x_c= data_shape[1]/2.;

		# - Compute mask
		if mask is None:
			logger.info("Computing image mask ...")
			mask= np.logical_and(data!=0, np.isfinite(data))	

		data_1d= data[mask]
	
		# - Threshold image at seed_thr
		zmap= (data-bkg)/rms
		binary_map= (zmap>merge_thr).astype(np.int32)
		binary_map[~mask]= 0
		zmap[~mask]= 0
	
		# - Extract source
		logger.info("Extracting sources ...")
		label_map= skimage.measure.label(binary_map)
		regprops= skimage.measure.regionprops(label_map, data)

		nsources= len(regprops)
		logger.info("#%d sources found ..." % nsources)

		# - Extract peaks
		kernsize= 3
		footprint = np.ones((kernsize, ) * data.ndim, dtype=bool)
		peaks= peak_local_max(np.copy(zmap), footprint=footprint, threshold_abs=seed_thr, min_distance=2, exclude_border=True)
		#print(peaks)
		
		if peaks.shape[0]<=0:
			logger.info("No peaks detected in this image, return None ...")
			return None
		
		# - Select best source
		regprops_sel= []
		peaks_sel= []
		binary_maps_sel= []
		polygons_sel= []
		contours_sel= []
		#binary_maps_sel= []
		#binary_map_sel= np.zeros_like(binary_map)

		for regprop in regprops:
			# - Check if region max is >=seed_thr
			sslice= regprop.slice
			zmask= zmap[sslice]
			zmask_1d= zmask[np.logical_and(zmask!=0, np.isfinite(zmask))]	
			zmax= zmask_1d.max()
			if zmax<seed_thr:
				logger.info("Skip source as zmax=%f<thr=%f" % (zmax, seed_thr))
				continue

			# - Set binary map with this source
			logger.debug("Get source binary mask  ...")
			bmap= np.zeros_like(binary_map)
			bmap[sslice]= binary_map[sslice]

			# - Extract contour and polygon from binary mask
			logger.info("Extracting contour and polygon from binary mask ...")
			contours= []
			polygon= None
			try:
				bmap_uint8= bmap.copy() # copy as OpenCV internally modify origin mask
				bmap_uint8= bmap_uint8.astype(np.uint8)
				contours= cv2.findContours(bmap_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				contours= imutils.grab_contours(contours)
				if len(contours)>0:
					contour= np.squeeze(contours[0])
					polygon = Polygon(contour)
			except Exception as e:
				logger.warn("Failed to compute mask contour (err=%s)!" % (str(e)))
			
			if polygon is None:
				logger.warn("Skip extracted blob as polygon failed to be computed...")
				continue

			# - Check if source has a local peak in the mask
			#   NB: Check if polygon is computed
			has_peak= False
			peak_sel= None
			if polygon is not None:
				for peak in peaks:
					point = Point(peak[1], peak[0])
					has_peak= polygon.contains(point)
					if has_peak:
						peak_sel= peak
						break
				
			if not has_peak: 
				logger.info("Skip extracted blob as no peak was found inside source contour polygon!")
				continue

			# - Check for source peak distance wrt image center
			if dist_thr>0:
				dist= np.sqrt( (peak_sel[1]-x_c)**2 + (peak_sel[0]-y_c)**2 )
				if dist>dist_thr:
					logger.info("Skip extracted source as peak-imcenter dist=%f<thr=%f" % (dist, dist_thr))
					continue

			# - Update global binary mask and regprops
			#binary_map_sel[sslice]= binary_map[sslice]
			regprops_sel.append(regprop)
			peaks_sel.append(peak_sel)
			binary_maps_sel.append(bmap)	
			polygons_sel.append(polygon)
			contours_sel.append(contours[0])
			
		# - Return None if no source is selected
		nsources_sel= len(regprops_sel)
		if nsources_sel<=0:
			logger.info("No sources selected for this image ...")
			return None

		# - If more than 1 source is selected, take the one with peak closer to image center
		peak_final= peaks_sel[0]
		bmap_final= binary_maps_sel[0]
		regprop_final= regprops_sel[0]
		polygon_final= polygons_sel[0]
		contour_final= contours_sel[0]

		if nsources_sel>1:
			logger.info("#%d sources selected, going to select the closest to image center ..." % (nsources_sel))
			
			dist_best= 1.e+99
			index_best= -1
			for j in range(len(peaks_sel)):
				peak= peaks_sel[j]
				bmap= binary_maps_sel[j]
				regprop= regprops_sel[j]
				polygon= polygons_sel[j]
				contour= contours_sel[j]
				dist= np.sqrt( (peak[1]-x_c)**2 + (peak[0]-y_c)**2 )
				if dist<dist_best:
					dist_best= dist
					peak_final= peak
					bmap_final= bmap
					regprop_final= regprop
					polygon_final= polygon
					contour_final= contour
					index_best= j
			
			logger.info("Selected source no. %d as the closest one to image center ..." % (index_best))							
		else:
			logger.info("#%d sources selected..." % (nsources_sel))
			
		# - Compute enclosing circle radius 
		try:
			(xc, yc), radius= cv2.minEnclosingCircle(contour_final)
			enclosing_circle= (xc,yc,radius)
		except Exception as e:
			logger.warn("Failed to compute min enclosing circle (err=%s)!" % (str(e)))
			enclosing_circle= None

		# - Draw figure
		if self.draw:
			fig, ax = plt.subplots()

			# - Draw map
			#plt.imshow(label_map)
			#plt.imshow(data)
			plt.imshow(zmap)
			#plt.imshow(bmap_final)
			plt.colorbar()

			# - Draw bbox rectangle
			bbox= regprop_final.bbox
			ymin= bbox[0]
			ymax= bbox[2]
			xmin= bbox[1]
			xmax= bbox[3]
			dx= xmax-xmin-1
			dy= ymax-ymin-1
			rect = patches.Rectangle((xmin,ymin), dx, dy, linewidth=1, edgecolor='r', facecolor='none')
			ax.add_patch(rect)

			# - Draw selected peak
			if peak_final is not None:
				plt.scatter(peak_final[1], peak_final[0], s=10)

			# - Draw contour polygon
			if polygon_final is not None:
				plt.plot(*polygon_final.exterior.xy)

			# - Draw enclosing circle
			circle = plt.Circle((xc, yc), radius, color='g', clip_on=False, fill=False)
			ax.add_patch(circle)

			plt.show()


		return (peak_final, bmap_final, regprop_final, enclosing_circle)


	def __compute_iou(self, mask1, mask2):
		""" Compute IOU between binary masks """
		mask1_area = np.count_nonzero(mask1 == 1)
		mask2_area = np.count_nonzero(mask2 == 1)
		intersection = np.count_nonzero( np.logical_and( mask1, mask2) )
		iou = intersection/(mask1_area+mask2_area-intersection)
		return iou


	#####################################
	##     SAVE DATA
	#####################################
	def __get_triu_indices(self, idx, N):
		""" Return chan i j index from global index """

		i= np.triu_indices(N, k=1)[0][idx]
		j= np.triu_indices(N, k=1)[1][idx]
		ch_i= i+1
		ch_j= j+1

		return ch_i, ch_j

	def fill_features(self):

		# - Save name
		self.param_dict["sname"]= self.sname

		# - Save source flux
		flux_ref= self.fluxes[self.refch]
		
		for j in range(len(self.fluxes)):
			flux= self.fluxes[j]
			parname= "flux_ch" + str(j+1)
			self.param_dict[parname]= flux

		# - Save source flux log ratios Fj/F_radio (i.e. colors)
		lgFluxRatio_safe= 0
		is_good_flux_ref= (flux_ref>0) and (np.isfinite(flux_ref))
		if not is_good_flux_ref:
			logger.warn("Flux for ref chan (%d) is <=0 or nan for image %s (id=%s),  will set all color index to %d..." % (self.refch, self.sname, self.label, lgFluxRatio_safe))

		for j in range(len(self.fluxes)):
			if j==self.refch:
				continue
			flux= self.fluxes[j] # if source is not detected this is the background level
			is_good_flux= (flux>0) and (np.isfinite(flux))
			
			lgFluxRatio= 0
			if is_good_flux_ref:
				if is_good_flux:
					lgFluxRatio= np.log10(flux/flux_ref)
				else:
					logger.warn("Flux for chan %d is <=0 or nan for image %s (id=%s),  will set this color index to %d..." % (self.refch, self.sname, self.label, lgFluxRatio_safe))
					lgFluxRatio= lgFluxRatio_safe
			else:
				lgFluxRatio= lgFluxRatio_safe
			 
			parname= "lgFratio_ch" + str(self.refch+1) + "_" + str(j+1)
			self.param_dict[parname]= lgFluxRatio

		
		# - Save source flux log ratios Fj/F_radio (i.e. colors)
		cind_safe= 0
		sflux_ref= self.sfluxes[self.refch]
		is_good_flux_ref= (sflux_ref is not None) and (sflux_ref>0) and (np.isfinite(sflux_ref))
		if not is_good_flux_ref:
			logger.warn("Flux for ref chan (%d) is <=0 or nan for image %s (id=%s),  will set all color index to %d..." % (self.refch, self.sname, self.label, cind_safe))

		for j in range(len(self.sfluxes)):
			if j==self.refch:
				continue
			sflux= self.sfluxes[j] 
			flux= self.fluxes[j]
			if sflux is None: # source is not detected, take sum of pixel fluxes inside ref source aperture (e.g. the background)
				logger.info("Source is not detected in chan %d, taking pixel sum over ref source aperture %f ..." % (j+1, flux))
				sflux= flux
				
			is_good_flux= (sflux>0) and (np.isfinite(sflux))
			
			cind= 0
			if is_good_flux_ref:
				if is_good_flux:
					cind= np.log10(sflux/sflux_ref)
				else:
					logger.warn("Flux for chan %d is <=0 or nan for image %s (id=%s),  will set this color index to %d..." % (self.refch, self.sname, self.label, cind_safe))
					cind= cind_safe
			else:
				cind= cind_safe
			
			parname= "color_ch" + str(self.refch+1) + "_" + str(j+1)
			self.param_dict[parname]= cind


		# - Save source IOU
		for j in range(len(self.sious)):
			ch_i, ch_j= self.__get_triu_indices(j, self.nchannels)
			iou= self.sious[j]
			parname= "iou_ch" + str(ch_i) + "_" + str(ch_j)
			self.param_dict[parname]= iou
			
		# - Save source peak dist
		for j in range(len(self.speaks_dists)):
			ch_i, ch_j= self.__get_triu_indices(j, self.nchannels)
			peak_dist= self.speaks_dists[j]
			parname= "dpeak_ch" + str(ch_i) + "_" + str(ch_j)
			self.param_dict[parname]= peak_dist


		# - Save img moments
		for i in range(len(self.moments_zern)):
			for j in range(len(self.moments_zern[i])):
				if j==0:
					continue # Skip as mom0 is always the same
				m= self.moments_zern[i][j]
				parname= "zernmom" + str(j+1) + "_ch" + str(i+1)
				self.param_dict[parname]= m

		# - Save ssim parameters
		if self.save_ssim_pars:
			for j in range(len(self.ssim_avg)):
				ch_i, ch_j= self.__get_triu_indices(j, self.nchannels)
				parname= "ssim_avg_ch{}_{}".format(ch_i,ch_j)
				self.param_dict[parname]= self.ssim_avg[j]
				
		# - Save class id
		self.param_dict["id"]= self.id


	def select_features(self, selcolids):
		""" Select feature cols (0 index is the first feature, not sname) """

		# - Check if param dict is filled
		if not self.param_dict or self.param_dict is None
			logger.error("Parameter dict is empty!")
			return -1

		# - Get list of sel keys given col indices
		keys= list(self.param_dict.keys())
		keys_sel= [keys[selcol+1] for selcol in selcolids] # +1 because 0 index is the first feature, not sname

		# - Create new dict with selected pars
		param_dict_sel= collections.OrderedDict()
		param_dict_sel["sname"]= self.param_dict["sname"]

		for key in key_sel:
			param_dict_sel[key]= self.param_dict[key]

		param_dict_sel["id"]= self.param_dict["id"]

		# - Override old dict
		self.param_dict= param_dict_sel
		
		return 0

#####################################
##      FeatExtractorMom class
#####################################
class FeatExtractorMom(object):
	
	def __init__(self, datalistfile="", datalistfile_mask=""):
		""" Return a FeatExtractorMom object """	

		# - Data options
		self.datalistfile= datalistfile
		self.datalistfile_mask= datalistfile_mask
		self.datalist= {}
		self.datalist_mask= {}
		self.datasize= 0
		self.classids= []
		self.classfract_map= {}
		self.labels= []
		self.snames= []
		self.nchannels= 0
		self.refch= 0

		# - Mask options
		self.kernsize= 5
		self.shrink_masks= False
		self.erode_kernels= []
		self.grow_masks= False
		self.dilate_kernels= []

		# - Bkg options
		self.subtract_bkg= False
		self.subtract_bkg_only_refch= False

		# - Source extraction options
		self.seed_thr= 4
		self.merge_thr= 2.5
		self.dist_thr= -1
		
		# - SSIM options
		self.ssim_winsize= 3
		self.save_ssim_pars= False

		# - Quality cuts
		self.negative_pix_fract_thr= 0.9
		self.bad_pix_fract_thr= 0.05

		# - Feature selection
		self.select_feat= False
		self.selfeatids= []

		# - Draw options
		self.draw= False

		# - Output options
		self.save= True
		self.par_dict_list= []
		self.outfile= "features_moments.csv"


	def run(self, datalist, datalist_mask):
		""" Run moment calculation passing data dict lists as inputs """

		# - Set data info
		logger.debug("Setting data info ...")
		self.datalist= datalist
		self.datalist_mask= datalist_mask

		self.datasize= len(self.datalist)
		self.labels= [item["label"] for item in self.datalist]
		self.snames= [item["sname"] for item in self.datalist]
		self.classids= 	[item["id"] for item in self.datalist]
		self.classfract_map= dict(Counter(self.classids).items())
		datasize_mask= len(self.datalist_mask)

		# - Check number of channels per image
		nchannels_set= set([len(item["filepaths"]) for item in self.datalist])
		if len(nchannels_set)!=1:
			logger.warn("Number of channels in each object instance is different (len(nchannels_set)=%d!=1)!" % (len(nchannels_set)))
			print(nchannels_set)
			return -1

		self.nchannels= list(nchannels_set)[0]
		
		# - Check data size for imgs and masks
		if self.datasize!=datasize_mask:
			logger.error("Img and mask datalist have different size!")
			return -1

		# - Loop over data and extract params per each source
		logger.info("Loop over data and extract params per each source ...")
		for i in range(self.datasize):
			if self.__process_sdata(i)<0:
				logger.error("Failed to read and process source data %d, skip to next..." % (i))
				continue
			
		# - Save data
		if self.save:
			logger.info("Saving data to file %s ..." % (self.outfile))
			self.__save_data()

		return 0


	def run(self):
		""" Run moment calculation """

		# - Read data filelist
		logger.info("Reading data filelist ...")
		if self.__read_datalist()<0:
			logger.error("Failed to read datalist!")
			return -1

		# - Loop over data and extract params per each source
		logger.info("Loop over data and extract params per each source ...")
		for i in range(self.datasize):
			if self.__process_sdata(i)<0:
				logger.error("Failed to read and process source data %d, skip to next..." % (i))
				continue
			
		# - Save data
		if self.save:
			logger.info("Saving data to file %s ..." % (self.outfile))
			self.__save_data()

		return 0


	def __save_data(self):
		""" Save data to file """

		if self.par_dict_list:
			logger.info("Saving parameter file %s ..." % (self.outfile))
			parnames = self.par_dict_list[0].keys()
			print("parnames")
			print(parnames)
		
			#with open(self.outfile, 'wb') as fp:
			with open(self.outfile, 'w') as fp:
				fp.write("# ")
				dict_writer = csv.DictWriter(fp, parnames)
				dict_writer.writeheader()
				dict_writer.writerows(self.par_dict_list)
		else:
			logger.warn("Parameter dict list is empty, no files will be written!")


	def __process_sdata(self, index):
		""" Process source data """

		#===========================
		#==    READ DATA
		#===========================
		# - Read source data
		logger.info("Reading source and source masked data %d ..." % (index))
		ret= self.__read_sdata(index)
		if ret is None:
			logger.error("Failed to read source data %d!" % (index))
			return -1

		sdata= ret[0]
		sdata_mask= ret[1]

		#===========================
		#==    MODIFY MASKS
		#===========================
		# - Shrink img & mask in masked sdata?		
		if self.shrink_masks: 
			logger.info("Shrinking img+mask on source masked data %d ..." % (index))
			if sdata_mask.shrink_masks(self.erode_kernels)<0:
				logger.warn("Failed to shrink mask for source masked data %d!" % (index))
				return -1

		# - Expand img & mask in masked sdata?
		if self.grow_masks:
			logger.info("Expanding img+mask on source masked data %d ..." % (index))
			if sdata_mask.grow_masks(self.dilate_kernels)<0:
				logger.warn("Failed to expand mask for source masked data %d!" % (index))
				return -1

		masks= sdata_mask.img_data_mask
		#mask_ref= masks[self.refch]

		#===========================
		#==  CHECK DATA INTEGRITY
		#===========================
		# - Check non-masked data
		has_good_data= sdata.has_good_data(check_mask=False, check_bad=True, check_neg=False, check_same=True)
		if not has_good_data:
			logger.warn("Source data %d are bad (too may NANs or equal pixel values)!" % (index))
			return -1

		# - Check masked data
		has_good_mask_data= sdata_mask.has_good_data(check_mask=False, check_bad=True, check_neg=True, check_same=True)
		if not has_good_mask_data:
			logger.warn("Source mask data %d are bad (too may NANs/negative or equal pixel values)!" % (index))
			return -1

		#===========================
		#==  CHECK AE RECO ACCURACY
		#===========================
		# ...
		# ...

		#===========================
		#==    COMPUTE BKG/FLUX
		#===========================
		# - Compute bkg on img over non-masked pixels
		logger.info("Computing bkg on source data %d ..." % (index))
		if sdata.compute_bkg(masks)<0:
			logger.warn("Failed to compute bkg for source data %d!" % (index))
			return -1

		bkg_levels= sdata.bkg_levels

		#print("--> bkg levels")
		#print(bkg_levels)

		# - Apply masks to sdata
		#   NB: Do this after bkg calculation (otherwise all non-masked pixels are set to 0, so bkg will be 0) and before subtract bkg
		logger.info("Applying masks to source data %d ..." % (index))
		sdata.apply_masks(masks)

		# - Subtract bkg on img
		#if self.subtract_bkg:
		#	logger.info("Subtracting bkg on source data %d ..." % (index))
		#	if sdata.subtract_bkg(bkg_levels, self.subtract_bkg_only_refch)<0:
		#		logger.warn("Failed to subtract bkg for source data %d!" % (index))
		#		return -1

		# - Compute integrated flux (no source extraction here, only sum of pixel fluxes in mask)
		logger.info("Computing flux on source data %d ..." % (index))
		sdata.compute_fluxes(subtract_bkg=self.subtract_bkg, subtract_only_refch=self.subtract_bkg_only_refch)

		# - Extract sources and compute pars 
		#   NB: source extraction may fail or not be accurate (e.g. miss source, contour not accurate, etc)
		logger.info("Extracting source blobs on source data %d ..." % (index))
		sdata.find_sources(
			seed_thr=self.seed_thr, merge_thr=self.merge_thr, dist_thr=self.dist_thr, 
			subtract_bkg=self.subtract_bkg, subtract_only_refch=self.subtract_bkg_only_refch
		)		

		#===========================
		#==    COMPUTE MOMENTS
		#===========================
		# - Compute centroids and moments on images (NB: masked before)
		logger.info("Computing moments on source data %d ..." % (index))	
		if sdata.compute_img_moments()<0:
			logger.warn("Failed to compute moments for source data %d!" % (index))
			return -1

		#===========================
		#==    COMPUTE SSIM
		#===========================
		if self.save_ssim_pars:
			logger.info("Computing ssim pars on source data %d ..." % (index))	
			if sdata.compute_ssim_pars(self.ssim_winsize)<0:
				logger.warn("Failed to compute SSIM pars for source data %d!" % (index))
				return -1

		#===========================
		#==   FILL SOURCE OUT DATA
		#===========================
		# - Fill and append features
		logger.info("Filling feature dict for source data %d ..." % (index))	
		sdata.fill_features()

		par_dict= sdata.param_dict
		if par_dict is None or not par_dict:
			logger.warn("Feature dict for source data %d is empty or None, skip it ..." % (index))
			
		else:
			# - Select features?
			if self.select_feat and self.selfeatids:
				ret= sdata.select_features(self.selfeatids)
				par_dict= sdata.param_dict

				if ret==0:
					self.par_dict_list.append(par_dict)
				else:
					logger.warn("Failed to select features for source data %d, skip it ..." % (index))

			else:
				self.par_dict_list.append(par_dict)
		
		return 0


	def __read_sdata(self, index):
		""" Read source data """

		# - Check index
		if index<0 or index>=self.datasize:
			logger.error("Invalid index %d given!" % index)
			return None

		# - Init sdata
		sdata= SData()
		sdata.refch= self.refch
		sdata.kernsize= self.kernsize
		sdata.draw= self.draw
		sdata.save_ssim_pars= self.save_ssim_pars
		sdata.negative_pix_fract_thr= self.negative_pix_fract_thr
		sdata.bad_pix_fract_thr= self.bad_pix_fract_thr
		
		sdata_mask= SData()
		sdata_mask.refch= self.refch
		sdata_mask.kernsize= self.kernsize
		sdata_mask.draw= self.draw
		sdata_mask.save_ssim_pars= self.save_ssim_pars
		sdata_mask.negative_pix_fract_thr= self.negative_pix_fract_thr
		sdata_mask.bad_pix_fract_thr= self.bad_pix_fract_thr

		# - Read source image data
		logger.debug("Reading source image data %d ..." % index)
		#d= self.datalist["data"][index]
		d= self.datalist[index]
		if sdata.set_from_dict(d)<0:
			logger.error("Failed to set source image data %d!" % index)
			return None

		if sdata.read_imgs()<0:
			logger.error("Failed to read source images %d!" % index)
			return None

		# - Read source masked image data
		logger.debug("Reading source masked image data %d ..." % index)
		#d= self.datalist_mask["data"][index]
		d= self.datalist_mask[index]
		
		if sdata_mask.set_from_dict(d)<0:
			logger.error("Failed to set source masked image data %d!" % index)
			return None

		if sdata_mask.read_imgs()<0:
			logger.error("Failed to read source masked images %d!" % index)
			return None

		return sdata, sdata_mask
		

	#############################
	##     READ DATA LIST
	#############################
	def __read_filelist(self, filename):
		""" Read a datalist """
			
		# - Read json filelist
		datalist= {}

		try:
			with open(filename) as fp:
				datalist= json.load(fp)
		except Exception as e:
			logger.error("Failed to read data filelist %s!" % (filename))
			return None

		# - Check number of channels per image
		nchannels_set= set([len(item["filepaths"]) for item in datalist["data"]])
		if len(nchannels_set)!=1:
			logger.warn("Number of channels in each object instance is different (len(nchannels_set)=%d!=1)!" % (len(nchannels_set)))
			print(nchannels_set)
			return -1

		return datalist, nchannels_set


	def __read_datalist(self):
		""" Read json filelist """

		# - Check datalist files
		if self.datalistfile=="" or self.datalistfile_mask=="":
			logger.error("Data list files are empty!")
			return -1

		# - Read data list for images and store number of instances per class, etc
		ret= self.__read_filelist(self.datalistfile)
		if ret is None:
			logger.error("Failed to read filelist for imgs!")
			return -1
		#self.datalist= ret[0]
		datadict= ret[0]
		nchannels_set= ret[1]
		
		self.datalist= datadict["data"]
		self.nchannels= list(nchannels_set)[0]
		#self.datasize= len(self.datalist["data"])
		#self.labels= [item["label"] for item in self.datalist["data"]]
		#self.snames= [item["sname"] for item in self.datalist["data"]]
		#self.classids= 	[item["id"] for item in self.datalist["data"]]
		
		self.datasize= len(self.datalist)
		self.labels= [item["label"] for item in self.datalist]
		self.snames= [item["sname"] for item in self.datalist]
		self.classids= 	[item["id"] for item in self.datalist]

		self.classfract_map= dict(Counter(self.classids).items())

		logger.info("#%d objects in dataset" % self.datasize)

		# - Read data list for masked images
		ret= self.__read_filelist(self.datalistfile_mask)
		if ret is None:
			logger.error("Failed to read filelist for masked imgs!")
			return -1
		#self.datalist_mask= ret[0]
		datadict_mask= ret[0]
		self.datalist_mask= datadict_mask["data"]
		#datasize_mask= len(self.datalist_mask["data"])
		datasize_mask= len(self.datalist_mask)

		# - Check data size for imgs and masks
		if self.datasize!=datasize_mask:
			logger.error("Img and mask datalist have different size!")
			return -1

		return 0



