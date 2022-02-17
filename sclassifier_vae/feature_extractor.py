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
import collections
import csv


##############################
##     GLOBAL VARS
##############################
from sclassifier_vae import logger

## SCI MODULES
from astropy.stats import sigma_clipped_stats
import skimage
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity
from skimage.measure import moments_central, moments_normalized, moments_hu, moments, regionprops
from skimage.feature import peak_local_max
from scipy.stats import kurtosis, skew, median_absolute_deviation
import mahotas
import cv2
import imutils
from shapely.geometry import Polygon
from shapely.geometry import Point

## GRAPHICS MODULES
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
from matplotlib import patches

## PACKAGE MODULES
from .utils import Utils
from .data_loader import DataLoader
from .data_loader import SourceData


##################################
##   FeatExtractorHelper CLASS
##################################
class FeatExtractorHelper(object):

	def __init__(self):
		""" Return a FeatExtractorHelper object """
		
		# - Source data
		self.sdata= None
		self.data= None
		self.nchans= -1
		self.sname= ""
		self.label= ""
		self.classid= 0
		self.refch= 0

		# - Data pre-processing options
		self.normalize_img= False
		self.scale_to_max= False
		self.chan_mins= []

		# - Source extraction options
		self.seed_thr= 5
		self.merge_thr= 3
		self.sigma_clip= 3
		self.subtract_bkg= True
		self.dilatemask= False
		self.kernsize= 5
		self.dist_thr= -1
		self.speaks= []
		self.scircles= []
		self.smasks= []
		self.sfluxes= []
		self.sious= []
		self.speaks_dists= []
		self.sseparations= []

		# - Validation options
		self.fthr_zeros= 0.1

		# - SSIM options & pars
		self.compute_ssim_params= False
		self.winsize= 3
		self.ssim_maps= []
		self.ssim_mean= []
		self.ssim_min= []
		self.ssim_max= []
		self.ssim_std= []
		self.ssim_median= []
		self.ssim_mad= []
		self.moments_ssim= []
		self.moments_hu_ssim= []
		self.moments_zern_ssim= []

		# - Color map options & pars
		self.compute_cind_params= False
		self.ssim_thr= 0.
		self.colorind_safe= 0
		self.colorind_thr= 6
		self.weight_colmap_with_ssim= False	
		self.colorind_maps= []
		self.colorind_mean= -999
		self.colorind_std= -999
		self.colorind_skew= -999
		self.colorind_kurt= -999
		self.colorind_min= -999
		self.colorind_max= -999
		self.colorind_median= -999
		self.colorind_mad= -999
		self.moments_colorind= []
		self.moments_hu_colorind= []
		self.moments_zern_colorind= []

		# - Moments
		self.use_sfind_mask= True
		self.mask= None
		self.nmoments= 16
		self.moments_img= []
		self.hu_moments_img= []
		self.zern_moments_img= []
		
		# - Output data
		self.param_dict= collections.OrderedDict()
		self.save_mom_pars= False
		self.save_mom_max_no= 1
		self.save_zern_mom_pars= False
		self.save_hu_mom_pars= False
		self.save_source_pars= True
		self.save_ssim_pars= False
		self.save_cind_pars= False

		# - Draw options
		self.draw= False

	#####################################
	##     SETTERS/GETTERS
	#####################################
	def __reset_data(self):
		""" Clear existing data """
	
		# - Reset existing data
		self.param_dict= collections.OrderedDict()
		self.speaks= []
		self.scircles= []
		self.smasks= []
		self.sfluxes= []
		self.sious= []
		self.speaks_dists= []
		self.sseparations= []

		self.ssim_maps= []
		self.ssim_mean= []
		self.ssim_min= []
		self.ssim_max= []
		self.ssim_std= []
		self.ssim_median= []
		self.ssim_mad= []
		self.moments_ssim= []
		self.moments_hu_ssim= []
		self.moments_zern_ssim= []

		self.colorind_maps= []
		self.colorind_mean= []
		self.colorind_std= []
		self.colorind_skew= []
		self.colorind_kurt= []
		self.colorind_min= []
		self.colorind_max= []
		self.colorind_median= []
		self.colorind_mad= []
		self.moments_colorind= []
		self.moments_hu_colorind= []
		self.moments_zern_colorind= []

		self.mask= None
		self.moments_img= []
		self.hu_moments_img= []
		self.zern_moments_img= []

	def set_data(self, sdata, data):
		""" Set data """

		# - Set data
		self.sdata= sdata
		self.data= data
		self.nchans= data.shape[3]
		self.sname= sdata.sname
		self.label= sdata.label
		self.classid= sdata.id

		# - Reset existing data
		self.__reset_data()

	#####################################
	##     EXTRACT FEATURES
	#####################################
	def extract_features(self):
		""" Extract features """		

		#============================
		#==    VALIDATE DATA
		#============================
		# - Validate data
		if not self.__validate_img():
			logger.warn("Failed to validate data for image %s (id=%s)!" % (self.sname, self.label))
			return -1

		#============================
		#==    COMPUTE FEATURES
		#============================
		# - Find sources and compute pars
		logger.info("Extracting sources for image %s (id=%s) ..." % (self.sname, self.label))
		if self.__find_sources()<0:
			logger.warn("Failed to extract sources for image %s (id=%s) ..." % (self.sname, self.label))
			return -1

		# - Compute image moments
		logger.info("Computing image moments for image %s (id=%s) ..." % (self.sname, self.label))
		self.__compute_img_moments()
		
		# - Compute SSIM params?
		if self.compute_ssim_params:
			logger.info("Computing SSIM parameters for image %s (id=%s) ..." % (self.sname, self.label))
			self.__compute_ssim_pars()

		# - Compute CIND params?
		if self.compute_cind_params:
			logger.info("Computing color index map parameters for image %s (id=%s) ..." % (self.sname, self.label))
			self.__compute_cind_pars()

		#============================
		#==    FILL FEATURES
		#============================
		logger.info("Filling feature parameters for image %s (id=%s) ..." % (self.sname, self.label))
		self.__fill_features()

		return 0
		
		
	#####################################
	##     FILL PARAMETERS
	#####################################
	def __get_triu_indices(self, idx, N):
		""" Return chan i j index from global index """

		i= np.triu_indices(N, k=1)[0][idx]
		j= np.triu_indices(N, k=1)[1][idx]
		ch_i= i+1
		ch_j= j+1

		return ch_i, ch_j


	def __fill_features(self):

		# - Save name
		self.param_dict["sname"]= self.sname

		# - Save source flux
		for j in range(len(self.sfluxes)):
			flux= self.sfluxes[j]
			parname= "flux_ch" + str(j+1)
			self.param_dict[parname]= flux

		# - Save source flux log ratios Fj/F_radio (i.e. colors)
		flux_ref= self.sfluxes[self.refch]
		cind_safe= 0
		is_good_flux_ref= (flux_ref>0) and (np.isfinite(flux_ref))
		if not is_good_flux_ref:
			logger.warn("Flux for ref chan (%d) is <=0 or nan for image %s (id=%s),  will set all color index to %d..." % (self.refch, self.sname, self.label, cind_safe))

		for j in range(len(self.sfluxes)):
			if j==self.refch:
				continue
			flux= self.sfluxes[j]
			is_good_flux= (flux>0) and (np.isfinite(flux))
			
			cind= 0
			if is_good_flux_ref:
				if is_good_flux:
					cind= np.log10(flux/flux_ref)
				else:
					logger.warn("Flux for chan %d is <=0 or nan for image %s (id=%s),  will set this color index to %d..." % (self.refch, self.sname, self.label, cind_safe))
					cind= cind_safe
			else:
				cind= cind_safe
			 
			parname= "color_ch" + str(i+1) + "_" + str(j+1)
			self.param_dict[parname]= cind


		# - Save source IOU
		for j in range(len(self.sious)):
			ch_i, ch_j= self.__get_triu_indices(j, self.nchans)
			iou= self.sious[j]
			parname= "iou_ch" + str(ch_i) + "_" + str(ch_j)
			self.param_dict[parname]= iou
			
		# - Save source peak dist
		for j in range(len(self.speaks_dists)):
			ch_i, ch_j= self.__get_triu_indices(j, self.nchans)
			peak_dist= self.speaks_dists[j]
			parname= "dpeak_ch" + str(ch_i) + "_" + str(ch_j)
			self.param_dict[parname]= peak_dist

		# - Save source separation
		for j in range(len(self.sseparations)):
			ch_i, ch_j= self.__get_triu_indices(j, self.nchans)
			sep= self.sseparations[j]
			parname= "sep_ch" + str(ch_i) + "_" + str(ch_j)
			self.param_dict[parname]= sep
			
		# - Save img moments
		if self.save_mom_pars:
			for i in range(len(self.moments_img)):
				for j in range(len(self.moments_img[i])):
					if j>=self.save_mom_max_no:
						continue
					m= self.moments_img[i][j]
					parname= "mom" + str(j+1) + "_ch" + str(i+1)
					self.param_dict[parname]= m

		if self.save_zern_mom_pars:
			for i in range(len(self.zern_moments_img)):
				for j in range(len(self.zern_moments_img[i])):
					if j==0:
						continue # Skip as mom0 is always the same
					m= self.zern_moments_img[i][j]
					parname= "zernmom" + str(j+1) + "_ch" + str(i+1)
					self.param_dict[parname]= m

		# - Save ssim parameters
		if self.save_ssim_pars:
			for j in range(len(self.ssim_mean)):
				ch_i, ch_j= self.__get_triu_indices(j, self.nchans)
				parname= "ssim_mean_ch{}_{}".format(ch_i,ch_j)
				self.param_dict[parname]= self.ssim_mean[j]
				parname= "ssim_min_ch{}_{}".format(ch_i,ch_j)
				self.param_dict[parname]= self.ssim_min[j]
				parname= "ssim_max_ch{}_{}".format(ch_i,ch_j)
				self.param_dict[parname]= self.ssim_max[j]
				parname= "ssim_std_ch{}_{}".format(ch_i,ch_j)
				self.param_dict[parname]= self.ssim_std[j]
				parname= "ssim_median_ch{}_{}".format(ch_i,ch_j)
				self.param_dict[parname]= self.ssim_median[j]
				parname= "ssim_mad_ch{}_{}".format(ch_i,ch_j)
				self.param_dict[parname]= self.ssim_mad[j]

				if self.save_mom_pars:
					for k in range(len(self.moments_ssim[j])):
						if k>=self.save_mom_max_no:
							continue
						m= self.moments_ssim[j][k]
						parname= "ssim_mom{}_ch{}_{}".format(k+1, ch_i, ch_j)
						self.param_dict[parname]= m

				if self.save_hu_mom_pars:
					for k in range(len(self.moments_hu_ssim[j])):
					#for k in range(self.nmoments_save):	
						m= self.moments_hu_ssim[j][k]
						parname= "ssim_humom{}_ch{}_{}".format(k+1, ch_i, ch_j)
						self.param_dict[parname]= m
			
				if self.save_zern_mom_pars:
					for k in range(len(self.moments_zern_ssim[j])):
						m= self.moments_zern_ssim[j][k]
						parname= "ssim_zernmom{}_ch{}_{}".format(k+1, ch_i, ch_j)
						self.param_dict[parname]= m
	
		# - Save cind parameters
		if self.save_cind_pars:
			for j in range(len(self.colorind_mean)):
				ch_i, ch_j= self.__get_triu_indices(j, self.nchans)
				parname= "cind_mean_ch{}_{}".format(ch_i,ch_j)
				self.param_dict[parname]= self.colorind_mean[j]
				parname= "cind_min_ch{}_{}".format(ch_i,ch_j)
				self.param_dict[parname]= self.colorind_min[j]
				parname= "cind_max_ch{}_{}".format(ch_i,ch_j)
				self.param_dict[parname]= self.colorind_max[j]
				parname= "cind_std_ch{}_{}".format(ch_i,ch_j)
				self.param_dict[parname]= self.colorind_std[j]
				parname= "cind_median_ch{}_{}".format(ch_i,ch_j)
				self.param_dict[parname]= self.colorind_median[j]
				parname= "cind_mad_ch{}_{}".format(ch_i,ch_j)
				self.param_dict[parname]= self.colorind_mad[j]
				parname= "cind_skew_ch{}_{}".format(ch_i,ch_j)
				self.param_dict[parname]= self.colorind_skew[j]
				parname= "cind_kurt_ch{}_{}".format(ch_i,ch_j)
				self.param_dict[parname]= self.colorind_kurt[j]

				if self.save_mom_pars:
					for k in range(len(self.moments_colorind[j])):
						if k>=self.save_mom_max_no:
							continue
						m= self.moments_colorind[j][k]
						parname= "cind_mom{}_ch{}_{}".format(k+1, ch_i, ch_j)
						self.param_dict[parname]= m

				if self.save_hu_mom_pars:
					for k in range(len(self.moments_hu_colorind[j])):
					#for k in range(self.nmoments_save):	
						m= self.moments_hu_colorind[j][k]
						parname= "cind_humom{}_ch{}_{}".format(k+1, ch_i, ch_j)
						self.param_dict[parname]= m
			
				if self.save_zern_mom_pars:
					for k in range(len(self.moments_zern_colorind[j])):
						m= self.moments_zern_colorind[j][k]
						parname= "cind_zernmom{}_ch{}_{}".format(k+1, ch_i, ch_j)
						self.param_dict[parname]= m

		# - Save class id
		self.param_dict["id"]= self.classid

	#####################################
	##     VALIDATE IMAGE DATA
	#####################################
	def __validate_img(self):
		""" Perform some validation on input image """

		# - Check for NANs
		has_naninf= np.any(~np.isfinite(self.data))
		if has_naninf:
			logger.warn("Image (name=%s, label=%s) has some nan/inf, validation failed!" % (self.sname, self.label))
			return False

		# - Check correct absolute norm
		data_abs_min= np.min(self.data[0,:,:,:])
		data_abs_max= np.max(self.data[0,:,:,:])
		if self.normalize_img:
			if self.scale_to_max:
				correct_norm= (data_max==1)
			else:
				correct_norm= (data_min==0 and data_max==1)
			if not correct_norm:
				logger.warn("Image chan %d (name=%s, label=%s) has invalid norm (%f,%f), validation failed!" % (i+1, self.sname, self.label, data_abs_min, data_abs_max))
				return False

		# - Check for fraction of zeros in radio mask
		cond= np.logical_and(self.data[0,:,:,0]!=0, np.isfinite(self.data[0,:,:,0]))

		for i in range(self.nchans):
			data_2d= self.data[0,:,:,i]
			data_1d= data_2d[cond]
			n= data_1d.size
			n_zeros= np.count_nonzero(data_1d==0)
			f= n_zeros/n
			if n_zeros>0:
				logger.debug("Image chan %d (name=%s, label=%s): n=%d, n_zeros=%d, f=%f" % (i+1, self.sname, self.label, n, n_zeros, f))
				
			if f>=self.fthr_zeros:
				logger.warn("Image chan %d (name=%s, label=%s) has a zero fraction %f>%f, validation failed!" % (i+1, self.sname, self.label, f, self.fthr_zeros))
				return False

			# - Check if channels have all equal values 
			data_min= np.min(data_2d)
			data_max= np.max(data_2d)
			same_values= (data_min==data_max)
			if same_values:
				logger.warn("Image chan %d (name=%s, label=%s) has all elements equal to %f, validation failed!" % (i+1, self.sname, self.label, data_min))
				return False

		return True

	#####################################
	##     COMPUTE IMG MOMENTS
	#####################################
	def __compute_img_moments(self):
		""" Compute image moments """

		# - Compute centroid from reference channel (needed for moment calculation)
		logger.info("Computing centroid from reference channel (ch=%d) for image %s (id=%s) ..." % (self.refch, self.sname, self.label))
		ret= self.__compute_moments(self.data[0,:,:,self.refch])
		if ret is None:
			logger.error("Failed to compute ref channel mask centroid for image %s (id=%s, ch=%d)!" % (self.sname, self.label, i+1))
			return None
		self.mask= ret[3]
		self.centroid= ret[4]
		self.radius= ret[5]
		logger.info("refch centroid=(%f,%f), radius=%f" % (self.centroid[0],self.centroid[1],self.radius))

		if self.use_sfind_mask and self.smasks:
			if self.smasks[self.refch] is None:
				logger.warn("Cannot use source mask from ref channel for image %s (id=%s), using default one" % (self.sname, self.label))
			else:
				logger.info("Using source finder mask & enclosing circle for ref channel for image %s (id=%s)..." % (self.sname, self.label))
				self.mask= self.smasks[self.refch]
				circle= self.scircles[self.refch]
				self.centroid= (circle[1], circle[0])
				self.radius= circle[2]
				logger.info("sfind centroid=(%f,%f), radius=%f" % (self.centroid[0],self.centroid[1],self.radius))

				# - Dilate image mask to enlarge area around source
				if self.dilatemask:
					logger.info("Dilating source mask for image %s (id=%s) ..." % (self.sname, self.label))
					try:
						structel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernsize, self.kernsize))
						mask_dil = cv2.dilate(self.mask.copy().astype(np.uint8), structel, iterations = 1)
						mask_dil_uint8= mask_dil.copy() # copy as OpenCV internally modify origin mask
						mask_dil_uint8= mask_dil_uint8.astype(np.uint8)
						contours= cv2.findContours(mask_dil_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
						contours= imutils.grab_contours(contours)

						if len(contours)>0:
							(xc,yc), radius= cv2.minEnclosingCircle(contours[0])
							self.centroid= (yc,xc)
							self.radius= radius
							self.mask= mask_dil
							logger.info("sfind dilated centroid=(%f,%f), radius=%f" % (self.centroid[0],self.centroid[1],self.radius))
							
							#plt.subplots(1,2)
							#plt.subplot(1,2,1)
							#plt.imshow(self.smasks[self.refch])
							#plt.subplot(1,2,2)
							#plt.imshow(self.mask)

					except Exception as e:
						logger.warn("Failed to compute dilate mask and/or enclosing circle for image %s (id=%s) (err=%s), using original mask ..." % (self.sname, self.label, str(e)))


		# - Compute moments (central, Hu, Zernike) of intensity images	
		#   NB: use same mask and centroid from refch for all channels
		for i in range(self.nchans):
			img_i= self.data[0,:,:,i]
			ret= self.__compute_moments(img_i, self.mask, self.centroid, self.radius)
			if ret is None:
				logger.error("Failed to compute moments for image %s (id=%s, ch=%d)!" % (self.sname, self.label, i+1))
				return None

			#print("--> Mom0 (orig)")
			#print(ret[0][0])			

			# - Override Moment 0 (excluding masked pixels)
			cond_i= np.logical_and(img_i!=0, np.isfinite(img_i))
			S= np.nansum(img_i[cond_i])
			ret[0][0]= S
			#print("--> Mom0 (mod)")
			#print(ret[0][0])			

			self.moments_img.append(ret[0])
			self.hu_moments_img.append(ret[1])
			self.zern_moments_img.append(ret[2])

			
	#####################################
	##     COMPUTE MOMENTS
	#####################################
	def __compute_moments(self, data, mask=None, centroid=None, radius=None, cm_peak_thr=5):

		# - Compute mask if not given 
		if mask is None:
			mask= np.copy(data)
			cond= np.logical_and(data!=0, np.isfinite(data))
			mask[cond]= 1
			mask= mask.astype(np.int32)

		if self.draw:
			#plt.imshow(data)
			plt.imshow(mask)
			plt.show()

		# - Compute raw moments of flux image
		M= moments(data)
		centroid_this= M[1, 0] / M[0, 0], M[0, 1] / M[0, 0]
		#print("--> centroid_this")
		#print(centroid_this)
	
		# - Compute peaks
		kernsize= 5
		footprint = np.ones((kernsize, ) * data.ndim, dtype=bool)
		##peaks= peak_local_max(np.copy(data), min_distance=2, exclude_border=True)
		peaks= peak_local_max(np.copy(data), footprint=footprint, min_distance=2, exclude_border=True)	
		#print("peaks")
		#print(peaks)
	
		d_best= 1.e+99
		peak_best= None
		for peak in peaks:
			d= np.sqrt((peak[0]-centroid_this[0])**2+(peak[1]-centroid_this[1])**2)
			if d<d_best and d<=cm_peak_thr:
				d_best= d
				peak_best= peak
		
		if peak_best is None:
			centroid_best= centroid_this
		else:
			centroid_best= tuple(peak_best)			

		#print("-> centroid_best")
		#print(centroid_best)

		# - Compute centroid if not given, otherwise override
		if centroid is None:
			centroid= centroid_best

		#print("-> centroid")
		#print(centroid)

		# - Compute central moments
		mom_c= moments_central(data, center=centroid, order=3)

		# - Compute normalized moments
		#   NB: Do not use class method as this will use the automatically computed centroid. We want to override centroid here
		mom_norm= moments_normalized(mom_c, 3)

		# - Compute Hu moments
		#   NB: Do not use class method as this will use the automatically computed centroid. We want to override centroid here
		mom_hu= moments_hu(mom_norm)

		# - Flatten moments
		mom_c= mom_c.flatten()

		# - Compute min enclosing circle if not given
		if radius is None:
			contours= []
			try:
				mask_uint8= mask.copy() # copy as OpenCV internally modify origin mask
				mask_uint8= mask_uint8.astype(np.uint8)
				contours= cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				contours= imutils.grab_contours(contours)
			except Exception as e:
				logger.warn("Failed to compute mask contour (err=%s)!" % (str(e)))
			
			if len(contours)>0:
				try:
					(xc,yc), radius= cv2.minEnclosingCircle(contours[0])
				except:
					logger.warn("Failed to compute min enclosing circle (err=%s)!" % (str(e)))
					
			#logger.info("Computed radius & centroid: %f" % (radius))
			#print("centroid")
			#print(centroid)
			

		# - Compute Zernike moments
		#   NB: mahotas takes only positive pixels and rescale image by sum(pix) internally
		poldeg= 4
		nmom_zernike= 9
		mom_zernike= [-999]*nmom_zernike
		if centroid is not None and radius is not None:
			try:
				mom_zernike = mahotas.features.zernike_moments(data, radius, degree=poldeg, cm=centroid)
				##mom_zernike = mahotas.features.zernike_moments(mask, radius, degree=poldeg, cm=centroid)
			except Exception as e:
				logger.warn("Failed to compute Zernike moments (err=%s)!" % (str(e)))

		#print("--> mom_zernike")
		#print(mom_zernike)
		
		return (mom_c, mom_hu, mom_zernike, mask, centroid, radius)

	####################################
	###        FIND SOURCES
	####################################
	def __find_sources(self):
		""" Find sources in all channels """

		# - Extract sources from each channel and compute integrated flux
		self.speaks= []
		self.smasks= []
		self.sfluxes= []
		regprops= []

		for i in range(self.nchans):
			img_i= self.data[0,:,:,i]
			ret= self.__extract_sources(img_i)
			peak= ret[0]
			bmap= ret[1]
			regprop= ret[2]
			bkg_level= ret[3]
			circle= ret[4]

			if regprop is None:
				self.speaks.append(None)
				self.smasks.append(None)
				regprops.append(None)
				self.sfluxes.append(bkg_level)
				self.scircles.append(None)

				if i==0:
					logger.warn("No sources detected in 1st channel for image %s (id=%s), skip this source ..." % (self.sname, self.label))
					return -1

			else:

				if circle is None and i==0:
					logger.warn("None enclosing circle computed in 1st channel for image %s (id=%s), skip this source ..." % (self.sname, self.label))
					return -1

				self.speaks.append(peak)
				self.smasks.append(bmap)
				self.scircles.append(circle)
				regprops.append(regprop)
				Stot= np.nansum(img_i[bmap==1])
				npix= np.count_nonzero(bmap)
				S= Stot
				if self.subtract_bkg:
					S= Stot-bkg_level*npix
				self.sfluxes.append(S)

				#print("--> Stot=%f" % (Stot))
				#print("--> npix=%f" % (npix))
				#print("--> bkg_level=%f" % (bkg_level))
				#print("--> S=%f" % (S))

		# - Compute IOU and other pars
		for i in range(self.nchans-1):
			smask_i= self.smasks[i]
			speak_i= self.speaks[i]
			circle_i= self.scircles[i]

			for j in range(i+1,self.nchans):
				smask_j= self.smasks[j]
				speak_j= self.speaks[j]
				circle_j= self.scircles[j]
				iou= 0
				speak_dist= 0
				has_smasks= (smask_i is not None) and (smask_j is not None)
				has_speaks= (speak_i is not None) and (speak_j is not None)
				has_circle= (circle_i is not None) and (circle_j is not None)
				has_radius= has_circle and circle_i[2]>0 and circle_j[2]>0

				if has_smasks:
					iou= self.__compute_iou(smask_i, smask_j)
				if has_speaks:
					speak_dist= np.sqrt( (speak_i[1]-speak_j[1])**2 + (speak_i[0]-speak_j[0])**2 )
						
				if has_radius:
					x_i= circle_i[0]
					x_j= circle_j[0]
					y_i= circle_i[1]
					y_j= circle_j[1]
					r_i= circle_i[2]
					r_j= circle_j[2]
					d= np.sqrt( (x_i-x_j)**2 + (y_i-y_j)**2 )
					sep= d/(r_i + r_j)			
				else:
					logger.info("Cannot scale peak distance by enclosing circle radii sum as they are not both measured for image %s (id=%s, ch=%d-%d), setting speak_dist=-1 ..." % (self.sname, self.label, i+1, j+1))
					sep= -1

				self.sious.append(iou)
				self.speaks_dists.append(speak_dist)
				self.sseparations.append(sep)

				#print("--> iou")
				#print(iou)
				#print("--> speak_dist")
				#print(speak_dist)

		return 0


	def __extract_sources(self, data):
		""" Find sources in channel data """
	
		# - Compute image center
		data_shape= data.shape
		y_c= data_shape[0]/2.;
		x_c= data_shape[1]/2.;

		# - Compute mask
		logger.info("Computing image mask ...")
		mask= np.logical_and(data!=0, np.isfinite(data))	
		data_1d= data[mask]
	
		# - Compute clipped stats
		logger.info("Computing image clipped stats of non-masked pixels ...")
		mean, median, stddev= sigma_clipped_stats(data_1d, sigma=self.sigma_clip)

		# - Set bkg level
		bkg_level= median

		# - Threshold image at seed_thr
		zmap= (data-median)/stddev
		binary_map= (zmap>self.merge_thr).astype(np.int32)
		binary_map[~mask]= 0
	
		# - Extract source
		logger.info("Extracting sources ...")
		label_map= skimage.measure.label(binary_map)
		regprops= skimage.measure.regionprops(label_map, data)

		nsources= len(regprops)
		logger.info("#%d sources found ..." % nsources)

		# - Extract peaks
		kernsize= 3
		footprint = np.ones((kernsize, ) * data.ndim, dtype=bool)
		peaks= peak_local_max(np.copy(zmap), footprint=footprint, threshold_abs=self.seed_thr, min_distance=2, exclude_border=True)
		#print(peaks)
		
		if peaks.shape[0]<=0:
			logger.info("No peaks detected in this image, return None ...")
			return (None, None, None, bkg_level, None)
		
		#if self.draw:
		#	fig, ax = plt.subplots()
		#	#plt.imshow(label_map)
		#	#plt.imshow(data)
		#	plt.imshow(zmap)
		#	plt.colorbar()

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
			if zmax<self.seed_thr:
				logger.info("Skip source as zmax=%f<thr=%f" % (zmax, self.seed_thr))
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
				
			# - Add stuff to plot
			#if self.draw:
			#	# - Draw bbox rectangle
			#	bbox= regprop.bbox
			#	ymin= bbox[0]
			#	ymax= bbox[2]
			#	xmin= bbox[1]
			#	xmax= bbox[3]
			#	dx= xmax-xmin-1
			#	dy= ymax-ymin-1
			#	rect = patches.Rectangle((xmin,ymin), dx, dy, linewidth=1, edgecolor='r', facecolor='none')
			#	ax.add_patch(rect)

			#	# - Draw selected peak
			#	if peak_sel is not None:
			#		plt.scatter(peak_sel[1], peak_sel[0], s=10)

			#	# - Draw polygon
			#	if polygon is not None:
			#		plt.plot(*polygon.exterior.xy)

			if not has_peak: 
				logger.info("Skip extracted blob as no peak was found inside source contour polygon!")
				continue

			# - Check for source peak distance wrt image center
			if self.dist_thr>0:
				#try:
				#	centroid= regprop.weighted_centroid
				#except:
				#	centroid= regprop.centroid_weighted
				#dist= np.sqrt( (centroid[0]-x_c)**2 + (centroid[1]-y_c)**2 )
				dist= np.sqrt( (peak_sel[1]-x_c)**2 + (peak_sel[0]-y_c)**2 )
				if dist>self.dist_thr:
					logger.info("Skip extracted source as peak-imcenter dist=%f<thr=%f" % (dist, self.dist_thr))
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
			return (None, None, None, bkg_level, None)

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
			#plt.imshow(zmap)
			plt.imshow(bmap_final)
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


		return (peak_final, bmap_final, regprop_final, bkg_level, enclosing_circle)


	def __compute_iou(self, mask1, mask2):
		""" Compute IOU between binary masks """
		mask1_area = np.count_nonzero(mask1 == 1)
		mask2_area = np.count_nonzero(mask2 == 1)
		intersection = np.count_nonzero( np.logical_and( mask1, mask2) )
		iou = intersection/(mask1_area+mask2_area-intersection)
		return iou

	####################################
	###  COMPUTE SSIM MAP PARAMETERS
	####################################
	def __compute_ssim_pars(self):
		""" Compute SSIM map pars """
		
		# - Loop over images and compute params
		index= 0
		for i in range(self.nchans-1):
			
			img_i= self.data[0,:,:,i]
			cond_i= np.logical_and(img_i!=0, np.isfinite(img_i))

			img_max_i= np.nanmax(img_i[cond_i])
			img_min_i= np.nanmin(img_i[cond_i])
			
			img_norm_i= (img_i-img_min_i)/(img_max_i-img_min_i)
			img_norm_i[~cond_i]= 0

			# - Compute SSIM maps
			for j in range(i+1,self.nchans):
				img_j= self.data[0,:,:,j]
				cond_j= np.logical_and(img_j!=0, np.isfinite(img_j))
				img_max_j= np.nanmax(img_j[cond_j])
				img_min_j= np.nanmin(img_j[cond_j])
				
				img_norm_j= (img_j-img_min_j)/(img_max_j-img_min_j)
				img_norm_j[~cond_j]= 0

				cond= np.logical_and(cond_i, cond_j)
				
				# - Compute SSIM moments
				#   NB: Need to normalize images to max otherwise the returned values are always ~1.
				logger.info("Computing SSIM for image %s (id=%s, ch=%d-%d) ..." % (self.sname, self.label, i+1, j+1))
				_, ssim_2d= structural_similarity(img_norm_i, img_norm_j, full=True, win_size=self.winsize, data_range=1)

				ssim_2d[ssim_2d<0]= 0
				ssim_2d[~cond]= 0
				self.ssim_maps.append(ssim_2d)

				ssim_1d= ssim_2d[cond]

				if self.draw:
					plt.subplot(1, 3, 1)
					plt.imshow(img_norm_i, origin='lower')
					plt.colorbar()

					plt.subplot(1, 3, 2)
					plt.imshow(img_norm_j, origin='lower')
					plt.colorbar()
					
					plt.subplot(1, 3, 3)
					plt.imshow(ssim_2d, origin='lower')
					plt.colorbar()

					plt.show()

				if ssim_1d.size>0:
					self.ssim_mean.append(np.nanmean(ssim_1d))
					self.ssim_min.append(np.nanmin(ssim_1d))
					self.ssim_max.append(np.nanmax(ssim_1d))
					self.ssim_std.append(np.nanstd(ssim_1d))
					self.ssim_median.append(np.nanmedian(ssim_1d))
					self.ssim_mad.append(median_absolute_deviation(ssim_1d))
				
					ret= self.__compute_moments(ssim_2d, self.mask, self.centroid)
					if ret is None:
						logger.warn("Failed to compute SSIM moments for image %s (id=%s, ch=%d-%d)!" % (self.sname, self.label, i+1, j+1))
						
					self.moments_ssim.append(ret[0])
					self.moments_hu_ssim.append(ret[1])
					self.moments_zern_ssim.append(ret[2])
						
				else:
					logger.warn("Image %s (chan=%d-%d): SSIM array is empty, setting estimators to -999..." % (self.sname, i+1, j+1))
					self.ssim_mean.append(-999)
					self.ssim_min.append(-999)
					self.ssim_max.append(-999)
					self.ssim_std.append(-999)
					self.ssim_median.append(-999)
					self.ssim_mad.append(-999)
					self.moments_ssim.append([-999]*16)		
					self.moments_hu_ssim.append([-999]*7)
					self.moments_zern_ssim.append([-999]*9)

		return 0
				
	####################################
	###  COMPUTE COLOR MAP PARAMETERS
	####################################
	def __compute_cind_pars(self):
		""" Compute color index map parameters """

		if not self.chan_mins:
			self.chan_mins= [0]*self.nchans
	
		index= 0

		for i in range(self.nchans-1):
			
			img_i= self.data[0,:,:,i]
			cond_i= np.logical_and(img_i!=0, np.isfinite(img_i))

			img_posdef_i= img_i - self.chan_mins[i]
			img_posdef_i[~cond_i]= 0

			cond_col_i= np.logical_and(img_posdef_i>0, np.isfinite(img_posdef_i))

			# - Compute color indices
			for j in range(i+1,self.nchans):
				img_j= self.data[0,:,:,j]
				cond_j= np.logical_and(img_j!=0, np.isfinite(img_j))
				
				img_posdef_j= img_j - self.chan_mins[j]
				img_posdef_j[~cond_j]= 0

				cond= np.logical_and(cond_i, cond_j)
				cond_col_j= np.logical_and(img_posdef_j>0, np.isfinite(img_posdef_j))
				cond_col_ij= np.logical_and(cond_col_i, cond_col_j)
				
				# - Compute color index map
				logger.info("Computing color index map for image %s (id=%s, ch=%d-%d) ..." % (self.sname, self.label, i+1, j+1))
				
				if self.ssim_maps:
					cond_colors= np.logical_and(cond_col_ij, self.ssim_maps[index]>self.ssim_thr)
				else:
					logger.warn("ssim_maps was not computed, won't apply ssim threshold in color index map for image %s (id=%s, ch=%d-%d) ..." % (self.sname, self.label, i+1, j+1))
					cond_colors= cond_col_ij

				colorind_2d= np.log10( np.divide(img_posdef_i, img_posdef_j, where=cond_colors, out=np.ones(img_posdef_i.shape)*1) )
				cond_colors_safe= np.logical_and(cond_colors, np.fabs(colorind_2d)<self.colorind_thr)
				colorind_2d+= self.colorind_thr
				
				if self.weight_colmap_with_ssim and self.ssim_maps:
					colorind_2d*= self.ssim_maps[index]
					
				colorind_2d[~cond_colors_safe]= self.colorind_safe
		
				self.colorind_maps.append(colorind_2d)
				
				#cond_colorind= np.isfinite(colorind_2d)
				cond_colorind= np.logical_and(np.isfinite(colorind_2d), colorind_2d!=self.colorind_safe)
				colorind_1d= colorind_2d[cond_colorind]

				if colorind_1d.size>0:
					self.colorind_mean.append(np.nanmean(colorind_1d))
					self.colorind_std.append(np.std(colorind_1d))
					self.colorind_skew.append(skew(colorind_1d))
					self.colorind_kurt.append(kurtosis(colorind_1d))	
					self.colorind_min.append(np.nanmin(colorind_1d))
					self.colorind_max.append(np.nanmax(colorind_1d))
					self.colorind_median.append(np.nanmedian(colorind_1d))
					self.colorind_mad.append(median_absolute_deviation(colorind_1d))
				
					ret= self.__compute_moments(colorind_2d, self.mask, self.centroid)
					if ret is None:
						logger.warn("Failed to compute moments for color index image %s (id=%s, ch=%d-%d)!" % (self.sname, self.label, i+1, j+1))
						
					self.moments_colorind.append(ret[0])
					self.moments_hu_colorind.append(ret[1])
					self.moments_zern_colorind.append(ret[2])

				else:
					logger.warn("Image %s (chan=%d-%d): color index array is empty, setting estimators to -999..." % (self.sname, i+1, j+1))
					self.colorind_mean.append(-999)
					self.colorind_std.append(-999)
					self.colorind_skew.append(-999)
					self.colorind_kurt.append(-999)
					self.colorind_min.append(-999)
					self.colorind_max.append(-999)
					self.colorind_median.append(-999)
					self.colorind_mad.append(-999)
					self.moments_colorind.append([-999]*16)
					self.moments_hu_colorind.append([-999]*7)
					self.moments_zern_colorind.append([-999]*9)

				index+= 1

		return 0

				
##################################
##     FeatExtractor CLASS
##################################
class FeatExtractor(object):
	""" Feature extraction class """
	
	def __init__(self, data_loader):
		""" Return a FeatExtractor object """

		# - Input data
		self.dl= data_loader

		# - Train data	
		self.nsamples= 0
		self.nx= 64 
		self.ny= 64
		self.nchannels= 0
		self.source_names= []
		self.source_labels= []
		self.source_ids= []
		
		# - Data pre-processing options
		self.refch= 0
		self.nmaximgs= -1
		self.data_generator= None
		self.resize= False
		self.normalize_img= False
		self.scale_to_abs_max= False 
		self.scale_to_max= False
		self.augmentation= False
		self.shuffle_data= False
		self.log_transform_img= False
		self.scale_img= False
		self.scale_img_factors= []

		self.standardize_img= False
		self.img_means= [] 
		self.img_sigmas= []
		self.chan_divide= False
		self.chan_mins= []
		self.erode= False
		self.erode_kernel= 5

		# - Data validation options
		self.fthr_zeros= 0.1

		# - Source extraction options
		self.seed_thr= 5
		self.merge_thr= 3
		self.sigma_clip= 3
		self.subtract_bkg= True
		self.dist_thr= -1
		self.dilatemask= False	
		self.kernsize= 5
		self.use_sfind_mask= True

		# - SSIM options
		self.compute_ssim_params= False
		self.winsize= 3
		self.ssim_thr= 0.

		# - Hu Moment options
		self.nmoments_save= 4
		
		# - Color index options
		self.compute_cind_params= False
		self.colorind_safe= 0
		self.colorind_thr= 6
		self.weight_colmap_with_ssim= False
		
		# - Draw options
		self.draw_plots= False
		self.marker_mapping= {
			'UNKNOWN': 'o', # unknown
			'MIXED_TYPE': 'X', # mixed type
			'STAR': 'x', # star
			'GALAXY': 'D', # galaxy
			'PN': '+', # PN
			'HII': 's', # HII
			'YSO': 'P', # YSO
			'QSO': 'v', # QSO
			'PULSAR': 'd', # pulsar
		}

		self.marker_color_mapping= {
			'UNKNOWN': 'k', # unknown
			'MIXED_TYPE': 'tab:gray', # mixed type
			'STAR': 'r', # star
			'GALAXY': 'm', # galaxy
			'PN': 'g', # PN
			'HII': 'b', # HII
			'YSO': 'y', # YSO
			'QSO': 'c', # QSO
			'PULSAR': 'tab:orange', # pulsar
		}
		
		# - Output data
		self.save_mom_pars= False
		self.save_mom_max_no= 1
		self.save_zern_mom_pars= False
		self.save_hu_mom_pars= False
		self.save_source_pars= True
		self.save_ssim_pars= False
		self.save_cind_pars= False

		self.outfile= "features.dat"

	#####################################
	##     SETTERS/GETTERS
	#####################################
	def set_image_size(self,nx,ny):
		""" Set image size """	
		self.nx= nx
		self.ny= ny


	#####################################
	##     SET DATA
	#####################################
	def __set_data(self):
		""" Set train data & generator from loader """

		# - Retrieve info from data loader
		self.nchannels= self.dl.nchannels
		self.source_labels= self.dl.labels
		self.source_ids= self.dl.classids
		self.source_names= self.dl.snames
		self.nsamples= len(self.source_labels)

		if self.nmaximgs==-1:
			self.nmaximgs= self.nsamples

		# - Create standard generator
		self.data_generator= self.dl.data_generator(
			batch_size=1, 
			shuffle=self.shuffle_data,
			resize=self.resize, nx=self.nx, ny=self.ny, 
			normalize=self.normalize_img, scale_to_abs_max=self.scale_to_abs_max, scale_to_max=self.scale_to_max,
			augment=self.augmentation,
			log_transform=self.log_transform_img,
			scale=self.scale_img, scale_factors=self.scale_img_factors,
			standardize=self.standardize_img, means=self.img_means, sigmas=self.img_sigmas,
			chan_divide=self.chan_divide, chan_mins=self.chan_mins,
			erode=self.erode, erode_kernel=self.erode_kernel,
			retsdata=True
		)

		return 0


	
	#####################################
	##     RUN
	#####################################
	def run(self):
		""" Extract features """

		#===========================
		#==   SET DATA
		#===========================	
		# - Read data
		logger.info("Setting input data from data loader ...")
		status= self.__set_data()
		if status<0:
			logger.error("Input data set failed!")
			return -1

		# - Set feature extractor helper
		feh= FeatExtractorHelper()
		feh.refch= self.refch
		feh.normalize_img= self.normalize_img
		feh.scale_to_max= self.scale_to_max
		feh.chan_mins= self.chan_mins
		feh.seed_thr= self.seed_thr
		feh.merge_thr= self.merge_thr
		feh.sigma_clip= self.sigma_clip
		feh.subtract_bkg= self.subtract_bkg
		feh.dist_thr= self.dist_thr
		feh.dilatemask= self.dilatemask
		feh.kernsize= self.kernsize
		feh.use_sfind_mask= self.use_sfind_mask

		feh.fthr_zeros= self.fthr_zeros
		feh.compute_ssim_params= self.compute_ssim_params
		feh.winsize= self.winsize
		feh.compute_cind_params= self.compute_cind_params
		feh.colorind_safe= self.colorind_safe
		feh.colorind_thr= self.colorind_thr
		feh.ssim_thr= self.ssim_thr
		feh.weight_colmap_with_ssim= self.weight_colmap_with_ssim

		feh.save_mom_pars= self.save_mom_pars
		feh.save_mom_max_no= self.save_mom_max_no
		feh.save_zern_mom_pars= self.save_zern_mom_pars
		feh.save_hu_mom_pars= self.save_hu_mom_pars
		feh.save_source_pars= self.save_source_pars
		feh.save_ssim_pars= self.save_ssim_pars
		feh.save_cind_pars= self.save_cind_pars

		feh.draw= self.draw_plots
		
		#===========================
		#==   EXTRACT FEATURES
		#===========================
		img_counter= 0
		par_dict_list= []
		failed= False
		
		while True:
			try:
				# - Stop generator?
				if img_counter>=self.nmaximgs:
					logger.info("Sample size (%d) reached, stop generation..." % self.nmaximgs)
					break

				# - Get data from generator
				data, sdata= next(self.data_generator)
				img_counter+= 1

				nchans= data.shape[3]
				sname= sdata.sname
				label= sdata.label
				classid= sdata.id
				
				# - Set data in feature extractor helper
				feh.set_data(sdata, data)

				# - Extracting features
				logger.info("Extracting features from image sample no. %d (name=%s, id=%d) ..." % (img_counter, sname, classid))
				if feh.extract_features()<0:
					logger.warn("Failed to extract features from image sample no. %d (name=%s, id=%d), skip it ..." % (img_counter, sname, classid))
					continue
				
				par_dict= feh.param_dict
				if par_dict is None or not par_dict:
					logger.warn("Feature dist for image sample no. %d (name=%s, id=%d) is empty or None, skip it ..." % (img_counter, sname, classid))
					continue

				par_dict_list.append(par_dict)

				# - Stop generator
				if img_counter>=self.nmaximgs:
					logger.info("Sample size (%d) reached, stop generation..." % self.nmaximgs)
					break

			except (GeneratorExit, KeyboardInterrupt):
				logger.info("Stop loop (keyboard interrupt) ...")
				break
			except Exception as e:
				logger.warn("Stop loop (exception catched %s) ..." % str(e))
				failed= True
				break

		if failed:
			return -1

		#===========================
		#==   SAVE PARAM FILE
		#===========================
		if par_dict_list:
			logger.info("Saving parameter file %s ..." % (self.outfile))
			parnames = par_dict_list[0].keys()
			print("parnames")
			print(parnames)
		
			#with open(self.outfile, 'wb') as fp:
			with open(self.outfile, 'w') as fp:
				fp.write("# ")
				dict_writer = csv.DictWriter(fp, parnames)
				dict_writer.writeheader()
				dict_writer.writerows(par_dict_list)
		else:
			logger.warn("Parameter dict list is empty, no files will be written!")

		return 0

	
