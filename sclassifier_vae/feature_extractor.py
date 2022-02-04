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
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity
from skimage.measure import moments_central, moments_normalized, moments_hu, moments, regionprops
from scipy.stats import kurtosis, skew, median_absolute_deviation

## GRAPHICS MODULES
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Agg')

## PACKAGE MODULES
from .utils import Utils
from .data_loader import DataLoader
from .data_loader import SourceData



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

		self.fthr_zeros= 0.1

		# - SSIM options
		self.winsize= 3
		self.ssim_thr= 0.

		# - Hu Moment options
		self.nmoments_save= 4
		
		# - Color index options
		self.colorind_safe= 0
		self.colorind_thr= 6
		self.weight_colmap_with_ssim= False
		
		# - Draw options
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
		self.save_imgs= False
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
	##     VALIDATE IMAGE DATA
	#####################################
	def __validate_img(self, data, sdata):
		""" Perform some validation on input image """

		# - Retrieve some data fields
		nchannels= data.shape[3]
		sname= sdata.sname
		label= sdata.label
		classid= sdata.id

		# - Check for NANs
		has_naninf= np.any(~np.isfinite(data))
		if has_naninf:
			logger.warn("Image (name=%s, label=%s) has some nan/inf, validation failed!" % (sname, label))
			return False

		# - Check for fraction of zeros in radio mask
		cond= np.logical_and(data[0,:,:,0]!=0, np.isfinite(data[0,:,:,0]))
		for i in range(1,nchannels):
			data_2d= data[0,:,:,i]
			data_1d= data_2d[cond]
			n= data_1d.size
			n_zeros= np.count_nonzero(data_1d==0)
			f= n_zeros/n
			if n_zeros>0:
				logger.debug("Image chan %d (name=%s, label=%s): n=%d, n_zeros=%d, f=%f" % (i+1, sname, label, n, n_zeros, f))
				
			if f>=self.fthr_zeros:
				logger.warn("Image chan %d (name=%s, label=%s) has a zero fraction %f>%f, validation failed!" % (i+1, sname, label, f, self.fthr_zeros))
				return False

			# - Check if channels have all equal values 
			for i in range(nchannels):
				data_min= np.min(data[0,:,:,i])
				data_max= np.max(data[0,:,:,i])
				same_values= (data_min==data_max)
				if same_values:
					logger.warn("Image chan %d (name=%s, label=%s) has all elements equal to %f, validation failed!" % (i+1, sname, label, data_min))
					return False
			
			# - Check correct norm
			if self.normalize_img:
				data_min= np.min(data[0,:,:,:])
				data_max= np.max(data[0,:,:,:])
				if self.scale_to_max:
					correct_norm= (data_max==1)
				else:
					correct_norm= (data_min==0 and data_max==1)
				if not correct_norm:
					logger.warn("Image chan %d (name=%s, label=%s) has invalid norm (%f,%f), validation failed!" % (i+1, sname, label, data_min, data_max))
					return False

		return True
		
	#####################################
	##     EXTRACT FEATURES
	#####################################
	def __extract_img_moments(self, data, mask=None, centroid=None):
		""" Extract moments from images """

		# - Compute mask if not given 
		if mask is None:
			mask= np.copy(data)
			cond= np.logical_and(data!=0, np.isfinite(data))
			mask[cond]= 1
			mask= mask.astype(np.int32)

		# - Compute region
		try:
			regprops= regionprops(label_image=mask, intensity_image=data) 
		except Exception as e:
			logger.error("Failed to extract region props (err=%s)" % str(e))
			return None

		if len(regprops)<=0:
			logger.error("No region with non-zero pixels detected, please check!")
			return None
		if len(regprops)>1:
			logger.warn("More than 1 region with non-zero pixels detected, please check!")
			return None

		regprop= regprops[0]
		
		# - Compute centroid if not given, otherwise override
		if centroid is None:
			try:
				# - from scikit API 0.17.2 (last supported for python3.6)
				centroid= regprop.weighted_local_centroid
				#centroid= regprop.centroid
				#centroid_local= regprop.local_centroid
				#centroid_w= regprop.weighted_centroid
				#centroid_local_w= regprop.weighted_local_centroid
			except:
				try:
					# - from scikit API >0.17.2
					centroid= regprop.centroid_weighted_local
					#centroid= regprop.centroid
					#centroid_local= regprop.centroid_local
					#centroid_w= regprop.centroid_weighted
					#centroid_local_w= regprop.centroid_weighted_local
				except:
					logger.error("Failed to get region centroids (check scikit-image API!)")
					return None
			
			print("--> centroid")
			print(centroid)

		else:
			# - Override centroid with passed one
			try:
				regprop.regprop.weighted_local_centroid= centroid
			except:
				try:
					regprop.centroid_weighted_local= centroid
				except:
					logger.error("Failed to get region centroids (check scikit-image API!)")
					return None
			print("--> overrdden contour")
			print(regprop.regprop.weighted_local_centroid)

		# - Compute Hu moments
		try:
			#moments= regprop.weighted_moments_hu	
			moments= regprop.weighted_moments_hu
		except:
			try:
				#moments= regprop.moments_weighted_hu
				moments= regprop.moments_weighted_hu
			except:
				logger.error("Failed to get region centroids (check scikit-image API!)")
				return None

		# - Compute central moments
		

		return (moments, mask, centroid)


	def __extract_features(self, data, sdata, save_imgs=False):
		""" Extract image features """

		# - Retrieve some data fields
		nchans= data.shape[3]
		sname= sdata.sname
		label= sdata.label
		classid= sdata.id

		if not self.chan_mins:
			self.chan_mins= [0]*nchans
		#print("--> self.chan_mins")
		#print(self.chan_mins)

		param_dict= collections.OrderedDict()
		param_dict["sname"]= sname
		param_dict["label"]= label
		param_dict["classid"]= classid

		# - Init save plot	
		if save_imgs:
			outfile_plot= sname + '_id' + str(classid) + '.png'		
			logger.info("Saving plot to file %s ..." % (outfile_plot))
			fig = plt.figure(figsize=(20, 10))

			plot_nrows= int(nchans*(nchans-1)/2)
			plot_ncols= 4

		# - Compute centroid from reference channel (needed for moment calculation)
		logger.info("Computing centroid from reference channel (ch=%d) for image %s (id=%s) ..." % (self.refch, sname, label))
		ret= self.__extract_img_moments(data[0,:,:,self.refch])
		if ret is None:
			logger.error("Failed to compute ref channel mask centroid for image %s (id=%s, ch=%d)!" % (sname, label, i+1))
			return None
		mask= ret[1]
		centroid= ret[2]
		
		# - Compute Hu moments of intensity images	
		#   NB: use same mask and centroid from refch for all channels
		#for i in range(nchans):
		#	img_i= data[0,:,:,i]
		#	ret= self.__extract_img_moments(img_i, mask, centroid)
		#	if ret is None:
		#		logger.error("Failed to compute moments for image %s (id=%s, ch=%d)!" % (sname, label, i+1))
		#		return None
		#	moments= ret[0]

		#	print("== IMG HU-MOMENTS (CH%d) ==" % (i+1))
		#	print(moments)
			
		#	for j in range(len(moments)):
		#		m= moments[j]
		#		#parname= "mom" + str(j+1) + "_ch" + str(i+1)
		#		#param_dict[parname]= m
			

		# - Loop over images and compute total flux	
		logger.info("Computing image stats (Stot, Smin/Smax) for image %s (id=%s) ..." % (sname, label))
		for i in range(nchans):
			img_i= data[0,:,:,i]
			cond_i= np.logical_and(img_i!=0, np.isfinite(img_i))	
			img_max_i= np.nanmax(img_i[cond_i])
			img_min_i= np.nanmin(img_i[cond_i])
			
			S= np.nansum(img_i[cond_i])
			parname= "Smin_ch" + str(i+1)
			param_dict[parname]= img_min_i
			parname= "Smax_ch" + str(i+1)
			param_dict[parname]= img_max_i
			parname= "Stot_ch" + str(i+1)
			param_dict[parname]= S

		# - Loop over images and compute params
		index= 0
		for i in range(nchans):
			
			img_i= data[0,:,:,i]
			cond_i= np.logical_and(img_i!=0, np.isfinite(img_i))

			
			img_max_i= np.nanmax(img_i[cond_i])
			img_min_i= np.nanmin(img_i[cond_i])
			

			img_norm_i= (img_i-img_min_i)/(img_max_i-img_min_i)
			img_norm_i[~cond_i]= 0

			img_posdef_i= img_i - self.chan_mins[i]
			img_posdef_i[~cond_i]= 0

			#cond_col_i= np.logical_and(img_i>0, np.isfinite(img_i))
			cond_col_i= np.logical_and(img_posdef_i>0, np.isfinite(img_posdef_i))


			# - Compute SSIM and color indices
			for j in range(i+1,nchans):
				img_j= data[0,:,:,j]
				cond_j= np.logical_and(img_j!=0, np.isfinite(img_j))
				img_max_j= np.nanmax(img_j[cond_j])
				img_min_j= np.nanmin(img_j[cond_j])
				
				img_norm_j= (img_j-img_min_j)/(img_max_j-img_min_j)
				img_norm_j[~cond_j]= 0

				img_posdef_j= img_j - self.chan_mins[j]
				img_posdef_j[~cond_j]= 0

				cond= np.logical_and(cond_i, cond_j)
				img_1d_i= img_i[cond]
				img_1d_j= img_j[cond]

				#cond_col_j= np.logical_and(img_j>0, np.isfinite(img_j))
				cond_col_j= np.logical_and(img_posdef_j>0, np.isfinite(img_posdef_j))
				cond_col_ij= np.logical_and(cond_col_i, cond_col_j)
				
				# - Compute SSIM moments
				#   NB: Need to normalize images to max otherwise the returned values are always ~1.
				###img_max= np.max([inputdata_img,recdata_img])
				#ssim_mean, ssim_2d= structural_similarity(img_i/img_max_i, img_j/img_max_j, full=True, win_size=self.winsize, data_range=1)

				logger.info("Computing SSIM for image %s (id=%s, ch=%d-%d) ..." % (sname, label, i+1, j+1))
				ssim_mean, ssim_2d= structural_similarity(img_norm_i, img_norm_j, full=True, win_size=self.winsize, data_range=1)

				ssim_2d[ssim_2d<0]= 0
				ssim_2d[~cond]= 0

				ssim_1d= ssim_2d[cond]

				#plt.subplot(1, 3, 1)
				#plt.imshow(img_norm_i, origin='lower')
				#plt.colorbar()

				#plt.subplot(1, 3, 2)
				#plt.imshow(img_norm_j, origin='lower')
				#plt.colorbar()
					
				#plt.subplot(1, 3, 3)
				#plt.imshow(ssim_2d, origin='lower')
				#plt.colorbar()

				#plt.show()

				if ssim_1d.size>0:
					ssim_mean_mask= np.nanmean(ssim_1d)
					ssim_min_mask= np.nanmin(ssim_1d)
					ssim_max_mask= np.nanmax(ssim_1d)
					ssim_std_mask= np.nanstd(ssim_1d)
					ssim_median_mask= np.nanmedian(ssim_1d)
					ssim_mad_mask= median_absolute_deviation(ssim_1d)
				
					ret= self.__extract_img_moments(ssim_2d, mask, centroid)
					if ret is None:
						logger.warn("Failed to compute SSIM moments for image %s (id=%s, ch=%d-%d)!" % (sname, label, i+1, j+1))
						
					moments_ssim= ret[0]
					badcounts= np.count_nonzero(~np.isfinite(moments_ssim))
					if badcounts>0:
						logger.warn("Some SSIM moments for image %s (id=%s, ch=%d-%d) is not-finite (%s), setting all to -999..." % (sname, label, i+1, j+1, str(moments_ssim)))
						moments_ssim= [-999]*7
						
				else:
					logger.warn("Image %s (chan=%d-%d): SSIM array is empty, setting estimators to -999..." % (sname, i+1, j+1))
					ssim_mean_mask= -999
					ssim_min_mask= -999
					ssim_max_mask= -999
					ssim_std_mask= -999
					ssim_median_mask= -999
					ssim_mad_mask= -999
					moments_ssim= [-999]*7
				
				if not np.isfinite(ssim_mean_mask):
					logger.warn("Image %s (chan=%d-%d): ssim_mean_mask is nan/inf!" % (sname, i+1, j+1))
					ssim_mean_mask= -999


				#print("== SSIM HU-MOMENTS CH%d-%d ==" % (i+1, j+1))
				#print(moments_ssim)

				parname= "ssim_mean_ch{}_{}".format(i+1,j+1)
				param_dict[parname]= ssim_mean_mask
				parname= "ssim_min_ch{}_{}".format(i+1,j+1)
				param_dict[parname]= ssim_min_mask
				parname= "ssim_max_ch{}_{}".format(i+1,j+1)
				param_dict[parname]= ssim_max_mask
				parname= "ssim_std_ch{}_{}".format(i+1,j+1)
				param_dict[parname]= ssim_std_mask
				parname= "ssim_median_ch{}_{}".format(i+1,j+1)
				param_dict[parname]= ssim_median_mask
				parname= "ssim_mad_ch{}_{}".format(i+1,j+1)
				param_dict[parname]= ssim_mad_mask

				#for k in range(len(moments_ssim)):
				for k in range(self.nmoments_save):
					m= moments_ssim[k]
					parname= "ssim_mom{}_ch{}_{}".format(k+1,i+1,j+1)
					param_dict[parname]= m
			

				# - Compute flux ratios and moments
				#####cond_colors= np.logical_and(cond_col_ij, ssim_2d>self.ssim_thr)
				#cond_colors= cond_col_ij
				#####fluxratio_2d= np.divide(img_i, img_j, where=cond_colors, out=np.ones(img_i.shape)*np.nan)
				#fluxratio_2d= np.divide(img_posdef_i, img_posdef_j, where=cond_colors, out=np.ones(img_posdef_i.shape)*np.nan)
				#cond_fluxratio= np.isfinite(fluxratio_2d)
				#fluxratio_1d= fluxratio_2d[cond_fluxratio]
				#fluxratio_ssim_1d= ssim_2d[cond_fluxratio]

				#if fluxratio_1d.size>0:
				#	fluxratio_mean= np.nanmean(fluxratio_1d)
				#	fluxratio_std= np.std(fluxratio_1d)
				#	fluxratio_skew= skew(fluxratio_1d)
				#	fluxratio_kurt= kurtosis(fluxratio_1d)	
				#	fluxratio_min= np.nanmin(fluxratio_1d)
				#	fluxratio_max= np.nanmax(fluxratio_1d)
				#else:
				#	logger.warn("Image %s (chan=%d-%d): flux ratio array is empty, setting estimators to -999..." % (sname, i+1, j+1))
				#	fluxratio_mean= -999
				#	fluxratio_std= -999
				#	fluxratio_skew= -999
				#	fluxratio_kurt= -999
				#	fluxratio_min= -999
				#	fluxratio_max= -999

				#if fluxratio_1d.size>0 and fluxratio_ssim_1d.size>0: 
				#	fluxratio_weighted_mean= Utils.weighted_mean(fluxratio_1d, fluxratio_ssim_1d)
				#	fluxratio_weighted_std= Utils.weighted_std(fluxratio_1d, fluxratio_ssim_1d)
				#	fluxratio_weighted_skew= Utils.weighted_skew(fluxratio_1d, fluxratio_ssim_1d)
				#	fluxratio_weighted_kurt= Utils.weighted_kurtosis(fluxratio_1d, fluxratio_ssim_1d)
				#else:
				#	logger.warn("Image %s (chan=%d-%d): flux ratio or weights array are empty, setting weighted estimators to -999..." % (sname, i+1, j+1))
				#	fluxratio_weighted_mean= -999
				#	fluxratio_weighted_std= -999
				#	fluxratio_weighted_skew= -999
				#	fluxratio_weighted_kurt= -999

				#parname= "fratio_mean_ch{}_{}".format(i+1,j+1)
				#param_dict[parname]= fluxratio_mean
				#parname= "fratio_std_ch{}_{}".format(i+1,j+1)
				#param_dict[parname]= fluxratio_std
				#parname= "fratio_skew_ch{}_{}".format(i+1,j+1)
				#param_dict[parname]= fluxratio_skew
				#parname= "fratio_kurt_ch{}_{}".format(i+1,j+1)
				#param_dict[parname]= fluxratio_kurt
				#parname= "fratio_min_ch{}_{}".format(i+1,j+1)
				#param_dict[parname]= fluxratio_min	
				#parname= "fratio_max_ch{}_{}".format(i+1,j+1)
				#param_dict[parname]= fluxratio_max

				#parname= "fratio_wmean_ch{}_{}".format(i+1,j+1)
				#param_dict[parname]= fluxratio_weighted_mean
				#parname= "fratio_wstd_ch{}_{}".format(i+1,j+1)
				#param_dict[parname]= fluxratio_weighted_std
				#parname= "fratio_wskew_ch{}_{}".format(i+1,j+1)
				#param_dict[parname]= fluxratio_weighted_skew
				#parname= "fratio_wkurt_ch{}_{}".format(i+1,j+1)
				#param_dict[parname]= fluxratio_weighted_kurt

				# - Compute color index map
				logger.info("Computing color index map for image %s (id=%s, ch=%d-%d) ..." % (sname, label, i+1, j+1))
				#cond_colors= cond_col_ij
				cond_colors= np.logical_and(cond_col_ij, ssim_2d>self.ssim_thr)
				colorind_2d= np.log10( np.divide(img_posdef_i, img_posdef_j, where=cond_colors, out=np.ones(img_posdef_i.shape)*1) )
				cond_colors_safe= np.logical_and(cond_colors, np.fabs(colorind_2d)<self.colorind_thr)
				colorind_2d+= self.colorind_thr
				
				if self.weight_colmap_with_ssim:
					colorind_2d*= ssim_2d
					
				colorind_2d[~cond_colors_safe]= self.colorind_safe
				
				#cond_colorind= np.isfinite(colorind_2d)
				cond_colorind= np.logical_and(np.isfinite(colorind_2d), colorind_2d!=self.colorind_safe)
				colorind_1d= colorind_2d[cond_colorind]

				if colorind_1d.size>0:
					colorind_mean= np.nanmean(colorind_1d)
					colorind_std= np.std(colorind_1d)
					colorind_skew= skew(colorind_1d)
					colorind_kurt= kurtosis(colorind_1d)	
					colorind_min= np.nanmin(colorind_1d)
					colorind_max= np.nanmax(colorind_1d)
					colorind_median= np.nanmedian(colorind_1d)
					colorind_mad= median_absolute_deviation(colorind_1d)
				
					ret= self.__extract_img_moments(colorind_2d, mask, centroid)
					if ret is None:
						logger.warn("Failed to compute moments for color index image %s (id=%s, ch=%d-%d)!" % (sname, label, i+1, j+1))
						
					moments_colorind= ret[0]
					badcounts= np.count_nonzero(~np.isfinite(moments_colorind))
					if badcounts>0:
						logger.warn("Some color index moments for image %s (id=%s, ch=%d-%d) is not-finite (%s), setting all to -999..." % (sname, label, i+1, j+1, str(moments_colorind)))
						moments_colorind= [-999]*7

				else:
					logger.warn("Image %s (chan=%d-%d): color index array is empty, setting estimators to -999..." % (sname, i+1, j+1))
					colorind_mean= -999
					colorind_std= -999
					colorind_skew= -999
					colorind_kurt= -999
					colorind_min= -999
					colorind_max= -999
					colorind_median= -999
					colorind_mad= -999
					moments_colorind= [-999]*7

				parname= "cind_mean_ch{}_{}".format(i+1,j+1)
				param_dict[parname]= colorind_mean
				parname= "cind_min_ch{}_{}".format(i+1,j+1)
				param_dict[parname]= colorind_min
				parname= "cind_max_ch{}_{}".format(i+1,j+1)
				param_dict[parname]= colorind_max
				parname= "cind_std_ch{}_{}".format(i+1,j+1)
				param_dict[parname]= colorind_std			
				parname= "cind_median_ch{}_{}".format(i+1,j+1)
				param_dict[parname]= colorind_median
				parname= "cind_mad_ch{}_{}".format(i+1,j+1)
				param_dict[parname]= colorind_mad			

				#print("== COLOR HU-MOMENTS CH%d-%d ==" % (i+1, j+1))
				#print(moments_colorind)

				#for k in range(len(moments_colorind)):
				for k in range(self.nmoments_save):
					m= moments_colorind[k]
					parname= "cind_mom{}_ch{}_{}".format(k+1,i+1,j+1)
					param_dict[parname]= m

				plt.subplot(2, 2, 1)
				plt.imshow(img_posdef_i, origin='lower')
				plt.colorbar()

				plt.subplot(2, 2, 2)
				plt.imshow(img_posdef_j, origin='lower')
				plt.colorbar()

				plt.subplot(2, 2, 3)
				plt.imshow(ssim_2d, origin='lower')
				plt.colorbar()
					
				plt.subplot(2, 2, 4)
				plt.imshow(colorind_2d, origin='lower')
				plt.colorbar()

				plt.show()

				# - Save images
				if save_imgs:
					
					# - Save ssim map
					plot_index= plot_ncols*index + 1
					logger.info("Adding subplot (%d,%d,%d) ..." % (plot_nrows,plot_ncols,plot_index))
					
					plt.subplot(plot_nrows, plot_ncols, plot_index)
					plt.imshow(ssim_2d, origin='lower')
					plt.colorbar()

					# - Save flux ratio map
					plot_index= plot_ncols*index + 2
					logger.info("Adding subplot (%d,%d,%d) ..." % (plot_nrows,plot_ncols,plot_index))
					
					plt.subplot(plot_nrows, plot_ncols, plot_index)
					plt.imshow(fluxratio_2d, origin='lower')
					plt.colorbar()

					# - Save flux ratio histogram
					plot_index= plot_ncols*index + 3
					logger.info("Adding subplot (%d,%d,%d) ..." % (plot_nrows,plot_ncols,plot_index))
					
					plt.subplot(plot_nrows, plot_ncols, plot_index)
					plt.hist(fluxratio_1d, bins='auto')

					# - Save weighted flux ratio histogram
					plot_index= plot_ncols*index + 4
					logger.info("Adding subplot (%d,%d,%d) ..." % (plot_nrows,plot_ncols,plot_index))
					
					plt.subplot(plot_nrows, plot_ncols, plot_index)
					plt.hist(fluxratio_1d, weights=fluxratio_ssim_1d, bins=20)

					
				index+= 1	

		# - Save image
		if save_imgs:
			plt.savefig(outfile_plot)
			#plt.tight_layout()
			#plt.show()
			plt.close()
	
		return param_dict

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

				# - Validate data
				if not self.__validate_img(data, sdata):
					logger.warn("Validation failed for image sample no. %d (name=%s, id=%d), skip it ..." % (img_counter, sname, classid))
					continue

				# - Extracting features
				logger.info("Extracting features from image sample no. %d (name=%s, id=%d) ..." % (img_counter, sname, classid))
				par_dict= self.__extract_features(data, sdata, save_imgs=self.save_imgs)
				if par_dict is None:
					logger.warn("Failed to extract features from image sample no. %d (name=%s, id=%d), skip it ..." % (img_counter, sname, classid))
					continue
				if par_dict:
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

	
