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
import shutil
import csv

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections

## MODULES
from astropy.io import fits
from astropy.wcs import WCS
import regions
from scipy.stats import linregress

## MODULES
from sclassifier_vae import logger
from sclassifier_vae.utils import Utils

#####################################
##   SpectralIndexTTHelper class
#####################################
class SpectralIndexTTHelper(object):

	def __init__(self, datadict):
		""" Return a SpectralIndexTTHelper object """	
	
		# - Source info
		self.filepaths= []
		self.sname= "XXX"
		self.label= "UNKNOWN"
		self.id= -1

		# - Set source info from input data dict
		self.set_from_dict(datadict)

		# - Image data
		self.img_data= []
		self.img_data_mask= []
		self.img_heads= []
		self.nx= 0
		self.ny= 0
		self.nchannels= 0
		self.refch= 0
		self.img_freqs_head= []
		self.img_freqs= []

		# - Quality check options
		self.negative_pix_fract_thr= 0.9
		self.bad_pix_fract_thr= 0.05

		# - Spectral index options
		self.alpha= -999
		self.alpha_err= -999
		self.rcoeff= -999
		self.rcoeff_thr= 0.9
		self.has_good_alpha= False

		# - Output options
		self.param_dict= collections.OrderedDict()


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

	#####################################
	##     CHECK DATA
	#####################################
	def __has_good_data(self, check_mask=False, check_bad=True, check_neg=True, check_same=True):
		""" Check data integrity (nan, negative, etc) """
			
		is_good= True		
		for i in range(self.nchannels):
			
			data= self.img_data[i]
			mask= self.img_data_mask[i]
			data_masked= data[mask==1]
			if check_mask:
				data= data_masked

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

	#============================
	#==      READ IMGS
	#============================
	def __read_imgs(self):
		""" Read image data from paths """

		# - Check data filelists
		if not self.filepaths:
			logger.error("Empty filelists given!")
			return -1

		# - Read images
		nimgs= len(self.filepaths)
		self.nchannels= nimgs
		has_freq_data= True

		for filename in self.filepaths:
			# - Read image
			logger.debug("Reading file %s ..." % (filename)) 
			data= None
			try:
				data, header, wcs= Utils.read_fits(filename)
			except Exception as e:
				logger.error("Failed to read image data from file %s (err=%s)!" % (filename, str(e)))
				return -1

			# - Compute data mask
			#   NB: =1 good values, =0 bad (pix=0 or pix=inf or pix=nan)
			data_mask= np.logical_and(data!=0, np.isfinite(data)).astype(np.uint8)
		
			# - Extract frequency information from header
			has_freq_in_header= False
			freq= -999
			if 'CRVAL3' in header and 'CTYPE3' in header:
				axis_type= header['CTYPE']
				if axis_type=="FREQ":
					freq= header['CRVAL3']
					has_freq_in_header= True	
				else:		
					has_freq_data= False
			else:
				has_freq_data= False

			# - Append image channel data to list
			self.img_data.append(data)
			self.img_heads.append(header)
			self.img_data_mask.append(data_mask)
			self.img_freqs_head.append(freq)
		
		# - Reset freq data if one of the channel has no data
		if not has_freq_data:
			self.img_freqs_head= []

		# - Check image sizes
		if not self.check_img_sizes():
			logger.error("Image channels for source %s do not have the same size, check your dataset!" % self.sname)
			return -1

		# - Set data shapes
		self.nx= self.img_data[0].shape[1]
		self.ny= self.img_data[0].shape[0]
		self.nchannels= len(self.img_data)

		return 0

	#============================
	#==  COMPUTE SPECTRAL INDEX
	#============================
	def __slope2alpha(self, slope, nu_x, nu_y):
		""" Compute alpha from slope """

		alpha= np.log10(slope)/np.log10(nu_y/nu_x)
		return alpha

	def __compute_alpha(self, data_1, data_2, nu1, nu2, smask):
		""" Compute alpha """

		# - Get array of pixels !=0 & finite in both maps
		cond_img1= np.logical_and(data_1!=0, np.isfinite(data_1))
		cond_img2= np.logical_and(data_2!=0, np.isfinite(data_2))
		cond_img12= np.logical_and(cond_img1, cond_img2)
		cond_final= np.logical_and(cond_img12, smask==1)

		indexes= np.where(cond_final)
		img_1d_1= data_1[indexes]
		img_1d_2= data_2[indexes]
	
		logger.info("#%d pixels in image 1 ..." % (len(img_1d_1)))
		logger.info("#%d pixels in image 2 ..." % (len(img_1d_2)))

		if len(img_1d_1)<=0 or len(img_1d_2)<0:
			logger.warn("No pixels left for T-T analysis after applying conditions (finite+mask) (hint: check if source is outside one or more channels)")
			return None

		# - Perform fit 1-2
		logger.info("Compute spectral index from T-T fit  ...")
		res_12 = linregress(img_1d_1, img_1d_2)
		slope_12= res_12.slope
		intercept_12= res_12.intercept
		alpha_12= self.__slope2alpha(slope_12, nu1, nu2)
		r_12= res_12.rvalue
	
		print("== FIT RES 1-2 ==")
		print(res_12)
		print("alpha_12=%f" % (alpha_12))

		# - Perform fit 2-1
		res_21 = linregress(img_1d_2, img_1d_1)
		slope_21= res_21.slope
		intercept_21= res_21.intercept
		alpha_21= self.__slope2alpha(slope_21, nu2, nu1)
		r_21= res_21.rvalue
	
		print("== FIT RES 2-1 ==")
		print(res_21)
		print("alpha_21=%f" % (alpha_21))

		# - Reject fits if any of them is nan
		goodvalues_12= np.isfinite(slope_12) and slope_12>0
		goodvalues_21= np.isfinite(slope_21) and slope_21>0
	
		# - Add some goodness of fit criteria
		obs_12 = img_1d_2
		pred_12 = slope_12 * img_1d_1 + intercept_12
		residuals_12= obs_12 - pred_12
		residuals_mean_12= np.mean(residuals_12)
		residuals_std_12= np.std(residuals_12)
		residuals_min_12= np.min(residuals_12)
		residuals_max_12= np.max(residuals_12)

		obs_21 = img_1d_1
		pred_21 = slope_21 * img_1d_2 + intercept_21
		residuals_21= obs_21 - pred_21
		residuals_mean_21= np.mean(residuals_21)
		residuals_std_21= np.std(residuals_21)
		residuals_min_21= np.min(residuals_21)
		residuals_max_21= np.max(residuals_21)

		# - Set return tuple
		outtuple= ()
		if goodvalues_12 and not goodvalues_21:
			outtuple= (alpha_12, r_12, residuals_mean_12, residuals_std_12, residuals_min_12, residuals_max_12)
		elif goodvalues_21 and not goodvalues_12:
			outtuple= (alpha_21, r_21, residuals_mean_21, residuals_std_21, residuals_min_21, residuals_max_21)
		else:
			# - Select best model	
			best_resbias_id= 1
			best_resstd_id= 1
			best_rcoeff_id= 1
			if np.abs(residuals_mean_21)<np.abs(residuals_mean_12): # check smallest residual bias
				best_resbias_id= 2
			if np.abs(residuals_std_21)<np.abs(residuals_std_12): # check smallest residual std dev
				best_resstd_id= 2
			if np.abs(r_21)>np.abs(r_12): # check larger (closer to 1) correlation coeff 
				best_rcoeff_id= 2
	
			if best_rcoeff_id==1:
				outtuple= (alpha_12, r_12, residuals_mean_12, residuals_std_12, residuals_min_12, residuals_max_12)
			else:
				outtuple= (alpha_21, r_21, residuals_mean_21, residuals_std_21, residuals_min_21, residuals_max_21)
		
		return outtuple
	
	
	#==================================
  #==   COMPUTE SPECTRAL INDEX
  #==================================
	def __compute_spectral_index(self, img_group_1, img_group_2):
		""" Compute spectral index alpha """

		# - Check first if frequency data are available
		print("self.img_freqs")
		print(self.img_freqs)
		print("len(self.img_freqs)")
		print(len(self.img_freqs))
		print("len(self.img_data)")
		print(len(self.img_data))
		print("img_group_1")
		print(img_group_1)
		print("img_group_2")
		print(img_group_2)
	

		freqs= []
		if self.img_freqs and len(self.img_freqs)==len(self.img_data):
			freqs= self.img_freqs
		else:
			if self.img_freqs_head and len(self.img_freqs_head)==len(self.img_data):
				freqs= self.img_freqs_head
			else:
				logger.error("No frequency data given (user/header)!")
				return -1

		# - Check group indexes
		if len(img_group_1)!=len(img_group_2):
			logger.error("Group indexes do not have the same length!")
			return -1

		# - Check group indices are within available channels
		for i in range(len(img_group_1)):
			index= img_group_1[i]
			if index<0 or index>=self.nchannels:	
				logger.error("Invalid index (%d) in group 1, must be in range [0,%d]!" % (index, self.nchannels-1))
				return -1

		for i in range(len(img_group_2)):
			index= img_group_2[i]
			if index<0 or index>=self.nchannels:	
				logger.error("Invalid index (%d) in group 2, must be in range [0,%d]!" % (index, self.nchannels-1))
				return -1

		# - Loop over img combinations and compute spectral indices
		logger.info("Computing spectral index (#%d combinations) ..." % (len(img_group_1)))
		alphas= []
		rcoeffs= []

		smask= self.img_data_mask[self.refch]

		for i in range(len(img_group_1)):
			index_1= img_group_1[i]
			index_2= img_group_2[i]
			data_1= self.img_data[index_1]
			data_2= self.img_data[index_2]
			
			# - Find frequency from header
			nu1= freqs[index_1]
			nu2= freqs[index_2]
			#alpha12, alpha21= compute_alpha(data_1, data_2, nu1, nu2, smask, draw_plots)
			#alpha= 0.5*(alpha12+alpha21)
			outtuple= self.__compute_alpha(data_1, data_2, nu1, nu2, smask)
			if outtuple is None:
				logger.warn("alpha calculation failed for map combination %d-%d, skip to next ..." % (index_1, index_2))
				continue

			alpha= outtuple[0]
			r= outtuple[1]
			alphas.append(alpha)
			rcoeffs.append(r)

		logger.info("Computing average spectral index ...")
		print(alphas)

		alphas= np.array(alphas)
		alphas_safe= alphas[np.isfinite(alphas)]
		alphas= alphas_safe
		if alphas.size==0:
			logger.warn("No alpha measurement left (all nans), will set alpha values to -999 ...")
			alpha_mean= -999
			alpha_median= -999
			alpha_min= -999
			alpha_max= -999
		else:
			alpha_mean= np.mean(alphas)
			alpha_median= np.median(alphas)
			alpha_min= np.min(alphas)
			alpha_max= np.max(alphas)

		rcoeffs= np.array(rcoeffs)
		rcoeffs_safe= rcoeffs[np.isfinite(rcoeffs)]
		rcoeffs= rcoeffs_safe
		if rcoeffs.size==0:
			logger.warn("No rcoeffs measurement left (all nans), will set alpha values to -999 ...")
			rcoeff_mean= -999
			rcoeff_median= -999
			rcoeff_min= -999
			rcoeff_max= -999
		else:
			rcoeff_mean= np.mean(rcoeffs)
			rcoeff_median= np.median(rcoeffs)
			rcoeff_min= np.min(rcoeffs)
			rcoeff_max= np.max(rcoeffs)

		# - Set spectral index
		self.alpha= alpha_mean
		self.rcoeff= rcoeff_mean
		if self.alpha!=-999 and self.rcoeff>=self.rcoeff_thr:
			self.has_good_alpha= True
		else:
			self.has_good_alpha= False

		return 0


	#============================
	#==      FILL DATA
	#============================
	def __fill_data(self):
		""" Fill data dictionary """
		
		# - Save name
		self.param_dict["sname"]= self.sname

		# - Save spectral index
		self.param_dict["alpha"]= self.alpha

		# - Save class id
		self.param_dict["id"]= self.id

	#============================
	#==      RUN
	#============================
	def run(self, img_group_1, img_group_2):
		""" Compute spectral index """

		# - Read image data
		if self.__read_imgs()<0:
			logger.error("Failed to read input imgs!")
			return -1
	
		# - Check data integrity
		good_data= self.__has_good_data(
			check_mask=True, 
			check_bad=True, 
			check_neg=True, 
			check_same=True
		)

		if not good_data:
			logger.warn("Source data selected as bad, skip this source...")
			return -1

		# - Compute spectral index
		if self.__compute_spectral_index(img_group_1, img_group_2)<0:
			logger.error("Failed to compute spectral index (see logs)!")
			return -1

		# - Fill dict data
		self.__fill_data()

		return 0

#####################################
##   SpectralIndexTTCalculator class
#####################################
class SpectralIndexTTCalculator(object):
	
	def __init__(self):
		""" Return a SpectralIndexTTCalculator object """	

		# - Input data options
		self.datalistfile= ""
		self.datalist= ""
		self.datasize= 0
		self.labels= []
		self.snames= []
		self.classids= []
		self.nchannels= 0

		# - Data quality options
		self.negative_pix_fract_thr= 0.9
		self.bad_pix_fract_thr= 0.05

		# - Alpha calculation options
		self.alpha_rcoeff_thr= 0.9
		self.img_freqs= []
		
		# - Output options
		self.save= True
		self.par_dict_list= []
		self.outfile= "features_alpha.csv"

	#===========================
	#==    RUN
	#===========================
	def run_from_datalist(self, datalist, img_group_1, img_group_2):
		""" Run spectral index calculation passing data dict lists as inputs """

		# - Set data info
		logger.debug("Setting data info ...")
		self.datalist= datalist
		self.datasize= len(self.datalist)
		self.labels= [item["label"] for item in self.datalist]
		self.snames= [item["sname"] for item in self.datalist]
		self.classids= 	[item["id"] for item in self.datalist]
		
		# - Check number of channels per image
		nchannels_set= set([len(item["filepaths"]) for item in self.datalist])
		if len(nchannels_set)!=1:
			logger.warn("Number of channels in each object instance is different (len(nchannels_set)=%d!=1)!" % (len(nchannels_set)))
			print(nchannels_set)
			return -1

		self.nchannels= list(nchannels_set)[0]

		# - Loop over data and extract params per each source
		logger.info("Loop over data and extract params per each source ...")
		for i in range(self.datasize):
			if self.__process_source(i, img_group_1, img_group_2)<0:
				logger.warn("Failed to process source %d, skip to next ..." % (i))
				continue 

		# - Save data
		if self.save:
			logger.info("Saving data to file %s ..." % (self.outfile))
			self.__save_data()

		return 0


	def run_from_datalistfile(self, datafile, img_group_1, img_group_2):
		""" Run spectral index calculation using data json filelist as input """

		# - Read data filelist
		logger.info("Reading data filelist ...")
		ret= self.__read_filelist(datafile)
		if ret is None:
			logger.error("Failed to read filelist %s for imgs!" % (datafile))
			return -1
		datadict= ret[0]
		nchannels_set= ret[1]
		
		self.datalistfile= datafile
		self.datalist= datadict["data"]
		self.nchannels= list(nchannels_set)[0]
		self.datasize= len(self.datalist)
		self.labels= [item["label"] for item in self.datalist]
		self.snames= [item["sname"] for item in self.datalist]
		self.classids= 	[item["id"] for item in self.datalist]

		logger.info("#%d objects in dataset" % self.datasize)

		# - Loop over data and extract params per each source
		logger.info("Loop over data and extract params per each source ...")
		for i in range(self.datasize):
			if self.__process_source(i, img_group_1, img_group_2)<0:
				logger.warn("Failed to process source %d, skip to next ..." % (i))
				continue 
			
		# - Save data
		if self.save:
			logger.info("Saving data to file %s ..." % (self.outfile))
			self.__save_data()

		return 0

	
	#===========================
	#==    PROCESS SOURCE
	#===========================
	def __process_source(self, index, img_group_1, img_group_2):
		""" Process source and compute spectral index data """

		# - Check index
		if index<0 or index>=self.datasize:
			logger.error("Invalid index %d given!" % (index))
			return -1

		# - Process source
		d= self.datalist[index]
		sih= SpectralIndexTTHelper(d)
		sih.negative_pix_fract_thr= self.negative_pix_fract_thr
		sih.bad_pix_fract_thr= self.bad_pix_fract_thr
		sih.rcoeff_thr= self.alpha_rcoeff_thr
		sih.img_freqs= self.img_freqs

		if sih.run(img_group_1, img_group_2)<0:
			logger.warn("Failed to compute spectral index for source %d ..." % (index))
			return -1

		# - Append out dict to list if a good index was estimated
		has_good_alpha= sih.has_good_alpha
		par_dict= sih.param_dict
		
		if par_dict is None or not par_dict:
			logger.warn("Feature dict for source data %d is empty or None, skip it ..." % (index))
			return -1

		if has_good_alpha:
			self.par_dict_list.append(par_dict)
		else:
			logger.warn("Spectral index computed for source %d is not reliable, skip it ..." % (index))

		return 0

	#============================
	#==     READ DATA LIST
	#============================
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
		datadict= ret[0]
		nchannels_set= ret[1]
		
		self.datalist= datadict["data"]
		self.nchannels= list(nchannels_set)[0]
		self.datasize= len(self.datalist)
		self.labels= [item["label"] for item in self.datalist]
		self.snames= [item["sname"] for item in self.datalist]
		self.classids= 	[item["id"] for item in self.datalist]

		self.classfract_map= dict(Counter(self.classids).items())

		logger.info("#%d objects in dataset" % self.datasize)


		return 0

	#===========================
	#==    SAVE DATA
	#===========================
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
			return -1

		return 0



