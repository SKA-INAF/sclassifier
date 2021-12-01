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

##############################
##     GLOBAL VARS
##############################
from sclassifier_vae import logger

## SCI MODULES
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity
from scipy.stats import kurtosis, skew

## GRAPHICS MODULES
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

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
		""" Return a Classifer object """

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
		self.nmaximgs= -1
		self.data_generator= None
		self.normalize_img= False
		self.shuffle_data= False
		self.log_transform_img= False
		self.scale_img= False
		self.scale_img_factors= []

		# - SSIM options
		self.winsize= 3
		self.ssim_thr= 0.
		
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

		# - Create standard generator (for reconstruction)
		self.data_generator= self.dl.data_generator(
			batch_size=1, 
			shuffle=self.shuffle_data,
			resize=True, nx=self.nx, ny=self.ny, 
			normalize=self.normalize_img, 
			augment=False,
			log_transform=self.log_transform_img,
			scale=self.scale_img, scale_factors=self.scale_img_factors,
			retsdata=True
		)
		
		return 0

		
	#####################################
	##     EXTRACT FEATURES
	#####################################
	def __extract_features(self, data, sdata, save_imgs=False):
		""" Extract image features """

		# - Retrieve some data fields
		nchans= data.shape[3]
		sname= sdata.sname
		label= sdata.label
		classid= sdata.id

		param_dict= collections.OrderedDict()
		param_dict["sname"]= sname
		param_dict["label"]= label
		param_dict["classid"]= classid

		# - Init save plot	
		if save_imgs:
			outfile_plot= sname + '_id' + str(classid) + '.png'		
			logger.info("Saving plot to file %s ..." % (outfile_plot))
			fig = plt.figure(figsize=(20, 10))

			plot_ncols= int(nchans*(nchans-1)/2)
			plot_nrows= 4

		# - Loop over images and compute pars
		index= 0

		for i in range(nchans):
			img_i= data[0,:,:,i]
			cond_i= np.logical_and(img_i!=0, np.isfinite(img_i))
			img_max_i= np.nanmax(img_i[cond_i])
			cond_col_i= np.logical_and(img_i>0, np.isfinite(img_i))

			# - Compute total flux
			S= np.nansum(img_i[cond_i])
			parname= "fluxSum_ch" + str(i+1)
			param_dict[parname]= S

			# - Compute SSIM and color indices
			for j in range(i+1,nchans):
				img_j= data[0,:,:,j]
				cond_j= np.logical_and(img_j!=0, np.isfinite(img_j))
				img_max_j= np.nanmax(img_j[cond_j])
				cond_col_j= np.logical_and(img_j>0, np.isfinite(img_j))

				cond= np.logical_and(cond_i,cond_j)
				img_1d_i= img_i[cond]
				img_1d_j= img_j[cond]
				cond_col_ij= np.logical_and(cond_col_i,cond_col_j)
				
				# - Compute SSIM
				#   NB: Need to normalize images to max otherwise the returned values are always ~1.
				#img_max= np.max([inputdata_img,recdata_img])
				ssim_mean, ssim_2d= structural_similarity(img_i/img_max_i, img_j/img_max_j, full=True, win_size=self.winsize, data_range=1)
				ssim_1d= ssim_2d[cond]

				if ssim_1d.size>0:
					ssim_mean_mask= np.nanmean(ssim_1d)
					ssim_min_mask= np.nanmin(ssim_1d)
					ssim_max_mask= np.nanmax(ssim_1d)
					ssim_std_mask= np.nanstd(ssim_1d)
				else:
					logger.warn("Image %s (chan=%d-%d): SSIM array is empty, setting estimators to -999..." % (sname, i+1, j+1))
					ssim_mean_mask= -999
					ssim_min_mask= -999
					ssim_max_mask= -999
					ssim_std_mask= -999
				
				if not np.isfinite(ssim_mean_mask):
					logger.warn("Image %s (chan=%d-%d): ssim_mean_mask is nan/inf!" % (sname, i+1, j+1))
					ssim_mean_mask= -999

				parname= "ssim_mean_ch{}_{}".format(i+1,j+1)
				param_dict[parname]= ssim_mean_mask
				parname= "ssim_min_ch{}_{}".format(i+1,j+1)
				param_dict[parname]= ssim_min_mask
				parname= "ssim_max_ch{}_{}".format(i+1,j+1)
				param_dict[parname]= ssim_max_mask
				parname= "ssim_std_ch{}_{}".format(i+1,j+1)
				param_dict[parname]= ssim_std_mask

				# - Compute flux ratios and moments
				cond_colors= np.logical_and(cond_col_ij, ssim_2d>self.ssim_thr)
				fluxratio_2d= np.divide(img_i, img_j, where=cond_colors, out=np.ones(img_i.shape)*np.nan)
				cond_fluxratio= np.isfinite(fluxratio_2d)
				fluxratio_1d= fluxratio_2d[cond_fluxratio]
				fluxratio_ssim_1d= ssim_2d[cond_fluxratio]

				if fluxratio_1d.size>0:
					fluxratio_mean= np.nanmean(fluxratio_1d)
					fluxratio_std= np.std(fluxratio_1d)
					fluxratio_skew= skew(fluxratio_1d)
					fluxratio_kurt= kurtosis(fluxratio_1d)	
					fluxratio_min= np.nanmin(fluxratio_1d)
					fluxratio_max= np.nanmax(fluxratio_1d)
				else:
					logger.warn("Image %s (chan=%d-%d): flux ratio array is empty, setting estimators to -999..." % (sname, i+1, j+1))
					fluxratio_mean= -999
					fluxratio_std= -999
					fluxratio_skew= -999
					fluxratio_kurt= -999
					fluxratio_min= -999
					fluxratio_max= -999

				if fluxratio_1d.size>0 and fluxratio_ssim_1d.size>0: 
					fluxratio_weighted_mean= Utils.weighted_mean(fluxratio_1d, fluxratio_ssim_1d)
					fluxratio_weighted_std= Utils.weighted_std(fluxratio_1d, fluxratio_ssim_1d)
					fluxratio_weighted_skew= Utils.weighted_skew(fluxratio_1d, fluxratio_ssim_1d)
					fluxratio_weighted_kurt= Utils.weighted_kurtosis(fluxratio_1d, fluxratio_ssim_1d)
				else:
					logger.warn("Image %s (chan=%d-%d): flux ratio or weights array are empty, setting weighted estimators to -999..." % (sname, i+1, j+1))
					fluxratio_weighted_mean= -999
					fluxratio_weighted_std= -999
					fluxratio_weighted_skew= -999
					fluxratio_weighted_kurt= -999

				parname= "fratio_mean_ch{}_{}".format(i+1,j+1)
				param_dict[parname]= fluxratio_mean
				parname= "fratio_std_ch{}_{}".format(i+1,j+1)
				param_dict[parname]= fluxratio_std
				parname= "fratio_skew_ch{}_{}".format(i+1,j+1)
				param_dict[parname]= fluxratio_skew
				parname= "fratio_kurt_ch{}_{}".format(i+1,j+1)
				param_dict[parname]= fluxratio_kurt
				parname= "fratio_min_ch{}_{}".format(i+1,j+1)
				param_dict[parname]= fluxratio_min	
				parname= "fratio_max_ch{}_{}".format(i+1,j+1)
				param_dict[parname]= fluxratio_max

				parname= "fratio_wmean_ch{}_{}".format(i+1,j+1)
				param_dict[parname]= fluxratio_weighted_mean
				parname= "fratio_wstd_ch{}_{}".format(i+1,j+1)
				param_dict[parname]= fluxratio_weighted_std
				parname= "fratio_wskew_ch{}_{}".format(i+1,j+1)
				param_dict[parname]= fluxratio_weighted_skew
				parname= "fratio_wkurt_ch{}_{}".format(i+1,j+1)
				param_dict[parname]= fluxratio_weighted_kurt

				# - Compute color index
				#colorind_2d= 2.5*np.log10(fluxratio_2d)

				# - Save images
				if save_imgs:
					
					# - Save ssim map
					plot_index= 3*index + 1
					plt.subplot(plot_nrows, plot_ncols, plot_index)
					plt.imshow(ssim_2d, origin='lower')
					plt.colorbar()

					# - Save flux ratio map
					plot_index= 3*index + 2
					plt.subplot(plot_nrows, plot_ncols, plot_index)
					plt.imshow(fluxratio_2d, origin='lower')
					plt.colorbar()

					# - Save flux ratio histogram
					plot_index= 3*index + 3
					plt.subplot(plot_nrows, plot_ncols, plot_index)
					plt.hist(fluxratio_1d, bins='auto')

					# - Save weighted flux ratio histogram
					plot_index= 3*index + 4
					plt.subplot(plot_nrows, plot_ncols, plot_index)
					plt.hist(fluxratio_1d, weights=fluxratio_ssim_1d, bins=20)

					plt.savefig(outfile_plot)
					#plt.tight_layout()
					#plt.show()
					plt.close()
	
				index+= 1	


		return param_dict

	#####################################
	##     RUN
	#####################################
	def run(self):
		""" Extract features """

		#===========================
		#==   SET DATA
		#===========================	
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
		
		while True:
			try:
				# - Get data from generator
				data, sdata= next(self.data_generator)
				img_counter+= 1

				nchans= data.shape[3]
				sname= sdata.sname
				label= sdata.label
				classid= sdata.id

				# - Extracting features
				logger.info("Extracting features from image sample no. %d (name=%s, id=%d) ..." % (img_counter, sname, classid))
				par_dict= self.__extract_features(data, sdata, save_imgs=self.save_imgs)
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
				break

		
		#===========================
		#==   SAVE PARAM FILE
		#===========================
		parnames = par_dict_list[0].keys()
		print("parnames")
		print(parnames)
		
		#with open(self.outfile, 'wb') as fp:
		with open(self.outfile, 'w') as fp:
			fp.write("# ")
			dict_writer = csv.DictWriter(fp, parnames)
			dict_writer.writeheader()
			dict_writer.writerows(spar_dict)

		return 0

	
