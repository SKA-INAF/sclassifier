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

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections

## IMAGE MODULES
import cv2
import imutils
from skimage.feature import peak_local_max
from skimage.measure import inertia_tensor_eigvals
from shapely.geometry import Polygon
from shapely.geometry import Point

## MODULES
from sclassifier import __version__, __date__
from sclassifier import logger
from sclassifier.data_loader import DataLoader
from sclassifier.utils import Utils
from sclassifier.data_generator import DataGenerator
from sclassifier.preprocessing import DataPreprocessor
from sclassifier.preprocessing import BkgSubtractor, SigmaClipper, Scaler, LogStretcher, Augmenter
from sclassifier.preprocessing import Resizer, MinMaxNormalizer, AbsMinMaxNormalizer, MaxScaler, AbsMaxScaler, ChanMaxScaler
from sclassifier.preprocessing import Shifter, Standardizer, ChanDivider, MaskShrinker, BorderMasker

import matplotlib.pyplot as plt

#### GET SCRIPT ARGS ####
def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

###########################
##     ARGS
###########################
def get_args():
	"""This function parses and return arguments passed in"""
	parser = argparse.ArgumentParser(description="Parse args.")

	# - Input options
	parser.add_argument('-datalist','--datalist', dest='datalist', required=True, type=str, help='Input data json filelist') 
	parser.add_argument('-nmax', '--nmax', dest='nmax', required=False, type=int, default=-1, action='store',help='Max number of images to be read (-1=all) (default=-1)')
	
	# - Data pre-processing options
	parser.add_argument('--resize', dest='resize', action='store_true',help='Resize images')	
	parser.set_defaults(resize=False)

	#parser.add_argument('-nx', '--nx', dest='nx', required=False, type=int, default=64, action='store',help='Image resize width in pixels (default=64)')
	#parser.add_argument('-nx', '--nx', dest='nx', required=False, type=int, default=64, action='store',help='Image resize width in pixels (default=64)')
	parser.add_argument('-resize_size', '--resize_size', dest='resize_size', required=False, type=int, default=64, action='store',help='Image resize in pixels (default=64)')	
	parser.add_argument('--downscale_with_antialiasing', dest='downscale_with_antialiasing', action='store_true', help='Use anti-aliasing when downsampling the image (default=no)')	
	parser.set_defaults(downscale_with_antialiasing=False)
	parser.add_argument('--upscale', dest='upscale', action='store_true', help='Upscale images to resize size when source size is smaller (default=no)')	
	parser.set_defaults(upscale=False)
	parser.add_argument('--set_pad_val_to_min', dest='set_pad_val_to_min', action='store_true', help='Set masked value in resized image to min, otherwise leave to masked values (default=no)')	
	parser.set_defaults(set_pad_val_to_min=False)

	parser.add_argument('--normalize_minmax', dest='normalize_minmax', action='store_true',help='Normalize each channel in range [0,1]')	
	parser.set_defaults(normalize_minmax=False)
	parser.add_argument('--normalize_absminmax', dest='normalize_absminmax', action='store_true',help='Normalize each channel in range using absolute min/max computed over all channels [0,1]')	
	parser.set_defaults(normalize_absminmax=False)

	parser.add_argument('--scale_to_abs_max', dest='scale_to_abs_max', action='store_true',help='Scale to global max across all channels')	
	parser.set_defaults(scale_to_abs_max=False)
	parser.add_argument('--scale_to_max', dest='scale_to_max', action='store_true',help='Scale to max not to min-max range')	
	parser.set_defaults(scale_to_max=False)
	parser.add_argument('--scale_to_selch_max', dest='scale_to_selch_max', action='store_true',help='Scale to selected channel max not to min-max range')	
	parser.set_defaults(scale_to_selch_max=False)
	parser.add_argument('--use_box_mask_in_chan_max_scaler', dest='use_box_mask_in_chan_max_scaler', action='store_true',help='Find chan max for scaling inside box mask')	
	parser.set_defaults(use_box_mask_in_chan_max_scaler=False)	
	parser.add_argument('-chan_max_scaler_box_mask_fract', '--chan_max_scaler_box_mask_fract', dest='chan_max_scaler_box_mask_fract', required=False, type=float, default=0.5, action='store',help='Size of mask box dimensions with respect to image size used in chan max scaler (default=0.5)')
	
	parser.add_argument('--log_transform', dest='log_transform', action='store_true',help='Apply log transform to images')	
	parser.set_defaults(log_transform=False)
	parser.add_argument('-log_transform_chid', '--log_transform_chid', dest='log_transform_chid', required=False, type=int, default=-1, action='store',help='Channel id to be excluded from log-transformed. -1=transform all (default=-1)')
	parser.add_argument('--log_transform_minmaxnorm', dest='log_transform_minmaxnorm', action='store_true',help='Apply min/max normalization after log transform to images')	
	parser.set_defaults(log_transform_minmaxnorm=False)
	parser.add_argument('-log_transform_normmin', '--log_transform_normmin', dest='log_transform_normmin', required=False, type=float, default=-6, action='store',help='Min data normalization value to be applied if log_transform_minmaxnorm is enabled (default=-6)')
	parser.add_argument('-log_transform_normmax', '--log_transform_normmax', dest='log_transform_normmax', required=False, type=float, default=6, action='store',help='Max data normalization value to be applied if log_transform_minmaxnorm is enabled (default=6)')
	parser.add_argument('--log_transform_clipneg', dest='log_transform_clipneg', action='store_true',help='Clip negative values to 0 after min/max norm')	
	parser.set_defaults(log_transform_clipneg=False)

	parser.add_argument('--scale', dest='scale', action='store_true',help='Apply scale factors to images')	
	parser.set_defaults(scale=False)
	parser.add_argument('-scale_factors', '--scale_factors', dest='scale_factors', required=False, type=str, default='', action='store',help='Image scale factors separated by commas (default=empty)')

	parser.add_argument('--standardize', dest='standardize', action='store_true',help='Apply standardization to images')	
	parser.set_defaults(standardize=False)
	parser.add_argument('--meanshift', dest='meanshift', action='store_true',help='Apply mean shift to images')	
	parser.set_defaults(meanshift=False)
	parser.add_argument('-img_means', '--img_means', dest='img_means', required=False, type=str, default='', action='store',help='Image means (separated by commas) to be used in standardization (default=empty)')
	parser.add_argument('-img_sigmas', '--img_sigmas', dest='img_sigmas', required=False, type=str, default='', action='store',help='Image sigmas (separated by commas) to be used in standardization (default=empty)')

	parser.add_argument('--chan_divide', dest='chan_divide', action='store_true',help='Apply channel division to images')	
	parser.set_defaults(chan_divide=False)
	parser.add_argument('-chref', '--chref', dest='chref', required=False, type=int, default=0, action='store',help='Image channel reference to be used in chan divide or scale to selch (default=0)')

	parser.add_argument('--erode', dest='erode', action='store_true',help='Apply erosion to image sourve mask')	
	parser.set_defaults(erode=False)	
	parser.add_argument('-erode_kernel', '--erode_kernel', dest='erode_kernel', required=False, type=int, default=5, action='store',help='Erosion kernel size in pixels (default=5)')	
	
	parser.add_argument('--augment', dest='augment', action='store_true',help='Augment images')	
	parser.set_defaults(augment=False)
	parser.add_argument('-augmenter', '--augmenter', dest='augmenter', required=False, type=str, default='cnn', action='store',help='Predefined augmenter to be used (default=cnn)')
	
	parser.add_argument('--shuffle', dest='shuffle', action='store_true',help='Shuffle images')	
	parser.set_defaults(shuffle=False)

	
	parser.add_argument('--subtract_bkg', dest='subtract_bkg', action='store_true',help='Subtract bkg from ref channel image')	
	parser.set_defaults(subtract_bkg=False)
	parser.add_argument('-sigma_bkg', '--sigma_bkg', dest='sigma_bkg', required=False, type=float, default=3, action='store',help='Sigma clip to be used in bkg calculation (default=3)')
	parser.add_argument('--use_box_mask_in_bkg', dest='use_box_mask_in_bkg', action='store_true',help='Compute bkg value in borders left from box mask')	
	parser.set_defaults(use_box_mask_in_bkg=False)	
	parser.add_argument('-bkg_box_mask_fract', '--bkg_box_mask_fract', dest='bkg_box_mask_fract', required=False, type=float, default=0.7, action='store',help='Size of mask box dimensions with respect to image size used in bkg calculation (default=0.7)')
	parser.add_argument('-bkg_chid', '--bkg_chid', dest='bkg_chid', required=False, type=int, default=-1, action='store',help='Channel to subtract background (-1=all) (default=-1)')

	parser.add_argument('--clip_data', dest='clip_data', action='store_true',help='Do sigma clipping')	
	parser.set_defaults(clip_data=False)
	parser.add_argument('-sigma_clip', '--sigma_clip', dest='sigma_clip', required=False, type=float, default=1, action='store',help='Sigma threshold to be used for clipping pixels (default=1)')
	parser.add_argument('-clip_chid', '--clip_chid', dest='clip_chid', required=False, type=int, default=-1, action='store',help='Channel to clip data (-1=all) (default=-1)')


	parser.add_argument('--mask_borders', dest='mask_borders', action='store_true',help='Mask image borders by desired width/height fraction')
	parser.set_defaults(mask_borders=False)
	parser.add_argument('-mask_border_fract', '--mask_border_fract', dest='mask_border_fract', required=False, type=float, default=0.7, action='store',help='Size of non-masked box dimensions with respect to image size (default=0.7)')

	parser.add_argument('--draw', dest='draw', action='store_true',help='Draw images')	
	parser.set_defaults(draw=False)

	parser.add_argument('--save_fits', dest='save_fits', action='store_true',help='Save images')	
	parser.set_defaults(save_fits=False)
	
	parser.add_argument('--dump_stats', dest='dump_stats', action='store_true',help='Dump image stats')	
	parser.set_defaults(dump_stats=False)

	parser.add_argument('--dump_sample_stats', dest='dump_sample_stats', action='store_true',help='Dump image stats over entire sample')	
	parser.set_defaults(dump_sample_stats=False)

	parser.add_argument('--dump_flags', dest='dump_flags', action='store_true',help='Dump image flags')	
	parser.set_defaults(dump_flags=False)

	parser.add_argument('--exit_on_fault', dest='exit_on_fault', action='store_true',help='Exit on fault')	
	parser.set_defaults(exit_on_fault=False)

	parser.add_argument('--skip_on_fault', dest='skip_on_fault', action='store_true',help='Skip to next source on fault')	
	parser.set_defaults(skip_on_fault=False)

	parser.add_argument('-fthr_zeros', '--fthr_zeros', dest='fthr_zeros', required=False, type=float, default=0.1, action='store',help='Max fraction of zeros above which channel is bad (default=0.1)')	
	
	
	args = parser.parse_args()	

	return args



##############
##   MAIN   ##
##############
def main():
	"""Main function"""

	#===========================
	#==   PARSE ARGS
	#===========================
	logger.info("Get script args ...")
	try:
		args= get_args()
	except Exception as ex:
		logger.error("Failed to get and parse options (err=%s)",str(ex))
		return 1

	# - Input filelist
	datalist= args.datalist
	nmax= args.nmax

	# - Data process options	
	#nx= args.nx
	#ny= args.ny
	normalize_minmax= args.normalize_minmax
	normalize_absminmax= args.normalize_absminmax
	scale_to_abs_max= args.scale_to_abs_max
	scale_to_max= args.scale_to_max
	scale_to_selch_max= args.scale_to_selch_max
	use_box_mask_in_chan_max_scaler= args.use_box_mask_in_chan_max_scaler
	chan_max_scaler_box_mask_fract= args.chan_max_scaler_box_mask_fract
	log_transform= args.log_transform
	log_transform_chid= args.log_transform_chid
	log_transform_minmaxnorm= args.log_transform_minmaxnorm
	log_transform_normmin= args.log_transform_normmin
	log_transform_normmax= args.log_transform_normmax
	log_transform_clipneg= args.log_transform_clipneg
	resize= args.resize
	resize_size= args.resize_size
	downscale_with_antialiasing= args.downscale_with_antialiasing
	upscale= args.upscale
	set_pad_val_to_min= args.set_pad_val_to_min
	subtract_bkg= args.subtract_bkg
	sigma_bkg= args.sigma_bkg
	use_box_mask_in_bkg= args.use_box_mask_in_bkg
	bkg_box_mask_fract= args.bkg_box_mask_fract
	bkg_chid= args.bkg_chid
	clip_data= args.clip_data
	sigma_clip= args.sigma_clip
	clip_chid= args.clip_chid
	augment= args.augment
	augmenter= args.augmenter
	shuffle= args.shuffle
	draw= args.draw
	dump_stats= args.dump_stats
	dump_sample_stats= args.dump_sample_stats
	dump_flags= args.dump_flags
	scale= args.scale
	scale_factors= []
	if args.scale_factors!="":
		scale_factors= [float(x.strip()) for x in args.scale_factors.split(',')]
	standardize= args.standardize
	meanshift= args.meanshift
	img_means= []
	img_sigmas= []
	if args.img_means!="":
		img_means= [float(x.strip()) for x in args.img_means.split(',')]
	if args.img_sigmas!="":
		img_sigmas= [float(x.strip()) for x in args.img_sigmas.split(',')]

	chan_divide= args.chan_divide
	chref= args.chref
	erode= args.erode	
	erode_kernel= args.erode_kernel

	mask_borders= args.mask_borders
	mask_border_fract= args.mask_border_fract

	outfile_stats= "stats_info.dat"
	outfile_flags= "stats_flags.dat"
	outfile_sample_stats= "stats_sample_info.dat"
	exit_on_fault= args.exit_on_fault
	skip_on_fault= args.skip_on_fault
	save_fits= args.save_fits
	fthr_zeros= args.fthr_zeros


	#===============================
	#==  CREATE DATA PRE-PROCESSOR
	#===============================
	# - Pre-process stage order
	#   1) Bkg sub
	#   2) Sigma clip
	#   3) Scale
	#   4) Stretch (e.g. log transform)
	#   5) Mask ops (shrinker, border masking)
	#   6) Augmentation
	#   7) Resize
	#   8) min/max (abs) norm, standardize, mean shift
	preprocess_stages= []

	if subtract_bkg:
		preprocess_stages.append(BkgSubtractor(sigma=sigma_bkg, use_mask_box=use_box_mask_in_bkg, mask_fract=bkg_box_mask_fract, chid=bkg_chid))

	if clip_data:
		preprocess_stages.append(SigmaClipper(sigma=sigma_clip, chid=clip_chid))

	if scale_to_abs_max:
		preprocess_stages.append(AbsMaxScaler())

	if scale_to_selch_max:
		preprocess_stages.append(ChanMaxScaler(chref=chref, use_mask_box=use_box_mask_in_chan_max_scaler, mask_fract=chan_max_scaler_box_mask_fract))

	if scale:
		preprocess_stages.append(Scaler(scale_factors))

	if log_transform:
		preprocess_stages.append(LogStretcher(chid=log_transform_chid, minmaxnorm=log_transform_minmaxnorm, data_norm_min=log_transform_normmin, log_transform_normmax=log_transform_normmax, clip_neg=log_transform_clipneg))
	
	if erode:
		preprocess_stages.append(MaskShrinker(kernel=erode_kernel))
	
	if augment:
		preprocess_stages.append(Augmenter(augmenter_choice=augmenter))

	if mask_borders:
		preprocess_stages.append(BorderMasker(mask_border_fract))

	if resize:
		preprocess_stages.append(Resizer(resize_size=resize_size, upscale=upscale, downscale_with_antialiasing=downscale_with_antialiasing, set_pad_val_to_min=set_pad_val_to_min))

	if normalize_minmax:
		preprocess_stages.append(MinMaxNormalizer())

	if normalize_absminmax:
		preprocess_stages.append(AbsMinMaxNormalizer())

	if scale_to_max:
		preprocess_stages.append(MaxScaler())

	#if scale_to_abs_max:
	#	preprocess_stages.append(AbsMaxScaler())

	if meanshift:
		preprocess_stages.append(Shifter(offsets=img_means))
	
	if standardize:
		preprocess_stages.append(Standardizer(means=img_means, sigmas=img_sigmas))
	
	if chan_divide:
		preprocess_stages.append(ChanDivider(chref=chref))

	print("== PRE-PROCESSING STAGES ==")
	print(preprocess_stages)

	dp= DataPreprocessor(preprocess_stages)


	#===============================
	#==  DATA GENERATOR
	#===============================
	dg= DataGenerator(filename=datalist, preprocessor=dp)

	# - Read datalist	
	logger.info("Reading datalist %s ..." % datalist)
	if dg.read_datalist()<0:
		logger.error("Failed to read input datalist!")
		return 1

	source_labels= dg.snames
	
	nsamples= len(source_labels)
	if nmax>0 and nmax<nsamples:
		nsamples= nmax

	logger.info("#%d samples to be read ..." % nsamples)

	# - Read data	
	logger.info("Running data generator ...")
	data_generator= dg.generate_data(
		batch_size=1, 
		shuffle=shuffle
	)	

	#===========================
	#==   READ DATA
	#===========================
	# - Create data loader
	#dl= DataLoader(filename=datalist)

	# - Read datalist	
	#logger.info("Reading datalist %s ..." % datalist)
	#if dl.read_datalist()<0:
	#	logger.error("Failed to read input datalist!")
	#	return 1

	#source_labels= dl.snames
	
	#nsamples= len(source_labels)
	#if nmax>0 and nmax<nsamples:
	#	nsamples= nmax

	#logger.info("#%d samples to be read ..." % nsamples)


	# - Read data	
	#logger.info("Running data loader ...")
	#data_generator= dl.data_generator(
	#	batch_size=1, 
	#	shuffle=shuffle,
	#	resize=resize, nx=nx, ny=ny, 	
	#	normalize=normalize, scale_to_abs_max=scale_to_abs_max, scale_to_max=scale_to_max,
	#	augment=augment,
	#	log_transform=log_transform,
	#	scale=scale, scale_factors=scale_factors,
	#	standardize=standardize, means=img_means, sigmas=img_sigmas,
	#	chan_divide=chan_divide, chan_mins=chan_mins,
	#	erode=erode, erode_kernel=erode_kernel,
	#	subtract_bkg_and_clip=subtract_bkg_and_clip,
	#	outdata_choice='sdata'
	#)	

	img_counter= 0
	img_stats_all= []
	img_flags_all= []
	pixel_values_per_channels= []
	
	while True:
		try:
			data, sdata= next(data_generator)
			img_counter+= 1

			sname= sdata.sname
			label= sdata.label
			classid= sdata.id

			logger.info("Reading image no. %d (name=%s, label=%s) ..." % (img_counter, sname, label))
			#print("data shape")
			#print(data.shape)

			# - Check if data is None
			if data is None:
				logger.warn("Image %d (name=%s, label=%s) is None (hint: some pre-processing stage failed, see logs), skipping it ..." % (img_counter, sname, label))
				continue

			nchannels= data.shape[3]
			
			# - Check for NANs
			has_naninf= np.any(~np.isfinite(data))
			if has_naninf:
				logger.warn("Image %d (name=%s, label=%s) has some nan/inf, check!" % (img_counter, sname, label))
				if exit_on_fault:
					return 1
				else:
					if skip_on_fault:
						break

			# - Check for fraction of zeros in radio mask
			cond= np.logical_and(data[0,:,:,0]!=0, np.isfinite(data[0,:,:,0]))
			for i in range(1,nchannels):
				data_2d= data[0,:,:,i]
				data_1d= data_2d[cond]
				n= data_1d.size
				n_zeros= np.count_nonzero(data_1d==0)
				f= n_zeros/n
				if n_zeros>0:
					logger.info("Image %d chan %d (name=%s, label=%s): n=%d, n_zeros=%d, f=%f" % (img_counter, i+1, sname, label, n, n_zeros, f))
				
				if f>=fthr_zeros:
					logger.warn("Image %d chan %d (name=%s, label=%s) has a zero fraction %f, check!" % (img_counter, i+1, sname, label, f))
					if skip_on_fault:
						break

			# - Check if channels have elements all equal
			for i in range(nchannels):
				data_min= np.min(data[0,:,:,i])
				data_max= np.max(data[0,:,:,i])
				same_values= (data_min==data_max)
				if same_values:
					logger.error("Image %d chan %d (name=%s, label=%s) has all elements equal to %f, check!" % (img_counter, i+1, sname, label, data_min))
					if exit_on_fault:
						return 1
					else:
						if skip_on_fault:
							break
			
			# - Check correct norm
			#correct_norm= True
			#data_min= np.min(data[0,:,:,:])
			#data_max= np.max(data[0,:,:,:])

			#if normalize_minmax or normalize_absminmax:				
			#	correct_norm= (data_min==0 and data_max==1)

			#if scale_to_max or scale_to_abs_max:
			#	correct_norm= (data_max==1)

			#if not correct_norm:
			#	logger.error("Image %d chan %d (name=%s, label=%s) has invalid norm (%f,%f), check!" % (img_counter, i+1, sname, label, data_min, data_max))
			#	if exit_on_fault:
			#		return 1
			#	else:
			#		if skip_on_fault:
			#			break

			# - Dump image flags
			if dump_flags:
				img_flags= [sname]

				for i in range(nchannels):
					##cond_i= np.logical_and(data[0,:,:,i]!=0, np.isfinite(data[0,:,:,i]))
					data_2d= data[0,:,:,i]
					data_1d= data_2d[cond] # pixel in radio mask
					n= data_1d.size
					n_bad= np.count_nonzero(np.logical_or(~np.isfinite(data_1d), data_1d==0))
					n_neg= np.count_nonzero(data_1d<0)
					f_bad= float(n_bad)/float(n)
					f_negative= float(n_neg)/float(n)
					data_min= np.nanmin(data_1d)
					data_max= np.nanmax(data_1d)
					same_values= int(data_min==data_max)

					
					img_flags.append(same_values)
					img_flags.append(f_bad)
					img_flags.append(f_negative)

				# - Compute peaks & aspect ratio of first channel
				kernsize= 7
				footprint = np.ones((kernsize, ) * data[0,:,:,i].ndim, dtype=bool)
				peaks= peak_local_max(np.copy(data[0,:,:,i]), footprint=footprint, min_distance=4, exclude_border=True)

				bmap= cond.astype(np.uint8)
				polygon= None
				try:
					contours= cv2.findContours(np.copy(bmap), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
					contours= imutils.grab_contours(contours)
					if len(contours)>0:
						contour= np.squeeze(contours[0])
						polygon = Polygon(contour)
				except Exception as e:
					logger.warn("Failed to compute mask contour (err=%s)!" % (str(e)))
			
				if polygon is None:
					peaks_sel= peaks
				else:
					peaks_sel= []
					for peak in peaks:
						point = Point(peak[1], peak[0])
						has_peak= polygon.contains(point)
						if has_peak:
							peaks_sel.append(peak)

				npeaks= len(peaks_sel)
	
				eigvals = inertia_tensor_eigvals(image=data_2d)
				aspect_ratio= eigvals[0]/eigvals[1]	
	
				img_flags.append(npeaks)
				img_flags.append(aspect_ratio)

				img_flags.append(classid)
				img_flags_all.append(img_flags)

			# - Dump image stats
			if dump_stats:
				img_stats= [sname]
				
				for i in range(nchannels):
					data_masked= np.ma.masked_equal(data[0,:,:,i], 0.0, copy=False)
					data_min= data_masked.min()
					data_max= data_masked.max()
					data_mean= data_masked.mean() 
					data_std= data_masked.std()
					
					img_stats.append(data_min)
					img_stats.append(data_max)
					img_stats.append(data_mean)
					img_stats.append(data_std)

				img_stats.append(classid)
				img_stats_all.append(img_stats)

			# - Dump sample image stats
			if dump_sample_stats:
				if not pixel_values_per_channels:
					pixel_values_per_channels= [[] for i in range(nchannels)]

				for i in range(nchannels):
					cond= np.logical_and(data[0,:,:,i]!=0, np.isfinite(data[0,:,:,i]))

					data_masked_1d= data[0,:,:,i][cond]
					data_masked_list= list(data_masked_1d)
					#data_masked= np.ma.masked_equal(data[0,:,:,i], 0.0, copy=False)
					#data_masked_list= data_masked[~data_masked.mask].tolist() # Extract non-masked values and put to list
					#print("type(data_masked_list)")
					#print(type(data_masked_list))
					#print(data_masked_list)

					if type(data_masked_list)!=list:
						logger.error("Collection of non-masked pixels in image %d chan %d (name=%s, label=%s) is not a list!" % (img_counter, i+1, sname, label))
						#print(type(data_masked_list))
						return 1
					else:
						for item in data_masked_list:
							item_type= type(item)
							if item_type!=float and item_type!=np.float and item_type!=np.float32:
								logger.error("Current pixel in collection of non-masked pixels in image %d chan %d (name=%s, label=%s) is not a float!" % (img_counter, i+1, sname, label))
								#print("item")
								#print(item)
								#print("item_type")
								#print(item_type)
								#print(data_masked_list)
								return 1

					if not data_masked_list:
						logger.error("Image %d chan %d (name=%s, label=%s) has non masked pixels!" % (img_counter, i+1, sname, label))
						if exit_on_fault:
							return 1
						else:
							if skip_on_fault:
								break
					pixel_values_per_channels[i].extend(data_masked_list)

			# - Draw data
			if draw:
				logger.info("Drawing data ...")
				fig = plt.figure(figsize=(20, 10))
				for i in range(nchannels):
					data_ch= data[0,:,:,i]
					data_masked= np.ma.masked_equal(data_ch, 0.0, copy=False)
					data_min= data_masked.min()
					data_max= data_masked.max()
					#data_ch[data_ch==0]= data_min

					#logger.info("Reading nchan %d ..." % i+1)
					plt.subplot(1, nchannels, i+1)
					im= plt.imshow(data_ch, origin='lower')
					cbar= plt.colorbar(im)			

				plt.tight_layout()
				plt.show()

			# - Dump fits
			if save_fits:
				logger.info("Writing FITS ...")
				for i in range(nchannels):
					outfile_fits= sname + '_id' + str(classid) + '_ch' + str(i+1) + '.fits'
					Utils.write_fits(data[0,:,:,i], outfile_fits)

			# - Stop generator
			if img_counter>=nsamples:
				logger.info("Sample size (%d) reached, stop generation..." % nsamples)
				break

		except (GeneratorExit, KeyboardInterrupt):
			logger.info("Stop loop (keyboard interrupt) ...")
			break
		except Exception as e:
			logger.warn("Stop loop (exception catched %s) ..." % str(e))
			break

	# - Dump img flags
	if dump_flags:
		logger.info("Dumping img flag info to file %s ..." % (outfile_flags))

		head= "# sname "

		for i in range(nchannels):
			ch= i+1
			s= 'equalPixValues_ch{i} badPixFract_ch{i} negativePixFract_ch{i} '.format(i=ch)
			head= head + s
		head= head + "npeaks_ch1 aspectRatio_ch1 "
		head= head + "id"
		logger.info("Flag file head: %s" % (head))
		
		# - Dump to file
		Utils.write_ascii(np.array(img_flags_all), outfile_flags, head)	


	# - Dump img stats
	if dump_stats:
		logger.info("Dumping img stats info to file %s ..." % (outfile_stats))

		head= "# sname "
		for i in range(nchannels):
			ch= i+1
			s= 'min_ch{i} max_ch{i} mean_ch{i} std_ch{i} '.format(i=ch)
			head= head + s
		head= head + "id"
		logger.info("Stats file head: %s" % (head))
		
		# - Dump to file
		Utils.write_ascii(np.array(img_stats_all), outfile_stats, head)	

	# - Dump sample pixel stats
	if dump_sample_stats:
		logger.info("Computing sample pixel stats ...")
		img_sample_stats= [[]]
		
		for i in range(len(pixel_values_per_channels)):
			#print("type(pixel_values_per_channels)")
			#print(type(pixel_values_per_channels))
			#print("type(pixel_values_per_channels[i])")
			#print(type(pixel_values_per_channels[i]))
			#print(pixel_values_per_channels[i])
			#print("len(pixel_values_per_channels[i])")
			#print(len(pixel_values_per_channels[i]))

			for j in range(len(pixel_values_per_channels[i])):
				item= pixel_values_per_channels[i][j]
				item_type= type(item)
				if item_type!=np.float32 and item_type!=np.float and item_type!=float:
					logger.error("Pixel no. %d not float (ch=%d)!" % (j+1, i+1))
					#print("item_type")
					#print(item_type)
					#print("item")
					#print(item)
					return 1
			data= np.array(pixel_values_per_channels[i], dtype=np.float)
			#print("type(data)")
			#print(type(data))
			data_min= data.min()
			data_max= data.max()
			data_mean= data.mean() 
			data_std= data.std()
			data_median= np.median(data)
			data_q3, data_q1= np.percentile(data, [75 ,25])
			data_iqr = data_q3 - data_q1

			img_sample_stats[0].append(data_min)
			img_sample_stats[0].append(data_max)
			img_sample_stats[0].append(data_mean)
			img_sample_stats[0].append(data_std)
			img_sample_stats[0].append(data_median)
			img_sample_stats[0].append(data_iqr)
			

		logger.info("Dumping pixel sample stats info to file %s ..." % (outfile_sample_stats))

		head= "# "
		for i in range(len(pixel_values_per_channels)):
			ch= i+1
			s= 'min_ch{i} max_ch{i} mean_ch{i} std_ch{i} median_ch{i} iqr_ch{i} '.format(i=ch)
			head= head + s
		logger.info("Sample stats file head: %s" % (head))
			
		Utils.write_ascii(np.array(img_sample_stats), outfile_sample_stats, head)	

	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

