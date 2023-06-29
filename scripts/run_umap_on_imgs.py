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
from sclassifier.preprocessing import BkgSubtractor, SigmaClipper, SigmaClipShifter, Scaler, LogStretcher, Augmenter
from sclassifier.preprocessing import Resizer, MinMaxNormalizer, AbsMinMaxNormalizer, MaxScaler, AbsMaxScaler, ChanMaxScaler
from sclassifier.preprocessing import Shifter, Standardizer, ChanDivider, MaskShrinker, BorderMasker
from sclassifier.preprocessing import ChanResizer, ZScaleTransformer, Chan3Trasformer
from sclassifier.preprocessing import PercentileThresholder, HistEqualizer
from sclassifier.preprocessing import BBoxResizer
from sclassifier.feature_extractor_umap import FeatExtractorUMAP
from sclassifier.clustering import Clusterer


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

	parser.add_argument('--clip_shift_data', dest='clip_shift_data', action='store_true',help='Do sigma clipp shifting')	
	parser.set_defaults(clip_shift_data=False)
	parser.add_argument('-sigma_clip', '--sigma_clip', dest='sigma_clip', required=False, type=float, default=1, action='store',help='Sigma threshold to be used for clip & shifting pixels (default=1)')
	parser.add_argument('--clip_data', dest='clip_data', action='store_true',help='Do sigma clipping')	
	parser.set_defaults(clip_data=False)
	parser.add_argument('-sigma_clip_low', '--sigma_clip_low', dest='sigma_clip_low', required=False, type=float, default=10, action='store',help='Lower sigma threshold to be used for clipping pixels below (mean-sigma_low*stddev) (default=10)')
	parser.add_argument('-sigma_clip_up', '--sigma_clip_up', dest='sigma_clip_up', required=False, type=float, default=10, action='store',help='Upper sigma threshold to be used for clipping pixels above (mean+sigma_up*stddev) (default=10)')	
	parser.add_argument('-clip_chid', '--clip_chid', dest='clip_chid', required=False, type=int, default=-1, action='store',help='Channel to clip data (-1=all) (default=-1)')

	parser.add_argument('--mask_borders', dest='mask_borders', action='store_true',help='Mask image borders by desired width/height fraction')
	parser.set_defaults(mask_borders=False)
	parser.add_argument('-mask_border_fract', '--mask_border_fract', dest='mask_border_fract', required=False, type=float, default=0.7, action='store',help='Size of non-masked box dimensions with respect to image size (default=0.7)')

	parser.add_argument('--resize_chans', dest='resize_chans', action='store_true',help='Resize channels to desired number specified in nchan_resize')	
	parser.set_defaults(resize_chans=False)
	parser.add_argument('-nchan_resize', '--nchan_resize', dest='nchan_resize', required=False, type=int, default=3, action='store',help='Desired number of channels for resizing (default=3)')

	parser.add_argument('--zscale_stretch', dest='zscale_stretch', action='store_true',help='Do zscale transform')	
	parser.set_defaults(zscale_stretch=False)
	parser.add_argument('--zscale_contrasts', dest='zscale_contrasts', required=False, type=str, default='0.25,0.25,0.25',help='zscale contrasts applied to all channels') 
	
	parser.add_argument('--chan3_preproc', dest='chan3_preproc', action='store_true',help='Use the 3 channel pre-processor')	
	parser.set_defaults(chan3_preproc=False)
	parser.add_argument('-sigma_clip_baseline', '--sigma_clip_baseline', dest='sigma_clip_baseline', required=False, type=float, default=0, action='store',help='Lower sigma threshold to be used for clipping pixels below (mean-sigma_low*stddev) in first channel of 3-channel preprocessing (default=0)')
	
	parser.add_argument('--apply_percentile_thr', dest='apply_percentile_thr', action='store_true',help='Apply percentile threshold to input image')	
	parser.set_defaults(apply_percentile_thr=False)
	parser.add_argument('-percentile_thr', '--percentile_thr', dest='percentile_thr', required=False, type=float, default=50, action='store',help='Percentile threshold (default=50)')

	parser.add_argument('--apply_histeq', dest='apply_histeq', action='store_true',help='Apply histogram equalization to input image')	
	parser.set_defaults(apply_histeq=False)
	
	parser.add_argument('--shrink_to_mask_bbox', dest='shrink_to_mask_bbox', action='store_true',help='Extract mask from image (where data!=0) and shrink image to mask bbox shape, eventually resizing to desired resize_size')
	parser.set_defaults(shrink_to_mask_bbox=False)

	parser.add_argument('--draw', dest='draw', action='store_true',help='Draw images')	
	parser.set_defaults(draw=False)
	
	# - Pre-processing options for UMAP/PCA
	parser.add_argument('--normalize', dest='normalize', action='store_true',help='Normalize feature data in range [0,1] before applying models (default=false)')	
	parser.set_defaults(normalize=False)
	parser.add_argument('-norm_transf', '--norm_transf', dest='norm_transf', required=False, type=str, default='minmax', action='store',help='Normalization transf to be applied in PCA: {"minmax","robust"} (default=minmax)')
	parser.add_argument('-scalerfile', '--scalerfile', dest='scalerfile', required=False, type=str, default='', action='store',help='Load and use data transform for UMAP stored in this file (.sav)')
	parser.add_argument('-scalerfile_pca', '--scalerfile_pca', dest='scalerfile_pca', required=False, type=str, default='', action='store',help='Load and use data transform for PCA stored in this file (.sav)')
	
	# - PCA options
	parser.add_argument('--run_pca', dest='run_pca', action='store_true',help='Run PCA before UMAP')	
	parser.set_defaults(run_pca=False)
	parser.add_argument('-modelfile_pca', '--modelfile_pca', dest='modelfile_pca', required=False, type=str, default='', action='store',help='PCA model filename (.h5)')
	parser.add_argument('-pca_ncomps', '--pca_ncomps', dest='pca_ncomps', required=False, type=int, default=-1, action='store',help='Number of PCA components to be used (-1=retain all cumulating a variance above threshold) (default=-1)')
	parser.add_argument('-pca_varthr', '--pca_varthr', dest='pca_varthr', required=False, type=float, default=0.9, action='store',help='Cumulative variance threshold used to retain PCA components (default=0.9)')

	# - UMAP classifier options
	parser.add_argument('-latentdim_umap', '--latentdim_umap', dest='latentdim_umap', required=False, type=int, default=2, action='store',help='Encoded data dim in UMAP (default=2)')
	parser.add_argument('-mindist_umap', '--mindist_umap', dest='mindist_umap', required=False, type=float, default=0.1, action='store',help='Min dist UMAP par (default=0.1)')
	parser.add_argument('-nneighbors_umap', '--nneighbors_umap', dest='nneighbors_umap', required=False, type=int, default=15, action='store',help='N neighbors UMAP par (default=15)')
	parser.add_argument('-outfile_umap_unsupervised', '--outfile_umap_unsupervised', dest='outfile_umap_unsupervised', required=False, type=str, default='latent_data_umap_unsupervised.dat', action='store',help='Name of UMAP encoded data output file')
	parser.add_argument('-outfile_umap_supervised', '--outfile_umap_supervised', dest='outfile_umap_supervised', required=False, type=str, default='latent_data_umap_supervised.dat', action='store',help='Name of UMAP output file with encoded data produced using supervised method')
	parser.add_argument('-outfile_umap_preclassified', '--outfile_umap_preclassified', dest='outfile_umap_preclassified', required=False, type=str, default='latent_data_umap_preclass.dat', action='store',help='Name of UMAP output file with encoded data produced from pre-classified data')

	parser.add_argument('-modelfile_umap', '--modelfile_umap', dest='modelfile_umap', required=False, type=str, action='store',help='UMAP model filename (.h5)')

	parser.add_argument('--predict', dest='predict', action='store_true',help='Only predict data according to loaded UMAP model (default=false)')	
	parser.set_defaults(predict=False)
	
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
	clip_shift_data= args.clip_shift_data
	clip_data= args.clip_data
	sigma_clip= args.sigma_clip
	sigma_clip_low= args.sigma_clip_low
	sigma_clip_up= args.sigma_clip_up
	clip_chid= args.clip_chid
	augment= args.augment
	augmenter= args.augmenter
	shuffle= args.shuffle
	draw= args.draw
	
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

	resize_chans= args.resize_chans
	nchan_resize= args.nchan_resize

	zscale_stretch= args.zscale_stretch
	zscale_contrasts= [float(x) for x in args.zscale_contrasts.split(',')]
	
	chan3_preproc= args.chan3_preproc
	sigma_clip_baseline= args.sigma_clip_baseline
	
	apply_percentile_thr= args.apply_percentile_thr
	percentile_thr= args.percentile_thr
	
	apply_histeq= args.apply_histeq
	shrink_to_mask_bbox= args.shrink_to_mask_bbox
	
	# - Data pre-processing for PCA/UMAP
	normalize= args.normalize
	norm_transf= args.norm_transf
	scalerfile= args.scalerfile
	scalerfile_pca= args.scalerfile_pca
	
	# - UMAP options
	latentdim_umap= args.latentdim_umap
	mindist_umap= args.mindist_umap
	nneighbors_umap= args.nneighbors_umap
	outfile_umap_unsupervised= args.outfile_umap_unsupervised
	outfile_umap_supervised= args.outfile_umap_supervised
	outfile_umap_preclassified= args.outfile_umap_preclassified

	modelfile_umap= args.modelfile_umap
	predict= args.predict
	
	# - PCA model options
	run_pca= args.run_pca
	modelfile_pca= args.modelfile_pca
	pca_ncomps= args.pca_ncomps
	pca_varthr= args.pca_varthr
	
	#===============================
	#==  CREATE DATA PRE-PROCESSOR
	#===============================
	# - Pre-process stage order
	#   1) Channel resizer 
	#   2) Bkg sub
	#   3) Sigma clip/shift
	#   4) Scale
	#   5) Stretch (e.g. log transform, zscale)
	#   6) Mask ops (shrinker, border masking)
	#   7) Augmentation
	#   8) Resize
	#   9) min/max (abs) norm, standardize, mean shift
	preprocess_stages= []

	if resize_chans:
		preprocess_stages.append(ChanResizer(nchans=nchan_resize))

	if shrink_to_mask_bbox:
		preprocess_stages.append(BBoxResizer(resize=resize, resize_size=resize_size))
		
	if subtract_bkg:
		preprocess_stages.append(BkgSubtractor(sigma=sigma_bkg, use_mask_box=use_box_mask_in_bkg, mask_fract=bkg_box_mask_fract, chid=bkg_chid))

	if clip_shift_data:
		preprocess_stages.append(SigmaClipShifter(sigma=sigma_clip, chid=clip_chid))

	if clip_data:
		preprocess_stages.append(SigmaClipper(sigma_low=sigma_clip_low, sigma_up=sigma_clip_up, chid=clip_chid))

	if scale_to_abs_max:
		preprocess_stages.append(AbsMaxScaler(use_mask_box=use_box_mask_in_chan_max_scaler, mask_fract=chan_max_scaler_box_mask_fract))

	if scale_to_selch_max:
		preprocess_stages.append(ChanMaxScaler(chref=chref, use_mask_box=use_box_mask_in_chan_max_scaler, mask_fract=chan_max_scaler_box_mask_fract))

	if scale:
		preprocess_stages.append(Scaler(scale_factors))

	if log_transform:
		preprocess_stages.append(LogStretcher(chid=log_transform_chid, minmaxnorm=log_transform_minmaxnorm, data_norm_min=log_transform_normmin, data_norm_max=log_transform_normmax, clip_neg=log_transform_clipneg))

	if zscale_stretch:
		preprocess_stages.append(ZScaleTransformer(contrasts=zscale_contrasts))
		
	if apply_histeq:
		preprocess_stages.append(HistEqualizer(adaptive=False))
	
	if erode:
		preprocess_stages.append(MaskShrinker(kernel=erode_kernel))
	
	if chan3_preproc:
		preprocess_stages.append( Chan3Trasformer(sigma_clip_baseline=sigma_clip_baseline, sigma_clip_low=sigma_clip_low, sigma_clip_up=sigma_clip_up, zscale_contrast=zscale_contrasts[0]) )

	if apply_percentile_thr:
		preprocess_stages.append(PercentileThresholder(percthr=percentile_thr))

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
	img_counter= 0
	featdata_list= []
	class_ids= []
	snames= []
	
	while True:
		try:
			data, sdata= next(data_generator)
			img_counter+= 1

			sname= sdata.sname
			label= sdata.label
			classid= sdata.id

			logger.info("Reading image no. %d (name=%s, label=%s) ..." % (img_counter, sname, label))
			
			# - Check if data is None
			if data is None:
				logger.warn("Image %d (name=%s, label=%s) is None (hint: some pre-processing stage failed, see logs), skipping it ..." % (img_counter, sname, label))
				continue

			nchannels= data.shape[3]
			
			# - Check for NANs
			has_naninf= np.any(~np.isfinite(data))
			if has_naninf:
				logger.warn("Image %d (name=%s, label=%s) has some nan/inf, skip ..." % (img_counter, sname, label))
				continue

			# - Check if channels have elements all equal
			for i in range(nchannels):
				data_min= np.min(data[0,:,:,i])
				data_max= np.max(data[0,:,:,i])
				same_values= (data_min==data_max)
				if same_values:
					logger.error("Image %d chan %d (name=%s, label=%s) has all elements equal to %f, check!" % (img_counter, i+1, sname, label, data_min))
					return 1
					
			
			# - Draw data
			if draw:
				logger.info("Drawing data ...")
				fig = plt.figure(figsize=(20, 10))
				plot_ncols= nchannels
				if nchannels==3:
					plot_ncols= 4
					
				for i in range(nchannels):
					data_ch= data[0,:,:,i]
					data_masked= np.ma.masked_equal(data_ch, 0.0, copy=False)
					data_min= data_masked.min()
					data_max= data_masked.max()
					#data_ch[data_ch==0]= data_min

					#logger.info("Reading nchan %d ..." % i+1)
					plt.subplot(1, plot_ncols, i+1)
					im= plt.imshow(data_ch, origin='lower')
					cbar= plt.colorbar(im)			

				# - Draw RGB if 3 channels
				if nchannels==3:
					plt.subplot(1, plot_ncols, nchannels+1)
					im= plt.imshow(data[0,:,:,:], origin='lower')
					cbar= plt.colorbar(im)

				plt.tight_layout()
				plt.show()
				plt.close()
				
			# - Flatten data & add to list
			featdata= data.flatten()
			snames.append(sname)
			class_ids.append(classid)
			featdata_list.append(featdata)
			
			#print("featdata.shape")
			#print(featdata.shape)
			
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


	# - Set feature data from all images
	logger.info("Setting image data ...")
	featdata= np.array(featdata_list)
	print("featdata.shape")
	print(featdata.shape)
	

	#==============================
	#==   RUN PCA
	#==============================
	if run_pca:
		logger.info("Running PCA on input feature data ...")
		clust= Clusterer()
		clust.normalize= normalize
		clust.norm_transf= norm_transf
		clust.pca_ncomps= pca_ncomps
		clust.pca_varthr= pca_varthr
	
		status= 0
		if clust.run_pca(featdata, class_ids=class_ids, snames=snames, modelfile=modelfile_pca, scalerfile=scalerfile_pca)<0:
			logger.error("PCA run failed!")
			return 1
			
		# - Set featdata to PCA transformed data
		featdata= clust.pca_transf_data	
		print("featdata.shape (after PCA)")
		print(featdata.shape)

	#==============================
	#==   RUN UMAP
	#==============================
	# - Create feat extractor 
	umap_class= FeatExtractorUMAP()
	umap_class.normalize= normalize
	umap_class.set_encoded_data_unsupervised_outfile(outfile_umap_unsupervised)
	umap_class.set_encoded_data_supervised_outfile(outfile_umap_supervised)
	umap_class.set_encoded_data_preclassified_outfile(outfile_umap_preclassified)
	umap_class.set_encoded_data_dim(latentdim_umap)
	umap_class.set_min_dist(mindist_umap)
	umap_class.set_n_neighbors(nneighbors_umap)
	umap_class.draw= draw
	
	
	# - Run UMAP
	status= 0
	if predict:
		logger.info("Running UMAP classifier prediction using modelfile %s on input feature data ..." % (modelfile_umap))
		if umap_class.run_predict(featdata, class_ids=class_ids, snames=snames, modelfile=modelfile_umap, scalerfile=scalerfile)<0:
			logger.error("UMAP prediction failed!")
			return 1
	else:
		logger.info("Running UMAP classifier training on input feature data ...")
		if umap_class.run_train(featdata, class_ids=class_ids, snames=snames, scalerfile=scalerfile)<0:
			logger.error("UMAP training failed!")
			return 1
	

	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

	
