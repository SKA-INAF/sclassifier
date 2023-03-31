#!/usr/bin/env python

from __future__ import print_function

##################################################
###    SET SEED FOR REPRODUCIBILITY (DEBUG)
##################################################
#from numpy.random import seed
#seed(1)
#import tensorflow
#tensorflow.random.set_seed(2)

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
import json

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections

## TENSORFLOW
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model


# DISPLAY
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
	parser.add_argument('-inputfile','--inputfile', dest='inputfile', required=False, type=str, help='Input data FITS file') 
	
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


	# - Model
	parser.add_argument('-weightfile', '--weightfile', dest='weightfile', required=False, type=str, default="", action='store',help='Weight file (hd5) to be loaded (default=no)')	
	parser.add_argument('-modelfile', '--modelfile', dest='modelfile', required=True, type=str, default="", action='store',help='Model architecture file (json) to be loaded (default=no)')
	parser.add_argument('--last_conv_layer_name', dest='last_conv_layer_name', required=False, type=str, default='layer4.1.relu2',help='Last conv layer size') 

	# - Override class target configuration
	parser.add_argument('--classid_remap', dest='classid_remap', required=False, type=str, default='', help='Class ID remap dictionary')
	parser.add_argument('--target_label_map', dest='target_label_map', required=False, type=str, default='', help='Target label dictionary')	
	parser.add_argument('--classid_label_map', dest='classid_label_map', required=False, type=str, default='', help='Class ID label dictionary')
	parser.add_argument('--target_names', dest='target_names', required=False, type=str, default='', help='Target names')
	parser.add_argument('-nclasses', '--nclasses', dest='nclasses', required=False, type=int, default=0, action='store', help='Number of classes (default=-1)')

	# - resnet18: layer4.1.relu2
	# - resnet50/resnet101: conv5_block3_out (or avg_pool?)
	# - custom: max_pooling2d_N  (es. max_pooling2d_3)

	# - Save
	parser.add_argument('--save', dest='save', action='store_true', help='Save heatmap superimposed over image')	
	parser.set_defaults(save=False)

	args = parser.parse_args()	

	return args


###########################
##     GRADCAM HEATMAP
###########################
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
	""" Create gradcam headmap """    

	# First, we create a model that maps the input image to the activations
	# of the last conv layer as well as the output predictions
	logger.info("Create grad_model ...")
	grad_model = tf.keras.models.Model(
		[model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
	)

	# Then, we compute the gradient of the top predicted class for our input image
	# with respect to the activations of the last conv layer
	logger.info("Compute the gradient ...")
	with tf.GradientTape() as tape:
		last_conv_layer_output, preds = grad_model(img_array)
		if pred_index is None:
			pred_index = tf.argmax(preds[0])
		class_channel = preds[:, pred_index]

	# This is the gradient of the output neuron (top predicted or chosen)
	# with regard to the output feature map of the last conv layer
	logger.info("Compute the gradient of the output neuron ...")
	grads = tape.gradient(class_channel, last_conv_layer_output)

	# This is a vector where each entry is the mean intensity of the gradient
	# over a specific feature map channel
	pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

	# We multiply each channel in the feature map array
	# by "how important this channel is" with regard to the top predicted class
	# then sum all the channels to obtain the heatmap class activation
	last_conv_layer_output = last_conv_layer_output[0]
	heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
	heatmap = tf.squeeze(heatmap)

	# For visualization purpose, we will also normalize the heatmap between 0 & 1
	heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
	return heatmap.numpy()


def load_model_from_file(modelfile, weightfile="", custom_objects=None):
	""" Load model & weights """

	# - Load model
	logger.info("Loading model from file %s ..." % (modelfile))
	try:
		model = load_model(modelfile, custom_objects=custom_objects, compile=False)
			
	except Exception as e:
		logger.warn("Failed to load model from file %s (err=%s)!" % (modelfile, str(e)))
		return None

	# - Load weights
	if weightfile!="":
		logger.info("Loading model weights from file %s ..." % (weightfile))
		try:
			model.load_weights(weightfile)
		except Exception as e:
			logger.warn("Failed to load weights from file %s (err=%s)!" % (weightfile, str(e)))
			return None

	return model



def save_and_display_gradcam(img, heatmap, alpha=0.4, draw=True, save=False, outfilename="cam.jpg"):
	""" Save and display heatmap superimposed over """

	# - Load the original image
	#img = keras.preprocessing.image.load_img(img_path)
	#img = keras.preprocessing.image.img_to_array(img)

	# - Rescale heatmap to a range 0-255
	heatmap = np.uint8(255 * heatmap)

	# - Use jet colormap to colorize heatmap
	jet = cm.get_cmap("jet")

	# - Use RGB values of the colormap
	jet_colors = jet(np.arange(256))[:, :3]
	jet_heatmap = jet_colors[heatmap]

	# - Create an image with RGB colorized heatmap
	jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
	jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
	jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

	# - Superimpose the heatmap on original image
	superimposed_img = jet_heatmap * alpha + img
	superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

	# - Save the superimposed image
	if save:
		superimposed_img.save(outfilename)

	# - Display Grad CAM
	if draw:
		display(Image(outfilename))


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
	inputfile= args.inputfile

	# - Model options
	modelfile= args.modelfile
	weightfile= args.weightfile
	last_conv_layer_name= args.last_conv_layer_name

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
	clip_shift_data= args.clip_shift_data
	clip_data= args.clip_data
	sigma_clip= args.sigma_clip
	sigma_clip_low= args.sigma_clip_low
	sigma_clip_up= args.sigma_clip_up
	clip_chid= args.clip_chid
	augment= args.augment
	augmenter= args.augmenter
	shuffle= args.shuffle
	
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

	# - Override target/class id configuration
	classid_remap= {}
	if args.classid_remap!="":
		try:
			classid_remap= ast.literal_eval(args.classid_remap)
		except Exception as e:
			logger.error("Failed to convert class id remap string to dict (err=%s)!" % (str(e)))
			return -1	
		
		print("== classid_remap ==")
		print(classid_remap)

	target_label_map= {}
	if args.target_label_map!="":
		try:
			target_label_map= ast.literal_eval(args.target_label_map)
		except Exception as e:
			logger.error("Failed to convert target label map string to dict (err=%s)!" % (str(e)))
			return -1	
		
		print("== target_label_map ==")
		print(target_label_map)

	classid_label_map= {}
	if args.classid_label_map!="":
		try:
			classid_label_map= ast.literal_eval(args.classid_label_map)
		except Exception as e:
			logger.error("Failed to convert classid label map string to dict (err=%s)!" % (str(e)))
			return -1	
		
		print("== classid_label_map ==")
		print(classid_label_map)


	target_names= []
	if args.target_names:
		target_names= [str(x) for x in args.target_names.split(',')]

	nclasses= args.nclasses

	# - Save options
	save= args.save

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
	
	if erode:
		preprocess_stages.append(MaskShrinker(kernel=erode_kernel))
	
	if chan3_preproc:
		preprocess_stages.append( Chan3Trasformer(sigma_clip_baseline=sigma_clip_baseline, sigma_clip_low=sigma_clip_low, sigma_clip_up=sigma_clip_up, zscale_contrast=zscale_contrasts[0]) )

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
	
	logger.info("#%d samples to be read ..." % nsamples)

	# - Read data	
	logger.info("Running data generator ...")
	data_generator= dg.generate_cnn_data(
		batch_size=1,
		shuffle=False,
		classtarget_map=classid_remap, nclasses=nclasses
	)

	#===============================
	#==  LOAD MODEL
	#===============================
	model= load_model_from_file(modelfile, weightfile)
	if model is None:
		logger.error("Failed to load model from file %s!" % (modelfile))
		return -1

	# Remove last layer's softmax
	#model.layers[-1].activation = None

	#===========================
	#==   READ DATA
	#===========================
	img_counter= 0
	
	
	while True:
		try:
			# - Read data from generator
			logger.info("Reading data from generator ...")
			#data, sdata= next(data_generator)
			data, _= next(data_generator)
			img_counter+= 1

			#sname= sdata.sname
			#label= sdata.label
			#classid= sdata.id

			##logger.info("Reading image no. %d (name=%s, label=%s) ..." % (img_counter, sname, label))
			logger.info("Reading image no. %d ..." % (img_counter))

			# - Check if data is None
			if data is None:
				#logger.warn("Image %d (name=%s, label=%s) is None (hint: some pre-processing stage failed, see logs), skipping it ..." % (img_counter, sname, label))
				logger.warn("Image %d is None (hint: some pre-processing stage failed, see logs), skipping it ..." % (img_counter))
				continue

			nchannels= data.shape[3]

			# - Predict data
			logger.info("Predicting data ...")
			predout= model.predict(
				x=data,
				batch_size=1,
    		verbose=2,
    		workers=1,
    		use_multiprocessing=False
			)

			# - Generate the heatmap
			heatmap = make_gradcam_heatmap(data, model, last_conv_layer_name)

			# - Display heatmap
			outfilename= sname + "_gradcam.jpg"
			save_and_display_gradcam(data, heatmap, alpha=0.4, draw=True, save=save, outfilename=outfilename)

		except (GeneratorExit, KeyboardInterrupt):
			logger.info("Stop loop (keyboard interrupt) ...")
			break
		except Exception as e:
			logger.warn("Stop loop (exception catched %s) ..." % str(e))
			break

	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())
