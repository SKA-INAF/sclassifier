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

## MODULES
from sclassifier import __version__, __date__
from sclassifier import logger
from sclassifier.data_loader import DataLoader
from sclassifier.feature_extractor_ae import FeatExtractorAE
from sclassifier.feature_extractor_umap import FeatExtractorUMAP
from sclassifier.clustering import Clusterer
from sclassifier.data_generator import DataGenerator
from sclassifier.preprocessing import DataPreprocessor
from sclassifier.preprocessing import BkgSubtractor, SigmaClipper, SigmaClipShifter, Scaler, LogStretcher, Augmenter
from sclassifier.preprocessing import Resizer, MinMaxNormalizer, AbsMinMaxNormalizer, MaxScaler, AbsMaxScaler, ChanMaxScaler
from sclassifier.preprocessing import Shifter, Standardizer, ChanDivider, MaskShrinker, BorderMasker
from sclassifier.preprocessing import ChanResizer, ZScaleTransformer

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
	parser.add_argument('-datalist_cv','--datalist_cv', dest='datalist_cv', required=False, default="", type=str, help='Input data json filelist for validation') 
	
	# - Data pre-processing options
	parser.add_argument('--no-resize', dest='resize', action='store_false',help='Resize images')	
	parser.set_defaults(resize=True)
	#parser.add_argument('-nx', '--nx', dest='nx', required=False, type=int, default=64, action='store',help='Image resize width in pixels (default=64)')
	#parser.add_argument('-ny', '--ny', dest='ny', required=False, type=int, default=64, action='store',help='Image resize height in pixels (default=64)')		
	parser.add_argument('-resize_size', '--resize_size', dest='resize_size', required=False, type=int, default=64, action='store',help='Image resize in pixels (default=64)')	
	parser.add_argument('--downscale_with_antialiasing', dest='downscale_with_antialiasing', action='store_true', help='Use anti-aliasing when downsampling the image (default=no)')	
	parser.set_defaults(downscale_with_antialiasing=False)
	parser.add_argument('--upscale', dest='upscale', action='store_true', help='Upscale images to resize size when source size is smaller (default=no)')	
	parser.set_defaults(upscale=False)

	parser.add_argument('--set_pad_val_to_min', dest='set_pad_val_to_min', action='store_true', help='Set masked value in resized image to min, otherwise leave to masked values (default=no)')	
	parser.set_defaults(set_pad_val_to_min=False)

	parser.add_argument('--augment', dest='augment', action='store_true',help='Augment images')	
	parser.set_defaults(augment=False)
	parser.add_argument('-augmenter', '--augmenter', dest='augmenter', required=False, type=str, default='cae', action='store',help='Predefined augmenter to be used (default=cnn)')
	parser.add_argument('-augment_scale_factor', '--augment_scale_factor', dest='augment_scale_factor', required=False, type=int, default=1, action='store',help='Number of times images are augmented. E.g. if 2, nsteps_per_epoch=2*nsamples/batch_size (default=1)')
	

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
	

	# - Network training options
	parser.add_argument('--predict', dest='predict', action='store_true',help='Predict model on input data (default=false)')	
	parser.set_defaults(predict=False)
	parser.add_argument('--reconstruct', dest='reconstruct', action='store_true',help='Reconstruct data using trained CAE model (default=false)')	
	parser.set_defaults(reconstruct=False)
	parser.add_argument('-weightfile_encoder', '--weightfile_encoder', dest='weightfile_encoder', required=False, type=str, default="", action='store',help='Encoder weights file (hd5) to be loaded (default=no)')	
	parser.add_argument('-weightfile_decoder', '--weightfile_decoder', dest='weightfile_decoder', required=False, type=str, default="", action='store',help='Decoder weights file (hd5) to be loaded (default=no)')	
	
	parser.add_argument('-latentdim', '--latentdim', dest='latentdim', required=False, type=int, default=2, action='store',help='Dimension of latent vector (default=2)')	
	parser.add_argument('-nepochs', '--nepochs', dest='nepochs', required=False, type=int, default=100, action='store',help='Number of epochs used in network training (default=100)')	
	parser.add_argument('-optimizer', '--optimizer', dest='optimizer', required=False, type=str, default='rmsprop', action='store',help='Optimizer used (default=rmsprop)')
	parser.add_argument('-learning_rate', '--learning_rate', dest='learning_rate', required=False, type=float, default=None, action='store',help='Learning rate. If None, use default for the selected optimizer (default=None)')
	parser.add_argument('-batch_size', '--batch_size', dest='batch_size', required=False, type=int, default=32, action='store',help='Batch size used in training (default=32)')
	parser.add_argument('-weight_seed', '--weight_seed', dest='weight_seed', required=False, type=int, default=None, action='store',help='Weight seed to set reproducible training (default=None)')
	parser.add_argument('--reproducible', dest='reproducible', action='store_true',help='Fix seed and make model reproducible from run to run')	
	parser.set_defaults(reproducible=False)
	parser.add_argument('-validation_steps', '--validation_steps', dest='validation_steps', required=False, type=int, default=10, action='store',help='Number of validation steps used in each epoch (default=10)')

	parser.add_argument('--no-multiprocessing', dest='multiprocessing', action='store_false',help='Disable multiprocessing in TF fit method (default=enabled)')	
	parser.set_defaults(multiprocessing=True)

	parser.add_argument('--load_cv_data_in_batches', dest='load_cv_data_in_batches', action='store_true',help='Load validation data in batches using train batch size (default=load all data in a single step)')	
	parser.set_defaults(load_cv_data_in_batches=False)

	parser.add_argument('--balance_classes_in_batch', dest='balance_classes_in_batch', action='store_true',help='Balance classes in batch generation')	
	parser.set_defaults(balance_classes_in_batch=False)
	##parser.add_argument('--class_probs', dest='class_probs', required=False, type=str, default='{3:1,6:1,23:1,24:1,1:1,2:1,6000:1}', help='Class weights used in batch rebalance') 
	parser.add_argument('--class_probs', dest='class_probs', required=False, type=str, default='{"PN":1,"HII":1,"PULSAR":1,"YSO":1,"STAR":1,"GALAXY":1,"QSO":1}', help='Class weights used in batch rebalance') 
	##parser.add_argument('--class_probs', dest='class_probs', required=False, type=str, default='', help='Class probs used in batch class resampling. If rand<prob accept generated data.') 
	
	parser.add_argument('--mse_loss', dest='mse_loss', action='store_true',help='Compute and include MSE reco loss in total loss')
	parser.add_argument('--no-mse_loss', dest='mse_loss', action='store_false',help='Skip MSE calculation and exclude MSE reco loss from total loss')
	parser.set_defaults(mse_loss=True)
	parser.add_argument('--scale_chan_mse_loss', dest='scale_chan_mse_loss', action='store_true',help='Scale MSE loss per channel by max(mean(ch))/mean(ch)')
	parser.set_defaults(scale_chan_mse_loss=False)
	
	parser.add_argument('--ssim_loss', dest='ssim_loss', action='store_true',help='Compute and include SSIM reco loss in total loss')
	parser.add_argument('--no-ssim_loss', dest='ssim_loss', action='store_false',help='Skip SSIM calculation and exclude SSIM reco loss from total loss')
	parser.set_defaults(ssim_loss=False)

	parser.add_argument('--kl_loss', dest='kl_loss', action='store_true',help='Compute and include KL reco loss in total loss (effective only for autoencoder model)')
	parser.add_argument('--no-kl_loss', dest='kl_loss', action='store_false',help='Skip KL calculation and exclude KL reco loss from total loss')
	parser.set_defaults(kl_loss=False)	

	parser.add_argument('-mse_loss_weight', '--mse_loss_weight', dest='mse_loss_weight', required=False, type=float, default=1.0, action='store',help='Reconstruction loss weight (default=1.0)')
	parser.add_argument('-kl_loss_weight', '--kl_loss_weight', dest='kl_loss_weight', required=False, type=float, default=1.0, action='store',help='KL loss weight (default=1.0)')
	parser.add_argument('-ssim_loss_weight', '--ssim_loss_weight', dest='ssim_loss_weight', required=False, type=float, default=1.0, action='store',help='SSIM loss weight (default=1.0)')
	parser.add_argument('-ssim_win_size', '--ssim_win_size', dest='ssim_win_size', required=False, type=int, default=3, action='store',help='SSIM filter window size in pixels (default=3)')

	# - Network architecture options
	parser.add_argument('--use_vae', dest='use_vae', action='store_true',help='Use variational autoencoders')	
	parser.set_defaults(use_vae=False)
	parser.add_argument('-modelfile_encoder', '--modelfile_encoder', dest='modelfile_encoder', required=False, type=str, default="", action='store',help='Model architecture file (json) to be loaded for encoder (default=no)')
	parser.add_argument('-modelfile_decoder', '--modelfile_decoder', dest='modelfile_decoder', required=False, type=str, default="", action='store',help='Model architecture file (json) to be loaded for decoder (default=no)')	

	parser.add_argument('--add_maxpooling_layer', dest='add_maxpooling_layer', action='store_true',help='Add max pooling layer after conv layers ')	
	parser.set_defaults(add_maxpooling_layer=False)	
	parser.add_argument('--add_batchnorm_layer', dest='add_batchnorm_layer', action='store_true',help='Add batch normalization layer after conv layers ')	
	parser.set_defaults(add_batchnorm_layer=False)	
	parser.add_argument('--add_leakyrelu', dest='add_leakyrelu', action='store_true',help='Add LeakyRELU after batch norm layers ')	
	parser.set_defaults(add_leakyrelu=False)
	parser.add_argument('--add_dense_layer', dest='add_dense_layer', action='store_true',help='Add dense layers in encoder after flattening layers ')	
	parser.set_defaults(add_dense_layer=False)
	parser.add_argument('--add_dropout_layer', dest='add_dropout_layer', action='store_true',help='Add dropout layers before dense layers')	
	parser.set_defaults(add_dropout_layer=False)
	parser.add_argument('-dropout_rate', '--dropout_rate', dest='dropout_rate', required=False, type=float, default=0.5, action='store',help='Dropout rate (default=0.5)')

	parser.add_argument('--add_conv_dropout_layer', dest='add_conv_dropout_layer', action='store_true',help='Add dropout layers after conv max pool layers')	
	parser.set_defaults(add_conv_dropout_layer=False)
	parser.add_argument('-conv_dropout_rate', '--conv_dropout_rate', dest='conv_dropout_rate', required=False, type=float, default=0.2, action='store',help='Dropout rate after conv layers (default=0.2)')

	parser.add_argument('--add_channorm_layer', dest='add_channorm_layer', action='store_true',help='Add norm layer before encoder input and denorm layer before decoder output')	
	parser.set_defaults(add_channorm_layer=False)

	parser.add_argument('-nfilters_cnn', '--nfilters_cnn', dest='nfilters_cnn', required=False, type=str, default='32,64,128', action='store',help='Number of convolution filters per each layer')
	parser.add_argument('-kernsizes_cnn', '--kernsizes_cnn', dest='kernsizes_cnn', required=False, type=str, default='3,5,7', action='store',help='Convolution filter kernel sizes per each layer')
	parser.add_argument('-strides_cnn', '--strides_cnn', dest='strides_cnn', required=False, type=str, default='2,2,2', action='store',help='Convolution strides per each layer')
	
	parser.add_argument('-dense_layer_sizes', '--dense_layer_sizes', dest='dense_layer_sizes', required=False, type=str, default='16', action='store',help='Dense layer sizes used (default=16)')
	parser.add_argument('-dense_layer_activation', '--dense_layer_activation', dest='dense_layer_activation', required=False, type=str, default='relu', action='store',help='Dense layer activation used {relu,softmax} (default=relu)')
	parser.add_argument('-decoder_output_layer_activation', '--decoder_output_layer_activation', dest='decoder_output_layer_activation', required=False, type=str, default='sigmoid', action='store',help='Output decoder layer activation used {sigmoid,softmax} (default=sigmoid)')

	# - Reco metrics & plot options
	parser.add_argument('-winsize', '--winsize', dest='winsize', required=False, type=int, default=3, action='store',help='Window size (odd) in pixels used to compute similarity index map (default=3)')	
	parser.add_argument('--save_plots', dest='save_plots', action='store_true',help='Save reco plots')	
	parser.set_defaults(save_plots=False)

	# - UMAP classifier options
	parser.add_argument('--run_umap', dest='run_umap', action='store_true',help='Run UMAP of autoencoder latent vector')	
	parser.set_defaults(run_umap=False)	
	parser.add_argument('-latentdim_umap', '--latentdim_umap', dest='latentdim_umap', required=False, type=int, default=2, action='store',help='Encoded data dim in UMAP (default=2)')
	parser.add_argument('-mindist_umap', '--mindist_umap', dest='mindist_umap', required=False, type=float, default=0.1, action='store',help='Min dist UMAP par (default=0.1)')
	parser.add_argument('-nneighbors_umap', '--nneighbors_umap', dest='nneighbors_umap', required=False, type=int, default=15, action='store',help='N neighbors UMAP par (default=15)')
	parser.add_argument('-outfile_umap_unsupervised', '--outfile_umap_unsupervised', dest='outfile_umap_unsupervised', required=False, type=str, default='latent_data_umap_unsupervised.dat', action='store',help='Name of UMAP encoded data output file')
	parser.add_argument('-outfile_umap_supervised', '--outfile_umap_supervised', dest='outfile_umap_supervised', required=False, type=str, default='latent_data_umap_supervised.dat', action='store',help='Name of UMAP output file with encoded data produced using supervised method')
	parser.add_argument('-outfile_umap_preclassified', '--outfile_umap_preclassified', dest='outfile_umap_preclassified', required=False, type=str, default='latent_data_umap_preclass.dat', action='store',help='Name of UMAP output file with encoded data produced from pre-classified data')

	# - Clustering options
	parser.add_argument('--run_clustering', dest='run_clustering', action='store_true',help='Run clustering on autoencoder latent vector')	
	parser.set_defaults(run_clustering=False)
	parser.add_argument('-min_cluster_size', '--min_cluster_size', dest='min_cluster_size', required=False, type=int, default=5, action='store',help='Minimum cluster size for HDBSCAN clustering (default=5)')
	parser.add_argument('-min_samples', '--min_samples', dest='min_samples', required=False, type=int, default=None, action='store',help='Minimum cluster sample parameter for HDBSCAN clustering. Typically equal to min_cluster_size (default=None')	
	parser.add_argument('-modelfile_clust', '--modelfile_clust', dest='modelfile_clust', required=False, type=str, action='store',help='Clustering model filename (.h5)')
	parser.add_argument('--predict_clust', dest='predict_clust', action='store_true',help='Only predict clustering according to current clustering model (default=false)')	
	parser.set_defaults(predict_clust=False)

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
	datalist_cv= args.datalist_cv
	
	# - Data process options	
	#nx= args.nx
	#ny= args.ny
	resize= args.resize
	resize_size= args.resize_size
	downscale_with_antialiasing= args.downscale_with_antialiasing
	upscale= args.upscale
	set_pad_val_to_min= args.set_pad_val_to_min
	augment= args.augment
	augmenter= args.augmenter
	augment_scale_factor= args.augment_scale_factor
	scale= args.scale
	scale_factors= []
	if args.scale_factors!="":
		scale_factors= [float(x.strip()) for x in args.scale_factors.split(',')]

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
	mask_borders= args.mask_borders
	mask_border_fract= args.mask_border_fract

	resize_chans= args.resize_chans
	nchan_resize= args.nchan_resize

	zscale_stretch= args.zscale_stretch
	zscale_contrasts= [float(x) for x in args.zscale_contrasts.split(',')]


	# - NN architecture
	use_vae= args.use_vae
	modelfile_encoder= args.modelfile_encoder
	modelfile_decoder= args.modelfile_decoder
	add_maxpooling_layer= args.add_maxpooling_layer
	add_batchnorm_layer= args.add_batchnorm_layer
	add_leakyrelu= args.add_leakyrelu
	add_dense_layer= args.add_dense_layer	
	add_channorm_layer= args.add_channorm_layer
	nfilters_cnn= [int(x.strip()) for x in args.nfilters_cnn.split(',')]
	kernsizes_cnn= [int(x.strip()) for x in args.kernsizes_cnn.split(',')]	
	strides_cnn= [int(x.strip()) for x in args.strides_cnn.split(',')]
	dense_layer_sizes= [int(x.strip()) for x in args.dense_layer_sizes.split(',')]
	dense_layer_activation= args.dense_layer_activation
	decoder_output_layer_activation= args.decoder_output_layer_activation
	add_dropout_layer= args.add_dropout_layer
	dropout_rate= args.dropout_rate
	add_conv_dropout_layer= args.add_conv_dropout_layer
	conv_dropout_rate= args.conv_dropout_rate
	
	# - Train options
	predict= args.predict
	reconstruct= args.reconstruct
	weightfile_encoder= args.weightfile_encoder
	weightfile_decoder= args.weightfile_decoder
	latentdim= args.latentdim
	optimizer= args.optimizer
	learning_rate= args.learning_rate
	batch_size= args.batch_size
	nepochs= args.nepochs
	mse_loss= args.mse_loss
	scale_chan_mse_loss= args.scale_chan_mse_loss
	kl_loss= args.kl_loss
	ssim_loss= args.ssim_loss
	mse_loss_weight= args.mse_loss_weight
	kl_loss_weight= args.kl_loss_weight
	ssim_loss_weight= args.ssim_loss_weight
	ssim_win_size= args.ssim_win_size
	weight_seed= args.weight_seed
	reproducible= args.reproducible
	validation_steps= args.validation_steps

	load_cv_data_in_batches= args.load_cv_data_in_batches
	multiprocessing= args.multiprocessing

	balance_classes_in_batch= args.balance_classes_in_batch
	#class_probs_dict= {}
	#if args.class_probs!="":
	#	class_probs= [float(x.strip()) for x in args.class_probs.split(',')]
	#	for i in range(len(class_probs)):
	#		class_probs_dict[i]= class_probs[i]
	#	print("== class_probs ==")
	#	print(class_probs_dict)

	class_probs_dict= {}
	if args.class_probs!="":
		try:
			class_probs_dict= json.loads(args.class_probs)
		except Exception as e:
			logger.error("Failed to convert class prob string to dict (err=%s)!" % (str(e)))
			return -1	

		print("== class_probs ==")
		print(class_probs_dict)

	# - Reco metrics & plot options
	winsize= args.winsize
	save_plots= args.save_plots

	# - UMAP options
	run_umap= args.run_umap
	latentdim_umap= args.latentdim_umap
	mindist_umap= args.mindist_umap
	nneighbors_umap= args.nneighbors_umap
	outfile_umap_unsupervised= args.outfile_umap_unsupervised
	outfile_umap_supervised= args.outfile_umap_supervised
	outfile_umap_preclassified= args.outfile_umap_preclassified
		
	# - Clustering options
	run_clustering= args.run_clustering
	min_cluster_size= args.min_cluster_size
	min_samples= args.min_samples	
	modelfile_clust= args.modelfile_clust
	predict_clust= args.predict_clust


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
	logger.info("Create train data pre-processor ...")
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
		preprocess_stages.append(ZScaleTransformer(constrasts=zscale_contrasts))

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

	print("== PRE-PROCESSING STAGES (TRAIN) ==")
	print(preprocess_stages)

	dp= DataPreprocessor(preprocess_stages)

	# - Creating data pre-processor for validation/test data
	logger.info("Creating data pre-processor for validation data ...")
	preprocess_stages_val= []
	for stage in preprocess_stages:
		if isinstance(stage, Augmenter):
			continue
		preprocess_stages_val.append(stage)

	print("== PRE-PROCESSING STAGES (VAL) ==")
	print(preprocess_stages_val)

	dp_val= DataPreprocessor(preprocess_stages_val)

	#===============================
	#==  DATA GENERATOR
	#===============================
	# - Create train data generator
	dg= DataGenerator(filename=datalist, preprocessor=dp)

	logger.info("Reading datalist %s ..." % datalist)
	if dg.read_datalist()<0:
		logger.error("Failed to read input datalist!")
		return 1

	# - Create validation data generator
	dg_cv= None
	if datalist_cv!="":
		dg_cv= DataGenerator(filename=datalist_cv, preprocessor=dp_val)
		
		logger.info("Reading datalist_cv %s ..." % (datalist_cv))
		if dg_cv.read_datalist()<0:
			logger.error("Failed to read input datalist for validation!")
			return 1

	#===========================
	#==   TRAIN AE
	#===========================
	ae= FeatExtractorAE(dg)
	ae.use_vae= use_vae
	ae.modelfile_encoder= modelfile_encoder
	ae.modelfile_decoder= modelfile_decoder
	ae.weightfile_encoder= weightfile_encoder
	ae.weightfile_decoder= weightfile_decoder
	ae.set_image_size(resize_size, resize_size)
	ae.augmentation= augment
	ae.augment_scale_factor= augment_scale_factor
	ae.latent_dim= latentdim

	ae.batch_size= batch_size
	ae.nepochs= nepochs
	ae.validation_steps= validation_steps
	ae.set_optimizer(optimizer, learning_rate)
	if reproducible:
		ae.set_reproducible_model()
	
	ae.add_max_pooling= add_maxpooling_layer
	ae.add_batchnorm= add_batchnorm_layer
	ae.add_leakyrelu= add_leakyrelu
	ae.add_dense= add_dense_layer
	ae.add_channorm_layer= add_channorm_layer
	ae.nfilters_cnn= nfilters_cnn
	ae.kernsizes_cnn= kernsizes_cnn
	ae.strides_cnn= strides_cnn
	ae.dense_layer_sizes= dense_layer_sizes
	ae.dense_layer_activation= dense_layer_activation
	ae.add_dropout_layer= add_dropout_layer
	ae.dropout_rate= dropout_rate
	ae.add_conv_dropout_layer= add_conv_dropout_layer
	ae.conv_dropout_rate= conv_dropout_rate

	ae.use_mse_loss= mse_loss
	ae.scale_chan_mse_loss= scale_chan_mse_loss
	ae.use_kl_loss= kl_loss
	ae.use_ssim_loss= ssim_loss
	ae.mse_loss_weight= mse_loss_weight
	ae.kl_loss_weight= kl_loss_weight
	ae.ssim_loss_weight= ssim_loss_weight
	ae.ssim_win_size= ssim_win_size
	ae.weight_seed= weight_seed

	ae.use_multiprocessing= multiprocessing
	ae.dg_cv= dg_cv
	ae.load_cv_data_in_batches= load_cv_data_in_batches

	ae.balance_classes= balance_classes_in_batch
	ae.class_probs= class_probs_dict

	if predict:
		logger.info("Running autoencoder predict ...")
		status= ae.predict_model(modelfile_encoder, weightfile_encoder)
	elif reconstruct:
		logger.info("Running autoencoder reconstruction ...")
		status= ae.reconstruct_data(
			modelfile_encoder, weightfile_encoder, 
			modelfile_decoder, weightfile_decoder,
			winsize= winsize,
			save_imgs= save_plots
		)
	else:
		logger.info("Running autoencoder training ...")
		status= ae.train_model()
			
	if status<0:
		logger.error("CAE run failed!")
		return 1

	#===========================
	#==   TRAIN UMAP
	#===========================
	if run_umap:
		# - Retrieve autoencoder latent data
		logger.info("Retrieve latent data from autoencoder model ...")
		snames= ae.source_names
		classids= ae.source_ids
		vae_data= ae.encoded_data

		# - Run UMAP	
		logger.info("Running UMAP classifier training on autoencoder latent data ...")
		umap_class= FeatExtractorUMAP()

		umap_class.set_encoded_data_unsupervised_outfile(outfile_umap_unsupervised)
		umap_class.set_encoded_data_supervised_outfile(outfile_umap_supervised)
		umap_class.set_encoded_data_preclassified_outfile(outfile_umap_preclassified)
		umap_class.set_encoded_data_dim(latentdim_umap)
		umap_class.set_min_dist(mindist_umap)
		umap_class.set_n_neighbors(nneighbors_umap)

		if umap_class.run_train(vae_data, class_ids=classids, snames=snames)<0:
			logger.error("UMAP training failed!")
			return 1

	#==============================
	#==   RUN CLUSTERING
	#==============================
	if run_clustering:
		# - Retrieve autoencoder latent data
		logger.info("Retrieve latent data from autoencoder ...")
		snames= ae.source_names
		classids= ae.source_ids
		vae_data= ae.encoded_data

		# - Run HDBSCAN clustering
		logger.info("Running HDBSCAN classifier prediction on autoencoder latent data ...")
		clust_class= Clusterer()
		clust_class.min_cluster_size= min_cluster_size
		clust_class.min_samples= min_samples
	
		status= 0
		if predict_clust:
			if clust_class.run_predict(vae_data, class_ids=classids, snames=snames, modelfile=modelfile_clust)<0:
				logger.error("Clustering predict failed!")
				return 1
		else:
			if clust_class.run_clustering(vae_data, class_ids=classids, snames=snames, modelfile=modelfile_clust)<0:
				logger.error("Clustering run failed!")
				return 1
	

	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

