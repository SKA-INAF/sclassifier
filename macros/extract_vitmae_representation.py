#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
## STANDARD MODULES
import os
from pathlib import Path
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
import io
import re
from _ctypes import PyObj_FromPtr  # see https://stackoverflow.com/a/15012814/355230

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections
import csv
import json
import pickle

## PYTORCH
import torch
import torchvision.transforms.functional as TF
#import torchvision.transforms as T
from PIL import Image

## TRANSFORMERS
from transformers import AutoProcessor, AutoModel
from transformers import AutoImageProcessor, ViTMAEModel, ViTForImageClassification

## ASTRO/IMG PROCESSING MODULES
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip
from astropy.visualization import ZScaleInterval
from astropy.stats import biweight_location, biweight_scale

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
	parser.add_argument('-inputfile','--inputfile', dest='inputfile', required=True, type=str, help='Path to file with image datalist (.json)') 
	parser.add_argument('-datalist_key','--datalist_key', dest='datalist_key', required=False, type=str, default="data", help='Dictionary key name to be read in input datalist (default=data)') 
	parser.add_argument('-nmax','--nmax', dest='nmax', required=False, type=int, default=-1, help='Max number of entries processed in filelist (-1=all)') 
	
	# - Data options
	parser.add_argument('-dataloader','--dataloader', dest='dataloader', required=False, type=str, default="tensor_3chan", help='Data loader mode {"pil_gray", "pil_rgb", "tensor_3chan"}') 
	
	parser.add_argument('--imgsize', default=256, type=int, help='Image resize size in pixels (default=256)')
	parser.add_argument('--reset_meanstd', dest='reset_meanstd', action='store_true', help='Reset original mean/std transform used in processor (default=false)')	
	parser.set_defaults(reset_meanstd=False)
	parser.add_argument('--reset_rescale', dest='reset_rescale', action='store_true', help='Reset original rescale transform used in processor (default=false)')	
	parser.set_defaults(reset_rescale=False)
	
	parser.add_argument('--clip_data', dest='clip_data', action='store_true',help='Apply sigma clipping transform (default=false)')	
	parser.set_defaults(clip_data=False)
	
	parser.add_argument('--robust_clip_data', dest='robust_clip_data', action='store_true',help='Apply robust sigma clipping transform (default=false)')	
	parser.set_defaults(robust_clip_data=False)
	parser.add_argument('--robust_clip_perc_min', dest='robust_clip_perc_min', default=2., type=float, help='Robust clipping min perc (default=2.0)')
	parser.add_argument('--robust_clip_perc_max', dest='robust_clip_perc_max', default=98., type=float, help='Robust clipping max perc (default=98.0)')
	
	parser.add_argument('--zscale', dest='zscale', action='store_true',help='Apply zscale transform (default=false)')	
	parser.set_defaults(zscale=False)
	parser.add_argument('--zscale_contrast', default=0.25, type=float, help='ZScale transform contrast (default=0.25)')
	
	parser.add_argument('--subtract_bkg', dest='subtract_bkg', action='store_true',help='Apply bkg subtraction transform (default=false)')	
	parser.set_defaults(subtract_bkg=False)
	parser.add_argument('--bkg_perc', dest='bkg_perc', default=25., type=float, help='Bkg percentile (default=25.0)')
	
	parser.add_argument('--biweight_normalization', dest='biweight_normalization', action='store_true',help='Apply biweight normalization transform (default=false)')	
	parser.set_defaults(biweight_normalization=False)
	
	parser.add_argument('--quantile_scaling', dest='quantile_scaling', action='store_true',help='Apply quantile scaling transform (default=false)')	
	parser.set_defaults(quantile_scaling=False)
	parser.add_argument('--quantile_perc_min', dest='quantile_perc_min', default=1., type=float, help='Quantile percentile min (default=1.0)')
	parser.add_argument('--quantile_perc_max', dest='quantile_perc_max', default=99., type=float, help='Quantile percentile max (default=99.0)')
	
	
	parser.add_argument('--norm_min', default=0., type=float, help='Norm min (default=0)')
	parser.add_argument('--norm_max', default=1., type=float, help='Norm max (default=1)')
	
	parser.add_argument('--to_uint8', dest='to_uint8', action='store_true',help='Convert to uint8 (default=false)')	
	parser.set_defaults(to_uint8=False)
	
	parser.add_argument('--in_chans', default = 1, type = int, help = 'Length of subset of dataset to use.')
	
	parser.add_argument('--set_zero_to_min', dest='shift_zero_to_min', action='store_true',help='Set blank pixels to min>0 (default=false)')	
	parser.set_defaults(set_zero_to_min=False)
	
	# - Model option
	parser.add_argument('-model','--model', dest='model', required=False, type=str, default="facebook/vit-mae-base", help='ViTMAE pretrained model {"facebook/vit-mae-base"}') 
	
	# - Run options
	parser.add_argument('-device','--device', dest='device', required=False, type=str, default="cuda", help='Device where to run inference. Default is cuda, if not found use cpu.') 
	
	# - Outfile option
	parser.add_argument('--save_to_json', dest='save_to_json', action='store_true',help='Save features to json (default=save to ascii)')
	parser.set_defaults(save_to_json=False)
	parser.add_argument('--save_labels_in_ascii', dest='save_labels_in_ascii', action='store_true',help='Save class labels to ascii (default=save classids)')
	parser.set_defaults(save_labels_in_ascii=False)
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='featdata.dat', help='Output filename (.dat) of feature data') 
	
	parser.add_argument('--verbose', dest='verbose', action='store_true',help='Turn on verbose printout')
	parser.set_defaults(verbose=False)
	
	args = parser.parse_args()	

	return args
	




def read_datalist(filename, key="data"):
  """ Read datalist """

  f= open(filename, "r")
  d= json.load(f)
  if key in d:
    return d[key]
  return d
	
def get_clipped_data(data, sigma_low=5, sigma_up=30):
	""" Apply sigma clipping to input data and return transformed data """

	# - Find NaNs pixels
	cond= np.logical_and(data!=0, np.isfinite(data))
	data_1d= data[cond]

	# - Clip all pixels that are below sigma clip
	res= sigma_clip(data_1d, sigma_lower=sigma_low, sigma_upper=sigma_up, masked=True, return_bounds=True)
	thr_low= res[1]
	thr_up= res[2]

	data_clipped= np.copy(data)
	data_clipped[data_clipped<thr_low]= thr_low
	data_clipped[data_clipped>thr_up]= thr_up

	# - Set NaNs to 0
	data_clipped[~cond]= 0

	return data_clipped


	
def get_zscaled_data(data, contrast=0.25):
	""" Apply sigma clipping to input data and return transformed data """

	# - Find NaNs pixels
	cond= np.logical_and(data!=0, np.isfinite(data))

	# - Apply zscale transform
	transform= ZScaleInterval(contrast=contrast)
	data_transf= transform(data)	

	# - Set NaNs to 0
	data_transf[~cond]= 0

	return data_transf
	
	
def get_robust_clipped_data(data, percentiles):
	"""
		Clips the input data to specified lower and upper percentiles.
	
			Purpose:
				Removes extreme outlier pixel values to stabilize the image dynamic range.

				Relation to radio astronomy:
					Radio images often contain outlier intensities due to bright compact sources,
					calibration artifacts, or residual RFI. Clipping to e.g. (2,98) percentiles
					ensures that the majority of the background noise and faint source structures
					are preserved without extreme pixel values dominating scaling.

			Caveats:
				- May suppress true flux of very bright sources.
				- The chosen percentile range should balance outlier removal with preserving astrophysically significant peaks.
	"""

	p_low, p_high = np.percentile(data, percentiles)
	return np.clip(data, p_low, p_high).astype(np.float32)

def get_background_subtracted_data(data, bg_percentile):
	"""
		Subtracts a background estimate computed as a specified percentile of the data.

		Purpose:
			Centers the image around zero by removing a robust estimate of the background level.

		Relation to radio astronomy:
			In typical radio cutouts, the majority of pixels represent background noise.
			Using a low percentile (e.g. 25th) provides a stable approximation of the
			noise floor, enhancing the relative contrast of faint sources.

		Caveats:
			- In images with extensive bright emission, this percentile may overestimate
				the true background, leading to negative residuals.
	"""
	background = np.percentile(data, bg_percentile)
	return (data - background).astype(np.float32)
  
  
def get_biweight_normalized_data(data):
	"""
		Normalizes data using biweight location and scale as robust estimators.

		Purpose:
			Robustly centers and scales the data to reduce the influence of outliers
			compared to standard mean/std normalization.

		Relation to radio astronomy:
			Radio images often include Gaussian-like noise with occasional bright sources.
			Biweight location (robust mean) and scale (robust std) provide stable estimates
			even in the presence of such outliers, maintaining background structure while
			preventing bright sources from dominating the normalization.

		Caveats:
			- If the image has extremely low SNR or is nearly empty, biweight scale can
				approach zero; a small floor (1e-6) is used to prevent division errors.
	"""
	flat = data.flatten()
	center = biweight_location(flat)
	scale = biweight_scale(flat)
	if scale < 1e-6:
		scale = 1e-6
	return ((data - center) / scale).astype(np.float32)      
	

def get_quantile_scaled_data(data, percentiles):
	"""
		Linearly rescales image pixel intensities to the [0,1] range based on specified lower and upper quantiles.

		Input:
			data (np.ndarray): Single-channel radio astronomy image as a float32 array.
      percentiles (tuple): Two percentile values (e.g. (1,99)) defining the intensity range to map to [0,1].

		Output:
			np.ndarray: Rescaled image as float32, with intensities clipped and linearly mapped to [0,1].

		Purpose:
			Normalizes each image to a fixed numeric range for stable neural network training and to ensure
			that differences in telescope flux scales or noise levels do not affect input distributions.

		Relation to radio astronomy:
			Radio images often span wide dynamic ranges due to strong sources and background noise variations.
			Quantile-based scaling ensures faint sources remain detectable while normalizing extreme outliers.

		Caveats:
			- If the specified quantile range is too narrow (e.g. in nearly flat images), output may be zero everywhere.	
			- Absolute flux density calibration is lost; this method is appropriate for tasks focusing on morphology
				rather than calibrated flux measurements.
	"""
	q_low, q_high = np.percentile(data, percentiles)
	if q_high - q_low < 1e-6:
		return np.zeros_like(data, dtype=np.float32)
	scaled = (data - q_low) / (q_high - q_low)
	return np.clip(scaled, 0, 1).astype(np.float32)	
	
	
def transform_img(data, args):
	""" Transform input data """
	
	data_transf= np.copy(data)

	# - Set NANs to image min
	if args.set_zero_to_min:
		cond= np.logical_and(data_transf!=0, np.isfinite(data_transf))
	else:
		cond= np.isfinite(data_transf)
				
	data_1d= data_transf[cond]
	if data_1d.size==0:
		print("WARN: Input data are all zeros/nan, return None!")
		return None
			
	data_min= np.min(data_1d)
	data_transf[~cond]= data_min

	if args.verbose:
		print("== DATA MIN/MAX ==")
		print(data_transf.min())
		print(data_transf.max())

	# - Clip data?
	if args.clip_data:
		data_clipped= get_clipped_data(data_transf, sigma_low=5, sigma_up=30)
		data_transf= data_clipped

	# - Apply zscale stretch
	if args.zscale:
		#print("Apply zscale stretch ...")
		data_stretched= get_zscaled_data(data_transf, contrast=args.zscale_contrast)
		data_transf= data_stretched

	# 2. Robust percentile clipping
	if args.robust_clip_data:
		data_clipped= get_robust_clipped_data(data_transf, (args.robust_clip_perc_min, args.robust_clip_perc_max) )
		data_transf= data_clipped
		
		if args.verbose:
			print("== DATA MIN/MAX (AFTER ROBUST CLIP) ==")
			print(data_transf.min())
			print(data_transf.max())

	# 3. Background subtraction
	if args.subtract_bkg:
		data_nobkg= get_background_subtracted_data(data_transf, args.bkg_perc)
		data_transf= data_nobkg
		
		if args.verbose:
			print("== DATA MIN/MAX (AFTER BKG SUB) ==")
			print(data_transf.min())
			print(data_transf.max())

	# 4. Biweight normalization
	if args.biweight_normalization:
		data_norm= get_biweight_normalized_data(data_transf)
		data_transf= data_norm
		
		if args.verbose:
			print("== DATA MIN/MAX (AFTER BIWEIGHT) ==")
			print(data_transf.min())
			print(data_transf.max())

	# 5. Final quantile scaling to [0,1]
	if args.quantile_scaling:
		data_scaled= get_quantile_scaled_data(data_transf, (args.quantile_perc_min, args.quantile_perc_max))
		data_transf= data_scaled
		
		if args.verbose:
			print("== DATA MIN/MAX (AFTER QUANTILE SCALING) ==")
			print(data_transf.min())
			print(data_transf.max())

	# 6. 3-channel tensor
	#tensor = torch.from_numpy(data).unsqueeze(0).repeat(3, 1, 1).contiguous().float()

	# - Expand 2D data to desired number of channels (if>1): shape=(ny,nx,nchans)
	ndim= data_transf.ndim
	nchans= args.in_chans
	if nchans>1 and ndim==2:
		data_transf= np.stack((data_transf,) * nchans, axis=-1)
	
	if args.verbose:
		print("== DATA MIN/MAX (AFTER IMG TRANSF) ==")
		print(data_transf.min())
		print(data_transf.max())
	
	# - Normalize to range
	data_min= data_transf.min()
	data_max= data_transf.max()
	norm_min= args.norm_min
	norm_max= args.norm_max
	if norm_min==data_min and norm_max==data_max:
		if args.verbose:
			print("INFO: Data already normalized in range (%f,%f)" % (norm_min, norm_max))
	else:
		data_norm= (data_transf-data_min)/(data_max-data_min) * (norm_max-norm_min) + norm_min
		data_transf= data_norm
			
	# - Convert to uint8
	if args.to_uint8:
		data_transf= data_transf.astype(np.uint8)
	
	if args.verbose:	
		print("== DATA MIN/MAX (AFTER NORM) ==")
		print(data_transf.min())
		print(data_transf.max())
	
	
	return data_transf


def read_img(filename, args):
	""" Read fits image """

	# - Check filename
	if filename=="":
		return None
		
	file_ext= os.path.splitext(filename)[1]
	
	# - Read fits image
	if file_ext=='.fits':
		with fits.open(filename, memmap=True) as hdul:
			data = hdul[0].data.astype(np.float32)
			data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
		
	elif file_ext in ['.png', '.jpg']:
		image= Image.open(filename)
		data= np.asarray(image)
	else:
		print("ERROR: Invalid or unrecognized file extension (%s)!" % (file_ext))
		return None
		
	if data is None:
		return None
				
	# - Apply transform to numpy array
	data_transf= transform_img(data, args)
	if data_transf is None:
		return None
	
	data= data_transf
	
	# - Convert numpy to PIL image
	image = Image.fromarray(data)
	
	# - Convert to RGB image
	if args.in_chans==3:
		image= image.convert("RGB")
		
	return image
	
	
def load_as_npy(filename, args):
	""" Read fits image """

	# - Check filename
	if filename=="":
		return None
		
	file_ext= os.path.splitext(filename)[1]
	
	# - Read fits image
	if file_ext=='.fits':
		with fits.open(filename, memmap=True) as hdul:
			data = hdul[0].data.astype(np.float32)
			data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
		
	elif file_ext in ['.png', '.jpg']:
		image= Image.open(filename)
		data= np.asarray(image)
	else:
		print("ERROR: Invalid or unrecognized file extension (%s)!" % (file_ext))
		return None
		
	if data is None:
		return None
		
	# - Check if image contains all zeros
	if data.min()==data.max():
		print("Image contains the same pixel value (%d), returning None!" % (data.min()))
		return None
				
	# - Apply transform to numpy array
	data_transf= transform_img(data, args)
	if data_transf is None:
		return None
		
	return data_transf
	
	
def load_as_pil_float(filename, args):
	""" Load image as PIL gray scale image """	
	
	# - Check number of channels is 1
	if args.in_chans!=1:
		print("WARN: Overriding args.in_chans (%d) to 1..." % (args.in_chans))
		args.in_chans= 1
		
	# - Check normalize to [0,1] range
	if args.norm_min!=0 or args.norm_max!=1:
		print("WARN: Overriding args.norm_min/args.norm_max (%d, %d) to [0,1] ..." % (args.norm_min, args.norm_max))	
		args.norm_min= 0.0
		args.norm_max= 1.0
		
	# - Read image as numpy (ny,nx) with all transforms applied
	data= load_as_npy(filename, args)
	if data is None:
		return None
	
	# - Convert numpy to PIL image
	#   NB: This works only for 2D images (e.g. 1-chan from FITS, args.in_chan=1)
	image = Image.fromarray(data)
	
	return image
	
	
def load_as_pil_rgb(filename, args):
	""" Load image as PIL RGB image """	
	
	# - Read image as numpy (1-chan from FITS, 3-chan from natural images)
	#   with all transforms applied
	# - Overriding args.in_chans=3
	if args.in_chans!=3:
		print("WARN: Overriding args.in_chans (%d) to 3..." % (args.in_chans))
		args.in_chans= 3
		
	# - Check normalize to [0,255] range
	if args.norm_min!=0 or args.norm_max!=1:
		print("WARN: Overriding args.norm_min/args.norm_max (%d, %d) to [0,1] ..." % (args.norm_min, args.norm_max))	
		args.norm_min= 0.0
		args.norm_max= 1.0
		
	# - Check uint8 is enabled
	if not args.to_uint8:
		print("WARN: Enabling uint8 conversion ...")	
		args.to_uint8= True
	
	data= load_as_npy(filename, args)
	if data is None:
		return None
	
	# - Convert numpy to PIL RGB image
	#   NB: This works only for 3D uint8 data (e.g. 1-chan from FITS has to be converted with args.in_chan=3)
	image = Image.fromarray(data).convert("RGB")
	
	return image
	
	
def load_as_tensor_3chan(filename, args):
	""" Load image as torch tensor """
	
	# - Read image as numpy (1-chan from FITS, 3-chan from natural images) with all transforms applied
	# - Overriding args.in_chans=3
	if args.in_chans!=3:
		print("WARN: Overriding args.in_chans (%d) to 3..." % (args.in_chans))
		args.in_chans= 3
	
	# - Check normalize to [0,1] range
	#if args.norm_min!=0 or args.norm_max!=1:
	#	print("WARN: Overriding args.norm_min/args.norm_max (%d, %d) to [0,1] ..." % (args.norm_min, args.norm_max))	
	#	args.norm_min= 0.0
	#	args.norm_max= 1.0
		
	data= load_as_npy(filename, args)
	if data is None:
		return None
		
	# - Convert to tensor
	tensor= torch.from_numpy(data.transpose((2, 0, 1))).contiguous().float()
		
	return tensor
	

	
	
def write_ascii(data, filename, header=''):
	""" Write data to ascii file """

	# - Skip if data is empty
	if data.size<=0:
		print("WARN: Empty data given, no file will be written!")
		return

	# - Open file and write header
	fout = open(filename, 'wt')
	if header:
		fout.write(header)
		fout.write('\n')	
		fout.flush()	
		
	# - Write data to file
	nrows= data.shape[0]
	ncols= data.shape[1]
	for i in range(nrows):
		fields= '  '.join(map(str, data[i,:]))
		fout.write(fields)
		fout.write('\n')	
		fout.flush()	

	fout.close()

## TAKEN FROM: https://stackoverflow.com/questions/42710879/write-two-dimensional-list-to-json-file
class NoIndent(object):
    """ Value wrapper. """
    def __init__(self, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError('Only lists and tuples can be wrapped')
        self.value = value


class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'  # Unique string pattern of NoIndent object ids.
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))  # compile(r'@@(\d+)@@')

    def __init__(self, **kwargs):
        # Keyword arguments to ignore when encoding NoIndent wrapped values.
        ignore = {'cls', 'indent'}

        # Save copy of any keyword argument values needed for use here.
        self._kwargs = {k: v for k, v in kwargs.items() if k not in ignore}
        super(MyEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
                    else super(MyEncoder, self).default(obj))

    def iterencode(self, obj, **kwargs):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.

        # Replace any marked-up NoIndent wrapped values in the JSON repr
        # with the json.dumps() of the corresponding wrapped Python object.
        for encoded in super(MyEncoder, self).iterencode(obj, **kwargs):
            match = self.regex.search(encoded)
            if match:
                id = int(match.group(1))
                no_indent = PyObj_FromPtr(id)
                json_repr = json.dumps(no_indent.value, **self._kwargs)
                # Replace the matched id string with json formatted representation
                # of the corresponding Python object.
                encoded = encoded.replace(
                            '"{}"'.format(format_spec.format(id)), json_repr)

            yield encoded


def save_features_to_ascii(datalist, outfile, save_labels=False):
	""" Save feature data to ascii file """

	# - Loop over datalist and fill lists
	features= []
	snames= []
	class_ids= []
	class_labels= []
	nsamples= len(datalist)

	for idx, item in enumerate(datalist):
		print("--> item")
		print(item)
		sname= item['sname']
		class_id= item['id']
		class_label= item['label']
		if 'feats' not in item: # Skip None Images
			continue
		feats= item['feats']
    
		features.append(feats.value)
		snames.append(sname)
		class_ids.append(class_id)
		class_labels.append(class_label)
		
	# - Save features to ascii file with format: sname, f1, f2, ..., fn, classid
	N= len(features)
	nfeats= len(features[0])
	print("INFO: Writing %d feature data (nfeats=%d) to file %s ..." % (N, nfeats, outfile))

	featdata_arr= np.array(features)
	snames_arr= np.array(snames).reshape(N,1)
	classids_arr= np.array(class_ids).reshape(N,1)
	classlabels_arr= np.array(class_labels).reshape(N,1)
  
	if save_labels:
		outdata= np.concatenate(
			(snames_arr, featdata_arr, classlabels_arr),
			axis=1
		)
	else:
		outdata= np.concatenate(
			(snames_arr, featdata_arr, classids_arr),
			axis=1
		)

	znames_counter= list(range(1,nfeats+1))
	znames= '{}{}'.format('z',' z'.join(str(item) for item in znames_counter))
	head= '{} {} {}'.format("# sname",znames,"id")
	write_ascii(outdata, outfile, head)

	return 0

def save_features_to_json(datalist, datalist_key, outfile):
	""" Save feat data to json """
	
	# - Create dict
	d= {datalist_key: datalist}
	
	# - Save to file
	print("Saving datalist to file %s ..." % (outfile))
	with open(outfile, 'w') as fp:
		#json.dump(d, fp, indent=2)
		#json.dump(d, fp, cls=MyEncoder, sort_keys=True, indent=2)
		json.dump(d, fp, cls=MyEncoder, indent=2)
		
	return 0

def extract_features(datalist, model, image_processor, device, args):
	""" Function to extract features from trained models """
	
	# - Print model
	print("--> model")
	print(model)
	
	# - Set data loader
	dataloader_fcn= load_as_tensor_3chan
	if args.dataloader=="pil_float":
		dataloader_fcn= load_as_pil_float
	elif args.dataloader=="pil_rgb":
		dataloader_fcn= load_as_pil_rgb
	elif args.dataloader=="tensor_3chan":
		dataloader_fcn= load_as_tensor_3chan
	
	# - Loop over datalist and extract features per each image
	nsamples= len(datalist)
	
	for idx, item in enumerate(datalist):
		if args.nmax!=-1 and idx>=args.nmax:
			print("INFO: Max number of entries processed (n=%d), exit." % (args.nmax))
			break

		if idx%1000==0:
			print("%d/%d entries processed ..." % (idx, nsamples))

		# - Read image
		filename= datalist[idx]["filepaths"][0]
		print("INFO: Reading image %s ..." % (filename))
		#img= read_img(filename, args)
		img= dataloader_fcn(filename, args)
		
		if img is None:
			print("WARN: Read/processed image %s is None, skip to next!" % (filename))
			continue
			
		# - Apply model pre-processing
		inputs = image_processor(images=img, return_tensors="pt").to(device)
		
		# - Extract image features
		with torch.no_grad():
			outputs = model(**inputs, output_hidden_states=True)
			
			if idx==0:
				print("outputs")
				print(type(outputs))
			
			# - Retrieve hidden states
			#   hidden_states are a tuple: (batch_size, num_patches + 1, hidden_size)
			#     - hidden_states[0]: output layer initial embedding
			#     - hidden_states[1]: output first layer Transformer.
      #     - ...
      #     - hidden_states[-1]: output last layer Transformer (we need this for feature extraction)
			hidden_states = outputs.hidden_states

			if idx==0:
				print("hidden_states")
				print(type(hidden_states))
				print(len(hidden_states))
				for hidden_state in hidden_states:
					print(hidden_state.shape)
    	
			# To get the features vector use token [CLS] (indice 0)
			features= hidden_states[-1][:,0,:]
    	
		if idx==0:
			print("features.shape")
			print(features.shape)
			
		features_numpy= features.cpu().numpy()[0]
		
		if idx==0:
			print("features_numpy.shape")
			print(features_numpy.shape)
			
		# - Append to main list
		feats_list= list(features_numpy)
		feats_list= [float(item) for item in feats_list]
    
		datalist[idx]["feats"]= NoIndent(feats_list)
		
	return datalist
		

##############
##   MAIN   ##
##############
def main():
	"""Main function"""

	#===========================
	#==   PARSE ARGS
	#===========================
	print("INFO: Get script args ...")
	try:
		args= get_args()
	except Exception as ex:
		print("ERROR: Failed to get and parse options (err=%s)",str(ex))
		return 1
		
	# - Input filelist
	if args.inputfile=="":
		print("ERROR: Empty datalist input file!")
		return 1	
	
	outfile= args.outfile
	
	#device = 'cuda' if torch.cuda.is_available() else 'cpu'
	device= args.device
	if "cuda" in device:
		if not torch.cuda.is_available():
			print("WARN: cuda not available, using cpu...")
			device= "cpu"
	
	print('device:',device)

	#===========================
	#==   READ DATALIST
	#===========================
	print("INFO: Read image dataset filelist %s ..." % (args.inputfile))
	datalist= read_datalist(args.inputfile, args.datalist_key)
	nfiles= len(datalist)
	
	#===========================
	#==   LOAD MODEL
	#===========================
	# - Load model
	print("INFO: Loading model %s ..." % (args.model))
	
	#model = AutoModel.from_pretrained(args.model).to(device)
	#model = ViTMAEModel.from_pretrained(args.model).to(device)
	model = ViTForImageClassification.from_pretrained(args.model).to(device)
	
	# - Load model processor
	print("INFO: Loading model processor ...")
	processor = AutoProcessor.from_pretrained(args.model)
	#image_processor= processor.image_processor
	image_processor = AutoImageProcessor.from_pretrained(args.model)
	
	
	print("image_processor")
	print(image_processor)
	#print("image_processor.size")
	#print(image_processor.size)
	
	print("== ORIGINAL PROCESSOR OPTIONS ==")
	imgsize_orig= (image_processor.size["height"], image_processor.size["width"])
	imgmean_orig= image_processor.image_mean
	imgstd_orig= image_processor.image_std
	print("imgsize_orig")
	print(imgsize_orig)
	print("imgmean_orig")
	print(imgmean_orig)
	print("imgstd_orig")
	print(imgstd_orig)
	
	# - Update processor options
	image_processor.size["height"]= args.imgsize
	image_processor.size["width"]= args.imgsize
	
	if args.reset_meanstd:
		image_processor.image_mean= [0.,0.,0.]
		image_processor.image_std= [1.,1.,1.]
		
	if args.reset_rescale:
		image_processor.do_rescale= False
		image_processor.rescale_factor= 1.0
		
	print("== FINAL PROCESSOR OPTIONS ==")
	print("image_processor")
	print(image_processor)	
	
	#===========================
	#==   EXTRACT FEATURES
	#===========================
	print("INFO: Extracting features from file %s ..." % (args.inputfile))
	datalist_out= extract_features(
		datalist,
		model, 
		image_processor,
		device, 
		args
	)
	if datalist_out is None:
	  print("ERROR: Failed to extract features!")
	  return 1

	print("datalist_out")
	print(datalist_out)

	#===========================
	#==   SAVE FEATURES
	#===========================
	print("INFO: Saving features to file %s ..." % (args.outfile))
	if args.save_to_json:
		save_features_to_json(
			datalist_out,
			args.datalist_key,
			args.outfile
		)
	else:
		save_features_to_ascii(
			datalist_out,
			args.outfile, 
			args.save_labels_in_ascii
		)
		
	return 0
	
###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())
