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
import io
import re

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections
import csv
import json
import pickle

## IMPORT IMAGE PROCESSING MODULES
import skimage
import skimage.transform
from PIL import Image

## TF MODULES
import tensorflow as tf
from tensorflow.keras.models import load_model

## ASTRO/IMG PROCESSING MODULES
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip
from astropy.visualization import ZScaleInterval
from astropy.io.fits.verify import VerifyWarning
from astropy.wcs import FITSFixedWarning
import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.simplefilter('ignore', category=VerifyWarning)
warnings.filterwarnings('ignore', category=FITSFixedWarning)

## IMPORT DRAW MODULES
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
	parser.add_argument('-model','--model', dest='model', required=True, type=str, help='Path to model architecture (.h5)') 
	parser.add_argument('-model_weights','--model_weights', dest='model_weights', required=True, type=str, help='Path to model weights (.h5)') 
	
	parser.add_argument('-datalist_key','--datalist_key', dest='datalist_key', required=False, type=str, default="data", help='Dictionary key name to be read in input datalist (default=data)') 
	
	# - Data options
	parser.add_argument('-nmax','--nmax', dest='nmax', required=False, type=int, default=-1, help='Max number of entries processed in filelist (-1=all)') 
	parser.add_argument('--imgsize', default=256, type=int, help='Image resize size in pixels (default=256)')
	
	#parser.add_argument('--clip_data', dest='clip_data', action='store_true',help='Apply sigma clipping transform (default=false)')	
	#parser.set_defaults(clip_data=False)
	parser.add_argument('--zscale', dest='zscale', action='store_true',help='Apply zscale transform (default=false)')	
	parser.set_defaults(zscale=False)
	parser.add_argument('--zscale_contrast', default=0.25, type=float, help='ZScale transform contrast (default=0.25)')
	
	#parser.add_argument('--norm_min', default=0., type=float, help='Norm min (default=0)')
	#parser.add_argument('--norm_max', default=1., type=float, help='Norm max (default=1)')
	#parser.add_argument('--to_uint8', dest='to_uint8', action='store_true',help='Convert to uint8 (default=false)')	
	#parser.set_defaults(to_uint8=False)
	#parser.add_argument('--in_chans', default = 1, type = int, help = 'Length of subset of dataset to use.')
	#parser.add_argument('--set_zero_to_min', dest='shift_zero_to_min', action='store_true',help='Set blank pixels to min>0 (default=false)')	
	#parser.set_defaults(set_zero_to_min=False)
	
	# - Run options
	#parser.add_argument('-device','--device', dest='device', required=False, type=str, default="cuda", help='Device where to run inference. Default is cuda, if not found use cpu.') 
	
	# - Outfile option
	parser.add_argument('--save_to_json', dest='save_to_json', action='store_true',help='Save features to json (default=save to ascii)')
	parser.set_defaults(save_to_json=False)
	parser.add_argument('--save_labels_in_ascii', dest='save_labels_in_ascii', action='store_true',help='Save class labels to ascii (default=save classids)')
	parser.set_defaults(save_labels_in_ascii=False)
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='featdata.dat', help='Output filename (.dat) of feature data') 
	
	args = parser.parse_args()

	return args
	
#################################
##      WRITE ASCII
#################################
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

#################################
##      DATA TRANSFORM
#################################
def resize_img(
  image,
  min_dim=None, max_dim=None, min_scale=None,
  mode="square",
  order=1,
  preserve_range=True,
  anti_aliasing=False
):
  """ Resize numpy array to desired size """

  # Keep track of image dtype and return results in the same dtype
  image_dtype = image.dtype
  image_ndims= image.ndim

  # - Default window (y1, x1, y2, x2) and default scale == 1.
  h, w = image.shape[:2]
  window = (0, 0, h, w)
  scale = 1
  if image_ndims==3:
    padding = [(0, 0), (0, 0), (0, 0)] # with multi-channel images
  elif image_ndims==2:
    padding = [(0, 0)] # with 2D images
  else:
    print("ERROR: Unsupported image ndims (%d), returning None!" % (image_ndims))
    return None

  crop = None

  if mode == "none":
    return image, window, scale, padding, crop

  # - Scale?
  if min_dim:
    # Scale up but not down
    scale = max(1, min_dim / min(h, w))

  if min_scale and scale < min_scale:
    scale = min_scale

  # Does it exceed max dim?
  if max_dim and mode == "square":
    image_max = max(h, w)
    if round(image_max * scale) > max_dim:
      scale = max_dim / image_max

  # Resize image using bilinear interpolation
  if scale != 1:
    image= skimage.transform.resize(
      image,
      (round(h * scale), round(w * scale)),
      order=order,
      mode="constant",
      cval=0, clip=True,
      preserve_range=preserve_range,
      anti_aliasing=anti_aliasing, anti_aliasing_sigma=None
    )

  # Need padding or cropping?
  if mode == "square":
    # Get new height and width
    h, w = image.shape[:2]
    top_pad = (max_dim - h) // 2
    bottom_pad = max_dim - h - top_pad
    left_pad = (max_dim - w) // 2
    right_pad = max_dim - w - left_pad

    if image_ndims==3:
      padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)] # multi-channel
    elif image_ndims==2:
      padding = [(top_pad, bottom_pad), (left_pad, right_pad)] # 2D images
    else:
      print("ERROR: Unsupported image ndims (%d), returning None!" % (image_ndims))
      return None

    image = np.pad(image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)

  elif mode == "pad64":
    h, w = image.shape[:2]
    # - Both sides must be divisible by 64
    if min_dim % 64 != 0:
      print("ERROR: Minimum dimension must be a multiple of 64, returning None!")
      return None

    # Height
    if h % 64 > 0:
      max_h = h - (h % 64) + 64
      top_pad = (max_h - h) // 2
      bottom_pad = max_h - h - top_pad
    else:
      top_pad = bottom_pad = 0

    # - Width
    if w % 64 > 0:
      max_w = w - (w % 64) + 64
      left_pad = (max_w - w) // 2
      right_pad = max_w - w - left_pad
    else:
      left_pad = right_pad = 0

    if image_ndims==3:
      padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    elif image_ndims==2:
      padding = [(top_pad, bottom_pad), (left_pad, right_pad)]
    else:
      print("ERROR: Unsupported image ndims (%d), returning None!" % (image_ndims))
      return None

    image = np.pad(image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)

  elif mode == "crop":
    # - Pick a random crop
    h, w = image.shape[:2]
    y = random.randint(0, (h - min_dim))
    x = random.randint(0, (w - min_dim))
    crop = (y, x, min_dim, min_dim)
    image = image[y:y + min_dim, x:x + min_dim]
    window = (0, 0, min_dim, min_dim)

  else:
    print("ERROR: Mode %s not supported!" % (mode))
    return None

  return image.astype(image_dtype)


def transform_img(data, nchans=1, norm_range=(0.,1.), resize=False, resize_size=224, apply_zscale=True, contrast=0.25, to_uint8=False, set_nans_to_min=False, verbose=False):
  """ Transform input image data and return transformed data """

  # - Replace NANs pixels with 0 or min
  cond_nonan= np.isfinite(data)
  cond_nonan_noblank= np.logical_and(data!=0, np.isfinite(data))
  data_1d= data[cond_nonan_noblank]
  if data_1d.size==0:
    print("WARN: Input data are all zeros/nan, return None!")
    return None

  if set_nans_to_min:
    data[~cond_nonan]= data_min
  else:
    data[~cond_nonan]= 0

  data_transf= data

  if verbose:
    print("== DATA MIN/MAX (BEFORE TRANSFORM)==")
    print(data_transf.min())
    print(data_transf.max())

  # - Apply zscale stretch?
  if apply_zscale:
    transform= ZScaleInterval(contrast=contrast)
    data_zscaled= transform(data_transf)
    data_transf= data_zscaled

  # - Resize image?
  if resize:
    interp_order= 3 # 1=bilinear, 2=biquadratic, 3=bicubic, 4=biquartic, 5=biquintic
    data_transf= resize_img(
      data_transf,
      min_dim=resize_size, max_dim=resize_size, min_scale=None,
      mode="square",
      order=interp_order,
      preserve_range=True,
      anti_aliasing=False
    )

  if verbose:
    print("== DATA MIN/MAX (AFTER TRANSFORM) ==")
    print(data_transf.shape)
    print(data_transf.min())
    print(data_transf.max())

  # - Apply min/max normalization
  data_min= data_transf.min()
  data_max= data_transf.max()
  norm_min= norm_range[0]
  norm_max= norm_range[1]
  data_norm= (data_transf-data_min)/(data_max-data_min) * (norm_max-norm_min) + norm_min
  data_transf= data_norm

  if verbose:
    print("== DATA MIN/MAX (AFTER TRANSFORM) ==")
    print(data_transf.shape)
    print(data_transf.min())
    print(data_transf.max())

  # - Expand 2D data to desired number of channels (if>1): shape=(ny,nx,nchans)
  if nchans>1:
    data_transf= np.stack((data_transf,) * nchans, axis=-1)

  # - Convert to uint8
  if to_uint8:
    data_transf= data_transf.astype(np.uint8)

  if verbose:
    print("== DATA MIN/MAX (AFTER RESHAPE) ==")
    print(data_transf.shape)
    print(data_transf.min())
    print(data_transf.max())

  return data_transf

#################################
##      READ FITS
#################################
def read_datalist(filename, key="data"):
  """ Read datalist """

  f= open(filename, "r")
  d= json.load(f)
  if key in d:
    return d[key]
  return d
  
  
def read_img(filename, nchans=1, norm_range=(0.,1.), resize=False, resize_size=224, apply_zscale=True, contrast=0.25, to_uint8=False, set_nans_to_min=False, verbose=False):
  """ Read fits image and returns a numpy array """

  # - Check filename
  if filename=="":
    return None
    
  fileext= os.path.splitext(filename)[1]
  #print("fileext=",fileext)

  # - Read FITS/PNG/JPEG image
  if fileext=='.fits':
    data= fits.open(filename)[0].data
  elif fileext in ['.png', '.jpg']:
    image= Image.open(filename)
    data= np.asarray(image)
  else:
    print("ERROR: Invalid or unrecognized file extension (%s)!" % (fileext))
    return None
  
  if data is None:
    return None

  # - Apply transform
  data_transf= transform_img(
    data,
    nchans=nchans,
    norm_range=norm_range,
    resize=resize, resize_size=resize_size,
    apply_zscale=apply_zscale, contrast=contrast,
    to_uint8=to_uint8,
    set_nans_to_min=set_nans_to_min,
    verbose=verbose
  )

  return data_transf

def load_img_as_npy_float(filename, add_chan_axis=True, add_batch_axis=True, resize=False, resize_size=224, apply_zscale=True, contrast=0.25, set_nans_to_min=False, verbose=False):
  """ Return numpy float image array norm to [0,1] """

  # - Read image from file and get transformed npy array
  data= read_img(
    filename,
    nchans=1,
    norm_range=(0.,1.),
    resize=resize, resize_size=resize_size,
    apply_zscale=apply_zscale, contrast=contrast,
    to_uint8=False,
    set_nans_to_min=set_nans_to_min,
    verbose=verbose
  )
  
  if data is None:
    print("WARN: Read image is None!")
    return None

  # - Add channel axis if missing?
  ndim= data.ndim
  if ndim==2 and add_chan_axis:
    data_reshaped= np.stack((data,), axis=-1)
    data= data_reshaped

    # - Add batch axis if requested
    if add_batch_axis:
      data_reshaped= np.stack((data,), axis=0)
      data= data_reshaped

  return data

def load_img_as_tftensor_float(filename, add_chan_axis=True, add_batch_axis=True, resize=False, resize_size=224, apply_zscale=True, contrast=0.25, set_nans_to_min=False, verbose=False):
  """ Return TensorFlow float image tensor norm to [0,1] """

  # - Read FITS from file and get transformed npy array
  data= load_img_as_npy_float(
    filename,
    add_chan_axis=add_chan_axis, add_batch_axis=add_batch_axis,
    resize=resize, resize_size=resize_size,
    apply_zscale=apply_zscale, contrast=contrast,
    set_nans_to_min=set_nans_to_min,
    verbose=verbose
  )
  if data is None:
    print("WARN: Read image is None!")
    return None

  return tf.convert_to_tensor(data)

def load_img_as_npy_rgb(filename, add_batch_axis=True, resize=False, resize_size=224, apply_zscale=True, contrast=0.25, set_nans_to_min=False, verbose=False):
  """ Return 3chan RGB image numpy norm to [0,255], uint8 """

  # - Read FITS from file and get transformed npy array
  data= read_img(
    filename,
    nchans=3,
    norm_range=(0.,255.),
    resize=resize, resize_size=resize_size,
    apply_zscale=apply_zscale, contrast=contrast,
    to_uint8=True,
    set_nans_to_min=set_nans_to_min,
    verbose=verbose
  )

  if data is None:
    print("WARN: Read image is None!")
    return None

  # - Add batch axis if requested
  if add_batch_axis:
    data_reshaped= np.stack((data,), axis=0)
    data= data_reshaped

  return data

def load_img_as_tftensor_rgb(filename, add_batch_axis=True, resize=False, resize_size=224, apply_zscale=True, contrast=0.25, set_nans_to_min=False, verbose=False):
  """ Return 3chan RGB image tensor norm to [0,255], uint8 """

  # - Read FITS from file and get transformed npy array
  data= load_img_as_npy_rgb(
    filename,
    add_chan_axis=add_chan_axis, add_batch_axis=add_batch_axis,
    resize=resize, resize_size=resize_size,
    apply_zscale=apply_zscale, contrast=contrast,
    set_nans_to_min=set_nans_to_min,
    verbose=verbose
  )

  if data is None:
    print("WARN: Read image is None!")
    return None

  return tf.convert_to_tensor(data)


def load_img_as_pil_float(filename, resize=False, resize_size=224, apply_zscale=True, contrast=0.25, set_nans_to_min=False, verbose=False):
  """ Convert numpy array to PIL float image norm to [0,1] """

  # - Read FITS from file and get transformed npy array
  data= read_img(
    filename,
    nchans=1,
    norm_range=(0.,1.),
    resize=resize, resize_size=resize_size,
    apply_zscale=apply_zscale, contrast=contrast,
    to_uint8=False,
    set_nans_to_min=set_nans_to_min,
    verbose=verbose
  )
  if data is None:
    print("WARN: Read image is None!")
    return None

  # - Convert to PIL image
  return Image.fromarray(data)

def load_img_as_pil_rgb(filename, resize=False, resize_size=224, apply_zscale=True, contrast=0.25, set_nans_to_min=False, verbose=False):
  """ Convert numpy array to PIL 3chan RGB image norm to [0,255], uint8 """

  # - Read FITS from file and get transformed npy array
  data= read_img(
    filename,
    nchans=3,
    norm_range=(0.,255.),
    resize=resize, resize_size=resize_size,
    apply_zscale=apply_zscale, contrast=contrast,
    to_uint8=True,
    set_nans_to_min=set_nans_to_min,
    verbose=verbose
  )
  if data is None:
    print("WARN: Read image is None!")
    return None

  # - Convert to PIL RGB image
  return Image.fromarray(data).convert("RGB")
  
  
def load_encoder_model(modelfile, weightfile):
  """ Load encoder model and weights from input h5 file """

  # - Load model
  try:
    print("INFO: Loading model from file %s ..." % (modelfile))
    encoder= load_model(modelfile, compile=False)
  except Exception as e:
    print("ERROR: Failed to load model %s (err=%s)!" % (modelfile, str(e)))
    return None

  # - Load weights
  try:
    print("INFO: Loading model weights from file %s ..." % (weightfile))
    encoder.load_weights(weightfile)
  except Exception as e:
    print("ERROR: Failed to load model weights %s (err=%s)!" % (weightfile, str(e)))
    return None

  return encoder


def extract_features(datalist, model, imgsize=224, zscale=True, contrast=0.25, nmax=-1):
  """ Function to extract features from SMGPS trained models """

  # - Loop over datalist and extract features per each image
  features= []
  snames= []
  class_ids= []
  nsamples= len(datalist)

  for idx, item in enumerate(datalist):
    if nmax!=-1 and idx>=nmax:
      print("INFO: Max number of entries processed (n=%d), exit." % (nmax))
      break

    if idx%1000==0:
      print("%d/%d entries processed ..." % (idx, nsamples))

    # - Read image as numpy float image
    class_id= item['id']
    sname= item['sname']
    filename= item["filepaths"][0]
    filename_fullpath= os.path.abspath(filename)
    fileext= os.path.splitext(filename)

    image_npy= load_img_as_npy_float(
      filename_fullpath,
      add_chan_axis=True, add_batch_axis=True,
      resize=True, resize_size=imgsize,
      apply_zscale=zscale, contrast=contrast,
      set_nans_to_min=False
    )
    if image_npy is None:
      print("WARN: Failed to load image %s, skip it ..." % (filename))
      continue

    # - Run model inference
    feats= model.predict(
      x=image_npy,
      batch_size=1,
      verbose=0,
      workers=1,
      use_multiprocessing=False
    )

    # - Append features
    feats_list= list(feats[0])
    feats_list= [float(item) for item in feats_list]
    
    datalist[idx]["feats"]= feats_list
    
  return datalist


def save_features_to_ascii(datalist, outfile, save_labels=False):
	""" Save feature data to ascii file """

	# - Loop over datalist and fill lists
	features= []
	snames= []
	class_ids= []
	class_labels= []
	nsamples= len(datalist)

	for idx, item in enumerate(datalist):
		sname= item['sname']
		class_id= item['id']
		class_label= item['label']
		feats= item['feats']
    
		features.append(feats)
		snames.append(sname)
		class_ids.append(class_id)
		class_labels.append(class_label)
		
	# - Save features to ascii file with format: sname, f1, f2, ..., fn, classid
	N= len(features)
	nfeats= features[0].shape[0]
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


def jsonIndentLimit(jsonString, indent, limit):
	""" Apply limits to json indent """
	regexPattern = re.compile(f'\n({indent}){{{limit}}}(({indent})+|(?=(}}|])))')
	return regexPattern.sub('', jsonString)

def save_features_to_json(datalist, outfile, limit_indent=False):
	""" Save feat data to json """
	
	# - Use above method to limit indentation 
	if limit_indent:
		print("INFO: Limiting indentation ...")
		jsonString= json.dumps(datalist, indent=2)
		jsonString= jsonIndentLimit(jsonString, '  ', 2)
		datalist= json.loads(jsonString)
	
	# - Save to file
	print("Saving datalist to file %s ..." % (outfile))
	with open(outfile, 'w') as fp:
		json.dump(datalist, fp, indent=1)
	
	return 0
	

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
	
	nmax= args.nmax
	
	#===========================
	#==   READ DATALIST
	#===========================
	print("INFO: Read image dataset filelist %s ..." % (args.inputfile))
	datalist= read_datalist(args.inputfile, args.datalist_key)
	nfiles= len(datalist)
	
	print("INFO: %d data entries read ..." % (nfiles))
	
	#===========================
	#==   LOAD MODEL
	#===========================
	print("INFO: Loading model (path=%s, weights=%s) ..." % (args.model, args.model_weights))
	model= load_encoder_model(args.model, args.model_weights)
	if model is None:
		print("ERROR: Failed to load model and/or weights!")
		return 1
	
	#===========================
	#==   EXTRACT FEATURES
	#===========================
	print("INFO: Extracting features from file %s ..." % (args.inputfile))
	datalist_out= extract_features(
		datalist, 
		model, 
		imgsize=args.imgsize, 
		zscale=args.zscale, contrast=args.zscale_contrast, 
		nmax=args.nmax
	)
	if datalist_out is None:
	  print("ERROR: Failed to extract features!")
	  return 1
  
	#===========================
	#==   SAVE FEATURES
	#===========================
	print("INFO: Saving features to file %s ..." % (args.outfile))
	if args.save_to_json:
		save_features_to_json(
			datalist_out,
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
