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

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections
import csv
import json
import pickle

## PYTORCH
import torch
import torchvision.transforms as T
from PIL import Image

## ASTRO/IMG PROCESSING MODULES
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip
from astropy.visualization import ZScaleInterval

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
	parser.add_argument('--imgsize', default=224, type=int, help='Image resize size in pixels')
	parser.add_argument('--clip_data', dest='clip_data', action='store_true',help='Apply sigma clipping transform (default=false)')	
	parser.set_defaults(clip_data=False)
	parser.add_argument('--zscale', dest='zscale', action='store_true',help='Apply zscale transform (default=false)')	
	parser.set_defaults(zscale=False)
	parser.add_argument('--zscale_contrast', default=0.25, type=float, help='ZScale transform contrast (default=0.25)')
	
	parser.add_argument('--norm_min', default=0., type=float, help='Norm min (default=0)')
	parser.add_argument('--norm_max', default=1., type=float, help='Norm max (default=1)')
	parser.add_argument('--to_uint8', dest='to_uint8', action='store_true',help='Convert to uint8 (default=false)')	
	parser.set_defaults(to_uint8=False)
	parser.add_argument('--set_zero_to_min', dest='shift_zero_to_min', action='store_true',help='Set blank pixels to min>0 (default=false)')	
	parser.set_defaults(set_zero_to_min=False)
	parser.add_argument('--in_chans', default = 1, type = int, help = 'Length of subset of dataset to use.')
	
	# - Model option
	parser.add_argument('-model','--model', dest='model', required=False, type=str, default="dinov2_vits14", help='DINO v2 pretrained model {dinov2_vits14,dinov2_vits14_reg,dinov2_vitl14,dinov2_vitl14_reg}') 
	
	# - Run options
	parser.add_argument('-device','--device', dest='device', required=False, type=str, default="cuda", help='Device where to run inference. Default is cuda, if not found use cpu.') 
	
	# - Outfile option
	parser.add_argument('--save_to_json', dest='save_to_json', action='store_true',help='Save features to json (default=save to ascii)')
	parser.set_defaults(save_to_json=False)
	parser.add_argument('--save_labels_in_ascii', dest='save_labels_in_ascii', action='store_true',help='Save class labels to ascii (default=save classids)')
	parser.set_defaults(save_labels_in_ascii=False)
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='featdata.dat', help='Output filename (.dat) of feature data') 
	
	args = parser.parse_args()	

	return args
	

	
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

	print("== DATA MIN/MAX ==")
	print(data_transf.min())
	print(data_transf.max())

	# - Clip data?
	if args.clip_data:
		data_clipped= get_clipped_data(data_transf, sigma_low=5, sigma_up=30)
		data_transf= data_clipped

	# - Apply zscale stretch
	if args.zscale:
		print("Apply zscale stretch ...")
		data_stretched= get_zscaled_data(data_transf, contrast=args.zscale_contrast)
		data_transf= data_stretched

	# - Convert to uint8
	#data_transf= (data_transf*255.).astype(np.uint8)
		
	# - Normalize to range
	data_min= data_transf.min()
	data_max= data_transf.max()
	norm_min= args.norm_min
	norm_max= args.norm_max
	if norm_min==data_min and norm_max==data_max:
		print("INFO: Data already normalized in range (%f,%f)" % (norm_min, norm_max))
	else:
		data_norm= (data_transf-data_min)/(data_max-data_min) * (norm_max-norm_min) + norm_min
		data_transf= data_norm
			
	print("== DATA MIN/MAX (AFTER TRANSF) ==")
	print(data_transf.min())
	print(data_transf.max())
	
	# - Convert to uint8
	if args.to_uint8:
		data_transf= data_transf.astype(np.uint8)
	
	return data_transf


def read_img(filename, args):
	""" Read fits image """

	# - Check filename
	if filename=="":
		return None
	
	file_ext= os.path.splitext(filename)[1]
		
	# - Read fits image
	if file_ext=='.fits':
		data= fits.open(filename)[0].data
	elif fileext in ['.png', '.jpg']:
		image= Image.open(filename)
		data= np.asarray(image)
	else:
		print("ERROR: Invalid or unrecognized file extension (%s)!" % (fileext))
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

	print("--> image.shape")
	print(np.asarray(image).shape)	
		
	# - Apply other transforms (e.g. resize, model-specific transforms)
	transform = T.Compose(
  	[
  	  T.Resize(args.imgsize, interpolation=T.InterpolationMode.BICUBIC),
  	  T.ToTensor(),
  	  #T.Normalize(mean=[0.5], std=[0.5])
  	  T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
  	]
	)
	
	image_tensor= transform(image)[:3].unsqueeze(0)
	
	return image_tensor
	
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


def extract_features(datalist, model, args):
	""" Function to extract features from trained models """
	
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
		sname= datalist[idx]["sname"]
		class_id= datalist[idx]["id"]
		
		print("INFO: Reading image %s ..." % (filename))
		image_tensor= read_img(filename, args)
		if image_tensor is None:
			print("WARN: Read/processed image %s is None, skip to next!" % (filename))
			continue
		
		# - Extract model prediction
		with torch.no_grad():
			features = model(image_tensor.to(device))[0]
			##features = model(image_tensor.to("cuda"))[0]
	
		features_numpy= features.cpu().numpy()
		
		if i==0:
			print("features.shape")
			print(features.shape)
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
	
	inputfile= args.inputfile
	model_id= args.model
	outfile= args.outfile
	
	device= args.device
	if "cuda" in device:
		if not torch.cuda.is_available():
			print("WARN: cuda not available, using cpu...")
			device= "cpu"
	
	print('device:',device)
		
	#===========================
	#==   READ DATALIST
	#===========================
	print("INFO: Read image dataset filelist %s ..." % (inputfile))
	fp= open(inputfile, "r")
	datalist= json.load(fp)[datalist_key]
	nfiles= len(datalist)
	
	#===========================
	#==   LOAD MODEL
	#===========================
	# - Load model
	print("INFO: Loading model %s ..." % (model_id))
	model= torch.hub.load('facebookresearch/dinov2', model_id)
	model.to(device)

	#===========================
	#==   EXTRACT FEATURES
	#===========================
	print("INFO: Extracting features from file %s ..." % (args.inputfile))
	datalist_out= extract_features(
		datalist, 
		model, 
		args
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
