#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
## STANDARD MODULES
import sys
import os
import numpy as np
import getopt
import argparse
import collections
import csv
import pickle
import json

## ASTRO MODULES
from astropy.io import ascii
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import KBinsDiscretizer

## NON-STANDARD MODULES
import gzip

## DRAW MODULES
import matplotlib.pyplot as plt

## MODULES
from sclassifier import logger
from sclassifier.utils import Utils
from sclassifier.preprocessing import DataPreprocessor
from sclassifier.preprocessing import ChanResizer, MinMaxNormalizer, Resizer, PercentileThresholder, MedianFilterer

###########################
##     ARGS
###########################
def get_args():
	"""This function parses and return arguments passed in"""
	parser = argparse.ArgumentParser(description="Parse args.")
	
	# - Input options
	parser.add_argument('-inputfile','--inputfile', dest='inputfile', required=False, type=str, default='', help='Input image file') 
	parser.add_argument('-filelist','--filelist', dest='filelist', required=False, type=str, default='', help='Input filelist with list of image files') 
	parser.add_argument('--jsonlist', dest='jsonlist', action='store_true', help='Treat input file as json filelist (default=no)')	
	parser.set_defaults(jsonlist=False)
	
	# - Transform options
	parser.add_argument('--resize', dest='resize', action='store_true', help='Resize images to resize_size (default=no)')	
	parser.set_defaults(resize=False)
	parser.add_argument('-resize_size', '--resize_size', dest='resize_size', required=False, type=int, default=64, action='store',help='Image resize in pixels (default=64)')	
	parser.add_argument('--downscale_with_antialiasing', dest='downscale_with_antialiasing', action='store_true', help='Use anti-aliasing when downsampling the image (default=no)')	
	parser.set_defaults(downscale_with_antialiasing=False)
	parser.add_argument('--upscale', dest='upscale', action='store_true', help='Upscale images to resize size when source size is smaller (default=no)')	
	parser.set_defaults(upscale=False)
	parser.add_argument('--set_pad_val_to_min', dest='set_pad_val_to_min', action='store_true', help='Set masked value in resized image to min, otherwise leave to masked values (default=no)')	
	parser.set_defaults(set_pad_val_to_min=False)
	
	parser.add_argument('-percentile_thr', '--percentile_thr', dest='percentile_thr', required=False, type=float, default=50, action='store',help='Percentile threshold (default=50)')
	parser.add_argument('-median_filt_size', '--median_filt_size', dest='median_filt_size', required=False, type=int, default=3, action='store',help='Median filter window size in pixels (default=3)')	
	
	# - Output options
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='complexity.dat', help='Output filename (.dat) with selected feature data') 

	# - Draw options
	parser.add_argument('--draw', dest='draw', action='store_true', help='Draw image (default=no)')	
	parser.set_defaults(draw=False)
	

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
		
	resize= args.resize
	resize_size= args.resize_size
	downscale_with_antialiasing= args.downscale_with_antialiasing
	upscale= args.upscale
	set_pad_val_to_min= args.set_pad_val_to_min
	percentile_thr= args.percentile_thr
	median_filt_size= args.median_filt_size
	
	outfile= args.outfile
	draw= args.draw
	
	# - Check args
	if args.inputfile=='':
		if args.filelist=='':
			logger.error("Missing both inputfile & filelist args, specicy one!")
			return 1
		else:
			filelist= []
			logger.info("Reading files from filelist %s ..." % (args.filelist))
			with open(args.filelist,'r') as f:
				for filename in f:
					filename = filename.strip()
					filelist.append(filename)
	else:
		if args.jsonlist:
			f= open(args.inputfile, "r")
			d= json.load(f)["data"]
			
			filelist= []
			for item in d:
				filename= item["filepaths"][0]
				sname= item["sname"]
				filelist.append(filename)
				
		else:
			filelist= [args.inputfile]
			
	print("filelist")
	print(str(filelist))
	
	#===============================
	#==  CREATE DATA PRE-PROCESSOR
	#===============================
	# - Pre-process stage order
	#   1) PercentileThresholder
	#   2) MedianFilterer
	#   3) Resize
	#   4) Min/max norm
	logger.info("Create train data pre-processor ...")
	preprocess_stages= []

	#preprocess_stages.append(ChanResizer(nchans=3))
	#preprocess_stages.append(PercentileThresholder(percthr=percentile_thr))
	#preprocess_stages.append(MedianFilterer(size=median_filt_size))
	#preprocess_stages.append(Resizer(resize_size=resize_size, upscale=upscale, downscale_with_antialiasing=downscale_with_antialiasing, set_pad_val_to_min=set_pad_val_to_min))
	#preprocess_stages.append(MinMaxNormalizer(norm_min=0.0, norm_max=1.0))
	
	preprocess_stages.append(ChanResizer(nchans=3))
	if resize:
		preprocess_stages.append(Resizer(resize_size=resize_size, upscale=upscale, downscale_with_antialiasing=downscale_with_antialiasing, set_pad_val_to_min=set_pad_val_to_min))
	preprocess_stages.append(PercentileThresholder(percthr=percentile_thr))
	preprocess_stages.append(MedianFilterer(size=median_filt_size))
	preprocess_stages.append(MinMaxNormalizer(norm_min=0.0, norm_max=1.0))
	
	print("== PRE-PROCESSING STAGES (TRAIN) ==")
	print(preprocess_stages)

	dp= DataPreprocessor(preprocess_stages)
	
	#===========================
	#==   READ FILELIST
	#===========================
	complexities= []
	tmpfile= "data.npy.gz"
	
	logger.info("Reading data ...")
	for filename in filelist:
		# - Read image
		logger.info("Reading image file %s ..." % (filename))
		fileext= os.path.splitext(filename)[1]
		if fileext=='.fits':
			data, header, wcs= Utils.read_fits(filename, strip_deg_axis=True)
		elif fileext in ['.png', '.jpg']:
			data= Utils.read_image(filename)
		else:
			logger.error("Invalid or unrecognized file extension (%s), exit!" % (fileext))
			return 1

		# - Find image min/max and convert tp uint8
		#cond= np.logical_and(data!=0, np.isfinite(data))
		#data_1d= data[cond]
		#data_min= data_1d.min()
		#data_max= data_1d.max()
		#logger.info("Converting data to uint8 ...")
		#data_norm= (data-data_min)/(data_max-data_min) * 255.
		#data_norm[~cond]= 0 # Restore 0 and nans set in original data
		#data= data_norm.astype(np.uint8)
		
		# - Apply transformation
		logger.info("Applying data transformation ...")
		data_transf= dp(data)
		
		# - Convert to uint8 after other transformations (Segal et al 2019 do this step before, see Section 3.1)
		logger.info("Converting data to uint8 ...")
		X= data_transf[:,:,0]
		X*= 255
		X= X.astype(np.uint8)
		
		# - Draw?
		if draw:
			logger.info("Drawing image after processing ...")
			plt.imshow(X, origin='lower')
			plt.show()
		
		# - Save array to file
		fout= gzip.GzipFile(tmpfile, "w")
		np.save(file=fout, arr=X)
		fout.close()
		complexity= os.path.getsize(tmpfile)
		
		complexities.append(complexity)
		logger.info("Image file %s: nbytes(original)=%f, complexity=%f" % (filename, X.nbytes, complexity))
		
	# - Remove temporary file
	logger.info("Removing tmp file %s ..." % (tmpfile))
	try:
		os.remove(tmpfile)
	except OSError:
		pass
	
	# - Save data
	N= len(complexities)
	filenames= np.array(filelist).reshape(N,1)
	#snames= np.array(snames).reshape(N,1)
	#classids= np.array(classids).reshape(N,1)
	complexities= np.array(complexities).reshape(N,1)
		
	outdata= np.concatenate(
		(filenames, complexities),
		axis=1
	)
	head= "# filename complexity"

	logger.info("Save complexity data to file %s ..." % (outfile))
	Utils.write_ascii(outdata, outfile, head)	
		
	return 0

	
###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())
		
		
