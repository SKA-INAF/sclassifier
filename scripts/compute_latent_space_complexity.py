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

## ASTRO MODULES
from astropy.io import ascii
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import KBinsDiscretizer

## NON-STANDARD MODULES
import gzip

## MODULES
from sclassifier import logger
from sclassifier.utils import Utils

###########################
##     ARGS
###########################
def get_args():
	"""This function parses and return arguments passed in"""
	parser = argparse.ArgumentParser(description="Parse args.")

	# - Input options
	parser.add_argument('-inputfile','--inputfile', dest='inputfile', required=True, type=str, help='Input feature data table filename') 
	
	# - Normalization options
	parser.add_argument('-scalerfile', '--scalerfile', dest='scalerfile', required=False, type=str, default='', action='store',help='Load and use data transform stored in this file (.sav)')
	parser.add_argument('-norm_min','--norm_min', dest='norm_min', required=False, type=int, default=0, help='Data normalization min') 	
	parser.add_argument('-norm_max','--norm_max', dest='norm_max', required=False, type=int, default=1, help='Data normalizaiton max') 	
	
	# - Quantization options
	parser.add_argument('-discretizerfile', '--discretizerfile', dest='discretizerfile', required=False, type=str, default='', action='store',help='Load and use data discretizer transform stored in this file (.sav)')
	parser.add_argument('-nbins','--nbins', dest='nbins', required=False, type=int, default=256, help='Data number of bins after normalization') 	
	parser.add_argument('-strategy','--strategy', dest='strategy', required=False, type=str, default='uniform', help='Discretizer strategy (uniform, kmeans)') 	
	
	# - Output options
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='complexity.dat', help='Output filename (.dat) with selected feature data') 

	args = parser.parse_args()	

	return args


#####################################
##     PRE-PROCESSING
#####################################
def transform_data(x, norm_min=0, norm_max=1, data_scaler=None, outfile_scaler="datascaler.sav"):
	""" Transform input data here or using a loaded scaler """

	# - Print input data min/max
	x_min= x.min(axis=0)
	x_max= x.max(axis=0)

	print("== INPUT DATA MIN/MAX ==")
	print(x_min)
	print(x_max)

	if data_scaler is None:
		# - Define and run scaler
		logger.info("Define and running data scaler ...")
		data_scaler= MinMaxScaler(feature_range=(norm_min, norm_max))
		x_transf= data_scaler.fit_transform(x)

		print("== TRANSFORM DATA MIN/MAX ==")
		print(data_scaler.data_min_)
		print(data_scaler.data_max_)

		# - Save scaler to file
		logger.info("Saving data scaler to file %s ..." % (outfile_scaler))
		pickle.dump(data_scaler, open(outfile_scaler, 'wb'))
			
	else:
		# - Transform data
		logger.info("Transforming input data using loaded scaler ...")
		x_transf = data_scaler.transform(x)

	# - Print transformed data min/max
	print("== TRANSFORMED DATA MIN/MAX ==")
	x_transf_min= x_transf.min(axis=0)
	x_transf_max= x_transf.max(axis=0)
	print(x_transf_min)
	print(x_transf_max)

	return x_transf

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
	inputfile= args.inputfile
	
	# - Normalization options
	norm_min= args.norm_min
	norm_max= args.norm_max
	scalerfile= args.scalerfile

	# - Discretizer options
	discretizerfile= args.discretizerfile
	nbins= args.nbins
	strategy= args.strategy
	
	# - Output options
	outfile= args.outfile
	outfile_scaler= "datascaler.sav"
	outfile_discretizer= "databinner.sav"
	
	#===========================
	#==   READ FEATURE DATA
	#===========================
	logger.info("Reading data %s from file ..." % (inputfile))
	ret= Utils.read_feature_data(inputfile)
	if not ret:
		logger.error("Failed to read data from file %s!" % (inputfile))
		return 1

	data= ret[0]
	snames= ret[1]
	classids= ret[2]
	
	#===========================
	#==   NORMALIZE DATA
	#===========================
	# - Load scaler from file?
	data_scaler= None
	if scalerfile!="":
		logger.info("Loading data scaler from file %s ..." % (scalerfile))
		try:
			data_scaler= pickle.load(open(scalerfile, 'rb'))
		except Exception as e:
			logger.error("Failed to load data scaler from file %s!" % (scalerfile))
			return 1
	
	# - Normalize data
	logger.info("Normalizing data to range [%f,%f] ..." % (norm_min, norm_max))		
	data_norm= transform_data(
		data, 
		norm_min=norm_min, norm_max=norm_max, 
		data_scaler=data_scaler, outfile_scaler=outfile_scaler
	)
	
	#===========================
	#==   CONVERT DATA TO INT
	#===========================
	#logger.info("Converting data to int ...")
	#data_norm= (np.rint(data_norm)).astype(int)
	
	# - Load scaler from file?
	data_discretizer= None
	if discretizerfile!="":
		logger.info("Loading data discretizer from file %s ..." % (discretizerfile))
		try:
			data_discretizer= pickle.load(open(discretizerfile, 'rb'))
		except Exception as e:
			logger.error("Failed to load data discretizer from file %s!" % (discretizerfile))
			return 1

	else:
		# - Init the discretizer
		data_discretizer= KBinsDiscretizer(
			n_bins=nbins, 
			encode="ordinal", 
			strategy=strategy, 
			#random_state=0
		)
		
		# - Fitting discretizer to data
		data_discretizer.fit(data_norm)
		
	# - Apply discretizer transform to data
	#   output represents the bin id of each data
	logger.info("Apply discretizer transform to data ...")
	data_transf= data_discretizer.transform(data_norm)
	
	# - Obtain the discretized data
	logger.info("Obtain the binned data ...")
	data_binned= data_discretizer.inverse_transform(data_transf)
	
	# - Save the discretizer to file
	logger.info("Saving data discretizer to file %s ..." % (outfile_discretizer))
	pickle.dump(data_discretizer, open(outfile_discretizer, 'wb'))
	
	print("nbytes(original)=", data_norm.nbytes)
	print("nbytes(binned)=", data_binned.nbytes)
	
	#===========================
	#==   COMPUTE COMPLEXITY
	#===========================
	# - Compute complexity for each row
	logger.info("Compute complexity row-by-row ...")
	data_shape= data.shape
	nrows= data_shape[0]
	ncols= data_shape[1]
	complexities= []
	tmpfile= "data.npy.gz"
	
	for i in range(nrows):
		X= data_binned[i,:]
		
		f= gzip.GzipFile(tmpfile, "w")
		np.save(file=f, arr=X)
		f.close()
		complexity= os.path.getsize(tmpfile)
		
		complexities.append(complexity)
		print("Row %d: nbytes(original)=%f, complexity=%f" % (i, X.nbytes, complexity))
	
	# - Remove temporary file
	logger.info("Removing tmp file %s ..." % (tmpfile))
	try:
		os.remove(tmpfile)
	except OSError:
		pass
	
	# - Save data
	N= data_shape[0]
	snames= np.array(snames).reshape(N,1)
	classids= np.array(classids).reshape(N,1)
	complexities= np.array(complexities).reshape(N,1)
		
	outdata= np.concatenate(
		(snames, np.array(complexities), classids),
		axis=1
	)
	head= "# sname complexity id"

	logger.info("Save complexity data to file %s ..." % (outfile))
	Utils.write_ascii(outdata, outfile, head)	
	
	return 0

	
###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())
	
	
	

