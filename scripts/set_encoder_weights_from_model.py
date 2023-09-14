#!/usr/bin/env python

from __future__ import print_function


##################################################
###          MODULE IMPORT
##################################################
## STANDARD MODULES
import os
import sys
import getopt
import argparse
import h5py
import numpy as np

## TF MODULE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from classification_models.tfkeras import Classifiers

from keras.saving.legacy.hdf5_format import load_attributes_from_hdf5_group

from sclassifier.tf_utils import SoftmaxCosineSim
from sclassifier.feature_extractor_simclr import WarmUpCosineDecay

## MODULE IMPORT
from sclassifier import logger

###########################
##     ARGS
###########################
def get_args():
	"""This function parses and return arguments passed in"""
	parser = argparse.ArgumentParser(description="Parse args.")

	parser.add_argument('-modelfile','--modelfile', dest='modelfile', required=True, type=str, help='Input .h5 model file') 
	parser.add_argument('-weightfile','--weightfile', dest='weightfile', required=True, type=str, help='Input .h5 weight file from which to extract encoder weights') 
	parser.add_argument('-modelfile_encoder','--modelfile_encoder', dest='modelfile_encoder', required=True, type=str, help='Input .h5 encoder model file') 
	parser.add_argument('-weightfile_encoder','--weightfile_encoder', dest='weightfile_encoder', required=True, type=str, help='Input .h5 encoder weight file that are to be updated') 
	
	parser.add_argument('-layername','--layername', dest='layername', default='base_model', required=False, type=str, help='Functional layer name to select') 
	parser.add_argument('-outfile_weights','--outfile_weights', dest='outfile_weights', default='encoder_weights_new.h5', required=False, type=str, help='Output filename where to store new encoder model weights') 
	
	args = parser.parse_args()	

	return args
	
###########################
##     READ WEIGHTS
###########################
def read_hdf5(path):
	""" Read weight file """

	weights = {}
	keys = []
	with h5py.File(path, 'r') as f: # open file
		print("f.attrs")
		print(f.attrs)
		layer_names= load_attributes_from_hdf5_group(f, "layer_names")
		print("layer_names")
		print(layer_names)
		for k, name in enumerate(layer_names):
			print(k)
			print(name)
		
		f.visit(keys.append) # append all keys to list
		for key in keys:
			if ':' in key: # contains data if ':' in key
				print(f[key].name)
				#weights[f[key].name] = f[key].value
				weights[f[key].name] = f[key][()]
    
	return weights
	
def get_non_trainable_model(model):
	""" Set each layer as non-trainable """
		
	for layer in model.layers:
		if hasattr(layer, 'layers'): # nested layer
			for nested_layer in layer.layers:
				nested_layer.trainable= False
		else:
			layer.trainable = False

	return model

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

	modelfile= args.modelfile
	weightfile= args.weightfile
	modelfile_encoder= args.modelfile_encoder
	weightfile_encoder= args.weightfile_encoder
	layername= args.layername
	outfile_weights= args.outfile_weights
	
	#===========================
	#==   LOAD MODEL
	#===========================
	# - Load model
	logger.info("Loading model from file %s ..." % (modelfile))
	model = load_model(modelfile, compile=False, custom_objects={'SoftmaxCosineSim': SoftmaxCosineSim, 'WarmUpCosineDecay': WarmUpCosineDecay})
	
	layer_names= []
	for layer in model.layers:
		layer_names.append(layer.name)
		
	model.summary()
	model.summary(expand_nested=True)
		
	print("layer_names")
	print(layer_names)
	
	#===========================
	#==   LOAD WEIGHTS
	#===========================
	# - Load weights
	logger.info("Loading model weights from file %s ..." % (weightfile))
	try:
		model.load_weights(weightfile)
	except Exception as e:
		logger.error("Failed to load weights from file %s (err=%s)!" % (weightfile, str(e)))
		return 1
		
	#===========================
	#==   LOAD ENCODER MODEL
	#===========================
	# - Load encoder model
	logger.info("Loading encoder model from file %s ..." % (modelfile_encoder))
	encoder_model = load_model(modelfile_encoder, compile=False, custom_objects={'SoftmaxCosineSim': SoftmaxCosineSim, 'WarmUpCosineDecay': WarmUpCosineDecay})
	
	layer_names= []
	for layer in model.layers:
		layer_names.append(layer.name)
		
	encoder_model.summary()
	encoder_model.summary(expand_nested=True)
		
	print("layer_names")
	print(layer_names)	
		
	#===========================
	#==   LOAD ENCODER WEIGHTS
	#===========================
	# - Load encoder weights
	logger.info("Loading encoder model weights from file %s ..." % (weightfile_encoder))
	try:
		encoder_model.load_weights(weightfile_encoder)
	except Exception as e:
		logger.error("Failed to load encoder weights from file %s (err=%s)!" % (weightfile_encoder, str(e)))
		return 1
		
	# - Get encoder loaded weight
	#logger.info("Retrieving encoder loaded weights ...")
	#try:
	#	encoder_weights= encoder_model.get_weights()
	#	print("type(weights)")
	#	print(type(weights))
	#except Exception as e:
	#	logger.error("Failed to retrieve loaded encoder weights!")
	#	return 1
		
		
	#======================================
	#==   GET ENCODER WEIGHTS FROM MODEL
	#======================================
	logger.info("Retrieving layer %s weights from model ..." % (layername))
	try:
		encoder_weights_new= model.get_layer(layername).get_weights()
		print("type(encoder_weights_new)")
		print(type(encoder_weights_new))
	except Exception as e:
		logger.error("Failed to retrieve layer %s weights from model!" % (layername))
		return 1
	
	
		
	#=======================================
	#==   UPDATE ENCODER WEIGHTS
	#=======================================
	# - Save current weights (original encoder weights loaded from input encoder weight file)
	original_weights = [layer.get_weights() for layer in encoder_model.layers]
	
	# - Update encoder weights
	logger.info("Update encoder weights ...")
	try:
		encoder_model.set_weights(encoder_weights_new)
	except Exception as e:
		logger.error("Failed to set updated weights in encoder model (err=%s)!" % (str(e)))
		return 1
		
	# - Check weights effectively changed
	logger.info("Check weights effectively changed ...")
	failed= False
	for layer, initial in zip(encoder_model.layers, original_weights):
		weights_curr= layer.get_weights()
		if weights_curr and all(tf.nest.map_structure(np.array_equal, weights_curr, initial)):
			logger.warning("Original and new weights for layer %s are equal ..." % (layer.name))
			failed= True
	
	if failed:
		logger.error("Weights not changed wrt original weights!")
		return 1
		
	# - Save new model weights
	logger.info("Saving new model weights to file %s ..." % (outfile_weights))
	encoder_model.save_weights(outfile_weights)
	
		
	
	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())  
    
