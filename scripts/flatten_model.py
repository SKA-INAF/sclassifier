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

	parser.add_argument('-weightfile','--weightfile', dest='weightfile', required=True, type=str, help='Input .h5 weight file') 
	parser.add_argument('-modelfile','--modelfile', dest='modelfile', required=True, type=str, help='Input .h5 model file') 
	parser.add_argument('-layername','--layername', dest='layername', default='model', required=False, type=str, help='Functional layer name to select') 
	parser.add_argument('-predefined_model','--predefined_model', dest='predefined_model', default='resnet18', required=False, type=str, help='Predefined model to create') 
	parser.add_argument('-outfile_weights','--outfile_weights', dest='outfile_weights', default='model_weights_new.h5', required=False, type=str, help='Output filename where to store new model weights') 
	
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

	weightfile= args.weightfile
	modelfile= args.modelfile
	layername= args.layername
	predefined_model= args.predefined_model
	outfile_weights= args.outfile_weights
	
	#===========================
	#==   LOAD MODEL
	#===========================
	# - Load model
	logger.info("Loading model ...")
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
	try:
		model.load_weights(weightfile)
	except Exception as e:
		logger.error("Failed to load weights from file %s (err=%s)!" % (weightfile, str(e)))
		return 1
			
	#==================================
	#==   GET SELECTED LAYER WEIGHTS
	#==================================
	logger.info("Retrieving layer %s weights ..." % (layername))
	try:
		weights= model.get_layer(layername).get_weights()
		print("type(weights)")
		print(type(weights))
	except Exception as e:
		logger.error("Failed to retrieve layer %s weights!" % (layername))
		return 1
	
	#==================================
	#==   CREATING NEW MODEL
	#==================================
	s= model.layers[0].input_shape[0]
	print(s)
	
	inputShape = (s[1], s[2], s[3])
	inputs= Input(shape=inputShape, dtype='float', name='inputs')
	
	try:
		model_new= Classifiers.get(predefined_model)[0](
			include_top=False,
			weights=None, 
			input_tensor=inputs, 
			input_shape=inputShape,
		)
	except Exception as e:
		logger.error("Failed to build new model %s (err=%s)!" % (predefined_model, str(e)))
		return 1
		
	model_new.summary()
		
	#=======================================
	#==   LOAD WEIGHT FROM SELECTED LAYER
	#=======================================
	# - Save current weights (random weights)
	initial_weights = [layer.get_weights() for layer in model_new.layers]
	
	# - Load weights from the other model
	logger.info("Loading weights in new model from selected layer ...")
	try:
		model_new.set_weights(weights)
	except Exception as e:
		logger.error("Failed to set weights in new model (err=%s)!" % (str(e)))
		return 1
		
	# - Check weights effectively changed
	logger.info("Check weights effectively changed ...")
	failed= False
	for layer, initial in zip(model_new.layers, initial_weights):
		weights_curr= layer.get_weights()
		if weights_curr and all(tf.nest.map_structure(np.array_equal, weights_curr, initial)):
			logger.warn("Given weight file %s contained no weights for layer %s ..." % (weights, layer.name))
			failed= True
	
	if failed:
		logger.error("Weights not changed wrt original weights!")
		return 1
		
	# - Save new model weights
	logger.info("Saving new model weights to file %s ..." % (outfile_weights))
	model_new.save_weights(outfile_weights)
	
	#print("Retrieve functional model layer ")
	#pretrained_model= model.get_layer("model")
	
	#print("Create flatten model ...")
	#model_flat2= keras.Model(
	#	[model.inputs],   # This is the Input() you've created yourself
	#	[pretrained_model.inbound_nodes[0].output_tensors,  # This is output created when you called `pretrained_model()` on your own `Input()`
	#	model.output]  # This is the output of the final `Dense()` layer you created
	#)
	
	#print("Printing flattened model ...")
	#model_flat2.summary()
	
	#layername_added= []
	#layers_flat = []
	#for layer in model.layers:
	#	try:
	#		layers_flat.extend(layer.layers)
	#	except AttributeError:
	#		if layer.name in layername_added:
	#			continue
	#		layers_flat.append(layer)
	#		layername_added.append(layer.name)

	#model_flat3 = models.Sequential(layers_flat)
	#print("Printing flattened model 3 ...")
	#model_flat3.summary()
	
	#model_flat = models.Sequential()
	#layername_added= []
	
	#for layer in model.layers:
	#	if layername!="" and layer.name!=layername:
	#		continue
	#	#if hasattr(layer, 'layers'): # nested layer
	#	if "Functional" == layer.__class__.__name__:
  #       
	#		for nested_layer in layer.layers:
	#			if nested_layer.name in layername_added:
	#				continue
	#			print("Adding layer %s ..." % (nested_layer.name))
	#			model_flat.add(nested_layer)
	#			layername_added.append(nested_layer.name)
	#			
	#			#weights= nested_layer.get_weights()
	#			#print("weights")
	#			#print(type(weights))
	#			#print(weights)
	#	#else:
	#		#if layer.name in layername_added:
	#		#	continue
	#		#print("Adding layer %s ..." % (layer.name))
	#		#model_flat.add(layer)
	#		#layername_added.append(layer.name)
#
#
#	print("Printing flat model ...")

	#model_flat.summary()

	#for layer_nested in model.get_layer('nested_model').layers:
  #  model_flat.add(layer_nested)		
	
	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())  
    
