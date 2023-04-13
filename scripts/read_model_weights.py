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

## MODULE IMPORT
from sclassifier import logger

###########################
##     ARGS
###########################
def get_args():
	"""This function parses and return arguments passed in"""
	parser = argparse.ArgumentParser(description="Parse args.")

	# - Input options
	parser.add_argument('-weightfile','--weightfile', dest='weightfile', required=True, type=str, help='Input .h5 weight file') 
	parser.add_argument('-modelfile','--modelfile', dest='modelfile', required=True, type=str, help='Input .h5 model file') 
	
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

	# - Input filelist
	weightfile= args.weightfile
	modelfile= args.modelfile
	
	#===========================
	#==   LOAD MODEL
	#===========================
	logger.info("Loading model ...")
	model = load_model(modelfile, compile=False)
	
	layer_names= []
	for layer in model.layers:
		layer_names.append(layer.name)
		
		
	print("layer_names")
	print(layer_names)
	
	try:
		model.load_weights(weightfile)
	except Exception as e:
		logger.warning("Failed to load weights from file %s (err=%s), retrying to load by layer name ..." % (weightfile, str(e)))
		try:
			model.load_weights(weightfile, by_name=True)
		except Exception as e:
			logger.error("Failed to load weights from file %s by name (err=%s), giving up!" % (weightfile, str(e)))	
			return -1
			
	logger.info("Making backbone non-trainable ...")
	for layer in model.layers:
		layer.trainable = False

	model.summary()
			
	#===========================
	#==   READ WEIGHTS
	#===========================
	logger.info("Reading weights %s ..." % (weightfile))
	weights = read_hdf5(weightfile)
	
	#===========================
	#==   SET NON TRAINABLE
	#===========================
	logger.info("Get non-trainable weights ...")
	model= get_non_trainable_model(model)
	
	model.summary()
	
	#===========================
	#==   LOAD WEIGHTS
	#===========================
	#inputShape = (224, 224, 3)
	#inputs= Input(shape=inputShape, dtype='float', name='inputs')
	#predefined_arch= "resnet18"
	
	# - Load model
	#logger.info("Creating backbone model ...")
	#try:
	#	backbone_model= Classifiers.get(predefined_arch)[0](
	#		include_top=False,
	#		weights=None, 
	#		input_tensor=inputs, 
	#		input_shape=inputShape,
	#	)
	#except Exception as e:
	#	logger.error("Failed to build base encoder %s (err=%s)!" % (predefined_arch, str(e)))
	#	return -1
		
	#layer_names= []
	#for layer in backbone_model.layers:
	#	layer_names.append(layer.name)
  	
	#print("layer_names (backbone)")
	#print(layer_names)
		
	#x= backbone_model(inputs)
	#x= layers.GlobalAveragePooling2D()(x) 
	#base_model= Model(inputs, x, name='base_model')
	
	# - Load weights
	#try:
	#	base_model.load_weights(weightfile)
	#	##backbone_model.load_weights(weightfile)
	#except Exception as e:
	#	logger.warning("Failed to load weights from file %s (err=%s), retrying to load by layer name ..." % (weightfile, str(e)))
	#	try:
	#		base_model.load_weights(weightfile, by_name=True)
	#		#backbone_model.load_weights(weightfile, by_name=True)	
	#	except Exception as e:
	#		logger.error("Failed to load weights from file %s by name (err=%s), giving up!" % (weightfile, str(e)))	
	#		return -1
		
	# - Create a new model to expand nested layers
	#flat_model= models.Sequential()
	
	#layer_counter= 0
	#layer_names= []
	
	#for i, layer in enumerate(model.layers):
	#	if hasattr(layer, 'layers'):
	#		for nested_layer in layer.layers:
	#			print("layer.name=%s" % (nested_layer.name))
	#			flat_model.add(nested_layer)
	#			#flat_model.layers[layer_counter].set_weights(nested_layer.get_weights())
	#			layer_counter+= 1
	#			layer_names.append(nested_layer.name)
	#	else:
	#		print("layer.name=%s" % (layer.name))
	#		print(type(layer))
	#		print("layer_names")
	#		print(layer_names)
	#		if layer.name in layer_names:
	#			logger.info("Layer %s was already added, skip it ..." % (layer.name)) 
	#			continue	
	#		
	#		if isinstance(layer, tf.keras.layers.InputLayer):
	#			print("--> input layer")
	#			print(layer.input_shape)
	#			print(layer.dtype)
	#			print(layer.name)
	#			
	#			inputs= Input(shape=(layer.input_shape[0][1],layer.input_shape[0][2],layer.input_shape[0][3]), dtype=layer.dtype, name="inputs")
	#				
	#			logger.info("Adding layer %s ..." % (layer.name))
	#			flat_model.add(inputs)
	#			layer_counter+= 1
	#			layer_names.append(layer.name)
	#		else:
	#			print("non input")
	#			
	#			logger.info("Adding layer %s ..." % (layer.name))
	#			flat_model.add(layer)
	#			layer_counter+= 1
	#			layer_names.append(layer.name)
	#			
	#		#flat_model.layers[layer_counter].set_weights(layer.get_weights())
	#		
	#	print("flat_model.layers")
	#	print(flat_model.layers)
	#		
	#print("layer_counter=%d" % (layer_counter))
	
	#flat_model.summary()
	
	#for i in range(len(model.layers)):
  #if model.layers[i].name not in new_layers:
  #model.layers[i].set_weights(model_pretrained.layers[i].get_weights())
	
	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())


if __name__ == '__main__':
    filename = 'comp_named.hdf5'
    
    
