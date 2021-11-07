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

##############################
##     GLOBAL VARS
##############################
#logger = logging.getLogger(__name__)
from sclassifier_vae import logger

## TENSORFLOW & KERAS MODULES
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
try:
	from tensorflow.keras.utils import plot_model
except:
	from tensorflow.keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
try:
	from tensorflow.keras.layers.normalization import BatchNormalization
except Exception as e:
	logger.warn("Failed to import BatchNormalization (err=%s), trying in another way ..." % str(e))
	from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.losses import mse, binary_crossentropy



#import keras
#from keras import layers
#from keras import models
#from keras import optimizers
#try:
#	from keras.utils import plot_model
#except:
#	from keras.utils.vis_utils import plot_model
#from keras import backend as K
#from keras.models import Model
#from keras.models import load_model
#try:
#	from keras.layers.normalization import BatchNormalization
#except Exception as e:
#	logger.warn("Failed to import BatchNormalization (err=%s), trying in another way ..." % str(e))
#	from keras.layers import BatchNormalization
#from keras.layers.convolutional import Conv2D
#from keras.layers.convolutional import MaxPooling2D
#from keras.layers.core import Activation
#from keras.layers.core import Dropout
#from keras.layers.core import Lambda
#from keras.layers.core import Dense
#from keras.layers import Flatten
#from keras.layers import Input
#from keras.utils.generic_utils import get_custom_objects
#import tensorflow as tf
#from keras.losses import mse, binary_crossentropy


from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

## GRAPHICS MODULES
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

## PACKAGE MODULES
from .utils import Utils
from .data_loader import DataLoader
from .data_loader import SourceData




class Sampling(layers.Layer):
	"""Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

	def call(self, inputs):
		z_mean, z_log_var = inputs
		batch = tf.shape(z_mean)[0]
		dim = tf.shape(z_mean)[1]
		epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
		return z_mean + tf.exp(0.5 * z_log_var) * epsilon

##############################
##     VAEClassifier CLASS
##############################
class VAEClassifier(object):
	""" Class to create and train a VAE classifier

			Arguments:
				- encoder_nnarc_file: File with encoder network architecture to be created
				- decoder_nnarc_file: File with decoder network architecture to be created
				- DataLoader class
	"""
	
	def __init__(self, data_loader):
		""" Return a Classifer object """

		# - Input data
		self.encoder_nnarc_file= ''
		self.decoder_nnarc_file= ''
		self.dl= data_loader

		# - Train data	
		self.nsamples= 0
		self.nx= 128 
		self.ny= 128
		self.nchannels= 0
		self.inputs= None	
		self.inputs_train= None
		self.input_labels= {}
		self.source_names= []
		self.flattened_inputs= None	
		self.input_data_dim= 0
		self.encoded_data= None
		self.train_data_generator= None
		self.crossval_data_generator= None
		self.test_data_generator= None
		self.augmentation= False	
		self.validation_steps= 10
		self.use_multiprocessing= True
		self.nworkers= 1
		
		# - NN architecture
		self.use_vae= False # create variational autoencoder, otherwise standard autoencoder
		self.fitout= None		
		self.vae= None
		self.encoder= None
		self.decoder= None	
		self.nfilters_cnn= [32,64,128]
		self.kernsizes_cnn= [3,5,7]
		self.strides_cnn= [2,2,2]
		self.add_max_pooling= False
		self.pool_size= 2
		self.add_leakyrelu= False
		self.leakyrelu_alpha= 0.2
		self.add_batchnorm= True
		self.activation_fcn_cnn= "relu"

		self.add_dense= False
		self.dense_layer_sizes= [16] 
		self.dense_layer_activation= 'relu'
		
		self.decoder_output_layer_activation= 'sigmoid'
		self.rec_loss_weight= 0.5
		self.kl_loss_weight= 0.5
		self.z_mean = None
		self.z_log_var = None
		self.z = None
		self.shape_before_flattening= 0
		self.batch_size= 32
		self.latent_dim= 2
		self.nepochs= 10
		
		self.learning_rate= 1.e-4
		self.optimizer_default= 'adam'
		self.optimizer= 'adam' # 'rmsprop'
		self.use_mse_loss= False

		self.weight_init_seed= None
		self.shuffle_train_data= True
		
		# - Draw options
		self.marker_mapping= {
			'UNKNOWN': 'o', # unknown
			'MIXED_TYPE': 'X', # mixed type
			'STAR': 'x', # star
			'GALAXY': 'D', # galaxy
			'PN': '+', # PN
			'HII': 's', # HII
			'YSO': 'P', # YSO
			'QSO': 'v', # QSO
			'PULSAR': 'd', # pulsar
		}

		self.marker_color_mapping= {
			'UNKNOWN': 'k', # unknown
			'MIXED_TYPE': 'tab:gray', # mixed type
			'STAR': 'r', # star
			'GALAXY': 'm', # galaxy
			'PN': 'g', # PN
			'HII': 'b', # HII
			'YSO': 'y', # YSO
			'QSO': 'c', # QSO
			'PULSAR': 'tab:orange', # pulsar
		}
		

		# - Output data
		self.outfile_loss= 'losses.png'
		self.outfile_accuracy= 'accuracy.png'
		self.outfile_model= 'model.png'
		self.outfile_nnout_metrics= 'losses.dat'
		self.outfile_encoded_data= 'latent_data.dat'

	#####################################
	##     SETTERS/GETTERS
	#####################################
	def set_image_size(self,nx,ny):
		""" Set image size """	
		self.nx= nx
		self.ny= ny

	def set_optimizer(self, opt, learning_rate=None):
		""" Set optimizer """

		if learning_rate is None or learning_rate<=0:
			self.optimizer= opt
		else:
			if opt=="rmsprop":
				self.optimizer= tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
			elif opt=="adam":	
				self.optimizer= tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
			else:
				logger.warn("Unknown optimizer selected (%s), setting to the default (%s) ..." % (opt, self.optimizer_default))
				self.optimizer= self.optimizer_default
		
	def set_reproducible_model(self):
		""" Set model in reproducible mode """

		logger.info("Set reproducible model ...")

		# - Fix numpy and tensorflow seeds
		#np.random.seed(1)
		#tf.set_random_seed(2)
		
		# - Do not shuffle data during training
		self.shuffle_train_data= False

		# - Initialize weight to same array
		if self.weight_init_seed is None:
			self.weight_init_seed= 1
		
	
	#####################################
	##     SET TRAIN DATA
	#####################################
	def __set_data(self):
		""" Set train data & generator from loader """

		# - Retrieve info from data loader
		self.nchannels= self.dl.nchannels
		self.source_labels= self.dl.labels
		self.source_ids= self.dl.classids
		self.source_names= self.dl.snames
		self.nsamples= len(self.source_labels)

		# - Create train data generator
		self.train_data_generator= self.dl.data_generator(
			batch_size=self.batch_size, 
			shuffle=self.shuffle_train_data,
			resize=True, nx=self.nx, ny=self.ny, 
			normalize=True, 
			augment=self.augmentation
		)	

		# - Create cross validation data generator
		self.crossval_data_generator= self.dl.data_generator(
			batch_size=self.batch_size, 
			shuffle=self.shuffle_train_data,
			resize=True, nx=self.nx, ny=self.ny, 
			normalize=True, 
			augment=self.augmentation
		)	

		# - Create test data generator
		self.test_data_generator= self.dl.data_generator(
			batch_size=self.nsamples, 
			shuffle=False,
			resize=True, nx=self.nx, ny=self.ny, 
			normalize=True, 
			augment=False
		)	
		
		return 0

	#####################################
	##     SAMPLING FUNCTION
	#####################################
	def __sampling(self,args):
		""" Reparameterization trick by sampling from an isotropic unit Gaussian.
			# Arguments
				args (tensor): mean and log of variance of Q(z|X)
			# Returns
				z (tensor): sampled latent vector
		"""

		z_mean, z_log_var = args
		batch = K.shape(z_mean)[0]
		dim = K.int_shape(z_mean)[1]
    
		# by default, random_normal has mean = 0 and std = 1.0
		epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=1.)
		#epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),mean=0., stddev=1.)

		return z_mean + K.exp(0.5 * z_log_var) * epsilon
		#return z_mean + K.exp(z_log_var) * epsilon

	

	#####################################
	##     BUILD PARAMETRIZED NETWORK
	#####################################
	def __build_parametrized_network(self):
		""" Build VAE model parametrized architecture """
	
		#===========================
		#==   CREATE ENCODER
		#===========================	
		logger.info("Creating parametrized encoder network ...")
		if self.__build_parametrized_encoder()<0:
			logger.error("Encoder model creation failed!")
			return -1
		
		#===========================
		#==   CREATE DECODER
		#===========================	
		logger.info("Creating parametrized decoder network ...")
		if self.__build_parametrized_decoder()<0:
			logger.error("Decoder model creation failed!")
			return -1

		#===========================
		#==   CREATE VAE MODEL
		#===========================	
		# - Build model
		logger.info("Creating autoencoder model ...")

		#vae_encoder_output = self.encoder(self.inputs)
		#print("vae_encoder_output shape")
		#print(K.int_shape(vae_encoder_output))

		#vae_decoder_output = self.decoder(vae_encoder_output)
		#print("vae_decoder_output shape")
		#print(K.int_shape(vae_decoder_output))

		if self.use_vae:
			self.outputs= self.decoder(self.encoder(self.inputs)[2])
		else:
			self.outputs= self.decoder(self.encoder(self.inputs))

		print("inputs shape")
		print(K.int_shape(self.inputs))
		print("outputs shape")
		print(K.int_shape(self.outputs))
		print("flattened inputs shape")
		print(K.int_shape(self.flattened_inputs))
		print("flattened outputs shape")
		print(K.int_shape(self.flattened_outputs))


		#self.flattened_outputs = self.decoder(self.encoder(self.inputs)[2])
		#self.outputs= layers.Reshape( (self.ny,self.nx,self.nchannels) )(self.flattened_outputs)
		#self.vae = Model(self.inputs, self.outputs, name='vae_mlp')
		self.vae = Model(inputs=self.inputs, outputs=self.outputs, name='vae')
		
		#===========================
		#==   SET LOSS & METRICS
		#===========================	
		#self.vae.compile(optimizer=self.optimizer, loss=self.loss_v2(self.z_mean, self.z_log_var), experimental_run_tf_function=False)
		#self.vae.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.reco_loss_metric, self.kl_loss_metric], experimental_run_tf_function=False)
		#self.vae.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.reco_loss_metric])
		#self.vae.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.reco_loss_metric], experimental_run_tf_function=False)
		self.vae.compile(optimizer=self.optimizer, loss=self.loss, experimental_run_tf_function=False)

		# - Print and draw model
		self.vae.summary()
		plot_model(self.vae,to_file='vae.png',show_shapes=True)

		return 0

	def __build_parametrized_encoder(self):
		""" Build encoder parametrized network """
		
		# - Initialize weights
		weight_initializer = tf.keras.initializers.HeUniform(seed=self.weight_init_seed)
		

		# - Input layer	
		inputShape = (self.ny, self.nx, self.nchannels)
		self.inputs= Input(shape=inputShape, dtype='float', name='encoder_input')
		x= self.inputs
	
		self.flattened_inputs= layers.Flatten()(x)
		self.input_data_dim= K.int_shape(x)
		print("Input data dim=", self.input_data_dim)

		# - Create a number of CNN layers
		for k in range(len(self.nfilters_cnn)):

			# - Add a Convolutional 2D layer
			padding= "same"
			if k==0:
				# - Set weights for the first layer
				x = layers.Conv2D(self.nfilters_cnn[k], (self.kernsizes_cnn[k], self.kernsizes_cnn[k]), strides=self.strides_cnn[k], activation=self.activation_fcn_cnn, padding=padding, kernel_initializer=weight_initializer)(x)
			else:
				x = layers.Conv2D(self.nfilters_cnn[k], (self.kernsizes_cnn[k], self.kernsizes_cnn[k]), strides=self.strides_cnn[k], activation=self.activation_fcn_cnn, padding=padding)(x)

			# - Add max pooling?
			if self.add_max_pooling:
				padding= "valid"
				x = layers.MaxPooling2D(pool_size=(self.pool_size,self.pool_size),strides=None,padding=padding)(x)
					
			# - Add Leaky RELU?	
			if self.add_leakyrelu:
				x = layers.LeakyReLU(alpha=self.leakyrelu_alpha)(x)

			# - Add batch normalization?
			if self.add_batchnorm:
				x = BatchNormalization(axis=-1)(x)
			

		# - Store layer size before flattening (needed for decoder network)
		self.shape_before_flattening= K.int_shape(x)

		# - Flatten layer
		x = layers.Flatten()(x)
		#self.flattened_inputs= x
		#self.input_data_dim= K.int_shape(x)
		#print("Input data dim=", self.input_data_dim)

		# - Add dense layer?
		if self.add_dense:
			for layer_size in self.dense_layer_sizes:
				x = layers.Dense(layer_size, activation=self.dense_layer_activation)(x)

		# - Output layers
		if self.use_vae:
			self.z_mean = layers.Dense(self.latent_dim,name='z_mean')(x)
			self.z_log_var = layers.Dense(self.latent_dim,name='z_log_var')(x)
			self.z = Lambda(self.__sampling, output_shape=(self.latent_dim,), name='z')([self.z_mean, self.z_log_var])
			#self.z = Sampling()([self.z_mean, self.z_log_var])
			encoder_output= Lambda(self.__sampling, name="z")([self.z_mean, self.z_log_var])

			# - Instantiate encoder model
			self.encoder = Model(self.inputs, [self.z_mean, self.z_log_var, self.z], name='encoder')
			#self.encoder = Model(self.inputs, encoder_output, name='encoder')
		else:
			self.z_mean = layers.Dense(self.latent_dim, name='encoder_output')(x)
			self.encoder = Model(self.inputs, self.z_mean, name='encoder')
		
		# - Print and plot model
		self.encoder.summary()
		plot_model(self.encoder, to_file='encoder.png', show_shapes=True)

		return 0


	def __build_parametrized_decoder(self):
		""" Build decoder parametrized network """

		# - Input layer (equal to encoder output)	
		if self.use_vae:
			latent_inputs = Input(shape=(self.latent_dim,), dtype='float', name='z_sampling')
		else:
			#decoder_input_shape = (None, self.latent_dim)
			#latent_inputs = Input(shape=decoder_input_shape[1:], dtype='float', name='decoder_input')
			latent_inputs = Input(shape=(self.latent_dim,), dtype='float', name='decoder_input')
			
		x= latent_inputs

		# - Add dense layers
		if self.add_dense:
			for layer_size in reversed(self.dense_layer_sizes):
				x = layers.Dense(layer_size, activation=self.dense_layer_activation)(x)

		# - Add dense layer and reshape
		x = layers.Dense(np.prod(self.shape_before_flattening[1:]))(x)
		x = layers.Reshape((self.shape_before_flattening[1], self.shape_before_flattening[2], self.shape_before_flattening[3]))(x)
		
		# - Create a number of CNN layers in reverse order
		for k in reversed(range(len(self.nfilters_cnn))):
			# - Add deconv 2D layer
			padding= "same"
			x = layers.Conv2DTranspose(self.nfilters_cnn[k], (self.kernsizes_cnn[k], self.kernsizes_cnn[k]), strides=self.strides_cnn[k], activation=self.activation_fcn_cnn, padding=padding)(x)
			
			# - Add max pooling?
			if self.add_max_pooling:
				x = layers.UpSampling2D((self.pool_size,self.pool_size),interpolation='nearest')(x)
	
			# - Add Leaky RELU?	
			if self.add_leakyrelu:
				x = layers.LeakyReLU(alpha=self.leakyrelu_alpha)(x)

			# - Add batch normalization?
			if self.add_batchnorm:
				x = BatchNormalization(axis=-1)(x)


		# - Apply a single conv (or Conv tranpose??) layer to recover the original depth of the image
		padding= "same"
		#x = layers.Conv2D(self.nchannels, (3, 3), activation='sigmoid', padding=padding)(x)
		x = layers.Conv2DTranspose(self.nchannels, (3, 3), activation=self.decoder_output_layer_activation, padding=padding)(x)
		outputs = x

		# - Flatten layer
		x = layers.Flatten()(x)
		self.flattened_outputs= x

		# - Create decoder model
		self.decoder = Model(latent_inputs, outputs, name='decoder')

		# - Print and draw model		
		self.decoder.summary()
		plot_model(self.decoder, to_file='decoder.png', show_shapes=True)

		return 0

	
	###########################
	##     LOSS DEFINITION
	###########################	
	@tf.function
	def reco_loss_metric(self, y_true, y_pred):
		""" Reconstruction loss function definition """
    
		y_true_shape= K.shape(y_true)
		img_cube_size= y_true_shape[1]*y_true_shape[2]*y_true_shape[3]

		if self.use_mse_loss:
			reco_loss = mse(K.flatten(y_true), K.flatten(y_pred))
		else:
			reco_loss = binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
      
		return reco_loss*tf.cast(img_cube_size, tf.float32)
		#return K.mean(y_pred-y_true)		
		#return 0	
		


	@tf.function
	def kl_loss_metric(self, y_true, y_pred):
		""" KL loss function definition """
	
		kl_loss= - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
		kl_loss_mean= K.mean(kl_loss)
		return kl_loss_mean

		
	#@tf.function
	#def reco_loss(self, *args, **kwargs):
	#	""" Reconstruction loss function definition """
    
	#	def fn(y_true, y_pred):
	#		y_true_shape= K.shape(y_true)
	#		img_cube_size= y_true_shape[1]*y_true_shape[2]*y_true_shape[3]

	#		if self.use_mse_loss:
	#			reco_loss = mse(K.flatten(y_true), K.flatten(y_pred))
	#		else:
	#			reco_loss = binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
      
	#		return reco_loss*tf.cast(img_cube_size, tf.float32)

	#	fn.__name__ = 'reco_loss'
	#	return fn

	
	#@tf.function
	#def kl_loss(self, *args, **kwargs):
	#	""" KL loss function definition """
	
	#	def fn(y_true, y_pred):
	#		kl_loss= - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
	#		kl_loss_mean= K.mean(kl_loss)
	#		return kl_loss_mean

	#	fn.__name__ = 'kl_loss'
	#	return fn

	@tf.function
	def reco_loss_fcn(self, y_true, y_pred):
		""" Reco loss function definition """

		if self.use_mse_loss:
			reco_loss = mse(y_true, y_pred)
		else:
			reco_loss = binary_crossentropy(y_true, y_pred)
			
		reco_loss= K.mean(reco_loss)
		return reco_loss
  

	@tf.function
	def loss(self, y_true, y_pred):
		""" Loss function definition """

		# - Print and fix numerical issues
		#logger.info("Print tensors and fix numerical issues before computing loss ...")		
		#tf.print("\n y_true_dim:", K.shape(y_true), output_stream=sys.stdout)
		#tf.print("\n y_pred_dim:", K.shape(y_pred), output_stream=sys.stdout)
		#tf.print("\n y_true min:", tf.math.reduce_min(y_true), output_stream=sys.stdout)
		#tf.print("\n y_true max:", tf.math.reduce_max(y_true), output_stream=sys.stdout)
		#tf.print("\n y_pred min:", tf.math.reduce_min(y_pred), output_stream=sys.stdout)
		#tf.print("\n y_pred max:", tf.math.reduce_max(y_pred), output_stream=sys.stdout)
		
		# - Compute flattened tensors
		y_true_shape= K.shape(y_true)
		img_cube_size= y_true_shape[1]*y_true_shape[2]*y_true_shape[3]
		y_true_flattened= K.flatten(y_true)
		y_pred_flattened= K.flatten(y_pred)
		#y_pred_flattened_nonans = tf.where(tf.math.is_nan(y_pred_flattened_nonans), tf.ones_like(w) * 0, y_pred_flattened_nonans) 

		#tf.print("\n img_cube_size:", img_cube_size, output_stream=sys.stdout)
		#tf.print("\n flatten y_true:", y_true_flattened, output_stream=sys.stdout)
		#tf.print("\n flatten y_pred:", y_pred_flattened, output_stream=sys.stdout)
		#tf.print("\n flatten y_true_dim:", K.int_shape(y_true_flattened), output_stream=sys.stdout)
		#tf.print("\n flatten y_pred_dim:", K.int_shape(y_pred_flattened), output_stream=sys.stdout)
		#tf.print("\n flatten y_true min:", tf.math.reduce_min(y_true_flattened), output_stream=sys.stdout)
		#tf.print("\n flatten y_true max:", tf.math.reduce_max(y_true_flattened), output_stream=sys.stdout)
		#tf.print("\n flatten y_pred min:", tf.math.reduce_min(y_pred_flattened), output_stream=sys.stdout)
		#tf.print("\n flatten y_pred max:", tf.math.reduce_max(y_pred_flattened), output_stream=sys.stdout)

		# - Extract sub tensorwith elements that are not NAN/inf.
		#   NB: Exclude also true elements that are =0 (i.e. masked in input data)
		mask= tf.logical_and(tf.logical_and(tf.math.is_finite(y_true_flattened),~tf.math.equal(y_true_flattened,0)), tf.math.is_finite(y_pred_flattened))
		indexes= tf.where(mask)		
		y_true_flattened_masked= tf.gather(y_true_flattened, indexes)
		y_pred_flattened_masked= tf.gather(y_pred_flattened, indexes)
		
		#tf.print("\n y_true_flattened_masked min:", tf.math.reduce_min(y_true_flattened_masked), output_stream=sys.stdout)
		#tf.print("\n y_true_flattened_masked max:", tf.math.reduce_max(y_true_flattened_masked), output_stream=sys.stdout)
		#tf.print("\n y_pred_flattened_masked min:", tf.math.reduce_min(y_pred_flattened_masked), output_stream=sys.stdout)
		#tf.print("\n y_pred_flattened_masked max:", tf.math.reduce_max(y_pred_flattened_masked), output_stream=sys.stdout)

		# - Check if vectors are not empty
		y_true_isempty= tf.equal(tf.size(y_true_flattened_masked),0)	
		y_pred_isempty= tf.equal(tf.size(y_pred_flattened_masked),0)
		are_empty= tf.logical_or(y_true_isempty, y_pred_isempty)
		
		#if tf.executing_eagerly():
		#	are_empty= are_empty_tensor.numpy()
		#else:
		#	are_empty= are_empty_tensor.eval()

		# - Compute reconstruction loss term
		#logger.info("Computing the reconstruction loss ...")
		reco_loss_default= 1.e+99
		reco_loss= tf.cond(are_empty, lambda: tf.constant(reco_loss_default), lambda: self.reco_loss_fcn(y_true_flattened_masked, y_pred_flattened_masked))

		#if self.use_mse_loss:
		#	###reco_loss = mse(y_true_flattened, y_pred_flattened)
		#	reco_loss = mse(y_true_flattened_masked, y_pred_flattened_masked)
		#else:
		#	###reco_loss = binary_crossentropy(y_true_flattened, y_pred_flattened)
		#	reco_loss = binary_crossentropy(y_true_flattened_masked, y_pred_flattened_masked)
			
		#reco_loss= K.mean(reco_loss)
      

		if self.rec_loss_weight>0:
			tf.print("\n reco_loss:", reco_loss, output_stream=sys.stdout)		
		#reco_loss*= tf.cast(img_cube_size, tf.float32)
		#tf.print("\n reco_loss (after mult):", reco_loss, output_stream=sys.stdout)
		

		# - Compute KL loss term
		#logger.info("Computing the KL loss ...")
		kl_loss= 0
		if self.use_vae:
			kl_loss= - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
			kl_loss_mean= K.mean(kl_loss)
			#print("kl_loss=", kl_loss)
			#tf.print("\n kl_loss:", kl_loss, output_stream=sys.stdout)

			if self.kl_loss_weight>0:
				tf.print("\n kl_loss_mean:", kl_loss_mean, output_stream=sys.stdout)
		
		# Total loss
		#logger.info("Computing the total loss ...")
		if self.use_vae:
			#vae_loss = K.mean(reco_loss + kl_loss)
			vae_loss = self.rec_loss_weight*reco_loss + self.kl_loss_weight*kl_loss_mean
			#tf.print("\n vae_loss:", vae_loss, output_stream=sys.stdout)
		else:
			vae_loss= reco_loss

		return vae_loss


	@tf.function
	def loss_v2(self, encoder_mu, encoder_log_variance):
		""" Loss function definition """

		# - Taken from https://blog.paperspace.com/how-to-build-variational-autoencoder-keras/

		def vae_reconstruction_loss(y_true, y_pred):
			""" Define reconstruction loss """
			reconstruction_loss_factor = 1000
			logger.info("Computing the reconstruction loss ...")
			reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.square(y_true-y_pred), axis=[1, 2, 3])	
			logger.info("Reconstruction loss done")
			return reconstruction_loss_factor * reconstruction_loss

		#def vae_kl_loss(encoder_mu, encoder_log_variance): # Original buggy code
		def vae_kl_loss(y_true, y_pred):
			""" Define KL loss """
			logger.info("Computing the KL loss ...")
			#kl_loss = -0.5 * tf.keras.backend.sum(1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) - tf.keras.backend.exp(encoder_log_variance), axis=1) # Original buggy code
			kl_loss = -0.5 * tf.keras.backend.sum(1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) - tf.keras.backend.exp(encoder_log_variance), axis=[1, 2, 3])	
			logger.info("KL loss done ...")
			return kl_loss

	
		#def vae_kl_loss_metric(y_true, y_pred):
		#	logger.info("Computing the KL loss metric ...")
		#	##kl_loss = -0.5 * tf.keras.backend.sum(1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) - tf.keras.backend.exp(encoder_log_variance), axis=1) # Original buggy code
		#	kl_loss = -0.5 * tf.keras.backend.sum(1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) - tf.keras.backend.exp(encoder_log_variance), axis=[1, 2, 3])
		#	logger.info("KL metric loss done ...")
		#	return kl_loss

		def vae_loss(y_true, y_pred):
			""" Define total loss"""

			y_true_dim= K.int_shape(y_true)
			y_pred_dim= K.int_shape(y_pred)
			tf.print("\n y_true_dim:", y_true_dim, output_stream=sys.stdout)
			tf.print("\n y_pred_dim:", y_pred_dim, output_stream=sys.stdout)
			tf.print("\n y_true min:", tf.math.reduce_min(y_true), output_stream=sys.stdout)
			tf.print("\n y_true max:", tf.math.reduce_max(y_true), output_stream=sys.stdout)
			tf.print("\n y_pred min:", tf.math.reduce_min(y_pred), output_stream=sys.stdout)
			tf.print("\n y_pred max:", tf.math.reduce_max(y_pred), output_stream=sys.stdout)
		
			reconstruction_loss = vae_reconstruction_loss(y_true, y_pred)
			kl_loss = vae_kl_loss(y_true, y_pred)
			loss = reconstruction_loss + kl_loss
			#loss = K.mean(reconstruction_loss + kl_loss)	

			tf.print("\n reconstruction_loss dim:", K.int_shape(reconstruction_loss), output_stream=sys.stdout)
			tf.print("\n reconstruction_loss:", reconstruction_loss, output_stream=sys.stdout)
			tf.print("\n kl_loss dim:", K.int_shape(kl_loss), output_stream=sys.stdout)
			tf.print("\n kl_loss:", kl_loss, output_stream=sys.stdout)
			tf.print("\n loss dim:", K.int_shape(loss), output_stream=sys.stdout)
			tf.print("\n loss:", loss, output_stream=sys.stdout)
		

			return loss

		return vae_loss

	###########################
	##     TRAIN NETWORK
	###########################
	def __train_network(self):
		""" Train deep network """
	
		# - Initialize train/test loss vs epoch
		self.train_loss_vs_epoch= np.zeros((1,self.nepochs))	
		deltaLoss_train= 0

		steps_per_epoch= self.nsamples // self.batch_size

		#===========================
		#==   TRAIN VAE
		#===========================
		logger.info("Start VAE training (dataset_size=%d, batch_size=%d, steps_per_epoch=%d) ..." % (self.nsamples, self.batch_size, steps_per_epoch))

		self.fitout= self.vae.fit(
			x=self.train_data_generator,
			epochs=self.nepochs,
			steps_per_epoch=steps_per_epoch,
			validation_data=self.crossval_data_generator,
			validation_steps=self.validation_steps,
			use_multiprocessing=self.use_multiprocessing,
			workers=self.nworkers,
			verbose=2
		)

		
		#===========================
		#==   SAVE NN
		#===========================
		#- Save the model weights
		logger.info("Saving NN weights ...")
		self.vae.save_weights('model_weights.h5')

		# -Save the model architecture in json format
		logger.info("Saving NN architecture in json format ...")
		with open('model_architecture.json', 'w') as f:
			f.write(self.vae.to_json())
		
		#- Save the model
		logger.info("Saving full NN model ...")
		self.vae.save('model.h5')

		# - Save the network architecture diagram
		logger.info("Saving network model architecture to file ...")
		plot_model(self.vae, to_file=self.outfile_model)


		#================================
		#==   SAVE TRAIN METRICS
		#================================
		# - Get losses and plot
		logger.info("Retrieving losses and plot ...")
		loss_train= self.fitout.history['loss']
		loss_val= self.fitout.history['val_loss']
		N= len(loss_train)
		epoch_ids= np.array(range(N))
		epoch_ids+= 1
		epoch_ids= epoch_ids.reshape(N,1)

		print(loss_train)
		
		plt.plot(loss_train, color='b')
		plt.plot(loss_val, color='r')		
		plt.title('VAE loss')
		plt.ylabel('loss')
		plt.xlabel('epochs')
		plt.xlim(left=0)
		plt.ylim(bottom=0)
		plt.legend(['train loss', 'val loss'], loc='upper right')
		#plt.show()
		plt.savefig('losses.png')				


		# - Saving losses to file
		logger.info("Saving train metrics (loss, ...) to file ...")
		

		metrics_data= np.concatenate(
			(epoch_ids,np.array(loss_train).reshape(N,1), np.array(loss_val).reshape(N,1)),
			axis=1
		)
			
		head= '# epoch loss loss_val'
		Utils.write_ascii(metrics_data,self.outfile_nnout_metrics,head)	

		#================================
		#==   SAVE ENCODED DATA
		#================================
		logger.info("Saving encoded data to file ...")
		if self.use_vae:
			self.encoded_data, _, _= self.encoder.predict(
				x=self.test_data_generator,	
				steps=1,
    		verbose=2,
    		workers=self.nworkers,
    		use_multiprocessing=self.use_multiprocessing
			)
		else:
			self.encoded_data= self.encoder.predict(
				x=self.test_data_generator,	
				steps=1,
    		verbose=2,
    		workers=self.nworkers,
    		use_multiprocessing=self.use_multiprocessing
			)

		#print("encoded_data type=",type(self.encoded_data))
		#print("encoded_data len=",len(self.encoded_data))
		#print("encoded_data=",self.encoded_data)
		print("encoded_data shape")
		print(self.encoded_data.shape)	
		print(self.encoded_data)
		N= self.encoded_data.shape[0]
		Nvar= self.encoded_data.shape[1]
		
		
		# - Merge encoded data
		obj_names= np.array(self.source_names).reshape(N,1)
		obj_ids= np.array(self.source_ids).reshape(N,1)
		enc_data= np.concatenate(
			(obj_names, self.encoded_data, obj_ids),
			axis=1
		)

		znames_counter= list(range(1,Nvar+1))
		znames= '{}{}'.format('z',' z'.join(str(item) for item in znames_counter))
		head= '{} {} {}'.format("# sname",znames,"id")
		#Utils.write_ascii(self.encoded_data,self.outfile_encoded_data,head)	
		Utils.write_ascii(enc_data,self.outfile_encoded_data,head)	


		return 0


	#####################################
	##     RUN TRAIN
	#####################################
	def train_model(self):
		""" Run network training """

		#===========================
		#==   SET TRAINING DATA
		#===========================	
		logger.info("Setting training data from data loader ...")
		status= self.__set_data()
		if status<0:
			logger.error("Train data set failed!")
			return -1

		#===========================
		#==   BUILD NN
		#===========================
		#- Create the network
		logger.info("Building network architecture ...")
		if self.__build_parametrized_network()<0:
			logger.error("NN build failed!")
			return -1

		#===========================
		#==   TRAIN NN
		#===========================
		logger.info("Training network ...")
		status= self.__train_network()
		if status<0:
			logger.error("NN train failed!")
			return -1

		#===========================
		#==   PLOT RESULTS
		#===========================
		logger.info("Plotting results ...")
		self.__plot_results()

		return 0


	#####################################
	##     RUN PREDICT
	#####################################
	def predict_model(self, modelfile):
		""" Run model prediction """

		#===========================
		#==   SET DATA
		#===========================	
		logger.info("Setting input data from data loader ...")
		status= self.__set_data()
		if status<0:
			logger.error("Input data set failed!")
			return -1

		#===========================
		#==   LOAD MODEL
		#===========================
		#- Create the network architecture and weights from file
		logger.info("Loading model architecture and weights from file %s ..." % (modelfile))
		if self.__load_model(modelfile)<0:
			logger.warn("Failed to load model!")
			return -1

		if self.vae is None:
			logger.error("Loaded model is None!")
			return -1

		# - Save the network architecture diagram
		logger.info("Saving network model architecture to file ...")
		plot_model(self.vae, to_file=self.outfile_model)
		
		# - Save the network architecture diagram
		logger.info("Saving network model architecture to file ...")
		plot_model(self.vae, to_file=self.outfile_model)

		#===========================
		#==   PREDICT
		#===========================
		predout= self.encoder.predict(
			x=self.test_data_generator,	
			steps=1,
    	verbose=2,
    	workers=self.nworkers,
    	use_multiprocessing=self.use_multiprocessing
		)

		if type(predout)==tuple and len(predout)>0:
			self.encoded_data= predout[0]
		else:
			self.encoded_data= predout

		print("encoded_data shape")
		print(self.encoded_data.shape)	
		print(self.encoded_data)
		N= self.encoded_data.shape[0]
		Nvar= self.encoded_data.shape[1]
		
		
		# - Merge encoded data
		obj_names= np.array(self.source_names).reshape(N,1)
		obj_ids= np.array(self.source_ids).reshape(N,1)
		enc_data= np.concatenate(
			(obj_names, self.encoded_data, obj_ids),
			axis=1
		)

		# - Save latent data to file
		logger.info("Saving predicted latent data to file %s ..." % (self.outfile_encoded_data))
		znames_counter= list(range(1, Nvar+1))
		znames= '{}{}'.format('z',' z'.join(str(item) for item in znames_counter))
		head= '{} {} {}'.format("# sname", znames, "id")
		Utils.write_ascii(enc_data, self.outfile_encoded_data, head)	

		return 0

	
	#####################################
	##     RECONSTRUCT DATA
	#####################################
	def reconstruct_data(self, modelfile, save_imgs=False):
		""" Reconstruct data """

		#===========================
		#==   SET DATA
		#===========================	
		logger.info("Setting input data from data loader ...")
		status= self.__set_data()
		if status<0:
			logger.error("Input data set failed!")
			return -1

		#===========================
		#==   LOAD MODEL
		#===========================
		#- Create the network architecture and weights from file
		logger.info("Loading model architecture and weights from file %s ..." % (modelfile))
		if self.__load_model(modelfile)<0:
			logger.warn("Failed to load model!")
			return -1

		if self.vae is None:
			logger.error("Loaded model is None!")
			return -1

		#===========================
		#==   RECONSTRUCT IMAGES
		#===========================
		img_counter= 0
		try:
			for item in self.test_data_generator:
				print("type(item)")
				print(type(item))
				data= item[0]

				print("type(data)")
				print(type(data))
				print("data shape")
				print(data.shape)	

				# - Get latent data for this output
				predout= self.encoder.predict(
					x= data,	
					steps=1,
    			verbose=2,
    			workers=self.nworkers,
    			use_multiprocessing=self.use_multiprocessing
				)

				if type(predout)==tuple and len(predout)>0:
					encoded_data= predout[0]
				else:
					encoded_data= predout

				print("encoded_data shape")
				print(encoded_data.shape)	
				print(encoded_data)
				N= encoded_data.shape[0]
				Nvar= encoded_data.shape[1]
		
				# - Compute reconstructed image
				logger.info("Reconstructing image sample no. %d ..." % (img_counter))
				decoded_imgs = self.decoder.predict(predout)
				#decoded_imgs = self.decoder.predict(self.encoded_data)
				print("type(decoded_imgs)")
				print(type(decoded_imgs))
				print("decoded_imgs.shape")
				print(decoded_imgs.shape)

				# - Compute some metrics
				# ...
				# ...

				# - Save input & reco images
				#if save_imgs:
				#	# ...	
				#	# ...

				img_counter+= 1

		except Exception as e:
			logger.warn("Generator throwes exception %s, stop loop." % (str(e)))				

		# - Save reco metrics
		# ...
		# ...
		
		return 0

	#####################################
	##     LOAD MODEL
	#####################################
	def __load_model(self, modelfile):
		""" Load model and weights from input h5 file """

		try:
			#self.vae= load_model(modelfile)
			self.vae = model_from_json(open(modelfile).read())
			self.vae.load_weights(os.path.join(os.path.dirname(modelfile), 'model_weights.h5'))

		except Exception as e:
			logger.warn("Failed to load model from file %s (err=%s)!" % (modelfile, str(e)))
			return -1

		return 0

	#####################################
	##     PLOT RESULTS
	#####################################
	def __plot_results(self):
		""" Plot training results """

		#================================
		#==   PLOT LOSS
		#================================
		# - Plot the total loss
		logger.info("Plot the network loss metric to file ...")
		plt.figure(figsize=(20,20))
		plt.xlabel("#epochs")
		plt.ylabel("loss")
		plt.title("Total Loss vs Epochs")
		plt.plot(np.arange(0, self.nepochs), self.train_loss_vs_epoch[0], label="TRAIN SAMPLE")
		plt.tight_layout()
		plt.savefig(self.outfile_loss)
		plt.close()

		#================================
		#==   PLOT ENCODED DATA
		#================================
		# - Display a 2D plot of the encoded data in the latent space
		logger.info("Plot a 2D plot of the encoded data in the latent space ...")
		plt.figure(figsize=(12, 10))

		N= self.encoded_data.shape[0]
		scatplots= ()
		legend_labels= ()
		print("N=%d" % N)
		for i in range(N):
			source_name= self.source_names[i]
			source_label= self.source_labels[i]
			marker= 'o'
			color= 'k'
			obj_id= 0
			has_label= source_label in self.marker_mapping
			if has_label:
				marker= self.marker_mapping[source_label]
				color= self.marker_color_mapping[source_label]
	
			scatplot= plt.scatter(self.encoded_data[i,0], self.encoded_data[i,1], color=color, marker=marker)
			
			# - Search if label was already encountered before
			try:
				legend_labels.index(source_label)
				label_found= True
			except:
				label_found= False
				
			if not label_found:
				legend_labels+= (source_label,)
				scatplots+= (scatplot,)

		plt.legend(scatplots,legend_labels, scatterpoints=1, loc='lower left', ncol=3, fontsize=8)
		plt.xlabel("z0")
		plt.ylabel("z1")
		plt.savefig('latent_data.png')
		

