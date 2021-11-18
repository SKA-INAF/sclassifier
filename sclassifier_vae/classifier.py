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
from tensorflow.keras.models import model_from_json
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

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

from tensorflow.image import convert_image_dtype
from tensorflow.python.ops.image_ops_impl import _fspecial_gauss, _ssim_helper, _verify_compatible_image_shapes


#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()

## SCIKIT MODULES
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity

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


def ssim_batchavg(img1, img2, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
	""" Compute SSIM averaged over all channels and batch """

	with ops.name_scope(None, 'MS-SSIM', [img1, img2]):
		# Convert to tensor if needed.
		img1 = ops.convert_to_tensor(img1, name='img1')
		img2 = ops.convert_to_tensor(img2, name='img2')

		# Shape checking.
		shape1, shape2, checks = _verify_compatible_image_shapes(img1, img2)
		with ops.control_dependencies(checks):
			img1 = array_ops.identity(img1)

		# Compute ssim for all batch and store in list		
		data_shape= tf.shape(img1)
		tf.print("data_shape:", data_shape, output_stream=sys.stdout)
		#data_shape_int= K.int_shape(img1)
		#tf.print("data_shape_int:", data_shape_int, output_stream=sys.stdout)
		#nsamples= data_shape[0]
		NN= data_shape[0].eval()
		tf.print("NN:", NN, output_stream=sys.stdout)

		nsamples= 5
		#nsamples= nsamples.numpy()
		tf.print("type(nsamples):", type(nsamples), output_stream=sys.stdout)
		tf.print("nsamples:", nsamples, output_stream=sys.stdout)
		ssim_list= []
		for i in range(nsamples):
			ssim_curr= ssim_chanavg(img1[i,:,:,:], img2[i,:,:,:], max_val, filter_size, filter_sigma, k1, k2)
			ssim_list.append(ssim_curr)
			tf.print("ssim_curr:", ssim_curr, output_stream=sys.stdout)

		# Compute mean over batch size
		ssim_tensor= tf.stack(ssim_list)
		ssim_batch_mean= tf.reduce_mean(ssim_tensor)
		##ssim_batch_mean= 1
		ssim_batch_mean= tf.cast(ssim_batch_mean, tf.float32)

		return ssim_batch_mean


def ssim_chanavg(img1, img2, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
	""" Compute SSIM averaged over all channels """

	with ops.name_scope(None, 'SSIM', [img1, img2]):
		# Convert to tensor if needed.
		img1 = ops.convert_to_tensor(img1, name='img1')
		img2 = ops.convert_to_tensor(img2, name='img2')

		# Shape checking.
		_, _, checks = _verify_compatible_image_shapes(img1, img2)
		with ops.control_dependencies(checks):
			img1 = array_ops.identity(img1)

		# Need to convert the images to float32.  Scale max_val accordingly so that
		# SSIM is computed correctly.
		max_val = math_ops.cast(max_val, img1.dtype)
		max_val = convert_image_dtype(max_val, dtypes.float32)
		img1 = convert_image_dtype(img1, dtypes.float32)
		img2 = convert_image_dtype(img2, dtypes.float32)
		ssim_per_channel, _ = compute_ssim_per_channel(img1, img2, max_val, filter_size, filter_sigma, k1, k2)
    
		# Compute average over color channels.
		return math_ops.reduce_mean(ssim_per_channel, [-1])


def compute_ssim_per_channel(img1, img2, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
	""" Computes SSIM index between img1 and img2 per channel """

	filter_size = constant_op.constant(filter_size, dtype=dtypes.int32)
	filter_sigma = constant_op.constant(filter_sigma, dtype=img1.dtype)

	shape1, shape2 = array_ops.shape_n([img1, img2])
	checks = [
		control_flow_ops.Assert(math_ops.reduce_all(math_ops.greater_equal(shape1[-3:-1], filter_size)), [shape1, filter_size], summarize=8),
		control_flow_ops.Assert(math_ops.reduce_all(math_ops.greater_equal(shape2[-3:-1], filter_size)), [shape2, filter_size], summarize=8)
	]

	# Enforce the check to run before computation.
	with ops.control_dependencies(checks):
		img1 = array_ops.identity(img1)

	# Replace nans & inf with image minimum
	img1_min= tf.reduce_min(tf.ragged.boolean_mask(img1, mask=tf.math.is_finite(img1)))
	img2_min= tf.reduce_min(tf.ragged.boolean_mask(img2, mask=tf.math.is_finite(img2)))
	cond_img1= tf.logical_and(tf.math.is_finite(img1),~tf.math.equal(img1,0))
	cond_img2= tf.math.is_finite(img2)
	img1= tf.where(~tf.math.is_finite(img1), tf.ones_like(img1) * img1_min, img1)
	img2= tf.where(~tf.math.is_finite(img2), tf.ones_like(img2) * img2_min, img2)


	# TODO(sjhwang): Try to cache kernels and compensation factor.
	kernel = _fspecial_gauss(filter_size, filter_sigma)
	kernel = array_ops.tile(kernel, multiples=[1, 1, shape1[-1], 1])

	# The correct compensation factor is `1.0 - tf.reduce_sum(tf.square(kernel))`,
	# but to match MATLAB implementation of MS-SSIM, we use 1.0 instead.
	compensation = 1.0

	# TODO(sjhwang): Try FFT.
	# TODO(sjhwang): Gaussian kernel is separable in space. Consider applying
	#   1-by-n and n-by-1 Gaussian filters instead of an n-by-n filter.
	def reducer(x):
		shape = array_ops.shape(x)
		x = array_ops.reshape(x, shape=array_ops.concat([[-1], shape[-3:]], 0))
		#y = nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
		y = nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
		return array_ops.reshape(y, array_ops.concat([shape[:-3], array_ops.shape(y)[1:]], 0))

	luminance, cs = _ssim_helper(img1, img2, reducer, max_val, compensation, k1, k2)

	# Average over the second and the third from the last: height, width.
	axes = constant_op.constant([-3, -2], dtype=dtypes.int32)
	ssim_val = math_ops.reduce_mean(luminance * cs, axes)
	cs = math_ops.reduce_mean(cs, axes)

	# Compute masked ssim (excluding nan/inf/0), averaged over 2D 
	ssim_img= luminance * cs
	cond_ssim= tf.math.is_finite(ssim_img)
	mask= tf.logical_and(tf.logical_and(cond_img1, cond_img2), cond_ssim)
	ssim_masked= math_ops.reduce_mean(tf.ragged.boolean_mask(ssim_img, mask=mask), axes)

	#return ssim_val, cs
	return ssim_masked, cs



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
		self.data_generator= None
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
		self.use_mse_loss= True
		self.use_kl_loss= False
		self.use_ssim_loss= False
		self.mse_loss_weight= 1.0
		self.kl_loss_weight= 1.0
		self.ssim_loss_weight= 1.0
		self.ssim_win_size= 3

		self.weight_init_seed= None
		self.shuffle_train_data= True
		self.augment_scale_factor= 1
		
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

		# - Create standard generator (for reconstruction)
		self.data_generator= self.dl.data_generator(
			batch_size=1, 
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
		###self.vae.compile(optimizer=self.optimizer, loss=self.loss, experimental_run_tf_function=False)
		self.vae.compile(optimizer=self.optimizer, loss=self.loss, run_eagerly=True)
		#self.vae.compile(optimizer=self.optimizer, loss=self.loss, run_eagerly=False)
		
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
	#@tf.function
	def mse_loss_fcn(self, y_true, y_pred):
		""" MSE loss function definition used for reconstruction loss """
		return K.mean(mse(y_true, y_pred))

	#@tf.function
	def ce_loss_fcn(self, y_true, y_pred):
		""" Cross-Entropy loss function definition used for reconstruction loss """
		return K.mean(binary_crossentropy(y_true, y_pred))
	
	#@tf.function
	def ssim_loss_fcn(self, y_true, y_pred):
		""" SSIM Loss function definition used for reconstruction loss """
	
		# - Compute ssim index averaged over channels and batch samples
		winsize= self.ssim_win_size
		max_val= 1.0
		filter_sigma= 1.5
		k1= 0.01
		k2= 0.03
		ssim_mean_sample= ssim_batchavg(y_true, y_pred, max_val=max_val, filter_size=winsize, filter_sigma=filter_sigma, k1=k1, k2=k2)
		#ssim_mean_sample= tf.py_function(func=ssim_batchavg, inp=[y_true, y_pred, max_val, winsize, filter_sigma, k1, k2], Tout=tf.float32)
		
		# - Compute ssim loss
		dssim= 0.5*(1.0-ssim_mean_sample)
		loss= tf.cast(dssim, tf.float32)
		#logger.info("ssim_mean_sample=%f, dssim=%f" % (ssim_mean_sample, dssim))
		tf.print("ssim_mean_sample:", ssim_mean_sample, output_stream=sys.stdout)
		tf.print("ssim loss:", loss, output_stream=sys.stdout)

		return loss


	#@tf.function
	def kl_loss_fcn(self):
		""" Kullback-Leibler loss function definition used for VAE latent space regularization """

		kl_loss= - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
		kl_loss_mean= K.mean(kl_loss)
		kl_loss_mean= tf.cast(kl_loss_mean, tf.float32)
		return kl_loss_mean


	#@tf.function
	def mse_reco_loss_fcn(self, y_true, y_pred):
		""" MSE reco loss function definition """

		# - Compute flattened tensors
		y_true_shape= K.shape(y_true)
		img_cube_size= y_true_shape[1]*y_true_shape[2]*y_true_shape[3]
		y_true_flattened= K.flatten(y_true)
		y_pred_flattened= K.flatten(y_pred)
		
		# - Extract sub tensorwith elements that are not NAN/inf.
		#   NB: Exclude also true elements that are =0 (i.e. masked in input data)
		mask= tf.logical_and(tf.logical_and(tf.math.is_finite(y_true_flattened),~tf.math.equal(y_true_flattened,0)), tf.math.is_finite(y_pred_flattened))
		indexes= tf.where(mask)		
		y_true_flattened_masked= tf.gather(y_true_flattened, indexes)
		y_pred_flattened_masked= tf.gather(y_pred_flattened, indexes)
		
		# - Check if vectors are not empty
		y_true_isempty= tf.equal(tf.size(y_true_flattened_masked),0)	
		y_pred_isempty= tf.equal(tf.size(y_pred_flattened_masked),0)
		are_empty= tf.logical_or(y_true_isempty, y_pred_isempty)
		
		# - Compute reconstruction loss term
		reco_loss_default= 1.e+99
		reco_loss= tf.cond(are_empty, lambda: tf.constant(reco_loss_default), lambda: self.mse_loss_fcn(y_true_flattened_masked, y_pred_flattened_masked))
		#reco_loss*= tf.cast(img_cube_size, tf.float32)
		reco_loss= tf.cast(reco_loss, tf.float32)

		return reco_loss

	
	#@tf.function
	def loss(self, y_true, y_pred):
		""" Loss function definition """

		#data_shape_int= K.int_shape(y_true)
		#tf.print("data_shape_int: ", data_shape_int, output_stream=sys.stdout)

		# - Compute MSE reconstruction loss term
		mse_loss= tf.zeros((),dtype=tf.float32)
		if self.use_mse_loss and self.mse_loss_weight>0:
			logger.info("Computing the MSE reconstruction loss ...")	
			mse_loss= self.mse_loss_weight*self.mse_reco_loss_fcn(y_true, y_pred)
		mse_loss= tf.cast(mse_loss, tf.float32)
	
		# - Compute SSIM reconstruction loss term
		ssim_loss= tf.zeros((),dtype=tf.float32)
		if self.use_ssim_loss and self.ssim_loss_weight>0:
			logger.info("Computing the SSIM reconstruction loss ...")	
			ssim_loss= self.ssim_loss_weight*self.ssim_loss_fcn(y_true, y_pred)
		ssim_loss= tf.cast(ssim_loss, tf.float32)
	
		# - Compute KL loss term (ONLY FOR VAE)
		kl_loss= tf.zeros((),dtype=tf.float32)
		if self.use_vae and self.use_kl_loss and self.kl_loss_weight>0:
			logger.info("Computing the KL loss ...")
			kl_loss= self.kl_loss_weight*self.kl_loss_fcn()
		kl_loss= tf.cast(kl_loss, tf.float32)

		# - Compute the total loss
		tot_loss= mse_loss + ssim_loss + kl_loss
		#logger.info("tot_loss=%f: mse=%f, ssim_loss=%f, kl_loss=%f" % (tot_loss, mse_loss, ssim_loss, kl_loss))
		#tf.print("tot_loss: ", tot_loss, output_stream=sys.stdout)
		#tf.print("mse_loss: ", mse_loss, output_stream=sys.stdout)
		#tf.print("ssim_loss: ", ssim_loss, output_stream=sys.stdout)
		#tf.print("kl_loss: ", kl_loss, output_stream=sys.stdout)
		tot_loss= tf.cast(tot_loss, tf.float32)

		return tot_loss


	###########################
	##     TRAIN NETWORK
	###########################
	def __train_network(self):
		""" Train deep network """
	
		# - Initialize train/test loss vs epoch
		self.train_loss_vs_epoch= np.zeros((1,self.nepochs))	
		deltaLoss_train= 0

		scale= 1
		if self.augmentation:
			scale= self.augment_scale_factor
		steps_per_epoch= scale*self.nsamples // self.batch_size

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
		self.encoder.save_weights('encoder_weights.h5')
		self.decoder.save_weights('decoder_weights.h5')

		# -Save the model architecture in json format
		logger.info("Saving NN architecture in json format ...")
		with open('model_architecture.json', 'w') as f:
			f.write(self.vae.to_json())

		with open('encoder_architecture.json', 'w') as f:
			f.write(self.encoder.to_json())

		with open('decoder_architecture.json', 'w') as f:
			f.write(self.decoder.to_json())
		
		#- Save the model
		logger.info("Saving full NN model ...")
		self.vae.save('model.h5')
		self.encoder.save('encoder.h5')
		self.decoder.save('decoder.h5')

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
	def predict_model(self, encoder_model, encoder_weights):
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
		logger.info("Loading encoder model architecture and weights from files %s %s ..." % (encoder_model, encoder_weights))
		if self.__load_encoder(encoder_model, encoder_weights)<0:
			logger.warn("Failed to load encoder model!")
			return -1

		if self.encoder is None:
			logger.error("Loaded model is None!")
			return -1

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
	def reconstruct_data(self, encoder_model, encoder_weights, decoder_model, decoder_weights, winsize=3, outfile_metrics="reco_metrics.dat", save_imgs=False):
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
		#==   LOAD MODELS
		#===========================
		#- Load encoder
		logger.info("Loading encoder model architecture and weights from files %s, %s ..." % (encoder_model, encoder_weights))
		if self.__load_encoder(encoder_model, encoder_weights)<0:
			logger.warn("Failed to load encoder model!")
			return -1

		if self.encoder is None:
			logger.error("Loaded encoder model is None!")
			return -1

		#- Load decoder
		logger.info("Loading decoder model architecture and weights from files %s, %s ..." % (decoder_model, decoder_weights))
		if self.__load_decoder(decoder_model, decoder_weights)<0:
			logger.warn("Failed to load decoder model!")
			return -1

		if self.decoder is None:
			logger.error("Loaded decoder model is None!")
			return -1
		

		#===========================
		#==   RECONSTRUCT IMAGES
		#===========================
		img_counter= 0
		reco_metrics= []
		
		while True:
			try:
				sname= self.source_names[img_counter]
				classid= self.source_ids[img_counter]

				data, _= next(self.data_generator)
				img_counter+= 1

				nchans= data.shape[3]
	
				#print("type(data)")
				#print(type(data))
				#print("data shape")
				#print(data.shape)				
				#print("nchans")
				#print(nchans)

				# - Get latent data for this output
				predout= self.encoder.predict(
					x= data,	
					batch_size=1,
    			verbose=2,
    			workers=self.nworkers,
    			use_multiprocessing=self.use_multiprocessing
				)

				#if type(predout)==tuple and len(predout)>0:
				#	encoded_data= predout[0]
				#else:
				#	encoded_data= predout

				#print("encoded_data shape")
				#print(encoded_data.shape)	
				#print(encoded_data)
				#N= encoded_data.shape[0]
				#Nvar= encoded_data.shape[1]
		
				# - Compute reconstructed image
				logger.info("Reconstructing image sample no. %d (name=%s, id=%d) ..." % (img_counter, sname, classid))
				decoded_imgs = self.decoder.predict(predout)
				###decoded_imgs = self.decoder.predict(self.encoded_data)
				#print("type(decoded_imgs)")
				#print(type(decoded_imgs))
				#print("decoded_imgs.shape")
				#print(decoded_imgs.shape)

				# - Compute metrics
				metric_list= []
				img_list= []
				metric_names= []

				#print("pto 1")

				for j in range(nchans):
					inputdata_img= data[0,:,:,j]
					recdata_img= decoded_imgs[0,:,:,j]
					
					cond= np.logical_and(inputdata_img!=0, np.isfinite(inputdata_img))

					inputdata_1d= inputdata_img[cond]
					recdata_1d= recdata_img[cond]
			
					#print("pto 2")

					#print("inputdata_img.shape")
					#print(inputdata_img.shape)
					#print("recdata_img.shape")
					#print(recdata_img.shape)
					#print("inputdata_1d shape")
					#print(inputdata_1d.shape)
					#print("recdata_1d shape")
					#print(recdata_1d.shape)
					#print("winsize")
					#print(winsize)

					# - Compute MSE
					mse= mean_squared_error(inputdata_1d, recdata_1d)
					
					#print("mse")
					#print(mse)

					# - Compute similarity index
					ssim_mean, ssim_2d= structural_similarity(inputdata_img, recdata_img, full=True, win_size=winsize)
					ssim_1d= ssim_2d[cond]
					ssim_mean_mask= np.nanmean(ssim_1d)
					ssim_min_mask= np.nanmin(ssim_1d)
					ssim_max_mask= np.nanmax(ssim_1d)
					ssim_std_mask= np.nanstd(ssim_1d)

					if not np.isfinite(ssim_mean_mask):
						logger.warn("Image no. %d (chan=%d): ssim_mean_mask is nan/inf!" % (img_counter, j+1))
						ssim_mean_mask= -999

					# - Append images
					recdata_img[~cond]= 0
					ssim_2d[~cond]= 0
					
					img_list.append([])		
					img_list[j].append(inputdata_img)
					img_list[j].append(recdata_img)
					img_list[j].append(ssim_2d)

					# - Append metrics
					metric_list.append(mse)
					metric_list.append(ssim_mean_mask)
					metric_list.append(ssim_min_mask)
					metric_list.append(ssim_max_mask)
					metric_list.append(ssim_std_mask)
	
					metric_names.append("mse_ch" + str(j+1))
					metric_names.append("ssim_mean_ch" + str(j+1))
					metric_names.append("ssim_min_ch" + str(j+1))
					metric_names.append("ssim_max_ch" + str(j+1))
					metric_names.append("ssim_std_ch" + str(j+1))
					
				reco_metrics.append(metric_list)
				
				# - Save input & reco images
				if save_imgs:
					outfile_plot= sname + '_id' + str(classid) + '.png'		
					logger.info("Saving reco plot to file %s ..." % (outfile_plot))
					fig = plt.figure(figsize=(20, 10))
					nrows= len(img_list)
					for i in range(nrows):
						ncols= len(img_list[i])
						for j in range(ncols):
							index= j + i*ncols + 1
							plt.subplot(nrows, ncols, index)
							plt.imshow(img_list[i][j], origin='lower')
							plt.colorbar()

							outfile_fits= sname + '_id' + str(classid) + '_ch' + str(i+1) + '_plot' + str(j+1) + '.fits'
							Utils.write_fits(img_list[i][j], outfile_fits)
					
					plt.savefig(outfile_plot)
					#plt.tight_layout()
					#plt.show()
					
				# - Stop generator
				if img_counter>=self.nsamples:
					logger.info("Sample size (%d) reached, stop generation..." % self.nsamples)
					break

			except (GeneratorExit, KeyboardInterrupt):
				logger.info("Stop loop (keyboard interrupt) ...")
				break
			except Exception as e:
				logger.warn("Stop loop (exception catched %s) ..." % str(e))
				break

		# - Check if read nsamples data
		N= len(reco_metrics)
		if N!=self.nsamples:
			logger.warn("Failed or stop data read (N=%d!=%d), stop." % (N, self.nsamples))
			return -1

		# - Save reco metrics
		logger.info("Setting metric out data ...")
		obj_names= np.array(self.source_names).reshape(N,1)
		obj_ids= np.array(self.source_ids).reshape(N,1)
		
		if not reco_metrics:
			logger.error("Empty reco metrics, check logs!")
			return -1
		nmetrics= len(reco_metrics[0])
		obj_metrics= np.array(reco_metrics).reshape(N, nmetrics)

		out_data= np.concatenate(
			(obj_names, obj_metrics, obj_ids),
			axis=1
		)

		logger.info("Saving reco metrics data to file %s ..." % (outfile_metrics))
		metric_names_str= ' '.join(str(item) for item in metric_names)
		head= '{} {} {}'.format("# sname", metric_names_str, "id")
		Utils.write_ascii(out_data, outfile_metrics, head)	


		return 0

	#####################################
	##     LOAD MODEL
	#####################################
	def __load_model(self, modelfile):
		""" Load model and weights from input h5 file """

		try:
			self.vae= load_model(modelfile)
		
		except Exception as e:
			logger.warn("Failed to load model from file %s (err=%s)!" % (modelfile, str(e)))
			return -1

		return 0


	def __load_model(self, modelfile_json, weightfile):
		""" Load model and weights from input h5 file """

		# - Load model
		try:
			self.vae = model_from_json(open(modelfile_json).read())
			self.vae.load_weights(weightfile)

		except Exception as e:
			logger.warn("Failed to load model from file %s (err=%s)!" % (modelfile_json, str(e)))
			return -1

		# - Build encoder & decoder
		#encoder = Model(autoencoder.input, autoencoder.layers[-2].output)
		#decoder_input = Input(shape=(encoding_dim,))
		#decoder = Model(decoder_input, autoencoder.layers[-1](decoder_input))

		return 0

	def __load_encoder(self, modelfile_json, weightfile):
		""" Load encoder model and weights from input h5 file """

		try:
			self.encoder = model_from_json(open(modelfile_json).read())
			self.encoder.load_weights(weightfile)

		except Exception as e:
			logger.warn("Failed to load encoder model from file %s (err=%s)!" % (modelfile_json, str(e)))
			return -1

		return 0

	def __load_decoder(self, modelfile_json, weightfile):
		""" Load decoder model and weights from input h5 file """

		try:
			self.decoder = model_from_json(open(modelfile_json).read())
			self.decoder.load_weights(weightfile)

		except Exception as e:
			logger.warn("Failed to load decoder model from file %s (err=%s)!" % (modelfile_json, str(e)))
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
		

