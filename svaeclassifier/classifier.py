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

## KERAS MODULES
import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.utils import plot_model
from keras import backend as K
from keras.models import Model
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf

## GRAPHICS MODULES
import matplotlib.pyplot as plt

## PACKAGE MODULES
from .utils import Utils
from .data_provider import DataProvider

##############################
##     GLOBAL VARS
##############################
logger = logging.getLogger(__name__)



##############################
##     Classifier CLASS
##############################
class VAEClassifier(object):
	""" Class to create and train a VAE classifier

			Arguments:
				- encoder_nnarc_file: File with encoder network architecture to be created
				- decoder_nnarc_file: File with decoder network architecture to be created
				- DataProvider class
	"""
	
	def __init__(self,data_provider):
		""" Return a Classifer object """

		# - Input data
		self.encoder_nnarc_file= ''
		self.decoder_nnarc_file= ''
		self.dp= data_provider

		# - Train data	
		self.nsamples_train= 0
		self.nx= 0
		self.ny= 0
		self.nchannels= 1
		self.inputs_train= None
		
		# - NN architecture
		self.use_shallow_network= True
		self.inputs= None	
		self.encoder= None
		self.nlayers_intermediate= 1
		self.intermediate_dim= 512
		self.intermediate_layer_activation= 'relu'
		self.output_layer_activation= 'sigmoid'
		self.decoder= None
		self.input_data_dim= 0
		self.z_mean = None
		self.z_log_var = None
		self.z = None
		self.shape_before_flattening= 0
		self.batch_size= 16
		self.latent_dim= 2
		self.nepochs= 10
		self.optimizer= 'adam' # 'rmsprop'
		self.learning_rate= 1.e-4
		self.use_mse_loss= True

		# - Output data
		self.outfile_loss= 'nn_loss.png'
		self.outfile_accuracy= 'nn_accuracy.png'
		self.outfile_model= 'nn_model.png'

	#####################################
	##     SETTERS/GETTERS
	#####################################
	def set_optimizer(self,opt):
		""" Set optimizer """
		self.optimizer= opt

	def set_learning_rate(self,lr):
		""" Set learning rate """
		self.learning_rate= lr

	def set_nepochs(self,w):
		""" Set number of train epochs """
		self.nepochs= w

	def set_batch_size(self,bs):
		""" Set batch size """
		self.batch_size= bs


	#####################################
	##     SET TRAIN DATA
	#####################################
	def __set_data(self):
		""" Set train data from provider """

		# - Retrieve input data info from provider
		self.inputs_train= self.dp.get_data()
		imgshape= self.inputs_train.shape
			
		# - Check if data provider has data filled
		if self.inputs_train.ndim!=4:
			logger.error("Invalid number of dimensions in train data (4 expected) (hint: check if data was read in provider!)")
			return -1

		# - Set data
		self.nsamples_train= imgshape[0]
		self.nx= imgshape[2]
		self.ny= imgshape[1]
		self.nchannels= imgshape[3] 
		
		logger.info("Train data size (N,nx,ny,nchan)=(%d,%d,%d)" % (self.nsamples_train,self.nx,self.ny,self.nchannels))
		

		return 0

	#####################################
	##     SALING FUNCTION
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
	##     BUILD SHALLOW NETWORK
	#####################################
	def __build_shallow_network(self):
		""" Build VAE model architecture """
	
		#===========================
		#==   CREATE ENCODER
		#===========================	
		logger.info("Creating shallow encoder network ...")
		status= self.__build_shallow_encoder()
		if status<0:
			logger.error("Encoder model creation failed!")
			return -1
		
		#===========================
		#==   CREATE DECODER
		#===========================	
		logger.info("Creating shallow decoder network ...")
		status= self.__build_shallow_decoder()
		if status<0:
			logger.error("Decoder model creation failed!")
			return -1

		#===========================
		#==   CREATE VAE MODEL
		#===========================	
		# - Build model
		logger.info("Creating VAE model ...")
		self.outputs = self.decoder(self.encoder(self.inputs)[2])
		self.vae = Model(self.inputs, self.outputs, name='vae_mlp')

		# - Set model loss = mse_loss or xent_loss + kl_loss
		# Reconstruction loss
    if self.use_mse_loss:
			reconstruction_loss = mse(self.inputs,self.outputs)
    else:
			reconstruction_loss = binary_crossentropy(self.inputs,self.outputs)

    reconstruction_loss*= self.input_data_dim

		# Kl loss
    kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss*= -0.5

		# Total loss
    vae_loss = K.mean(reconstruction_loss + kl_loss)

    self.vae.add_loss(vae_loss)    
		self.vae.compile(optimizer=self.optimizer)
    
		# - Print and draw model
		self.vae.summary()
    plot_model(self.vae,to_file='vae_mlp.png',show_shapes=True)

		return 0


	#####################################
	##     BUILD SHALLOW ENCODER
	#####################################
	def __build_shallow_encoder(self):
		""" Set encoder shallow network """

		# - Input layer	
		inputShape = (self.ny, self.nx, self.nchannels)
		self.inputs= Input(shape=inputShape,dtype='float', name='encoder_input')
		x= self.inputs

		# - Flatten layer
		x = layers.Flatten()(x)
		self.input_data_dim= K.int_shape(x)
		logger.info("Input data dim= %d" % (self.input_data_dim))

		# - Intermediate layers
		for index in range(self.nlayers_intermediate):
			x = layers.Dense(self.intermediate_dim, activation=self.intermediate_layer_activation)(x)

		# - Output layers
		self.z_mean = layers.Dense(self.latent_dim,name='z_mean')(x)
		self.z_log_var = layers.Dense(self.latent_dim,name='z_log_var')(x)
		self.z = Lambda(self.__sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])

		# - Instantiate encoder model
		self.encoder = Model(self.inputs, [z_mean, z_log_var, z], name='encoder')

		# - Print and plot model
		self.encoder.summary()
		plot_model(self.encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

		return 0
		

	#####################################
	##     BUILD SHALLOW DECODER
	#####################################
	def __build_shallow_decoder(self):
		""" Set decoder shallow network """

		# - Input layer	
		latent_inputs = Input(shape=(self.latent_dim,),dtype='float', name='z_sampling')
		x= latent_inputs

		# - Intermediate layers
		for index in range(self.nlayers_intermediate):
			x = layers.Dense(self.intermediate_dim, activation=self.intermediate_layer_activation)(x)

		# - Output layer
		outputs = layers.Dense(self.input_data_dim, activation=self.output_layer_activation)(x)

		# - Create decoder model
		self.decoder = Model(latent_inputs, outputs, name='decoder')

		# - Print and draw model		
		self.decoder.summary()
		plot_model(self.decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

		return 0


	#####################################
	##     BUILD NETWORK FROM SPEC FILE
	#####################################
	def __build_network(self,encoder_filename,decoder_filename):
		""" Build VAE model architecture """
	
		#===========================
		#==   CREATE ENCODER
		#===========================	
		logger.info("Creating encoder ...")
		status= self.__build_encoder(encoder_filename)
		if status<0:
			logger.error("Encoder model creation failed!")
			return -1
		
		#===========================
		#==   CREATE DECODER
		#===========================	
		logger.info("Creating decoder ...")
		status= self.__build_decoder(decoder_filename)
		if status<0:
			logger.error("Decoder model creation failed!")
			return -1

		return 0

	#####################################
	##     BUILD ENCODER
	#####################################
	def __build_encoder(self,filename):
		""" Set encoder network """
	
		# - Read NN architecture file
		nn_data= []
		skip_patterns= ['#']
		try:
			nn_data= Utils.read_ascii(filename,skip_patterns)
		except IOError:
			print("ERROR: Failed to read nn arc file %d!" % filename)
			return -1

		nlayers= np.shape(nn_data)[0]		
		
		# - Input layer	
		inputShape = (self.ny, self.nx, self.nchannels)
		self.inputs= Input(shape=inputShape,dtype='float', name='input')
		x= self.inputs

		# - Parse NN architecture file and create intermediate layers
		for index in range(nlayers):
			layer_info= nn_data[index]
			logger.info("Layer no. %d: %s" % (index,layer_info))

			layer_type= layer_info[0]

			# - Add Conv2D layer?
			if layer_type=='Conv2D':
				nfields= len(layer_info)
				if nfields!=5:
					logger.error("Invalid number of fields (n=%d) given in Conv2D layer specification (5 expected)" % nfields)
					return -1
				nfilters= int(layer_info[1])
				kernSize= int(layer_info[2])
				activation= str(layer_info[3])
				padding= str(layer_info[4])
				x = layers.Conv2D(filters=nfilters, kernel_size=(kernSize,kernSize), activation=activation, padding=padding)(x)	
				self.shape_before_flattening= K.int_shape(x)		
	
			# - Add MaxPooling2D layer?
			elif layer_type=='MaxPooling2D':
				nfields= len(layer_info)
				if nfields!=3:
					logger.error("Invalid number of fields (n=%d) given in MaxPooling2D layer specification (3 expected)" % nfields)
					return -1
				poolSize= int(layer_info[1])
				padding= str(layer_info[2])
				x = layers.MaxPooling2D(pool_size=(poolSize,poolSize),strides=None,padding=padding)(x)
				self.shape_before_flattening= K.int_shape(x)

			# - Add Dropout layer?
			elif layer_type=='Dropout':
				nfields= len(layer_info)
				if nfields!=2:
					logger.error("Invalid number of fields (n=%d) given in Dropout layer specification (2 expected)" % nfields)
					return -1
				dropout= float(layer_info[1])
				x = layers.Dropout(dropout)(x)

			# - Add BatchNormalization layer?
			elif layer_type=='BatchNormalization':
				x = layers.BatchNormalization()(x)
	
			# - Add Flatten layer?
			elif layer_type=='Flatten':
				x = layers.Flatten()(x)

			# - Add Dense layer?
			elif layer_type=='Dense':
				nfields= len(layer_info)
				if nfields!=3:
					logger.error("Invalid number of fields (n=%d) given in Dense layer specification (3 expected)" % nfields)
				nNeurons= int(layer_info[1])
				activation= str(layer_info[2])
				x = layers.Dense(nNeurons, activation=activation)(x)

			else:
				logger.error("Invalid/unknown layer type parsed (%s)!" % layer_type)
				return -1

		# - Output layers
		z_mean = layers.Dense(self.latent_dim,name='z_mean')(x)
		z_log_var = layers.Dense(self.latent_dim,name='z_log_var')(x)
		z = Lambda(self.__sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])

		# - Instantiate encoder model
		self.encoder = Model(self.inputs, [z_mean, z_log_var, z], name='encoder')

		# - Print and plot model
		self.encoder.summary()
		plot_model(self.encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

		return 0


	#####################################
	##     BUILD DECODER
	#####################################
	def __build_decoder(self,filename):
		""" Set decoder network """

		# - Read NN architecture file
		nn_data= []
		skip_patterns= ['#']
		try:
			nn_data= Utils.read_ascii(filename,skip_patterns)
		except IOError:
			print("ERROR: Failed to read nn arc file %d!" % filename)
			return -1

		nlayers= np.shape(nn_data)[0]		

		# - Set decoder inputs
		latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
		#decoder_input = Input(K.int_shape(z)[1:])

		#x = Dense(intermediate_dim, activation='relu')(latent_inputs)
		#outputs = Dense(original_dim, activation='sigmoid')(x)
		# ...
		# ...

		return 0

	###########################
	##     TRAIN NETWORK
	###########################
	def __train_network(self):
		""" Train deep network """
	
		#===========================
		#==   TRAIN VAE
		#===========================
		logger.info("Start VAE training ...")
		self.vae.fit(
			x=self.inputs_train,
			epochs=self.nepochs,
			batch_size=self.batch_size,
			validation_data=(self.inputs_train, None)
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

		# - Save the networkarchitecture diagram
		logger.info("Saving network model architecture to file ...")
		plot_model(self.vae, to_file=self.outfile_model)


		return 0


	#####################################
	##     RUN NN TRAIN
	#####################################
	def train_model(self):
		""" Run network training """

		#===========================
		#==   SET TRAINING DATA
		#===========================	
		logger.info("Setting training data from provider ...")
		status= self.__set_data()
		if status<0:
			logger.error("Train data set failed!")
			return -1

		#===========================
		#==   BUILD NN
		#===========================
		#- Create the network
		logger.info("Building network architecture ...")
		#if self.use_shallow_network:
			status= self.__build_shallow_network()
		#else:
		#	status= self.__build_network(self.nnarc_file)

		if status<0:
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

		return 0

