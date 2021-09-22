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

## KERAS MODULES
import keras
from keras import layers
from keras import models
from keras import optimizers
try:
	from keras.utils import plot_model
except:
	from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras.models import Model
from keras.models import load_model
try:
	from keras.layers.normalization import BatchNormalization
except Exception as e:
	logger.warn("Failed to import BatchNormalization (err=%s), trying in another way ..." % str(e))
	from keras.layers import BatchNormalization
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
from keras.losses import mse, binary_crossentropy


## GRAPHICS MODULES
import matplotlib.pyplot as plt

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
		self.nsamples_train= 0
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
		self.test_data_generator= None
		self.augmentation= False	
		self.validation_steps= 10
		self.use_multiprocessing= True
		self.nworkers= 1
		
		# - NN architecture
		self.use_shallow_network= False
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
		self.nlayers_intermediate= 1
		self.intermediate_layer_size_factor= 1
		self.intermediate_dim= 512
		self.intermediate_layer_activation= 'relu'
		self.output_layer_activation= 'sigmoid'
		self.z_mean = None
		self.z_log_var = None
		self.z = None
		self.shape_before_flattening= 0
		self.batch_size= 32
		self.latent_dim= 2
		self.nepochs= 10
		self.optimizer= 'adam' # 'rmsprop'
		self.learning_rate= 1.e-4
		self.use_mse_loss= False

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
		self.outfile_loss= 'nn_loss.png'
		self.outfile_accuracy= 'nn_accuracy.png'
		self.outfile_model= 'nn_model.png'
		self.outfile_nnout_metrics= 'nnout_metrics.dat'
		self.outfile_encoded_data= 'encoded_data.dat'

	#####################################
	##     SETTERS/GETTERS
	#####################################
	def set_image_size(self,nx,ny):
		""" Set image size """	
		self.nx= nx
		self.ny= ny

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

	def set_intermediate_layer_size(self,n):
		""" Set intermediate layer size """
		self.intermediate_dim= n

	def set_n_intermediate_layers(self,n):
		""" Set number of intermediate layers """
		self.nlayers_intermediate= n

	def set_intermediate_layer_size_factor(self,f):
		""" Set reduction factor to compute number of neurons in dense layers """
		self.intermediate_layer_size_factor= f

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
		self.nsamples_train= len(self.source_labels)

		# - Create train data generator
		self.train_data_generator= self.dl.data_generator(
			batch_size=self.batch_size, 
			shuffle=True,
			resize=True, nx=self.nx, ny=self.ny, 
			normalize=True, 
			augment=self.augmentation
		)	

		self.test_data_generator= self.dl.data_generator(
			batch_size=self.nsamples_train, 
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
		logger.info("Creating VAE model ...")

		#vae_encoder_output = self.encoder(self.inputs)
		#print("vae_encoder_output shape")
		#print(K.int_shape(vae_encoder_output))

		#vae_decoder_output = self.decoder(vae_encoder_output)
		#print("vae_decoder_output shape")
		#print(K.int_shape(vae_decoder_output))

		#self.outputs= self.decoder(self.encoder(self.inputs))
		self.outputs= self.decoder(self.encoder(self.inputs)[2]) ## TEST
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
		#==   SET LOSS
		#===========================	
		# - Set model loss = mse_loss or xent_loss + kl_loss
		# Reconstruction loss
		if self.use_mse_loss:
			#reconstruction_loss = mse(self.flattened_inputs,self.flattened_outputs)
			reconstruction_loss = mse(K.flatten(self.inputs), K.flatten(self.outputs))
		else:
			#reconstruction_loss = binary_crossentropy(self.flattened_inputs,self.flattened_outputs)
			reconstruction_loss = binary_crossentropy(K.flatten(self.inputs), K.flatten(self.outputs))
      

		flatten_datadim= K.int_shape(self.flattened_inputs)[1]
		print("flatten_datadim")
		print(flatten_datadim)		

		
		reconstruction_loss*= flatten_datadim

		# Kl loss
		kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
		kl_loss = K.sum(kl_loss, axis=-1)
		kl_loss*= -0.5

		# Total loss
		vae_loss = K.mean(reconstruction_loss + kl_loss)


		#self.vae.add_loss(vae_loss)
		#self.vae.compile(optimizer=self.optimizer)
		#self.vae.compile(optimizer=self.optimizer, loss=self.loss_func(self.z_mean, self.z_log_var))
		self.vae.compile(optimizer=self.optimizer, loss=self.loss)

		# - Print and draw model
		self.vae.summary()
		plot_model(self.vae,to_file='vae_mlp.png',show_shapes=True)

		return 0

	def __build_parametrized_encoder(self):
		""" Build encoder parametrized network """
		
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
			x = layers.Conv2D(self.nfilters_cnn[k], (self.kernsizes_cnn[k], self.kernsizes_cnn[k]), strides=self.strides_cnn[k], padding=padding)(x)

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

		# - Output layers
		self.z_mean = layers.Dense(self.latent_dim,name='z_mean')(x)
		self.z_log_var = layers.Dense(self.latent_dim,name='z_log_var')(x)
		self.z = Lambda(self.__sampling, output_shape=(self.latent_dim,), name='z')([self.z_mean, self.z_log_var])
		#self.z = Sampling()([self.z_mean, self.z_log_var])
		encoder_output= Lambda(self.__sampling, name="z")([self.z_mean, self.z_log_var])

		# - Instantiate encoder model
		self.encoder = Model(self.inputs, [self.z_mean, self.z_log_var, self.z], name='encoder')
		#self.encoder = Model(self.inputs, encoder_output, name='encoder')
		
		
		# - Print and plot model
		self.encoder.summary()
		plot_model(self.encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

		return 0


	def __build_parametrized_decoder(self):
		""" Build decoder parametrized network """

		# - Input layer (equal to encoder output)
		latent_inputs = Input(shape=(self.latent_dim,), dtype='float', name='z_sampling')
		x= latent_inputs

		# - Add dense layer and reshape
		x = layers.Dense(np.prod(self.shape_before_flattening[1:]))(x)
		x = layers.Reshape((self.shape_before_flattening[1], self.shape_before_flattening[2], self.shape_before_flattening[3]))(x)
		
		# - Create a number of CNN layers in reverse order
		for k in reversed(range(len(self.nfilters_cnn))):
			# - Add deconv 2D layer
			padding= "same"
			x = layers.Conv2DTranspose(self.nfilters_cnn[k], (self.kernsizes_cnn[k], self.kernsizes_cnn[k]), strides=self.strides_cnn[k], padding=padding)(x)

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


		# - Apply a single conv (or Conv tranpose??) layer to recover the original depth of the image
		padding= "same"
		#x = layers.Conv2D(self.nchannels, (3, 3), activation='sigmoid', padding=padding)(x)
		x = layers.Conv2DTranspose(self.nchannels, (3, 3), activation='sigmoid', padding=padding)(x)
		outputs = x

		# - Flatten layer
		x = layers.Flatten()(x)
		self.flattened_outputs= x

		# - Create decoder model
		self.decoder = Model(latent_inputs, outputs, name='decoder')

		# - Print and draw model		
		self.decoder.summary()
		plot_model(self.decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

		return 0

	
	###########################
	##     LOSS DEFINITION
	###########################
	def loss(self, y_true, y_pred):
		""" Loss function definition """

		# - Compute reconstruction loss term
		logger.info("Computing the reconstruction loss ...")
		if self.use_mse_loss:
			reconstruction_loss = mse(K.flatten(y_true), K.flatten(y_pred))
		else:
			reconstruction_loss = binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
      
		# - Compute KL loss term
		logger.info("Computing the KL loss ...")
		kl_loss= - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
		
		# Total loss
		logger.info("Computing the total loss ...")
		vae_loss = K.mean(reconstruction_loss + kl_loss)

		return vae_loss



	def loss_v2(self, encoder_mu, encoder_log_variance):
		""" Loss function definition """

		def vae_reconstruction_loss(y_true, y_predict):
			reconstruction_loss_factor = 1000
			logger.info("Computing the reconstruction loss ...")
			reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.square(y_true-y_predict), axis=[1, 2, 3])
			logger.info("Reconstruction loss done")
			return reconstruction_loss_factor * reconstruction_loss

		def vae_kl_loss(encoder_mu, encoder_log_variance):
			logger.info("Computing the KL loss ...")
			kl_loss = -0.5 * tf.keras.backend.sum(1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) - tf.keras.backend.exp(encoder_log_variance), axis=1)
			logger.info("KL loss done ...")
			return kl_loss

		def vae_kl_loss_metric(y_true, y_predict):
			logger.info("Computing the KL loss metric ...")
			kl_loss = -0.5 * tf.keras.backend.sum(1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) - tf.keras.backend.exp(encoder_log_variance), axis=1)
			logger.info("KL metric loss done ...")
			return kl_loss

		def vae_loss(y_true, y_predict):

			y_true_dim= K.int_shape(y_true)
			y_predict_dim= K.int_shape(y_predict)
			print("y_true_dim")
			print(y_true_dim)
			print("y_predict_dim")
			print(y_predict_dim)

			reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
			kl_loss = vae_kl_loss(y_true, y_predict)

			loss = reconstruction_loss + kl_loss
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

		steps_per_epoch= self.nsamples_train // self.batch_size

		#===========================
		#==   TRAIN VAE
		#===========================
		logger.info("Start VAE training (dataset_size=%d, batch_size=%d, steps_per_epoch=%d) ..." % (self.nsamples_train, self.batch_size, steps_per_epoch))
		for epoch in range(self.nepochs):
	
			#self.fitout= self.vae.fit(
			#	x=self.inputs_train,
			#	epochs=1,
			#	batch_size=self.batch_size,
			#	validation_data=(self.inputs_train, None),
			#	verbose=1
			#)

			self.fitout= self.vae.fit(
				x=self.train_data_generator,
				epochs=1,
				steps_per_epoch=steps_per_epoch,
				#validation_data=self.train_data_generator,
				#validation_steps=self.validation_steps,
				use_multiprocessing=self.use_multiprocessing,
				workers=self.nworkers,
				verbose=2
			)

			loss_train= self.fitout.history['loss'][0]
			self.train_loss_vs_epoch[0,epoch]= loss_train
			if epoch>=1:
				deltaLoss_train= (loss_train/self.train_loss_vs_epoch[0,epoch-1]-1)*100.

			logger.info("EPOCH %d: loss(train)=%s (dl=%s)" % (epoch,loss_train,deltaLoss_train))
				
				

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


		#================================
		#==   SAVE TRAIN METRICS
		#================================
		logger.info("Saving train metrics (loss, ...) to file ...")
		N= self.train_loss_vs_epoch.shape[1]
		epoch_ids= np.array(range(N))
		epoch_ids+= 1
		epoch_ids= epoch_ids.reshape(N,1)

		metrics_data= np.concatenate(
			(epoch_ids,self.train_loss_vs_epoch.reshape(N,1)),
			axis=1
		)
			
		head= '# epoch - loss'
		Utils.write_ascii(metrics_data,self.outfile_nnout_metrics,head)	

		#================================
		#==   SAVE ENCODED DATA
		#================================
		logger.info("Saving encoded data to file ...")
		#self.encoded_data, _, _= self.encoder.predict(self.inputs_train, batch_size=self.batch_size)
		self.encoded_data, _, _= self.encoder.predict(
			x=self.test_data_generator,
    	verbose=2,
    	workers=self.nworkers,
    	use_multiprocessing=self.use_multiprocessing
		)

		#print("encoded_data type=",type(self.encoded_data))
		#print("encoded_data len=",len(self.encoded_data))
		#print("encoded_data=",self.encoded_data)
		print("encoded_data shape")
		print(self.encoded_data.shape)
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
	##     RUN NN TRAIN
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

			plt.scatter(self.encoded_data[i,0], self.encoded_data[i,1], color=color, marker=marker)

		#plt.scatter(self.encoded_data[:, 0], self.encoded_data[:, 1])
		#plt.colorbar()
		plt.xlabel("z0")
		plt.ylabel("z1")
		plt.savefig('encoded_data.png')
		plt.show()


