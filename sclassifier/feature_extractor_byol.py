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
import collections
import csv
import pickle
from copy import deepcopy
from pathlib import Path

##############################
##     GLOBAL VARS
##############################
from sclassifier import logger

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
	from tensorflow.keras.layers import BatchNormalization
except Exception as e:
	logger.warn("Failed to import BatchNormalization (err=%s), trying in another way ..." % str(e))
	from tensorflow.keras.layers.normalization import BatchNormalization	
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
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import (
	ModelCheckpoint,
	EarlyStopping,
	ReduceLROnPlateau,
)

from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution 
#disable_eager_execution()
#enable_eager_execution()

## SCIKIT MODULES
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity
from skimage.util import img_as_float64
from PIL import Image

## GRAPHICS MODULES
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

## PACKAGE MODULES
from .utils import Utils
from .tf_utils import byol_loss
##from .models import ResNet18, ResNet34
from .models import resnet18, resnet34
from .models import ProjectionHead, ClassificationHead

################################
##   FeatExtractorByol CLASS
################################
# - Implementation following: 
#     https://github.com/garder14/byol-tensorflow2/blob/main/pretraining.py
#     https://www.kaggle.com/code/vedantj/byol-tensorflow-2-0/notebook

class FeatExtractorByol(object):
	""" Class to create and train a feature extractor based on Byol contrastive learning framework

			Arguments:
				- DataGenerator class
	"""
	
	def __init__(self, data_generator):
		""" Return a feature extractor Byol object """

		self.dg= data_generator
		self.dg_cv= None
		self.has_cvdata= False

		# *****************************
		# ** Input data
		# *****************************
		self.nsamples= 0
		self.nsamples_cv= 0
		self.nx= 64 
		self.ny= 64
		self.nchannels= 0
		self.inputs= None	
		self.inputs_train= None
		self.input_labels= {}
		self.source_names= []
		self.input_data_dim= 0
		self.encoded_data= None
		self.train_data_generator= None
		self.crossval_data_generator= None
		self.test_data_generator= None
		self.test_data_generator_embeddings= None
		self.augmentation= False	
		self.validation_steps= 10
		self.use_multiprocessing= True
		self.nworkers= 0

		# *****************************
		# ** Model
		# *****************************
		# - NN architecture
		#self.model= None
		#self.modelfile= ""
		#self.weightfile= ""
		self.modelfile_encoder= ""
		self.weightfile_encoder= ""
		self.modelfile_projector= ""
		self.weightfile_projector= ""
		self.modelfile_predictor= ""
		self.weightfile_predictor= ""
		#self.fitout= None		
		self.f_online= None
		self.g_online= None
		self.q_online= None
		self.f_target= None
		self.g_target= None

		self.add_channorm_layer= False
		self.channorm_min= 0.0
		self.channorm_max= 1.0
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
		self.add_dropout_layer= False
		self.add_conv_dropout_layer= False
		self.conv_dropout_rate= 0.2
		self.dense_layer_sizes= [256,128]
		self.dense_layer_activation= 'relu'
		self.latent_dim= 2
		self.use_global_avg_pooling= False
		self.use_predefined_arch= False
		self.predefined_arch= "resnet50"

		# - Training options
		self.nepochs= 10
		self.batch_size= 32
		self.learning_rate= 1.e-4
		self.optimizer= tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
		self.ph_regul= 0.005
		self.loss_type= "categorical_crossentropy"
		self.weight_init_seed= None
		self.shuffle_train_data= True
		self.augment_scale_factor= 1

		self.load_cv_data_in_batches= True
		self.balance_classes= False
		self.class_probs= {}
		
		# *****************************
		# ** Output
		# *****************************
		self.outfile_loss= 'losses.png'
		self.outfile_nnout_metrics= 'losses.dat'
		self.outfile_encoded_data= 'latent_data.dat'

		self.save_embeddings= True
		self.save_tb_embeddings= False
		self.nembeddings_save= 1000
		self.img_embedding_scale= 1.0 
		self.shuffle_embeddings= False
		self.outfile_tb_embeddings= 'feature_vecs.tsv'

	#####################################
	##     SETTERS/GETTERS
	#####################################
	def set_image_size(self,nx,ny):
		""" Set image size """	
		self.nx= nx
		self.ny= ny

	def set_optimizer(self, opt, learning_rate=1.e-4):
		""" Set optimizer """

		if opt=="rmsprop":
			logger.info("Setting rmsprop optimizer with lr=%f ..." % (learning_rate))
			self.optimizer= tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
		elif opt=="adam":
			logger.info("Setting adam optimizer with lr=%f ..." % (learning_rate))
			self.optimizer= tf.keras.optimizers.Adam(learning_rate=learning_rate)
		else:
			logger.warn("Unknown optimizer selected (%s), won't change the default ..." % (opt))
			

		
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
		self.nchannels= self.dg.nchannels
		self.source_labels= self.dg.labels
		self.source_ids= self.dg.classids
		self.source_names= self.dg.snames
		self.nsamples= len(self.source_labels)

		# - Create train data generator
		self.train_data_generator= self.dg.generate_byol_data(
			batch_size=self.batch_size, 
			shuffle=self.shuffle_train_data,
			balance_classes=self.balance_classes, class_probs=self.class_probs
		)


		# - Create cross validation data generator
		if self.dg_cv is None:
			logger.info("Creating validation data generator (deep-copying train data generator) ...")
			self.dg_cv= deepcopy(self.dg)
			logger.info("Disabling data augmentation in validation data generator ...")
			self.dg_cv.disable_augmentation()
			self.has_cvdata= False
			self.nsamples_cv= 0
			batch_size_cv= 0
			self.crossval_data_generator= None

		else:
			self.has_cvdata= True
			self.nsamples_cv= len(self.dg_cv.labels)
			logger.info("#nsamples_cv=%d" % (self.nsamples_cv))

			if self.load_cv_data_in_batches:
				batch_size_cv= self.batch_size
			else:
				batch_size_cv= self.nsamples_cv

			logger.info("Loading cv data in batches? %d (batch_size_cv=%d)" % (self.load_cv_data_in_batches, batch_size_cv))

			self.crossval_data_generator= self.dg_cv.generate_byol_data(
				batch_size=batch_size_cv, 
				shuffle=False
			)


		# - Create test data generator
		logger.info("Creating test data generator (deep-copying train data generator) ...")
		self.dg_test= deepcopy(self.dg)
		logger.info("Disabling data augmentation in test data generator ...")
		self.dg_test.disable_augmentation()

		self.test_data_generator= self.dg_test.generate_cae_data(
			batch_size=1, 
			shuffle=False
		)

		# - Create embeddings data generator
		logger.info("Creating test data generator for embeddings (deep-copying train data generator) ...")
		self.dg_test_embeddings= deepcopy(self.dg)
		logger.info("Disabling data augmentation in test data generator for embeddings ...")
		self.dg_test_embeddings.disable_augmentation()

		self.test_data_generator_embeddings= self.dg_test_embeddings.generate_data(
			batch_size=1, 
			shuffle=self.shuffle_embeddings
		)
		
		return 0



	#####################################
	##     CREATE BASE MODEL (CUSTOM)
	#####################################
	def __create_custom_base_model(self, inputShape, model_name='base_model'):
		""" Create the encoder base model using a custom parametrized CNN """

		#===========================
		#==  INIT WEIGHTS
		#===========================
		logger.info("Initializing weights ...")
		try:
			weight_initializer = tf.keras.initializers.HeUniform(seed=self.weight_init_seed)
		except:
			logger.info("Failed to find tf.keras.initializers.HeUniform, trying with tf.keras.initializers.he_uniform ...")
			weight_initializer= tf.keras.initializers.he_uniform(seed=self.weight_init_seed) 

		#===========================
		#==  INPUT LAYER
		#===========================
		#x= inputs
		inputs= Input(shape=inputShape)
		input_data_dim= K.int_shape(inputs)
		x= inputs
		print("Base model input data dim=", input_data_dim)
		
		#===========================
		#==  CONV LAYER
		#===========================
		# - Create encoder or base model
		for k in range(len(self.nfilters_cnn)):

			# - Add a Convolutional 2D layer
			padding= "same"
			if k==0:
				# - Set weights for the first layer
				x = layers.Conv2D(self.nfilters_cnn[k], (self.kernsizes_cnn[k], self.kernsizes_cnn[k]), strides=self.strides_cnn[k], padding=padding, kernel_initializer=weight_initializer)(x)
			else:
				x = layers.Conv2D(self.nfilters_cnn[k], (self.kernsizes_cnn[k], self.kernsizes_cnn[k]), strides=self.strides_cnn[k], padding=padding)(x)

			# - Add batch normalization?
			if self.add_batchnorm:
				x = BatchNormalization(axis=-1)(x)

			# - Add Leaky RELU?	
			if self.add_leakyrelu:
				x = layers.LeakyReLU(alpha=self.leakyrelu_alpha)(x)
			else:
				x = layers.ReLU()(x)

			# - Add max pooling?
			if self.add_max_pooling:
				padding= "valid"
				x = layers.MaxPooling2D(pool_size=(self.pool_size,self.pool_size), strides=None, padding=padding)(x)

			# - Add dropout?
			if self.add_conv_dropout_layer:
				x= layers.Dropout(self.conv_dropout_rate)(x)
			
		#===========================
		#==  FLATTEN LAYER
		#===========================
		if self.use_global_avg_pooling:
			x= layers.GlobalAveragePooling2D()(x)
		else:
			x = layers.Flatten()(x)

		#===========================
		#==  DENSE LAYER
		#===========================
		if self.add_dense:
			x = layers.Dense(self.latent_dim, activation=self.dense_layer_activation)(x)

		#===========================
		#==  BUILD MODEL
		#===========================
		model= Model(inputs, x, name=model_name)
		
		return model

	########################################
	##     CREATE BASE MODEL (PREDEFINED)
	########################################
	def __create_predefined_base_model(self, inputShape, model_name='base_model'):
		""" Create the encoder base model """

		#===========================
		#==  INIT WEIGHTS
		#===========================
		logger.info("Initializing weights ...")
		try:
			weight_initializer = tf.keras.initializers.HeUniform(seed=self.weight_init_seed)
		except:
			logger.info("Failed to find tf.keras.initializers.HeUniform, trying with tf.keras.initializers.he_uniform ...")
			weight_initializer= tf.keras.initializers.he_uniform(seed=self.weight_init_seed) 

		#===========================
		#==  INPUT LAYER
		#===========================
		#x= inputs
		inputs= Input(shape=inputShape)
		input_data_dim= K.int_shape(inputs)
		x= inputs
		print("Base model input data dim=", input_data_dim)
		
		#===========================
		#==  RES NET 
		#===========================
		if self.predefined_arch=="resnet50":
			logger.info("Using resnet50 as base encoder ...")
			resnet50= tf.keras.applications.resnet50.ResNet50(
				include_top=False, # disgard the fully-connected layer as we are training from scratch
				weights=None,  # random initialization
				input_tensor=inputs,
				input_shape=inputShape,
				pooling="avg" #global average pooling will be applied to the output of the last convolutional block
			)
			x= resnet50(x)

		elif self.predefined_arch=="resnet101":
			logger.info("Using resnet101 as base encoder ...")
			resnet101= tf.keras.applications.resnet.ResNet101(
				include_top=False, # disgard the fully-connected layer as we are training from scratch
				weights=None,  # random initialization
				input_tensor=inputs,
				input_shape=inputShape,
				pooling="avg" #global average pooling will be applied to the output of the last convolutional block
			)
			x= resnet101(x)

		elif self.predefined_arch=="resnet18":
			logger.info("Using resnet18 as base encoder ...")
			x= resnet18(x, include_top=False)

		elif self.predefined_arch=="resnet34":
			logger.info("Using resnet34 as base encoder ...")
			x= resnet34(x, include_top=False)			
		else:
			logger.error("Unknown/unsupported predefined backbone architecture given (%s)!" % (self.predefined_arch))
			return None

		#===========================
		#==  FLATTEN LAYER
		#===========================
		###x = layers.Flatten()(x) # done already inside resnet block
		###x= layers.GlobalAveragePooling2D()(x) # done already inside pooling

		#===========================
		#==  DENSE LAYER
		#===========================
		# - Needed only to reduce a bit resnet output (2048)
		if self.add_dense:
			x = layers.Dense(self.latent_dim, activation=self.dense_layer_activation)(x)

		#===========================
		#==  BUILD MODEL
		#===========================
		model= Model(inputs, x, name=model_name)
		
		return model

		

	#####################################
	##     CREATE BASE MODEL
	#####################################
	def __create_base_model(self, inputShape, model_name='base_model'):
		""" Create the encoder base model """

		if self.use_predefined_arch:
			return self.__create_predefined_base_model(inputShape, model_name)	
		else:
			return self.__create_custom_base_model(inputShape, model_name)


	##########################################
	##     CREATE PREDICTOR/PROJECTOR MODEL
	##########################################
	def __create_proj_model(self, inputShape, model_name='proj_model'):
		""" Create the projector model """

		#===========================
		#==  INPUT LAYER
		#===========================
		inputs= Input(shape=inputShape)
		input_data_dim= K.int_shape(inputs)
		x= inputs
		
		#===========================
		#==  DENSE LAYER
		#===========================
		# - Original implementation seems to have 2 layers, e.g. 256-128
		num_layers_ph= len(self.dense_layer_sizes)

		for j in range(num_layers_ph):
			layer_size= self.dense_layer_sizes[j]

			if j < num_layers_ph - 1:
				# - Add linear dense layer
				x = layers.Dense(layer_size)(x)
				###x = layers.Dense(layer_size, activation=self.dense_layer_activation, kernel_regularizer=l1(self.ph_regul))(x) # probably wrong, activation function is linear in original work?
				###if self.add_dropout_layer:
				###	x= layers.Dropout(self.dropout_rate)(x)

				# - Add batch normalization?
				if self.add_batchnorm:
					x = BatchNormalization()(x)

				# - Add activation (RELU to make non-linear)	
				if self.add_leakyrelu:
					x = layers.LeakyReLU(alpha=self.leakyrelu_alpha)(x)
				else:
					x = layers.ReLU()(x)

			else:
				# - Add final linear dense layer
				x = layers.Dense(layer_size)(x)

    
		#===========================
		#==  BUILD MODEL
		#===========================
		model= Model(inputs, x, name=model_name)
		
		return model

	

	#####################################
	##     CREATE MODEL
	#####################################
	def __create_model(self):
		""" Create the model """
		
		# - Create inputs
		inputShape = (self.ny, self.nx, self.nchannels)
		self.inputs= Input(shape=inputShape, dtype='float', name='inputs')
		self.input_data_dim= K.int_shape(self.inputs)
		
		print("Input data dim=", self.input_data_dim)
		print("inputs shape")
		print(K.int_shape(self.inputs))

		# - Create online encoder base model (f_online)
		logger.info("Creating online encoder base model ...")
		self.f_online= self.__create_base_model(inputShape, 'f_online')
		self.f_online.summary()

		h= self.f_online(self.inputs)

		# - Create online projector model
		logger.info("Creating online projector model ...")
		self.g_online= self.__create_proj_model(h.shape[-1], 'g_online')
		self.g_online.summary()

		z= self.g_online(h)

		# - Create online predictor model
		logger.info("Creating online predictor model ...")
		self.q_online= self.__create_proj_model(z.shape[-1], 'q_online') 
		self.q_online.summary()

		p= self.q_online(z)

		# - Create target encoder base model (f_target)
		logger.info("Creating target encoder base model ...")
		self.f_target= self.__create_base_model(inputShape, 'f_target')
		self.f_target.summary()

		h = self.f_target(self.inputs)

		# - Create target projector model
		logger.info("Creating target projector model ...")
		self.g_target= self.__create_proj_model(h.shape[-1], 'g_target')
		self.g_target.summary()

		return 0


	#####################################
	##     RUN TRAIN
	#####################################
	def run_train(self, modelfile_encoder="", weightfile_encoder="", modelfile_projector="", weightfile_projector="", modelfile_predictor="", weightfile_predictor=""):
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
		#==   BUILD MODEL
		#===========================
		# - Load model architecture & weights from input files
		if modelfile_encoder!="" and modelfile_projector!="" and modelfile_predictor!="":
			# - Load encoder
			logger.info("Loading encoder model architecture from file: %s, %s ..." % (modelfile_encoder, weightfile_encoder))
			if self.__load_encoder(modelfile_encoder, weightfile_encoder)<0:
				logger.error("Encoder model loading failed!")
				return -1

			self.modelfile_encoder= modelfile_encoder
			self.weightfile_encoder= weightfile_encoder

			# - Load projector
			logger.info("Loading projector model architecture from file: %s, %s ..." % (modelfile_projector, weightfile_projector))
			if self.__load_projector(modelfile_projector, weightfile_projector)<0:
				logger.error("Projector model loading failed!")
				return -1

			self.modelfile_projector= modelfile_projector
			self.weightfile_projector= weightfile_projector

			# - Load predictor
			logger.info("Loading predictor model architecture from file: %s, %s ..." % (modelfile_predictor, weightfile_predictor))
			if self.__load_predictor(modelfile_predictor, weightfile_predictor)<0:
				logger.error("Predictor model loading failed!")
				return -1

			self.modelfile_predictor= modelfile_predictor
			self.weightfile_predictor= weightfile_predictor

		else:
			# - Build model
			logger.info("Building network architecture ...")
			if self.__create_model()<0:
				logger.error("Model build failed!")
				return -1

		#===========================
		#==   TRAIN MODEL
		#===========================
		logger.info("Training BYOL model ...")
		status= self.__train_network()
		if status<0:
			logger.error("Model training failed!")
			return -1

		logger.info("End training run")

		return 0

	#####################################
	##     TRAIN NN
	#####################################
	@tf.function
	def __train_step_pretraining(self, x1, x2):  # (bs, 32, 32, 3), (bs, 32, 32, 3)
		""" Train step pretraining """
		
		# Forward pass
		h_target_1 = self.f_target(x1, training=True)
		z_target_1 = self.g_target(h_target_1, training=True)

		h_target_2 = self.f_target(x2, training=True)
		z_target_2 = self.g_target(h_target_2, training=True)

		with tf.GradientTape(persistent=True) as tape:
			h_online_1 = self.f_online(x1, training=True)
			z_online_1 = self.g_online(h_online_1, training=True)
			p_online_1 = self.q_online(z_online_1, training=True)
            
			h_online_2 = self.f_online(x2, training=True)
			z_online_2 = self.g_online(h_online_2, training=True)
			p_online_2 = self.q_online(z_online_2, training=True)
            
			p_online = tf.concat([p_online_1, p_online_2], axis=0)
			z_target = tf.concat([z_target_2, z_target_1], axis=0)
			loss = byol_loss(p_online, z_target)

		# Backward pass (update online networks)
		grads = tape.gradient(loss, self.f_online.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.f_online.trainable_variables))

		grads = tape.gradient(loss, self.g_online.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.g_online.trainable_variables))

		grads = tape.gradient(loss, self.q_online.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.q_online.trainable_variables))        

		del tape
		
		return loss

	@tf.function
	def __val_step_pretraining(self, x1, x2):  # (bs, 32, 32, 3), (bs, 32, 32, 3)
		""" Validation step pretraining """

		h_target_1 = self.f_target(x1)
		z_target_1 = self.g_target(h_target_1)

		h_target_2 = self.f_target(x2)
		z_target_2 = self.g_target(h_target_2)

		h_online_1 = self.f_online(x1)
		z_online_1 = self.g_online(h_online_1)
		p_online_1 = self.q_online(z_online_1)
            
		h_online_2 = self.f_online(x2)
		z_online_2 = self.g_online(h_online_2)
		p_online_2 = self.q_online(z_online_2)
            
		p_online = tf.concat([p_online_1, p_online_2], axis=0)
		z_target = tf.concat([z_target_2, z_target_1], axis=0)
		loss = byol_loss(p_online, z_target)

		return loss

	def __train_network(self):
		""" Train BYOL model """

		#===========================
		#==   INIT
		#===========================
		# - Initialize train/test loss vs epoch
		self.train_loss_vs_epoch= np.zeros((1,self.nepochs))	
		steps_per_epoch= self.nsamples // self.batch_size

		# - Set validation steps
		val_steps_per_epoch= self.validation_steps
		if self.has_cvdata:
			if self.load_cv_data_in_batches:
				val_steps_per_epoch= self.nsamples_cv // self.batch_size
			else:
				val_steps_per_epoch= 1

		#===========================
		#==   TRAIN NETWORK
		#===========================
		# - Train model
		logger.info("Start BYOL training (dataset_size=%d, batch_size=%d, steps_per_epoch=%d, val_steps_per_epoch=%d) ..." % (self.nsamples, self.batch_size, steps_per_epoch, val_steps_per_epoch))

		losses_train = []
		losses_val = []
		log_every= 1

		for epoch_id in range(self.nepochs):
			  
			# - Run train
			losses_train_batch= []

			for batch_id in range(steps_per_epoch):
				# - Fetch train data from generator
				x1, x2= next(self.train_data_generator)
				loss = self.__train_step_pretraining(x1, x2)
				losses_train_batch.append(float(loss))

				# - Update target networks (exponential moving average of online networks)
				beta = 0.99

				f_target_weights = self.f_target.get_weights()
				f_online_weights = self.f_online.get_weights()
				for i in range(len(f_online_weights)):
					f_target_weights[i] = beta * f_target_weights[i] + (1 - beta) * f_online_weights[i]
				self.f_target.set_weights(f_target_weights)
            
				g_target_weights = self.g_target.get_weights()
				g_online_weights = self.g_online.get_weights()
				for i in range(len(g_online_weights)):
					g_target_weights[i] = beta * g_target_weights[i] + (1 - beta) * g_online_weights[i]
				self.g_target.set_weights(g_target_weights)

				# - Print train losses in batch				
				if (batch_id + 1) % log_every == 0:
					logger.info("Epoch %d/%d [Batch %d/%d]: train_loss=%f" % (epoch_id+1, self.nepochs, batch_id+1, steps_per_epoch, loss))

			# - Run validation?
			losses_val_batch= []
			if self.has_cvdata:
				for batch_id in range(val_steps_per_epoch):
					x1, x2= next(self.crossval_data_generator)
					loss = self.__val_step_pretraining(x1, x2)
					losses_val_batch.append(float(loss))
				
					if (batch_id + 1) % log_every == 0:
						logger.info("Epoch %d/%d [Batch %d/%d]: val_loss=%f" % (epoch_id+1, self.nepochs, batch_id+1, val_steps_per_epoch, loss))

			# - Compute average train & validation loss
			loss_train= np.nanmean(losses_train_batch)
			losses_train.append(loss_train)
			if losses_val_batch:
				loss_val= np.nanmean(losses_val_batch)
				losses_val.append(loss_val)
				logger.info("Epoch %d/%d: train_loss=%f, val_loss=%f " % (epoch_id+1, self.nepochs, loss_train, loss_val))
			else:
				logger.info("Epoch %d/%d: train_loss=%f" % (epoch_id+1, self.nepochs, loss_train))
		
		
		#===========================
		#==   SAVE NN
		#===========================
		# - Save encoder base		
		if self.f_online:
			logger.info("Saving online encoder network/weights to file ...")
			self.f_online.save_weights('encoder_weights.h5')

			with open('encoder_architecture.json', 'w') as f:
				f.write(self.f_online.to_json())

			self.f_online.save('encoder.h5')

		# - Save online projector		
		if self.g_online:
			logger.info("Saving online projector network/weights to file ...")
			self.g_online.save_weights('projector_weights.h5')

			with open('projector_architecture.json', 'w') as f:
				f.write(self.g_online.to_json())

			self.g_online.save('projector.h5')

		# - Save online predictor
		if self.q_online:
			logger.info("Saving online predictor network/weights to file ...")
			self.q_online.save_weights('predictor_weights.h5')

			with open('predictor_architecture.json', 'w') as f:
				f.write(self.q_online.to_json())

			self.q_online.save('predictor.h5')
		
		#================================
		#==   SAVE TRAIN METRICS
		#================================
		# - Saving losses to file
		logger.info("Saving train/val losses to file ...")
		
		N= len(losses_train)		
		if not losses_val:
			losses_val= [0]*N
			
		epoch_ids= np.array(range(N))
		epoch_ids+= 1
		epoch_ids= epoch_ids.reshape(N,1)

		metrics_data= np.concatenate(
			(epoch_ids, np.array(losses_train).reshape(N,1), np.array(losses_val).reshape(N,1)),
			axis=1
		)
		
		head= '# epoch loss loss_val'
		Utils.write_ascii(metrics_data,self.outfile_nnout_metrics,head)	

		#================================
		#==   PREDICT & SAVE EMBEDDINGS
		#================================
		if self.save_embeddings and self.__save_embeddings()<0:
			logger.warn("Failed to save latent space embeddings to file ...")
			return -1

		return 0


	#####################################
	##     RUN PREDICT
	#####################################
	def run_predict(self, modelfile_encoder="", weightfile_encoder=""):
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
		# - Load encoder
		logger.info("Loading encoder model architecture from file: %s, %s ..." % (modelfile_encoder, weightfile_encoder))
		if self.__load_encoder(modelfile_encoder, weightfile_encoder)<0:
			logger.error("Encoder model loading failed!")
			return -1

		self.modelfile_encoder= modelfile_encoder
		self.weightfile_encoder= weightfile_encoder

		#================================
		#==   PREDICT & SAVE EMBEDDINGS
		#================================
		if self.save_embeddings and self.__save_embeddings()<0:
			logger.warn("Failed to save latent space embeddings to file ...")
			return -1

		#================================
		#==  SAVE EMBEDDINGS TO TB
		#================================
		if self.save_tb_embeddings and self.__save_tb_embeddings()<0:
			logger.warn("Failed to save embeddings in tensorboard format ...")

		logger.info("End predict run")

		return 0

	########################################
	##     SAVE EMBEDDINGS
	########################################
	def __save_embeddings(self):
		""" Save embeddings """

		# - Apply model to input
		logger.info("Running BYOL prediction on input data ...")
		predout= self.f_online.predict(
			x=self.test_data_generator,	
			steps=self.nsamples,
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
		N= self.encoded_data.shape[0]
		Nvar= self.encoded_data.shape[1]
		
		# - Merge encoded data
		logger.info("Adding source info data to encoded data ...")
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


	########################################
	##     SAVE EMBEDDINGS TENSORBOARD
	########################################
	def __save_tb_embeddings(self):
		""" Save embeddings for tensorboard visualization """
		
		# - Set nembeddings to be saved: -1=ALL
		if self.nembeddings_save==-1 or self.nembeddings_save>self.nsamples:
			n_embeddings= self.nsamples
		else:
			n_embeddings= self.nembeddings_save

		# - Loop and save
		imgs= []
		img_embeddings= []
		labels= []

		for i in range(n_embeddings):

			# - Get data from generator
			data, sdata= next(self.test_data_generator_embeddings)
			class_id= sdata.id
			class_name= sdata.label
			nimgs= data.shape[0]
			nchannels= data.shape[3]

			# - Get latent data for this output
			predout= self.f_online.predict(
				x= data,	
				batch_size=1,
    		verbose=2,
    		workers=self.nworkers,
    		use_multiprocessing=self.use_multiprocessing
			)

			# - Save embeddings & labels	
			for j in range(nimgs):
				img_embeddings.append(predout[j])
				labels.append(class_id)

			# - Save images (if nchan=1 or nchan=3)
			if nchannels==1 or nchannels==3:
				for j in range(nimgs):
					data_arr= data[j]
					if nchannels==1:
						data_arr= data_arr[:,:,0]

					img_h= data_arr.shape[0]
					img_w= data_arr.shape[1]

					# - Downscale image previews
					scale= self.img_embedding_scale
					if scale>0 and scale<1 and scale!=1:
						try:
							data_resized= Utils.resize_img(img_as_float64(data_arr), (round(img_h * scale), round(img_w * scale)), preserve_range=True, order=1, anti_aliasing=True)
							data_arr= data_resized
						except Exception as e:
							logger.error("Failed to resize image with scale=%f (err=%s)!" % (self.img_embedding_scale, str(e)))

					# - Convert data to [0,255] range and create PIL image (convert to RGB)
					data_norm= (data_arr-np.min(data_arr))/(np.max(data_arr)-np.min(data_arr))
					data_norm= data_norm*255
					img= Image.fromarray(data_norm.astype('uint8')).convert('RGB')
					imgs.append(img)
				
		# - Check there are embedding data
		if not img_embeddings:
			logger.warn("No embeddings retrieved from generator, nothing to be saved ...")
			return -1

		# - Create output log directory
		currdir= os.getcwd()
		savedir= os.path.join(currdir, 'logs', 'embeddings')
		logger.info("Creating embedding save dir %s ..." % (savedir))
		p= Path(savedir)
		p.mkdir(parents=True, exist_ok= True)

		# - Save embeddings
		logger.info("Save embeddings to file %s ..." % (self.outfile_tb_embeddings))
		outfile_fullpath= os.path.join(savedir, self.outfile_tb_embeddings)

		#with open(outfile_fullpath, 'w') as fw:
		#	csv_writer = csv.writer(fw, delimiter='\t')
		#	csv_writer.writerows(img_embeddings)

		embeddings_variable = tf.Variable(img_embeddings) # Create a checkpoint from embedding, the filename and key are the # name of the tensor. 
		checkpoint = tf.train.Checkpoint(embedding=embeddings_variable) 
		checkpoint.save(os.path.join(savedir, "embedding.ckpt"))

		# - Save labels
		outfile_labels_fullpath= os.path.join(savedir, 'metadata.tsv')
		logger.info("Saving label metadata to file %s ..." % (outfile_labels_fullpath))
		
		with open(outfile_labels_fullpath, 'w') as fp: 
			for label in labels:
				#fp.write(f"{label}\n")
				fp.write("{}\n".format(label))

		# - Create config projector
		#   The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
		config = projector.ProjectorConfig()
		embedding = config.embeddings.add()
		embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
		embedding.metadata_path = 'metadata.tsv'
		
		# - Save image sprite (if imgs available)
		nimgs= len(imgs)
		if nimgs>0:
			# - Set the width and height of a single thumbnail	
			nx= imgs[0].width
			ny= imgs[0].height
			embedding.sprite.image_path = 'sprite.jpg' # Specify the width and height of a single thumbnail. 
			embedding.sprite.single_image_dim.extend([ny, nx])
			
			one_square_size = int(np.ceil(np.sqrt(nimgs)))
			master_width = ny * one_square_size
			master_height = nx * one_square_size
			spriteimage = Image.new(
				mode='RGBA',
				size=(master_width, master_height),
				color=(0,0,0,0) # fully transparent
			)

			for count, image in enumerate(imgs):
				div, mod = divmod(count, one_square_size)
				h_loc = nx * div
				w_loc = ny * mod
				spriteimage.paste(image, (w_loc, h_loc))

			outfile_sprite_fullpath= os.path.join(savedir, 'sprite.jpg')
			logger.info("Saving sprite image to file %s ..." % (outfile_sprite_fullpath))
			spriteimage.convert("RGB").save(outfile_sprite_fullpath, transparency=0)


		# - Visualize embeddings
		logger.info("Visualize embeddings ...")
		projector.visualize_embeddings(savedir, config)
	
		return 0
	
	#####################################
	##     LOAD ENCODER MODEL
	#####################################
	def __load_encoder(self, modelfile, weightfile=""):
		""" Load encoder model and weights from input h5 file """

		#==============================
		#==   LOAD MODEL ARCHITECTURE
		#==============================
		# - Load online encoder
		logger.info("Loading online encoder from file %s ..." % (modelfile))
		try:
			self.f_online= load_model(modelfile)
			
		except Exception as e:
			logger.warn("Failed to load online encoder model from file %s (err=%s)!" % (modelfile, str(e)))
			return -1

		if not self.f_online or self.f_online is None:
			logger.error("Encoder online model object is None, loading failed!")
			return -1

		# - Load target encoder
		logger.info("Loading target encoder from file %s ..." % (modelfile))
		try:
			self.f_target= load_model(modelfile)
			
		except Exception as e:
			logger.warn("Failed to load target encoder model from file %s (err=%s)!" % (modelfile, str(e)))
			return -1

		if not self.f_target or self.f_target is None:
			logger.error("Encoder target model object is None, loading failed!")
			return -1

		#==============================
		#==   LOAD MODEL WEIGHTS
		#==============================
		if weightfile:
			# - Load online encoder weights
			logger.info("Loading online encoder weights from file %s ..." % (weightfile))
			try:
				self.f_online.load_weights(weightfile)
				
			except Exception as e:
				logger.warn("Failed to load online encoder model weights from file %s (err=%s)!" % (weightfile, str(e)))
				return -1

			# - Load target encoder weights
			logger.info("Loading target encoder weights from file %s ..." % (weightfile))
			try:
				self.f_target.load_weights(weightfile)

			except Exception as e:
				logger.warn("Failed to load target encoder model weights from file %s (err=%s)!" % (weightfile, str(e)))
				return -1

		return 0

	#####################################
	##     LOAD PROJECTOR MODEL
	#####################################
	def __load_projector(self, modelfile, weightfile=""):
		""" Load projector model and weights from input h5 file """

		#==============================
		#==   LOAD MODEL ARCHITECTURE
		#==============================
		try:
			self.g_online= load_model(modelfile)
			self.g_target= load_model(modelfile)
			
		except Exception as e:
			logger.warn("Failed to load projector model from file %s (err=%s)!" % (modelfile, str(e)))
			return -1

		if not self.g_online or self.g_online is None:
			logger.error("Projector online model object is None, loading failed!")
			return -1

		if not self.g_target or self.g_target is None:
			logger.error("Projector target model object is None, loading failed!")
			return -1

		#==============================
		#==   LOAD MODEL WEIGHTS
		#==============================
		if weightfile:
			try:
				self.g_online.load_weights(weightfile)
				self.g_target.load_weights(weightfile)

			except Exception as e:
				logger.warn("Failed to load projector model weights from file %s (err=%s)!" % (weightfile, str(e)))
				return -1

		return 0

	#####################################
	##     LOAD PREDICTOR MODEL
	#####################################
	def __load_predictor(self, modelfile, weightfile=""):
		""" Load predictor model and weights from input h5 file """

		#==============================
		#==   LOAD MODEL ARCHITECTURE
		#==============================
		try:
			self.q_online= load_model(modelfile)
			
		except Exception as e:
			logger.warn("Failed to load predictor model from file %s (err=%s)!" % (modelfile, str(e)))
			return -1

		if not self.q_online or self.q_online is None:
			logger.error("Predictor model object is None, loading failed!")
			return -1

		
		#==============================
		#==   LOAD MODEL WEIGHTS
		#==============================
		if weightfile:
			try:
				self.q_online.load_weights(weightfile)
				
			except Exception as e:
				logger.warn("Failed to load predictor model weights from file %s (err=%s)!" % (weightfile, str(e)))
				return -1

		return 0
	
