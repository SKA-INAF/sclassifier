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
from tensorboard.plugins import projector
from classification_models.tfkeras import Classifiers

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
#from .data_loader import DataLoader
#from .data_loader import SourceData
from .tf_utils import SoftmaxCosineSim, nt_xent_loss
from .models import resnet18, resnet34


###################################
##   WarmUpCosineDecay CLASS
###################################
# - Taken from https://stackabuse.com/learning-rate-warmup-with-cosine-decay-in-keras-and-tensorflow/
class WarmUpCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
	""" Implement linear warm-up + cosine decay learning rate update """    
	
	def __init__(self, target_lr, warmup_steps, total_steps, steps_done, lr_min=1.e-6):
		super().__init__()
		self.target_lr = target_lr
		self.warmup_steps = warmup_steps
		self.total_steps = total_steps
		self.steps_done= steps_done  # Number of steps done before this training run
		self.lr_min= lr_min

	def __call__(self, step):
		""" Override base method """
		
		# - Compute global step (accounting also previous epochs
		global_step= self.steps_done + step + 1 # step is starting from 0
		
		# - Compute learning rate
		lr = self.lr_warmup_cosine_decay(global_step)

		return tf.where(
			global_step > self.total_steps, self.lr_min, lr, name="learning_rate"
		)

	def lr_warmup_cosine_decay(self, global_step):
		""" Helper function """
	
		# - Compute warmup LR
		#   Target LR * progress of warmup (=1 at the final warmup step)
		warmup_lr = self.target_lr * (global_step / self.warmup_steps)
		logger.info("global_step=%d, warmup_steps=%d, target_lr=%f ==> warmup_lr=%f" % (global_step, self.warmup_steps, self.target_lr, warmup_lr))
		
		# - Compute cosine decay LR
		decay_lr= self.lr_min + 0.5 * (self.target_lr - self.lr_min) * (1 + tf.cos(tf.constant(np.pi) * (global_step - self.warmup_steps) / float(self.total_steps - self.warmup_steps)))
		logger.info("global_step=%d, warmup_steps=%d, decay_lr=%f ==> decay_lr=%f" % (global_step, self.warmup_steps, self.target_lr, decay_lr))

		# - Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether `global_step < warmup_steps` and we're still holding.
		learning_rate = tf.where(global_step < self.warmup_steps, warmup_lr, decay_lr)
		logger.info("Final lr=%f" % (learning_rate))		
		
		return learning_rate
        
	def get_config(self):
		config = {
			'target_lr': self.target_lr,
			'warmup_steps': self.warmup_steps,
			'total_steps': self.total_steps,
			'steps_done': self.steps_done,
			'lr_min': self.lr_min
		}
		return config

################################
##   FeatExtractorSimCLR CLASS
################################
class FeatExtractorSimCLR(object):
	""" Class to create and train a feature extractor based on SimCLR contrastive learning framework

			Arguments:
				- DataGenerator class
	"""
	
	def __init__(self, data_generator):
		""" Return a feature extractor SimCLR object """

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
		self.model= None
		self.modelfile= ""
		self.modelfile_encoder= ""
		self.modelfile_projhead= ""
		self.weightfile= ""
		self.weightfile_encoder= ""	
		self.weightfile_projhead= ""
		self.fitout= None		
		self.encoder= None
		self.projhead= None
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
		self.use_backbone_impl_v2= False
		self.weightfile_backbone= ""	
		self.reg_factor= 0.01 # tf default
		self.add_regularization= False	

		# - Training options
		self.nepochs= 30
		self.batch_size= 32
		self.learning_rate= 1.e-4
		self.optimizer_type= "adam"
		self.set_optimizer(self.optimizer_type, self.learning_rate)
		#self.optimizer= tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
		self.use_warmup_lr_schedule= False
		self.lr_schedule= None
		self.nepochs_done= 0
		self.nepochs_warmup= 10
		self.nepochs_schedule_tot= 100
		
		self.temperature = 0.1 # 0.5
		self.ph_regul= 0.005
		self.loss_type= "categorical_crossentropy"
		self.weight_init_seed= None
		self.shuffle_train_data= True
		self.augment_scale_factor= 1

		self.load_cv_data_in_batches= True
		self.balance_classes= False
		self.class_probs= {}

		self.use_simclr_impl_v2= False

		# *****************************
		# ** Output
		# *****************************
		self.outfile_loss= 'losses.png'
		self.outfile_model= 'model.png'
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

		if type(learning_rate)==float:
			logger.info("Setting %s optimizer with lr=%f ..." % (opt, learning_rate))
		else: 
			logger.info("Setting %s optimizer with lr schedule ..." % (opt))				
		
		if opt=="rmsprop":
			self.optimizer= tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
			self.optimizer_type= opt
			self.learning_rate= learning_rate
		elif opt=="adam":
			self.optimizer= tf.keras.optimizers.Adam(learning_rate=learning_rate)
			self.optimizer_type= opt
			self.learning_rate= learning_rate
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
		if self.nchannels<=0 or self.nchannels is None:
			self.nchannels= self.dg.nchannels # NB: if pre-processor modifies the tensor dimension, you must exclicitly set nchannels!
		self.source_labels= self.dg.labels
		self.source_ids= self.dg.classids
		self.source_names= self.dg.snames
		self.nsamples= len(self.source_labels)

		# - Create train data generator
		if self.use_simclr_impl_v2:
			self.train_data_generator= self.dg.generate_simclr_data_v2(
				batch_size=self.batch_size, 
				shuffle=self.shuffle_train_data,
				balance_classes=self.balance_classes, class_probs=self.class_probs
			)
		else:
			self.train_data_generator= self.dg.generate_simclr_data(
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

			if self.use_simclr_impl_v2:
				self.crossval_data_generator= self.dg_cv.generate_simclr_data_v2(
					batch_size=batch_size_cv, 
					shuffle=False
				)
			else:
				self.crossval_data_generator= self.dg_cv.generate_simclr_data(
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
	def __create_custom_base_model(self, inputShape):
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
		logger.info("Printing base model ...")
		self.encoder = Model(inputs, x, name='base_model')
		
		# - Print and plot model
		logger.info("Printing base model architecture ...")
		self.encoder.summary()

		return x

	########################################
	##     CREATE BASE MODEL (PREDEFINED)
	########################################
	def __create_predefined_base_model(self, inputShape, backbone_weights=""):
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
		# - Set backbone weights
		logger.info("Using %s as base encoder ..." % (self.predefined_arch))
		base_weights= None
		if backbone_weights!="":
			base_weights= backbone_weights
			logger.info("Loading backbone weights from file %s ..." % (base_weights))
		
		# - Add backbone net using image-classifier implementation
		#		NB: This is the implementation used in mrcnn tf2 porting and for which resnet18/34 imagenet weights are available (image-classifier module) 	
		if self.use_backbone_impl_v2:
			logger.info("Using backbone image-classifier implementation ...")
			try:
				backbone_model= Classifiers.get(self.predefined_arch)[0](
					include_top=False,
					weights=base_weights,
					input_tensor=inputs, 
					input_shape=inputShape,
				)
			except Exception as e:
				logger.error("Failed to build base encoder %s (err=%s)!" % (self.predefined_arch, str(e)))
				return None

			# - Add regularization?
			if self.add_regularization:
				logger.info("Applying regularization to built backbone ...")
				regularizer= tf.keras.regularizers.l2(self.reg_factor)
				backbone_model= self.__get_regularized_model(backbone_model, regularizer)

			backbone_model.summary()

			# - Apply model to inputs
			x= backbone_model(x)
		
		else:
			# - Use default tf impl for resnet50/101 & resnet18/resnet34 implementation provided in model.py (from authors of simclr tf porting)
			if self.predefined_arch=="resnet50":
				logger.info("Using resnet50 as base encoder ...")
				resnet50= tf.keras.applications.resnet50.ResNet50(
					include_top=False, # disgard the fully-connected layer as we are training from scratch
					weights=base_weights,  # random initialization
					input_tensor=inputs,
					input_shape=inputShape,
					pooling="avg" #global average pooling will be applied to the output of the last convolutional block
				)
				x= resnet50(x)

			elif self.predefined_arch=="resnet101":
				logger.info("Using resnet101 as base encoder ...")
				resnet101= tf.keras.applications.resnet.ResNet101(
					include_top=False, # disgard the fully-connected layer as we are training from scratch
					weights=base_weights,  # random initialization
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
		# - Not done for impl1 as flattening is done already inside resnet block
		if self.use_backbone_impl_v2:
			###x = layers.Flatten()(x) 
			x= layers.GlobalAveragePooling2D()(x) # done already inside pooling

		#===========================
		#==  DENSE LAYER
		#===========================
		# - Needed only to reduce a bit resnet output (resnet50=2048, resnet18=512)
		if self.add_dense:
			x = layers.Dense(self.latent_dim, activation=self.dense_layer_activation)(x)

		#===========================
		#==  BUILD MODEL
		#===========================
		logger.info("Printing base model ...")
		self.encoder = Model(inputs, x, name='base_model')
		
		# - Print and plot model
		logger.info("Printing base model architecture ...")
		self.encoder.summary()

		return x

	#####################################
	##     ADD REGULARIZATION
	#####################################
	# NB: Saving and reloading both model & weights is needed, otherwise the regularization is not applied in the loss or the pretrained weights are reset 
	#     See: https://sthalles.github.io/keras-regularizer/
	def __get_regularized_model(self, model, regularizer=tf.keras.regularizers.l2(0.01)):
		""" Add regularization to existing backbone model """

		if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
			logger.warn("Regularizer must be a subclass of tf.keras.regularizers.Regularizer, returning same model...")
			return model

		for layer in model.layers:
			for attr in ['kernel_regularizer']:
				if hasattr(layer, attr):
					setattr(layer, attr, regularizer)

		# - When we change the layers attributes, the change only happens in the model config file
		model_json = model.to_json()

		# - Save the weights before reloading the model.
		tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
		model.save_weights(tmp_weights_path)

		# - Load the model from the config
		model = tf.keras.models.model_from_json(model_json)
    
		# Reload the model weights
		model.load_weights(tmp_weights_path, by_name=True)

		return model
		
	#####################################
	##     CREATE BASE MODEL
	#####################################
	def __create_base_model(self, inputShape):
		""" Create the encoder base model """

		if self.use_predefined_arch:
			return self.__create_predefined_base_model(inputShape, backbone_weights=self.weightfile_backbone)	
		else:
			return self.__create_custom_base_model(inputShape)


	#####################################
	##     CREATE PROJECTION HEAD MODEL
	#####################################
	def __create_projhead_model(self, inputShape):
		""" Create the projection head model """

		#===========================
		#==  INPUT LAYER
		#===========================
		inputs= Input(shape=inputShape)
		input_data_dim= K.int_shape(inputs)
		x= inputs
		print("Projection head model input data dim=", input_data_dim)

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
				x = layers.Dense(layer_size, name='projhead_output')(x)

    
		#===========================
		#==  BUILD MODEL
		#===========================
		logger.info("Printing proj head model ...")
		self.projhead = Model(inputs, x, name='projhead_model')
		
		# - Print and plot model
		logger.info("Printing proj head model architecture ...")
		self.projhead.summary()			
		
		return x


	#####################################
	##     CREATE MODEL
	#####################################
	def __create_model(self):
		""" Create the model """
		
		#===========================
		#==  BUILD MODEL
		#===========================
		# - Create inputs
		inputShape = (self.ny, self.nx, self.nchannels)
		self.inputs= Input(shape=inputShape, dtype='float', name='inputs')
		self.input_data_dim= K.int_shape(self.inputs)
		
		print("Input data dim=", self.input_data_dim)
		print("inputs shape")
		print(K.int_shape(self.inputs))

		# - Create encoder base model
		logger.info("Creating encoder model ...")
		encoder_outputs= self.__create_base_model(inputShape)

		# - Create projection head model
		logger.info("Creating projection head model ...")
		projhead_input_data_dim= K.int_shape(encoder_outputs)[1]
		print("== projhead_input_data_dim ==")
		print(projhead_input_data_dim)
		projhead_outputs= self.__create_projhead_model(projhead_input_data_dim)

		
		# - Create model
		#   Inputs: list of 2xbatch_size inputs
		model_inputs= []
		projhead_outputs= []
		for i in range(2 * self.batch_size):
			model_inputs.append( Input(shape=inputShape) )
			projhead_outputs.append(self.projhead(self.encoder(model_inputs[i])))
		
		print("== projhead_outputs shape ==")
		print(type(projhead_outputs))
		print(len(projhead_outputs))
		print(K.int_shape(projhead_outputs[0]))
		
		# - Create softmax cosine similarity layer
		feat_dim= K.int_shape(projhead_outputs[0])[1]
		##soft_cos_sim= SoftmaxCosineSim(batch_size=self.batch_size, feat_dim=self.latent_dim) # NB: Don't understand why in original code feat_dim is equal to encoder output shape
		soft_cos_sim= SoftmaxCosineSim(batch_size=self.batch_size, feat_dim=feat_dim, temperature=self.temperature)

		model_outputs= soft_cos_sim(projhead_outputs)

		print("== model_outputs shape ==")
		print(type(model_outputs))
		print(K.int_shape(model_outputs))

		#===========================
		#==   COMPILE MODEL
		#===========================
		# - Define and compile model
		logger.info("Creating SimCLR model ...")
		self.model= Model(inputs=model_inputs, outputs=model_outputs, name='SimCLR')

		logger.info("Compiling SimCLR model ...")
		self.model.compile(optimizer=self.optimizer, loss=self.loss_type, run_eagerly=True)	

		# - Print model summary
		logger.info("Printing SimCLR model architecture ...")
		self.model.summary()
		
		return 0


	#####################################
	##     CREATE MODEL v2
	##      (https://github.com/garder14/simclr-tensorflow2)   
	#####################################
	def __create_model_v2(self):
		""" Create the model (version 2) """

		#===========================
		#==  BUILD MODEL
		#===========================
		# - Create inputs
		inputShape = (self.ny, self.nx, self.nchannels)
		self.inputs= Input(shape=inputShape, dtype='float', name='inputs')
		self.input_data_dim= K.int_shape(self.inputs)
		
		print("Input data dim=", self.input_data_dim)
		print("inputs shape")
		print(K.int_shape(self.inputs))

		# - Create encoder base model
		logger.info("Creating encoder model ...")
		encoder_outputs= self.__create_base_model(inputShape)

		# - Create projection head model
		logger.info("Creating projection head model ...")
		projhead_input_data_dim= K.int_shape(encoder_outputs)[1]
		print("== projhead_input_data_dim ==")
		print(projhead_input_data_dim)
		projhead_outputs= self.__create_projhead_model(projhead_input_data_dim)

		return 0


	#####################################
	##     CREATE CALLBACKS
	#####################################
	def get_callbacks(self):
		""" Returns callbacks used while training """

		# Time stamp for checkpoint
		now = datetime.datetime.now()
		dt_string = now.strftime("_%m_%d_%Hh_%M")

		checkpoint = ModelCheckpoint(
			"simclr_" + dt_string + ".h5",
			monitor="val_loss",
			verbose=1,
			save_best_only=True,	
			save_weights_only=False,
			mode="auto",
		)

		earlyStopping = EarlyStopping(
			monitor="val_loss",
			patience=10,
			verbose=0,
			mode="auto",
			restore_best_weights=True,
		)
    
		reduce_lr = ReduceLROnPlateau(
			monitor="val_loss", patience=5, verbose=1, factor=0.5,
		)
		    
		return checkpoint, earlyStopping, reduce_lr

	
	#####################################
	##     RUN TRAIN
	#####################################
	def run_train(self, modelfile="", weightfile="", modelfile_encoder="", weightfile_encoder=""):
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
		#- Create the network or load it from file?
		if modelfile!="":
			logger.info("Loading model architecture from file: %s, %s ..." % (modelfile, weightfile))
			if self.__load_model(modelfile, weightfile)<0:
				logger.error("Model loading failed!")
				return -1

			self.modelfile= modelfile
			self.weightfile= weightfile

			# - Load encoder?
			if modelfile_encoder!="":
				logger.info("Loading encoder model architecture from file: %s, %s ..." % (modelfile_encoder, weightfile_encoder))
				if self.__load_encoder(modelfile_encoder, weightfile_encoder)<0:
					logger.error("Encoder model loading failed!")
					return -1

				self.modelfile_encoder= modelfile_encoder
				self.weightfile_encoder= weightfile_encoder

		else:
			logger.info("Building network architecture ...")
			if self.__create_model()<0:
				logger.error("Model build failed!")
				return -1

		#===========================
		#==   TRAIN MODEL
		#===========================
		logger.info("Training SimCLR model ...")
		status= self.__train_network()
		if status<0:
			logger.error("Model training failed!")
			return -1

		logger.info("End training run")

		return 0


	#####################################
	##     RUN TRAIN (v2)
	#####################################
	def run_train_v2(self, modelfile_encoder="", weightfile_encoder="", modelfile_projhead="", weightfile_projhead="", ):
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
		#- Create the network or load it from file?
		if modelfile_encoder!="":
			# - Load encoder?
			logger.info("Loading encoder model architecture from file: %s, %s ..." % (modelfile_encoder, weightfile_encoder))
			if self.__load_encoder(modelfile_encoder, weightfile_encoder)<0:
				logger.error("Encoder model loading failed!")
				return -1

			self.modelfile_encoder= modelfile_encoder
			self.weightfile_encoder= weightfile_encoder

			# - Load proj head?
			if modelfile_projhead!="":
				logger.info("Loading projhead model architecture from file: %s, %s ..." % (modelfile_projhead, weightfile_projhead))
				if self.__load_projhead(modelfile_projhead, weightfile_projhead)<0:
					logger.error("Encoder model loading failed!")
					return -1

				self.modelfile_projhead= modelfile_projhead
				self.weightfile_projhead= weightfile_projhead

		else:
			logger.info("Building network architecture ...")
			if self.__create_model_v2()<0:
				logger.error("Model build failed!")
				return -1

		#===========================
		#==   TRAIN MODEL
		#===========================
		logger.info("Training SimCLR model ...")
		status= self.__train_network_v2()
		if status<0:
			logger.error("Model training failed!")
			return -1

		logger.info("End training run")

		return 0

	#####################################
	##     TRAIN NN
	#####################################
	@tf.function
	def __train_step_pretraining(self, x):  # (2*bs, ny, nx, 3)
		""" Train step pretraining """
        
		# Forward pass
		with tf.GradientTape(persistent=True) as tape:
			h = self.encoder(x, training=True)  # (2*bs, 256)
			z = self.projhead(h, training=True)  # (2*bs, 128)
			loss = nt_xent_loss(z, temperature=tf.constant(self.temperature))
        
		# Backward pass
		grads = tape.gradient(loss, self.encoder.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.encoder.trainable_variables))
		grads = tape.gradient(loss, self.projhead.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.projhead.trainable_variables))
		del tape

		return loss

	@tf.function
	def __val_step_pretraining(self, x):  # (2*bs, ny, nx, 3)
		""" Validation step pretraining """

		h = self.encoder(x, training=False)
		z = self.projhead(h, training=False)
		loss = nt_xent_loss(z, temperature=tf.constant(self.temperature))

		return loss


	def __train_network(self):
		""" Training the SimCLR model and saving best model with time stamp
				Transfers adapted weights to base_model
		"""

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
		#==  SET LR SCHEDULE
		#===========================
		if self.use_warmup_lr_schedule:
			# - Define learning rate schedule
			total_steps = steps_per_epoch*self.nepochs_schedule_tot
			warmup_steps= steps_per_epoch*self.nepochs_warmup
			steps_done= steps_per_epoch*self.nepochs_done
			#target_lr= 0.1*self.batch_size/256. # taken from BYOL paper (https://hal.inria.fr/hal-02869787v1/document)
			target_lr= self.learning_rate
			lr_min= 1.e-6
			
			logger.info("Defining lr schedule with pars: total_steps=%d, warmup_steps=%d, steps_done=%d, target_lr=%f" % (total_steps, warmup_steps, steps_done, target_lr))
			self.lr_schedule= WarmUpCosineDecay(
				target_lr=target_lr, 
				warmup_steps=warmup_steps, 
				total_steps=total_steps, 
				steps_done=steps_done, 
				lr_min=lr_min
			)
			
			# - Redefine optimizer
			logger.info("Redefining optimizer ...")
			self.set_optimizer(self.optimizer_type, learning_rate=self.lr_schedule)
			
			# - Compile model again
			logger.info("Compiling SimCLR model after optimizer update ...")
			self.model.compile(optimizer=self.optimizer, loss=self.loss_type, run_eagerly=True)	


		#===========================
		#==   TRAIN NETWORK
		#===========================
		# - Define train callbacks
		#   NB: Disabling callbacks as val_loss is not reliable
		#checkpoint, earlyStopping, reduce_lr = self.get_callbacks()

		# - Train model
		logger.info("Start SimCLR training (dataset_size=%d, batch_size=%d, steps_per_epoch=%d, val_steps_per_epoch=%d) ..." % (self.nsamples, self.batch_size, steps_per_epoch, val_steps_per_epoch))

		self.fitout= self.model.fit(
			x=self.train_data_generator,
			epochs=self.nepochs,
			steps_per_epoch=steps_per_epoch,
			validation_data=self.crossval_data_generator,
			validation_steps=val_steps_per_epoch,
			#callbacks=[checkpoint, earlyStopping, reduce_lr],
			use_multiprocessing=self.use_multiprocessing,
			workers=self.nworkers,
			verbose=2
		)

		#===========================
		#==   SAVE NN
		#===========================
		#- Save the model weights
		logger.info("Saving model weights ...")
		self.model.save_weights('model_weights.h5')
				
		# -Save the model architecture in json format
		logger.info("Saving model architecture in json format ...")
		with open('model_architecture.json', 'w') as f:
			f.write(self.model.to_json())

		#- Save the model
		logger.info("Saving full model ...")
		self.model.save('model.h5')
		
		# - Save the network architecture diagram
		logger.info("Saving model architecture to file %s ..." % (self.outfile_model))
		plot_model(self.model, to_file=self.outfile_model, show_shapes=True)

		# - Save encoder
		if self.encoder:
			self.encoder.save_weights('encoder_weights.h5')

			with open('encoder_architecture.json', 'w') as f:
				f.write(self.encoder.to_json())

			self.encoder.save('encoder.h5')

		
		#================================
		#==   SAVE TRAIN METRICS
		#================================
		# - Get losses and plot
		logger.info("Retrieving losses and plot ...")
		loss_train= self.fitout.history['loss']
		N= len(loss_train)
				
		loss_val= [0]*N
		if 'val_loss' in self.fitout.history:
			loss_val= self.fitout.history['val_loss']
		epoch_ids= np.array(range(N))
		epoch_ids+= 1
		epoch_ids= epoch_ids.reshape(N,1)

		print(loss_train)
		
		plt.plot(loss_train, color='b')
		plt.plot(loss_val, color='r')		
		plt.title('NN loss')
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
		#==   PREDICT & SAVE EMBEDDINGS
		#================================
		if self.save_embeddings and self.__save_embeddings()<0:
			logger.warn("Failed to save latent space embeddings to file ...")
			return -1

		return 0


	def __train_network_v2(self):
		""" Training the SimCLR model and saving best model with time stamp
				Transfers adapted weights to base_model
		"""

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
		logger.info("Start SimCLR training (dataset_size=%d, batch_size=%d, steps_per_epoch=%d, val_steps_per_epoch=%d) ..." % (self.nsamples, self.batch_size, steps_per_epoch, val_steps_per_epoch))

		losses_train = []
		losses_val = []
		log_every= 1

		for epoch_id in range(self.nepochs):
			  
			# - Run train
			losses_train_batch= []

			for batch_id in range(steps_per_epoch):
				# - Fetch train data from generator
				x= next(self.train_data_generator)
				loss = self.__train_step_pretraining(x)
				losses_train_batch.append(float(loss))

				# - Print train losses in batch				
				if (batch_id + 1) % log_every == 0:
					logger.info("Epoch %d/%d [Batch %d/%d]: train_loss=%f" % (epoch_id+1, self.nepochs, batch_id+1, steps_per_epoch, loss))

			# - Run validation?
			losses_val_batch= []
			if self.has_cvdata:
				for batch_id in range(val_steps_per_epoch):
					x= next(self.crossval_data_generator)
					loss = self.__val_step_pretraining(x)
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
		# - Save encoder
		if self.encoder:
			self.encoder.save_weights('encoder_weights.h5')

			with open('encoder_architecture.json', 'w') as f:
				f.write(self.encoder.to_json())

			self.encoder.save('encoder.h5')

		# - Save proj head
		if self.projhead:
			self.projhead.save_weights('projhead_weights.h5')

			with open('projhead_architecture.json', 'w') as f:
				f.write(self.projhead.to_json())

			self.projhead.save('projhead.h5')

		#================================
		#==   SAVE TRAIN METRICS
		#================================
		# - Get losses and plot
		logger.info("Retrieving losses and plot ...")
		loss_train= self.fitout.history['loss']
		N= len(loss_train)
				
		loss_val= [0]*N
		if 'val_loss' in self.fitout.history:
			loss_val= self.fitout.history['val_loss']
		epoch_ids= np.array(range(N))
		epoch_ids+= 1
		epoch_ids= epoch_ids.reshape(N,1)

		print(loss_train)
		
		plt.plot(loss_train, color='b')
		plt.plot(loss_val, color='r')		
		plt.title('NN loss')
		plt.ylabel('loss')
		plt.xlabel('epochs')
		plt.xlim(left=0)
		plt.ylim(bottom=0)
		plt.legend(['train loss', 'val loss'], loc='upper right')
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
		#==   PREDICT & SAVE EMBEDDINGS
		#================================
		if self.save_embeddings and self.__save_embeddings()<0:
			logger.warn("Failed to save latent space embeddings to file ...")
			return -1

		return 0

	#####################################
	##     RUN PREDICT
	#####################################
	def run_predict(self, modelfile_encoder, weightfile_encoder):
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
		logger.info("Loading encoder model architecture and weights from files %s %s ..." % (modelfile_encoder, weightfile_encoder))
		if self.__load_encoder(modelfile_encoder, weightfile_encoder)<0:
			logger.warn("Failed to load encoder model from files!")
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
		logger.info("Running SimCLR prediction on input data ...")
		predout= self.encoder.predict(
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

			#print("== data shape ==")
			#print(data.shape)

			# - Get latent data for this output
			predout= self.encoder.predict(
				x= data,	
				batch_size=1,
    		verbose=2,
    		workers=self.nworkers,
    		use_multiprocessing=self.use_multiprocessing
			)

			#print("== predout shape ==")
			#print(predout.shape)

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
	##     LOAD MODEL
	#####################################
	def __load_model(self, modelfile, weightfile=""):
		""" Load model and weights from input h5 file """

		#==============================
		#==   LOAD MODEL ARCHITECTURE
		#==============================
		try:
			self.model= load_model(modelfile, custom_objects={'SoftmaxCosineSim': SoftmaxCosineSim, 'WarmupCosineDecay': WarmupCosineDecay})
			
		except Exception as e:
			logger.warn("Failed to load model from file %s (err=%s)!" % (modelfile, str(e)))
			return -1

		if not self.model or self.model is None:
			logger.error("Model object is None, loading failed!")
			return -1

		#if not self.encoder or self.encoder is None:
		#	logger.error("Encoder model object is None, loading failed!")
		#	return -1

		#==============================
		#==   LOAD MODEL WEIGHTS
		#==============================
		if weightfile:
			try:
				self.model.load_weights(weightfile)
			except Exception as e:
				logger.warn("Failed to load model weights from file %s (err=%s)!" % (weightfile, str(e)))
				return -1
		
		#===========================
		#==   SET LOSS & METRICS
		#===========================	
		self.model.compile(optimizer=self.optimizer, loss=self.loss_type, run_eagerly=True)

		# - Print and draw model
		self.model.summary()
		plot_model(self.model, to_file=self.outfile_model, show_shapes=True)

		return 0

	#####################################
	##     LOAD ENCODER MODEL
	#####################################
	def __load_encoder(self, modelfile, weightfile=""):
		""" Load encoder model and weights from input h5 file """

		#==============================
		#==   LOAD MODEL ARCHITECTURE
		#==============================
		try:
			self.encoder= load_model(modelfile)
			
		except Exception as e:
			logger.warn("Failed to load encoder model from file %s (err=%s)!" % (modelfile, str(e)))
			return -1

		if not self.encoder or self.encoder is None:
			logger.error("Encoder model object is None, loading failed!")
			return -1

		#==============================
		#==   LOAD MODEL WEIGHTS
		#==============================
		if weightfile:
			try:
				self.encoder.load_weights(weightfile)
			except Exception as e:
				logger.warn("Failed to load encoder model weights from file %s (err=%s)!" % (weightfile, str(e)))
				return -1

		return 0

	#####################################
	##     LOAD PROJHEAD MODEL
	#####################################
	def __load_projhead(self, modelfile, weightfile=""):
		""" Load proj head model and weights from input h5 file """

		#==============================
		#==   LOAD MODEL ARCHITECTURE
		#==============================
		try:
			self.projhead= load_model(modelfile)
			
		except Exception as e:
			logger.warn("Failed to load projhead model from file %s (err=%s)!" % (modelfile, str(e)))
			return -1

		if not self.projhead or self.projhead is None:
			logger.error("Projector head model object is None, loading failed!")
			return -1

		#==============================
		#==   LOAD MODEL WEIGHTS
		#==============================
		if weightfile:
			try:
				self.projhead.load_weights(weightfile)
			except Exception as e:
				logger.warn("Failed to load projhead model weights from file %s (err=%s)!" % (weightfile, str(e)))
				return -1

		return 0
	
