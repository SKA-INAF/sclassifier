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

## GRAPHICS MODULES
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

## PACKAGE MODULES
from .utils import Utils
#from .data_loader import DataLoader
#from .data_loader import SourceData
from .tf_utils import SoftmaxCosineSim

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
		self.weightfile= ""
		self.weightfile_encoder= ""
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
		self.dense_layer_sizes= [16] 
		self.dense_layer_activation= 'relu'
		self.latent_dim= 2

		# - Training options
		self.nepochs= 10
		self.batch_size= 32
		self.learning_rate= 1.e-4
		self.optimizer_default= 'adam'
		self.optimizer= 'adam' # 'rmsprop'
		self.ph_regul= 0.005
		self.loss_type= "categorical_crossentropy"
		self.weight_init_seed= None
		self.shuffle_train_data= True
		self.augment_scale_factor= 1

		self.load_cv_data_in_batches= True
		self.balance_classes= False
		self.class_probs= {}

		# *****************************
		# ** Pre-processing
		# *****************************
		#self.normalize= False
		#self.scale_to_abs_max= False
		#self.scale_to_max= False
		#self.resize= True
		#self.log_transform_img= False
		#self.scale_img= False
		#self.scale_img_factors= []
		#self.standardize_img= False		
		#self.img_means= []
		#self.img_sigmas= []	
		#self.chan_divide= False
		#self.chan_mins= []
		#self.erode= False
		#self.erode_kernel= 5
		
		# *****************************
		# ** Output
		# *****************************
		self.outfile_loss= 'losses.png'
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
			logger.info("Setting %s optimizer (no lr given) ..." % (opt))
			self.optimizer= opt
		else:
			if opt=="rmsprop":
				logger.info("Setting rmsprop optimizer with lr=%f ..." % (learning_rate))
				self.optimizer= tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
			elif opt=="adam":
				logger.info("Setting adam optimizer with lr=%f ..." % (learning_rate))
				self.optimizer= tf.keras.optimizers.Adam(learning_rate=learning_rate)
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
		self.nchannels= self.dg.nchannels
		self.source_labels= self.dg.labels
		self.source_ids= self.dg.classids
		self.source_names= self.dg.snames
		self.nsamples= len(self.source_labels)

		# - Create train data generator
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

			self.crossval_data_generator= self.dg_cv.generate_simclr_data(
				batch_size=batch_size_cv, 
				shuffle=False
			)


		# - Create test data generator
		logger.info("Creating test data generator (deep-copying train data generator) ...")
		self.dg_test= deepcopy(self.dg)
		logger.info("Disabling data augmentation in test data generator ...")
		self.dg_test.disable_augmentation()

		#self.test_data_generator= self.dg_test.generate_simclr_data(
		self.test_data_generator= self.dg_test.generate_cae_data(
			#batch_size=self.nsamples,
			batch_size=1, 
			shuffle=False
		)
		
		return 0


	#####################################
	##     CREATE BASE MODEL
	#####################################
	def __create_base_model(self, inputShape):
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
		num_layers_ph= len(self.dense_layer_sizes)

		for j in range(num_layers_ph):
			layer_size= self.dense_layer_sizes[j]

			if j < num_layers_ph - 1:
				x = layers.Dense(layer_size, activation=self.dense_layer_activation, kernel_regularizer=l1(self.ph_regul))(x)
				if self.add_dropout_layer:
					x= layers.Dropout(self.dropout_rate)(x)
			else:
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
	##     CREATE PROJECTION HEAD
	#####################################
	#def __create_projhead_layers(self):
	#	""" Create the projection head layers """
	#
	#	# - Init layers
	#	self.ph_l = []
	#	num_layers_ph= len(self.dense_layer_sizes)
	#
	#	if self.add_dense:
	#		for j in range(num_layers_ph):	
	#			layer_size= self.dense_layer_sizes[j]
	#			self.ph_l.append(
	#				layers.Dense(layer_size, activation=self.dense_layer_activation, kernel_regularizer=l1(self.ph_regul))
	#			)
	#
	#	self.ph_l.append(
	#		layers.Dense(self.latent_dim, kernel_regularizer=l1(self.ph_regul), name='projhead_output')
	#	)


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

		# - Create softmax cosine similarity layer
		soft_cos_sim= SoftmaxCosineSim(batch_size=self.batch_size, feat_dim=self.latent_dim) # NB: Dn't understand why in original code feat_dim is equal to encoder output shape
		
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

		model_outputs= soft_cos_sim(projhead_outputs)

		print("== model_outputs shape ==")
		print(type(model_outputs))
		print(K.int_shape(model_outputs))

		# - Create projection head layers
		#self.__create_projhead_layers()

		# - Create softmax cosine similarity layer
		#soft_cos_sim = SoftmaxCosineSim(batch_size=self.batch_size, feat_dim=self.latent_dim)
		
		# - Create model
		#num_layers_ph= len(self.ph_l)
		#i = []  # Inputs (# = 2 x batch_size)
		#f_x = []  # Output base_model
		###h = []  # Flattened feature representation
		#g = []  # Projection head
		#for j in range(num_layers_ph):
		#	g.append([])

		#for index in range(2 * self.batch_size):
		#	i.append(Input(shape=inputShape, dtype='float'))
		#	f_x.append(self.encoder(i[index]))
		#	###h.append(layers.Flatten()(f_x[index]))
		#	for j in range(num_layers_ph):
		#		if j == 0:
		#			###g[j].append(self.ph_l[j](h[index]))
		#			g[j].append(self.ph_l[j](f_x[index]))
		#		else:
		#			g[j].append(self.ph_l[j](g[j - 1][index]))

		#o = soft_cos_sim(g[-1])  # Output = Last layer of projection head

		#logger.info("isinstance(i, list)? %d" % (isinstance(i, list)))
		#logger.info("isinstance(o, list)? %d" % (isinstance(o, list)))
     
		#===========================
		#==   COMPILE MODEL
		#===========================
		# - Define and compile model
		logger.info("Creating SimCLR model ...")
		#self.model= Model(inputs=i, outputs=o, name='SimCLR')
		self.model= Model(inputs=model_inputs, outputs=model_outputs, name='SimCLR')

		logger.info("Compiling SimCLR model ...")
		self.model.compile(optimizer=self.optimizer, loss=self.loss_type, run_eagerly=True)	

		# - Print model summary
		logger.info("Printing SimCLR model architecture ...")
		self.model.summary()
		
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
	##     TRAIN NN
	#####################################
	def __train_network(self):
		""" Training the SimCLR model and saving best model with time stamp
				Transfers adapted weights to base_model
		"""

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
		# - Define train callbacks
		checkpoint, earlyStopping, reduce_lr = self.get_callbacks()

		# - Train model
		logger.info("Start SimCLR training (dataset_size=%d, batch_size=%d, steps_per_epoch=%d, val_steps_per_epoch=%d) ..." % (self.nsamples, self.batch_size, steps_per_epoch, val_steps_per_epoch))

		self.fitout= self.model.fit(
			x=self.train_data_generator,
			epochs=self.nepochs,
			steps_per_epoch=steps_per_epoch,
			validation_data=self.crossval_data_generator,
			validation_steps=val_steps_per_epoch,
			callbacks=[checkpoint, earlyStopping, reduce_lr],
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
		#==   SAVE ENCODED DATA
		#================================
		logger.info("Running SimCLR prediction on input data ...")
		self.encoded_data= self.encoder.predict(
			x=self.test_data_generator,	
			steps=1,
    	verbose=2,
    	workers=self.nworkers,
    	use_multiprocessing=self.use_multiprocessing
		)#.flatten()

		print("encoded_data shape")
		print(self.encoded_data.shape)	
		print(self.encoded_data)
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

		# - Save to file
		logger.info("Saving encoded data to file %s ..." % (self.outfile_encoded_data))
		znames_counter= list(range(1,Nvar+1))
		znames= '{}{}'.format('z',' z'.join(str(item) for item in znames_counter))
		head= '{} {} {}'.format("# sname",znames,"id")
		Utils.write_ascii(enc_data, self.outfile_encoded_data, head)	

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

		#===========================
		#==   PREDICT
		#===========================
		logger.info("Running SimCLR prediction on input data ...")
		predout= self.encoder.predict(
			x=self.test_data_generator,	
			#steps=1,
			steps=self.nsamples,
    	verbose=2,
    	workers=self.nworkers,
    	use_multiprocessing=self.use_multiprocessing
		)#.flatten()

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

		logger.info("End predict run")

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
			self.model= load_model(modelfile, custom_objects={'SoftmaxCosineSim': SoftmaxCosineSim})
			
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
	
