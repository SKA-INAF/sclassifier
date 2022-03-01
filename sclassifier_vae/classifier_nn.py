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

##############################
##     GLOBAL VARS
##############################
from sclassifier_vae import logger

## SCI MODULES
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity
from scipy.stats import kurtosis, skew
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.tree import export_text

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from lightgbm import LGBMClassifier

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
from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution 

from tensorflow.keras.utils import to_categorical

## GRAPHICS MODULES
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

## PACKAGE MODULES
from .utils import Utils
from .data_loader import DataLoader
from .data_loader import SourceData


def recall(y_true, y_pred):
	y_true = K.ones_like(y_true) 
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
	recall = true_positives / (all_positives + K.epsilon())
	return recall

def precision(y_true, y_pred):
	y_true = K.ones_like(y_true) 
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

def f1_score(y_true, y_pred):
	precision = precision(y_true, y_pred)
	recall = recall(y_true, y_pred)
	return 2*((precision*recall)/(precision+recall+K.epsilon()))

##################################
##     SClassifierNN CLASS
##################################
class SClassifierNN(object):
	""" Source classifier class """
	
	def __init__(self, data_loader, multiclass=True):
		""" Return a SClassifierNN object """


		self.dl= data_loader
		self.multiclass= multiclass

		# *****************************
		# ** Input data
		# *****************************
		self.nsamples= 0
		self.nx= 64 
		self.ny= 64
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

		# *****************************
		# ** Model
		# *****************************
		
		self.model= None
		self.classids_pred= None
		self.targets_pred= None
		self.probs_pred= None
		self.accuracy= None
		self.precision= None
		self.recall= None   
		self.f1score= None
		self.class_precisions= []
		self.class_recalls= []  
		self.class_f1scores= []
		self.feat_ranks= []
		self.nclasses= 7
		

		# - NN architecture
		self.modelfile= ""
		self.weightfile= ""
		self.fitout= None		
		self.model= None
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
		self.dense_layer_sizes= [16] 
		self.dense_layer_activation= 'relu'

		# - Training options
		self.batch_size= 32
		self.nepochs= 10
		self.learning_rate= 1.e-4
		self.optimizer_default= 'adam'
		self.optimizer= 'adam' # 'rmsprop'
		self.weight_init_seed= None
		self.shuffle_train_data= True
		self.augment_scale_factor= 1
		self.loss_type= "categorical_crossentropy"
		

		self.__set_target_labels(multiclass)

		# *****************************
		# ** Pre-processing
		# *****************************
		self.normalize= False
		self.scale_to_abs_max= False
		self.scale_to_max= False
		self.resize= True
		self.log_transform_img= False
		self.scale_img= False
		self.scale_img_factors= []
		self.standardize_img= False		
		self.img_means= []
		self.img_sigmas= []	
		self.chan_divide= False
		self.chan_mins= []
		self.erode= False
		self.erode_kernel= 5


		# *****************************
		# ** Output
		# *****************************
		self.outfile_model= 'model.png'
		self.outfile_metrics= "metrics.dat"
		self.outfile_loss= 'losses.png'
		self.outfile_nnout_metrics= 'losses.dat'
		self.outfile= 'classified_data.dat'
		
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
	##     CREATE CLASS LABELS
	#####################################
	def __set_target_labels(self, multiclass=True):
		""" Create class labels """

		if multiclass:
			logger.info("Setting multi class targets ...")

			self.nclasses= 7

			self.classid_remap= {
				0: -1,
				1: 4,
				2: 5,
				3: 0,
				6: 1,
				23: 2,
				24: 3,			
				6000: 6,
			}

			self.target_label_map= {
				-1: "UNKNOWN",
				0: "PN",
				1: "HII",
				2: "PULSAR",
				3: "YSO",
				4: "STAR",
				5: "GALAXY",
				6: "QSO",
			}

			self.classid_label_map= {
				0: "UNKNOWN",
				1: "STAR",
				2: "GALAXY",
				3: "PN",
				6: "HII",
				23: "PULSAR",
				24: "YSO",			
				6000: "QSO",
			}

			self.target_names= ["PN","HII","PULSAR","YSO","STAR","GALAXY","QSO"]
			self.loss_type= "categorical_crossentropy"
	
		else: # binary (GAL vs EGAL)

			self.nclasses= 2

			logger.info("Setting binary class targets ...")
			self.classid_remap= {
				0: -1,
				1: 1,
				2: 0,
				3: 1,
				6: 1,
				23: 1,
				24: 1,			
				6000: 0,
			}

			self.target_label_map= {
				-1: "UNKNOWN",
				0: "EGAL",
				1: "GAL",
			}

			self.classid_label_map= {
				0: "UNKNOWN",
				1: "GAL",
				2: "EGAL",
				3: "GAL",
				6: "GAL",
				23: "GAL",
				24: "GAL",			
				6000: "EGAL",
			}

			self.target_names= ["EGAL","GAL"]
			self.loss_type= "binary_crossentropy"

		self.classid_remap_inv= {v: k for k, v in self.classid_remap.items()}
		self.classid_label_map_inv= {v: k for k, v in self.classid_label_map.items()}

		
	#####################################
	##     SET TRAIN DATA
	#####################################
	def __set_data(self):
		""" Set train data & generator from loader """

		# - Retrieve info from data loader
		self.nchannels= self.dl.nchannels
		if self.chan_divide:
			self.nchannels-= 1
		self.source_labels= self.dl.labels
		self.source_ids= self.dl.classids
		self.source_names= self.dl.snames
		self.nsamples= len(self.source_labels)

		# - Set model targets
		self.target_ids= []

		for i in range(self.nsamples):
			source_name= self.source_names[i]
			obj_id= self.source_ids[i]
			label= self.source_labels[i]
			target_id= self.classid_remap[obj_id] # remap obj id to target class ids
				
			if obj_id!=0 and obj_id!=-1:
				self.target_ids.append(target_id)	

		self.class_names= list(set(self.source_labels))
		self.target_names= []
		self.nclasses_targets= 0
		self.output_targets= None
		if self.target_ids:
			self.target_names= [self.target_label_map[item] for item in set(sorted(self.target_ids))]
			self.nclasses_targets= len(self.target_names)
			#self.output_targets= to_categorical(np.array(self.target_ids), num_classes=self.nclasses_targets)
			self.output_targets= to_categorical(np.array(self.target_ids), num_classes=self.nclasses)
		else:
			logger.info("No known class found in dataset (not a problem if predicting) ...")

		print("== CLASS NAMES ==")
		print(self.class_names)
	
		print("== TARGET NO/NAMES ==")
		print(self.nclasses_targets)		
		print(self.target_names)

		print("== OUTPUT TARGETS ==")
		print(self.output_targets)
		
	
		# - Create train data generator
		self.train_data_generator= self.dl.data_generator(
			batch_size=self.batch_size, 
			shuffle=self.shuffle_train_data,
			resize=self.resize, nx=self.nx, ny=self.ny, 
			normalize=self.normalize, scale_to_abs_max=self.scale_to_abs_max, scale_to_max=self.scale_to_max,
			augment=self.augmentation,
			log_transform=self.log_transform_img,
			scale=self.scale_img, scale_factors=self.scale_img_factors,
			standardize=self.standardize_img, means=self.img_means, sigmas=self.img_sigmas,
			chan_divide=self.chan_divide, chan_mins=self.chan_mins,
			erode=self.erode, erode_kernel=self.erode_kernel,
			retsdata=False, 
			ret_classtargets=True, classtarget_map=self.classid_remap, nclasses=self.nclasses
		)

		# - Create cross validation data generator
		self.crossval_data_generator= self.dl.data_generator(
			batch_size=self.batch_size, 
			shuffle=self.shuffle_train_data,
			resize=self.resize, nx=self.nx, ny=self.ny, 
			normalize=self.normalize, scale_to_abs_max=self.scale_to_abs_max, scale_to_max=self.scale_to_max,
			augment=self.augmentation,
			log_transform=self.log_transform_img,
			scale=self.scale_img, scale_factors=self.scale_img_factors,
			standardize=self.standardize_img, means=self.img_means, sigmas=self.img_sigmas,
			chan_divide=self.chan_divide, chan_mins=self.chan_mins,
			erode=self.erode, erode_kernel=self.erode_kernel,
			retsdata=False, 
			ret_classtargets=True, classtarget_map=self.classid_remap, nclasses=self.nclasses	
		)	

		# - Create test data generator
		self.test_data_generator= self.dl.data_generator(
			batch_size=self.nsamples, 
			shuffle=False,
			resize=self.resize, nx=self.nx, ny=self.ny, 
			normalize=self.normalize, scale_to_abs_max=self.scale_to_abs_max, scale_to_max=self.scale_to_max,
			augment=False,
			log_transform=self.log_transform_img,
			scale=self.scale_img, scale_factors=self.scale_img_factors,
			standardize=self.standardize_img, means=self.img_means, sigmas=self.img_sigmas,
			chan_divide=self.chan_divide, chan_mins=self.chan_mins,
			erode=self.erode, erode_kernel=self.erode_kernel,
			retsdata=False, 
			ret_classtargets=True, classtarget_map=self.classid_remap, nclasses=self.nclasses
		)

		return 0
	

	#####################################
	##     RUN TRAIN
	#####################################
	def run_train(self):
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
		#- Create the network or load it from file?
		if self.modelfile!="":
			logger.info("Loading network architecture from file: %s, %s ..." % (self.modelfile))
			if self.__load_model(self.modelfile, self.weightfile)<0:
				logger.error("NN loading failed!")
				return -1
		else:
			logger.info("Building network architecture ...")
			if self.__create_model()<0:
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
	def run_predict(self, modelfile, weightfile):
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
		logger.info("Loading model architecture and weights from files %s %s ..." % (modelfile, weightfile))
		if self.__load_model(modelfile, weightfile)<0:
			logger.warn("Failed to load model from files!")
			return -1

		if self.model is None:
			logger.error("Loaded model is None!")
			return -1

		#===========================
		#==   PREDICT
		#===========================
		# - Get predicted output data
		logger.info("Predicting model output data ...")
		predout= self.model.predict(
			x=self.test_data_generator,	
			steps=1,
    	verbose=2,
    	workers=self.nworkers,
    	use_multiprocessing=self.use_multiprocessing
		)

		print("predout")
		print(type(predout))
		print(predout.shape)

		logger.info("Retriving target ids from predicted output ...")
		targetids_pred= np.argmax(predout, axis=None, out=None)

		print("targetids_pred")
		print(type(targetids_pred))
		print(targetids_pred.shape)

		# - Get predicted output class id
		logger.info("Predicting output classid ...")
		predclasses= self.model.predict_classes(
			x=self.test_data_generator,	
			steps=1,
    	verbose=2,
    	workers=self.nworkers,
    	use_multiprocessing=self.use_multiprocessing
		)

		print("predclasses")
		print(type(predclasses))
		print(predclasses.shape)
		
		# - Get predicted output class prob
		logger.info("Predicting output classid ...")
		predprobs= self.model.predict_proba(
			x=self.test_data_generator,	
			steps=1,
    	verbose=2,
    	workers=self.nworkers,
    	use_multiprocessing=self.use_multiprocessing
		)

		print("predprobs")
		print(type(predprobs))
		print(predprobs.shape)
		

		if type(predout)==tuple and len(predout)>0:
			self.output_data= predout[0]
		else:
			self.output_data= predout

		print("output_data shape")
		print(self.output_data.shape)	
		print(self.output_data)
		N= self.output_data.shape[0]
		Nvar= self.output_data.shape[1]
		
		
		# - Merge output data
		obj_names= np.array(self.source_names).reshape(N,1)
		obj_ids= np.array(self.source_ids).reshape(N,1)
		out_data= np.concatenate(
			(obj_names, self.output_data, obj_ids),
			axis=1
		)

		# - Save output data to file
		logger.info("Saving predicted output data to file %s ..." % (self.outfile))
		znames_counter= list(range(1, Nvar+1))
		znames= '{}{}'.format('z',' z'.join(str(item) for item in znames_counter))
		head= '{} {} {}'.format("# sname", znames, "id")
		Utils.write_ascii(out_data, self.outfile, head)	

		return 0

	#####################################
	##     CREATE MODEL
	#####################################
	
	def __create_model(self):
		""" Create the model """
				
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
		#==  BUILD MODEL
		#===========================
		# - Init model
		self.model = models.Sequential()

		# - Input layer	
		inputShape = (self.ny, self.nx, self.nchannels)
		self.inputs= Input(shape=inputShape, dtype='float', name='inputs')
		self.input_data_dim= K.int_shape(self.inputs)
		self.model.add(self.inputs)

		print("Input data dim=", self.input_data_dim)
		print("inputs shape")
		print(K.int_shape(self.inputs))

		# - Add a number of CNN layers
		for k in range(len(self.nfilters_cnn)):

			# - Add a Convolutional 2D layer
			padding= "same"
			if k==0:
				# - Set weights for the first layer
				x = layers.Conv2D(self.nfilters_cnn[k], (self.kernsizes_cnn[k], self.kernsizes_cnn[k]), strides=self.strides_cnn[k], padding=padding, kernel_initializer=weight_initializer)
			else:
				x = layers.Conv2D(self.nfilters_cnn[k], (self.kernsizes_cnn[k], self.kernsizes_cnn[k]), strides=self.strides_cnn[k], padding=padding)

			self.model.add(x)

			# - Add batch normalization?
			if self.add_batchnorm:
				x = BatchNormalization(axis=-1)
				self.model.add(x)

			# - Add Leaky RELU?	
			if self.add_leakyrelu:
				x = layers.LeakyReLU(alpha=self.leakyrelu_alpha)
			else:
				x = layers.ReLU()

			self.model.add(x)

			# - Add max pooling?
			if self.add_max_pooling:
				padding= "valid"
				x = layers.MaxPooling2D(pool_size=(self.pool_size,self.pool_size),strides=None,padding=padding)
				self.model.add(x)

		# - Add flatten layer
		x = layers.Flatten()
		self.model.add(x)		

		#===========================
		#==  MODEL OUTPUT LAYERS
		#===========================
		# - Add dense layer?
		if self.add_dense:
			for layer_size in self.dense_layer_sizes:
				x = layers.Dense(layer_size, activation=self.dense_layer_activation)
				self.model.add(x)
		
		# - Output layer
		self.outputs = layers.Dense(self.nclasses, name='outputs', activation='softmax')
		self.model.add(self.outputs)
		
		
		#print("outputs shape")
		#print(K.int_shape(self.outputs))
		
		#===========================
		#==   BUILD MODEL
		#===========================
		# - Define and compile model
		#self.model = Model(inputs=self.inputs, outputs=self.output_targets, name='classifier')
		#self.model = Model(inputs=self.inputs, outputs=self.outputs, name='classifier')
		self.model.compile(optimizer=self.optimizer, loss=self.loss_type, metrics=['accuracy', f1_score, precision, recall], run_eagerly=True)
		
		# - Print model summary
		self.model.summary()
		
		return 0

	
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
		logger.info("Start model training (dataset_size=%d, batch_size=%d, steps_per_epoch=%d) ..." % (self.nsamples, self.batch_size, steps_per_epoch))

		self.fitout= self.model.fit(
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
		self.model.save_weights('model_weights.h5')
		
		# -Save the model architecture in json format
		logger.info("Saving NN architecture in json format ...")
		with open('model_architecture.json', 'w') as f:
			f.write(self.model.to_json())

		#- Save the model
		logger.info("Saving full NN model ...")
		self.model.save('model.h5')
		
		# - Save the network architecture diagram
		logger.info("Saving network model architecture to file ...")
		plot_model(self.model, to_file=self.outfile_model, show_shapes=True)
		

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
		#==   SAVE OUTPUT DATA
		#================================
		logger.info("Saving output data to file ...")
		
		self.output_data= self.model.predict(
			x=self.test_data_generator,	
			steps=1,
    	verbose=2,
    	workers=self.nworkers,
    	use_multiprocessing=self.use_multiprocessing
		)

		print("output_data shape")
		print(self.output_data.shape)	
		print(self.output_data)
		N= self.output_data.shape[0]
		Nvar= self.output_data.shape[1]
		
		
		# - Merge output data
		obj_names= np.array(self.source_names).reshape(N,1)
		obj_ids= np.array(self.source_ids).reshape(N,1)
		out_data= np.concatenate(
			(obj_names, self.output_data, obj_ids),
			axis=1
		)

		znames_counter= list(range(1,Nvar+1))
		znames= '{}{}'.format('z',' z'.join(str(item) for item in znames_counter))
		head= '{} {} {}'.format("# sname",znames,"id")
		Utils.write_ascii(out_data, self.outfile, head)	


		return 0


	
	
	#####################################
	##     LOAD MODEL
	#####################################
	def __load_model(self, modelfile):
		""" Load model and weights from input h5 file """

		#==============================
		#==   LOAD MODEL ARCHITECTURE
		#==============================
		try:
			self.model= load_model(modelfile)
			
		except Exception as e:
			logger.warn("Failed to load model from file %s (err=%s)!" % (modelfile, str(e)))
			return -1

		if not self.model or self.model is None:
			logger.error("Model object is None, loading failed!")
			return -1

		
		#===========================
		#==   SET LOSS & METRICS
		#===========================	
		self.model.compile(optimizer=self.optimizer, loss=self.loss_type, run_eagerly=True)
		
		# - Print and draw model
		self.model.summary()
		plot_model(self.model,to_file='model.png',show_shapes=True)

		return 0


	def __load_model(self, modelfile_json, weightfile):
		""" Load model and weights from input h5 file """

		#==============================
		#==   LOAD MODEL ARCHITECTURE
		#==============================
		# - Load model
		try:
			self.model = model_from_json(open(modelfile_json).read())
			self.model.load_weights(weightfile)

		except Exception as e:
			logger.warn("Failed to load model from file %s (err=%s)!" % (modelfile_json, str(e)))
			return -1

		if not self.model or self.model is None:
			logger.error("Model object is None, loading failed!")
			return -1

		#===========================
		#==   SET LOSS & METRICS
		#===========================	
		self.model.compile(optimizer=self.optimizer, loss=self.loss_type, run_eagerly=True)
		
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



