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
from itertools import chain
import json

##############################
##     GLOBAL VARS
##############################
from sclassifier import logger

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
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

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
from .data_generator import DataGenerator
from .tf_utils import ChanMinMaxNorm, ChanMaxScale, ChanMeanRatio, ChanMaxRatio, ChanPosDef

##################################
##     METRICS CLASS
##################################
def recall_metric(y_true, y_pred):
	""" Compute recall=TP/(TP+FN) """
	#y_true = K.ones_like(y_true) 
	#true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	#all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  #recall = true_positives / (all_positives + K.epsilon())
	
	# - Convert one-hot encoding to labels 
	l_true= tf.argmax(y_true, axis = 1)
	l_pred= tf.argmax(y_pred, axis = 1)

	# - Compute recall=TP/all
	#TP= tf.math.count_nonzero(l_true==l_pred)
	#TP_FN= tf.size(l_true)
	#recall= tf.math.divide(tf.cast(TP,tf.float32), tf.cast(TP_FN,tf.float32))	
 
	recall= recall_score(l_true, l_pred, average='micro')

	return recall

def precision_metric(y_true, y_pred):
	""" Compute precision=TP/(TP+FN) """
	#y_true = K.ones_like(y_true) 
	#true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	#predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	#precision = true_positives / (predicted_positives + K.epsilon())

	# - Convert one-hot encoding to labels 
	l_true= tf.argmax(y_true, axis = 1)
	l_pred= tf.argmax(y_pred, axis = 1)
	
	# - Compute precision=TP/(TP+FN)
	precision= precision_score(l_true, l_pred, average='micro')

	return precision

def f1score_metric(y_true, y_pred):
	""" Compute F1-score metric"""
	#prec = precision_metric(y_true, y_pred)
	#rec = recall_metric(y_true, y_pred)
	#f1score= 2*((prec*rec)/(prec+rec+K.epsilon()))

	# - Convert one-hot encoding to labels 
	l_true= tf.argmax(y_true, axis = 1)
	l_pred= tf.argmax(y_pred, axis = 1)

	# - Compute F1-score
	f1score= f1_score(l_true, l_pred, average='weighted')

	return f1score



##################################
##     SClassifierNN CLASS
##################################
class SClassifierNN(object):
	""" Source classifier class """
	
	def __init__(self, data_generator, multiclass=True, multilabel=False):
		""" Return a SClassifierNN object """

		self.dg= data_generator
		self.dg_cv= None
		self.has_cvdata= False
		self.multiclass= multiclass
		self.multilabel= multilabel

		self.excluded_objids_train= [-1,0] # Sources with these ids are considered not labelled and therefore excluded from training or metric calculation

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
		self.flattened_inputs= None	
		self.input_data_dim= 0
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
		self.use_predefined_arch= False
		self.predefined_arch= "resnet50"
		self.channorm_min= 0.0
		self.channorm_max= 1.0
		self.nfilters_cnn= [32,64,128]
		self.kernsizes_cnn= [3,5,7]
		self.strides_cnn= [2,2,2]
		self.add_max_pooling= False
		self.pool_size= 2
		self.add_leakyrelu= False
		self.leakyrelu_alpha= 0.3
		self.add_batchnorm= True
		self.activation_fcn_cnn= "relu"
		self.add_dense= False
		self.dense_layer_sizes= [16] 
		self.dense_layer_activation= 'relu'
		self.add_dropout_layer= False
		self.dropout_rate= 0.5
		self.add_conv_dropout_layer= False
		self.conv_dropout_rate= 0.2
		self.add_chanminmaxnorm_layer= False
		self.add_chanmaxscale_layer= False
		self.add_chanmeanratio_layer= False
		self.add_chanmaxratio_layer= False
		self.add_chanposdef_layer= False
		self.use_global_avg_pooling= False

		# - Training options
		self.batch_size= 32
		self.nepochs= 10
		self.learning_rate= 1.e-4
		self.optimizer_default= 'adam'
		self.optimizer= 'adam' # 'rmsprop'
		self.weight_init_seed= None
		self.shuffle_train_data= True
		self.augment_scale_factor= 1
		self.loss_type= "categorical_crossentropy" # Loss are: binary_crossentropy (binary classification), categorical_crossentropy (multiclass classification), tf.nn.sigmoid_cross_entropy_with_logits (multilabel classification)
		self.load_cv_data_in_batches= True
		self.save_model_period= 100

		self.balance_classes= False
		self.class_probs= {}

		self.__set_target_labels(multiclass)

		self.sigmoid_thr= 0.5 # used in multilabel

		# *****************************
		# ** Output
		# *****************************
		self.outfile_model= 'model.png'
		self.outfile_metrics= "metrics.dat"
		self.outfile_loss= 'losses.png'
		self.outfile_nnout_metrics= 'losses.dat'
		self.outfile= 'classified_data.dat'
		self.outfile_cm= "confusion_matrix.dat"
		self.outfile_cm_norm= "confusion_matrix_norm.dat"
		
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

	def set_classid_remap(self, cid_remap):
		""" Set class id remap and update inverted map """
	
		self.classid_remap= cid_remap
		self.classid_remap_inv= {v: k for k, v in self.classid_remap.items()}
		
	def set_classid_label_map(self, cid_label_map):
		""" Set class id label map and update inverted map """
	
		self.classid_label_map= cid_label_map
		self.classid_label_map_inv= {v: k for k, v in self.classid_label_map.items()}
		

		
	def __multilabel_loss(self, y_true, y_pred):
		""" Loss function used for multilabel classification """
		return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

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

		# - Set model targets
		self.target_ids= []
		self.target_ids_all= []

		for i in range(self.nsamples):
			source_name= self.source_names[i]
			obj_id= self.source_ids[i]
			label= self.source_labels[i]
			target_id= self.classid_remap[obj_id] # remap obj id to target class ids
			
			add_to_train_list= True
			for obj_id_excl in self.excluded_objids_train:
				if obj_id==obj_id_excl:
					add_to_train_list=False
					break

			if add_to_train_list:
				self.target_ids.append(target_id)
				self.target_ids_all.append(target_id)
			else:
				self.target_ids_all.append(-1)

		# - Set target names
		self.target_names= []
		if self.target_ids:
			logger.info("#%d labelled sources found in dataset ..." % (len(self.target_ids)))
			self.target_names= [self.target_label_map[item] for item in set(sorted(self.target_ids))]
		else:
			logger.info("No known class found in dataset (not a problem if predicting) ...")

		print("== TARGET NAMES ==")
		print(self.target_names)
		
		# - Create data generators
		logger.info("Creating data generators ...")
		self.__set_data_generators()

		return 0
	
	#####################################
	##     SET DATA (MULTILABEL)
	#####################################
	def __set_data_multilabel(self):
		""" Create dataset for multi-label classification """

		# - Retrieve info from data loader
		self.nchannels= self.dg.nchannels
		self.source_labels= self.dg.labels  # this should be a 2D list
		self.source_ids= self.dg.classids   # this should be a 2D list
		self.source_names= self.dg.snames
		self.nsamples= len(self.source_labels)
	
		# - Set model targets
		self.target_ids= []
		self.target_ids_all= []

		for i in range(self.nsamples):
			source_name= self.source_names[i]
			obj_ids= self.source_ids[i]
			labels= self.source_labels[i]

			# - Check if input data has single or multi-labels
			has_multilabels= isinstance(obj_ids, list) and isinstance(labels, list)
			if not has_multilabels:
				logger.warn("Class id/label for input data no. %d (name=%s) are not lists (as required for multilabel classification), skip it ..." % (i+1, source_name))
				continue

			# - Set target ids
			target_id_list= []
			add_to_train_list= True

			for obj_id in obj_ids:
				target_id_list.append(self.classid_remap[obj_id]) # remap obj id to target class ids)

				for obj_id_excl in self.excluded_objids_train:
					if obj_id==obj_id_excl:
						add_to_train_list=False

			##print("target_id_list=%s, add_to_train_list=%d, excluded_objids_train=%s" % (str(target_id_list), add_to_train_list, str(self.excluded_objids_train)))
			
			if add_to_train_list:
				self.target_ids.append(target_id_list)
				self.target_ids_all.append(target_id_list)
			else:
				self.target_ids_all.append([-1])		

		# - Check there are data left
		if not self.target_ids_all:
			logger.error("No data left (check given labels)!")
			return -1

		# - Set target names
		self.target_names= []
		if self.target_ids:
			logger.info("#%d labelled sources found in dataset ..." % (len(self.target_ids)))
			self.target_names= [self.target_label_map[item] for item in set(sorted(list(chain.from_iterable(self.target_ids))))] # chain.from_iterable flattens 2D list to 1D
		else:
			logger.info("No known class found in dataset (not a problem if predicting) ...")

		print("== TARGET NAMES ==")
		print(self.target_names)

		# - Create data generators
		logger.info("Creating data generators ...")
		self.__set_data_generators()
		
		return 0

	#####################################
	##     SET DATA GENERATOR
	#####################################
	def __set_data_generators(self):
		""" Create data generators """

		# - Create train data generator
		self.train_data_generator= self.dg.generate_cnn_data(
			batch_size=self.batch_size, 
			shuffle=self.shuffle_train_data,
			classtarget_map=self.classid_remap, nclasses=self.nclasses,
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

			self.crossval_data_generator= self.dg_cv.generate_cnn_data(
				batch_size=batch_size_cv, 
				shuffle=False,
				classtarget_map=self.classid_remap, nclasses=self.nclasses
			)

		
		# - Create test data generator
		logger.info("Creating test data generator (deep-copying train data generator) ...")
		self.dg_test= deepcopy(self.dg)
		logger.info("Disabling data augmentation in test data generator ...")
		self.dg_test.disable_augmentation()

		self.test_data_generator= self.dg_test.generate_cnn_data(
			#batch_size=self.nsamples,
			batch_size=1,
			shuffle=False,
			classtarget_map=self.classid_remap, nclasses=self.nclasses
		)


	#####################################
	##     RUN TRAIN
	#####################################
	def run_train(self):
		""" Run network training """

		#===========================
		#==   SET TRAINING DATA
		#===========================	
		logger.info("Setting training data from data loader ...")
		if self.multilabel:
			status= self.__set_data_multilabel()
		else:
			status= self.__set_data()
		if status<0:
			logger.error("Train data set failed!")
			return -1

		#===========================
		#==   BUILD NN
		#===========================
		#- Create the network or load it from file?
		if self.modelfile!="":
			logger.info("Loading network architecture from file: %s, %s ..." % (self.modelfile, self.weightfile))
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
		if self.multilabel:
			status= self.__set_data_multilabel()
		else:
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
			steps=self.nsamples,
    	verbose=2,
    	workers=self.nworkers,
    	use_multiprocessing=self.use_multiprocessing
		)

		print("predout")
		print(type(predout))
		print(predout.shape)

		# - Save prediction data to file
		logger.info("Saving predicted data to file ...")
		if self.multilabel:
			self.__save_predicted_data_multilabel(predout)
		else:
			self.__save_predicted_data(predout)

		#================================
		#==   COMPUTE AND SAVE METRICS
		#================================
		# - Performed only for data with a class label set (not for unknown sources)
		if self.target_ids:
			logger.info("Compute and save metrics ...")
			if self.multilabel:
				self.__compute_metrics_multilabel()
			else:
				self.__compute_metrics()
			
		return 0

	
	#####################################
	##     SAVE PREDICTED DATA
	#####################################
	def __save_predicted_data(self, predout):
		""" Save prediction data """

		# - Convert one-hot encoding to target ids
		logger.info("Retrieving target ids from predicted output ...")	
		self.targets_pred= np.argmax(predout, axis=1)

		print("targets_pred")
		print(self.targets_pred)
		print(type(self.targets_pred))
		print(self.targets_pred.shape)

		# - Get predicted output class id
		logger.info("Computing predicted class ids from targets ...")
		self.classids_pred= [self.classid_remap_inv[item] for item in self.targets_pred]
		
		print("classids_pred")
		print(self.classids_pred)
		print(type(self.classids_pred))
		
		# - Get predicted output class prob
		logger.info("Predicting output classid ...")
		self.probs_pred= [predout[i,self.targets_pred[i]] for i in range(predout.shape[0])]

		print("probs_pred")
		print(self.probs_pred)
		print(type(self.probs_pred))
		
		# - Save predicted data to file
		logger.info("Saving prediction data to file %s ..." % (self.outfile))
		N= predout.shape[0]
		snames= np.array(self.source_names).reshape(N,1)
		objids= np.array(self.source_ids).reshape(N,1)
		objids_pred= np.array(self.classids_pred).reshape(N,1)
		probs_pred= np.array(self.probs_pred).reshape(N,1)

		outdata= np.concatenate(
			(snames, objids, objids_pred, probs_pred),
			axis=1
		)

		head= "# sname id id_pred prob"
		Utils.write_ascii(outdata, self.outfile, head)
		
		return 0

	##########################################
	##     SAVE PREDICTED DATA (MULTILABEL)
	##########################################
	def __save_predicted_data_multilabel(self, predout):
		""" Save prediction data for multilabel classification """

		# - Convert one-hot encoding to target ids (2D list)
		logger.info("Retrieving target ids from predicted output ...")
		self.targets_pred= [list( np.flatnonzero(row > self.sigmoid_thr) ) for row in predout]

		# - Check if some source was not classified (e.g. all probs are below threshold)
		#   In this case the list will be empty, replace it with a [-1]
		for i in range(len(self.targets_pred)):
			if not self.targets_pred[i]:
				self.targets_pred[i]= [-1]

		print("targets_pred")
		print(self.targets_pred)
		print(type(self.targets_pred))

		# - Get predicted output class id (2D list)
		logger.info("Computing predicted class ids from targets ...")
		self.classids_pred= [[self.classid_remap_inv[target_id] for target_id in item] for item in self.targets_pred]
		
		print("classids_pred")
		print(self.classids_pred)
		print(type(self.classids_pred))
		
		# - Get predicted output class prob (2D list)
		logger.info("Predicting output classid ...")
		self.probs_pred= [list(row[row>self.sigmoid_thr].astype('float')) for row in predout]
		
		# - Check if some source was not classified (e.g. all probs are below threshold)
		#   In this case the list will be empty, replace it with a [0.]
		for i in range(len(self.probs_pred)):
			if not self.probs_pred[i]:
				self.probs_pred[i]= [0.]

		print("probs_pred")
		print(self.probs_pred)
		print(type(self.probs_pred))

		# - Get original labels from target ids
		labels= [[self.target_label_map[target_id] for target_id in item] for item in self.target_ids_all]

		# - Get predicted labels from predicted target ids
		labels_pred= [[self.target_label_map[target_id] for target_id in item] for item in self.targets_pred]
		
		# - Save predicted data to file
		logger.info("Saving prediction data to file %s ..." % (self.outfile))
		ddlist= []
		N= predout.shape[0]
		for i in range(N):

			#print("type(target_id)")
			#print(type(self.target_ids_all[i]))
			#print(type(self.target_ids_all[i][0]))

			print("type(target_id_pred)")
			print(type(self.targets_pred[i]))
			print(type(self.targets_pred[i][0]))

			dd= {
				"sname": self.source_names[i],
				"id": self.source_ids[i],
				"target_id": [int(item) for item in self.target_ids_all[i]],
				"label": labels[i],
				"id_pred": self.classids_pred[i],
				"target_id_pred": [int(item) for item in self.targets_pred[i]],
				"label_pred": labels_pred[i],
				"prob": self.probs_pred[i]
			}
			ddlist.append(dd)

		outdata= {"data": ddlist}

		with open(self.outfile, "w") as fp:
			json.dump(outdata, fp)
		
		
		return 0


	#####################################
	##     COMPUTE AND SAVE METRICS
	#####################################
	def __compute_metrics(self):
		""" Compute and save metrics """

		# - Compute target/pred vectors (must have a class label set)
		y_true= []
		y_pred= []
		for i in range(len(self.target_ids_all)):
			target_id= self.target_ids_all[i]
			pred_id= self.targets_pred[i]
			if target_id<0:
				continue
			y_true.append(target_id)
			y_pred.append(pred_id)

		# - Compute classification metrics
		logger.info("Computing classification metrics on predicted data ...")
		##report= classification_report(self.target_ids, self.targets_pred, target_names=self.target_names, output_dict=True)
		report= classification_report(y_true, y_pred, target_names=self.target_names, output_dict=True)
		self.accuracy= report['accuracy']
		self.precision= report['weighted avg']['precision']
		self.recall= report['weighted avg']['recall']    
		self.f1score= report['weighted avg']['f1-score']

		self.class_precisions= []
		self.class_recalls= []  
		self.class_f1scores= []
		for class_name in self.target_names:
			class_precision= report[class_name]['precision']
			class_recall= report[class_name]['recall']    
			class_f1score= report[class_name]['f1-score']
			self.class_precisions.append(class_precision)
			self.class_recalls.append(class_recall)
			self.class_f1scores.append(class_f1score)
			
		logger.info("accuracy=%f" % (self.accuracy))
		logger.info("precision=%f" % (self.precision))
		logger.info("recall=%f" % (self.recall))
		logger.info("f1score=%f" % (self.f1score))
		logger.info("--> Metrics per class")
		print("classnames")
		print(self.target_names)
		print("precisions")
		print(self.class_precisions)
		print("recall")
		print(self.class_recalls)
		print("f1score")
		print(self.class_f1scores)

		# - Retrieving confusion matrix
		logger.info("Retrieving confusion matrix ...")
		#cm= confusion_matrix(self.target_ids, self.targets_pred)
		#cm_norm= confusion_matrix(self.target_ids, self.targets_pred, normalize="true")
		cm= confusion_matrix(y_true, y_pred)
		cm_norm= confusion_matrix(y_true, y_pred, normalize="true")

		print("confusion matrix")
		print(cm)

		print("confusion matrix (norm)")
		print(cm_norm)

		# - Saving metrics to file
		logger.info("Saving metrics to file %s ..." % (self.outfile_metrics))
		metrics= [self.accuracy, self.precision, self.recall, self.f1score]
		metric_names= ["accuracy","precision","recall","f1score"]
		
		for i in range(len(self.target_names)):
			classname= self.target_names[i]
			precision= self.class_precisions[i]
			recall= self.class_recalls[i]
			f1score= self.class_f1scores[i]
			metrics.append(precision)
			metrics.append(recall)
			metrics.append(f1score)
			metric_names.append("precision_" + classname)
			metric_names.append("recall_" + classname)
			metric_names.append("f1score_" + classname)
			
		Nmetrics= len(metrics)
		metric_data= np.array(metrics).reshape(1,Nmetrics)

		metric_names_str= ' '.join(str(item) for item in metric_names)
		head= '{} {}'.format("# ",metric_names_str)

		print("metric_data")
		print(metrics)
		print(len(metrics))
		print(metric_data.shape)
		
		Utils.write_ascii(metric_data, self.outfile_metrics, head)

		# - Save confusion matrix to file
		logger.info("Saving confusion matrix to file %s ..." % (self.outfile_cm))		
		np.savetxt(self.outfile_cm, cm, delimiter=',')
		np.savetxt(self.outfile_cm_norm, cm_norm, delimiter=',')


		return 0


	def __compute_metrics_multilabel(self):
		""" Compute metrics for multilabel classification """

		# - Compute target/pred vectors (must have a class label set)
		target_ids_true= []
		target_ids_pred= []
		for i in range(len(self.target_ids_all)):
			target_ids= self.target_ids_all[i]
			pred_ids= self.targets_pred[i]
			if len(target_ids)==1 and target_ids[0]<0:
				continue
			target_ids_true.append(target_ids)
			target_ids_pred.append(pred_ids)

		# - Convert target vector in hot encoding format (needed for metrics), e.g. [0,1,0,0,1,0]
		mlb = MultiLabelBinarizer(classes=np.arange(0,self.nclasses))
		y_true= mlb.fit_transform(target_ids_true)
		y_pred= mlb.fit_transform(target_ids_pred)
		
		# - Compute classification metrics
		logger.info("Computing classification metrics on predicted data ...")
		report= classification_report(y_true, y_pred, target_names=self.target_names, output_dict=True)
		self.accuracy= report['accuracy']
		self.precision= report['weighted avg']['precision']
		self.recall= report['weighted avg']['recall']    
		self.f1score= report['weighted avg']['f1-score']

		print("report")
		print(report)

		self.class_precisions= []
		self.class_recalls= []  
		self.class_f1scores= []
		for class_name in self.target_names:
			class_precision= report[class_name]['precision']
			class_recall= report[class_name]['recall']    
			class_f1score= report[class_name]['f1-score']
			self.class_precisions.append(class_precision)
			self.class_recalls.append(class_recall)
			self.class_f1scores.append(class_f1score)
			
		logger.info("accuracy=%f" % (self.accuracy))
		logger.info("precision=%f" % (self.precision))
		logger.info("recall=%f" % (self.recall))
		logger.info("f1score=%f" % (self.f1score))
		logger.info("--> Metrics per class")
		print("classnames")
		print(self.target_names)
		print("precisions")
		print(self.class_precisions)
		print("recall")
		print(self.class_recalls)
		print("f1score")
		print(self.class_f1scores)

		return 0


	#####################################
	##     CREATE MODEL
	#####################################
	def __create_model(self):
		""" Create model (custom vs predefined) """

		if self.use_predefined_arch:
			return self.__create_predefined_model()
		else:
			return self.__create_custom_model()

	#####################################
	##     CREATE PREDEFINED MODEL
	#####################################
	def __create_predefined_model(self):
		""" Create model using a predefined architecture as backbone """

		#===========================
		#==  INIT MODEL
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

		#===========================
		#==  BACKBONE NET
		#===========================
		# - Add backbone net
		if self.predefined_arch=="resnet50":
			backbone= tf.keras.applications.resnet50.ResNet50(
				include_top=False, # disgard the fully-connected layer as we are training from scratch
				weights=None,  # random initialization
				input_tensor=self.inputs,
				input_shape=inputShape,
				pooling=None
			)
		elif self.predefined_arch=="resnet101":
			backbone= tf.keras.applications.resnet50.ResNet50(
				include_top=False, # disgard the fully-connected layer as we are training from scratch
				weights=None,  # random initialization
				input_tensor=self.inputs,
				input_shape=inputShape,
				pooling=None
			)
		else:
			logger.error("Unknown/unsupported predefined backbone architecture given (%s)!" % (self.predefined_arch))
			return -1
	
		self.model.add(backbone)

		# - Add flatten layer or global average pooling?
		if self.use_global_avg_pooling:
			x= layers.GlobalAveragePooling2D()
		else:
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

				if self.add_dropout_layer:
					x= layers.Dropout(self.dropout_rate)
					self.model.add(x)
			
		# - Output layer
		#   NB: see https://glassboxmedicine.com/2019/05/26/classification-sigmoid-vs-softmax/
		if self.multilabel:
			self.outputs = layers.Dense(self.nclasses, name='outputs', activation='sigmoid') 
		else:
			self.outputs = layers.Dense(self.nclasses, name='outputs', activation='softmax') 

		self.model.add(self.outputs)
		
		#print("outputs shape")
		#print(K.int_shape(self.outputs))
		
		#===========================
		#==   BUILD MODEL
		#===========================
		# - Define and compile model
		if self.multilabel:
			self.model.compile(
				optimizer=self.optimizer, 
				loss=self.__multilabel_loss, 
				#metrics=['accuracy', f1score_metric, precision_metric, recall_metric], 
				run_eagerly=True
			)
		
		else:
			self.model.compile(
				optimizer=self.optimizer, 
				loss=self.loss_type, 
				metrics=['accuracy', f1score_metric, precision_metric, recall_metric], 
				run_eagerly=True
		)
		
		# - Print model summary
		self.model.summary()
		
		return 0

	#####################################
	##     CREATE CUSTOM MODEL
	#####################################
	def __create_custom_model(self):
		""" Create the model """
		
		if self.add_chanmaxratio_layer:
			return self.__create_custom_model_nonsequencial()
		elif self.add_chanmeanratio_layer:
			return self.__create_custom_model_nonsequencial()
		else:
			return self.__create_custom_model_sequencial()

	#####################################
	##     CREATE CUSTOM MODEL (SEQUENTIAL)
	#####################################
	def __create_custom_model_sequencial(self):
		""" Create the model in a sequencial way """

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

		# - Add channel to make images always positive
		if self.add_chanposdef_layer:
			x= ChanPosDef(name='chan_posdef_maker')
			self.model.add(x)

		# - Add channel min-max normalization layer?
		if self.add_chanminmaxnorm_layer:
			x= ChanMinMaxNorm(name='chan_minmax_norm')
			self.model.add(x)

		# - Add channel max scale layer?
		if self.add_chanmaxscale_layer:
			x= ChanMaxScale(name='chan_max_scaler')
			self.model.add(x)

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
				x = layers.MaxPooling2D(pool_size=(self.pool_size,self.pool_size), strides=None, padding=padding)
				self.model.add(x)

			# - Add dropout?
			if self.add_conv_dropout_layer:
				x= layers.Dropout(self.conv_dropout_rate)
				self.model.add(x)
			

		# - Add flatten layer or global average pooling
		if self.use_global_avg_pooling:
			x= layers.GlobalAveragePooling2D()
		else:
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

				if self.add_dropout_layer:
					x= layers.Dropout(self.dropout_rate)
					self.model.add(x)
			
		# - Output layer
		if self.multilabel:
			self.outputs = layers.Dense(self.nclasses, name='outputs', activation='sigmoid')
		else:
			self.outputs = layers.Dense(self.nclasses, name='outputs', activation='softmax')
		
		self.model.add(self.outputs)
		

		#print("outputs shape")
		#print(K.int_shape(self.outputs))
		
		#===========================
		#==   BUILD MODEL
		#===========================
		# - Define and compile model
		if self.multilabel:
			self.model.compile(
				optimizer=self.optimizer, 
				loss=self.__multilabel_loss,
				#metrics=['accuracy', f1score_metric, precision_metric, recall_metric], 
				run_eagerly=True
			)
		else:
			self.model.compile(
				optimizer=self.optimizer, 
				loss=self.loss_type, 
				metrics=['accuracy', f1score_metric, precision_metric, recall_metric], 
				run_eagerly=True
			)
		
		# - Print model summary
		self.model.summary()
		
		return 0


	########################################
	##     CREATE CUSTOM MODEL (NON SEQUENTIAL)
	########################################
	def __create_custom_model_nonsequencial(self):
		""" Create the model in a non-sequencial way """

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
		# - Input layer	
		inputShape = (self.ny, self.nx, self.nchannels)
		self.inputs= Input(shape=inputShape, dtype='float', name='inputs')
		self.input_data_dim= K.int_shape(self.inputs)
		x= self.inputs

		print("Input data dim=", self.input_data_dim)
		print("inputs shape")
		print(K.int_shape(self.inputs))

		# - Compute chan max ratios?
		if self.add_chanmaxratio_layer:
			x_maxratios= ChanMaxRatio(name='chanmaxratios')(self.inputs)
			x_maxratios_flattened= layers.Flatten()(x_maxratios)

		# - Compute chan mean ratios?
		if self.add_chanmeanratio_layer:
			x_meanratios= ChanMeanRatio(name='chanmeanratios')(self.inputs)
			x_meanratios_flattened= layers.Flatten()(x_meanratios)

		# - Add channel to make images always positive
		if self.add_chanposdef_layer:
			x= ChanPosDef(name='chan_posdef_maker')(x)

		# - Add channel min-max normalization layer?
		if self.add_chanminmaxnorm_layer:
			x= ChanMinMaxNorm(name='chan_minmax_norm')(x)

		# - Add channel max scale layer?
		if self.add_chanmaxscale_layer:
			x= ChanMaxScale(name='chan_max_scaler')(x)
			
		# - Add a number of CNN layers
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
				x = layers.MaxPooling2D(pool_size=(self.pool_size,self.pool_size),strides=None,padding=padding)(x)
		
			# - Add dropout?
			if self.add_conv_dropout_layer:
				x= layers.Dropout(self.conv_dropout_rate)(x)

		# - Add flatten layer
		if self.use_global_avg_pooling:
			x= layers.GlobalAveragePooling2D()(x)
		else:
			x = layers.Flatten()(x)

		# - Concatenate flattened CNN output + channel max scale layer?
		xconcat_list= [x]
		if self.add_chanmaxratio_layer:
			xconcat_list.append(x_maxratios_flattened)
		if self.add_chanmeanratio_layer:
			xconcat_list.append(x_meanratios_flattened)
		
		if len(xconcat_list)>1:
			#x= layers.Concatenate(axis=1)([x, x_maxratios_flattened])
			x= layers.Concatenate(axis=1)(xconcat_list)

		#===========================
		#==  MODEL OUTPUT LAYERS
		#===========================
		# - Add dense layer?
		if self.add_dense:
			for layer_size in self.dense_layer_sizes:
				x = layers.Dense(layer_size, activation=self.dense_layer_activation)(x)

				if self.add_dropout_layer:
					x= layers.Dropout(self.dropout_rate)(x)
			
		# - Output layer
		if self.multilabel:
			self.outputs = layers.Dense(self.nclasses, name='outputs', activation='sigmoid')(x)
		else:
			self.outputs = layers.Dense(self.nclasses, name='outputs', activation='softmax')(x)
		
		#print("outputs shape")
		#print(K.int_shape(self.outputs))
		
		#===========================
		#==   BUILD MODEL
		#===========================
		# - Define and compile model
		self.model = Model(inputs=self.inputs, outputs=self.outputs, name='classifier')

		if self.multilabel:
			self.model.compile(
				optimizer=self.optimizer, 
				loss=self.__multilabel_loss,
				#metrics=['accuracy', f1score_metric, precision_metric, recall_metric], 
				run_eagerly=True
			)
		else:
			self.model.compile(
				optimizer=self.optimizer, 
				loss=self.loss_type, 
				metrics=['accuracy', f1score_metric, precision_metric, recall_metric], 
				run_eagerly=True
			)
		
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

		# - Set validation steps
		val_steps_per_epoch= self.validation_steps
		if self.has_cvdata:
			if self.load_cv_data_in_batches:
				val_steps_per_epoch= self.nsamples_cv // self.batch_size
			else:
				val_steps_per_epoch= 1

		# - Set callback
		cb= tf.keras.callbacks.ModelCheckpoint('weights{epoch:08d}.h5', save_weights_only=False, period=self.save_model_period)

		#===========================
		#==   TRAIN MODEL
		#===========================
		logger.info("Start model training (dataset_size=%d, batch_size=%d, steps_per_epoch=%d, val_steps_per_epoch=%d) ..." % (self.nsamples, self.batch_size, steps_per_epoch, val_steps_per_epoch))

		self.fitout= self.model.fit(
			x=self.train_data_generator,
			epochs=self.nepochs,
			steps_per_epoch=steps_per_epoch,
			validation_data=self.crossval_data_generator,
			validation_steps=val_steps_per_epoch,
			#callbacks=[cb],
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
		#==   SAVE OUTPUT DATA
		#================================
		logger.info("Saving output data to file ...")
		if self.multilabel:
			self.__save_model_output_data_multilabel()
		else:
			self.__save_model_output_data()
		
		return 0



	def __save_model_output_data(self):
		""" Save model output data to file """
	
		self.output_data= self.model.predict(
			x=self.test_data_generator,	
			steps=self.nsamples,
    	verbose=2,
    	workers=self.nworkers,
    	use_multiprocessing=self.use_multiprocessing
		)

		print("output_data shape")
		print(self.output_data.shape)	
		#print(self.output_data)
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

	

	def __save_model_output_data_multilabel(self):
		""" Save model output data to file """

		self.output_data= self.model.predict(
			x=self.test_data_generator,	
			steps=self.nsamples,
    	verbose=2,
    	workers=self.nworkers,
    	use_multiprocessing=self.use_multiprocessing
		)

		print("output_data shape")
		print(self.output_data.shape)	
		#print(self.output_data)
		N= self.output_data.shape[0]
		Nvar= self.output_data.shape[1]

		# - Save predicted data to file
		logger.info("Saving prediction data to file %s ..." % (self.outfile))
		dd_list= []
		
		for i in range(N):
			dd= {
				"sname": self.source_names[i],
				"id": self.source_ids[i],
				"probs": self.output_data[i].tolist()
			}
			dd_list.append(dd)

		out_data= {"data": dd_list}

		with open(self.outfile, "w") as fp:
			json.dump(out_data, fp)

		return 0
	
	#####################################
	##     LOAD MODEL
	#####################################
	def __load_model(self, modelfile):
		""" Load model and weights from input h5 file """

		#==============================
		#==   LOAD MODEL ARCHITECTURE
		#==============================
		custom_objects= None
		if self.add_chanminmaxnorm_layer:
			custom_objects['ChanMinMaxNorm']= ChanMinMaxNorm 
		if self.add_chanmaxscale_layer:
			custom_objects['ChanMaxScale']= ChanMaxScale 
		if self.add_chanmeanratio_layer:
			custom_objects['ChanMeanRatio']= ChanMeanRatio
		if self.add_chanmaxratio_layer:
			custom_objects['ChanMaxRatio']= ChanMaxRatio
		if self.add_chanposdef_layer:
			custom_objects['ChanPosDef']= ChanPosDef

		if self.multilabel:
			custom_objects['loss']= self.__multilabel_loss

		print("== custom_objects ==")
		print(custom_objects)

		try:
			self.model= load_model(modelfile, custom_objects=custom_objects, compile=False)
			
		except Exception as e:
			logger.warn("Failed to load model from file %s (err=%s)!" % (modelfile, str(e)))
			return -1

		if not self.model or self.model is None:
			logger.error("Model object is None, loading failed!")
			return -1

		
		#===========================
		#==   SET LOSS & METRICS
		#===========================
		if self.multilabel:
			self.model.compile(
				optimizer=self.optimizer, 
				loss=self.__multilabel_loss,
				#metrics=['accuracy', f1score_metric, precision_metric, recall_metric], 
				run_eagerly=True
			)
		else:
			self.model.compile(
				optimizer=self.optimizer, 
				loss=self.loss_type, 
				metrics=['accuracy', f1score_metric, precision_metric, recall_metric], 
				run_eagerly=True
			)
		
		# - Print and draw model
		self.model.summary()
		plot_model(self.model,to_file='model.png',show_shapes=True)

		return 0


	def __load_model(self, modelfile, weightfile):
		""" Load model and weights from input h5 file """

		#==============================
		#==   LOAD MODEL ARCHITECTURE
		#==============================
		# - Set custom objects
		custom_objects={'recall_metric': recall_metric, 'precision_metric': precision_metric, 'f1score_metric': f1score_metric}
		if self.add_chanminmaxnorm_layer:
			custom_objects['ChanMinMaxNorm']= ChanMinMaxNorm 
		if self.add_chanmaxscale_layer:
			custom_objects['ChanMaxScale']= ChanMaxScale 
		if self.add_chanmeanratio_layer:
			custom_objects['ChanMeanRatio']= ChanMeanRatio
		if self.add_chanmaxratio_layer:
			custom_objects['ChanMaxRatio']= ChanMaxRatio
		if self.add_chanposdef_layer:
			custom_objects['ChanPosDef']= ChanPosDef

		if self.multilabel:
			custom_objects['loss']= self.__multilabel_loss

		print("== custom_objects ==")
		print(custom_objects)

		# - Load model
		try:
			####self.model = model_from_json(open(modelfile_json).read())
			#self.model = load_model(modelfile, custom_objects={'recall_metric': recall_metric, 'precision_metric': precision_metric, 'f1score_metric': f1score_metric})
			self.model = load_model(modelfile, custom_objects=custom_objects, compile=False)
			self.model.load_weights(weightfile)

		except Exception as e:
			logger.warn("Failed to load model from file %s (err=%s)!" % (modelfile, str(e)))
			return -1

		if not self.model or self.model is None:
			logger.error("Model object is None, loading failed!")
			return -1

		#===========================
		#==   SET LOSS & METRICS
		#===========================
		if self.multilabel:
			self.model.compile(
				optimizer=self.optimizer, 
				loss=self.__multilabel_loss,
				#metrics=['accuracy', f1score_metric, precision_metric, recall_metric], 
				run_eagerly=True
			)
		else:
			self.model.compile(
				optimizer=self.optimizer, 
				loss=self.loss_type, 
				metrics=['accuracy', f1score_metric, precision_metric, recall_metric], 
				run_eagerly=True
			)
		
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



