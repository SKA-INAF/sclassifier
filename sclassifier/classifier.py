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
from sclassifier import logger

## SCI MODULES
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity
from scipy.stats import kurtosis, skew
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import Perceptron
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.tree import export_text
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from lightgbm import LGBMClassifier
from lightgbm import early_stopping, log_evaluation, record_evaluation
from lightgbm import plot_tree, plot_importance

import pandas as pd

## OPTUNA
import optuna
from optuna.integration import LightGBMPruningCallback

## GRAPHICS MODULES
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

## PACKAGE MODULES
from .utils import Utils
from .data_loader import DataLoader
from .data_loader import SourceData
from .outlier_finder import OutlierFinder


def save_feature_importance_df(model, feat_names="", outfile="feature_importance.csv"):
	""" Save feature importance sorted by gain """

	# - Get pandas data frame with feature importance sorted by gain
	logger.info("Creating Panda data frame with trained model feature importance, sorted by importance ...")
	
	feature_names= model.feature_name_
	if feat_names!="":
		feature_names= feat_names
	
	importance_df = (
		pd.DataFrame({
			'feature_name': feature_names,
			'importance': model.feature_importances_
			#'importance_gain': model.feature_importance(importance_type='gain'),
			#'importance_split': model.feature_importance(importance_type='split'),
		})
		.sort_values('importance', ascending=False)
		.reset_index(drop=True)
	)
	
	# - Save to file
	logger.info("Saving feature importance to file %s ..." % (outfile))
	importance_df.to_csv(outfile, index=False)


##################################
##     SClassifier CLASS
##################################
class SClassifier(object):
	""" Source classifier class """
	
	def __init__(self, multiclass=True):
		""" Return a SClassifier object """

		# - Input data
		self.nsamples= 0
		self.nfeatures= 0
		self.data= None
		self.data_labels= []
		self.data_classids= []
		self.data_targets= []
		self.data_preclassified= None
		self.data_preclassified_labels= None
		self.data_preclassified_classids= None
		self.data_preclassified_targets= None
		self.data_preclassified_classnames= None
		self.data_preclassified_targetnames= None
		self.source_names= []
		self.source_names_preclassified= []

		self.excluded_objids_train= [-1,0] # Sources with these ids are considered not labelled and therefore excluded from training or metric calculation

		# - Validation data
		self.nsamples_cv= 0
		self.nfeatures_cv= 0
		self.data_cv= None
		self.data_labels_cv= []
		self.data_classids_cv= []
		self.data_targets_cv= []
		self.data_preclassified_cv= None
		self.data_preclassified_labels_cv= None
		self.data_preclassified_classids_cv= None
		self.data_preclassified_targets_cv= None
		self.data_preclassified_classnames_cv= None
		self.data_preclassified_targetnames_cv= None
		self.source_names_cv= []
		self.source_names_preclassified_cv= []
		

		# *****************************
		# ** Model
		# *****************************
		self.data_scaler= None
		self.max_depth= None
		self.min_samples_split= 2
		self.min_samples_leaf= 1
		self.criterion= 'gini'
		self.num_leaves= 31
		self.n_estimators= 100
		self.learning_rate= 0.1
		self.niters= 100
		self.classifier_inventory= {}
		self.classifier= 'DecisionTreeClassifier'
		self.model= None
		self.classids_pred= None
		self.labels_pred= None
		self.targets_pred= None
		self.probs_pred= None
		self.accuracy= None
		self.precision= None
		self.recall= None   
		self.f1score= None
		self.cm= None
		self.cm_norm= None
		self.class_precisions= []
		self.class_recalls= []  
		self.class_f1scores= []
		self.feat_ranks= []
		self.nclasses= 7
		self.multiclass= multiclass
		self.balance_classes= False

		# - LGBM custom options
		self.early_stop_round= 10
		self.metric_lgbm= 'multi_logloss'
		self.lgbm_eval_dict= {}
		self.importance_type= 'split' # 'gain'
		self.scan_test_size= 0.3
		
		# - Linear classifier custom options
		self.tol= None # 1.e-3
		self.verbosity= 1

		# - Set class label names
		self.__set_target_labels(multiclass)

		# *****************************
		# ** Pre-processing
		# *****************************
		self.normalize= False
		self.norm_min= 0
		self.norm_max= 1

		
		# *****************************
		# ** Anomaly detection
		# *****************************
		self.find_outliers= False
		self.outlier_modelfile= ""
		self.outlier_thr= 0.9
		self.outlier_max_samples= "auto"
		self.outlier_max_features= 1
		self.outlier_flags= None
		self.outlier_scores= None
		self.save_outlier= False
		self.outlier_outfile= "outlier_data.dat"
		
		# *****************************
		# ** Output
		# *****************************
		self.save_labels= False
		self.dump_model= True
		self.outfile_scaler = 'datascaler.sav'
		self.outfile_model= "classifier.sav"
		self.outfile_metrics= "metrics.dat"
		self.outfile_cm= "confusion_matrix.dat"
		self.outfile_cm_norm= "confusion_matrix_norm.dat"
		self.outfile= 'classified_data.dat'
		self.outfile_losses= 'losses.dat'
		self.plotfile_decisiontree= 'decision_tree.png'

		# *****************************
		# ** Draw
		# *****************************
		self.feature_names= ''

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
				

		
		self.classid_remap_inv= {v: k for k, v in self.classid_remap.items()}
		self.classid_label_map_inv= {v: k for k, v in self.classid_label_map.items()}

		print("classid_remap")
		print(self.classid_remap)
		print("target_label_map")
		print(self.target_label_map)
		print("classid_label_map")
		print(self.classid_label_map)
		print("classid_remap_inv")
		print(self.classid_remap_inv)
		print("classid_label_map_inv")
		print(self.classid_label_map_inv)

	def set_classid_remap(self, cid_remap):
		""" Set class id remap and update inverted map """
	
		self.classid_remap= cid_remap
		self.classid_remap_inv= {v: k for k, v in self.classid_remap.items()}
		
	def set_classid_label_map(self, cid_label_map):
		""" Set class id label map and update inverted map """
	
		self.classid_label_map= cid_label_map
		self.classid_label_map_inv= {v: k for k, v in self.classid_label_map.items()}
		


	#####################################
	##     CREATE CLASSIFIER
	#####################################
	def __create_classifier_inventory(self):
		""" Create classifier inventory """

		# - Set LGBM classifier
		max_depth_lgbm= self.max_depth
		if max_depth_lgbm is None:
			max_depth_lgbm= -1

		if self.multiclass:
			#logger.info("Setting LGBM classifier for multiclass classification ...")
			objective_lgbm= 'multiclass'
			self.metric_lgbm= 'multi_logloss'
			
			class_weight= None
			if self.balance_classes:
				class_weight= 'balanced'

			lgbm= LGBMClassifier(
				#n_estimators=self.n_estimators, 
				max_depth=max_depth_lgbm, 
				min_data_in_leaf=self.min_samples_leaf, 
				num_leaves=self.num_leaves,
				learning_rate=self.learning_rate,
				num_iterations=self.niters,
				objective=objective_lgbm,
				metric=self.metric_lgbm,
				is_provide_training_metric=True,
				boosting_type='gbdt',
				class_weight=class_weight,
				importance_type=self.importance_type,
				verbose=1
				#num_class=self.nclasses
			)

		else:
			#logger.info("Setting LGBM classifier for binary classification ...")
			objective_lgbm= 'binary'
			self.metric_lgbm= 'binary_logloss'

			is_unbalance= False
			if self.balance_classes:
				is_unbalance= True

			lgbm= LGBMClassifier(
				#n_estimators=self.n_estimators, 
				max_depth=max_depth_lgbm, 
				min_data_in_leaf=self.min_samples_leaf, 
				num_leaves=self.num_leaves,
				learning_rate=self.learning_rate,
				num_iterations=self.niters,
				objective=objective_lgbm,
				metric=self.metric_lgbm,
				is_provide_training_metric=True,
				boosting_type='gbdt',
				is_unbalance=is_unbalance,
				importance_type=self.importance_type,
				verbose=1
				#num_class=self.nclasses
			)

		# - Set DecisionTree classifier
		dt= DecisionTreeClassifier(
			max_depth=self.max_depth, 
			min_samples_split=self.min_samples_split, 
			min_samples_leaf=self.min_samples_leaf
		)

		# - Set RandomForest classifier
		rf= RandomForestClassifier(
			max_depth=self.max_depth, 
			min_samples_split=self.min_samples_split, 
			min_samples_leaf=self.min_samples_leaf, 
			n_estimators=self.n_estimators, 
			max_features=1
		)
		
		# - Set linear classifier
		class_weight= None
		if self.balance_classes:
			class_weight= 'balanced'
				
		lc= Perceptron(
			eta0=self.learning_rate,
			tol=self.tol,
			penalty=None,
			max_iter=self.niters,
			fit_intercept=True,
			class_weight=class_weight,
			early_stopping=False,
			warm_start=False,
			shuffle=True,
			verbose=self.verbosity,
			random_state=0
		)

		# - Set inventory
		self.classifier_inventory= {
			"DecisionTreeClassifier": dt,
			"RandomForestClassifier": rf,
			"GradientBoostingClassifier": GradientBoostingClassifier(),
			"MLPClassifier": MLPClassifier(alpha=1, max_iter=1000),
			#"SVC": SVC(gamma=2, C=1),
			"SVC": SVC(),
			"QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),
			"LinearClassifier": lc,
			"LGBMClassifier": lgbm
    }


	def __create_model(self):
		""" Create the model """
		
		# - Create classifier inventory
		logger.info("Creating classifier inventory ...")
		self.__create_classifier_inventory()		

		# - Check if model type exists in classifier inventory
		if self.classifier not in self.classifier_inventory:
			logger.error("Chosen classifier (%s) is not in the inventory, returning None!" % (self.classifier))
			return None

		# - Return classifier
		return self.classifier_inventory[self.classifier]


	

	#####################################
	##     PRE-PROCESSING
	#####################################
	def __transform_data(self, x, norm_min=0, norm_max=1):
		""" Transform input data here or using a loaded scaler """

		# - Print input data min/max
		x_min= x.min(axis=0)
		x_max= x.max(axis=0)

		print("== INPUT DATA MIN/MAX ==")
		print(x_min)
		print(x_max)

		if self.data_scaler is None:
			# - Define and run scaler
			logger.info("Define and running data scaler ...")
			self.data_scaler= MinMaxScaler(feature_range=(norm_min, norm_max))
			x_transf= self.data_scaler.fit_transform(x)

			print("== TRANSFORM DATA MIN/MAX ==")
			print(self.data_scaler.data_min_)
			print(self.data_scaler.data_max_)

			# - Save scaler to file
			logger.info("Saving data scaler to file %s ..." % (self.outfile_scaler))
			pickle.dump(self.data_scaler, open(self.outfile_scaler, 'wb'))
			
		else:
			# - Transform data
			logger.info("Transforming input data using loaded scaler ...")
			x_transf = self.data_scaler.transform(x)

		# - Print transformed data min/max
		print("== TRANSFORMED DATA MIN/MAX ==")
		x_transf_min= x_transf.min(axis=0)
		x_transf_max= x_transf.max(axis=0)
		print(x_transf_min)
		print(x_transf_max)

		
		return x_transf


	def __normalize_data(self, x, norm_min, norm_max):
		""" Normalize input data to desired range """
		
		x_min= x.min(axis=0)
		x_max= x.max(axis=0)
		print("x_min")
		print(x_min)
		print("x_max")
		print(x_max)
	
		x_norm = norm_min + (x-x_min)/(x_max-x_min) * (norm_max-norm_min)
		return x_norm


	def __set_preclass_data(self):
		""" Set pre-classified data """

		# - Set preclassified data
		row_list= []
		label_list= []
		classid_list= []	
		targetid_list= []

		for i in range(self.nsamples):
			source_name= self.source_names[i]
			obj_id= self.data_classids[i]
			label= self.data_labels[i]
			target_id= self.classid_remap[obj_id] # remap obj id to target class ids
				
			add_to_train_list= True
			for obj_id_excl in self.excluded_objids_train:
				if obj_id==obj_id_excl:
					add_to_train_list=False
					break	
				
			#if obj_id!=0 and obj_id!=-1:
			if add_to_train_list:
				row_list.append(i)
				classid_list.append(obj_id)
				targetid_list.append(target_id)	
				label_list.append(label)
				self.source_names_preclassified.append(source_name)				

		if row_list:	
			self.data_preclassified= self.data[row_list,:]
			self.data_preclassified_labels= np.array(label_list)
			self.data_preclassified_classids= np.array(classid_list)
			self.data_preclassified_targets= np.array(targetid_list)
			self.data_preclassified_classnames= list(set(label_list))
			self.data_preclassified_targetnames= [self.target_label_map[item] for item in set(sorted(targetid_list))]
			
			print("data_preclassified_targetnames")
			print(self.data_preclassified_targetnames)

			print("self.data_preclassified_targets")
			print(self.data_preclassified_targets)

		if self.data_preclassified is not None:
			logger.info("#nsamples_preclass=%d" % (len(self.data_preclassified_labels)))
		else:
			logger.info("No pre-classified objects found in this file ...")

		return 0

	def __set_preclass_val_data(self):
		""" Set pre-classified validation data """

		# - Set preclassified data
		row_list= []
		label_list= []
		classid_list= []	
		targetid_list= []

		for i in range(self.nsamples_cv):
			source_name= self.source_names_cv[i]
			obj_id= self.data_classids_cv[i]
			label= self.data_labels_cv[i]
			target_id= self.classid_remap[obj_id] # remap obj id to target class ids
				
			add_to_train_list= True
			for obj_id_excl in self.excluded_objids_train:
				if obj_id==obj_id_excl:
					add_to_train_list=False
					break		
				
			#if obj_id!=0 and obj_id!=-1:
			if add_to_train_list:
				row_list.append(i)
				classid_list.append(obj_id)
				targetid_list.append(target_id)	
				label_list.append(label)
				self.source_names_preclassified_cv.append(source_name)				

		if row_list:	
			self.data_preclassified_cv= self.data_cv[row_list,:]
			self.data_preclassified_labels_cv= np.array(label_list)
			self.data_preclassified_classids_cv= np.array(classid_list)
			self.data_preclassified_targets_cv= np.array(targetid_list)
			self.data_preclassified_classnames_cv= list(set(label_list))
			self.data_preclassified_targetnames_cv= [self.target_label_map[item] for item in set(sorted(targetid_list))]
			
			print("data_preclassified_targetnames_cv")
			print(self.data_preclassified_targetnames_cv)

			print("self.data_preclassified_targets_cv")
			print(self.data_preclassified_targets_cv)

		if self.data_preclassified_cv is not None:
			logger.info("#nsamples_preclass_cv=%d" % (len(self.data_preclassified_labels_cv)))

		return 0

	#####################################
	##     SET DATA FROM FILE
	#####################################	
	def set_data_from_file(self, filename):
		""" Set data from input file. Expected format: sname, N features, classid """

		# - Read table
		row_start= 0
		try:
			table= ascii.read(filename, data_start=row_start)
		except:
			logger.error("Failed to read feature file %s!" % filename)
			return -1
	
		print(table.colnames)
		print(table)

		ncols= len(table.colnames)
		nfeat= ncols-2

		# - Set data vectors
		rowIndex= 0
		self.data_labels= []
		self.data_classids= []
		self.data_targets= []
		self.source_names= []
		featdata= []

		for data in table:
			sname= data[0]
			obj_id= data[ncols-1]
			label= self.classid_label_map[classid]
			targetid= self.classid_remap[obj_id] # remap obj id in class id

			self.source_names.append(sname)
			self.data_labels.append(label)
			self.data_classids.append(obj_id)
			self.data_targets.append(targetid)
			featdata_curr= []
			for k in range(nfeat):
				featdata_curr.append(data[k+1])
			featdata.append(featdata_curr)

		self.data= np.array(featdata)
		if self.data.size==0:
			logger.error("Empty feature data vector read!")
			return -1

		data_shape= self.data.shape
		self.nsamples= data_shape[0]
		self.nfeatures= data_shape[1]
		logger.info("#nsamples=%d, #nfeatures=%d" % (self.nsamples, self.nfeatures))
		
		# - Normalize feature data?
		if self.normalize:
			logger.info("Normalizing feature data ...")
			#data_norm= self.__normalize_data(self.data, self.norm_min, self.norm_max)
			data_norm= self.__transform_data(self.data, self.norm_min, self.norm_max)
			if data_norm is None:
				logger.error("Data transformation failed!")
				return -1
			self.data= data_norm

		# - Set pre-classified data
		logger.info("Setting pre-classified data (if any) ...")
		self.__set_preclass_data()

		return 0



	def set_val_data_from_file(self, filename):
		""" Set validation data from input file. Expected format: sname, N features, classid """

		# - Read table
		row_start= 0
		try:
			table= ascii.read(filename, data_start=row_start)
		except:
			logger.error("Failed to read feature file %s!" % filename)
			return -1
	
		print(table.colnames)
		print(table)

		ncols= len(table.colnames)
		nfeat= ncols-2

		# - Set data vectors
		rowIndex= 0
		self.data_labels_cv= []
		self.data_classids_cv= []
		self.data_targets_cv= []
		self.source_names_cv= []
		featdata= []

		for data in table:
			sname= data[0]
			obj_id= data[ncols-1]
			label= self.classid_label_map[classid]
			targetid= self.classid_remap[obj_id] # remap obj id in class id

			self.source_names_cv.append(sname)
			self.data_labels_cv.append(label)
			self.data_classids_cv.append(obj_id)
			self.data_targets_cv.append(targetid)
			featdata_curr= []
			for k in range(nfeat):
				featdata_curr.append(data[k+1])
			featdata.append(featdata_curr)

		self.data_cv= np.array(featdata)
		if self.data.size_cv==0:
			logger.error("Empty feature data vector read!")
			return -1

		data_shape= self.data_cv.shape
		self.nsamples_cv= data_shape[0]
		self.nfeatures_cv= data_shape[1]
		logger.info("#nsamples=%d, #nfeatures=%d" % (self.nsamples_cv, self.nfeatures_cv))
		
		# - Normalize feature data?
		if self.normalize:
			logger.info("Normalizing feature data ...")
			#data_norm= self.__normalize_data(self.data, self.norm_min, self.norm_max)
			data_norm= self.__transform_data(self.data_cv, self.norm_min, self.norm_max)
			if data_norm is None:
				logger.error("Data transformation failed!")
				return -1
			self.data_cv= data_norm

		# - Set pre-classified data
		logger.info("Setting pre-classified validation data (if any) ...")
		self.__set_preclass_val_data()

		return 0

	#####################################
	##     SET DATA FROM VECTOR
	#####################################
	def set_data(self, featdata, class_ids=[], snames=[]):
		""" Set data from input array. Optionally give labels & obj names """

		# - Set data vector
		self.data_labels= []
		self.data_classids= []
		self.data_targets= []
		self.source_names= []

		# - Set feature data
		self.data= featdata
		data_shape= self.data.shape

		if self.data.size==0:
			logger.error("Empty feature data vector given!")
			return -1

		self.nsamples= data_shape[0]
		self.nfeatures= data_shape[1]

		# - Set class ids & labels
		if class_ids:
			nids= len(class_ids)
			if nids!=self.nsamples:
				logger.error("Given class ids have size (%d) different than feature data (%d)!" % (nids,self.nsamples))
				return -1
			self.data_classids= class_ids

			for classid in self.data_classids:
				label= self.classid_label_map[classid]
				self.data_labels.append(label)

		else:
			self.data_classids= [0]*self.nsamples # Init to unknown type
			self.data_labels= ["UNKNOWN"]*self.nsamples
		
		# - Set target ids
		for j in range(len(self.data_classids)):
			obj_id= self.data_classids[j]
			targetid= self.classid_remap[obj_id] # remap obj id in class id
			self.data_targets.append(targetid)
		
		# - Set obj names
		if snames:
			n= len(snames)	
			if n!=self.nsamples:
				logger.error("Given source names have size (%d) different than feature data (%d)!" % (n,self.nsamples))
				return -1
			self.source_names= snames
		else:
			self.source_names= ["XXX"]*self.nsamples # Init to unclassified
		
		logger.info("#nsamples=%d, #nfeatures=%d" % (self.nsamples, self.nfeatures))
		
		# - Normalize feature data?
		if self.normalize:
			logger.info("Normalizing feature data ...")
			#data_norm= self.__normalize_data(self.data, self.norm_min, self.norm_max)
			data_norm= self.__transform_data(self.data, self.norm_min, self.norm_max)
			if data_norm is None:
				logger.error("Data transformation failed!")
				return -1
			self.data= data_norm

		# - Set pre-classified data
		logger.info("Setting pre-classified data (if any) ...")
		self.__set_preclass_data()

		return 0


	def set_val_data(self, featdata, class_ids=[], snames=[]):
		""" Set validation data from input array. Optionally give labels & obj names """

		# - Set data vector
		self.data_labels_cv= []
		self.data_classids_cv= []
		self.data_targets_cv= []
		self.source_names_cv= []

		# - Set feature data
		self.data_cv= featdata
		data_shape= self.data_cv.shape

		if self.data_cv.size==0:
			logger.error("Empty feature data vector given!")
			return -1

		self.nsamples_cv= data_shape[0]
		self.nfeatures_cv= data_shape[1]

		# - Set class ids & labels
		if class_ids:
			nids= len(class_ids)
			if nids!=self.nsamples_cv:
				logger.error("Given class ids have size (%d) different than feature data (%d)!" % (nids,self.nsamples_cv))
				return -1
			self.data_classids_cv= class_ids

			for classid in self.data_classids_cv:
				label= self.classid_label_map[classid]
				self.data_labels_cv.append(label)

		else:
			self.data_classids_cv= [0]*self.nsamples_cv # Init to unknown type
			self.data_labels_cv= ["UNKNOWN"]*self.nsamples_cv
		
		# - Set target ids
		for j in range(len(self.data_classids_cv)):
			obj_id= self.data_classids_cv[j]
			targetid= self.classid_remap[obj_id] # remap obj id in class id
			self.data_targets_cv.append(targetid)
		
		# - Set obj names
		if snames:
			n= len(snames)	
			if n!=self.nsamples_cv:
				logger.error("Given source names have size (%d) different than feature data (%d)!" % (n,self.nsamples_cv))
				return -1
			self.source_names_cv= snames
		else:
			self.source_names_cv= ["XXX"]*self.nsamples_cv # Init to unclassified
		
		logger.info("#nsamples=%d, #nfeatures=%d" % (self.nsamples_cv, self.nfeatures_cv))
		
		# - Normalize feature data?
		if self.normalize:
			logger.info("Normalizing feature data ...")
			data_norm= self.__transform_data(self.data_cv, self.norm_min, self.norm_max)
			if data_norm is None:
				logger.error("Data transformation failed!")
				return -1
			self.data_cv= data_norm

		# - Set pre-classified data
		logger.info("Setting pre-classified validation data (if any) ...")
		self.__set_preclass_val_data()

		return 0

	#####################################
	##     RUN TRAIN
	#####################################
	def run_train(self, datafile, modelfile='', scalerfile='', datafile_cv=''):
		""" Run train using input dataset """

		#================================
		#==   LOAD DATA SCALER
		#================================
		# - Load scaler from file?
		if scalerfile!="":
			logger.info("Loading data scaler from file %s ..." % (scalerfile))
			try:
				self.data_scaler= pickle.load(open(scalerfile, 'rb'))
			except Exception as e:
				logger.error("Failed to load data scaler from file %s!" % (scalerfile))
				return -1

		#================================
		#==   LOAD DATA
		#================================
		# - Check inputs
		if datafile=="":
			logger.error("Empty data file specified!")
			return -1

		if self.set_data_from_file(datafile)<0:
			logger.error("Failed to read datafile %s!" % datafile)
			return -1

		#================================
		#==   LOAD CROSS-VALIDATION DATA
		#================================
		if datafile_cv!="":
			logger.info("Reading validation data from file %s ..." % datafile_cv)
			
			if self.set_val_data_from_file(datafile_cv)<0:
				logger.error("Failed to read validation datafile %s!" % datafile_cv)
				return -1
		
		#================================
		#==   LOAD MODEL
		#================================
		if modelfile and modelfile is not None:
			logger.info("Loading the model from file %s ..." % modelfile)
			try:
				#self.model, self.prediction_extra_data = pickle.load((open(modelfile, 'rb')))
				self.model = pickle.load((open(modelfile, 'rb')))
			except Exception as e:
				logger.error("Failed to load model from file %s!" % (modelfile))
				return -1

			# - Retrieve classifier name from loaded object
			self.classifier= self.model.__class__.__name__
			logger.info("Loaded model classifier is: %s" % (self.classifier))

		else:
			logger.info("Creating the clustering model ...")
			self.model= self.__create_model()
			#self.predition_extra_data= None

		#================================
		#==   TRAIN
		#================================
		logger.info("Training model ...")
		if self.__train()<0:
			logger.error("Failed to train model!")
			return -1

		#================================
		#==   SAVE
		#================================
		logger.info("Saving results ...")
		if self.__save_train()<0:
			logger.error("Failed to save results!")
			return -1

		return 0



	def run_train(self, data, class_ids=[], snames=[], modelfile='', scalerfile='', data_cv=None, class_ids_cv=[], snames_cv=[]):
		""" Run train using input dataset """

		#================================
		#==   LOAD DATA SCALER
		#================================
		# - Load scaler from file?
		if scalerfile!="":
			logger.info("Loading data scaler from file %s ..." % (scalerfile))
			try:
				self.data_scaler= pickle.load(open(scalerfile, 'rb'))
			except Exception as e:
				logger.error("Failed to load data scaler from file %s!" % (scalerfile))
				return -1

		#================================
		#==   LOAD DATA
		#================================
		# - Check inputs
		if data is None:
			logger.error("None input data specified!")
			return -1

		if self.set_data(data, class_ids, snames)<0:
			logger.error("Failed to set data!")
			return -1

		#================================
		#==   LOAD CROSS-VALIDATION DATA
		#================================
		if data_cv is not None:
			logger.info("Loading validation data ...")
			
			if self.set_val_data(data_cv, class_ids_cv, snames_cv)<0:
				logger.error("Failed to set validation data!")
				return -1

		#================================
		#==   LOAD MODEL
		#================================
		if modelfile and modelfile is not None:
			logger.info("Loading the model from file %s ..." % modelfile)
			try:
				#self.model, self.prediction_extra_data = pickle.load((open(modelfile, 'rb')))
				self.model= pickle.load((open(modelfile, 'rb')))
			except Exception as e:
				logger.error("Failed to load model from file %s!" % (modelfile))
				return -1

			# - Retrieve classifier name from loaded object
			self.classifier= self.model.__class__.__name__
			logger.info("Loaded model classifier is: %s" % (self.classifier))

		else:
			logger.info("Creating the model ...")
			self.model= self.__create_model()
			#self.predition_extra_data= None

		#================================
		#==   TRAIN
		#================================
		logger.info("Training model ...")
		if self.__train()<0:
			logger.error("Failed to train model!")
			return -1

		#================================
		#==   SAVE
		#================================
		logger.info("Saving results ...")
		if self.__save_train()<0:
			logger.error("Failed to save results!")
			return -1

		return 0




	def __lgbm_scan_objective(self, trial, X, y):
		""" Define optuna objective function for multiclass/binary classification scan """
    
		# - Define parameters to be optimized
		if self.multiclass:
			objective_lgbm= 'multiclass'
			metric_lgbm= 'multi_logloss'
			
			class_weight= None
			is_unbalance= False
			if self.balance_classes:
				class_weight= 'balanced'

		else:
			objective_lgbm= 'binary'
			metric_lgbm= 'binary_logloss'

			class_weight= None
			is_unbalance= False
			if self.balance_classes:
				is_unbalance= True

	
		param_grid = {
			"num_iterations": self.niters,
			"objective": objective_lgbm,
			"metric": metric_lgbm,
			"verbosity": 1,
			"boosting_type": "gbdt",
			"is_provide_training_metric": True,
			###"learning_rate": self.learning_rate,
			#"learning_rate": trial.suggest_categorical("learning_rate", [0.1,0.2,0.5,0.01,0.05,0.001]),
			"learning_rate": trial.suggest_categorical("learning_rate", [0.1]),
			###"min_data_in_leaf": self.min_samples_leaf,
			#"min_data_in_leaf": trial.suggest_categorical("min_data_in_leaf", [1,2,5,10,20,50,100]),
			"min_data_in_leaf": trial.suggest_categorical("min_data_in_leaf", [5]),
			"class_weight": class_weight,
			"is_unbalance": is_unbalance,
			### "device_type": trial.suggest_categorical("device_type", ['gpu']),
			#"n_estimators": trial.suggest_categorical("n_estimators", [1,2,5,10,100,200,1000]),
			"n_estimators": trial.suggest_categorical("n_estimators", [100]),
			##"num_leaves": trial.suggest_categorical("num_leaves", [2,5,10,20,30,40,50,100]),
			"num_leaves": trial.suggest_categorical("num_leaves", [5]),
			#"max_depth": trial.suggest_int("max_depth", 2, 10, step=1),
			"max_depth": trial.suggest_categorical("max_depth", [-1]),
			###"lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
			###"lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
			###"min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
			###"bagging_fraction": trial.suggest_float(
			###	"bagging_fraction", 0.2, 0.95, step=0.1
			###),
 			###"bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
			###"feature_fraction": trial.suggest_float(
			###	"feature_fraction", 0.2, 0.95, step=0.1
			###),
		}

		# - Define data split
		nsplits= 5
		#cv = StratifiedKFold(n_splits=nsplits, shuffle=True, random_state=1121218)
		cv = StratifiedShuffleSplit(n_splits=nsplits, test_size=self.scan_test_size, random_state=1121218)
		cv_scores = np.empty(nsplits)
		cv_losses = np.empty(nsplits)
		
		# - Define callbacks
		earlystop_cb= early_stopping(
			stopping_rounds=self.early_stop_round, 
			first_metric_only=True, verbose=True
		)
			
		logeval_cb= log_evaluation(period=1, show_stdv=True)

		# - Scan over parameters
		for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
			print(f"Fold {idx}:")
			print(f"  Train: {len(train_idx)}, index={train_idx}")
			print(f"  Test:  {len(test_idx)}, index={test_idx}")

			X_train, X_test = X[train_idx], X[test_idx]
			y_train, y_test = y[train_idx], y[test_idx]
			
			# - Create model
			model= LGBMClassifier(**param_grid)

			# - Fit model	
			model.fit(
				X_train, y_train,
				eval_set=[(X_test, y_test), (X_train, y_train)],
				eval_names=["test", "train"],
				eval_metric=metric_lgbm,
				#early_stopping_rounds=100,
				callbacks=[	
					earlystop_cb, 
					logeval_cb, 
					#receval_cb
				]
			)
			
			# - Retrieve val loss
			print("model.best_score")
			print(model._best_score)
			loss_val= model._best_score['test'][metric_lgbm]
			cv_losses[idx]= loss_val
			print("loss_val")
			print(loss_val)
		
			# - Predict model on pre-classified data
			y_pred= model.predict(X_test)
		
			# - Retrieve metrics
			logger.info("Computing classification metrics on train data ...")
			report= classification_report(y_test, y_pred, target_names=self.target_names, output_dict=True)
			print("report")
			print(report)
			f1score= report['weighted avg']['f1-score']
			cv_scores[idx]= f1score
			
		# - Compute mean scores/losses
		mean_cv_loss_val= np.mean(cv_losses)
		mean_cv_score= np.mean(cv_scores)
		print("cv_scores")
		print(cv_scores)
		print("mean_cv_score")
		print(mean_cv_score)
		print("cv_losses")
		print(cv_losses)
		print("mean_cv_loss_val")
		print(mean_cv_loss_val)
		
		#return mean_cv_score
		return mean_cv_loss_val



	def run_lgbm_scan(self, datafile, n_trials=1):
		""" Run LGBM par scan """

		#================================
		#==   LOAD DATA SCALER
		#================================
		# - Load scaler from file?
		if scalerfile!="":
			logger.info("Loading data scaler from file %s ..." % (scalerfile))
			try:
				self.data_scaler= pickle.load(open(scalerfile, 'rb'))
			except Exception as e:
				logger.error("Failed to load data scaler from file %s!" % (scalerfile))
				return -1

		#================================
		#==   LOAD DATA
		#================================
		# - Check inputs
		if datafile=="":
			logger.error("Empty data file specified!")
			return -1

		if self.set_data_from_file(datafile)<0:
			logger.error("Failed to read datafile %s!" % datafile)
			return -1

		#================================
		#==   CREATE SCAN MODEL
		#================================
		# - Set scan data
		X= self.data_preclassified
		y= self.data_preclassified_targets

		# - Define optuna study	
		logger.info("Define optuna study ...")
		#study = optuna.create_study(direction="maximize", study_name="LGBM Classifier")
		study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")
		func= lambda trial: self.__lgbm_scan_objective(trial, X, y)
		
		#================================
		#==   RUN SCAN
		#================================
		# - Run study
		logger.info("Run optuna study ...")
		study.optimize(func, n_trials=n_trials)

		print(f"\tBest value (F1-score): {study.best_value:.5f}")
		print(f"\tBest params:")

		for key, value in study.best_params.items():
			print(f"\t\t{key}: {value}")

		return 0


	def run_lgbm_scan(self, data, class_ids=[], snames=[], scalerfile='', n_trials=1):
		""" Run train using input dataset """

		#================================
		#==   LOAD DATA SCALER
		#================================
		# - Load scaler from file?
		if scalerfile!="":
			logger.info("Loading data scaler from file %s ..." % (scalerfile))
			try:
				self.data_scaler= pickle.load(open(scalerfile, 'rb'))
			except Exception as e:
				logger.error("Failed to load data scaler from file %s!" % (scalerfile))
				return -1

		#================================
		#==   LOAD DATA
		#================================
		# - Check inputs
		if data is None:
			logger.error("None input data specified!")
			return -1

		if self.set_data(data, class_ids, snames)<0:
			logger.error("Failed to set data!")
			return -1

		#================================
		#==   CREATE SCAN MODEL
		#================================
		# - Set scan data
		X= self.data_preclassified
		y= self.data_preclassified_targets

		# - Define optuna study	
		logger.info("Define optuna study ...")
		#study = optuna.create_study(direction="maximize", study_name="LGBM Classifier")
		study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")
		func= lambda trial: self.__lgbm_scan_objective(trial, X, y)

		#================================
		#==   RUN SCAN
		#================================
		# - Run study
		logger.info("Run optuna study ...")
		study.optimize(func, n_trials=n_trials)

		print(f"\tBest value (F1-score): {study.best_value:.5f}")
		print(f"\tBest params:")

		for key, value in study.best_params.items():
			print(f"\t\t{key}: {value}")

		return 0

	def __fit_model(self):
		""" Fit model """

		has_cv_data= (self.data_preclassified_cv is not None) and (self.data_preclassified_targets_cv is not None)

		if self.classifier=='LGBMClassifier':
			
			featname_opt= 'auto'
			if self.feature_names!="":
				featname_opt= self.feature_names
		
			if has_cv_data:
				# - Custom LGBM scikit fit method with early stopping
				earlystop_cb= early_stopping(
					stopping_rounds=self.early_stop_round, 
					first_metric_only=True, verbose=True
				)
			
				logeval_cb= log_evaluation(period=1, show_stdv=True)	
				receval_cb= record_evaluation(self.lgbm_eval_dict)

				try:
					self.model.fit(
						self.data_preclassified, self.data_preclassified_targets,
						eval_set=[(self.data_preclassified_cv, self.data_preclassified_targets_cv), (self.data_preclassified, self.data_preclassified_targets)],
						eval_names=["cv", "train"],
						eval_metric=self.metric_lgbm,
						callbacks=[earlystop_cb, logeval_cb, receval_cb],
						feature_name= featname_opt
					)
		
					print("--> lgbm eval dict")
					print(self.lgbm_eval_dict)

				except Exception as e:
					logger.error("Failed to fit model on data (err=%s)!" % (str(e)))
					return -1
			else:
				# - Custom LGBM scikit fit method
				try:
					self.model.fit(
						self.data_preclassified, self.data_preclassified_targets,
						feature_name= featname_opt
					)
				except Exception as e:
					logger.error("Failed to fit model on data (err=%s)!" % (str(e)))
					return -1
		else:
			# - Standard scikit learn fit method
			try:
				self.model.fit(self.data_preclassified, self.data_preclassified_targets)
			except Exception as e:
				logger.error("Failed to fit model on data (err=%s)!" % (str(e)))
				return -1
		
		return 0


	def __train(self):
		""" Train model """
		
		# - Check if data are set
		if self.data_preclassified is None:
			logger.error("Input pre-classified data is None, check if provided data have labels!")
			return -1
		if self.data is None:
			logger.error("Input data is None!")
			return -1

		# - Check if model is set
		if self.model is None:
			logger.error("Model is not set!")
			return -1

		# - Fit model on pre-classified data
		logger.info("Fit model on train data ...")
		if self.__fit_model()<0:
			logger.error("Failed to fit model on data!")
			return -1
		#try:
		#	self.model.fit(self.data_preclassified, self.data_preclassified_targets)
		#except Exception as e:
		#	logger.error("Failed to fit model on data (err=%s)!" % (str(e)))
		#	return -1

		# - Predict model on pre-classified data
		logger.info("Predicting class and probabilities on train data ...")
		try:
			self.targets_pred= self.model.predict(self.data_preclassified)
			#class_probs_pred= self.model.predict_proba(self.data_preclassified)
			#print("== class_probs_pred ==")
			#print(class_probs_pred.shape)
			#self.probs_pred= np.max(class_probs_pred, axis=1)

		except Exception as e:
			logger.error("Failed to predict model on data (err=%s)!" % (str(e)))
			return -1
			
		try:
			class_probs_pred= self.model.predict_proba(self.data_preclassified)
			print("== class_probs_pred ==")
			print(class_probs_pred.shape)
			self.probs_pred= np.max(class_probs_pred, axis=1)

		except Exception as e:
			logger.warn("Failed to get model prob on data (err=%s), setting them to -1..." % (str(e)))
			N= self.data_preclassified.shape[0]
			self.probs_pred= [-1]*N

		if self.multiclass:
			# - Convert targets to obj ids
			logger.info("Converting predicted targets to class ids ...")
			self.classids_pred= [self.classid_remap_inv[item] for item in self.targets_pred]

			# - Convert obj ids to labels
			self.labels_pred= [self.classid_label_map[item] for item in self.classids_pred]
		else:
			# - Set class ids to targets for binary class
			logger.info("Setting predicted class ids to target ids for binary class ...")
			self.classids_pred=	self.targets_pred	

			# - Convert obj ids to labels
			self.labels_pred= [self.target_label_map[item] for item in self.classids_pred]


		nclasses= len(self.target_names)
		labels= list(range(0,nclasses))
		print("target_names")
		print(self.target_names)
		print("labels")
		print(labels)

		# - Retrieve metrics
		logger.info("Computing classification metrics on train data ...")
		#report= classification_report(self.data_preclassified_targets, self.targets_pred, target_names=self.target_names, output_dict=True)
		report= classification_report(self.data_preclassified_targets, self.targets_pred, target_names=self.target_names, labels=labels, output_dict=True)	
		self.accuracy= 0
		if 'accuracy' in report:
			self.accuracy= report['accuracy']
		self.precision= report['weighted avg']['precision']
		self.recall= report['weighted avg']['recall']    
		self.f1score= report['weighted avg']['f1-score']

		self.class_precisions= []
		self.class_recalls= []  
		self.class_f1scores= []
		for class_name in self.data_preclassified_targetnames:
			class_precision= report[class_name]['precision']
			class_recall= report[class_name]['recall']    
			class_f1score= report[class_name]['f1-score']
			self.class_precisions.append(class_precision)
			self.class_recalls.append(class_recall)
			self.class_f1scores.append(class_f1score)
			
		print("--> Classification report")
		print(report)

		logger.info("accuracy=%f" % (self.accuracy))
		logger.info("precision=%f" % (self.precision))
		logger.info("recall=%f" % (self.recall))
		logger.info("f1score=%f" % (self.f1score))
				
		logger.info("--> Metrics per class")
		print("classnames")
		print(self.data_preclassified_targetnames)
		print("class precisions")
		print(self.class_precisions)
		print("class recall")
		print(self.class_recalls)
		print("class f1score")
		print(self.class_f1scores)

		# - Retrieving confusion matrix
		logger.info("Retrieving confusion matrix on train data ...")
		self.cm= confusion_matrix(self.data_preclassified_targets, self.targets_pred)

		print("confusion matrix")
		print(self.cm)

		self.cm_norm= confusion_matrix(self.data_preclassified_targets, self.targets_pred, normalize="true")

		print("confusion matrix norm")
		print(self.cm_norm)

		# - Retrieving the feature importances
		try:
			self.feat_ranks= self.model.feature_importances_
			print("feat ranks")
			print(self.feat_ranks)
			
		except Exception as e:
			logger.warn("Failed to get feature ranks from model (err=%s)..." % (str(e)))
	
	
		return 0


	#####################################
	##     RUN PREDICT
	#####################################
	def run_predict(self, datafile, modelfile='', scalerfile=''):
		""" Run model prediction using input dataset """

		#================================
		#==   LOAD DATA SCALER
		#================================
		# - Load scaler from file?
		if scalerfile!="":
			logger.info("Loading data scaler from file %s ..." % (scalerfile))
			try:
				self.data_scaler= pickle.load(open(scalerfile, 'rb'))
			except Exception as e:
				logger.error("Failed to load data scaler from file %s!" % (scalerfile))
				return -1

		#================================
		#==   LOAD DATA
		#================================
		# - Check inputs
		if datafile=="":
			logger.error("Empty data file specified!")
			return -1

		if self.set_data_from_file(datafile)<0:
			logger.error("Failed to read datafile %s!" % datafile)
			return -1

		#================================
		#==   LOAD MODEL
		#================================
		if modelfile is not None:
			logger.info("Loading the model from file %s ..." % modelfile)
			try:
				#self.model, self.prediction_extra_data = pickle.load((open(modelfile, 'rb')))
				self.model = pickle.load((open(modelfile, 'rb')))
			except Exception as e:
				logger.error("Failed to load model from file %s!" % (modelfile))
				return -1

			# - Retrieve classifier name from loaded object
			self.classifier= self.model.__class__.__name__
			logger.info("Loaded model classifier is: %s" % (self.classifier))

		else:
			logger.info("Creating the clustering model ...")
			self.model= self.__create_model()
			#self.predition_extra_data= None

		#================================
		#==   RUN PREDICT
		#================================
		logger.info("Run model predict ...")
		if self.__predict()<0:
			logger.warn("Failed to run model predict on input data!")
			return -1

		#================================
		#==   RUN OUTLIER FINDER
		#================================
		if self.find_outliers:
			logger.info("Run outlier finder ...")
			if self.__find_outliers(datafile, self.outlier_modelfile, scalerfile)<0:
				logger.warn("Failed to run outlier finder on input data!")
				return -1

		#================================
		#==   SAVE
		#================================
		logger.info("Saving results ...")
		if self.__save_predict()<0:
			logger.error("Failed to save results!")
			return -1

		return 0

	def run_predict(self, data, class_ids=[], snames=[], modelfile='', scalerfile=''):
		""" Run model prediction using input dataset """

		#================================
		#==   LOAD DATA SCALER
		#================================
		# - Load scaler from file?
		if scalerfile!="":
			logger.info("Loading data scaler from file %s ..." % (scalerfile))
			try:
				self.data_scaler= pickle.load(open(scalerfile, 'rb'))
			except Exception as e:
				logger.error("Failed to load data scaler from file %s!" % (scalerfile))
				return -1

		#================================
		#==   LOAD DATA
		#================================
		# - Check inputs
		if data is None:
			logger.error("None input data specified!")
			return -1

		if self.set_data(data, class_ids, snames)<0:
			logger.error("Failed to read datafile %s!" % datafile)
			return -1

		#================================
		#==   LOAD MODEL
		#================================
		if modelfile is not None:
			logger.info("Loading the model from file %s ..." % modelfile)
			try:
				#self.model, self.prediction_extra_data = pickle.load((open(modelfile, 'rb')))
				self.model= pickle.load((open(modelfile, 'rb')))
			except Exception as e:
				logger.error("Failed to load model from file %s!" % (modelfile))
				return -1

			# - Retrieve classifier name from loaded object
			self.classifier= self.model.__class__.__name__
			logger.info("Loaded model classifier is: %s" % (self.classifier))

		else:
			logger.info("Creating the model ...")
			self.model= self.__create_model()
			#self.predition_extra_data= None

		#================================
		#==   RUN PREDICT
		#================================
		logger.info("Run model predict ...")
		if self.__predict()<0:
			logger.warn("Failed to run model predict on input data!")
			return -1
	
		#================================
		#==   RUN OUTLIER FINDER
		#================================
		if self.find_outliers:
			logger.info("Run outlier finder ...")
			if self.__find_outliers(data, class_ids, snames, self.outlier_modelfile, scalerfile)<0:
				logger.warn("Failed to run outlier finder on input data!")
				return -1

		#================================
		#==   SAVE
		#================================
		logger.info("Saving results ...")
		if self.__save_predict()<0:
			logger.error("Failed to save results!")
			return -1

		return 0


	def __predict(self):
		""" Predict model """
		
		# - Check if data are set
		if self.data is None:
			logger.error("Input data is None!")
			return -1

		# - Check if model is set
		if self.model is None:
			logger.error("Model is not set!")
			return -1

		# - Predict model on data
		logger.info("Predicting class and probabilities on input data ...")
		try:
			self.targets_pred= self.model.predict(self.data)
			#class_probs_pred= self.model.predict_proba(self.data)
			#print("== class_probs_pred ==")
			#print(class_probs_pred.shape)
			#self.probs_pred= np.max(class_probs_pred, axis=1)

		except Exception as e:
			logger.error("Failed to predict model on data (err=%s)!" % (str(e)))
			return -1
			
		try:
			class_probs_pred= self.model.predict_proba(self.data)
			print("== class_probs_pred ==")
			print(class_probs_pred.shape)
			self.probs_pred= np.max(class_probs_pred, axis=1)

		except Exception as e:
			logger.warn("Failed to get model prob on data (err=%s), setting them to -1 ..." % (str(e)))
			N= self.data.shape[0]
			self.probs_pred= [-1]*N

		
		if self.multiclass:
			# - Convert targets to obj ids
			logger.info("Converting predicted targets to class ids ...")
			self.classids_pred= [self.classid_remap_inv[item] for item in self.targets_pred]

			# - Compute pred labels
			self.labels_pred= [self.classid_label_map[item] for item in self.classids_pred]

		else:
			# - Set class ids to targets for binary class
			logger.info("Setting predicted class ids to target ids for binary class ...")
			self.classids_pred=	self.targets_pred	

			# - Convert obj ids to labels
			self.labels_pred= [self.target_label_map[item] for item in self.classids_pred]


		# - Predict model on pre-classified data (if any)
		if self.data_preclassified is not None:
			logger.info("Predicting class and probabilities on input pre-classified data ...")
			try:
				targets_pred_preclass= self.model.predict(self.data_preclassified)
				#class_probs_pred_preclass= self.model.predict_proba(self.data_preclassified)
				#print("== class_probs_pred (preclass data) ==")
				#print(class_probs_pred_preclass.shape)
				#probs_pred_preclass= np.max(class_probs_pred_preclass, axis=1)

			except Exception as e:
				logger.error("Failed to predict model on pre-classified data (err=%s)!" % (str(e)))
				return -1
				
			try:
				class_probs_pred_preclass= self.model.predict_proba(self.data_preclassified)
				print("== class_probs_pred (preclass data) ==")
				print(class_probs_pred_preclass.shape)
				probs_pred_preclass= np.max(class_probs_pred_preclass, axis=1)

			except Exception as e:
				logger.warn("Failed to obtain model probability on pre-classified data (err=%s) ..." % (str(e)))

			print("target_names")
			print(self.target_names)
			print("targets_pred_preclass.shape")
			print(targets_pred_preclass.shape)
			print(targets_pred_preclass)
			print("data_preclassified_targets.shape")
			print(self.data_preclassified_targets.shape)
			print(self.data_preclassified_targets)
			nclasses= len(self.target_names)
			labels= list(range(0,nclasses))
			print("labels")
			print(labels)

			# - Retrieve metrics
			logger.info("Computing classification metrics on pre-classified data ...")
			report= classification_report(self.data_preclassified_targets, targets_pred_preclass, target_names=self.target_names, labels=labels, output_dict=True)
			#report= classification_report(self.data_preclassified_targets, targets_pred_preclass, target_names=self.target_names, output_dict=True)
			print(report)
			self.accuracy= 0
			if 'accuracy' in report:
				self.accuracy= report['accuracy']
			self.precision= report['weighted avg']['precision']
			self.recall= report['weighted avg']['recall']    
			self.f1score= report['weighted avg']['f1-score']

			self.class_precisions= []
			self.class_recalls= []  
			self.class_f1scores= []
			for class_name in self.data_preclassified_targetnames:
				class_precision= report[class_name]['precision']
				class_recall= report[class_name]['recall']    
				class_f1score= report[class_name]['f1-score']
				self.class_precisions.append(class_precision)
				self.class_recalls.append(class_recall)
				self.class_f1scores.append(class_f1score)
			
			print("--> Classification report")
			print(report)

			logger.info("accuracy=%f" % (self.accuracy))
			logger.info("precision=%f" % (self.precision))
			logger.info("recall=%f" % (self.recall))
			logger.info("f1score=%f" % (self.f1score))			
			logger.info("--> Metrics per class")
			print("classnames")
			print(self.data_preclassified_targetnames)
			print("class precisions")
			print(self.class_precisions)
			print("class recall")
			print(self.class_recalls)
			print("class f1score")
			print(self.class_f1scores)

			# - Retrieving confusion matrix
			logger.info("Retrieving confusion matrix on pre-classified data ...")
			self.cm= confusion_matrix(self.data_preclassified_targets, targets_pred_preclass)

			print("confusion matrix")
			print(self.cm)

			self.cm_norm= confusion_matrix(self.data_preclassified_targets, targets_pred_preclass, normalize="true")

			print("confusion matrix norm")
			print(self.cm_norm)


		return 0

	#####################################
	##     FIND OUTLIERS
	#####################################
	def __find_outliers(self, datafile, modelfile="", scalerfile=""):
		""" Find outliers """
		
		# - Check if outlier model file has been given
		if modelfile=="":
			logger.error("No trained outlier model file given!")
			return -1
	
		# - Run outlier prediction on input data
		ofinder= OutlierFinder()
		ofinder.normalize= self.normalize
		ofinder.anomaly_thr= self.outlier_thr
		ofinder.max_samples= self.outlier_max_samples
		ofinder.max_features= self.outlier_max_features
		ofinder.save_to_file= self.save_outlier
		ofinder.outfile= self.outlier_outfile

		status= ofinder.run(
			datafile, 
			modelfile=modelfile,
			scalerfile=scalerfile
		)

		if status<0:
			logger.error("Failed to run outlier finder!")
			return -1

		# - Retrieve results
		self.outlier_flags= ofinder.data_pred
		self.outlier_scores= ofinder.anomaly_scores_orig

		return 0


	def __find_outliers(self, data, class_ids, snames, modelfile="", scalerfile=""):
		""" Find outliers """
		
		# - Check if outlier model file has been given
		if modelfile=="":
			logger.error("No trained outlier model file given!")
			return -1
	
		# - Run outlier prediction on input data
		ofinder= OutlierFinder()
		ofinder.normalize= self.normalize
		ofinder.anomaly_thr= self.outlier_thr
		ofinder.max_samples= self.outlier_max_samples
		ofinder.max_features= self.outlier_max_features
		ofinder.save_to_file= self.save_outlier
		ofinder.outfile= self.outlier_outfile

		status= ofinder.run(
			data, class_ids, snames, 
			modelfile=modelfile,
			scalerfile=scalerfile
		)

		if status<0:
			logger.error("Failed to run outlier finder!")
			return -1

		# - Retrieve results
		self.outlier_flags= ofinder.data_pred
		self.outlier_scores= ofinder.anomaly_scores_orig

		return 0

	#####################################
	##     SAVE
	#####################################
	def __save_predict(self):
		""" Save prediction results """

		#================================
		#==   SAVE METRICS
		#================================
		if self.data_preclassified is not None:
			logger.info("Saving metrics on pre-classified data ...")
			metrics= [self.accuracy, self.precision, self.recall, self.f1score]
			metric_names= ["accuracy","precision","recall","f1score"]
		
			for i in range(len(self.data_preclassified_targetnames)):
				classname= self.data_preclassified_targetnames[i]
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

			# - Save confusion matrix
			logger.info("Saving confusion matrix to file %s ..." % (self.outfile_cm))
			#with open(self.outfile_cm, 'w') as f:
			#	f.write(np.array2string(self.cm, separator=', '))

			np.savetxt(self.outfile_cm, self.cm, delimiter=',')

			#with open(self.outfile_cm_norm, 'w') as f:
			#	f.write(np.array2string(self.cm_norm, separator=', '))
		
			np.savetxt(self.outfile_cm_norm, self.cm_norm, delimiter=',')

		#================================
		#==   SAVE PREDICTION DATA
		#================================
		logger.info("Saving prediction data to file %s ..." % (self.outfile))
		N= self.data.shape[0]
		snames= np.array(self.source_names).reshape(N,1)
		objids= np.array(self.data_classids).reshape(N,1)
		objids_pred= np.array(self.classids_pred).reshape(N,1)
		probs_pred= np.array(self.probs_pred).reshape(N,1)
		objlabels_pred= np.array(self.labels_pred).reshape(N,1)
		objlabels= np.array(self.data_labels).reshape(N,1)

		if self.find_outliers:
			N_outliers= self.outlier_flags.shape[0]
			if N!=N_outliers:
				logger.error("Number of entries in outlier data out (%d) is different from classifier data out (%d)!" % (N_outliers, N))
				return -1
			outlier_flags= np.array(self.outlier_flags).reshape(N_outliers,1)
			outlier_scores= np.array(self.outlier_scores).reshape(N_outliers,1)

		if self.save_labels:
			if self.find_outliers:
				outdata= np.concatenate(
					(snames, self.data, outlier_flags, outlier_scores, objlabels, objlabels_pred, probs_pred),
					axis=1
				)
			else:
				outdata= np.concatenate(
					(snames, self.data, objlabels, objlabels_pred, probs_pred),
					axis=1
				)
		else:
			if self.find_outliers:
				outdata= np.concatenate(
					(snames, self.data, outlier_flags, outlier_scores, objids, objids_pred, probs_pred),
					axis=1
				)
			else:
				outdata= np.concatenate(
					(snames, self.data, objids, objids_pred, probs_pred),
					axis=1
				)	

		znames_counter= list(range(1,self.nfeatures+1))
		znames= '{}{}'.format('z',' z'.join(str(item) for item in znames_counter))
		
		if self.feature_names:
			feat_names= self.feature_names
		else:
			feat_names= ['z'+str(item) for item in znames_counter]
		
		if self.save_labels:
			if self.find_outliers:
				head= '{} {} {}'.format("# sname",znames,"is_outlier outlier_score label label_pred prob")
			else:
				head= '{} {} {}'.format("# sname",znames,"label label_pred prob")
		else:
			if self.find_outliers:
				head= '{} {} {}'.format("# sname",znames,"is_outlier outlier_score id id_pred prob")
			else:
				head= '{} {} {}'.format("# sname",znames,"id id_pred prob")
			

		Utils.write_ascii(outdata, self.outfile, head)	

		#================================
		#==   SAVE FEATURE IMPORTANCE
		#================================
		if self.classifier=='LGBMClassifier':
			logger.info("Saving LGBM feature importance ...")
			save_feature_importance_df(self.model, feat_names)	
			ax= plot_importance(self.model, importance_type="gain", figsize=(15,15), title="LightGBM Feature Importance (Gain)")
			plt.savefig("lgbm_feature_importance.png")	

		return 0


	def __save_train(self):
		""" Save train results """

		#================================
		#==   SAVE DECISION TREE PLOT
		#================================
		# - If classifier is decision tree, save it
		if self.classifier=='DecisionTreeClassifier':
			logger.info("Saving decision tree plots ...")
			self.__save_decisiontree()

		# - If classifier is LGBM plot tree
		if self.classifier=='LGBMClassifier':
			logger.info("Saving LGBM tree plot ...")
			#lightgbm.plot_tree(self.model, ax=None, tree_index=0, figsize=None, dpi=None, show_info=None, precision=3, orientation='horizontal', **kwargs)
			#lightgbm.plot_tree(self.model)
			ax= plot_tree(self.model, tree_index=0, figsize=(15, 15), show_info=['split_gain'])
			plt.savefig("lgbm_tree.png")	
			
		#================================
		#==   SAVE FEATURE IMPORTANCE
		#================================
		znames_counter= list(range(1,self.nfeatures+1))
		if self.feature_names:
			feat_names= self.feature_names
		else:
			feat_names= ['z'+str(item) for item in znames_counter]
			
		if self.classifier=='LGBMClassifier':
			logger.info("Saving LGBM feature importance ...")
			save_feature_importance_df(self.model, feat_names)
			ax= plot_importance(self.model, importance_type="gain", figsize=(15,15), title="LightGBM Feature Importance (Gain)")
			plt.savefig("lgbm_feature_importance.png")	
				

		#================================
		#==   SAVE MODEL
		#================================
		# - Save model to file
		if self.dump_model:
			logger.info("Dumping model to file %s ..." % self.outfile_model)
			pickle.dump(self.model, open(self.outfile_model, 'wb'))
			
		#================================
		#==   SAVE LOSSES
		#================================
		if self.classifier=='LGBMClassifier' and self.lgbm_eval_dict:
			logger.info("Saving LGBM loss metrics ...")
			losses_train= self.lgbm_eval_dict['train'][self.metric_lgbm]
			losses_cv= self.lgbm_eval_dict['cv'][self.metric_lgbm]
			n_iters= len(losses_train)
			losses_train_arr= np.array(losses_train).reshape(n_iters,1)			
			losses_cv_arr= np.array(losses_cv).reshape(n_iters,1)			
			iters= np.array(list(range(1, n_iters+1))).reshape(n_iters, 1)
			head= "# iter loss_train loss_cv"

			outdata_losses= np.concatenate(
				(iters, losses_train_arr, losses_cv_arr),
				axis=1
			)
			Utils.write_ascii(outdata_losses, self.outfile_losses, head)	


		#================================
		#==   SAVE METRICS
		#================================
		logger.info("Saving train metrics ...")
		metrics= [self.accuracy, self.precision, self.recall, self.f1score]
		metric_names= ["accuracy","precision","recall","f1score"]
		
		for i in range(len(self.data_preclassified_targetnames)):
			classname= self.data_preclassified_targetnames[i]
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

		# - Save confusion matrix
		logger.info("Saving confusion matrix to file ...")
		#with open(self.outfile_cm, 'w') as f:
		#	f.write(np.array2string(self.cm, separator=', '))

		np.savetxt(self.outfile_cm, self.cm, delimiter=',')

		#with open(self.outfile_cm_norm, 'w') as f:
		#	f.write(np.array2string(self.cm_norm, separator=', '))
		
		np.savetxt(self.outfile_cm_norm, self.cm_norm, delimiter=',')

		#===================================
		#==   SAVE TRAIN PREDICTION DATA
		#===================================
		logger.info("Saving train prediction data to file %s ..." % (self.outfile))
		N= self.data_preclassified.shape[0]
		snames= np.array(self.source_names_preclassified).reshape(N,1)
		objids= np.array(self.data_preclassified_classids).reshape(N,1)
		objids_pred= np.array(self.classids_pred).reshape(N,1)
		probs_pred= np.array(self.probs_pred).reshape(N,1)
		objlabels_pred= np.array(self.labels_pred).reshape(N,1)
		objlabels= np.array(self.data_preclassified_labels).reshape(N,1)

		if self.save_labels:
			outdata= np.concatenate(
				(snames, self.data_preclassified, objlabels, objlabels_pred, probs_pred),
				axis=1
			)
		else:
			outdata= np.concatenate(
				(snames, self.data_preclassified, objids, objids_pred, probs_pred),
				axis=1
			)

		znames_counter= list(range(1,self.nfeatures+1))
		znames= '{}{}'.format('z',' z'.join(str(item) for item in znames_counter))
		head= '{} {} {}'.format("# sname",znames,"id id_pred prob")

		Utils.write_ascii(outdata, self.outfile, head)	

		return 0



	def __save_decisiontree(self, class_names=[]):
		""" Print and save decision tree """

		# - Set class names
		if not class_names:
			class_names= self.data_preclassified_targetnames

		# - Set feature names
		feat_counter= list(range(1,self.nfeatures+1))
		feat_names= ['z'+str(item) for item in feat_counter]

		# - Print decision rules
		logger.info("Printing decision tree rules ...")
		tree_rules= export_text(self.model, feature_names=feat_names)
		print(tree_rules)

		# - Save figure with decision tree	
		logger.info("Saving decision tree plot ...")
		fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
		tree.plot_tree(self.model,
               feature_names = feat_names, 
               class_names=class_names,
               filled = True,
							 proportion= True,
               label='all',
               rounded=True,
							 impurity= False,
							 precision=2)
		fig.savefig(self.plotfile_decisiontree)


