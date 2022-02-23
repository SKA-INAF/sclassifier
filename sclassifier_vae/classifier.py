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

from lightgbm import LGBMClassifier

## GRAPHICS MODULES
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

## PACKAGE MODULES
from .utils import Utils
from .data_loader import DataLoader
from .data_loader import SourceData



##################################
##     SClassifier CLASS
##################################
class SClassifier(object):
	""" Source classifier class """
	
	def __init__(self):
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

		# *****************************
		# ** Model
		# *****************************
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
		self.targets_pred= None
		self.probs_pred= None
		self.accuracy= None
		self.precision= None
		self.recall= None   
		self.f1score= None
		self.class_precisions= []
		self.class_recalls= []  
		self.class_f1scores= []
		
		# *****************************
		# ** Pre-processing
		# *****************************
		self.normalize= False
		self.norm_min= 0
		self.norm_max= 1

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

		self.classid_remap_inv= {v: k for k, v in self.classid_remap.items()}
		self.classid_label_map_inv= {v: k for k, v in self.classid_label_map.items()}

		#print("")

		# *****************************
		# ** Output
		# *****************************
		self.dump_model= True
		self.outfile_model= "classifier.sav"
		self.outfile_metrics= "metrics.dat"
		self.outfile= 'classified_data.dat'
		self.plotfile_decisiontree= 'decision_tree.png'


	#####################################
	##     CREATE CLASSIFIER
	#####################################
	def __create_classifier_inventory(self):
		""" Create classifier inventory """

		# - Set LGBM classifier
		max_depth_lgbm= self.max_depth
		if max_depth_lgbm is None:
			max_depth_lgbm= -1

		lgbm= LGBMClassifier(
			n_estimators=self.n_estimators, 
			max_depth=max_depth_lgbm, 
			min_data_in_leaf=self.min_samples_leaf, 
			num_leaves=self.num_leaves,
			learning_rate=self.learning_rate,
			num_iterations=self.niters,
			objective='multiclass',
			boosting_type='gbdt'
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

		# - Set inventory
		self.classifier_inventory= {
			"DecisionTreeClassifier": dt,
			"RandomForestClassifier": rf,
			"GradientBoostingClassifier": GradientBoostingClassifier(),
			"MLPClassifier": MLPClassifier(alpha=1, max_iter=1000),
			#"SVC": SVC(gamma=2, C=1),
			"SVC": SVC(),
			"QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),
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
	def __normalize_data(self, x, norm_min, norm_max):
		""" Normalize input data to desired range """
		
		x_min= x.min(axis=0)
		x_max= x.max(axis=0)
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
				
			if obj_id!=0 and obj_id!=-1:
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

		if self.data_preclassified is not None:
			logger.info("#nsamples_preclass=%d" % (len(self.data_preclassified_labels)))

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
			data_norm= self.__normalize_data(self.data, self.norm_min, self.norm_max)
			self.data= data_norm

		# - Set pre-classified data
		logger.info("Setting pre-classified data (if any) ...")
		self.__set_preclass_data()

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
			data_norm= self.__normalize_data(self.data, self.norm_min, self.norm_max)
			self.data= data_norm

		# - Set pre-classified data
		logger.info("Setting pre-classified data (if any) ...")
		self.__set_preclass_data()

		return 0


	#####################################
	##     RUN TRAIN
	#####################################
	def run_train(self, datafile, modelfile=''):
		""" Run train using input dataset """

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
		if modelfile and modelfile is not None:
			logger.info("Loading the model from file %s ..." % modelfile)
			try:
				#self.model, self.prediction_extra_data = pickle.load((open(modelfile, 'rb')))
				self.model = pickle.load((open(modelfile, 'rb')))
			except Exception as e:
				logger.error("Failed to load model from file %s!" % (modelfile))
				return -1

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


	def run_train(self, data, class_ids=[], snames=[], modelfile=''):
		""" Run train using input dataset """

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
		if modelfile and modelfile is not None:
			logger.info("Loading the model from file %s ..." % modelfile)
			try:
				#self.model, self.prediction_extra_data = pickle.load((open(modelfile, 'rb')))
				self.model= pickle.load((open(modelfile, 'rb')))
			except Exception as e:
				logger.error("Failed to load model from file %s!" % (modelfile))
				return -1

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
		try:
			self.model.fit(self.data_preclassified, self.data_preclassified_targets)
		except Exception as e:
			logger.error("Failed to fit model on data (err=%s)!" % (str(e)))
			return -1

		# - Predict model on pre-classified data
		logger.info("Predicting class and probabilities on train data ...")
		try:
			self.targets_pred= self.model.predict(self.data_preclassified)
			class_probs_pred= self.model.predict_proba(self.data_preclassified)
			print("== class_probs_pred ==")
			print(class_probs_pred.shape)
			self.probs_pred= np.max(class_probs_pred, axis=1)

		except Exception as e:
			logger.error("Failed to predict model on data (err=%s)!" % (str(e)))
			return -1

		# - Convert targets to obj ids
		logger.info("Converting predicted targets to class ids ...")
		self.classids_pred= [self.classid_remap_inv[item] for item in self.targets_pred]
		

		# - Retrieve metrics
		logger.info("Computing classification metrics on train data ...")
		report= classification_report(self.data_preclassified_targets, self.targets_pred, target_names=self.data_preclassified_targetnames, output_dict=True)
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
			
		logger.info("accuracy=%f" % (self.accuracy))
		logger.info("precision=%f" % (self.precision))
		logger.info("recall=%f" % (self.recall))
		logger.info("f1score=%f" % (self.f1score))
		logger.info("--> Metrics per class")
		print("classnames")
		print(self.data_preclassified_targetnames)
		print("precisions")
		print(self.class_precisions)
		print("recall")
		print(self.class_recalls)
		print("f1score")
		print(self.class_f1scores)

		return 0


	#####################################
	##     RUN PREDICT
	#####################################
	def run_predict(self, datafile, modelfile=''):
		""" Run model prediction using input dataset """

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
		#==   SAVE
		#================================
		logger.info("Saving results ...")
		if self.__save_train()<0:
			logger.error("Failed to save results!")
			return -1

		return 0

	def run_predict(self, data, class_ids=[], snames=[], modelfile=''):
		""" Run model prediction using input dataset """

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
		#==   SAVE
		#================================
		logger.info("Saving results ...")
		if self.__save_train()<0:
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
			class_probs_pred= self.model.predict_proba(self.data)
			print("== class_probs_pred ==")
			print(class_probs_pred.shape)
			self.probs_pred= np.max(class_probs_pred, axis=1)

		except Exception as e:
			logger.error("Failed to predict model on data (err=%s)!" % (str(e)))
			return -1

		# - Convert targets to obj ids
		logger.info("Converting predicted targets to class ids ...")
		self.classids_pred= [self.classid_remap_inv[item] for item in self.targets_pred]


		# - Predict model on pre-classified data (if any)
		if self.data_preclassified is not None:
			logger.info("Predicting class and probabilities on input pre-classified data ...")
			try:
				targets_pred_preclass= self.model.predict(self.data_preclassified)
				class_probs_pred_preclass= self.model.predict_proba(self.data_preclassified)
				print("== class_probs_pred (preclass data) ==")
				print(class_probs_pred_preclass.shape)
				probs_pred_preclass= np.max(class_probs_pred_preclass, axis=1)

			except Exception as e:
				logger.error("Failed to predict model on pre-classified data (err=%s)!" % (str(e)))
				return -1


			# - Retrieve metrics
			logger.info("Computing classification metrics on pre-classified data ...")
			report= classification_report(self.data_preclassified_targets, targets_pred_preclass, target_names=self.data_preclassified_targetnames, output_dict=True)
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
			
			logger.info("accuracy=%f" % (self.accuracy))
			logger.info("precision=%f" % (self.precision))
			logger.info("recall=%f" % (self.recall))
			logger.info("f1score=%f" % (self.f1score))
			logger.info("--> Metrics per class")
			print("classnames")
			print(self.data_preclassified_targetnames)
			print("precisions")
			print(self.class_precisions)
			print("recall")
			print(self.class_recalls)
			print("f1score")
			print(self.class_f1scores)

		return 0

	#####################################
	##     SAVE
	#####################################
	def __save_train(self):
		""" Save train results """

		#================================
		#==   SAVE DECISION TREE PLOT
		#================================
		# - If classifier is decision tree, save it
		if self.classifier=='DecisionTreeClassifier':
			logger.info("Saving decision tree plots ...")
			self.__save_decisiontree()

		#================================
		#==   SAVE MODEL
		#================================
		# - Save model to file
		if self.dump_model:
			logger.info("Dumping model to file %s ..." % self.outfile_model)
			pickle.dump(self.model, open(self.outfile_model, 'wb'))
			
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
		
		#================================
		#==   SAVE TRAIN PREDICTION DATA
		#================================
		logger.info("Saving train prediction data to file %s ..." % (self.outfile))
		N= self.data_preclassified.shape[0]
		snames= np.array(self.source_names_preclassified).reshape(N,1)
		objids= np.array(self.data_preclassified_classids).reshape(N,1)
		objids_pred= np.array(self.classids_pred).reshape(N,1)
		probs_pred= np.array(self.probs_pred).reshape(N,1)

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


