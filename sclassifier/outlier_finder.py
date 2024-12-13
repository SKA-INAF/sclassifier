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
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from sklearn.metrics import make_scorer, f1_score

## GRAPHICS MODULES
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

## PACKAGE MODULES
from .utils import Utils
from .utils import NoIndent, MyEncoder

##################################
##     OutlierFinder CLASS
##################################
class OutlierFinder(object):
	""" Outlier finder class """
	
	def __init__(self):
		""" Return a OutlierFinder object """

		# *****************************
		# ** Input data
		# *****************************
		self.nsamples= 0
		self.nfeatures= 0
		self.data= None
		self.data_preclassified= None
		self.data_preclassified_labels= None
		self.data_preclassified_classids= None
		self.datalist_json= None
		self.data_labels= []
		self.data_classids= []
		self.source_names= []
		self.source_names_preclassified= []
		
		#self.excluded_objids_train= [-1,0] # Sources with these ids are considered not labelled and therefore excluded from training or metric calculation
		self.excluded_objids_train= [] # Sources with these ids are considered not labelled and therefore excluded from training or metric calculation
		
		self.classid_label_map= {
			1: "INLIER",
			-1: "OUTLIER"
		}
		
		# *****************************
		# ** Pre-processing
		# *****************************
		self.normalize= False
		self.norm_min= 0
		self.norm_max= 1
		self.data_scaler= None

		# *****************************
		# ** Isolation Forest pars
		# *****************************
		self.model= None
		self.n_estimators= 100
		self.contamination= 'auto'	
		self.max_samples= 'auto'
		self.bootstrap= False
		self.max_features= 1
		self.verbose= 0
		self.ncores= 1
		self.random_state= None
		self.run_scan= False
		self.scan_nestimators= False
		self.scan_maxfeatures= False
		self.scan_maxsamples= False
		self.scan_contamination= False
		
		self.data_pred= None
		self.anomaly_scores= None
		self.anomaly_scores_df= None
		self.anomaly_scores_orig= None
		self.anomaly_thr= 0.9
		
		self.predict= False

		# *****************************
		# ** Output data
		# *****************************
		self.save_ascii= True
		self.save_json= True
		self.save_features= True
		self.save_model= True
		self.outfile= "outlier_data.dat"
		self.outfile_model= "outlier_model.sav"
		self.outfile_scaler = 'datascaler.sav'
		self.outfile_json= "outlier_data.json"
		
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
		x_norm = norm_min + (x-x_min)/(x_max-x_min) * (norm_max-norm_min)
		return x_norm


	def __set_preclass_data(self):
		""" Set pre-classified data """

		# - Set preclassified data
		row_list= []
		label_list= []
		classid_list= []

		for i in range(self.nsamples):
			source_name= self.source_names[i]
			obj_id= self.data_classids[i]
			label= self.data_labels[i]
				
			add_to_train_list= True
			for obj_id_excl in self.excluded_objids_train:
				if isinstance(obj_id, list):
					for item in obj_id:
						if item==obj_id_excl:
							add_to_train_list= False
							break
					if not add_to_train_list:
						break
				else:
					if obj_id==obj_id_excl:
						add_to_train_list= False
						break

			if add_to_train_list:
			#if obj_id!=0 and obj_id!=-1:
				row_list.append(i)
				classid_list.append(obj_id)	
				label_list.append(label)
				self.source_names_preclassified.append(source_name)
			else:
				logger.debug("Exclude source with id=%d from list (excluded_ids=%s) ..." % (obj_id, str(self.excluded_objids_train)))
				

		if row_list:	
			self.data_preclassified= self.data[row_list,:]
			#self.data_preclassified_labels= np.array(label_list)
			#self.data_preclassified_classids= np.array(classid_list)
			self.data_preclassified_labels= label_list
			self.data_preclassified_classids= classid_list
		
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
		self.source_names= []
		featdata= []

		for data in table:
			sname= data[0]
			classid= data[ncols-1]
			if self.classid_label_map:
				label= self.classid_label_map[classid]
			else:
				label= str(classid)

			self.source_names.append(sname)
			self.data_labels.append(label)
			self.data_classids.append(classid)
			featdata_curr= []
			for k in range(nfeat):
				featdata_curr.append(data[k+1])
			featdata.append(featdata_curr)

		self.data= np.array(featdata)
		if self.data.size==0:
			logger.error("Empty feature data vector read!")
			return -1

		self.nsamples= data_shape[0]
		self.nfeatures= data_shape[1]
		logger.info("#nsamples=%d" % (self.nsamples))
		
		# - Normalize feature data?
		if self.normalize:
			logger.info("Normalizing feature data ...")
			data_norm= self.__transform_data(self.data, self.norm_min, self.norm_max)
			if data_norm is None:
				logger.error("Data transformation failed!")
				return -1
			self.data= data_norm

		# - Set pre-classified data
		logger.info("Setting pre-classified data (if any) ...")
		self.__set_preclass_data()

		return 0

	
	#####################################
	##     SET DATA FROM ASCII FILE
	#####################################
	def set_data_from_ascii_file(self, filename):
		""" Set data from input ascii file. Expected format: sname, N features, classid """

		# - Read ascii file
		ret= Utils.read_feature_data(filename)
		if not ret:
			logger.error("Failed to read data from file %s!" % (filename))
			return -1

		data= ret[0]
		snames= ret[1]
		classids= ret[2]

		return self.set_data(data, snames, class_ids)
	
	#####################################
	##     SET DATA FROM JSON FILE
	#####################################
	def set_data_from_json_file(self, filename, datakey="data"):
		""" Set data from input json file. Expected format: dict with featdata key """

		# - Read json file
		try:
			f= open(filename, "r")
			self.datalist_json= json.load(f)[datakey]
		except Exception as e:
			logger.error("Failed to read feature file %s (err=%s)!" % (filename, str(e)))
			return -1
		
		# - Set data vectors
		self.data_labels= []
		self.data_classids= []
		self.source_names= []
		featdata= []
		
		for item in self.datalist_json:
			sname= item['sname']
			classid= item['id']
			label= item['label']
			feats= item['feats']
			nfeat= len(feats)
			
			# - Remap labels?
			if self.classid_label_map:
				if isinstance(classid, list):
					label= [self.classid_label_map[i] for i in classid]
				else:
					label= self.classid_label_map[classid]
			
			# - Add entries to list
			self.source_names.append(sname)
			self.data_labels.append(label)
			self.data_classids.append(classid)
			featdata.append(feats)

		self.data= np.array(featdata)
		if self.data.size==0:
			logger.error("Empty feature data vector read!")
			return -1

		data_shape= self.data.shape
		self.nsamples= data_shape[0]
		self.nfeatures= data_shape[1]
		logger.info("#nsamples=%d" % (self.nsamples))
		
		# - Normalize feature data?
		if self.normalize:
			logger.info("Normalizing feature data ...")
			data_norm= self.__transform_data(self.data, self.norm_min, self.norm_max)
			if data_norm is None:
				logger.error("Data transformation failed!")
				return -1
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
				if self.classid_label_map:
					label= self.classid_label_map[classid]
				else:
					label= str(classid)
				self.data_labels.append(label)

		else:
			self.data_classids= [0]*self.nsamples # Init to unknown type
			self.data_labels= ["UNKNOWN"]**self.nsamples
		
		
		# - Set obj names
		if snames:
			n= len(snames)	
			if n!=self.nsamples:
				logger.error("Given source names have size (%d) different than feature data (%d)!" % (n,self.nsamples))
				return -1
			self.source_names= snames
		else:
			self.source_names= ["XXX"]*self.nsamples # Init to unclassified
		
		logger.info("#nsamples=%d" % (self.nsamples))
		
		# - Normalize feature data?
		if self.normalize:
			logger.info("Normalizing feature data ...")
			data_norm= self.__transform_data(self.data, self.norm_min, self.norm_max)
			if data_norm is None:
				logger.error("Data transformation failed!")
				return -1
			self.data= data_norm

		# - Set pre-classified data
		logger.info("Setting pre-classified data (if any) ...")
		self.__set_preclass_data()

		return 0


	#####################################
	##     CREATE/LOAD MODEL
	#####################################
	def __create_model(self):
		""" Create model """

		# - Create random seed
		if self.random_state is None:
			self.random_state= np.random.RandomState(42)

		# - Init isolation forest
		model= IsolationForest(
			n_estimators=self.n_estimators,
			max_samples=self.max_samples,
			contamination=self.contamination,
			bootstrap=self.bootstrap,
			max_features=self.max_features,
			n_jobs=self.ncores,
			random_state=self.random_state,
			verbose=self.verbose
		)
		
		print("== MODEL INIT PARAMETERS ==")
		print("n_estimators: ", self.n_estimators)
		print("max_samples: ", self.max_samples)
		print("contamination: ", self.contamination)
		print("max_features: ", self.max_features)
		print("bootstrap: ", self.bootstrap)
		print("random_state: ", self.random_state)
		print("======================")
			
		return model

	#####################################
	##     SCAN PARAMETERS
	#####################################
	def __scan_parameters(self):
		""" Scan IF hyperparameters """

		# - Check if we have preclassified data
		if self.data_preclassified is None:
			logger.warn("No preclassfied data available, no scan performed!")
			return -1

		# - Define parameter grid to be searched
		nestimators_scan= [self.n_estimators]
		if self.scan_nestimators:
			nestimators_scan= [100,200,500,1000]
			
		maxfeatures_scan= [self.nfeatures]
		if self.scan_maxfeatures:
			maxfeatures_scan= [1,2,3,5,self.nfeatures]
			
		maxsamples_scan= [self.max_samples]
		if self.scan_maxsamples:
			maxsamples_scan= ['auto',0.001,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
			###maxsamples_scan= ['auto',0.2]
			
		contamination_scan= [self.contamination]
		if self.scan_contamination:
			contamination_scan= ['auto',0.001,0.01,0.1]
					
		param_grid = {
			'n_estimators': nestimators_scan, 
			'max_samples': maxsamples_scan, 
			'contamination': contamination_scan,
			'max_features': maxfeatures_scan,
		}
		
		logger.info("Scan grid parameters: %s" % (str(param_grid)))
		
		#f1sc= make_scorer(f1_score(average='micro'))

		# - Run grid search
		logger.info("Running parameter grid scan ...")
		grid_search = model_selection.GridSearchCV(
			self.model,
			param_grid,
			#scoring=f1sc, 
			scoring='f1_micro',
			refit=True,
			cv=10, 
			return_train_score=True
		)
		
		res= grid_search.fit(self.data_preclassified, self.data_preclassified_classids)
		#best_pars= grid_search.best_params_
		#best_model= grid_search.best_estimator_
		best_pars= res.best_params_
		best_model= res.best_estimator_

		print("== BEST SCAN MODEL PARAMETERS ==")
		print('Optimum parameters', best_pars)
		print("type(best_model)", type(best_model))
		print("type(res)", type(res))
		print("n_estimators: ", len(best_model.estimators_))
		print("max_samples: ", best_model.max_samples_)
		print("contamination: ", best_model.contamination)
		#print("max_features: ", len(best_model.estimators_features_))
		print("max_features: ", best_model.n_features_in_)
		print("offset: ", best_model.offset_)
		print("======================")
		
		# - Setting model parameters to best model
		logger.info("Setting model to best model found in scan ...")
		self.model= best_model
		##self.model= res
		
		return 0

	#####################################
	##     DETECT OUTLIERS
	#####################################
	def __find_outliers(self, fitdata):
		""" Find outliers """

		# - Fit data?
		if fitdata: 
			logger.info("Fitting input data ...")
			self.model.fit(self.data)
			
			print("== FIT MODEL PARAMETERS ==")
			print("n_estimators: ", len(self.model.estimators_))
			print("max_samples: ", self.model.max_samples_)
			print("contamination: ", self.model.contamination)
			#print("max_features: ", len(self.model.estimators_features_))
			print("max_features: ", self.model.n_features_in_)
			print("offset: ", self.model.offset_)
			print("======================")

		# - Predict outliers (-1=outlier, 1=inlier)
		logger.info("Predicting outliers ...")
		self.data_pred= self.model.predict(self.data)
		
		# - Retrieve the anomaly scores
		#   NB: The lower, the more abnormal. Negative scores represent outliers, positive scores represent inliers
		logger.info("Retrieving the anomaly score (-1 or negative values means outliers) ...")
		self.anomaly_scores_df= self.model.decision_function(self.data)
		self.anomaly_scores= self.model.score_samples(self.data)
		self.anomaly_scores_orig= -self.anomaly_scores

		# - Apply user threshold
		N= self.data_pred.shape[0]
		for i in range(N):
			score= self.anomaly_scores_orig[i]
			if score>self.anomaly_thr:
				#self.data_pred[i]= -1
				self.data_pred[i]= 1
			else:
				#self.data_pred[i]= 1
				self.data_pred[i]= 0

		return 0		


	def run(self, datafile, modelfile='', scalerfile='', datalist_key='data'):
		""" Find outliers in input data """
		
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
			
		# - Check input file extension
		file_ext= os.path.splitext(datafile)[1]
		
		if file_ext=='.json':
			if self.set_data_from_json_file(datafile, datalist_key)<0:
				logger.error("Failed to read datafile %s!" % datafile)
				return -1
		else:
			#if self.set_data_from_file(datafile)<0:
			if self.set_data_from_ascii_file(datafile)<0:
				logger.error("Failed to read datafile %s!" % datafile)
				return -1
	
		#================================
		#==   LOAD MODEL
		#================================
		if modelfile and modelfile is not None:
			logger.info("Loading the model from file %s ..." % modelfile)
			try:
				self.model = pickle.load((open(modelfile, 'rb')))
			except Exception as e:
				logger.error("Failed to load model from file %s!" % (modelfile))
				return -1

		else:
			logger.info("Creating the model ...")
			self.model= self.__create_model()
			
		#================================
		#==   RUN SCAN FIRST?
		#================================	
		if self.run_scan:
			if self.__scan_parameters()<0:
				logger.warn("No parameter optimization scan performed, will run with user-supplied parameters ...")
				
		#================================
		#==   FIND OUTLIERS
		#================================
		# - Do not fit data if predict or if a fit scan was already run before	
		fitdata= True
		#if self.predict or self.run_scan:
		if self.predict:
			fitdata= False
			
		logger.info("Searching for outliers ...")
		if self.__find_outliers(fitdata)<0:
			logger.error("Failed to search outliers!")
			return -1
		
		#================================
		#==   SAVE
		#================================
		if self.save_to_file:
			logger.info("Saving results ...")
			if self.__save()<0:
				logger.error("Failed to save outlier search results!")
				return -1

		return 0


	def run(self, data, class_ids=[], snames=[], modelfile='', scalerfile=''):
		""" Find outliers in input data """

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
		#==   LOAD MODEL
		#================================
		if modelfile and modelfile is not None:
			logger.info("Loading the model from file %s ..." % modelfile)
			try:
				self.model = pickle.load((open(modelfile, 'rb')))
			except Exception as e:
				logger.error("Failed to load model from file %s!" % (modelfile))
				return -1

		else:
			logger.info("Creating the model ...")
			self.model= self.__create_model()
			
		#================================
		#==   RUN SCAN FIRST?
		#================================	
		if self.run_scan:
			if self.__scan_parameters()<0:
				logger.warn("No parameter optimization scan performed, will run with user-supplied parameters ...")

		#================================
		#==   FIND OUTLIERS
		#================================	
		# - Do not fit data if predict or if a fit scan was already run before	
		fitdata= True
		#if self.predict or self.run_scan:
		if self.predict:
			fitdata= False
			
		logger.info("Searching for outliers ...")
		if self.__find_outliers(fitdata)<0:
			logger.error("Failed to search outliers!")
			return -1
		
		#================================
		#==   SAVE
		#================================
		logger.info("Saving results ...")
		if self.__save()<0:
			logger.error("Failed to save outlier search results!")
			return -1

		return 0

	#####################################
	##     SAVE DATA
	#####################################
	def __save(self):
		""" Save data  """
		
		###################################
		##     SAVE ASCII
		###################################
		if self.save_ascii and self.__save_to_ascii()<0:
			logger.warn("Failed to save outputs to ascii format...")
		
		###################################
		##     SAVE JSON
		###################################
		if self.save_json and self.__save_to_json()<0:
			logger.warn("Failed to save outputs to json format...")
			
		###################################
		##     SAVE MODEL
		###################################	
		if self.model and self.save_model:
			logger.info("Saving model to file %s ..." % (self.outfile_model))
			pickle.dump(self.model, open(self.outfile_model, 'wb'))
	
		return 0
		
	def __save_to_json(self):
		""" Save data to ascii """
		
		# - Save unsupervised encoded data
		if self.datalist_json is not None:
			N= self.data.shape[0]
		
			# - Loop over data and replace original feats with UMAP feats
			outdata= {"data": []}
			
			for i in range(N):
				feats= list(self.data[i])
				feats= [float(item) for item in feats]
				is_outlier= self.data_pred[i]
				outlier_score= self.anomaly_scores_orig[i]
				
				d= self.datalist_json[i]
				if self.save_features:
					d['feats']= NoIndent(feats_umap)
				else:
					del d['feats']
				d['is_outlier']= is_outlier
				d['outlier_score']= outlier_score
				outdata["data"].append(d)

			# - Save to json 
			logger.info("Saving unsupervised encoded data to json file %s ..." % (self.outfile_encoded_data_unsupervised_json))	
			with open(self.outfile_encoded_data_unsupervised_json, 'w') as fp:
				json.dump(outdata, fp, cls=MyEncoder, indent=2)
		
		return 0
	
	def __save_to_ascii(self):
		""" Save data to ascii """

		# - Check if selected data is available
		if self.data is None:
			logger.error("Input data is None!")
			return -1

		if self.anomaly_scores is None or self.data_pred is None:
			logger.error("Predicted outlier data are None!")
			return -1

		# - Concatenate data for saving
		logger.info("Concatenate feature-selected data for saving ...")
		N= self.data.shape[0]
		Nfeat= self.data.shape[1]
		snames= np.array(self.source_names).reshape(N,1)
		
		if isinstance(self.data_classids[0], list): # concatenate ids/labels comma separated
			objids_concat= [','.join(map(str,item)) for item in self.data_classids]
			objlabels_concat= [','.join(map(str,item)) for item in self.data_labels]
			objids= np.array(objids_concat).reshape(N,1)
			objlabels= np.array(objlabels_concat).reshape(N,1)
		else:
			objids= np.array(self.data_classids).reshape(N,1)
			objlabels= np.array(self.data_labels).reshape(N,1)
		
		outlier_outputs= np.array(self.data_pred).reshape(N,1)
		#outlier_outputs[outlier_outputs==1]= 0   # set non-outliers to 0 
		#outlier_outputs[outlier_outputs==-1]= 1  # set outliers to 1
		outlier_scores= np.array(self.anomaly_scores).reshape(N,1)
		outlier_scores_df= np.array(self.anomaly_scores_df).reshape(N,1)
		outlier_score_orig= np.array(self.anomaly_scores_orig).reshape(N,1)

		if self.save_features:
			znames_counter= list(range(1, Nfeat+1))
			znames= '{}{}'.format('z',' z'.join(str(item) for item in znames_counter))
			
			if self.save_labels_in_ascii:
				outdata= np.concatenate(
					(snames, self.data, objlabels, outlier_outputs, outlier_score_orig),
					axis=1
				)
				head= '{} {} {}'.format("# sname",znames," label is_outlier outlier_score")
			else:
				outdata= np.concatenate(
					(snames, self.data, objids, outlier_outputs, outlier_score_orig),
					axis=1
				)
				head= '{} {} {}'.format("# sname",znames," id is_outlier outlier_score")
		else:
			if self.save_labels_in_ascii:
				outdata= np.concatenate(
					(snames, objlabels, outlier_outputs, outlier_score_orig),
					axis=1
				)
				head= "# sname label is_outlier outlier_score"
			else:
				outdata= np.concatenate(
					(snames, objids, outlier_outputs, outlier_score_orig),
					axis=1
				)
				head= "# sname id is_outlier outlier_score"
		
		# - Save outlier data 
		logger.info("Saving outlier output data to file %s ..." % (self.outfile))
		Utils.write_ascii(outdata, self.outfile, head)

		return 0

