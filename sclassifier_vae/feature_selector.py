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
from sklearn.feature_selection import RFE, RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline

## GRAPHICS MODULES
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

## PACKAGE MODULES
from .utils import Utils
from .data_loader import DataLoader
from .data_loader import SourceData



##################################
##     FeatSelector CLASS
##################################
class FeatSelector(object):
	""" Feature selector class """
	
	def __init__(self):
		""" Return a FeatSelector object """

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
		self.data_sel= None
		self.data_preclassified_sel= None
		self.source_names= []
		self.source_names_preclassified= []

		# *****************************
		# ** Scoring
		# *****************************
		self.cv_nsplits= 5
		self.cv_nrepeats= 3
		self.cv_seed= 1
		self.scoring= 'f1_weighted'
		#self.scoring= 'accuracy'
		self.ncores= 1

		# *****************************
		# ** Model
		# *****************************
		self.max_depth= None
		self.classifier_inventory= {}
		self.classifier= 'DecisionTreeClassifier'
		self.model= None
		self.models= []
		self.rfe= None
		self.pipeline= None
		self.pipelines= []
		self.cv= None

		# *****************************
		# ** Pre-processing
		# *****************************
		self.normalize= False
		self.norm_min= 0
		self.norm_max= 1

		self.classid_remap= {
			0: -1,
			-1: -1,
			1: 4,
			2: 5,
			3: 0,
			6: 1,
			23: 2,
			24: 3,			
			6000: 6,
		}

		self.classid_label_map= {
			0: "UNKNOWN",
			-1: "MIXED_TYPE",
			1: "STAR",
			2: "GALAXY",
			3: "PN",
			6: "HII",
			23: "PULSAR",
			24: "YSO",			
			6000: "QSO",
		}

		# *****************************
		# ** Output
		# *****************************
		self.outfile= 'featdata_sel.dat'
		self.outfile_scores= 'featscores.png'


	#####################################
	##     BUILD PIPELINE
	#####################################
	def __create_classifier_inventory(self):
		""" Create classifier inventory """

		self.classifier_inventory= {
			"DecisionTreeClassifier": DecisionTreeClassifier(max_depth=self.max_depth),
			"RandomForestClassifier": RandomForestClassifier(max_depth=self.max_depth, n_estimators=10, max_features=1),
			"GradientBoostingClassifier": GradientBoostingClassifier(),
			"MLPClassifier": MLPClassifier(alpha=1, max_iter=1000),
			"SVC": SVC(gamma=2, C=1),
			"QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis()
    }


	def __create_model(self):
		""" Create the model """"
		
		# - Check if model type exists in classifier inventory
		if self.classifier not in self.classifier_inventory:
			return None

		# - Return classifier
		return self.classifier_inventory[self.classifier]
		

	def __create_pipeline(self):
		""" Build the feature selector pipeline """

		# - Create classifier inventory
		logger.info("Creating classifier inventory ...")
		self.__create_classifier_inventory()

		# - Create models
		self.model= self.__create_model()
		if self.model is None:
			logger.error("Created model is None!")
			return -1

		for i in range(1,self.nfeatures):
			m= self.__create_model()
			self.models.append(m)

		# - Define dataset split (unique for all models)
		self.cv= StratifiedKFold(n_splits=self.cv_nsplits, random_state=self.cv_seed)

		# - Create RFE & pipeline
		self.rfe= RFECV(
			estimator=self.model,
			step=1,
			#cv=self.cv, 
			scoring=self.scoring,
			min_features_to_select=1,
			n_jobs=self.ncores
		)
		self.pipeline = Pipeline(
			steps=[('featsel', self.rfe),('model', self.model)]
		)

		for i in range(1,self.nfeatures):
			r= RFE(
				estimator=self.models[i-1],
				#cv=self.cv, 
				scoring=self.scoring,
				n_features_to_select=i,
				n_jobs=self.ncores
			)
			p= Pipeline(steps=[('featsel', r),('model', self.models[i-1])])
			self.pipelines.append(p)
		
		
		return 0
		
	#####################################
	##     EVALUATE MODEL
	#####################################
	def __evaluate_model(self):
		""" Evaluate model """

		# - Create pipeline and models
		logger.info("Creating pipeline and model ...")
		if self.__create_pipeline()<0:
			logger.error("Failed to create pipeline and model!")
			return -1

		# - Evaluate models
		logger.info("Evaluating models as a function of #features ...")
		results, nfeats = list(), list()
		for i in range(1,self.nfeatures):
			p= self.pipelines[i-1]
			scores= cross_val_score(
				p, 
				self.data_preclassified, self.data_preclassified_targets, 
				scoring=self.scoring, 
				cv=self.cv, 
				n_jobs=self.ncores, 
				error_score='raise'
			)
			scores_mean= mean(scores)
			scores_std= std(scores)
			results.append(scores)
			nfeats.append(i)
			logger.info('--> nfeats=%d: score=%.3f (std=%.3f)' % (i, scores_mean, scores_std))
			
		# - Evaluate automatically-selected model
		logger.info("Evaluate model (automated feature selection) ...")
		scores= cross_val_score(
			self.pipeline, 
			self.data_preclassified, self.data_preclassified_targets, 
			scoring=self.scoring, 
			cv=self.cv, 
			n_jobs=self.ncores, 
			error_score='raise'
		)

		best_scores_mean= mean(scores)
		best_scores_std= std(scores)
		logger.info('Best scores: %.3f (std=%.3f)' % (best_scores_mean, best_scores_std))

		# - Fit data and show which features were selected
		logger.info("Fitting RFE model on dataset ...")
		self.rfe.fit(self.data_preclassified, self.data_preclassified_targets)
	
		selfeats= self.rfe.support_
		featranks= self.rfe.ranking_ 
		for i in range(self.data_preclassified.shape[1]):
			logger.info('Feature %d: selected? %d (rank=%.3f)' % (i, selfeats[i], featranks[i]))

		# - Extract selected data columns
		logger.info("Extracting selected data columns from original data ...")
		self.data_sel= self.data[:,selfeats]
		self.data_preclassified_sel= self.data_preclassified[:,selfeats]

		# - Plot results
		logger.info("Plotting and saving feature score results ...")
		plt.boxplot(results, labels=nfeats, showmeans=True)
		#plt.show()
		plt.savefig(self.outfile_scores)

		return 0
	

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
	##     SAVE DATA
	#####################################
	def __save(self):
		""" Save selected data """

		# - Check if selected data is available
		if self.data_sel is None:
			logger.error("Selected data is None!")
			return -1	

		# - Concatenate sel data for saving
		logger.info("Concatenate feature-selected data for saving ...")
		N= self.data_sel.shape[0]
		Nfeat= self.data_sel.shape[1]
		print("Selected data shape=",self.data_sel.shape)
		
		snames= np.array(self.source_names).reshape(N,1)
		objids= np.array(self.data_classids).reshape(N,1)
			
		outdata= np.concatenate(
			(snames, self.data_sel, objids),
			axis=1
		)

		znames_counter= list(range(1,Nfeat+1))
		znames= '{}{}'.format('z',' z'.join(str(item) for item in znames_counter))
		head= '{} {} {}'.format("# sname",znames,"id")

		# - Save feature selected data 
		logger.info("Saving feature-selected data to file %s ..." % (self.outfile))
		Utils.write_ascii(outdata, self.outfile, head)	

		return 0

	#####################################
	##     RUN
	#####################################
	def run(self, datafile):
		""" Run feature selection """
		
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
		#==   EVALUATE MODELS
		#================================
		logger.info("Evaluating models ...")
		if self.__evaluate_model()<0:
			logger.error("Failed to evaluate models!")
			return -1

		#================================
		#==   SAVE
		#================================
		logger.info("Saving results ...")
		if self.__save()<0:
			logger.error("Failed to save results!")
			return -1

		return 0



	def run(self, data, class_ids=[], snames=[]):
		""" Run feature selection using input dataset """

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
		#==   EVALUATE MODELS
		#================================
		logger.info("Evaluating models ...")
		if self.__evaluate_model()<0:
			logger.error("Failed to evaluate models!")
			return -1

		#================================
		#==   SAVE
		#================================
		logger.info("Saving results ...")
		if self.__save()<0:
			logger.error("Failed to save results!")
			return -1

		return 0

