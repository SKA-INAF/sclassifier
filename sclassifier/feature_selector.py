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
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.feature_selection import RFE, RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from lightgbm import LGBMClassifier
from lightgbm import early_stopping, log_evaluation, record_evaluation
from lightgbm import plot_tree

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
	
	def __init__(self, multiclass=True):
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
		#self.cv_nrepeats= 3
		self.cv_seed= 1
		self.scoring= 'f1_weighted'
		#self.scoring= 'accuracy'
		self.ncores= 1

		# *****************************
		# ** Model
		# *****************************
		self.nfeat_min= 1
		self.nfeat_max= -1 # -1 means =nfeat
		self.nfeats= []
		self.selfeatids= []
		self.auto_selection= True
		self.max_depth= None
		self.min_samples_split= 2
		self.min_samples_leaf= 1
		self.n_estimators= 100
		self.classifier_inventory= {}
		self.classifier= 'DecisionTreeClassifier'
		self.model= None
		self.models= []
		self.rfe= None
		self.pipeline= None
		self.pipelines= []
		self.cv= None
		
		# - LGBM custom options
		self.early_stop_round= 10
		self.metric_lgbm= 'multi_logloss'
		self.lgbm_eval_dict= {}
		self.balance_classes= False	
		self.learning_rate= 0.1
		#self.criterion= 'gini'
		self.num_leaves= 31
		self.niters= 100

		# - Set target labels
		#self.classid_remap= {
		#	0: -1,
		#	-1: -1,
		#	1: 4,
		#	2: 5,
		#	3: 0,
		#	6: 1,
		#	23: 2,
		#	24: 3,			
		#	6000: 6,
		#}

		#self.classid_label_map= {
		#	0: "UNKNOWN",
		#	-1: "MIXED_TYPE",
		#	1: "STAR",
		#	2: "GALAXY",
		#	3: "PN",
		#	6: "HII",
		#	23: "PULSAR",
		#	24: "YSO",			
		#	6000: "QSO",
		#}

		self.multiclass= multiclass
		self.__set_target_labels(multiclass)

		# *****************************
		# ** Pre-processing
		# *****************************
		self.normalize= False	
		self.data_scaler= None
		self.norm_min= 0
		self.norm_max= 1
		

		# *****************************
		# ** Output
		# *****************************
		self.outfile= 'featdata_sel.dat'
		self.outfile_scores= 'featscores.png'
		self.outfile_scorestats= 'featscores.dat'
		self.outfile_featranks= 'featranks.dat'
		self.outfile_featimportance= 'feat_importance.dat'
		self.outfile_selfeat= 'selfeatids.dat'
		self.outfile_scaler = 'datascaler.sav'

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
				verbose=1
				#num_class=self.nclasses
			)

		else:
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


		self.classifier_inventory= {
			#"DecisionTreeClassifier": DecisionTreeClassifier(max_depth=self.max_depth),
			#"RandomForestClassifier": RandomForestClassifier(max_depth=self.max_depth, n_estimators=self.n_estimators, max_features=1),
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
		
		# - Check if model type exists in classifier inventory
		if self.classifier not in self.classifier_inventory:
			logger.error("Chosen classifier (%s) is not in the inventory, returning None!" % (self.classifier))
			return None

		# - Return classifier
		return self.classifier_inventory[self.classifier]
		

	#####################################
	##     CREATE PIPELINE
	#####################################
	def __create_pipeline(self):
		""" Build the feature selector pipeline """

		# - Create classifier inventory
		logger.info("Creating classifier inventory ...")
		self.__create_classifier_inventory()

		# - Set min/max nfeat range
		nf_min= self.nfeat_min
		nf_max= self.nfeat_max
		if nf_max==-1:
			nf_max= self.nfeatures

		self.nfeats= []	
		for i in range(nf_min,nf_max+1):
			self.nfeats.append(i)

		# - Create models
		self.model= self.__create_model()
		if self.model is None:
			logger.error("Created model is None!")
			return -1

		for i in range(len(self.nfeats)):
			m= self.__create_model()
			self.models.append(m)

		# - Define dataset split (unique for all models)
		self.cv= StratifiedKFold(n_splits=self.cv_nsplits, shuffle=True, random_state=self.cv_seed)

		# - Create RFE & pipeline
		self.rfe= RFECV(
			estimator=self.model,
			step=1,
			#cv=self.cv,
			min_features_to_select=self.nfeat_min
		)
		self.pipeline = Pipeline(
			steps=[('featsel', self.rfe),('model', self.model)]
		)

		for i in range(len(self.nfeats)):
			n= self.nfeats[i]
			r= RFE(
				estimator=self.models[i],
				#cv=self.cv,
				n_features_to_select=n
			)
			p= Pipeline(steps=[('featsel', r),('model', self.models[i])])
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
		#results, nfeats = list(), list()
		results= list()
		rfe_best= None
		score_best= -1
		nfeat_best= -1
		rfe_best_index= -1
		scores_stats= []

		#for i in range(1,self.nfeatures):
		for i in range(len(self.nfeats)):
			n= self.nfeats[i]
			p= self.pipelines[i]
			scores= cross_val_score(
				p, 
				self.data_preclassified, self.data_preclassified_targets, 
				scoring=self.scoring, 
				cv=self.cv, 
				n_jobs=self.ncores, 
				error_score='raise'
			)
			scores_mean= np.mean(scores)
			scores_std= np.std(scores)
			scores_min= np.min(scores)
			scores_max= np.max(scores)
			scores_median= np.median(scores)
			scores_q1= np.percentile(scores, 25)
			scores_q3= np.percentile(scores, 75)
			scores_stats.append([n, scores_mean, scores_std, scores_min, scores_max, scores_median, scores_q1, scores_q3])

			results.append(scores)
			#nfeats.append(i)

			if scores_mean>score_best:
				score_best= scores_mean
				nfeat_best= n
				rfe_best_index= i
			logger.info('--> nfeats=%d: score=%.3f (std=%.3f)' % (n, scores_mean, scores_std))
			
		# - Save scores stats
		logger.info("Saving score stats ...")
		scores_head= "# n mean std min max median q1 q3"
		scores_stats= np.array(scores_stats).reshape(len(self.nfeats), 8)
		Utils.write_ascii(scores_stats, self.outfile_scorestats, scores_head)			

		# - Evaluate automatically-selected model?
		rfe_best= None

		if self.auto_selection:
			logger.info("Evaluate model (automated feature selection) ...")
			scores= cross_val_score(
				self.pipeline, 
				self.data_preclassified, self.data_preclassified_targets, 
				scoring=self.scoring, 
				cv=self.cv, 
				n_jobs=self.ncores, 
				error_score='raise'
			)

			best_scores_mean= np.mean(scores)
			best_scores_std= np.std(scores)
			logger.info('Selecting best scores automatically: %.3f (std=%.3f)' % (best_scores_mean, best_scores_std))

			rfe_best= self.rfe

		else:
			logger.info("Selecting best model after scan: index=%d, n_feat=%d, score=%.3f" % (rfe_best_index, nfeat_best, score_best))
		
			rfe_best= RFE(
				estimator=self.models[rfe_best_index],
				#cv=self.cv,
				n_features_to_select=nfeat_best
			)

		# - Fit data and show which features were selected
		logger.info("Fitting RFE model on dataset ...")
		rfe_best.fit(self.data_preclassified, self.data_preclassified_targets)
	
		selfeats= rfe_best.support_
		featranks= rfe_best.ranking_
		nfeat_sel= rfe_best.n_features_
		self.selfeatids= []
		for i in range(self.data_preclassified.shape[1]):
			logger.info('Feature %d: selected? %d (rank=%.3f)' % (i, selfeats[i], featranks[i]))
			if selfeats[i]:
				self.selfeatids.append(i)

		self.selfeatids.sort()

		# - Save feature ranks
		logger.info("Saving feature ranks ...")
		N= self.data_preclassified.shape[1]
		outdata_featranks= np.concatenate(
			(np.arange(0,N).reshape(N,1), np.array(selfeats).reshape(N,1), np.array(featranks).reshape(N,1)),
			axis=1
		)

		featrank_head= "# featid selected rank"
		Utils.write_ascii(outdata_featranks, self.outfile_featranks, featrank_head)			

		# - Compute feature ranks with RFECV
		logger.info("Fitting RFECV model on dataset ...")
		self.rfe.fit(self.data_preclassified, self.data_preclassified_targets)

		feat_importances= self.rfe.estimator_.feature_importances_
		max_importance= np.max(feat_importances)
		feat_scaled_importances= feat_importances/max_importance
		
		for i in range(self.data_preclassified.shape[1]):
			logger.info('Feature %d: selected? %d (rank=%.3f, importance=%.3f)' % (i, self.rfe.support_[i], self.rfe.ranking_[i], feat_scaled_importances[i]))

		# - Save feature importance
		logger.info("Saving feature importance ...")
		N= self.data_preclassified.shape[1]
		outdata_featimportance= np.concatenate(
			(np.arange(0,N).reshape(N,1), np.array(feat_scaled_importances).reshape(N,1)),
			axis=1
		)

		featimportance_head= "# featid importance"
		Utils.write_ascii(outdata_featimportance, self.outfile_featimportance, featimportance_head)			


		# - Extract selected data columns
		logger.info("Extracting selected data columns (N=%d) from original data ..." % (nfeat_sel))
		self.data_sel= self.data[:,selfeats]
		self.data_preclassified_sel= self.data_preclassified[:,selfeats]

		# - Plot results
		logger.info("Plotting and saving feature score results ...")
		plt.boxplot(results, labels=self.nfeats, showmeans=True)
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
		else:
			logger.info("No pre-classified objects found in this file ...")

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
	
		#print(table.colnames)
		#print(table)

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

		# - Save selected feature column ids
		logger.info("Saving selected feature column ids, separated by commas ...")
		selcolids_str= ','.join(str(item) for item in self.selfeatids)
		with open(self.outfile_selfeat, 'w') as f:
			f.write(selcolids_str)
		

		return 0

	#####################################
	##     RUN
	#####################################
	def run(self, datafile, scalerfile=''):
		""" Run feature selection """
		
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



	def run(self, data, class_ids=[], snames=[], scalerfile=''):
		""" Run feature selection using input dataset """

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


	#####################################
	##     SELECT COLUMNS
	#####################################
	def __select_cols(self, selcols):
		""" Select data columns provided in selcols list """

		# - Check sel cols
		if not selcols:
			logger.error("Empty sel col list!")
			return -1

		# - Remove any duplicated col ids, sort and set colsel flags
		selcols= list(set(selcols))
		selcols.sort()
		selcolflags= [False]*self.nfeatures
		for col in selcols:
			if col<0 or col>=self.nfeatures:
				logger.error("Given sel col id %d is not in nfeature col range [0,%d]!" % (col, self.nfeatures-1))
				return -1
			selcolflags[col]= True

		print("--> Selected columns")
		print(selcols)
		print("--> Selected column flags")
		print(selcolflags)
			
		# - Extract selected data columns
		logger.info("Extracting selected data columns (N=%d) from original data ..." % (len(selcols)))
		self.data_sel= self.data[:,selcolflags]
		if self.data_preclassified_sel is not None:
			self.data_preclassified_sel= self.data_preclassified[:,selcolflags]
		self.selfeatids= selcols

		return 0


	def select_from_file(self, datafile, selcols, scalerfile=''):
		""" Select data columns provided in selcols list """
		
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
		#==   SELECT COLUMNS
		#================================
		logger.info("Extracting columns ...")
		if self.__select_cols(selcols)<0:
			logger.error("Failed to select data columns!")
			return -1

		#================================
		#==   SAVE
		#================================
		logger.info("Saving results ...")
		if self.__save()<0:
			logger.error("Failed to save results!")
			return -1

		return 0

	def select(self, data, selcols, class_ids=[], snames=[], scalerfile=''):
		""" Select data columns provided in selcols list """

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
		#==   SELECT COLUMNS
		#================================
		logger.info("Extracting columns ...")
		if self.__select_cols(selcols)<0:
			logger.error("Failed to select data columns!")
			return -1

		#================================
		#==   SAVE
		#================================
		logger.info("Saving results ...")
		if self.__save()<0:
			logger.error("Failed to save results!")
			return -1

		return 0

