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
import pickle
import json

## ASTRO MODULES
from astropy.io import ascii 

## CLUSTERING MODULES
import hdbscan
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

## GRAPHICS MODULES
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

## SCLASSIFIER MODULES
from .utils import Utils
from .utils import NoIndent, MyEncoder

##############################
##     GLOBAL VARS
##############################
from sclassifier import logger

##############################
##     ClusteringExtraData CLASS
##############################
class ClusteringExtraData(object):
	""" Class to store clustering addon data """
	
	def __init__(self):
		""" Return a ClusteringExtraData object """

		self.class_ids= []
		self.class_labels= []
		self.snames= []
		
##############################
##     Clusterer CLASS
##############################
class Clusterer(object):
	""" Class to create and train a clustering classifier
	"""
	
	def __init__(self):
		""" Return a Clusterer object """

		# - Input data
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
		
		self.excluded_objids_train= [-1,0] # Sources with these ids are considered not labelled and therefore excluded from training or metric calculation
		self.classid_label_map= {}
		self.selcols= []
		
		# *****************************
		# ** Pre-processing
		# *****************************
		self.data_scaler= None
		self.normalize= False
		self.norm_transf= "minmax" # "robust"
		self.norm_min= 0
		self.norm_max= 1
		self.reduce_dim= False
		self.reduce_dim_method= 'pca'
		self.pca= None
		self.pca_ncomps= -1
		self.pca_varthr= 0.9
		self.pca_transf_data= None
		
		# *****************************
		# ** Clustering parameters
		# *****************************
		# - Clustering model & results 
		self.clusterer= None
		self.prediction_extra_data= None
		self.add_prediction_data= False
		self.prediction_data= None
		self.nclusters= 0
		self.labels= None
		self.probs= None
		self.outlier_scores= None
		self.exemplars= None
		self.cluster_persistence= None
		self.labels_pred= None
		self.probs_pred= None

		# - Clustering model & results for pre-classified data (historical + new)
		self.clusterer_preclass= None
		self.prediction_data_preclass= None
		self.labels_preclass= None
		self.probs_preclass= None

		# - Metrics options: {'braycurtis','canberra','chebyshev','cityblock','dice','euclidean','hamming','haversine','infinity',
		#											'jaccard','kulsinski','l1','l2','mahalanobis','manhattan','matching','minkowski','p','pyfunc','rogerstanimoto',
		#                     'russellrao','seuclidean','sokalmichener','sokalsneath','wminkowski'}	
		self.metric= 'euclidean'   # this is the default
		self.metric_args= {}
		self.min_cluster_size= 5   # this is the default
		self.min_samples= None     # this is the default
		self.cluster_selection_epsilon= 0.0 # this is the defaul

		self.use_preclassified_data= True
		self.preclassified_data_minsize= 20
		#self.encoded_data_dim= 2
		#self.encoded_data_unsupervised= None
		#self.encoded_data_preclassified= None
		#self.encoded_data_supervised= None
		
		# *****************************
		# ** Draw
		# *****************************
		self.draw= False
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

		# - Draw options
		self.marker_mapping= {
			'UNKNOWN': 'o', # unknown
			'MIXED_TYPE': 'X', # mixed type
			'STAR': 'x', # star
			'GALAXY': 'D', # galaxy
			'PN': '+', # PN
			'HII': 's', # HII
			'YSO': 'P', # YSO
			'QSO': 'v', # QSO
			'PULSAR': 'd', # pulsar
		}

		self.marker_color_mapping= {
			'UNKNOWN': 'k', # unknown
			'MIXED_TYPE': 'tab:gray', # mixed type
			'STAR': 'r', # star
			'GALAXY': 'm', # galaxy
			'PN': 'g', # PN
			'HII': 'b', # HII
			'YSO': 'y', # YSO
			'QSO': 'c', # QSO
			'PULSAR': 'tab:orange', # pulsar
		}
		
		# *****************************
		# ** Output
		# *****************************
		# - Output data
		self.save_ascii= True
		self.save_json= True
		self.save_features= True
		self.save_model= True
		self.outfile_model= "clustering_model.sav"
		self.outfile_model_preclass= "clustering_preclass_model.sav"
		self.outfile= 'clustered_data.dat'
		self.outfile_json= 'clustered_data.json'
		self.outfile_plot= 'clusters.png'
		self.outfile_scaler = 'datascaler.sav'
		self.outfile_pca= 'pca_data.dat'
		self.outfile_model_pca= "pca_model.sav"
		#self.outfile_encoded_data_unsupervised= 'encoded_data_unsupervised.dat'
		#self.outfile_encoded_data_supervised= 'encoded_data_supervised.dat'
		#self.outfile_encoded_data_preclassified= 'encoded_data_preclassified.dat'

	#####################################
	##     PRE-PROCESSING
	#####################################
	def __create_scaler(self):
		""" Create scaler model for normalization """

		if self.norm_transf=="minmax":
			self.data_scaler= MinMaxScaler(feature_range=(self.norm_min, self.norm_max))
		elif self.norm_transf=="robust":
			self.data_scaler= RobustScaler()
		else:
			logger.error("Undefined/unsupported norm transf model %s!" % (self.norm_transf))	
			return -1

		return 0


	def __transform_data(self, x):
		""" Transform input data here or using a loaded scaler """

		# - Print input data min/max
		x_min= x.min(axis=0)
		x_max= x.max(axis=0)

		print("== INPUT DATA MIN/MAX ==")
		print(x_min)
		print(x_max)

		if self.data_scaler is None:
			# - Define scaler
			logger.info("Define and running data scaler ...")
			if self.__create_scaler()<0:
				logger.error("Failed to create norm scaler model!")
				return None

			# - Run scaler
			x_transf= self.data_scaler.fit_transform(x)

			if self.norm_transf=="minmax":
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


	def __reduce_data_dim(self, x, normalize=False):
		""" Reduce input data dimensionality """

		if self.reduce_dim_method=="pca":
			x_transf= self.__pca_transform(x)
		else:
			logger.error("Unknown/unsupported dimensionality reduction method (%s) given!" % (self.reduce_dim_method))
			return None

		if normalize:
			logger.info("Normalizing dim reducted data ...")
			x_norm= self.__transform_data(x_transf)
			x_transf= x_norm

		return x_transf
		

	def __pca_transform(self, x, fit=True):
		""" Apply PCA to input data """
	
		# - Check if PCA model was built
		if self.pca is None:
			logger.info("PCA model was not created, creating it now ...")
			self.__build_pca_model()

		# - Run PCA		
		nfeat= x.shape[1]
		logger.info("Running PCA on #%d dim data ..." % (nfeat))
		if fit:
			x_transf= self.pca.fit_transform(x)
		else:
			x_transf= self.pca.transform(x)

		logger.info("=> PCA variance ratio")
		print(self.pca.explained_variance_ratio_)

		#logger.info("=> PCA singular values")
		#print(self.pca.singular_values_)

		return x_transf


	def __build_pca_model(self):
		""" Build PCA model """

		if self.pca_ncomps==-1:
			logger.info("Creating PCA and selecting components with total variance ratio >= than %f ..." % (self.pca_varthr))
			self.pca= PCA(n_components=self.pca_varthr, svd_solver='full')
		else:
			logger.info("Applying PCA and selecting %d components ..." % (self.pca_ncomps))
			self.pca= PCA(n_components=self.pca_ncomps, svd_solver='full')
			
		return 0


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
				
			#if obj_id!=0 and obj_id!=-1:
			if add_to_train_list:
				row_list.append(i)
				classid_list.append(obj_id)	
				label_list.append(label)
				self.source_names_preclassified.append(source_name)
			else:
				logger.info("Exclude source with id=%s from list (excluded_ids=%s) ..." % (str(obj_id), str(self.excluded_objids_train)))
					

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

		data_shape= self.data.shape
		self.nsamples= data_shape[0]
		self.nfeatures= data_shape[1]
		logger.info("#nsamples=%d, #nfeatures=%d" % (self.nsamples, self.nfeatures))
		
		# - Normalize feature data?
		if self.normalize:
			logger.info("Normalizing feature data ...")
			data_norm= self.__transform_data(self.data)
			if data_norm is None:
				logger.error("Data transformation failed!")
				return -1
			self.data= data_norm

		# - Apply dimensionality reduction?
		if self.reduce_dim:
			logger.info("Reducing data dimensionality using method %s ..." % (self.reduce_dim_method))
			post_normalize= False
			data_transf= self.__reduce_data_dim(self.data, normalize=post_normalize)
			self.data= data_transf
			data_shape= self.data.shape
			self.nsamples= data_shape[0]
			self.nfeatures= data_shape[1]
			logger.info("#nsamples=%d, #nfeatures=%d (after dim reduction)" % (self.nsamples, self.nfeatures))

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
		ret= Utils.read_feature_data(filename, self.selcols)
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

		if self.selcols:
			self.data= Utils.get_selected_data_cols(np.array(featdata), self.selcols)
		else:
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
			data_norm= self.__transform_data(self.data)
			if data_norm is None:
				logger.error("Data transformation failed!")
				return -1
			self.data= data_norm

		# - Apply dimensionality reduction?
		if self.reduce_dim:
			logger.info("Reducing data dimensionality using method %s ..." % (self.reduce_dim_method))
			post_normalize= False
			data_transf= self.__reduce_data_dim(self.data, normalize=post_normalize)
			self.data= data_transf
			data_shape= self.data.shape
			self.nsamples= data_shape[0]
			self.nfeatures= data_shape[1]
			logger.info("#nsamples=%d, #nfeatures=%d (after dim reduction)" % (self.nsamples, self.nfeatures))

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
			self.data_labels= ["UNKNOWN"]*self.nsamples
		
		
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
			data_norm= self.__transform_data(self.data)
			if data_norm is None:
				logger.error("Data transformation failed!")
				return -1
			self.data= data_norm

		# - Apply dimensionality reduction?
		if self.reduce_dim:
			logger.info("Reducing data dimensionality using method %s ..." % (self.reduce_dim_method))
			post_normalize= False
			data_transf= self.__reduce_data_dim(self.data, normalize=post_normalize)
			self.data= data_transf
			data_shape= self.data.shape
			self.nsamples= data_shape[0]
			self.nfeatures= data_shape[1]
			logger.info("#nsamples=%d, #nfeatures=%d (after dim reduction)" % (self.nsamples, self.nfeatures))


		# - Set pre-classified data
		logger.info("Setting pre-classified data (if any) ...")
		self.__set_preclass_data()

		return 0


	def __build_model(self):
		""" Create clustering model """
	
		clusterer = hdbscan.HDBSCAN(
			min_cluster_size=self.min_cluster_size,
			min_samples=self.min_samples,
			cluster_selection_epsilon= self.cluster_selection_epsilon,
			metric=self.metric,
			prediction_data=True,
		)

		return clusterer

	#####################################
	##     PREDICT
	#####################################
	def run_predict_from_file(self, datafile, modelfile, scalerfile='', datalist_key='data'):
		""" Run precit using input dataset """

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
		logger.info("Loading the clustering model from file %s ..." % modelfile)
		try:
			self.clusterer, self.prediction_extra_data = pickle.load((open(modelfile, 'rb')))
		except Exception as e:
			logger.error("Failed to load model from file %s!" % (modelfile))
			return -1

		#================================
		#==   PREDICT
		#================================
		if self.__predict()<0:
			logger.error("Predict failed!")
			return -1

		return 0


	def run_predict(self, data, class_ids=[], snames=[], modelfile='', scalerfile=''):
		""" Run precit using input dataset """

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
		logger.info("Loading the clustering model from file %s ..." % modelfile)
		try:
			self.clusterer, self.prediction_extra_data = pickle.load((open(modelfile, 'rb')))
		except Exception as e:
			logger.error("Failed to load model from file %s!" % (modelfile))
			return -1

		#================================
		#==   PREDICT
		#================================
		if self.__predict()<0:
			logger.error("Predict failed!")
			return -1

		return 0



	def __predict(self):

		#====================================================
		#==   CHECK DATA & MODEL
		#====================================================
		# - Check if data are set
		if self.data is None:
			logger.error("Input data array is None!")
			return -1

		# - Check if clustering model is set
		if self.clusterer is None:
			logger.error("Clusterer is not set!")
			return -1

		# - Retrieve prediction data from current model
		logger.info("Retrieving prediction data from current model (if any) ...")
		self.prediction_data= self.clusterer.prediction_data_
		
		#====================================================
		#==   CLUSTER DATA USING SAVED MODEL
		#====================================================
		logger.info("Encode input data using loaded model ...")
		self.labels, self.probs = hdbscan.approximate_predict(self.clusterer, self.data)

		#================================
		#==   SAVE CLUSTERED DATA
		#================================
		logger.info("Saving results ...")
		if self.__save(self.data, self.source_names, self.data_classids, self.data_labels, self.labels, self.probs, None)<0:
			logger.error("Failed to save clustering results!")
			return -1

		#================================
		#==   PLOT
		#================================
		if self.draw:
			logger.info("Plotting results ...")
			self.__plot_predict(
				self.clusterer, 
				self.data, self.labels, self.source_names, self.data_labels, 
				self.prediction_data, self.prediction_extra_data, 
				self.outfile_plot
			)

		return 0

	
	#####################################
	##     RUN CLUSTERING
	#####################################
	def run_clustering_from_file(self, datafile, modelfile='', scalerfile='', datalist_key='data'):
		""" Run clustering using input dataset """

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
			logger.info("Loading the clustering model from file %s ..." % modelfile)
			try:
				self.clusterer, self.prediction_extra_data = pickle.load((open(modelfile, 'rb')))
			except Exception as e:
				logger.error("Failed to load model from file %s!" % (modelfile))
				return -1

		else:
			logger.info("Creating the clustering model ...")
			self.clusterer= self.__build_model()
			self.prediction_extra_data= None

		#================================
		#==   CLUSTER DATA
		#================================
		if self.__cluster()<0:
			logger.error("Failed to cluster data!")
			return -1

		return 0


	def run_clustering(self, data, class_ids=[], snames=[], modelfile='', scalerfile=''):
		""" Run clustering using input dataset """

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
		if modelfile and modelfile is not None:
			logger.info("Loading the clustering model from file %s ..." % modelfile)
			try:
				self.clusterer, self.prediction_extra_data = pickle.load((open(modelfile, 'rb')))
			except Exception as e:
				logger.error("Failed to load model from file %s!" % (modelfile))
				return -1

		else:
			logger.info("Creating the clustering model ...")
			self.clusterer= self.__build_model()
			self.prediction_extra_data= None

		#================================
		#==   CLUSTER DATA
		#================================
		if self.__cluster()<0:
			logger.error("Failed to cluster data!")
			return -1

		return 0


	def __cluster(self):
		""" Build clusterer and cluster data """

		# - Check if data are set
		if self.data is None:
			logger.error("Input data array is None!")
			return -1

		# - Check if clusterer is set
		if self.clusterer is None:
			logger.error("Clusterer is not set!")
			return -1
	
		# - Retrieve prediction data from current model
		logger.info("Retrieving prediction data from current model (if any) ...")
		try:
			self.prediction_data= self.clusterer.prediction_data_
		except Exception as e:
			self.prediction_data= None

		data_hist= None
		snames_hist= None
		classids_hist= None
		classlabels_hist= None

		if self.prediction_data is None:
			logger.info("No prediction data present in current model ...")	
		else:
			data_hist= self.prediction_data.data
			logger.info("Found prediction data (%d,%d) in current model ..." % (data_hist.shape[0], data_hist.shape[1]))

		# - Check prediction historical data (if to be used)
		if self.add_prediction_data:
			if data_hist is None:
				logger.error("Selected to cluster also historical data but current model has none!")
				return -1
			if data_hist.shape[1]!=self.data.shape[1]:
				logger.error("Historical and current data to be clustered have different shapes (%d,%d)!" % (data_hist.shape[1],self.data.shape[1]))
				return -1
			if self.prediction_extra_data is None:
				logger.error("Historical extra data is None!")
				return -1

			snames_hist= self.prediction_extra_data.snames
			classids_hist= self.prediction_extra_data.class_ids
			classlabels_hist= self.prediction_extra_data.class_labels
			if not snames_hist or not classids_hist:
				logger.error("Historical extra data lists are empty!")
				return -1

		#================================
		#==   CLUSTER DATA
		#================================
		# - Set data to be clustered
		data_all= self.data
		data_ids_all= self.data_classids	
		data_labels_all= self.data_labels
		source_names_all= self.source_names
		if self.add_prediction_data:	
			data_all= np.concatenate((data_hist, self.data), axis=0)
			data_ids_all.extend(classids_hist)
			data_labels_all.extend(classlabels_hist)
			source_names_all.extend(snames_hist)
		
		# - Cluster data
		logger.info("Clustering input data ...")
		#self.clusterer= self.clusterer.fit(data_all)
		self.clusterer.fit(data_all)

		# - Retrieve clustering results
		self.labels = self.clusterer.labels_ # shape (n_samples, )
		self.probs= self.clusterer.probabilities_ # shape (n_samples, )
		self.outlier_scores= self.clusterer.outlier_scores_ # shape (n_samples, )
		self.exemplars= self.clusterer.exemplars_  # list
		self.cluster_persistence= self.clusterer.cluster_persistence_  # shape (n_clusters, )
		
		print("self.labels")
		print(type(self.labels))
		print(self.labels)
		print(self.labels.tolist())
		print(type(self.labels.tolist()))
		

		labels_unique= set(self.labels.tolist())
		labels_unique.discard(-1) # get set of unique labels, without -1=noise
		print("labels_unique")
		print(labels_unique)
		#self.nclusters= self.labels.max()
		self.nclusters= len(labels_unique)
		logger.info("#%d clusters found ..." % (self.nclusters))
	
		
		#================================
		#==   SAVE CLUSTERED DATA
		#================================
		logger.info("Saving clustered data to file ...")
		if self.__save(data_all, source_names_all, data_ids_all, data_labels_all, self.labels, self.probs, self.outlier_scores)<0:
			logger.error("Failed to save clustering results!")
			return -1
		

		#N= data_all.shape[0]
		#print("Clustered data size is N=",N)

		#snames= np.array(source_names_all).reshape(N,1)
		#objids= np.array(data_ids_all).reshape(N,1)
		#objlabels= np.array(data_labels_all).reshape(N,1)

		#print("snames shape")
		#print(snames.shape)
		#print("objids shape")
		#print(objids.shape)
		#print("objlabels shape")
		#print(objlabels.shape)
		#print("self.labels shape")
		#print(self.labels.shape)
		#print("self.probs shape")
		#print(self.probs.shape)
		#print("self.outlier_scores")
		#print(self.outlier_scores.shape)

		#clustered_data= np.concatenate(
		#	(snames, data_all, objids, self.labels.reshape(N,1), self.probs.reshape(N,1), self.outlier_scores.reshape(N,1)),
		#	axis=1
		#)

		#znames_counter= list(range(1, self.nfeatures+1))
		#znames= '{}{}'.format('z',' z'.join(str(item) for item in znames_counter))

		#head= '{} {} {}'.format("# sname",znames,"id clust_id clust_prob outlier_score")
		#Utils.write_ascii(clustered_data, self.outfile, head)	

		#================================
		#==   SAVE MODEL
		#================================
		# - Save model to file
		if self.save_model:
			# - Create clustering extra data
			clust_extra_data= ClusteringExtraData()
			clust_extra_data.snames= source_names_all
			clust_extra_data.class_ids= data_ids_all
			clust_extra_data.class_labels= data_labels_all

			# - Dump model & extra data
			logger.info("Dumping model & extra data to file %s ..." % self.outfile_model)
			#pickle.dump(self.clusterer, open(self.outfile_model, 'wb'))
			pickle.dump([self.clusterer, clust_extra_data], open(self.outfile_model, 'wb'))

		#================================
		#==   PLOT
		#================================
		if self.draw:
			logger.info("Plotting results ...")
			self.__plot(
				self.clusterer, 
				data_all, source_names_all, data_labels_all, 
				self.outfile_plot
			)

		
		return 0



	#####################################
	##     SAVE DATA
	#####################################
	def __save(self, data, snames, data_classids, data_labels, clust_labels, clust_probs, clust_outlier_scores=None):
		""" Save data  """
		
		###################################
		##     SAVE ASCII
		###################################
		if self.save_ascii and self.__save_to_ascii(data, snames, data_classids, data_labels, clust_labels, clust_probs, clust_outlier_scores)<0:
			logger.warn("Failed to save outputs to ascii format...")
		
		###################################
		##     SAVE JSON
		###################################
		if self.save_json and self.__save_to_json()<0:
			logger.warn("Failed to save outputs to json format...")
			
		###################################
		##     SAVE MODEL
		###################################	
		#if self.model and self.save_model:
		#	logger.info("Saving model to file %s ..." % (self.outfile_model))
		#	pickle.dump(self.model, open(self.outfile_model, 'wb'))
	
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
				clust_id= self.labels[i]
				clust_prob= self.probs[i]
				
				d= self.datalist_json[i]
				if self.save_features:
					d['feats']= NoIndent(feats)
				else:
					del d['feats']
				d['clust_id']= int(clust_id)
				d['clust_prob']= float(clust_prob)
				if self.outlier_scores is not None:
					d['clust_outlier_score']= float(self.outlier_scores[i])
					
				outdata["data"].append(d)

			# - Save to json 
			logger.info("Saving unsupervised encoded data to json file %s ..." % (self.outfile_json))	
			with open(self.outfile_json, 'w') as fp:
				json.dump(outdata, fp, cls=MyEncoder, indent=2)
		
		return 0
	
	def __save_to_ascii(self, data, source_names, data_classids, data_labels, clust_labels, clust_probs, clust_outlier_scores):
		""" Save data to ascii """

		# - Check if selected data is available
		if data is None:
			logger.error("Input data is None!")
			return -1
			
		N= data.shape[0]
		Nfeat= data.shape[1]
		
		print("Saving cluster data (N=%d) to file ..." % N)

		# - Concatenate data for saving
		logger.info("Concatenate data for saving ...")
		snames= np.array(source_names).reshape(N,1)
		clust_labels= np.array(clust_labels).reshape(N,1)
		clust_probs= np.array(clust_probs).reshape(N,1)
		if clust_outlier_scores is not None:
			clust_outlier_scores= np.array(clust_outlier_scores).reshape(N,1)
		
		if isinstance(data_classids[0], list): # concatenate ids/labels comma separated
			objids_concat= [','.join(map(str,item)) for item in data_classids]
			objlabels_concat= [','.join(map(str,item)) for item in data_labels]
			objids= np.array(objids_concat).reshape(N,1)
			objlabels= np.array(objlabels_concat).reshape(N,1)
		else:
			objids= np.array(data_classids).reshape(N,1)
			objlabels= np.array(data_labels).reshape(N,1)
			
		if self.save_features:
			znames_counter= list(range(1, Nfeat+1))
			znames= '{}{}'.format('z',' z'.join(str(item) for item in znames_counter))
		
			if self.save_labels_in_ascii:
				if clust_outlier_scores is None:
					clustered_data= np.concatenate(
						(snames, data, objlabels, clust_labels, clust_probs),
						axis=1
					)
					head= '{} {} {}'.format("# sname",znames,"label clust_id clust_prob")
				else:
					clustered_data= np.concatenate(
						(snames, data, objlabels, clust_labels, clust_probs, clust_outlier_scores),
						axis=1
					)
					head= '{} {} {}'.format("# sname",znames,"label clust_id clust_prob clust_outlier_score")
			else:
				if clust_outlier_scores is None:
					clustered_data= np.concatenate(
						(snames, data, objids, clust_labels, clust_probs),
						axis=1
					)
					head= '{} {} {}'.format("# sname",znames,"id clust_id clust_prob")
				else:
					clustered_data= np.concatenate(
						(snames, data, objids, clust_labels, clust_probs, clust_outlier_scores),
						axis=1
					)
					head= '{} {} {}'.format("# sname",znames,"id clust_id clust_prob clust_outlier_score")
		else:
			if self.save_labels_in_ascii:
				if clust_outlier_scores is None:
					clustered_data= np.concatenate(
						(snames, objlabels, clust_labels, clust_probs),
						axis=1
					)
					head= "# sname label clust_id clust_prob"
				else:
					clustered_data= np.concatenate(
						(snames, objlabels, clust_labels, clust_probs, clust_outlier_scores),
						axis=1
					)
					head= "# sname label clust_id clust_prob clust_outlier_score"
			else:
				if clust_outlier_scores is None:
					clustered_data= np.concatenate(
						(snames, objids, clust_labels, clust_probs),
						axis=1
					)
					head= "# sname id clust_id clust_prob"
				else:
					clustered_data= np.concatenate(
						(snames, objids, clust_labels, clust_probs, clust_outlier_scores),
						axis=1
					)
					head= "# sname id clust_id clust_prob clust_outlier_score"
				
		# - Save clustering data 
		logger.info("Saving clustering output data to file %s ..." % (self.outfile))
		Utils.write_ascii(clustered_data, self.outfile, head)	

		return 0

	#####################################
	##     RUN PCA
	#####################################
	def run_pca_from_file(self, datafile, modelfile='', scalerfile='', datalist_key='data'):
		""" Run PCA using input dataset """

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
			logger.info("Loading the PCA model from file %s ..." % (modelfile))
			fit_pca= False
			try:
				self.pca = pickle.load((open(modelfile, 'rb')))
			except Exception as e:
				logger.error("Failed to load PCA model from file %s!" % (modelfile))
				return -1

		else:
			logger.info("Creating the PCA model ...")
			fit_pca= True
			self.__build_pca_model()

		#================================
		#==   APPLY PCA TO DATA
		#================================
		if self.__run_pca(fit_pca)<0:
			logger.error("Failed to run PCA on input data!")
			return -1

		return 0
	
	def run_pca(self, data, class_ids=[], snames=[], modelfile='', scalerfile=''):
		""" Run clustering using input dataset """

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
			logger.info("Loading the PCA model from file %s ..." % (modelfile))
			fit_pca= False
			try:
				self.pca = pickle.load((open(modelfile, 'rb')))
			except Exception as e:
				logger.error("Failed to load PCA model from file %s!" % (modelfile))
				return -1

		else:
			logger.info("Creating the PCA model ...")
			fit_pca= True
			self.__build_pca_model()

		#================================
		#==   APPLY PCA TO DATA
		#================================
		if self.__run_pca(fit_pca)<0:
			logger.error("Failed to apply PCA to data!")
			return -1

		return 0

	def __run_pca(self, fit=True):
		""" Interna method to run PCA """

		# - Check if data are set
		if self.data is None:
			logger.error("Input data array is None!")
			return -1

		#================================
		#==   APPLY PCA
		#================================
		# - Set data 
		data_all= self.data
		data_ids_all= self.data_classids	
		data_labels_all= self.data_labels
		source_names_all= self.source_names
		
		# - PCA
		logger.info("Apply PCA to input data  ...")
		data_transf= self.__pca_transform(data_all, fit)

		#================================
		#==   SAVE PCA DATA
		#================================
		logger.info("Saving PCA data to file ...")
		N= data_all.shape[0]
		snames= np.array(source_names_all).reshape(N,1)
		objids= np.array(data_ids_all).reshape(N,1)
		
		pca_data= np.concatenate(
			(snames, data_transf, objids),
			axis=1
		)
		
		self.pca_transf_data= data_transf

		Nfeat_pca= data_transf.shape[1]
		znames_counter= list(range(1,Nfeat_pca+1))
		znames= '{}{}'.format('z',' z'.join(str(item) for item in znames_counter))

		head= '{} {} {}'.format("# sname",znames,"id")
		Utils.write_ascii(pca_data, self.outfile_pca, head)	
		
		#================================
		#==   SAVE PCA MODEL
		#================================
		# - Save model to file
		if self.save_model:
			logger.info("Dumping PCA to file %s ..." % self.outfile_model_pca)
			pickle.dump(self.pca, open(self.outfile_model_pca, 'wb'))
			

		return 0

	#####################################
	##     PLOTTING
	#####################################
	def __plot(self, clusterer, data, snames, class_labels, outfile):
		""" Plot clusters """

		# - Set variable names
		data_shape= data.shape
		ndim= data_shape[1]
		N= data_shape[0]
		varnames_counter= list(range(1,ndim+1))
		varnames= '{}{}'.format('z',' z'.join(str(item) for item in varnames_counter))
		#nclusters= clusterer.labels_.max()
		labels_unique= set(clusterer.labels_.tolist())
		labels_unique.discard(-1)
		nclusters= len(labels_unique)

		print("len(clusterer.labels_)")
		print(len(clusterer.labels_))
		print(clusterer.labels_)
		print("nclusters")
		print(nclusters)

		# - Set cluster colors
		palette = sns.color_palette('deep', nclusters+1)
		cluster_colors = [palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusterer.labels_]
		cluster_member_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)]
		
		# - Pair-wise Scatter Plots
		###cols = ['density', 'residual sugar', 'total sulfur dioxide', 'fixed acidity']
		###pp = sns.pairplot(wines[cols], size=1.8, aspect=1.8, plot_kws=dict(edgecolor="k", linewidth=0.5), diag_kind="kde", diag_kws=dict(shade=True))
		#pp = sns.pairplot(data, size=1.8, aspect=1.8, plot_kws=dict(edgecolor="k", s=50, linewidth=0, c=cluster_member_colors, alpha=0.25), diag_kind="kde", diag_kws=dict(shade=True))
		#fig = pp.fig 
		#fig.subplots_adjust(top=0.93, wspace=0.3)
		#t = fig.suptitle('Clustering Plots', fontsize=14)

		# - If data dimension is >2 project with tSNE
		data_draw= data
		if ndim>2:
			logger.info("Data size is >2, projecting to 2D with tSNE (assuming default pars) ...")
			data_draw= TSNE().fit_transform(data)

		# - Display a 2D plot of clustered data
		logger.info("Plot a 2D plot of the clustered data ...")
		plt.figure(figsize=(12, 10))
	
		#plt.scatter(*data.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)

		for i in range(N):
			source_name= snames[i]
			source_label= class_labels[i]
			marker= 'o'
			color= 'k'
			obj_id= 0
			has_label= source_label in self.marker_mapping
			if has_label:
				marker= self.marker_mapping[source_label]
				color= self.marker_color_mapping[source_label]

			cluster_color= cluster_member_colors[i]

			#plt.scatter(data_draw[i,0], data_draw[i,1], s=50, linewidth=0, color=color, marker=marker, alpha=0.25)
			plt.scatter(data_draw[i,0], data_draw[i,1], s=50, linewidth=0, color=cluster_color, marker=marker, alpha=0.25)

		
		plt.xlabel("z0")
		plt.ylabel("z1")
		plt.savefig(outfile)
		#plt.show()


	def __plot_predict(self, clusterer, data_test, labels_test, snames_test, class_labels_test, prediction_data, prediction_extra_data, outfile):
		""" Plot clusters """

		###########################################################################
		## NB: data is the data clustered assuming the loaded clustering model
		##     data_test is the new data passed by user 
		###########################################################################
		
		# - Retrieve prediction data
		data= prediction_data.data
		snames= prediction_extra_data.snames
		class_labels= prediction_extra_data.class_labels

		# - Set variable names
		data_shape= data.shape
		ndim= data_shape[1]
		N= data_shape[0]
		varnames_counter= list(range(1,ndim+1))
		varnames= '{}{}'.format('z',' z'.join(str(item) for item in varnames_counter))
		#nclusters= clusterer.labels_.max()
		labels_unique= set(clusterer.labels_.tolist())
		labels_unique.discard(-1)
		nclusters= len(labels_unique)

		# - Set cluster colors
		palette = sns.color_palette('deep', nclusters+1)
		cluster_colors = [palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusterer.labels_]
		cluster_member_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)]
				
		# - If data dimension is >2, merge data & data_test and project all with tSNE
		data_all= np.concatenate((data, data_test), axis=0)
		data_merged_draw= data_all
		if ndim>2:
			logger.info("Data size is >2, projecting all data to 2D with tSNE (assuming default pars) ...")
			data_merged_draw= TSNE().fit_transform(data_all)

		# - Display a 2D plot of clustered data
		logger.info("Plot a 2D plot of the clustered data ...")
		plt.figure(figsize=(12, 10))
	
		for i in range(N):
			source_name= snames[i]
			source_label= class_labels[i]
			marker= 'o'
			color= 'k'
			obj_id= 0
			has_label= source_label in self.marker_mapping
			if has_label:
				marker= self.marker_mapping[source_label]
				color= self.marker_color_mapping[source_label]

			cluster_color= cluster_member_colors[i]

			index= i
			#plt.scatter(data_merged_draw[index,0], data_merged_draw[index,1], s=50, linewidth=0, color=color, marker=marker, alpha=0.25)
			plt.scatter(data_merged_draw[index,0], data_merged_draw[index,1], s=50, linewidth=0, color=cluster_color, marker=marker, alpha=0.25)

		# - Plot test data
		N_test= data_test.shape[0]
		cluster_colors_test= [palette[x] if x >= 0 else (0.1, 0.1, 0.1) for x in labels_test]
			
		for i in range(N_test):
			source_name= snames_test[i]
			source_label= class_labels_test[i]
			marker= 'o'
			color= 'k'
			obj_id= 0
			has_label= source_label in self.marker_mapping
			if has_label:
				marker= self.marker_mapping[source_label]
				color= self.marker_color_mapping[source_label]

			cluster_color= cluster_colors_test[i]

			index= N + i
			plt.scatter(data_merged_draw[index,0], data_merged_draw[index,1], s=80, linewidths=1, edgecolors='k', c=cluster_color, marker=marker)		

		plt.xlabel("z0")
		plt.ylabel("z1")
		plt.savefig(outfile)
		#plt.show()
	

