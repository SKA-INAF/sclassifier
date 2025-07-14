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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

## UMAP MODULES
import umap

## GRAPHICS MODULES
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
##     FeatExtractorUMAP CLASS
##############################
class FeatExtractorUMAP(object):
	""" Class to create and train a UMAP feature extractor """
	
	def __init__(self):
		""" Return a UMAP feature extractor object """

		# *****************************
		# ** Input data
		# *****************************
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
		self.selcols= []
		
		self.excluded_objids_train= [-1,0] # Sources with these ids are considered not labelled and therefore excluded from training or metric calculation
		
		# *****************************
		# ** Pre-processing
		# *****************************
		self.normalize= False
		self.norm_min= 0
		self.norm_max= 1
		self.data_scaler= None

		# *****************************
		# ** UMAP parameters
		# *****************************
		# - Reducer & parameters
		self.reducer= None
		self.preclassified_data_minsize= 20
		self.encoded_data_dim= 2
		self.encoded_data_unsupervised= None
		self.encoded_data_preclassified= None
		self.encoded_data_supervised= None
		self.learned_transf= None
		self.metric= 'euclidean' # {'manhattan','chebyshev','minkowski','mahalanobis','seuclidean',...}
		self.metric_args= {}
		self.target_metric= 'categorical'
		self.target_metric_args= {}
		self.min_dist= 0.1 # 0.1 is default, larger values (close to 1) --> broad structures, small values (close to 0) --> cluster objects
		self.n_neighbors= 15 # 15 is default
		self.embedding_init= 'spectral' # {'spectral','random'}
		self.embedding_spread= 1.0 # default=1.0
		self.embedding_apar= None # 1.576943460405378 in digit example
		self.embedding_bpar= None # 0.8950608781227859 in digit example
		self.op_mix_ratio= 1.0 # default=1.0, in range [0,1]
		self.negative_sample_rate= 5 # default=5
		self.transform_queue_size= 4.0 # default=4
		self.angular_rp_forest= False # default=false
		self.local_connectivity= 1.0 # default=1
		self.nepochs= None # default=None
		self.random_seed= 42

		self.classid_label_map= {}
		
		#self.run_semisupervised= False
		self.run_supervised= False
		#self.use_preclassified_data= True

		
		# *****************************
		# ** Draw
		# *****************************
		self.draw= False
		self.marker_mapping= {}
		self.marker_color_mapping= {}

		# *****************************
		# ** Output
		# *****************************
		self.outfile_encoded_data_unsupervised= 'encoded_data_unsupervised.dat'
		self.outfile_encoded_data_supervised= 'encoded_data_supervised.dat'
		self.outfile_encoded_data_preclassified= 'encoded_data_preclassified.dat'
		self.outfile_encoded_data_unsupervised_json= 'encoded_data_unsupervised.json'
		self.outfile_scaler = 'datascaler.sav'
		self.outfile_model= 'umap_model.sav'
		self.save_labels_in_ascii= False
		self.save_ascii= True
		self.save_json= True
		self.save_model= True
		self.save_scaler= True

	#####################################
	##     SETTERS/GETTERS
	#####################################
	def set_encoded_data_unsupervised_outfile(self,outfile):
		""" Set name of encoded data output unsupervised file """
		self.outfile_encoded_data_unsupervised= outfile	

	def set_encoded_data_supervised_outfile(self,outfile):
		""" Set name of encoded data output supervised file """
		self.outfile_encoded_data_supervised= outfile	

	def set_encoded_data_preclassified_outfile(self,outfile):
		""" Set name of encoded preclassified data output file """
		self.outfile_encoded_data_preclassified= outfile	
		
	def set_encoded_data_unsupervised_json_outfile(self,outfile):
		""" Set name of encoded data json output unsupervised file """
		self.outfile_encoded_data_unsupervised_json= outfile	

	def set_encoded_data_dim(self,dim):
		""" Set encoded data dim """
		self.encoded_data_dim= dim

	def set_n_neighbors(self,n):
		""" Set neighbor number parameter """
		self.n_neighbors= n

	def set_min_dist(self,d):
		""" Set min distance parameter"""
		self.min_dist= d


	def set_classid_label_map_astroclass(self):
		""" Set class id-label map """

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

	def set_classid_label_map_morphclass(self):
		""" Set class id-label map """

		self.classid_label_map= {
			0: "UNKNOWN",
			1: "POINT-LIKE",
			2: "COMPACT",
			3: "EXTENDED",
			4: "EXTENDED-MULTISLAND",
			5: "DIFFUSE"
		}


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
				logger.debug("Exclude source with id=%s from list (excluded_ids=%s) ..." % (str(obj_id), str(self.excluded_objids_train)))
				

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

		# - Select data columns?
		if self.selcols:
			self.data= Utils.get_selected_data_cols(np.array(featdata), self.selcols)
		else:
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
		ret= Utils.read_feature_data(filename, self.selcols)
		if not ret:
			logger.error("Failed to read data from file %s!" % (filename))
			return -1

		data= ret[0]
		snames= ret[1]
		classids= ret[2]

		return self.set_data(data, snames, classids)
		
	
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
	##     SET MODEL
	#####################################
	def __build_model(self):
		""" Create UMAP model """
	
		reducer= umap.UMAP(
			random_state=self.random_seed,
			n_components=self.encoded_data_dim,
			metric=self.metric,
			n_neighbors=self.n_neighbors,
			min_dist=self.min_dist, 
			a=self.embedding_apar, 
			angular_rp_forest=self.angular_rp_forest,
			b=self.embedding_bpar, 
			init=self.embedding_init,
			local_connectivity=self.local_connectivity, 
			metric_kwds=self.metric_args,
			n_epochs=self.nepochs,
   		negative_sample_rate=self.negative_sample_rate, 
			set_op_mix_ratio=self.op_mix_ratio,
   		spread=self.embedding_spread, 
			target_metric=self.target_metric, 
			target_metric_kwds=self.target_metric_args,
   		transform_queue_size=self.transform_queue_size, 
			transform_seed=self.random_seed, 
			verbose=False
		)

		return reducer


	#####################################
	##     PREDICT
	#####################################
	def run_predict_from_file(self, datafile, modelfile, scalerfile='', datalist_key='data'):
		""" Run predict using input dataset """

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
		logger.info("Loading the UMAP reducer from file %s ..." % modelfile)
		try:
			self.reducer= pickle.load((open(modelfile, 'rb')))
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
		""" Run predict using input dataset """

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
		logger.info("Loading the UMAP reducer from file %s ..." % modelfile)
		try:
			self.reducer= pickle.load((open(modelfile, 'rb')))
		except Exception as e:
			logger.error("Failed to load model from file %s!" % (modelfile))
			return -1

		#================================
		#==   PREDICT
		#================================
		if self.__predict()<0:
			logger.error("Predict failed!")
			return -1

		#================================
		#==   PLOT
		#================================
		if self.draw:
			logger.info("Plotting results ...")
			self.plot()

		return 0



	def __predict(self):

		#====================================================
		#==   CHECK DATA & MODEL
		#====================================================
		# - Check if data are set
		if self.data is None:
			logger.error("Input data array is None!")
			return -1

		# - Check if reducer is set
		if self.reducer is None:
			logger.error("UMAP reducer is not set!")
			return -1

		#====================================================
		#==   ENCODE DATA
		#====================================================
		logger.info("Encode input data using loaded model ...")
		self.encoded_data_unsupervised= self.reducer.transform(self.data)

		#================================
		#==   SAVE ENCODED DATA
		#================================
		if self.__save_encoded_data()<0:
			logger.warn("Failed to save encoded data to file!")

		return 0

	
	#####################################
	##     TRAIN
	#####################################
	def run_train_from_file(self, datafile, modelfile='', scalerfile='', datalist_key='data'):
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
		if modelfile!="":
			logger.info("Loading the UMAP reducer from file %s ..." % modelfile)
			try:
				self.reducer= pickle.load((open(modelfile, 'rb')))
			except Exception as e:
				logger.error("Failed to load model from file %s!" % (modelfile))
				return -1

		else:
			logger.info("Creating the UMAP reducer ...")
			self.reducer= self.__build_model()

		#================================
		#==   TRAIN MODEL
		#================================
		if self.__train()<0:
			logger.error("Failed to train!")
			return -1
			
		#================================
		#==   PLOT
		#================================
		if self.draw:
			logger.info("Plotting results ...")
			self.plot()

		return 0


	def run_train(self, data, class_ids=[], snames=[], modelfile='', scalerfile=''):
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
			logger.error("Failed to read datafile %s!" % datafile)
			return -1

		#================================
		#==   LOAD MODEL
		#================================
		if modelfile!="":
			logger.info("Loading the UMAP reducer from file %s ..." % modelfile)
			try:
				self.reducer= pickle.load((open(modelfile, 'rb')))
			except Exception as e:
				logger.error("Failed to load model from file %s!" % (modelfile))
				return -1

		else:
			logger.info("Creating the UMAP reducer ...")
			self.reducer= self.__build_model()

		#================================
		#==   TRAIN MODEL
		#================================
		if self.__train()<0:
			logger.error("Failed to train!")
			return -1

		#================================
		#==   PLOT
		#================================
		if self.draw:
			logger.info("Plotting results ...")
			self.plot()

		return 0


	def __train(self):
		""" Build and train/test reducer """

		# - Check if data are set
		if self.data is None:
			logger.error("Input data array is None!")
			return -1

		# - Check if reducer is set
		if self.reducer is None:
			logger.error("UMAP reducer is not set!")
			return -1

		#==========================================================
		#==   FIT PRE-CLASSIFIED DATA (IF AVAILABLE) SUPERVISED
		#==========================================================
		# - This is performed if we have preclassified data AND if labels are not lists (e.g. single-label classification)
		#if self.use_preclassified_data:
		if self.run_supervised:
			is_single_label= (self.data_preclassified.ndim==1)
			if self.data_preclassified is not None and len(self.data_preclassified)>=self.preclassified_data_minsize and is_single_label:
				logger.info("Fitting input pre-classified data in a supervised way ...")
				#self.learned_transf= self.reducer.fit(self.data_preclassified, self.data_preclassified_classids)
				self.learned_transf= self.reducer.fit(self.data_preclassified, np.array(self.data_preclassified_classids))
				self.encoded_data_preclassified= self.learned_transf.transform(self.data_preclassified)

		#================================
		#==   FIT DATA UNSUPERVISED
		#================================
		logger.info("Fitting input data in a completely unsupervised way ...")
		self.encoded_data_unsupervised= self.reducer.fit_transform(self.data)

		# - Save model to file
		if self.save_model:
			logger.info("Dumping model to file %s ..." % self.outfile_model)
			pickle.dump(self.reducer, open(self.outfile_model, 'wb'))

		#====================================================
		#==   ENCODE DATA USING LEARNED TRANSFORM (IF DONE)
		#====================================================
		if self.learned_transf is not None:
			logger.info("Encode input data using learned transform on pre-classified data ...")
			self.encoded_data_supervised= self.learned_transf.transform(self.data)

		#================================
		#==   SAVE ENCODED DATA
		#================================
		if self.__save_encoded_data()<0:
			logger.warn("Failed to save encoded data to file!")
		
		return 0


	def __save_encoded_data_to_ascii(self):
		""" Save encoded data to ascii """
		
		# - Unsupervised encoded data
		logger.info("Saving unsupervised encoded data to file ...")
		N= self.encoded_data_unsupervised.shape[0]
		print("Unsupervised encoded data shape=", self.encoded_data_unsupervised.shape)
		print("Unsupervised encoded data N=", N)

		snames= np.array(self.source_names).reshape(N,1)
		
		if isinstance(self.data_classids[0], list): # concatenate ids/labels comma separated
			objids_concat= [','.join(map(str,item)) for item in self.data_classids]
			objlabels_concat= [','.join(map(str,item)) for item in self.data_labels]
			objids= np.array(objids_concat).reshape(N,1)
			objlabels= np.array(objlabels_concat).reshape(N,1)
		else:
			objids= np.array(self.data_classids).reshape(N,1)
			objlabels= np.array(self.data_labels).reshape(N,1)
			
		# - Save unsupervised encoded data
		znames_counter= list(range(1,self.encoded_data_dim+1))
		znames= '{}{}'.format('z',' z'.join(str(item) for item in znames_counter))
		
		if self.save_labels_in_ascii:
			enc_data= np.concatenate(
				(snames, self.encoded_data_unsupervised, objlabels),
				axis=1
			)
			head= '{} {} {}'.format("# sname",znames,"label")
		else:
			enc_data= np.concatenate(
				(snames, self.encoded_data_unsupervised, objids),
				axis=1
			)
			head= '{} {} {}'.format("# sname",znames,"id")


		print("snames")
		print(snames[0])
		print("objids")
		print(objids[0])
		print("enc_data")
		print(enc_data[0])
		

		Utils.write_ascii(enc_data, self.outfile_encoded_data_unsupervised, head)	

		# - Supervised encoded data
		if self.encoded_data_supervised is not None:
			logger.info("Saving supervised encoded data to file ...")
			N= self.encoded_data_supervised.shape[0]
			print("Supervised encoded data shape=",self.encoded_data_supervised.shape)
			print("Supervised encoded data N=",N)

			if self.save_labels_in_ascii:
				enc_data= np.concatenate(
					(snames, self.encoded_data_supervised, objids),
					axis=1
				)
			else:
				enc_data= np.concatenate(
					(snames, self.encoded_data_supervised, objlabels),
					axis=1
				)
			
			Utils.write_ascii(enc_data, self.outfile_encoded_data_supervised, head)	

		# - Pre-classified data
		if self.encoded_data_preclassified is not None:
			logger.info("Saving pre-classified encoded data to file ...")
			N= self.encoded_data_preclassified.shape[0]
			print("Pre-classified encoded data shape=", self.encoded_data_preclassified.shape)
			print("Pre-classified encoded data N=", N)

			snames_preclass= np.array(self.source_names_preclassified).reshape(N,1)
			
			if isinstance(self.data_classids[0], list): # concatenate ids/labels comma separated
				objids_preclass_concat= [','.join(map(str,item)) for item in self.data_preclassified_classids]
				objlabels_preclass_concat= [','.join(map(str,item)) for item in self.data_preclassified_labels]
				objids_preclass= np.array(objids_preclass_concat).reshape(N,1)
				objlabels_preclass= np.array(objlabels_preclass_concat).reshape(N,1)
			else:
				objids_preclass= np.array(self.data_preclassified_classids).reshape(N,1)
				objlabels_preclass= np.array(self.data_preclassified_labels).reshape(N,1)
			
			if self.save_labels_in_ascii:
				enc_data= np.concatenate(
					(snames_preclass, self.encoded_data_preclassified, objlabels_preclass),
					axis=1
				)
			else:
				enc_data= np.concatenate(
					(snames_preclass, self.encoded_data_preclassified, objids_preclass),
					axis=1
				)

			Utils.write_ascii(enc_data, self.outfile_encoded_data_preclassified, head)	


		return 0
		
	def __save_encoded_data_to_json(self):
		""" Save encoded data to json format """
		
		# - Save unsupervised encoded data
		if self.datalist_json is not None:
			N= self.encoded_data_unsupervised.shape[0]
		
			# - Loop over data and replace original feats with UMAP feats
			outdata= {"data": []}
			
			for i in range(N):
				feats_umap= list(self.encoded_data_unsupervised[i])
				feats_umap= [float(item) for item in feats_umap]
				
				d= self.datalist_json[i]
				d['feats']= NoIndent(feats_umap)
				outdata["data"].append(d)

			# - Save to json 
			logger.info("Saving unsupervised encoded data to json file %s ..." % (self.outfile_encoded_data_unsupervised_json))	
			with open(self.outfile_encoded_data_unsupervised_json, 'w') as fp:
				json.dump(outdata, fp, cls=MyEncoder, indent=2)
		
		return 0
		
		
	def __save_encoded_data(self):
		""" Save encoded data """

		###################################
		##     SAVE ASCII
		###################################
		if self.save_ascii and self.__save_encoded_data_to_ascii()<0:
			logger.warn("Failed to save outputs to ascii format...")
		
		###################################
		##     SAVE JSON
		###################################
		if self.save_json and self.__save_encoded_data_to_json()<0:
			logger.warn("Failed to save outputs to json format...")

		return 0


	def plot(self):
		""" Plot results """

		#================================
		#==   PLOT ENCODED DATA
		#================================
		# - Display a 2D plot of the encoded data in the latent space
		logger.info("Plot a 2D plot of the UMAP encoded data in the latent space ...")
		plt.figure(figsize=(12, 10))

		N= self.encoded_data_unsupervised.shape[0]
		print("N=%d" % N)
		for i in range(N):
			source_name= self.source_names[i]
			source_label= self.data_labels[i]
			marker= 'o'
			color= 'k'
			obj_id= 0
			has_label= source_label in self.marker_mapping
			if has_label:
				marker= self.marker_mapping[source_label]
				color= self.marker_color_mapping[source_label]

			plt.scatter(self.encoded_data_unsupervised[i,0], self.encoded_data_unsupervised[i,1], color=color, marker=marker)

		#plt.scatter(self.encoded_data[:, 0], self.encoded_data[:, 1])
		#plt.colorbar()
		plt.xlabel("z0")
		plt.ylabel("z1")
		plt.savefig('latent_data_umap.png')
		#plt.show()

