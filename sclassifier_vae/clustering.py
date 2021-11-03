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

## ASTRO MODULES
from astropy.io import ascii 

## CLUSTERING MODULES
import hdbscan

## GRAPHICS MODULES
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

## SCLASSIFIER MODULES
from .utils import Utils

##############################
##     GLOBAL VARS
##############################
from sclassifier_vae import logger

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
		self.data_labels= []
		self.data_classids= []
		self.source_names= []
		self.source_names_preclassified= []
		
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

		self.use_preclassified_data= True
		self.preclassified_data_minsize= 20
		#self.encoded_data_dim= 2
		#self.encoded_data_unsupervised= None
		#self.encoded_data_preclassified= None
		#self.encoded_data_supervised= None
		
		
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
		

		# - Output data
		self.dump_model= True
		self.outfile_model= "clustering_model.sav"
		self.outfile_model_preclass= "clustering_preclass_model.sav"
		self.outfile= 'clustered_data.dat'
		self.outfile_plot= 'clusters.png'
		#self.outfile_encoded_data_unsupervised= 'encoded_data_unsupervised.dat'
		#self.outfile_encoded_data_supervised= 'encoded_data_supervised.dat'
		#self.outfile_encoded_data_preclassified= 'encoded_data_preclassified.dat'

	#####################################
	##     SETTERS/GETTERS
	#####################################
	#def set_encoded_data_unsupervised_outfile(self,outfile):
	#	""" Set name of encoded data output unsupervised file """
	#	self.outfile_encoded_data_unsupervised= outfile	

	#def set_encoded_data_supervised_outfile(self,outfile):
	#	""" Set name of encoded data output supervised file """
	#	self.outfile_encoded_data_supervised= outfile	

	#def set_encoded_data_preclassified_outfile(self,outfile):
	#	""" Set name of encoded preclassified data output file """
	#	self.outfile_encoded_data_preclassified= outfile	

	#def set_encoded_data_dim(self,dim):
	#	""" Set encoded data dim """
	#	self.encoded_data_dim= dim

	

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
				
			if obj_id!=0 and obj_id!=-1:
				row_list.append(i)
				classid_list.append(obj_id)	
				label_list.append(label)
				self.source_names_preclassified.append(source_name)				

		if row_list:	
			self.data_preclassified= self.data[row_list,:]
			self.data_preclassified_labels= np.array(label_list)
			self.data_preclassified_classids= np.array(classid_list)

		
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
			label= self.classid_label_map[classid]

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
				label= self.classid_label_map[classid]
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
		
		# - Set pre-classified data
		logger.info("Setting pre-classified data (if any) ...")
		self.__set_preclass_data()

		return 0


	def __build_model(self):
		""" Create clustering model """
	
		clusterer = hdbscan.HDBSCAN(
			min_cluster_size=self.min_cluster_size,
			min_samples=self.min_samples,
			metric=self.metric,
			prediction_data=True,
		)

		return clusterer

	#####################################
	##     PREDICT
	#####################################
	def run_predict(self, datafile, modelfile):
		""" Run precit using input dataset """

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


	def run_predict(self, data, class_ids=[], snames=[], modelfile=''):
		""" Run precit using input dataset """

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
		logger.info("Saving unsupervised encoded data to file ...")
		N= self.data.shape[0]
		print("Cluster data N=",N)

		snames= np.array(self.source_names).reshape(N,1)
		objids= np.array(self.data_classids).reshape(N,1)	
		clustered_data= np.concatenate(
			(snames, objids, self.labels, self.probs),
			axis=1
		)

		head= "# sname id clustid clustprob"
		Utils.write_ascii(clustered_data, self.outfile, head)	

		#================================
		#==   PLOT
		#================================
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
	def run_clustering(self, datafile, modelfile=''):
		""" Run clustering using input dataset """

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
			logger.info("Loading the clustering model from file %s ..." % modelfile)
			try:
				self.clusterer, self.prediction_extra_data = pickle.load((open(modelfile, 'rb')))
			except Exception as e:
				logger.error("Failed to load model from file %s!" % (modelfile))
				return -1

		else:
			logger.info("Creating the clustering model ...")
			self.clusterer= self.__build_model()
			self.predition_extra_data= None

		#================================
		#==   CLUSTER DATA
		#================================
		if self.__cluster()<0:
			logger.error("Failed to cluster data!")
			return -1

		return 0


	def run_clustering(self, data, class_ids=[], snames=[], modelfile=''):
		""" Run clustering using input dataset """

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
			logger.info("Loading the clustering model from file %s ..." % modelfile)
			try:
				self.clusterer, self.prediction_extra_data = pickle.load((open(modelfile, 'rb')))
			except Exception as e:
				logger.error("Failed to load model from file %s!" % (modelfile))
				return -1

		else:
			logger.info("Creating the clustering model ...")
			self.clusterer= self.__build_model()
			self.predition_extra_data= None

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
		
		self.nclusters= self.labels.max()
		logger.info("#%d clusters found ..." % (self.nclusters))
	

		#================================
		#==   SAVE CLUSTERED DATA
		#================================
		logger.info("Saving clustered data to file ...")
		N= data_all.shape[0]
		print("Clustered data size is N=",N)

		snames= np.array(source_names_all).reshape(N,1)
		objids= np.array(data_ids_all).reshape(N,1)
		objlabels= np.array(data_labels_all).reshape(N,1)

		clustered_data= np.concatenate(
			(snames, objids, objlabels, self.labels, self.probs, self.outlier_scores),
			axis=1
		)

		head= "# sname id clust_id clust_prob outlier_score"
		Utils.write_ascii(clustered_data, self.outfile, head)	

		#================================
		#==   SAVE MODEL
		#================================
		# - Save model to file
		if self.dump_model:
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
		logger.info("Plotting results ...")
		self.__plot(
			self.clusterer, 
			data_all, source_names_all, data_labels_all, 
			self.outfile_plot
		)

		
		return 0
	
	#####################################
	##     PLOTTING
	#####################################
	def __plot(self, clusterer, data, snames, class_labels, outfile):
		""" Plot clusters """

		# - Set cluster colors
		palette = sns.color_palette('deep', 8)
		cluster_colors = [palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusterer.labels_]
		cluster_member_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)]
		
		# - Set variable names
		data_shape= data.shape
		ndim= data_shape[1]
		N= data_shape[0]
		varnames_counter= list(range(1,ndim+1))
		varnames= '{}{}'.format('z',' z'.join(str(item) for item in varnames_counter))

		# - Pair-wise Scatter Plots
		###cols = ['density', 'residual sugar', 'total sulfur dioxide', 'fixed acidity']
		###pp = sns.pairplot(wines[cols], size=1.8, aspect=1.8, plot_kws=dict(edgecolor="k", linewidth=0.5), diag_kind="kde", diag_kws=dict(shade=True))
		#pp = sns.pairplot(data, size=1.8, aspect=1.8, plot_kws=dict(edgecolor="k", s=50, linewidth=0, c=cluster_member_colors, alpha=0.25), diag_kind="kde", diag_kws=dict(shade=True))
		#fig = pp.fig 
		#fig.subplots_adjust(top=0.93, wspace=0.3)
		#t = fig.suptitle('Clustering Plots', fontsize=14)

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

			#plt.scatter(data[i,0], data[i,1], s=50, linewidth=0, color=color, marker=marker, alpha=0.25)
			plt.scatter(data[i,0], data[i,1], s=50, linewidth=0, color=cluster_color, marker=marker, alpha=0.25)

		
		plt.xlabel("z0")
		plt.ylabel("z1")
		plt.savefig(outfile)
		#plt.show()


	def __plot_predict(self, clusterer, data_test, labels_test, snames_test, class_labels_test, prediction_data, prediction_extra_data, outfile):
		""" Plot clusters """

		# - Retrieve prediction data
		data= prediction_data.data
		snames= prediction_extra_data.snames
		class_labels= prediction_extra_data.class_labels

		# - Set cluster colors
		palette = sns.color_palette('deep', 8)
		cluster_colors = [palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusterer.labels_]
		cluster_member_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)]
		
		# - Set variable names
		data_shape= data.shape
		ndim= data_shape[1]
		N= data_shape[0]
		varnames_counter= list(range(1,ndim+1))
		varnames= '{}{}'.format('z',' z'.join(str(item) for item in varnames_counter))

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

			#plt.scatter(data[i,0], data[i,1], s=50, linewidth=0, color=color, marker=marker, alpha=0.25)
			plt.scatter(data[i,0], data[i,1], s=50, linewidth=0, color=cluster_color, marker=marker, alpha=0.25)

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

			plt.scatter(data_test[i,0], data_test[i,1], s=80, linewidths=1, edgecolors='k', c=cluster_color, marker=marker)		

		plt.xlabel("z0")
		plt.ylabel("z1")
		plt.savefig(outfile)
		#plt.show()
	

