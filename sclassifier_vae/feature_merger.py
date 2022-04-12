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
import collections
from collections import Counter
import csv
import io

## ASTRO MODULES
from astropy.io import ascii

## SCLASSIFIER MODULES
from .utils import Utils

##############################
##     GLOBAL VARS
##############################
from sclassifier_vae import logger


##############################
##     FeatMerger CLASS
##############################
class FeatMerger(object):
	""" Class to store feature merger """
	
	def __init__(self):
		""" Return a FeatMerger object """

		self.par_dict_list= []
		self.save_csv= False

	#===========================
	#==   MERGE FEATURE DATA
	#===========================
	def __merge_data(self, dlist):
		""" Merge feature data """

		# - Check input list
		if not dlist:
			logger.error("Empty data dict list given!")
			return -1

		for i in range(len(dlist)):
			d= dlist[i]
			if not d:
				logger.error("Data dict %d is empty!" % (i+1))
				return -1

		# - Compute number of vars
		nvars_tot= 0
		for d in dlist:
			print("d")
			print(d)
			nentries= len(d.keys())
			firstitem= next(iter(d.items()))
			nvars= len(firstitem[1].keys()) - 2
			nvars_tot+= nvars
			logger.info("Data dict has #%d entries (#%d vars) ..." % (nentries, nvars))

		logger.info("Merged set is expected to have %d vars ..." % (nvars_tot))

		# - Merge features
		logger.info("Merging feature data for input data dict ...")

		dmerged= collections.OrderedDict()

		for d in dlist:
			for key, value in d.items():
				if key not in dmerged:
					dmerged[key]= collections.OrderedDict({})
				dmerged[key].update(value)
				dmerged[key].move_to_end("id")

		# - Remove rows with less number of entries
		logger.info("Removing rows with number of vars !=%d ..." % (nvars_tot))

		self.par_dict_list= []
		for key, value in dmerged.items():
			nvars= len(value.keys())-2
			if nvars!=nvars_tot:
				logger.info("Removing entry (%s) as number of vars (%d) is !=%d ..." % (key, nvars, nvars_tot))
				#del dmerged[key]
				continue
			self.par_dict_list.append(value)

		return 0

	#===========================
	#==   READ FEATURE DATA
	#===========================
	def __read_and_merge_data(self, inputfiles, selcolids=[], allow_novars=False):
		""" Read and merge feature data """
	
		# - Check selcolids has format [[selcol_1],[selcol_2]]
		if selcolids:
			if len(selcolids)!=len(inputfiles):
				logger.error("Given selcolid length (%d) must be equal to inputfile list length (%d)!" % (len(selcolids),len(inputfiles)))
				return -1

		# - Read features
		dlist= []
		nvars_tot= 0

		for i in range(len(inputfiles)):
			inputfile= inputfiles[i]
			colprefix= "featset" + str(i+1) + "_"

			if selcolids:
				selcols_i= selcolids[i]
				if selcols_i:
					d= Utils.read_sel_feature_data_dict(inputfile, selcols_i, colprefix=colprefix)
				else:
					logger.error("Empty selcols for file %s given!" % (inputfile))
					return -1	
			else:
				d= Utils.read_feature_data_dict(inputfile, colprefix=colprefix, allow_novars=allow_novars)
			if not d or d is None:
				logger.error("Failed to read data from file %s!" % (inputfile))
				return -1

			nentries= len(d.keys())
			firstitem= next(iter(d.items()))
			nvars= len(firstitem[1].keys()) - 2
			nvars_tot+= nvars
			logger.info("Data file %s has #%d entries (#%d vars) ..." % (inputfile, nentries, nvars))

			dlist.append(d)
			
		logger.info("Merged set is expected to have %d vars ..." % (nvars_tot))

		# - Merge features
		logger.info("Merging feature data for all input files ...")

		dmerged= collections.OrderedDict()

		for d in dlist:
			for key, value in d.items():
				if key not in dmerged:
					dmerged[key]= collections.OrderedDict({})
				dmerged[key].update(value)
				dmerged[key].move_to_end("id")

		# - Remove rows with less number of entries
		logger.info("Removing rows with number of vars !=%d ..." % (nvars_tot))

		self.par_dict_list= []
		for key, value in dmerged.items():
			nvars= len(value.keys())-2
			if nvars!=nvars_tot:
				logger.info("Removing entry (%s) as number of vars (%d) is !=%d ..." % (key, nvars, nvars_tot))
				#del dmerged[key]
				continue
			self.par_dict_list.append(value)

		return 0

	#===========================
	#==   SAVE
	#===========================
	def __save(self, outfile):
		""" Save merged data """
	
		# - Write CSV
		outfile_noext= os.path.splitext(outfile)[0]
		outfile_csv= outfile_noext + '.csv'
		logger.info("Saving merged feature data to file %s ..." % (outfile_csv))

		parnames = self.par_dict_list[0].keys()
		#print("parnames")
		#print(parnames)
		
		with open(outfile_csv, 'w') as fp:
			fp.write("# ")
			dict_writer = csv.DictWriter(fp, parnames)
			dict_writer.writeheader()
			dict_writer.writerows(self.par_dict_list)

		# - Read CSV
		logger.info("Reading CSV written table %s ..." % (outfile_csv))
		row_start= 0
		table= ascii.read(outfile_csv, data_start=row_start)
		colnames= table.colnames
		colnames_mod= np.copy(colnames)
		colnames_mod[0]= '# ' + colnames[0]
		
		# - Write ASCII (replace quotes that astropy puts in colnames after #)
		logger.info("Writing ascii table to file %s ..." % (outfile))
		buf= io.StringIO()
		ascii.write(table, buf, format="basic", names=colnames_mod, overwrite=True)
		with open(outfile, 'w') as f:
			f.write(buf.getvalue().replace('"',""))

		# - Remove CSV file
		if not self.save_csv:
			logger.info("Removing tmp file %s ..." % (outfile_csv))
			if os.path.exists(outfile_csv):
				os.remove(outfile_csv)

	#===========================
	#==   RUN
	#===========================
	def run(self, inputfiles, outfile='featdata_merged.dat', selcolids=[], allow_novars=False):
		""" Run feature merger """

		# - Read feature data and merge 
		logger.info("Reading and merging input feature files ...")
		if self.__read_and_merge_data(inputfiles, selcolids, allow_novars)<0:
			logger.error("Failed to read and merge data!")
			return -1	

		# - Save data
		logger.info("Saving merged data to file %s ..." % (outfile))
		self.__save(outfile)
		
		return 0

	def run_from_dictlist(self, dlist, outfile='featdata_merged.dat'):
		""" Run feature merger """

		# - Read feature data and merge 
		logger.info("Merging input feature data dicts ...")
		if self.__merge_data(dlist)<0:
			logger.error("Failed to merge data!")
			return -1	

		# - Save data
		logger.info("Saving merged data to file %s ..." % (outfile))
		self.__save(outfile)
		
		return 0

