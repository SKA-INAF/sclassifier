#!/usr/bin/env python

##################################################
###          MODULE IMPORT
##################################################
## STANDARD MODULES
import os
import sys
import string
import logging
import numpy as np
from distutils.version import LooseVersion
import shutil
import subprocess
import time
import signal
from threading import Thread
import datetime
import random
import math
import logging
import re
import glob
import json
import collections
from collections import OrderedDict
from collections import defaultdict

## ASTRO MODULES
from astropy.io import fits
from astropy.wcs import WCS
from astropy.io import ascii
from astropy.table import Column
import regions

# Suppress `Invalid 'BLANK' keyword in header.` warnings
import warnings
from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter('ignore', category=VerifyWarning)

## MONTAGE MODULES
from montage_wrapper.commands import mImgtbl

## IMG PROCESSING MODULES
import skimage.color
import skimage.io
import skimage.transform
##from mahotas.features import zernike

## SCUTOUT MODULES
import scutout
from scutout.config import Config

## GRAPHICS MODULES
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


###########################
##   GLOBAL DEFINITIONS
###########################
g_class_labels= ["UNKNOWN","PN","HII","PULSAR","YSO","STAR","GALAXY","QSO"]
g_class_label_id_map= {
	"UNKNOWN": 0,
	"STAR": 1,
	"GALAXY": 2,
	"PN": 3,
	"HII": 6,
	"PULSAR": 23,
	"YSO": 24,
	"QSO": 6000
}

###########################
##     CLASS DEFINITIONS
###########################
class Utils(object):
	""" Class collecting utility methods

			Attributes:
				None
	"""

	def __init__(self):
		""" Return a Utils object """
		#self.logger = logging.getLogger(__name__)
		#_logger = logging.getLogger(__name__)

	@classmethod
	def has_patterns_in_string(cls,s,patterns):
		""" Return true if patterns are found in string """
		if not patterns:		
			return False

		found= False
		for pattern in patterns:
			found= pattern in s
			if found:
				break

		return found

	@classmethod
	def mkdir(cls, path, delete_if_exists=False):
		""" Create a directory """
		try:
			if delete_if_exists and os.path.isdir(path):
				shutil.rmtree(path)
			os.makedirs(path)
		except OSError as exc:
			if exc.errno != errno.EEXIST:
				logger.error('Failed to create directory ' + path + '!')
				return -1

		return 0

	@classmethod
	def write_ascii(cls,data,filename,header=''):
		""" Write data to ascii file """

		# - Skip if data is empty
		if data.size<=0:
			#cls._logger.warn("Empty data given, no file will be written!")
			logger.warn("Empty data given, no file will be written!")
			return

		# - Open file and write header
		fout = open(filename, 'wt')
		if header:
			fout.write(header)
			fout.write('\n')	
			fout.flush()	
		
		# - Write data to file
		nrows= data.shape[0]
		ncols= data.shape[1]
		for i in range(nrows):
			fields= '  '.join(map(str, data[i,:]))
			fout.write(fields)
			fout.write('\n')	
			fout.flush()	

		fout.close()

	@classmethod
	def read_ascii(cls,filename,skip_patterns=[]):
		""" Read an ascii file line by line """
	
		try:
			f = open(filename, 'r')
		except IOError:
			errmsg= 'Could not read file: ' + filename
			#cls._logger.error(errmsg)
			logger.error(errmsg)
			raise IOError(errmsg)

		fields= []
		for line in f:
			line = line.strip()
			line_fields = line.split()
			if not line_fields:
				continue

			# Skip pattern
			skipline= cls.has_patterns_in_string(line_fields[0],skip_patterns)
			if skipline:
				continue 		

			fields.append(line_fields)

		f.close()	

		return fields

	@classmethod
	def read_ascii_table(cls,filename,row_start=0,delimiter='|'):
		""" Read an ascii table file line by line """

		table= ascii.read(filename,data_start=row_start, delimiter=delimiter)
		return table

	@classmethod
	def read_feature_data(cls, filename):
		""" Read data table. Format: sname data classid """	

		# - Read table
		row_start= 0
		table= ascii.read(filename, data_start=row_start)
		colnames= table.colnames
		print(colnames)

		ndim= len(colnames)
		nvars= ndim-2
		if nvars<=0:
			logger.error("Too few cols present in file (ndim=%d)!" % (ndim))
			return ()

		# - Read data columns
		snames= table[colnames[0]].data.tolist()
		classids= table[colnames[ndim-1]].data.tolist()
		x= np.lib.recfunctions.structured_to_unstructured(table.as_array())
		data= x[:,1:1+nvars].astype(np.float32)

		return (data, snames, classids)

	@classmethod
	def read_feature_data_dict(cls, filename, colprefix="", allow_novars=False):
		""" Read data table and return dict. Format: sname data classid """	

		# - Read table
		row_start= 0
		table= ascii.read(filename, data_start=row_start)
		colnames= table.colnames
		print(colnames)

		ndim= len(colnames)
		nvars= ndim-2
		if nvars<=0:
			if allow_novars:
				logger.warn("No var columns present in file (ndim=%d) ..." % (ndim))
			else:
				logger.error("Too few cols present in file (ndim=%d)!" % (ndim))
				return ()

		# - Check if prefix has to be given to vars
		colnames_mod= colnames
		if colprefix!="":
			colnames_mod= [colprefix + item for item in colnames]

		# - Iterate over table and create dict
		d= OrderedDict()

		for row in table:
			sname= row[0]
			classid= row[ndim-1]
			if sname in d:
				logger.warn("Source %s is already present in data dict, overwriting it ..." % (sname))
			d[sname]= OrderedDict()
			d[sname][colnames[0]]= sname
			if nvars>0:
				for	col in range(1, nvars+1):
					colname= colnames_mod[col]
					var= row[col]
					d[sname][colname]= var
			d[sname][colnames[ndim-1]]= classid

		return d

	@classmethod
	def read_sel_feature_data_dict(cls, filename, selcolids, colprefix=""):
		""" Read data table and return dict. Format: sname data classid """	

		# - Read table
		row_start= 0
		table= ascii.read(filename, data_start=row_start)
		colnames= table.colnames
		print(colnames)

		ndim= len(colnames)
		nvars= ndim-2
		nvars_sel= len(selcolids)

		# - Check vars
		if nvars<=0:
			logger.error("Too few cols present in file (ndim=%d)!" % (ndim))
			return ()
		if nvars_sel>nvars:
			logger.error("Number of column vars to be selected (%d) exceeds available columns (%d)!" % (nvars_sel, nvars))
			return ()
		for colid in selcolids:
			if colid<0 or colid>=nvars:
				logger.error("Given sel column id (%d) exceed range of available columns [0,%d]!" % (colid, nvars))
				return ()

		# - Check if prefix has to be given to vars
		#colnames_mod= []
		#for i in range(nvars_sel):
		#	colid= selcolids[
		#	colname= colnames[1+i]
		#	if colprefix=="":
		#		colname_mod= colprefix + colname 
		#	else:
		#		colname_mod= colname
		#	colnames_mod.append(colname)

		#print("colnames_mod")
		#print(colnames_mod)

		# - Iterate over table and create dict
		d= OrderedDict()

		for row in table:
			sname= row[0]
			classid= row[ndim-1]
			if sname in d:
				logger.warn("Source %s is already present in data dict, overwriting it ..." % (sname))
			d[sname]= OrderedDict()
			d[sname][colnames[0]]= sname
			
			for i in range(nvars_sel):
				col= selcolids[i] + 1 # NB: selcolid=0 is the first feature not sname
				#colname= colnames_mod[i]
				colname= colprefix+colnames[col]
				var= row[col]
				d[sname][colname]= var
			d[sname][colnames[ndim-1]]= classid

		return d


	@classmethod
	def write_fits(cls,data,filename):
		""" Read data to FITS image """

		hdu= fits.PrimaryHDU(data)
		hdul= fits.HDUList([hdu])
		hdul.writeto(filename,overwrite=True)

	@classmethod
	def read_fits(cls, filename):
		""" Read FITS image and return data """

		# - Open file
		try:
			hdu= fits.open(filename, memmap=False)
		except Exception as ex:
			errmsg= 'Cannot read image file: ' + filename
			#cls._logger.error(errmsg)
			logger.error(errmsg)
			raise IOError(errmsg)

		# - Read data
		data= hdu[0].data
		data_size= np.shape(data)
		nchan= len(data.shape)
		if nchan==4:
			output_data= data[0,0,:,:]
		elif nchan==2:
			output_data= data	
		else:
			errmsg= 'Invalid/unsupported number of channels found in file ' + filename + ' (nchan=' + str(nchan) + ')!'
			#cls._logger.error(errmsg)
			logger.error(errmsg)
			hdu.close()
			raise IOError(errmsg)

		# - Read metadata
		header= hdu[0].header

		# - Get WCS
		wcs = WCS(header)
		if wcs is None:
			logger.warn("No WCS in input image!")
	
		# - Close file
		hdu.close()

		#return output_data, header
		return output_data, header, wcs


	#===========================
	#==   MAKE IMG METADATA
	#===========================
	@classmethod
	def write_montage_fits_metadata(cls, inputfile, metadata_file="metadata.tbl", jobdir=""):
		""" Generate Montage metadata for input image path """

		# - Set output dir
		if jobdir=="":
			jobdir= os.getcwd()

		inputfile_base= os.path.basename(inputfile)
		inputfile_dir= os.path.dirname(inputfile)

		# - Write fieldlist file
		fieldlist_file= os.path.join(jobdir, "fieldlist.txt")
		logger.info("Writing Montage fieldlist file %s ..." % (fieldlist_file))	
		fout = open(fieldlist_file, 'wt')
		fout.write("BUNIT char 15")
		fout.flush()
		fout.close()

		# - Write imglist file		
		imglist_file= os.path.join(jobdir, "imglist.txt")
		logger.info("Writing Montage imglist file %s ..." % (imglist_file))	
		fout = open(imglist_file, 'wt')
		fout.write("|                            fname|\n")
		fout.write("|                             char|\n")
		fout.write(inputfile_base)
		fout.flush()
		fout.close()
			
		# - Write metadata file
		status_file= os.path.join(jobdir,"imgtbl_status.txt")
		logger.info("Writing Montage metadata file %s ..." % (metadata_file))	
		try:
			mImgtbl(
				directory= inputfile_dir,
				images_table=metadata_file,
				corners=True,
				status_file=status_file,
				fieldlist=fieldlist_file,
				img_list=imglist_file
			)
	
			# - Parse status from file
			# ...
			# ...

			# - Update metadata (Montage put fname without absolute path if img_list option is given!)
			t= ascii.read(metadata_file)
			
			if t["fname"]!=inputfile:
				coldata= [inputfile]
				col= Column(data=coldata, name='fname')
				t["fname"]= col				
				ascii.write(t, metadata_file, format="ipac", overwrite=True)

		except Exception as e:
			logger.error("Exception occurred when executing mImgTbl command (err=%s)!" % (str(e)))
			return -1
					
		return 0
		
	
	@classmethod
	def crop_img(cls,data,x0,y0,dx,dy):
		""" Extract sub image of size (dx,dy) around pixel (x0,y0) """

		#- Extract crop data
		xmin= int(x0-dx/2)
		xmax= int(x0+dx/2)
		ymin= int(y0-dy/2)
		ymax= int(y0+dy/2)		
		crop_data= data[ymin:ymax+1,xmin:xmax+1]
	
		#- Replace NAN with zeros and inf with large numbers
		np.nan_to_num(crop_data,False)

		return crop_data

	@classmethod
	def resize_img(cls, image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
		"""A wrapper for Scikit-Image resize().

		Scikit-Image generates warnings on every call to resize() if it doesn't
		receive the right parameters. The right parameters depend on the version
		of skimage. This solves the problem by using different parameters per
		version. And it provides a central place to control resizing defaults.
		"""
		if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
			# New in 0.14: anti_aliasing. Default it to False for backward
			# compatibility with skimage 0.13.
			return skimage.transform.resize(
				image, output_shape,
				order=order, mode=mode, cval=cval, clip=clip,
				preserve_range=preserve_range, anti_aliasing=anti_aliasing,
				anti_aliasing_sigma=anti_aliasing_sigma)
		else:
			return skimage.transform.resize(
				image, output_shape,
				order=order, mode=mode, cval=cval, clip=clip,
				preserve_range=preserve_range)

	@classmethod
	def draw_histo(cls,data,nbins=100,logscale=False):
		""" Draw input array histogram """

		# - Do nothing if data is empty
		if data.ndim<=0:
			return

		# - Flatten array 
		x= data.flatten()

		# - Set histogram from data
		hist, bins = np.histogram(x, bins=nbins)
		width = 0.7 * (bins[1] - bins[0])
		center = (bins[:-1] + bins[1:]) / 2

		# - Draw plots
		plt.bar(center, hist, align='center', width=width)
		if logscale:
			plt.yscale('log')

		plt.show()


	@classmethod
	def weighted_mean(cls, x, wts):
		""" Calculates the weighted mean of a data sample """
		return np.average(x, weights=wts)

	@classmethod
	def weighted_variance(cls, x, wts):
		""" Calculates the weighted variance of a data sample """
		return np.average((x - cls.weighted_mean(x, wts))**2, weights=wts)
	
	@classmethod
	def weighted_std(cls, x, wts):
		""" Calculates the weighted standard deviation of a data sample """
		return np.sqrt(cls.weighted_variance(x, wts))

	@classmethod
	def weighted_skew(cls, x, wts):
		""" Calculates the weighted skewness of a data sample """
		return (np.average((x - cls.weighted_mean(x, wts))**3, weights=wts) /
			cls.weighted_variance(x, wts)**(1.5))

	@classmethod
	def weighted_kurtosis(cls, x, wts):
		""" Calculates the weighted skewness """
		return (np.average((x - cls.weighted_mean(x, wts))**4, weights=wts) /
			cls.weighted_variance(x, wts)**(2))

	
	#=========================
	#==   READ DS9 REGION
	#=========================
	@classmethod
	def find_duplicates(cls, seq):
		""" Return dict with duplicated item in list"""

		tally = defaultdict(list)
		for i,item in enumerate(seq):
			tally[item].append(i)

  	#return ({key:locs} for key,locs in tally.items() if len(locs)>0)
		return (locs for key,locs in tally.items() if len(locs)>0)

	@classmethod
	def read_regions(cls, regionfiles):
		""" Read input regions """

		# - Read regions
		regs= []
		snames= []
		slabels= []

		for regionfile in regionfiles:
			region_list= regions.read_ds9(regionfile)
			logger.info("#%d regions found in file %s ..." % (len(region_list), regionfile))
			regs.extend(region_list)
			
		logger.info("#%d source regions read ..." % (len(regs)))

		# - Check if region are PolygonSkyRegion and get names
		for i in range(len(regs)):
			region= regs[i]

			# - Check region type
			is_polygon_sky= isinstance(region, regions.PolygonSkyRegion)
			if not is_polygon_sky:
				logger.error("Region no. %d is not a PolygonSkyRegion (check input region)!" % (i+1))
				return None

			# - Set source name
			sname= "S" + str(i+1)
			if 'text' in region.meta:
				sname= region.meta['text']
			snames.append(sname)

			# - Set source class label
			label= "UNKNOWN"
			if 'tag' in region.meta:
				tags= region.meta['tag']
				for tag in tags:
					tag_value= re.sub('[{}]','',tag)
					if tag_value in g_class_labels:
						label= tag_value
						break

			slabels.append(label)


		# - Create compound regions from union of regions with same name
		logger.info("Creating merged multi-island regions ...")
		source_indices= sorted(cls.find_duplicates(snames))
		scounter= 0
		regions_merged= []
		snames_merged= []
		slabels_merged= []

		for sindex_list in source_indices:
			if not sindex_list:
				continue
			nsources= len(sindex_list)

			if nsources==1:
				sindex= sindex_list[0]
				regions_merged.append(regs[sindex])
				snames_merged.append(snames[sindex])
				slabels_merged.append(slabels[sindex])
				
			else:
				mergedRegion= copy.deepcopy(regs[sindex_list[0]])
				
				for i in range(1,len(sindex_list)):
					tmpRegion= mergedRegion.union(regs[sindex_list[i]])
					mergedRegion= tmpRegion

				regions_merged.append(mergedRegion)

		regs= regions_merged
		snames= snames_merged
		slabels= slabels_merged

		logger.info("#%d source regions left after merging multi-islands ..." % (len(regs)))

		return regs, snames, slabels

	#===========================
	#==   SELECT REGIONS
	#===========================
	@classmethod
	def select_regions(cls, regs, seltags):
		""" Select regions by tags """
	
		regs_sel= []
		snames_sel= []
		slabels_sel= []
		region_counter= 0
	
		for r in regs:
			# - Set source name
			sname= "S" + str(region_counter+1)
			if 'text' in r.meta:
				sname= r.meta['text']

			# - Set labels
			if 'tag' not in r.meta:
				continue
			tags= r.meta['tag']

		
			label= "UNKNOWN"
			for tag in tags:
				tag_value= re.sub('[{}]','',tag)
				if tag_value in g_class_labels:
					label= tag_value
					break

			has_all_tags= True

			for seltag in seltags:	
				has_tag= False
		
				for tag in tags:
					tag_value= re.sub('[{}]','',tag)
					if tag_value==seltag:
						has_tag= True
						break

				if not has_tag:	
					has_all_tags= False
					break

			if has_all_tags:
				regs_sel.append(r)
				snames_sel.append(sname)
				slabels_sel.append(label)
				region_counter+= 1


		logger.info("#%d region selected by tags..." % (len(regs_sel)))

		return regs_sel, snames_sel, slabels_sel


	#=================================
	#==   FIND REGION BBOX
	#=================================
	@classmethod
	def compute_region_centroid(cls, vertices):
		""" Compute bbox from region vertices """

		ra_list= [item.ra.value for item in vertices]
		dec_list= [item.dec.value for item in vertices]
		ra_min= np.min(ra_list)
		ra_max= np.max(ra_list)
		dec_min= np.min(dec_list)
		dec_max= np.max(dec_list)
		dra= ra_max-ra_min
		ddec= dec_max-dec_min
		ra_c= ra_min + dra/2.
		dec_c= dec_min + ddec/2.
		radius= np.sqrt(dra**2 + ddec**2)/2. # in deg
		radius_arcsec= radius*3600
	
		return ra_c, dec_c, radius_arcsec

	@classmethod
	def compute_region_info(cls, regs):
		""" Find region bbox info """

		centroids= []
		radii= []

		for r in regs:
			vertices= r.vertices
			ra, dec, radius= cls.compute_region_centroid(vertices)
			centroids.append((ra,dec))
			radii.append(radius)

		return centroids, radii


	#===========================
	#==   MAKE DATA LISTS
	#===========================
	@classmethod
	def clear_cutout_dirs(cls, datadir, datadir_mask, nsurveys):
		""" Remove cutout dirs with less than desired survey files """

		# - List all directories with masked cutouts
		sdirs_mask= []
		for item in os.listdir(datadir_mask):
			fullpath= os.path.join(datadir_mask, item)
			if os.path.isdir(fullpath):
				sdirs_mask.append(fullpath)

		print("sdirs_mask")
		print(sdirs_mask)

		# - Delete both cutout and masked cutout dirs without enough files
		for sdir_mask in sdirs_mask:
			files= glob.glob(os.path.join(sdir_mask,"*.fits"))
			nfiles= len(files)
			logger.info("#%d files in masked cutout dir %s ..." % (nfiles, sdir_mask))

			if nfiles==nsurveys: # nothing to be done if we have all files per survey
				continue

			if os.path.exists(sdir_mask):
				logger.info("Removing masked cutout dir %s ..." % (sdir_mask))
				shutil.rmtree(sdir_mask)

				sdir_base= os.path.basename(os.path.normpath(sdir_mask))
				sdir= os.path.join(datadir, sdir_base)
				if os.path.exists(sdir):
					logger.info("Removing cutout dir %s ..." % (sdir))
					shutil.rmtree(sdir)

		# - Do the same on cutout dirs (e.g. maybe masked cutouts are missed due to a fail on masking routine)
		sdirs= []
		for item in os.listdir(datadir):
			fullpath= os.path.join(datadir, item)
			if os.path.isdir(fullpath):
				sdirs.append(fullpath)

		for sdir in sdirs:
			files= glob.glob(os.path.join(sdir,"*.fits"))
			nfiles= len(files)
			if nfiles==nsurveys: # nothing to be done if we have all files per survey
				continue
		
			if os.path.exists(sdir):
				logger.info("Removing cutout dir %s ..." % (sdir))
				shutil.rmtree(sdir)

		return 0

	@classmethod
	def file_sorter(cls, item):
		""" Custom sorter of filename according to rank """
		return item[1]

	@classmethod
	def get_file_rank(cls, filename):
		""" Return file order rank from filename """

		# - Check if radio
		is_radio= False
		score= 0
		if "meerkat_gps" in filename:
			is_radio= True
		if "askap" in filename:
			is_radio= True
		if "first" in filename:
			is_radio= True
		if "custom_survey" in filename:
			is_radio= True
		if is_radio:
			score= 0

		# - Check if 12 um
		if "wise_12" in filename:
			score= 1

		# - Check if 22 um
		if "wise_22" in filename:
			score= 2

		# - Check if 3.4 um
		if "wise_3_4" in filename:
			score= 3

		# - Check if 4.6 um
		if "wise_4_6" in filename:
			score= 4

		# - Check if 8 um
		if "irac_8" in filename:
			score= 5

		# - Check if 70 um
		if "higal_70" in filename:
			score= 6

		return score

	@classmethod
	def make_datalists(cls, datadir, slabelmap, outfile):
		""" Create json datalists for cutouts """

		# - Init data dictionary
		data_dict= {"data": []}
		normalizable_flag= 1
	
		# - Create dir list
		sdirs= []
		dirlist= os.listdir(datadir)
		dirlist.sort()

		for item in dirlist:
			fullpath= os.path.join(datadir, item)
			if os.path.isdir(fullpath):
				sdirs.append(fullpath)

		for sdir in sdirs:
			# - Get all FITS files in dir
			filenames= glob.glob(os.path.join(sdir,"*.fits"))
			nfiles= len(filenames)
			if nfiles<=0:
				continue
		
			# - Sort filename (bash equivalent)
			filenames_sorted= sorted(filenames)
			filenames= filenames_sorted	

			# - Sort filename according to specified ranks
			filenames_ranks = []
			for filename in filenames:
				rank= cls.get_file_rank(filename)
				filenames_ranks.append(rank)

			filenames_tuple= [(filename,rank) for filename,rank in zip(filenames,filenames_ranks)]
			filenames_tuple_sorted= sorted(filenames_tuple, key=cls.file_sorter)
			filenames_sorted= []
			for item in filenames_tuple_sorted:
				filenames_sorted.append(item[0])

			filenames= filenames_sorted

			# - Create normalizable flags
			normalizable= [normalizable_flag]*len(filenames)

			# - Compute source name from directory name
			sname= os.path.basename(os.path.normpath(sdir))

			# - Find class label & id
			class_label= "UNKNOWN"
			if sname not in slabelmap:
				logger.warn("Source %s not present in class label map, setting class label to UNKNOWN..." % (sname))
				class_label= "UNKNOWN"
			else:
				class_label= slabelmap[sname]
		
			class_id= g_class_label_id_map[class_label]
		
			# - Add entry in dictionary
			d= {}
			d["filepaths"]= filenames
			d["normalizable"]= normalizable
			d["sname"]= sname
			d["id"]= class_id
			d["label"]= class_label
			data_dict["data"].append(d)

		# - Save json filelist
		logger.info("Saving json datalist to file %s ..." % (outfile))
		with open(outfile, 'w') as fp:
			json.dump(data_dict, fp)

	#=============================
	#==   READ SCUTOUT CONFIG
	#=============================
	@classmethod
	def make_scutout_config(cls, configfile, surveys, jobdir, add_survey=False, img_metadata=""):
		""" Set scutout config class from config file template """

		# - Read scutout config
		logger.info("Parsing scutout config file %s ..." % (configfile))
		config= Config()

		if config.parse(configfile, add_survey, img_metadata)<0:
			logger.error("Failed to read and parse scutout config %s!" % (configfile))
			return None
		
		# - Set desired surveys and workdir
		config.workdir= jobdir
		config.surveys= []
		#if self.imgfile!="":
		#	config.surveys.append("custom_survey")
		if add_survey:
			config.surveys.append("custom_survey")

		if surveys:
			config.surveys.extend(surveys)

		if config.validate()<0:
			logger.error("Failed to validate scutout config after setting surveys & workdir!")
			return None

		#nsurveys= len(config.surveys)

		return config

		
