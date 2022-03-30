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
import re
import shutil
import glob

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections
from collections import defaultdict

## ASTRO MODULES
from astropy.io import fits
from astropy.wcs import WCS
from astropy.io import ascii
from astropy.table import Column
import regions

## SCUTOUT MODULES
import scutout
from scutout.config import Config

## MONTAGE MODULES
from montage_wrapper.commands import mImgtbl

## PLOT MODULES
import matplotlib.pyplot as plt


## MODULES
from sclassifier_vae import __version__, __date__
from sclassifier_vae import logger
from sclassifier_vae.data_loader import DataLoader
from sclassifier_vae.utils import Utils
from sclassifier_vae.classifier import SClassifier
from sclassifier_vae.cutout_maker import SCutoutMaker


#===========================
#==   IMPORT MPI
#===========================
MASTER=0
try:
	from mpi4py import MPI as MPI
	comm= MPI.COMM_WORLD
	nproc= comm.Get_size()
	procId= comm.Get_rank()
except Exception as e:
	logger.warn("Failed to import mpi4py module (err=%s), cannot run in parallel ..." % str(e))
	MPI= None
	comm= None
	nproc= 1
	procId= 0

#### GET SCRIPT ARGS ####
def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

###########################
##     ARGS
###########################
def get_args():
	"""This function parses and return arguments passed in"""
	parser = argparse.ArgumentParser(description="Parse args.")

	# - Input image options
	parser.add_argument('-img','--img', dest='img', required=False, type=str, help='Input 2D radio image filename (.fits)') 
	
	# - Region options
	parser.add_argument('-region','--region', dest='region', required=True, type=str, help='Input DS9 region filename with sources to be classified (.reg)') 
	parser.add_argument('--filter_regions_by_tags', dest='filter_regions_by_tags', action='store_true')	
	parser.set_defaults(filter_regions_by_tags=False)
	parser.add_argument('-tags','--tags', dest='tags', required=False, type=str, help='List of region tags to be used for region selection.') 
	
	# - Source cutout options
	parser.add_argument('-scutout_config','--scutout_config', dest='scutout_config', required=True, type=str, help='scutout configuration filename (.ini)') 
	parser.add_argument('-surveys','--surveys', dest='surveys', required=False, type=str, help='List of surveys to be used for cutouts, separated by comma. First survey is radio.') 
	
	# - Pre-processing options
	parser.add_argument('--normalize', dest='normalize', action='store_true',help='Normalize feature data in range [0,1] before applying models (default=false)')	
	parser.set_defaults(normalize=False)
	parser.add_argument('-scalerfile', '--scalerfile', dest='scalerfile', required=False, type=str, default='', action='store',help='Load and use data transform stored in this file (.sav)')
	
	# - Model options
	parser.add_argument('-modelfile', '--modelfile', dest='modelfile', required=False, type=str, default='', action='store',help='Classifier model filename (.sav)')
	parser.add_argument('--binary_class', dest='binary_class', action='store_true',help='Perform a binary classification {0=EGAL,1=GAL} (default=multiclass)')	
	parser.set_defaults(binary_class=False)
	
	# - Output options
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, type=str, default='classified_data.dat', help='Output filename (.dat) with classified data') 
	parser.add_argument('-jobdir','--jobdir', dest='jobdir', required=False, type=str, default='', help='Job directory. Set to PWD if empty') 
	
	args = parser.parse_args()	

	return args


#===========================
#==   READ REGIONS
#===========================
class_labels= ["UNKNOWN","PN","HII","PULSAR","YSO","STAR","GALAXY","QSO"]
class_label_id_map= {
	"UNKNOWN": 0,
	"STAR": 1,
	"GALAXY": 2,
	"PN": 3,
	"HII": 6,
	"PULSAR": 23,
	"YSO": 24,
	"QSO": 6000
}


def find_duplicates(seq):
	""" Return dict with duplicated item in list"""
	tally = defaultdict(list)
	for i,item in enumerate(seq):
		tally[item].append(i)

  #return ({key:locs} for key,locs in tally.items() if len(locs)>0)
	return (locs for key,locs in tally.items() if len(locs)>0)

def read_regions(regionfiles):
	""" Read input regions """

	# - Read regions
	regs= []
	snames= []
	slabels= []

	for regionfile in regionfiles:
		region_list= regions.read_ds9(regionfile)
		logger.info("[PROC %d] #%d regions found in file %s ..." % (procId, len(region_list), regionfile))
		regs.extend(region_list)
			
	logger.info("[PROC %d] #%d source regions read ..." % (procId, len(regs)))

	# - Check if region are PolygonSkyRegion and get names
	for i in range(len(regs)):
		region= regs[i]

		# - Check region type
		is_polygon_sky= isinstance(region, regions.PolygonSkyRegion)
		if not is_polygon_sky:
			logger.error("[PROC %d] Region no. %d is not a PolygonSkyRegion (check input region)!" % (procId, i+1))
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
				if tag_value in class_labels:
					label= tag_value
					break

		slabels.append(label)


	# - Create compound regions from union of regions with same name
	logger.info("[PROC %d] Creating merged multi-island regions ..." % (procId))
	source_indices= sorted(find_duplicates(snames))
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

	logger.info("[PROC %d] #%d source regions left after merging multi-islands ..." % (procId, len(regs)))

	return regs, snames, slabels


#===========================
#==   SELECT REGIONS
#===========================
def select_regions(regs, seltags):
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
			if tag_value in class_labels:
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


	logger.info("[PROC %d] #%d region selected by tags..." % (procId, len(regs_sel)))

	return regs_sel, snames_sel, slabels_sel
	
#=================================
#==   FIND REGION BBOX
#=================================
def compute_region_centroid(vertices):
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

def compute_region_info(regs):
	""" Find region bbox info """

	centroids= []
	radii= []

	for r in regs:
		vertices= r.vertices
		ra, dec, radius= compute_region_centroid(vertices)
		centroids.append((ra,dec))
		radii.append(radius)

	return centroids, radii

#===========================
#==   READ IMAGE
#===========================
def read_img(inputfile, metadata_file="metadata.tbl", jobdir=""):
	""" Read image """

	# - Set output dir
	if jobdir=="":
		jobdir= os.getcwd()

	# - Read input image
	try:
		hdu= fits.open(inputfile)[0]

	except Exception as e:
		logger.error("[PROC %d] Failed to read image file %s!" % (procId, inputfile))
		return None		
	
	data= hdu.data
	header= hdu.header
	nchan = len(data.shape)
	if nchan == 4:
		data = data[0, 0, :, :]
	
	shape= data.shape	

	wcs = WCS(header)
	if wcs is None:
		logger.warn("[PROC %d] No WCS in input image!" % (procId))
		return None

	#cs= wcs_to_celestial_frame(wcs)
	#cs_name= cs.name
	#iau_name_prefix= 'G'

	#pixSizeX= header['CDELT1']
	#pixSizeY= header['CDELT2']

	# - Generate Montage metadata for this image (PROC 0)
	#   PROC 0 broadcast status to other PROC
	status= -1

	if procId==MASTER:
		# - Write fieldlist file
		#fieldlist_file= "fieldlist.txt"
		fieldlist_file= os.path.join(jobdir, "fieldlist.txt")
		logger.info("[PROC %d] Writing Montage fieldlist file %s ..." % (procId, fieldlist_file))	
		fout = open(fieldlist_file, 'wt')
		fout.write("BUNIT char 15")
		fout.flush()
		fout.close()

		# - Write imglist file
		inputfile_base= os.path.basename(inputfile)
		#imglist_file= "imglist.txt"
		imglist_file= os.path.join(jobdir, "imglist.txt")
		logger.info("[PROC %d] Writing Montage imglist file %s ..." % (procId, imglist_file))	
		fout = open(imglist_file, 'wt')
		fout.write("|                            fname|\n")
		fout.write("|                             char|\n")
		fout.write(inputfile_base)
		fout.flush()
		fout.close()
			
		# - Write metadata file
		status_file= os.path.join(jobdir,"imgtbl_status.txt")
		inputfile_dir= os.path.dirname(inputfile)
		logger.info("[PROC %d] Writing Montage metadata file %s ..." % (procId, metadata_file))	
		try:
			mImgtbl(
				directory= inputfile_dir,
				images_table=metadata_file,
				corners=True,
				status_file=status_file,
				fieldlist=fieldlist_file,
				img_list=imglist_file
			)
	
			status= 0

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

				#fin = open(metadata_file, "rt")
				#data = fin.read()
				#data = data.replace(inputfile_base, inputfile)
				#fin.close()
				
				#fin = open(metadata_file, "wt")
				#fin.write(data)
				#fin.close()

		except Exception as e:
			logger.error("[PROC %d] Exception occurred when executing mImgTbl command (err=%s)!" % (procId, str(e)))
			status= -1
				
	else: # OTHER PROCS
		status= -1
			
	if comm is not None:
		status= comm.bcast(status, root=MASTER)

	if status<0:
		logger.error("[PROC %d] Failed to generate Montage metadata for input image, exit!" % (procId))
		return None

	return data, header, wcs


def clear_cutout_dirs(datadir, datadir_mask, nsurveys):
	""" Remove cutout dirs with less than desired survey files """

	# - List all directories with masked cutouts
	sdirs_mask= []
	for item in os.listdir(datadir_mask):
		if os.path.isdir(os.path.join(datadir_mask, item)):
			sdirs_mask.append(item)

	print("sdirs_mask")
	print(sdirs_mask)

	# - Delete both cutout and masked cutout dirs without enough files
	for sdir_mask in sdirs_mask:
		files= glob.glob(os.path.join(sdir_mask,"*.fits"))
		nfiles= len(files)
		logger.info("[PROC %d] #%d files in masked cutout dir %s ..." % (procId, nfiles, sdir_mask))

		if nfiles==nsurveys: # nothing to be done if we have all files per survey
			continue

		if os.path.exists(sdir_mask):
			logger.info("[PROC %d] Removing masked cutout dir %s ..." % (procId, sdir_mask))
			#shutil.rmtree(sdir_mask)

			sdir_base= os.path.basename(os.path.normpath(sdir_mask))
			sdir= os.path.join(datadir, sdir_base)
			if os.path.exists(sdir):
				logger.info("[PROC %d] Removing cutout dir %s ..." % (procId, sdir))
				#shutil.rmtree(sdir)

	# - Do the same on cutout dirs (e.g. maybe masked cutouts are missed due to a fail on masking routine)
	sdirs= []
	for item in os.listdir(datadir):
		if os.path.isdir(os.path.join(datadir, item)):
			sdirs.append(item)

	for sdir in sdirs:
		files= glob.glob(os.path.join(sdir,"*.fits"))
		nfiles= len(files)
		if nfiles==nsurveys: # nothing to be done if we have all files per survey
			continue
		
		if os.path.exists(sdir):
			logger.info("[PROC %d] Removing cutout dir %s ..." % (procId, sdir))
			#shutil.rmtree(sdir)

	return 0


##############
##   MAIN   ##
##############
def main():
	"""Main function"""

	#===========================
	#==   PARSE ARGS
	#==     (ALL PROCS)
	#===========================
	if procId==MASTER:
		logger.info("[PROC %d] Parsing script args ..." % (procId))
	try:
		args= get_args()
	except Exception as ex:
		logger.error("[PROC %d] Failed to get and parse options (err=%s)" % (procId, str(ex)))
		return 1

	imgfile= args.img
	regionfile= args.region
	configfile= args.scutout_config 

	surveys= []
	if args.surveys!="":
		surveys= [str(x.strip()) for x in args.surveys.split(',')]

	if imgfile=="" and not surveys:
		logger.error("[PROC %d] No image passed, surveys option cannot be empty!" % (procId))
		return 1

	filter_regions_by_tags= args.filter_regions_by_tags
	tags= []
	if args.tags!="":
		tags= [str(x.strip()) for x in args.tags.split(',')]

	jobdir= os.getcwd()
	if args.jobdir!="":
		if not os.path.exists(args.jobdir):
			logger.error("[PROC %d] Given job dir %s does not exist!" % (procId, args.jobdir))
			return 1
		jobdir= args.jobdir

	#==================================
	#==   READ IMAGE DATA   
	#==     (READ - ALL PROC)
	#==     (METADATA GEN - PROC 0) 
	#==================================
	img_metadata= ""
	data= None
	header= None
	wcs= None
	add_survey= False

	if imgfile!="":
		add_survey= True
		##img_metadata= "metadata.tbl"
		img_metadata= os.path.join(jobdir, "metadata.tbl")

		imgfile_fullpath= os.path.abspath(imgfile)

		logger.info("[PROC %d] Reading input image %s ..." % (procId, imgfile))
		ret= read_img(imgfile_fullpath, img_metadata, jobdir)
		if ret is None:
			logger.error("[PROC %d] Failed to read input image %s!" % (procId, imgfile))
			return 1

		data= ret[0]
		header= ret[1]
		wcs= ret[2]


	#=============================
	#==   READ SCUTOUT CONFIG
	#==      (ALL PROCS)
	#=============================
	# - Read scutout config
	logger.info("[PROC %d] Parsing scutout config file %s ..." % (procId, configfile))
	config= Config()

	if config.parse(configfile, add_survey, img_metadata)<0:
		logger.error("[PROC %d] Failed to read and parse scutout config %s!" % (procId, configfile))
		return 1
		
	# - Set desired surveys and workdir
	config.workdir= jobdir
	config.surveys= []
	if imgfile!="":
		config.surveys.append("custom_survey")
	if surveys:
		#config.surveys= surveys
		config.surveys.extend(surveys)
	if config.validate()<0:
		logger.error("[PROC %d] Failed to validate scutout config after setting surveys & workdir!" % (procId))
		return 1

	nsurveys= len(config.surveys)
	
	#===========================
	#==   READ REGIONS
	#==     (ALL PROCS)
	#===========================
	# - Read regions
	logger.info("[PROC %d] Reading DS9 region file %s ..." % (procId, regionfile))
	ret= read_regions([regionfile])
	if ret is None:
		logger.error("[PROC %d] Failed to read regions (check format)!" % (procId))
		return 1
	
	regs= ret[0]
	snames= ret[1]
	slabels= ret[2]

	# - Select region by tag
	regs_sel= regs
	snames_sel= snames
	slabels_sel= slabels
	if filter_regions_by_tags and tags:
		logger.info("[PROC %d] Selecting DS9 region with desired tags ..." % (procId))
		regs_sel, snames_sel, slabels_sel= select_regions(regs, tags)
		
	if not regs_sel:
		logger.warn("[PROC %d] No region left for processing (check input region file)!" % (procId))
		return 1

	sname_label_map= {}
	for i in range(len(snames_sel)):
		sname= snames_sel[i]
		slabel= slabels_sel[i]
		sname_label_map[sname]= slabel

	# - Compute centroids & radius
	centroids, radii= compute_region_info(regs_sel)

	# - Assign sources to each processor
	nsources= len(regs_sel)
	source_indices= list(range(0,nsources))
	source_indices_split= np.array_split(source_indices, nproc)
	source_indices_proc= list(source_indices_split[procId])
	nsources_proc= len(source_indices_proc)
	imin= source_indices_proc[0]
	imax= source_indices_proc[nsources_proc-1]
	
	snames_proc= snames_sel[imin:imax+1]
	slabels_proc= slabels_sel[imin:imax+1]
	regions_proc= regs_sel[imin:imax+1]
	centroids_proc= centroids[imin:imax+1]
	radii_proc= radii[imin:imax+1]
	logger.info("[PROC %d] #%d sources assigned to this processor ..." % (procId, nsources_proc))
	
	print("snames_proc %d" % (procId))
	print(snames_proc)

	#=============================
	#==   MAKE CUTOUTS
	#== (DISTRIBUTE AMONG PROCS)
	#=============================
	# - Prepare dir
	datadir= os.path.join(jobdir, "cutouts")
	datadir_mask= os.path.join(jobdir, "cutouts_masked")

	mkdir_status= -1
		
	if procId==MASTER:
		if not os.path.exists(datadir):
			logger.info("Creating cutout data dir %s ..." % (datadir))
			Utils.mkdir(datadir, delete_if_exists=False)

		if not os.path.exists(datadir_mask):
			logger.info("Creating cutout masked data dir %s ..." % (datadir_mask))
			Utils.mkdir(datadir_mask, delete_if_exists=False)

		mkdir_status= 0

	if comm is not None:
		mkdir_status= comm.bcast(mkdir_status, root=MASTER)

	if mkdir_status<0:
		logger.error("[PROC %d] Failed to create cutout data directory, exit!" % (procId))
		return 1

	# - Make cutouts
	logger.info("[PROC %d] Making cutouts for #%d sources ..." % (procId, nsources_proc))
	cm= SCutoutMaker(config)
	cm.datadir= datadir
	cm.datadir_mask= datadir_mask

	for i in range(nsources_proc):
		sname= snames_proc[i]
		centroid= centroids_proc[i]
		radius= radii_proc[i]
		region= regions_proc[i]

		if cm.make_cutout(centroid, radius, sname, region)<0:
			logger.warn("[PROC %d] Failed to make cutout of source %s, skip to next ..." % (procId, sname))
			continue

	
	#===========================
	#==   CLEAR CUTOUT DIRS
	#==      (ONLY PROC 0)
	#===========================
	# - Remove source cutout directories if having less than desired survey files
	if comm is not None:
		comm.Barrier()

	if procId==MASTER:
		logger.info("[PROC %d] Ensuring that cutout directories contain exactly #%d survey files ..." % (procId, nsurveys))
		clear_cutout_dirs(datadir, datadir_mask, nsurveys)

	#===========================
	#==   MAKE FILELISTS
	#===========================
	# - Create data filelists
	# ...



	#===========================
	#==   CLASSIFY SOURCES
	#===========================
	# ...

	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())
