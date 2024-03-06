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
import functools

## ASTRO MODULES
import warnings
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.simplefilter('ignore', category=VerifyWarning)
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning)

## MONTAGE MODULES
from montage_wrapper.commands import mImgtbl

logger = logging.getLogger(__name__)


###########################
##     CLASS DEFINITIONS
###########################
class MontageUtils(object):
	""" Class collecting utility methods

			Attributes:
				None
	"""

	def __init__(self):
		""" Return a Utils object """

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

