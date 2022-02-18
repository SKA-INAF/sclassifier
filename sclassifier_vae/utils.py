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
from collections import OrderedDict

## ASTRO MODULES
from astropy.io import fits
from astropy.io import ascii 

## IMG PROCESSING MODULES
import skimage.color
import skimage.io
import skimage.transform
##from mahotas.features import zernike

## GRAPHICS MODULES
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


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

		fout.close();

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
	def read_feature_data_dict(cls, filename, colprefix=""):
		""" Read data table and return dict. Format: sname data classid """	

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

		# - Check if prefix has to be given to vars
		colnames_mod= colnames
		if colprefix!="":
			colnames_mod= [colprefix + item for item in colnames]

		# - Iterate over table and create dict
		d= OrderedDict()

		for row in table:
			sname= row[0]
			classid= row[ndim-1]
			d[sname]= OrderedDict()
			d[sname][colnames[0]]= sname
			for	col in range(1, nvars+1):
				colname= colnames_mod[col]
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
	def read_fits(cls,filename):
		""" Read FITS image and return data """

		# - Open file
		try:
			hdu= fits.open(filename,memmap=False)
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

		# - Close file
		hdu.close()

		return output_data, header

	
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

#	@classmethod
#	def zernike_moments(cls, im, radius, degree=8, cm=None, scale=True, pospix=True):
#		""" Compute Zernike moments """
#		zvalues = []
#		if cm is None:
#			c0,c1 = center_of_mass(im)
#		else:
#			c0,c1 = cm
#
#		Y,X = np.mgrid[:im.shape[0],:im.shape[1]]
#		P = im.ravel()
#
#		def rescale(C, centre):
#			Cn = C.astype(np.double)
#			Cn -= centre
#			Cn /= radius
#			return Cn.ravel()
#		Yn = rescale(Y, c0)
#		Xn = rescale(X, c1)
#
#		Dn = Xn**2
#		Dn += Yn**2
#		np.sqrt(Dn, Dn)
#		np.maximum(Dn, 1e-9, out=Dn)
#		k = (Dn <= 1.)
#		if pospix:
#			k &= (P > 0)
#		else:
#			k &= np.logical_and(P!=0, np.isfinite(P))
#
#		frac_center = np.array(P[k], np.double)
#		frac_center = frac_center.ravel()
#		if scale:
#			frac_center /= frac_center.sum()
#		Yn = Yn[k]
#		Xn = Xn[k]
#		Dn = Dn[k]
#		An = np.empty(Yn.shape, np.complex_)
#		An.real = (Xn/Dn)
#		An.imag = (Yn/Dn)
#
#		Ans = [An**p for p in range(2,degree+2)]
#		Ans.insert(0, An) # An**1
#		Ans.insert(0, np.ones_like(An)) # An**0
#		for n in range(degree+1):
#			for l in range(n+1):
#				if (n-l)%2 == 0:
#					z = zernike.znl(Dn, Ans[l], frac_center, n, l)
#					zvalues.append(abs(z))
#    return np.array(zvalues)

