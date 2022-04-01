#! /usr/bin/env python
"""
Setup for sclassifier-vae
"""
import os
import sys
from setuptools import setup


def read(fname):
	"""Read a file"""
	return open(os.path.join(os.path.dirname(__file__), fname)).read()


def get_version():
	""" Get the package version number """
	import sclassifier_vae
	return sclassifier_vae.__version__


#reqs = ['numpy>=1.10',
#        'astropy>=2.0, <3',
#        'keras>=2.0',
#        'tensorflow>=1.13']

PY_MAJOR_VERSION=sys.version_info.major
PY_MINOR_VERSION=sys.version_info.minor
print("PY VERSION: maj=%s, min=%s" % (PY_MAJOR_VERSION,PY_MINOR_VERSION))

reqs= []
reqs.append('numpy>=1.18')
#reqs.append('astropy>=2.0, <3')
reqs.append('astropy>=2.0')


if PY_MAJOR_VERSION<=2:
	print("PYTHON 2 detected")
	reqs.append('future')
	reqs.append('scipy<=1.2.1')
	reqs.append('scikit-learn<=0.20')
	reqs.append('pyparsing>=2.0.1')
	reqs.append('matplotlib<=2.2.4')
else:
	print("PYTHON 3 detected")
	reqs.append('scipy')
	reqs.append('scikit-learn')
	reqs.append('pyparsing')
	reqs.append('matplotlib')

#reqs.append('keras>=2.0')

#reqs.append('six<1.16')
reqs.append('six==1.15.0')
reqs.append('numpy==1.19.5')

#reqs.append('tensorflow>=2.3')
#reqs.append('tensorflow==2.6.0')
reqs.append('tensorflow>=2.6.1')

#reqs.append('imgaug')
reqs.append('imgaug>=0.4.0')

reqs.append('umap-learn')
reqs.append('hdbscan')
reqs.append('seaborn')
##reqs.append('scikit-image<=0.15.0')
reqs.append('scikit-image')
reqs.append('lightgbm')
reqs.append('opencv-python')
reqs.append('mahotas')
reqs.append('imutils')
reqs.append('montage_wrapper')
reqs.append('scutout')
reqs.append('mpi4py')
reqs.append('optuna')


data_dir = 'data'

setup(
	name="sclassifier_vae",
	version=get_version(),
	author="Simone Riggi",
	author_email="simone.riggi@gmail.com",
	description="Unsupervised source classification using variational autoencoders (VAE)",
	license = "GPL3",
	url="https://github.com/SKA-INAF/sclassifier-vae",
	long_description=read('README.md'),
	packages=['sclassifier_vae'],
	install_requires=reqs,
	scripts=['scripts/check_data.py','scripts/run_train.py','scripts/run_predict.py','scripts/run_clustering.py','scripts/reconstruct_data.py','scripts/extract_features.py','scripts/select_features.py','scripts/run_classifier.py','scripts/merge_features.py','scripts/run_classifier_nn.py','scripts/classify_source.py','scripts/find_outliers.py'],
)
