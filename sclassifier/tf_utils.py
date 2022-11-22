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

## TENSORFLOW & KERAS MODULES
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
try:
	from tensorflow.keras.utils import plot_model
except:
	from tensorflow.keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
try:
	from tensorflow.keras.layers.normalization import BatchNormalization
except Exception as e:
	logger.warn("Failed to import BatchNormalization (err=%s), trying in another way ..." % str(e))
	from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.losses import mse, binary_crossentropy

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

from tensorflow.image import convert_image_dtype
from tensorflow.python.ops.image_ops_impl import _fspecial_gauss, _ssim_helper, _verify_compatible_image_shapes
from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution 

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.activations import softmax


###############################################
##     ChanMinMaxNorm LAYER
###############################################
class ChanMinMaxNorm(layers.Layer):
	"""Scale inputs in range.
	The rescaling is applied both during training and inference.
	Input shape:
		Arbitrary.
	Output shape:
		Same as input.
	Arguments:
		norm_min: Float, the data min to the inputs.
		norm_max: Float, the offset to apply to the inputs.
		name: A string, the name of the layer.
	"""

	def __init__(self, norm_min=0., norm_max=1., name=None, **kwargs):
		self.norm_min = norm_min
		self.norm_max = norm_max
		super(ChanMinMaxNorm, self).__init__(name=name, **kwargs)

	def build(self, input_shape):
		super(ChanMinMaxNorm, self).build(input_shape)

	def call(self, inputs, training=False):
		# - Init stuff
		input_shape = tf.shape( inputs )
		norm_min= self.norm_min
		norm_max= self.norm_max
		
		# - Compute input data min & max, excluding NANs & zeros
		cond= tf.logical_and(tf.math.is_finite(inputs), tf.math.not_equal(inputs, 0.))
		
		data_min= tf.reduce_min(tf.where(~cond, tf.ones_like(inputs) * 1.e+99, inputs), axis=(1,2))
		data_max= tf.reduce_max(tf.where(~cond, tf.ones_like(inputs) * -1.e+99, inputs), axis=(1,2))
		
		##### DEBUG ############
		#tf.print("data_min (before norm)", data_min, output_stream=sys.stdout)
		#tf.print("data_max (before norm)", data_max, output_stream=sys.stdout)
		#########################		

		# - Normalize data in range (norm_min, norm_max)
		data_min= tf.expand_dims(tf.expand_dims(data_min, axis=1),axis=1)
		data_max= tf.expand_dims(tf.expand_dims(data_max, axis=1),axis=1)
		data_norm= (inputs-data_min)/(data_max-data_min) * (norm_max-norm_min) + norm_min
		
		# - Set masked values (NANs, zeros) to norm_min
		data_norm= tf.where(~cond, tf.ones_like(data_norm) * norm_min, data_norm)
		
		#######  DEBUG ###########
		data_min= tf.reduce_min(data_norm, axis=(1,2))
		data_max= tf.reduce_max(data_norm, axis=(1,2))
		#data_min= tf.expand_dims(tf.expand_dims(data_min, axis=1), axis=1)
		#data_max= tf.expand_dims(tf.expand_dims(data_max, axis=1), axis=1)
		
		#tf.print("data_min (after norm)", data_min, output_stream=sys.stdout)
		#tf.print("data_max (after norm)", data_max, output_stream=sys.stdout)
		###########################

		return tf.reshape(data_norm, self.compute_output_shape(input_shape))
		
	def compute_output_shape(self, input_shape):
		return input_shape

	def get_config(self):
		config = {
			'norm_min': self.norm_min,
			'norm_max': self.norm_max,
		}
		base_config = super(ChanMinMaxNorm, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))



###############################################
##     ChanMaxScale LAYER
###############################################
class ChanMaxScale(layers.Layer):
	"""Scale inputs to channel maximum.
	The rescaling is applied both during training and inference.
	Input shape:
		Arbitrary.
	Output shape:
		Same as input.
	"""

	def __init__(self, name=None, **kwargs):
		super(ChanMaxScale, self).__init__(name=name, **kwargs)

	def build(self, input_shape):
		super(ChanMaxScale, self).build(input_shape)

	def call(self, inputs, training=False):
		# - Init stuff
		input_shape = tf.shape(inputs)
		
		# - Compute input data min & max, excluding NANs & zeros
		cond= tf.logical_and(tf.math.is_finite(inputs), tf.math.not_equal(inputs, 0.))
		
		data_min= tf.reduce_min(tf.where(~cond, tf.ones_like(inputs) * 1.e+99, inputs), axis=(1,2))
		data_max= tf.reduce_max(tf.where(~cond, tf.ones_like(inputs) * -1.e+99, inputs), axis=(1,2))
		data_min= tf.expand_dims(tf.expand_dims(data_min, axis=1),axis=1)
		data_max= tf.expand_dims(tf.expand_dims(data_max, axis=1),axis=1)
		
		##### DEBUG ############
		#tf.print("data_min (before max scale)", data_min, output_stream=sys.stdout)
		#tf.print("data_max (before max scale)", data_max, output_stream=sys.stdout)
		#########################		

		# - Scale data to max
		inputs_scaled= inputs/data_max
		
		# - Set masked values (NANs, zeros) to norm_min
		norm_min= 0
		inputs_scaled= tf.where(~cond, tf.ones_like(inputs_scaled) * norm_min, inputs_scaled)
		
		#######  DEBUG ###########
		#data_min= tf.reduce_min(inputs_scaled, axis=(1,2))
		#data_max= tf.reduce_max(inputs_scaled, axis=(1,2))
		#data_min= tf.expand_dims(tf.expand_dims(data_min, axis=1), axis=1)
		#data_max= tf.expand_dims(tf.expand_dims(data_max, axis=1), axis=1)
		
		#tf.print("data_min (after max scale)", data_min, output_stream=sys.stdout)
		#tf.print("data_max (after max scale)", data_max, output_stream=sys.stdout)
		###########################

		return tf.reshape(inputs_scaled, self.compute_output_shape(input_shape))
		
	def compute_output_shape(self, input_shape):
		return input_shape





###############################################
##     ChanPosDef LAYER
###############################################
class ChanPosDef(layers.Layer):
	"""Make images positive, as subtract chan minimum
	Input shape:
		Arbitrary.
	Output shape:
		Same as input.
	"""

	def __init__(self, name=None, **kwargs):
		super(ChanPosDef, self).__init__(name=name, **kwargs)

	def build(self, input_shape):
		super(ChanPosDef, self).build(input_shape)

	def call(self, inputs, training=False):
		# - Init stuff
		input_shape = tf.shape(inputs)
		
		# - Compute input data min & max, excluding NANs & zeros
		cond= tf.logical_and(tf.math.is_finite(inputs), tf.math.not_equal(inputs, 0.))
		
		data_min= tf.reduce_min(tf.where(~cond, tf.ones_like(inputs) * 1.e+99, inputs), axis=(1,2))
		#data_max= tf.reduce_max(tf.where(~cond, tf.ones_like(inputs) * -1.e+99, inputs), axis=(1,2))
		data_min= tf.expand_dims(tf.expand_dims(data_min, axis=1),axis=1)
		#data_max= tf.expand_dims(tf.expand_dims(data_max, axis=1),axis=1)

		##### DEBUG ############
		#tf.print("data_min (before posdef)", data_min, output_stream=sys.stdout)
		#tf.print("data_max (before posdef)", data_max, output_stream=sys.stdout)
		#########################		

		# - Subtract data_min on channels with negative data_min
		cond2= tf.math.less(data_min, 0)
		inputs_scaled= tf.where(cond2, inputs - data_min, inputs)

		# - Set masked values (NANs, zeros) to norm_min
		norm_min= 0
		inputs_scaled= tf.where(~cond, tf.ones_like(inputs_scaled) * norm_min, inputs_scaled)
		
		#######  DEBUG ###########
		#data_min= tf.reduce_min(inputs_scaled, axis=(1,2))
		#data_max= tf.reduce_max(inputs_scaled, axis=(1,2))
		#data_min_nozeros= tf.reduce_min(tf.where(~cond, tf.ones_like(inputs_scaled) * 1.e+99, inputs_scaled), axis=(1,2))
		#data_max_nozeros= tf.reduce_max(tf.where(~cond, tf.ones_like(inputs_scaled) * -1.e+99, inputs_scaled), axis=(1,2))
		#tf.print("data_min (nozeros, after posdef)", data_min_nozeros, output_stream=sys.stdout)
		#tf.print("data_max (nozeros, after posdef)", data_max_nozeros, output_stream=sys.stdout)
		#tf.print("data_min (after posdef)", data_min, output_stream=sys.stdout)
		#tf.print("data_max (after posdef)", data_max, output_stream=sys.stdout)
		###########################

		return tf.reshape(inputs_scaled, self.compute_output_shape(input_shape))
		
	def compute_output_shape(self, input_shape):
		return input_shape

	
###############################################
##     ChanMaxRatio LAYER
###############################################
class ChanMaxRatio(layers.Layer):
	""".
	Input shape:
		Arbitrary.
	Output shape:
		[nbatches, nchans]
	"""

	def __init__(self, name=None, **kwargs):
		super(ChanMaxRatio, self).__init__(name=name, **kwargs)

	def build(self, input_shape):
		super(ChanMaxRatio, self).build(input_shape)

	def call(self, inputs, training=False):
		# - Init stuff
		input_shape = tf.shape(inputs)
		
		# - Compute input data channel max, excluding NANs & zeros
		cond= tf.logical_and(tf.math.is_finite(inputs), tf.math.not_equal(inputs, 0.))
		data_max= tf.reduce_max(tf.where(~cond, tf.ones_like(inputs) * -1.e+99, inputs), axis=(1,2))
		#data_max= tf.expand_dims(tf.expand_dims(data_max, axis=1),axis=1)
		
		# - Compute absolute max across channels
		data_absmax= tf.reduce_max(data_max, axis=1)
		data_absmax= tf.expand_dims(data_absmax, axis=1)

		# - Compute max ratios
		xmax_ratio= data_max/data_absmax

		return xmax_ratio
		

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[-1])


###############################################
##     ChanMeanRatio LAYER
###############################################
class ChanMeanRatio(layers.Layer):
	""".
	Input shape:
		Arbitrary.
	Output shape:
		[nbatches, nchans]
	"""

	def __init__(self, name=None, **kwargs):
		super(ChanMeanRatio, self).__init__(name=name, **kwargs)

	def build(self, input_shape):
		super(ChanMeanRatio, self).build(input_shape)

	def call(self, inputs, training=False):
		# - Init stuff
		input_shape = tf.shape(inputs)
		
		# - Compute input data channel max, excluding NANs & zeros
		cond= tf.logical_and(tf.math.is_finite(inputs), tf.math.not_equal(inputs, 0.))
		npix= tf.reduce_sum( tf.cast(cond, tf.float32), axis=(1,2) )
		pix_sum= tf.reduce_sum(tf.where(~cond, tf.ones_like(inputs) * 0, inputs), axis=(1,2))
		data_mean= tf.math.divide_no_nan(pix_sum, npix)
		
		# - Compute max of means across channels
		data_mean_max= tf.reduce_max(data_mean, axis=1)
		data_mean_max= tf.expand_dims(data_mean_max, axis=1)
		
		# - Compute mean ratios
		data_mean_ratio= data_mean/data_mean_max

		return data_mean_ratio
		

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[-1])


###############################################
##     ChanSumRatio LAYER
###############################################
class ChanSumRatio(layers.Layer):
	""".
	Input shape:
		Arbitrary.
	Output shape:
		[nbatches, nchans]
	"""

	def __init__(self, name=None, **kwargs):
		super(ChanSumRatio, self).__init__(name=name, **kwargs)

	def build(self, input_shape):
		super(ChanSumRatio, self).build(input_shape)

	def call(self, inputs, training=False):
		# - Init stuff
		input_shape = tf.shape(inputs)
		
		# - Compute input data channel max, excluding NANs & zeros
		cond= tf.logical_and(tf.math.is_finite(inputs), tf.math.not_equal(inputs, 0.))
		data_sum= tf.reduce_sum(tf.where(~cond, tf.ones_like(inputs) * 0, inputs), axis=(1,2))
		
		# - Compute max of pixel sums across channels
		data_sum_max= tf.reduce_max(data_sum, axis=1)
		data_sum_max= tf.expand_dims(data_sum_max, axis=1)

		# - Compute sum ratios
		data_sum_ratio= data_sum/data_sum_max

		return data_sum_ratio
		

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[-1])


###############################################
##     SoftmaxCosineSim LAYER
###############################################
# Taken from https://github.com/mwdhont/SimCLRv1-keras-tensorflow/blob/master/SoftmaxCosineSim.py

# ==============================================================================
# Code modified from NT-XENT-loss:
# https://github.com/google-research/simclr/blob/master/objective.py
# ==============================================================================
# coding=utf-8
# Copyright 2020 The SimCLR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================


class SoftmaxCosineSim(layers.Layer):
	""" Custom Keras layer: takes all z-projections as input and calculates
			output matrix which needs to match to [I|O|I|O], where
				I = Unity matrix of size (batch_size x batch_size)
				O = Zero matrix of size (batch_size x batch_size)
	"""

	def __init__(self, batch_size, feat_dim, **kwargs):
		super(SoftmaxCosineSim, self).__init__()
		self.batch_size = batch_size
		self.feat_dim = feat_dim
		self.units = (batch_size, 4 * feat_dim)
		self.input_dim = [(None, feat_dim)] * (batch_size * 2)
		self.temperature = 0.1
		self.LARGE_NUM = 1e9

	def get_config(self):
		config = super().get_config().copy()
		config.update(
			{
				"batch_size": self.batch_size,
				"feat_dim": self.feat_dim,
				"units": self.units,
				"input_dim": self.input_dim,
				"temperature": self.temperature,
				"LARGE_NUM": self.LARGE_NUM,
			}
		)
		return config

	def call(self, inputs):
		z1 = []
		z2 = []

		for index in range(self.batch_size):
			# 0-index assumes that batch_size in generator is equal to 1
			z1.append(
				tf.math.l2_normalize(inputs[index][0], -1)
			)
			z2.append(
				tf.math.l2_normalize(inputs[self.batch_size + index][0], -1)
			)

		# Gather hidden1/hidden2 across replicas and create local labels.
		z1_large = z1
		z2_large = z2

		masks = tf.one_hot(tf.range(self.batch_size), self.batch_size)

		# Products of vectors of same side of network (z_i), count as negative examples
		# Values on the diagonal are put equal to a very small value
		# -> exclude product between 2 identical values, no added value
		logits_aa = tf.matmul(z1, z1_large, transpose_b=True) / self.temperature
		logits_aa = logits_aa - masks * self.LARGE_NUM

		logits_bb = tf.matmul(z2, z2_large, transpose_b=True) / self.temperature
		logits_bb = logits_bb - masks * self.LARGE_NUM

		# Similarity between two transformation sides of the network (z_i and z_j)
		# -> diagonal should be as close as possible to 1
		logits_ab = tf.matmul(z1, z2_large, transpose_b=True) / self.temperature
		logits_ba = tf.matmul(z2, z1_large, transpose_b=True) / self.temperature

		part1 = softmax(tf.concat([logits_ab, logits_aa], 1))
		part2 = softmax(tf.concat([logits_ba, logits_bb], 1))
		output = tf.concat([part1, part2], 1)

		return output

