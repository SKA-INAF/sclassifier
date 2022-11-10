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
		tf.print("data_min (before norm)", data_min, output_stream=sys.stdout)
		tf.print("data_max (before norm)", data_max, output_stream=sys.stdout)
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
		
		tf.print("data_min (after norm)", data_min, output_stream=sys.stdout)
		tf.print("data_max (after norm)", data_max, output_stream=sys.stdout)
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
		data_mean= tf.reduce_mean(tf.where(~cond, tf.ones_like(inputs) * -1.e+99, inputs), axis=(1,2))
		#data_mean= tf.expand_dims(tf.expand_dims(data_mean, axis=1),axis=1)
		
		# - Compute max of means across channels
		data_mean_max= tf.reduce_max(data_mean, axis=1)
		data_mean_max= tf.expand_dims(data_mean_max, axis=1)

		# - Compute mean ratios
		data_mean_ratio= data_mean/data_mean_max

		tf.print("data_mean_ratio", data_mean_ratio, output_stream=sys.stdout)

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
		data_sum= tf.reduce_sum(tf.where(~cond, tf.ones_like(inputs) * -1.e+99, inputs), axis=(1,2))
		#data_sum= tf.expand_dims(tf.expand_dims(data_sum, axis=1),axis=1)
		
		# - Compute max of pixel sums across channels
		data_sum_max= tf.reduce_max(data_sum, axis=1)
		data_sum_max= tf.expand_dims(data_sum_max, axis=1)

		# - Compute sum ratios
		data_sum_ratio= data_sum/data_sum_max

		return data_sum_ratio
		

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[-1])

