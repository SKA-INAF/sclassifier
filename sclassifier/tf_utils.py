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
##     ChanMinMaxNormalization LAYER
###############################################
class ChanMinMaxNormalization(layers.Layer):
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
		super(ChanMinMaxNormalization, self).__init__(name=name, **kwargs)

	def build(self, input_shape):
		super(ChanMinMaxNormalization, self).build(input_shape)

	def call(self, inputs, training=False):
		# - Init stuff
		input_shape = tf.shape( inputs )
		norm_min= self.norm_min
		norm_max= self.norm_max
		
		#tf.print("call(): input_shape", input_shape, output_stream=sys.stdout)
		#tf.print("call(): K.int_shape", K.int_shape(inputs), output_stream=sys.stdout)

		# - Compute input data min & max, excluding NANs & zeros
		cond= tf.logical_and(tf.math.is_finite(inputs), tf.math.not_equal(inputs, 0.))
		#mask= tf.ragged.boolean_mask(inputs, mask=cond)
		#data_min= tf.reduce_min(mask, axis=(1,2)) ## NB: WRONG not providing correct results with ragged tensor, don't use!!!
		#data_max= tf.reduce_max(mask, axis=(1,2)) ## NB: WRONG not providing correct results with ragged tensor, don't use!!!

		data_min= tf.reduce_min(tf.where(~cond, tf.ones_like(inputs) * 1.e+99, inputs), axis=(1,2))
		data_max= tf.reduce_max(tf.where(~cond, tf.ones_like(inputs) * -1.e+99, inputs), axis=(1,2))
		data_min= tf.expand_dims(tf.expand_dims(data_min, axis=1),axis=1)
		data_max= tf.expand_dims(tf.expand_dims(data_max, axis=1),axis=1)
		
		##### DEBUG ############
		#tf.print("data_min raw", data_min, output_stream=sys.stdout)
		#tf.print("data_max raw", data_max, output_stream=sys.stdout)
		#data_min= data_min.to_tensor()
		#data_max= data_max.to_tensor()

		#tf.print("data_min shape", K.int_shape(data_min), output_stream=sys.stdout)
		#tf.print("data_max shape", K.int_shape(data_max), output_stream=sys.stdout)
		
		#sample= 0
		#ch= 0
		#iy= 31
		#ix= 31
		#tf.print("data_min (before norm)", data_min, output_stream=sys.stdout)
		#tf.print("data_max (before norm)", data_max, output_stream=sys.stdout)
		#tf.print("data_min[sample,:,:,:] (before norm)", data_min[sample,:,:,:], output_stream=sys.stdout)
		#tf.print("data_max[sample,:,:,:] (before norm)", data_max[sample,:,:,:], output_stream=sys.stdout)
		#tf.print("inputs[sample,iy,ix,:] (before norm)", inputs[sample,iy,ix,:], output_stream=sys.stdout)
		#########################		

		# - Normalize data in range (norm_min, norm_max)
		data_norm= (inputs-data_min)/(data_max-data_min) * (norm_max-norm_min) + norm_min
		
		# - Set masked values (NANs, zeros) to norm_min
		data_norm= tf.where(~cond, tf.ones_like(data_norm) * norm_min, data_norm)
		
		#######  DEBUG ###########
		#data_min= tf.reduce_min(data_norm, axis=(1,2))
		#data_max= tf.reduce_max(data_norm, axis=(1,2))
		#data_min= tf.expand_dims(tf.expand_dims(data_min, axis=1), axis=1)
		#data_max= tf.expand_dims(tf.expand_dims(data_max, axis=1), axis=1)
		
		#tf.print("data_min (after norm)", data_min, output_stream=sys.stdout)
		#tf.print("data_max (after norm)", data_max, output_stream=sys.stdout)
		#tf.print("data_min[sample,:,:,:] (after norm)", data_min[sample,:,:,:], output_stream=sys.stdout)
		#tf.print("data_max[sample,:,:,:] (after norm)", data_max[sample,:,:,:], output_stream=sys.stdout)
		#tf.print("inputs[sample,iy,ix,:] (after norm)", data_norm[sample,iy,ix,:], output_stream=sys.stdout)
		###########################

		return tf.reshape(data_norm, self.compute_output_shape(input_shape))
		
	def compute_output_shape(self, input_shape):
		return input_shape

	def get_config(self):
		config = {
			'norm_min': self.norm_min,
			'norm_max': self.norm_max,
		}
		base_config = super(ChanMinMaxNormalization, self).get_config()
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
		
		tf.print("call(): input_shape", input_shape, output_stream=sys.stdout)
		tf.print("call(): K.int_shape", K.int_shape(inputs), output_stream=sys.stdout)

		# - Compute input data min & max, excluding NANs & zeros
		cond= tf.logical_and(tf.math.is_finite(inputs), tf.math.not_equal(inputs, 0.))
		
		data_min= tf.reduce_min(tf.where(~cond, tf.ones_like(inputs) * 1.e+99, inputs), axis=(1,2))
		data_max= tf.reduce_max(tf.where(~cond, tf.ones_like(inputs) * -1.e+99, inputs), axis=(1,2))
		data_min= tf.expand_dims(tf.expand_dims(data_min, axis=1),axis=1)
		data_max= tf.expand_dims(tf.expand_dims(data_max, axis=1),axis=1)
		
		##### DEBUG ############
		#tf.print("data_min raw", data_min, output_stream=sys.stdout)
		#tf.print("data_max raw", data_max, output_stream=sys.stdout)
		#data_min= data_min.to_tensor()
		#data_max= data_max.to_tensor()

		#tf.print("data_min shape", K.int_shape(data_min), output_stream=sys.stdout)
		#tf.print("data_max shape", K.int_shape(data_max), output_stream=sys.stdout)
		
		#sample= 0
		#ch= 0
		#iy= 31
		#ix= 31
		tf.print("data_min (before max scale)", data_min, output_stream=sys.stdout)
		tf.print("data_max (before max scale)", data_max, output_stream=sys.stdout)
		#tf.print("data_min[sample,:,:,:] (before norm)", data_min[sample,:,:,:], output_stream=sys.stdout)
		#tf.print("data_max[sample,:,:,:] (before norm)", data_max[sample,:,:,:], output_stream=sys.stdout)
		#tf.print("inputs[sample,iy,ix,:] (before norm)", inputs[sample,iy,ix,:], output_stream=sys.stdout)
		#########################		

		# - Scale data to max
		inputs_scaled= inputs/data_max
		
		# - Set masked values (NANs, zeros) to norm_min
		norm_min= 0
		inputs_scaled= tf.where(~cond, tf.ones_like(inputs_scaled) * norm_min, inputs_scaled)
		
		#######  DEBUG ###########
		data_min= tf.reduce_min(inputs_scaled, axis=(1,2))
		data_max= tf.reduce_max(inputs_scaled, axis=(1,2))
		data_min= tf.expand_dims(tf.expand_dims(data_min, axis=1), axis=1)
		data_max= tf.expand_dims(tf.expand_dims(data_max, axis=1), axis=1)
		
		tf.print("data_min (after max scale)", data_min, output_stream=sys.stdout)
		tf.print("data_max (after max scale)", data_max, output_stream=sys.stdout)
		#tf.print("data_min[sample,:,:,:] (after norm)", data_min[sample,:,:,:], output_stream=sys.stdout)
		#tf.print("data_max[sample,:,:,:] (after norm)", data_max[sample,:,:,:], output_stream=sys.stdout)
		#tf.print("inputs[sample,iy,ix,:] (after norm)", data_norm[sample,iy,ix,:], output_stream=sys.stdout)
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
		data_max= tf.reduce_max(tf.where(~cond, tf.ones_like(inputs) * -1.e+99, inputs), axis=(1,2))
		#data_min= tf.expand_dims(tf.expand_dims(data_min, axis=1),axis=1)
		#data_max= tf.expand_dims(tf.expand_dims(data_max, axis=1),axis=1)
		
		##### DEBUG ############
		tf.print("data_min (before posdef)", data_min, output_stream=sys.stdout)
		tf.print("data_max (before posdef)", data_max, output_stream=sys.stdout)
		#########################		

		# - Subtract data_min on channels with negative data_min
		cond2= tf.math.greater(data_min, 0)
		inputs_scaled= tf.where(~cond2, inputs - data_min, inputs)

		# - Set masked values (NANs, zeros) to norm_min
		norm_min= 0
		inputs_scaled= tf.where(~cond, tf.ones_like(inputs_scaled) * norm_min, inputs_scaled)
		
		#######  DEBUG ###########
		data_min= tf.reduce_min(inputs_scaled, axis=(1,2))
		data_max= tf.reduce_max(inputs_scaled, axis=(1,2))
		#data_min= tf.expand_dims(tf.expand_dims(data_min, axis=1), axis=1)
		#data_max= tf.expand_dims(tf.expand_dims(data_max, axis=1), axis=1)
		
		tf.print("data_min (after posdef)", data_min, output_stream=sys.stdout)
		tf.print("data_max (after posdef)", data_max, output_stream=sys.stdout)
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


