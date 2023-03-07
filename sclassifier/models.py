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

##############################
##     GLOBAL VARS
##############################
from sclassifier import logger

## TENSORFLOW & KERAS MODULES
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#===============================
#==     DEFINE RESNET 18/34
#===============================
# Taken from https://github.com/garder14/byol-tensorflow2/blob/main/models.py

class BasicBlock(tf.keras.layers.Layer):
	""" Define base block """

	def __init__(self, filters, strides):
		super(BasicBlock, self).__init__()
		self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, padding='same')
		self.bn1 = tf.keras.layers.BatchNormalization()
		self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')
		self.bn2 = tf.keras.layers.BatchNormalization()

		if strides != 1:
			self.convdown = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=strides)
			self.bndown = tf.keras.layers.BatchNormalization()
		self.strides = strides

	def call(self, inp, training=False):
		x1 = self.conv1(inp)
		x1 = self.bn1(x1, training=training)
		x1 = tf.nn.relu(x1)
		x1 = self.conv2(x1)
		x1 = self.bn2(x1, training=training)

		if self.strides != 1:
			x2 = self.convdown(inp)
			x2 = self.bndown(x2, training=training)
		else:
			x2 = inp

		x = tf.keras.layers.add([x1, x2])
		x = tf.nn.relu(x)
		return x


# ResNet with BasicBlock (adapted to CIFAR-10)
class BasicResNet(tf.keras.Model):

	def __init__(self, layer_blocks):
		super(BasicResNet, self).__init__()

		self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')
		self.bn1 = tf.keras.layers.BatchNormalization()

		self.blocks = []
		self.blocks.append(BasicBlock(filters=64, strides=1))
		for _ in range(layer_blocks[0] - 1):
			self.blocks.append(BasicBlock(filters=64, strides=1))
        
		self.blocks.append(BasicBlock(filters=128, strides=2))
		for _ in range(layer_blocks[1] - 1):
			self.blocks.append(BasicBlock(filters=128, strides=1))
        
		self.blocks.append(BasicBlock(filters=256, strides=2))
		for _ in range(layer_blocks[2] - 1):
			self.blocks.append(BasicBlock(filters=256, strides=1))
        
		self.blocks.append(BasicBlock(filters=512, strides=2))
		for _ in range(layer_blocks[3] - 1):
			self.blocks.append(BasicBlock(filters=512, strides=1))

		self.avgpool = tf.keras.layers.GlobalAveragePooling2D()

	def call(self, inp, training=False):
		x = self.conv1(inp)
		x = self.bn1(x, training=training)
		x = tf.nn.relu(x)
		for block in self.blocks:
			x = block(x, training=training)
		x = self.avgpool(x)    
		return x


def ResNet18():
	""" Create resnet18 model """
	return BasicResNet(layer_blocks=[2, 2, 2, 2])


def ResNet34():
	""" Create resnet34 model """
	return BasicResNet(layer_blocks=[3, 4, 6, 3])

#=================================
#==     DEFINE PROJECTION HEAD
#=================================
# 512 (h) -> 256 -> 128 (z)
class ProjectionHead(tf.keras.Model):
	""" Define projection head model """

	def __init__(self):
		super(ProjectionHead, self).__init__()
		self.fc1 = tf.keras.layers.Dense(units=256)
		self.bn = tf.keras.layers.BatchNormalization()
		self.fc2 = tf.keras.layers.Dense(units=128)

	def call(self, inp, training=False):
		x = self.fc1(inp)
		x = self.bn(x, training=training)
		x = tf.nn.relu(x)
		x = self.fc2(x)
		return x

#====================================
#==     DEFINE CLASSIFICATION HEAD
#====================================
# 512 (h) -> 10 (s)
class ClassificationHead(tf.keras.Model):
	""" Define classification head model """

	def __init__(self):
		super(ClassificationHead, self).__init__()
		self.fc = tf.keras.layers.Dense(units=10)

	def call(self, inp):
		x = self.fc(inp)
		return x




#===============================
#==     DEFINE RESNET 18/34
#===============================
# Taken from https://github.com/jimmyyhwu/resnet18-tf2
# Adapted from https://github.com/pytorch/vision/blob/v0.4.0/torchvision/models/resnet.py


kaiming_normal = keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')

def conv3x3(x, out_planes, stride=1, name=None):
	x = layers.ZeroPadding2D(padding=1, name=f'{name}_pad')(x)
	return layers.Conv2D(filters=out_planes, kernel_size=3, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=name)(x)

def basic_block(x, planes, stride=1, downsample=None, name=None):
	identity = x

	out = conv3x3(x, planes, stride=stride, name=f'{name}.conv1')
	out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn1')(out)
	out = layers.ReLU(name=f'{name}.relu1')(out)

	out = conv3x3(out, planes, name=f'{name}.conv2')
	out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn2')(out)

	if downsample is not None:
		for layer in downsample:
			identity = layer(identity)

	out = layers.Add(name=f'{name}.add')([identity, out])
	out = layers.ReLU(name=f'{name}.relu2')(out)

	return out

def make_layer(x, planes, blocks, stride=1, name=None):
	""" Create conv layer block """
	downsample = None
	inplanes = x.shape[3]
	if stride != 1 or inplanes != planes:
		downsample = [
			layers.Conv2D(filters=planes, kernel_size=1, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=f'{name}.0.downsample.0'),
			layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.0.downsample.1'),
		]

	x = basic_block(x, planes, stride, downsample, name=f'{name}.0')
	for i in range(1, blocks):
		x = basic_block(x, planes, name=f'{name}.{i}')

	return x

def resnet(x, blocks_per_layer, include_top=True, num_classes=1000):
	""" Define resnet network """
	x = layers.ZeroPadding2D(padding=3, name='conv1_pad')(x)
	x = layers.Conv2D(filters=64, kernel_size=7, strides=2, use_bias=False, kernel_initializer=kaiming_normal, name='conv1')(x)
	x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
	x = layers.ReLU(name='relu1')(x)
	x = layers.ZeroPadding2D(padding=1, name='maxpool_pad')(x)
	x = layers.MaxPool2D(pool_size=3, strides=2, name='maxpool')(x)

	x = make_layer(x, 64, blocks_per_layer[0], name='layer1')
	x = make_layer(x, 128, blocks_per_layer[1], stride=2, name='layer2')
	x = make_layer(x, 256, blocks_per_layer[2], stride=2, name='layer3')
	x = make_layer(x, 512, blocks_per_layer[3], stride=2, name='layer4')

	x = layers.GlobalAveragePooling2D(name='avgpool')(x)

	if include_top:
		initializer = keras.initializers.RandomUniform(-1.0 / math.sqrt(512), 1.0 / math.sqrt(512))
		x = layers.Dense(units=num_classes, kernel_initializer=initializer, bias_initializer=initializer, name='fc')(x)

	return x

def resnet18(x, **kwargs):
	""" Define resnet18 """
	return resnet(x, [2, 2, 2, 2], **kwargs)

def resnet34(x, **kwargs):
	""" Define resnet34 """
	return resnet(x, [3, 4, 6, 3], **kwargs)

