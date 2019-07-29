from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

# Basic conceptual model
class PlebModel(Model):
	def __init__(self):
		super(PlebModel, self).__init__()
		self.conv1 = Conv2D(32, 3, activation="relu")
		self.flatten = Flatten()
		self.d1 = (128, activation="relu")
		self.d2 = Dense(4, activation="softmax")

	def call(self, x):
		x = self.conv1(x)
		x = self.flatten(x)
		x = self.d1(x)
		return self.d2(x)