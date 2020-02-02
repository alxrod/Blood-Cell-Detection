import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import Model
from resnet_block import ResnetBlock
from load_data import load_folder

class ResnetModel(Model):
	def __init__(self):
		super(ResnetModel, self).__init__()
		
		self.conv1 = Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l1(0.4))
		self.pool1 = MaxPool2D((2,2))
		self.conv2 = Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l1(0.4))
		self.pool2 = MaxPool2D((2,2))
		self.res1 = ResnetBlock([32, 32, 64], 1)
		self.res2 = ResnetBlock([32, 32, 64], 1)
		self.flatten = Flatten()
		self.d1 = Dense(512, activation="relu")
		self.dropout1 = Dropout(0.8)
		self.d2 = Dense(128, activation="relu")
		self.dropout2 = Dropout(0.8)
		self.d3 = Dense(32, activation="relu")
		self.dropout3 = Dropout(0.8)
		self.d4 = Dense(4, activation="softmax")

	def call(self, x):
		x = self.conv1(x)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.pool2(x)
		x = self.res1(x, training=True)
		x = self.res2(x, training=True)
		x = self.flatten(x)
		x = self.d1(x)
		x = self.dropout1(x)
		x = self.d2(x)
		x = self.dropout2(x)
		x = self.d3(x)
		x = self.dropout3(x)
		x = self.d4(x)
		return x

