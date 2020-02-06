import tensorflow as tf
# import the necessary packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from resnet_block import ResnetBlock


class ResnetModel(Model):
	def __init__(self):
		super(ResnetModel, self).__init__()
		self.stages = (3,4,6)
		self.filters = (64, 128, 256, 512)

		self.reg = 0.1
		self.bnEps=2e-5
		self.bnMom=0.9
		    
		    #I hardcoded this one
		self.inputShape = (112,112,3)
		self.chanDim = -1

		self.bn0 = BatchNormalization(axis=self.chanDim, epsilon=self.bnEps, momentum=self.bnMom)
		self.conv1 = Conv2D(self.filters[0], (5,5), use_bias=False, padding="same", kernel_regularizer=l2(self.reg))
		self.bn1 = BatchNormalization(axis=self.chanDim, epsilon=self.bnEps, momentum=self.bnMom)
		self.act1 = Activation("relu")
		self.zp1 = ZeroPadding2D((1,1))
		self.mp1 = MaxPooling2D((3,3), strides=(2,2))
		self.resBlocks = []
		for i in range(0, len(self.stages)):
			stride = (1,1) if i == 0 else (2,2)
			rb = ResnetBlock(self.filters[i+1], stride, self.chanDim, red=True, bnEps=self.bnEps, bnMom=self.bnMom)
			self.resBlocks.append(rb)

			for j in range(0, self.stages[i]-1):
				rb = ResnetBlock(self.filters[i+1], (1,1), self.chanDim, bnEps=self.bnEps, bnMom=self.bnMom)
			self.resBlocks.append(rb)

		self.bn2 = BatchNormalization(axis=self.chanDim, epsilon=self.bnEps, momentum=self.bnMom)
		self.act2 = Activation("relu")
		#        self.ap1 = AveragePooling2D((8, 8))
		self.f = Flatten()
		self.d1 = Dense(300, activation="relu", kernel_regularizer=l2(self.reg))
		self.do1 = Dropout(0.6)
		self.d2 = Dense(100, activation="relu", kernel_regularizer=l2(self.reg))
		self.do2 = Dropout(0.6)
		self.d3 = Dense(25, activation="relu", kernel_regularizer=l2(self.reg))
		self.do3 = Dropout(0.6)
		self.df = Dense(4, kernel_regularizer=l2(self.reg))
		self.act3 = Activation("softmax")

	def call(self, inputData):
		x = self.bn0(inputData)
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.act1(x)
		x = self.zp1(x)
		x = self.mp1(x)

		for rb in self.resBlocks:
		    x = rb(x, training=True)

		x = self.bn2(x)
		x = self.act2(x)

		x = self.f(x)
		x = self.d1(x)
		x = self.do1(x)
		x = self.d2(x)
		x = self.do2(x)
		x = self.d3(x)
		x = self.do3(x)
		x = self.df(x)
		return self.act3(x)
