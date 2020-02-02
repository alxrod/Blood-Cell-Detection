import tensorflow as tf

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation                          
from tensorflow.keras.layers import Dense                                 
from tensorflow.keras.models import Model                             
from tensorflow.keras.layers import add                                   
from tensorflow.keras.regularizers import l2                           

class ResnetBlock(tf.keras.Model):
	def __init__(self, K, stride, chanDim, red=False, reg=0.0001, bnEps=2e-5, bnMom=0.9):
            super(ResnetBlock, self).__init__()
            self.red = red
            self.bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)
            self.act1 = Activation("relu")
            self.conv1 = Conv2D(int(K*0.25), (1,1), use_bias=False, kernel_regularizer=l2(reg))
            
            self.bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)
            self.act2 = Activation("relu")
            self.conv2 = Conv2D(int(K*0.25), (3,3), strides=stride, padding="same", use_bias=False, kernel_regularizer=l2(reg))

            self.bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)
            self.act3 = Activation("relu")
            self.conv3 = Conv2D(K, (1,1), use_bias=False, kernel_regularizer=l2(reg))
            self.scConv = Conv2D(K, (1,1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))


	def call(self, data, training=False):
            shortcut = data
            x = self.bn1(data)
            act1 = self.act1(x)
            x = self.conv1(act1)

            x = self.bn2(x)
            x = self.act2(x)
            x = self.conv2(x)

            x = self.bn3(x)
            x = self.act3(x)
            x = self.conv3(x)
            
            if self.red:
                shortcut = self.scConv(act1)

            x = add([x, shortcut])
            return x

            
