from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convol
class ResnetBlock(tf.keras.Model):
	def __init__(self, filters, kernel_size):
		super(ResnetBlock, self).__init__(name="")
		filters1, filters2, filters3 = filters

		self.conv1 = tf.keras.layers.Conv2D(filters1, (1,1))
		self.bn1 = tf.keras.layers.BatchNormalization()
		
		self.conv2 = tf.keras.layers.Conv2D(filters2, kernel_size, padding="same")
		self.bn2 = tf.keras.layers.BatchNormalization()

		self.conv3 = tf.keras.layers.Conv2D(filters3, (1,1))
		self.bn3 = tf.keras.layers.BatchNormalization()
		
	
	def call(self, input_tensor, training=False):
		x = self.conv1(input_tensor)
		x = self.bn1(x, training=training)
		x = tf.nn.relu(x)

		x = self.conv2(x)
		x = self.bn2(x, training=training)
		x = tf.nn.relu(x)
		
		x = self.conv3(x)
		x = self.bn3(x, training=training)
	
		x = x + input_tensor
		return tf.nn.relu(x)
