import tensorflow as tf
from load_data import load_folder
import tensorflow as tf
import random
#labels, images, lti = load_folder("../dataset/processed_images/train")

class Process(object):
	def __init__(self, labels, images):
		self.labels = labels
		self.images = images
		self.batchSize = 32
		self.shuffle_buffer = 10000
	
	def buildDataset(self):
		data = list(zip(self.images, self.labels))
		random.shuffle(data)
		inputs, outputs = zip(*data)
		print("Here we go:")
		print(inputs[:10])
		train_ds = tf.data.Dataset.from_tensor_slices((list(inputs),list(outputs) ))

		train_ds = train_ds.map(self._parse_image, num_parallel_calls=4)
		train_ds = train_ds.batch(self.batchSize)
		train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
		return train_ds

	def _parse_image(self, raw, label, training=False):
		image = tf.io.read_file(raw)
		image = tf.image.decode_jpeg(image, channels=3)
		image = tf.image.convert_image_dtype(image, tf.float32)
		image = tf.image.per_image_standardization(image)
		

		return image, label

#p = Process(labels, images)
#ds = p.buildDataset()
#print("Built")
#print(ds)






