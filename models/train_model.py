import tensorflow as tf
from load_data import load_folder
from vanilla_model import VanillaModel

# labels, images, lti = load_folder("../dataset/processed_images/train/")
# test_labels, test_images, test_lti = load_folder("../dataset/processed_images/test/")

train_ds = tf.data.Dataset.from_tensor_slices((images, labels)).shuffle(10000).batch(32)

model = VanillaModel()

loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")



@tf.function
def train_step(images, labels):
	with tf.GradientTape() as tape:
		# predictions = model(images)
		loss = loss_object(labels, predictions)
	gradients = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	train_loss(loss)
	train_accuracy(labels, predictions)

EPOCHS = 5
for epoch in range(EPOCHS):
	for images, labels in train_ds:
		train_step(images, labels)

	model.save_weights("model_content", save_format="tf")

	print("Epoch:", str(epoch+1)," Loss:",str(train_loss.result())," Accuracy:",str(train_accuracy.result()*100))
	train_loss.reset_states()
	train_accuracy.reset_states()

	predictions = np.argmax(model(test_images),axis=1)