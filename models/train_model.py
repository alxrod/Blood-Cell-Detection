import tensorflow as tf
from load_data import load_folder
from vanilla_model import VanillaModel
from resnet_model import ResnetModel
from dataset_compiler import Process
import numpy as np

labels, images, lti = load_folder("../dataset/processed_images/train/")
test_labels, test_images, test_lti = load_folder("../dataset/processed_images/test/")
p = Process(labels, images)
train_ds = p.buildDataset()
pT = Process(test_labels, test_images)
test_ds = pT.buildDataset()
print(train_ds)
print("Dataset loaded")

model = ResnetModel()

loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")



@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


def validation_predict(model):
    total = 0
    total_correct = 0  
    for images, labels in test_ds:
        predictions = np.argmax(model(images),axis=1)
        correct = tf.math.reduce_sum(tf.dtypes.cast(tf.math.equal(predictions,np.argmax(labels,axis=1)),dtype=tf.int32))
        total_correct += correct
        total += 32
    
    print("Validation Accuracy: " + str(total_correct/total))

EPOCHS = 10
for epoch in range(EPOCHS):
    for images, labels in train_ds:
    	train_step(images, labels)
    model.save_weights("./model_save/checkpoint")

    print("Epoch:", str(epoch+1)," Loss:",str(train_loss.result())," Accuracy:",str(train_accuracy.result()*100))
    train_loss.reset_states()
    train_accuracy.reset_states()
    validation_predict(model)




