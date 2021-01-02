import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

### Import data ###
mnist = tf.keras.datasets.mnist
(train_images, train_label), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)
print(train_label.shape)

### Pre-process data ###

train_images = train_images[:3000]/255.0
test_images = test_images[:3000]/255.0
train_label = train_label[:3000]
test_labels = test_labels[:3000]
train_images = tf.expand_dims(train_images, -1)
test_images = tf.expand_dims(test_images, -1)
train_labels_processed = np.zeros((len(train_label), 10))
test_labels_processed = np.zeros((len(test_labels), 10))

for i in range(len(train_label)):
    train_labels_processed[i,train_label[i]] = 1.0
    test_labels_processed[i, test_labels[i]] = 1.0

train_labels_processed = tf.convert_to_tensor(train_labels_processed)
test_labels_processed = tf.convert_to_tensor(test_labels_processed)

print(train_label[0])
print(train_labels_processed[0])
print(test_labels[0])
print(test_labels_processed[0])
### Build model ###

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.Dropout1 = tf.keras.layers.Dropout(0.2)
        self.Conv1 = tf.keras.layers.Conv2D(16, 2, padding='valid',activation='relu', input_shape=(1,28,28,1), name='Conv1')
        self.MaxPool1 = tf.keras.layers.MaxPool2D(padding='same')
        self.Dropout2 = tf.keras.layers.Dropout(0.2)
        self.Conv2 = tf.keras.layers.Conv2D(16, 2, padding='valid', activation='relu', input_shape=(1,28,28,1), name='Conv2')
        self.MaxPool2 = tf.keras.layers.MaxPool2D(padding='same')
        self.Dropout3 = tf.keras.layers.Dropout(0.2)
        self.Flat = tf.keras.layers.Flatten()
        self.Fc1 = tf.keras.layers.Dense(256, activation='relu', name='Fc1')
        self.Fc2 = tf.keras.layers.Dense(128, activation='relu', name='Fc2')
        self.Fc3 = tf.keras.layers.Dense(10, activation='softmax', name='Fc3')

    def call(self, inputs):
        x = self.Dropout1(inputs)
        x = self.Conv1(x)
        x = self.MaxPool1(x)
        x = self.Dropout2(x)
        x = self.Conv2(x)
        x = self.MaxPool2(x)
        x = self.Dropout3(x)
        #Flat
        x = self.Flat(x)
        x = self.Fc1(x)
        x = self.Fc2(x)
        output = self.Fc3(x)

        return output

model = MyModel()
output_data = model(tf.reshape(train_images[0],(1,28,28,1)))
print(output_data)
print(output_data.shape)

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
checkpoint_path = './train_model/cp.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='.\logs', histogram_freq=1)
model.fit(x=train_images, y=train_labels_processed, epochs=15, callbacks=[tensorboard_callback, cp_callback])

test_loss, test_acc = model.evaluate(test_images, test_labels_processed, verbose=1)



