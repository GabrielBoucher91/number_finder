import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import image_processing as imp
import NN_use as NN

"""
image = imp.get_picture()
imp.display_img(image)
image2 = imp.convert_to_PIL(image)
print(np.array(imp.convert_to_grayscale(image2)).shape)
imp.display_img(imp.convert_to_grayscale(image2))
w = imp.get_window(image, size=(150, 150), pos=(10, 5), channel=3)
print(w.shape)
imp.display_img(w)
"""

model = NN.createNetwork()

dataset = tf.keras.datasets.mnist
(train_images, train_label), (test_images, test_labels) = dataset.load_data()
train_images = train_images[:3000]/255.0
test_images = test_images[:3000]/255.0
train_label = train_label[:3000]
test_labels = test_labels[:3000]
train_images = tf.expand_dims(train_images, -1)
test_images = tf.expand_dims(test_images, -1)

loss, acc = model.evaluate(test_images, test_labels, verbose=2)




