import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import image_processing as imp
import NN_use as NN
import cv2
import PIL

"""
image = imp.get_picture()
imp.display_img(image)
image2 = imp.convert_to_PIL(image)
print(np.array(imp.convert_to_grayscale(image2)).shape)
imp.display_img(imp.convert_to_grayscale(image2))
w = imp.get_window(image, size=(150, 150), pos=(10, 5), channel=3)
print(w.shape)
imp.display_img(w)

cv2.imwrite('./image_test/imagetest2.jpg', image*255)
"""


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

imp.display_img(test_images[0])
"""

model = NN.createNetwork()

image = cv2.imread('./image_test/imagetest2.jpg')/255.0
imp.display_img(image)
image2 = imp.convert_to_grayscale(imp.convert_to_PIL(image))
imp.display_img(image2)

#window = imp.get_window(image, size=(36, 36), pos=(287, 360)) #im
#window = imp.get_window(image, size=(36, 36), pos=(204, 486)) #im
#window = imp.get_window(image, size=(32, 32), pos=(70, 229)) #im
#window = imp.get_window(image, size=(32, 32), pos=(68, 379)) #im1
#window = imp.get_window(image, size=(38, 38), pos=(187, 504)) #im1
#window = imp.get_window(image, size=(28, 28), pos=(287, 360)) #im
#window = imp.get_window(image, pos=(204, 486)) #im
#window = imp.get_window(image, pos=(70, 229)) #im

#window = imp.get_window(image, size=(38, 38), pos=(55, 161)) #im2
#window = imp.get_window(image, size=(38, 38), pos=(110, 329)) #im2
#window = imp.get_window(image, size=(38, 38), pos=(108, 505)) #im2
window = imp.get_window(image, size=(56, 56), pos=(220, 362)) #im2

window2 = imp.convert_to_grayscale(imp.convert_to_PIL(window))
window2 = imp.resize_image(window2)
nump_window2 = (1 - np.asarray(window2))/255.0

nump_window4 = imp.increase_contrast(nump_window2)
nump_window2 = imp.clean_backgroundV2(nump_window2)
nump_window4 = imp.clean_backgroundV2(nump_window4)

imp.display_img(window)
imp.display_img(nump_window2)
imp.display_img(nump_window4)

nump_window_in = np.expand_dims(nump_window4, [0, -1])
nump_window_in = nump_window_in



output = model.predict(nump_window_in)
print(output)
print("With contrast, his is a " + str(np.argmax(output)) + " with a confidence of " + str(output[0, np.argmax(output)]*100)+"%")
