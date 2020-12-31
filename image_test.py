import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import PIL
import PIL.Image as Image

mnist = tf.keras.datasets.mnist
(train_images, train_label), (test_images, test_labels) = mnist.load_data()

print(train_images[0].shape)
image = train_images[0]/255.0
im = Image.fromarray(np.uint8(cm.gist_earth(image)*255))
im = im.resize((56, 56))
plt.figure()
plt.imshow(im)
plt.show()