import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(train_images, train_label), (test_images, test_labels) = mnist.load_data()

train_images = train_images[:500]/255.0
train_images = np.expand_dims(train_images, -1)

plt.figure()
for i in range(64):
    plt.subplot(8, 8, i+1)
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)

plt.show()
"""
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2
)
"""

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    zoom_range=0.0
)

train_aug = datagen.flow(train_images, shuffle=False)

print(train_aug.next().shape)
train_aug.reset()
train_aug_np = train_aug.next()
for i in range(train_aug.__len__()):
    if i == 0:
        pass
    else:
        train_aug_np = np.concatenate((train_aug_np, train_aug.next()))


print(train_aug_np.shape)

plt.figure()
for i in range(64):
    plt.subplot(8, 8, i+1)
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)


plt.figure()
for i in range(64):
    plt.subplot(8, 8, i+1)
    plt.grid(False)
    plt.imshow(train_aug_np[i], cmap=plt.cm.binary)
    plt.ylabel(train_label[i])

plt.show()


