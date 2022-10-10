"""
File to create white background image so that the nn can recognize just white background.


"""
#import tensorflow as tf
import numpy as np
import image_processing as impro
import matplotlib.pyplot as plt
import cv2

image = cv2.imread("image_test/imagetest1.jpg")

"""
impro.display_img(image)

image = impro.convert_to_PIL(image)
image = impro.convert_to_grayscale(image)
image = np.asarray(image)/255.0

impro.display_img(image)

image = impro.get_window(image, size=(28, 28), pos=(75, 75))

impro.display_img(image)

image = impro.increase_contrast(image)
image = impro.clean_backgroundV2(image)

impro = impro.display_img(image)
"""

### Convert image to pil and greyscale then back to numpy
image = impro.convert_to_PIL(image)
image = impro.convert_to_grayscale(image)
image = np.asarray(image)/255.0
print(image.shape)
images = []
labels = np.ones(6000)*10

for i in range(6000):
    positions = np.random.randint(30, 450, size=2)
    temp_im = impro.get_window(image, pos=positions)
    temp_im = impro.increase_contrast(temp_im)
    temp_im = impro.clean_backgroundV2(temp_im)
    images.append(temp_im)

npimg = np.asarray(images)
npimg = np.expand_dims(npimg, -1)
print(npimg.shape)
print(labels.shape)
"""
fig, ax = plt.subplots(5, 5)
ax = ax.flatten()

for i in range(25):
    ax[i].imshow(images[i])

plt.show()
"""