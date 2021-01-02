"""
This file contains the functions for the image processing.
First we open the image.
Then a function extract parts of the image and format them for the NN.
Then from the results of the NN, the image is displayed with the heatmap next to it.

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import PIL
import os
import cv2
from PIL import Image, ImageOps

def get_picture():
    """
    That function gets the picture from the webcam
    :return:
    """
    cam = cv2.VideoCapture(0)
    s, im = cam.read()
    print(type(im))
    return im/255.0

def display_img(img):
    plt.figure()
    plt.imshow(img)
    plt.show()

def get_window(img, size=(28,28),pos=(0,0),channel=1):
    """
    That function gets a window and processes it so that it is the right size for the NN.
    :return:
    """
    window = np.ones((size[0], size[1], channel))
    image_dims = img.shape
    # Check the position too see if the window goes out of bound.
    #in x
    if size[0]/2 > pos[0]:
        deltay = size[0] / 2 - pos[0]
        range_y = (0, int(size[0]-(size[0] / 2 - pos[0])))
        range_y_window = (int(deltay), size[0])
    elif size[0]/2 > (image_dims[0]-pos[0]):
        deltay = size[0] / 2 - (image_dims[0]-pos[0])
        range_y = (int(pos[0]-size[0]/2), int(image_dims[0]))
        range_y_window = (0, int(size[0]-deltay))
    else:
        deltay=None
        range_y = (int(pos[0]-size[0]/2), int(pos[0]+size[0]/2))

    if size[1] / 2 > pos[1]:
        deltax = size[1] / 2 - pos[1]
        range_x = (0, int(size[1]-(size[1] / 2 - pos[1])))
        range_x_window = (int(deltax), size[1])
    elif size[1] / 2 > (image_dims[1] - pos[1]):
        deltax = size[1] / 2 - (image_dims[1] - pos[1])
        range_x = (int(pos[1]-size[1]/2), int(image_dims[1]))
        range_x_window = (0, int(size[1] - deltax))
    else:
        deltax = None
        range_x = (int(pos[1]-size[1]/2), int(pos[1]+size[1]/2))

    if deltax == None and deltay ==None:
        window = img[int(pos[0]-size[0]/2):int(pos[0]+size[0]/2), int(pos[1]-size[1]/2):int(pos[1]+size[1]/2),:]
    else:
        window[range_y_window[0]:range_y_window[1], range_x_window[0]: range_x_window[1], :] = img[range_y[0]:range_y[1], range_x[0]:range_x[1], :]
    return window

def convert_to_PIL(image):
    im = Image.fromarray((image*255).astype(np.uint8))
    return im

def convert_to_grayscale(image):
    im = ImageOps.grayscale(image)
    return im

def clean_background(image, threshold = 0.5):
    image[image < threshold] = 0.0
    return image

def resize_image(image, target_size=(28,28)):
    output = image.resize(target_size)
    return output

def display_heatmap():
    """
    That function displays the image and the heatmap next to it.
    :return:
    """