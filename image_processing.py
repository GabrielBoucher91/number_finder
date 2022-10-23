"""
This file contains the functions for the image processing.
First we open the image.
Then a function extract parts of the image and format them for the NN.
Then from the results of the NN, the image is displayed with the data found in it.

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

    range_y = (int(pos[0]-size[0]/2), int(pos[0]+size[0]/2))
    range_y_window = (int(pos[0]-size[0]/2), int(pos[0]+size[0]/2))

    range_x = (int(pos[1]-size[1]/2), int(pos[1]+size[1]/2))
    range_x_window = (int(pos[1]-size[1]/2), int(pos[1]+size[1]/2))

    if channel!=1:
        window = img[int(pos[0]-size[0]/2):int(pos[0]+size[0]/2), int(pos[1]-size[1]/2):int(pos[1]+size[1]/2), :]
    else:
        window = img[int(pos[0] - size[0] / 2):int(pos[0] + size[0] / 2), int(pos[1] - size[1] / 2):int(pos[1] + size[1] / 2)]
    return window

def convert_to_PIL(image):
    im = Image.fromarray((image*255).astype(np.uint8))
    return im

def convert_to_grayscale(image):
    im = ImageOps.grayscale(image)
    return im

def clean_background(image, threshold = 0.5):
    image[image < np.mean(image)*1.3] = 0.0
    return image

def clean_backgroundV2(image):
    image[image < np.mean(image)*1.2] = 0.0
    image[image > 0.2] = 1
    return image

def clean_backgroundV3(image, threshold):
    ret, img = cv2.threshold(image, min(threshold * np.mean(image), 225), 255, cv2.THRESH_BINARY)
    #img = cv2.adaptiveThreshold(image.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    #img = cv2.adaptiveThreshold(image.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return img.astype(np.uint8)

def increase_contrast(image):
    min_val = np.amin(image)
    max_val = np.amax(image)
    delta = max_val-min_val
    im = image-min_val
    im = im/delta
    return im

def resize_image(image, target_size=(28,28)):
    output = image.resize(target_size)
    return output


def intersectionOverUnion(pos1, window1, pos2, window2):
    x11 = pos1[0] - window1[0]/2
    x12 = pos1[0] + window1[0]/2
    y11 = pos1[1] - window1[1]/2
    y12 = pos1[1] + window1[1]/2
    x21 = pos2[0] - window2[0]/2
    x22 = pos2[0] + window2[0]/2
    y21 = pos2[1] - window2[1]/2
    y22 = pos2[1] + window2[1]/2



    intersection = (max(0, min(x12, x22)-max(x11, x21)))*(max(0, min(y12, y22)-max(y11, y21)))
    union = (x12 - x11)*(y12 - y11) + (x22-x21)*(y22-y21) - intersection
    return intersection/union

def window_split(window, window_pos):
    window_out = []
    window_out_pos = []
    window_pos_split = []
    windows_split = []
    window_w = 0
    window_h = 0
    if window.shape[0] < 160 and window.shape[1] < 160:
        window_out.append(window)
        window_out_pos.append((window_pos[0], window_pos[1]))
    elif window.shape[0] >= 160 and window.shape[1] < 160:
        window_h = int((window.shape[0]/2) * 1.2)
        windows_split.append(window[0:window_h, :])
        window_pos_split.append((window_pos[0], window_pos[1]))
        windows_split.append(window[window.shape[0]-window_h:, :])
        window_pos_split.append((window_pos[0] + window.shape[0]-window_h, window_pos[1]))
    elif window.shape[1] >= 160 and window.shape[0] < 160:
        window_w = int((window.shape[1] / 2) * 1.2)
        windows_split.append(window[:, 0:window_w])
        window_pos_split.append((window_pos[0], window_pos[1]))
        windows_split.append(window[:, window.shape[1] - window_w:])
        window_pos_split.append((window_pos[0], window_pos[1] + window.shape[1] - window_w))
    elif window.shape[1] >= 160 and window.shape[0] >= 160:
        window_h = int((window.shape[0] / 2) * 1.2)
        window_w = int((window.shape[1] / 2) * 1.2)

        windows_split.append(window[0:window_h, 0:window_w])
        window_pos_split.append((window_pos[0], window_pos[1]))

        windows_split.append(window[0:window_h, window.shape[1] - window_w:])
        window_pos_split.append((window_pos[0], window_pos[1] + window.shape[1] - window_w))

        windows_split.append(window[window.shape[0] - window_h:, 0:window_w:])
        window_pos_split.append((window_pos[0] + window.shape[0] - window_h, window_pos[1]))

        windows_split.append(window[window.shape[0] - window_h:, window.shape[1] - window_w:])
        window_pos_split.append((window_pos[0] + window.shape[0] - window_h, window_pos[1] + window.shape[1] - window_w))

    if len(windows_split) > 0:
        for i in range(len(windows_split)):
            win, pos = window_split(windows_split[i], window_pos_split[i])
            window_out = window_out + win
            window_out_pos = window_out_pos + pos

    return window_out, window_out_pos
