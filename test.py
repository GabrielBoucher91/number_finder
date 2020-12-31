import numpy as np
import matplotlib.pyplot as plt
import image_processing as imp

image = imp.get_picture()
imp.display_img(image)
image2 = imp.convert_to_PIL(image)
print(np.array(imp.convert_to_grayscale(image2)).shape)
imp.display_img(imp.convert_to_grayscale(image2))
w = imp.get_window(image, size=(150, 150), pos=(10, 5), channel=3)
imp.display_img(w)