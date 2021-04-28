Number Finder
Author: Gabriel Boucher

Number Finder is an application used to find hand written numbers on a white or light background.
Disclaimer: This application might not be completely stable as it's mostly a test to see if I could use the MNIST
dataset for an actual application. There is no error handling yet and things might crash. It works if the proper steps
are followed.

## Dependencies
Numpy
Tensorflow 2
PyQT5
OpenCV
Matplotlib

#Usage


Connect a webcam before starting the program

Launch it using the main.py script

Once launched, use the Take Picture button to take a picture with your webcam. You can retry as many time as you want.

Once a suitable picture has been taken, use your mouse to click and drag over the area where the numbers are.

When the window has been created, click the Process Window button. You can use the box next to the classify button to adjust
the processing (default value is 1,6). Higher value will thin out the lines, lower values will make them thicker

Once the processing is done, press the Classify button to find the number and classify them.

If another picture is taken, it erases the current one and the steps have to be done again.




