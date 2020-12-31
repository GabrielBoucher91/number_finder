"""
Main file for the project

The steps are as follow:
1. Load the trained NN
2. Capture an image from the webcam
3. Pre-process the image so that it is in the right format to be scanned
4. Scan the image at each pixel
5. For each scan, send it through the network.
6. Retrieve the output and log it in the heatmap.
"""