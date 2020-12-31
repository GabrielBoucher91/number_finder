"""
This file is used for the NN functions when using the network to evaluation.
It also includes the heatmap class.
"""
import numpy as np


def createNetwork():
    pass


class NumberHeatmap():
    def __init__(self, windowSize, size):
        self.__windowSize = windowSize
        self.__size = size
        self.__heatmap = np.zeros(size)

    def updateHeatmap(self, position=(0, 0), value=0):
        self.__heatmap[position] = value

    def get_heatmapValue(self, position=(0, 0)):
        return self.__heatmap[position]

    def get_size(self):
        return self.__size

    def get_windwoSize(self):
        return self.__windowSize


def extractValueForHeatmap(netOutput, threshold):
    pass
