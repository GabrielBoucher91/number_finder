import UI_Number_Finder as UI
from PyQt5 import QtWidgets, QtGui, QtCore
import cv2
import numpy as np
import image_processing as imp
import NN_use as NN



class ApplicationV2(QtWidgets.QMainWindow, UI.Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.videoCam = cv2.VideoCapture(0)
        self.model = NN.createNetworkV2()
        self.mainPictureData = None
        self.originalPixmap = None
        self.actualPixmap = None
        self.windowRawData = None
        self.windowProcessedData = None

        self.takePictureButton.clicked.connect(self.takePicture)
        self.mainPicture.mousePressEvent = self.getWindowPos
        self.ywindowSize.valueChanged.connect(self.getWindow)
        self.updateWindowButton.clicked.connect(self.processWindow)
        self.classifyButton.clicked.connect(self.classify)

    def takePicture(self):
        s, c_img = self.videoCam.read()
        c_img = np.flip(c_img, 2).copy()
        self.mainPictureData = c_img
        img = QtGui.QImage(c_img, c_img.shape[1], c_img.shape[0], QtGui.QImage.Format_RGB888)
        self.mainPicture.setPixmap(QtGui.QPixmap(img))
        self.originalPixmap = QtGui.QPixmap(img)

    def getWindowPos(self, event):
        x = event.pos().x()
        y = event.pos().y()
        self.xPosition.display(x)
        self.yPosition.display(y)
        self.getWindow()

    def getWindow(self):
        size = (self.ywindowSize.value(), self.ywindowSize.value())
        position = (self.yPosition.value(), self.xPosition.value())
        self.windowRawData = imp.get_window(self.mainPictureData, size=size, pos=position, channel=3)
        windowPIL = imp.convert_to_PIL(self.windowRawData/255.0)
        windowPIL = imp.resize_image(windowPIL, (112, 112))
        windowDisplay = np.asarray(windowPIL)
        img = QtGui.QImage(windowDisplay, windowDisplay.shape[1], windowDisplay.shape[0], QtGui.QImage.Format_RGB888)
        self.windowRaw.setPixmap(QtGui.QPixmap(img))

        self.actualPixmap = self.originalPixmap.copy()
        painterInstance = QtGui.QPainter(self.actualPixmap)
        rectangle = QtGui.QPen(QtCore.Qt.green)
        rectangle.setWidth(3)
        painterInstance.setPen(rectangle)
        painterInstance.drawRect(position[1]-size[0]/2, position[0]-size[0]/2, size[0], size[0])
        self.mainPicture.setPixmap(self.actualPixmap)

    def processWindow(self):
        windowPIL = imp.convert_to_PIL(self.windowRawData/255.0)
        windowGrayscale = (1-np.asarray(imp.convert_to_grayscale(windowPIL)))/255.0
        windowGrayscale = imp.increase_contrast(windowGrayscale)
        #windowGrayscale = imp.clean_backgroundV2(windowGrayscale)
        self.windowProcessedData = windowGrayscale
        windowPIL = imp.convert_to_PIL(windowGrayscale)
        windowPIL = imp.resize_image(windowPIL, (112, 112))
        windowDisplay = np.asarray(windowPIL)
        img = QtGui.QImage(windowDisplay, windowDisplay.shape[1], windowDisplay.shape[0], QtGui.QImage.Format_Grayscale8)
        self.windowProcessed.setPixmap(QtGui.QPixmap(img))

    def classify(self):
        windowPIL = imp.convert_to_PIL(self.windowProcessedData)
        windowPIL = imp.resize_image(windowPIL)
        nump_window = np.asarray(windowPIL)/255.0
        nump_window_in = np.expand_dims(nump_window, [0, -1])
        output = self.model.predict(nump_window_in)
        output = output.round(decimals=3)
        number = np.argmax(output)
        confidance = output[0, number]
        self.numberLCD.display(number)
        self.confidanceLCD.display(confidance*100)
