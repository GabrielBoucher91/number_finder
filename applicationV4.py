import UI_Number_Finder as UI
from PyQt5 import QtWidgets, QtGui, QtCore
import cv2
import numpy as np
import image_processing as imp
import NN_use as NN
import matplotlib.pyplot as plt



class ApplicationV4(QtWidgets.QMainWindow, UI.Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.videoCam = cv2.VideoCapture(1)
        self.model = NN.createNetworkV4()
        self.mainPictureData = None
        self.originalPixmap = None
        self.actualPixmap = None
        self.windowRawData = None
        self.windowProcessedData = None
        self.corner1 = None
        self.corner2 = None
        self.listofboxes = []
        self.listofboxfin = []
        self.list_of_windows = None
        self.list_of_pos = None

        self.takePictureButton.clicked.connect(self.takePicture)
        self.mainPicture.mousePressEvent = self.getCorner1
        self.mainPicture.mouseReleaseEvent = self.getCorner2
        self.updateWindowButton.clicked.connect(self.processWindow)
        self.classifyButton.clicked.connect(self.classifyV2)

    def takePicture(self):
        s, c_img = self.videoCam.read()
        c_img = np.flip(c_img, 2).copy()
        self.mainPictureData = c_img
        img = QtGui.QImage(c_img, c_img.shape[1], c_img.shape[0], QtGui.QImage.Format_RGB888)
        self.mainPicture.setPixmap(QtGui.QPixmap(img))
        self.originalPixmap = QtGui.QPixmap(img)

    def getCorner1(self, event):
        x = event.pos().x()
        y = event.pos().y()
        self.corner1 = (y, x)

    def getCorner2(self, event):
        x = event.pos().x()
        y = event.pos().y()
        self.corner2 = (y, x)
        self.getWindow()

    def getWindow(self):
        size = (abs(self.corner2[0]-self.corner1[0]), abs(self.corner2[1]-self.corner1[1]))
        #print(size)
        position = (min(self.corner1[0], self.corner2[0])+int(size[0]/2), min(self.corner1[1], self.corner2[1])+int(size[1]/2))
        #print(position)
        self.windowRawData = imp.get_window(self.mainPictureData, size=size, pos=position, channel=3)
        windowPIL = imp.convert_to_PIL(self.windowRawData/255.0)
        windowPIL = imp.resize_image(windowPIL, (261, 211))
        windowDisplay = np.asarray(windowPIL).astype(np.uint8)
        totalBytes = windowDisplay.nbytes
        bytesPerLine = windowDisplay.shape[1]*3
        img = QtGui.QImage(windowDisplay, windowDisplay.shape[1], windowDisplay.shape[0], bytesPerLine, QtGui.QImage.Format_RGB888)
        self.windowRaw.setPixmap(QtGui.QPixmap(img))

        self.actualPixmap = self.originalPixmap.copy()
        painterInstance = QtGui.QPainter(self.actualPixmap)
        rectangle = QtGui.QPen(QtCore.Qt.green)
        rectangle.setWidth(3)
        painterInstance.setPen(rectangle)
        painterInstance.drawRect(min(self.corner1[1], self.corner2[1]), min(self.corner1[0], self.corner2[0]), size[1], size[0])
        self.mainPicture.setPixmap(self.actualPixmap)

    def processWindow(self):
        self.windowSplit()
        windowPIL = imp.convert_to_PIL(self.windowRawData/255.0)
        windowGrayscale = (1-np.asarray(imp.convert_to_grayscale(windowPIL)))
        windowGrayscale = imp.increase_contrast(windowGrayscale)*255
        windowGrayscale = imp.clean_backgroundV3(windowGrayscale, self.thresholdMult.value())
        self.windowProcessedData = windowGrayscale
        windowPIL = imp.convert_to_PIL(windowGrayscale)
        windowPIL = imp.resize_image(windowPIL, (261, 211))
        windowDisplay = np.asarray(windowPIL)*255
        bytesPerLine = windowDisplay.shape[1]
        img = QtGui.QImage(windowDisplay, windowDisplay.shape[1], windowDisplay.shape[0], bytesPerLine, QtGui.QImage.Format_Grayscale8)
        self.windowProcessed.setPixmap(QtGui.QPixmap(img))
        for i in range(len(self.list_of_windows)):
            self.list_of_windows[i] = self.processWindowInput(self.list_of_windows[i])

    def processWindowInput(self, window):
        windowPIL = imp.convert_to_PIL(window/255.0)
        windowGrayscale = (1-np.asarray(imp.convert_to_grayscale(windowPIL)))
        windowGrayscale = imp.increase_contrast(windowGrayscale)*255
        windowGrayscale = imp.clean_backgroundV3(windowGrayscale, self.thresholdMult.value())
        return windowGrayscale

    def classify(self):
        self.listofboxes = []
        contours, hier = cv2.findContours(self.windowProcessedData, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
        #cv2.imshow("contours", self.windowProcessedData)
        #cv2.waitKey(0)
        d = 0
        for ctr in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(ctr)
            # Getting ROI
            roi = self.windowProcessedData[y:y + h, x:x + w]
            if roi.shape[0] > 10 and roi.shape[1] > 10:
                #cv2.imshow('character: %d' % d, roi)
                self.listofboxes.append([y, x, w, h, roi])
                # cv2.imwrite('character_%d.png'%d, roi)
                #cv2.waitKey(0)
            cv2.destroyAllWindows()
            d += 1

        self.actualPixmap = self.originalPixmap.copy()
        painterInstance = QtGui.QPainter(self.actualPixmap)
        rectangle = QtGui.QPen(QtCore.Qt.blue)
        rectangle.setWidth(3)
        painterInstance.setPen(rectangle)
        for box in self.listofboxes:
            imPIL = imp.convert_to_PIL(np.pad(box[4], pad_width=4))
            imPIL = imp.resize_image(imPIL)
            nnInput = np.asarray(imPIL)/1.0
            #plt.figure()
            #plt.imshow(nnInput)
            nnInput = np.expand_dims(nnInput, [0, -1])
            out = self.model.predict(nnInput)
            if out[0, np.argmax(out[0])] > 0.6:
                painterInstance.drawRect(min(self.corner1[1], self.corner2[1]) + int(box[1]),
                                        min(self.corner1[0], self.corner2[0]) + int(box[0]), box[2], box[3])
                painterInstance.drawText(min(self.corner1[1], self.corner2[1]) + int(box[1]),
                                        min(self.corner1[0], self.corner2[0]) + int(box[0]), str(np.argmax(out[0])))
            print(out)
            print(np.argmax(out[0]))
        self.mainPicture.setPixmap(self.actualPixmap)

    def windowSplit(self):
        self.list_of_windows, self.list_of_pos = imp.window_split(self.windowRawData, (0, 0))

    def classifyV2(self):
        self.listofboxes = []
        self.listofboxfin = []
        padding = [[0,0],[0,0]]
        for i in range(len(self.list_of_windows)):
            contours, hier = cv2.findContours(self.list_of_windows[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
            d = 0
            for ctr in contours:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(ctr)
                # Getting ROI
                roi = self.list_of_windows[i][y:y + h, x:x + w]
                if roi.shape[0] > 10 and roi.shape[1] > 2:
                    #plt.figure()
                    #plt.imshow(roi)
                    # cv2.imshow('character: %d' % d, roi)
                    self.listofboxes.append([y+self.list_of_pos[i][0], x+self.list_of_pos[i][1], w, h, roi])
                    # cv2.imwrite('character_%d.png'%d, roi)
                    # cv2.waitKey(0)
                cv2.destroyAllWindows()
                d += 1
        #self.actualPixmap = self.originalPixmap.copy()
        #painterInstance = QtGui.QPainter(self.actualPixmap)
        #rectangle = QtGui.QPen(QtCore.Qt.blue)
        #rectangle.setWidth(3)
        #painterInstance.setPen(rectangle)
        for box in self.listofboxes:
            max_size = (max(box[4].shape[0], box[4].shape[1]), np.argmax(box[4].shape))
            min_size = (min(box[4].shape[0], box[4].shape[1]), np.argmin(box[4].shape))
            padding_diff = int((max_size[0]-min_size[0])/2)
            padding[max_size[1]][0] = 6
            padding[max_size[1]][1] = 6
            padding[min_size[1]][0] = 6 + padding_diff
            padding[min_size[1]][1] = 6 + padding_diff
            padded_box = np.pad(box[4], padding)
            imPIL = imp.convert_to_PIL(padded_box)
            imPIL = imp.resize_image(imPIL)
            nnInput = np.asarray(imPIL) / 1.0
            #plt.figure()
            #plt.imshow(nnInput)
            nnInput = np.expand_dims(nnInput, [0, -1])
            out = self.model.predict(nnInput)
            if out[0, np.argmax(out[0])] > 0.6:
                self.listofboxfin.append([box, np.argmax(out[0]), out[0, np.argmax(out[0])]])
                #painterInstance.drawRect(min(self.corner1[1], self.corner2[1]) + int(box[1]),
                #                         min(self.corner1[0], self.corner2[0]) + int(box[0]), box[2], box[3])
                #painterInstance.drawText(min(self.corner1[1], self.corner2[1]) + int(box[1]),
                #                         min(self.corner1[0], self.corner2[0]) + int(box[0]), str(np.argmax(out[0])) + " " + str(out[0, np.argmax(out[0])]))
        self.removeDouble()
            #print(out)
            #print(np.argmax(out[0]))
        #self.mainPicture.setPixmap(self.actualPixmap)
        #plt.show()
    def removeDouble(self):
        max_confidence = 0
        temp_list = self.listofboxfin.copy()
        #print("List length: " + str(len(temp_list)))
        for i in range(len(temp_list)):
            if temp_list[i] != None:
                for j in range(len(temp_list)):
                    if i != j and temp_list[i] != None and temp_list[j] != None:
                        ioveru = imp.intersectionOverUnion((temp_list[i][0][1], temp_list[i][0][0]),
                                                           (temp_list[i][0][3], temp_list[i][0][2]),
                                                           (temp_list[j][0][1], temp_list[j][0][0]),
                                                           (temp_list[j][0][3], temp_list[j][0][2]))

                        #print(str(ioveru) + " " + str(temp_list[i][1]) + " " + str(temp_list[j][1]))
                        if ioveru > 0.0:
                            if temp_list[i][0][4].shape[1] > temp_list[j][0][4].shape[1]:
                                temp_list[j] = None
                            else:
                                temp_list[i] = None
        self.actualPixmap = self.originalPixmap.copy()
        painterInstance = QtGui.QPainter(self.actualPixmap)
        rectangle = QtGui.QPen(QtCore.Qt.blue)
        rectangle.setWidth(3)
        painterInstance.setPen(rectangle)
        for box in temp_list:
            if box is None:
                pass
            else:
                painterInstance.drawRect(min(self.corner1[1], self.corner2[1]) + int(box[0][1]),
                                         min(self.corner1[0], self.corner2[0]) + int(box[0][0]), box[0][2], box[0][3])
                painterInstance.drawText(min(self.corner1[1], self.corner2[1]) + int(box[0][1]),
                                         min(self.corner1[0], self.corner2[0]) + int(box[0][0]),
                                         str(box[1]))
        self.mainPicture.setPixmap(self.actualPixmap)
        #print(temp_list)
