"""
This file is used for the NN functions when using the network to evaluation.
It also includes the heatmap class.
"""
import numpy as np
import tensorflow as tf
import image_processing as impro

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.Dropout1 = tf.keras.layers.Dropout(0.2)
        self.Conv1 = tf.keras.layers.Conv2D(16, 2, padding='valid',activation='relu', input_shape=(1,28,28,1), name='Conv1')
        self.MaxPool1 = tf.keras.layers.MaxPool2D(padding='same')
        self.Dropout2 = tf.keras.layers.Dropout(0.2)
        self.Conv2 = tf.keras.layers.Conv2D(16, 2, padding='valid', activation='relu', input_shape=(1,28,28,1), name='Conv2')
        self.MaxPool2 = tf.keras.layers.MaxPool2D(padding='same')
        self.Dropout3 = tf.keras.layers.Dropout(0.2)
        self.Flat = tf.keras.layers.Flatten()
        self.Fc1 = tf.keras.layers.Dense(256, activation='relu', name='Fc1')
        self.Fc2 = tf.keras.layers.Dense(128, activation='relu', name='Fc2')
        self.Fc3 = tf.keras.layers.Dense(10, activation='softmax', name='Fc3')

    def call(self, inputs):
        x = self.Dropout1(inputs)
        x = self.Conv1(x)
        x = self.MaxPool1(x)
        x = self.Dropout2(x)
        x = self.Conv2(x)
        x = self.MaxPool2(x)
        x = self.Dropout3(x)
        #Flat
        x = self.Flat(x)
        x = self.Fc1(x)
        x = self.Fc2(x)
        output = self.Fc3(x)

        return output

class MyModelV2(tf.keras.Model):
    def __init__(self):
        super(MyModelV2, self).__init__()
        self.Dropout1 = tf.keras.layers.Dropout(0.2)
        self.Conv1 = tf.keras.layers.Conv2D(64, 2, padding='valid', activation='relu', input_shape=(1,28,28,1), name='Conv1')
        self.MaxPool1 = tf.keras.layers.MaxPool2D(padding='same')
        self.Dropout2 = tf.keras.layers.Dropout(0.2)
        self.Conv2 = tf.keras.layers.Conv2D(128, 2, padding='valid', activation='relu', input_shape=(1,28,28,1), name='Conv2')
        self.MaxPool2 = tf.keras.layers.MaxPool2D(padding='same')
        self.Dropout3 = tf.keras.layers.Dropout(0.2)
        self.Flat = tf.keras.layers.Flatten()
        self.Fc1 = tf.keras.layers.Dense(256, activation='relu', name='Fc1')
        self.Fc2 = tf.keras.layers.Dense(128, activation='relu', name='Fc2')
        self.Fc3 = tf.keras.layers.Dense(10, activation='softmax', name='Fc3')

    def call(self, inputs):
        x = self.Dropout1(inputs)
        x = self.Conv1(x)
        x = self.MaxPool1(x)
        x = self.Dropout2(x)
        x = self.Conv2(x)
        x = self.MaxPool2(x)
        x = self.Dropout3(x)
        #Flat
        x = self.Flat(x)
        x = self.Fc1(x)
        x = self.Fc2(x)
        output = self.Fc3(x)

        return output

class MyModelV3(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.Noise = tf.keras.layers.GaussianNoise(0.2)
        self.Dropout1 = tf.keras.layers.Dropout(0.2)
        self.Conv1 = tf.keras.layers.Conv2D(64, 2, padding='valid', activation='relu', input_shape=(1,28,28,1), name='Conv1')
        self.MaxPool1 = tf.keras.layers.MaxPool2D(padding='same')
        self.Dropout2 = tf.keras.layers.Dropout(0.2)
        self.Conv2 = tf.keras.layers.Conv2D(128, 2, padding='valid', activation='relu', input_shape=(1,28,28,1), name='Conv2')
        self.MaxPool2 = tf.keras.layers.MaxPool2D(padding='same')
        self.Dropout3 = tf.keras.layers.Dropout(0.2)
        self.Flat = tf.keras.layers.Flatten()
        self.Fc1 = tf.keras.layers.Dense(256, activation='relu', name='Fc1')
        self.Fc2 = tf.keras.layers.Dense(128, activation='relu', name='Fc2')
        self.Fc3 = tf.keras.layers.Dense(10, activation='softmax', name='Fc3')

    def call(self, inputs):
        x = self.Noise(inputs)
        x = self.Dropout1(x)
        x = self.Conv1(x)
        x = self.MaxPool1(x)
        x = self.Dropout2(x)
        x = self.Conv2(x)
        x = self.MaxPool2(x)
        x = self.Dropout3(x)
        #Flat
        x = self.Flat(x)
        x = self.Fc1(x)
        x = self.Fc2(x)
        output = self.Fc3(x)

        return output

class MyModelV4(tf.keras.Model):
    def __init__(self):
        super(MyModelV4, self).__init__()
        self.Noise = tf.keras.layers.GaussianNoise(0.2)
        self.Dropout1 = tf.keras.layers.Dropout(0.2)
        self.Conv1 = tf.keras.layers.Conv2D(64, 2, padding='valid', activation='relu', input_shape=(1,28,28,1), name='Conv1')
        self.MaxPool1 = tf.keras.layers.MaxPool2D(padding='same')
        self.Dropout2 = tf.keras.layers.Dropout(0.2)
        self.Conv2 = tf.keras.layers.Conv2D(128, 2, padding='valid', activation='relu', input_shape=(1,28,28,1), name='Conv2')
        self.MaxPool2 = tf.keras.layers.MaxPool2D(padding='same')
        self.Dropout3 = tf.keras.layers.Dropout(0.2)
        self.Flat = tf.keras.layers.Flatten()
        self.Fc1 = tf.keras.layers.Dense(256, activation='relu', name='Fc1')
        self.Fc2 = tf.keras.layers.Dense(128, activation='relu', name='Fc2')
        self.Fc3 = tf.keras.layers.Dense(11, activation='softmax', name='Fc3')

    def call(self, inputs):
        x = self.Noise(inputs)
        x = self.Dropout1(x)
        x = self.Conv1(x)
        x = self.MaxPool1(x)
        x = self.Dropout2(x)
        x = self.Conv2(x)
        x = self.MaxPool2(x)
        x = self.Dropout3(x)
        #Flat
        x = self.Flat(x)
        x = self.Fc1(x)
        x = self.Fc2(x)
        output = self.Fc3(x)

        return output

def createNetwork():
    MyNetwork = MyModelV3()
    MyNetwork.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[tf.metrics.SparseCategoricalAccuracy()])
    MyNetwork.load_weights(filepath="./train_model_2/cp2.ckpt")

    return MyNetwork

def createNetworkV2():
    MyNetwork = MyModelV2()
    MyNetwork.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[tf.metrics.SparseCategoricalAccuracy()])
    MyNetwork.load_weights(filepath="./train_model_aug_2/cpaug2.ckpt")

    return MyNetwork

def createNetworkV3():
    MyNetwork = MyModelV2()
    MyNetwork.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[tf.metrics.SparseCategoricalAccuracy()])
    MyNetwork.load_weights(filepath="./train_model_aug_3/cpaug3.ckpt")

    return MyNetwork

def createNetworkV4():
    MyNetwork = MyModelV4()
    MyNetwork.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[tf.metrics.SparseCategoricalAccuracy()])
    MyNetwork.load_weights(filepath="./train_model_aug_4/cpaug4.ckpt")

    return MyNetwork


class NumberHeatmap():
    def __init__(self, image_size, window_sizes=(28, 32, 40)):
        self.__windowSizes = window_sizes
        self.__size = image_size
        self.__heatmap = np.zeros((image_size[0]-max(window_sizes), image_size[1]-max(window_sizes), 3))

    def get_heatmap(self):
        return self.__heatmap

    def reset_heatmap(self):
        self.__heatmap = np.zeros((self.image_size[0] - max(self.window_sizes), self.image_size[1] - max(self.window_sizes), 3))

    def updateHeatmap(self, position=(0, 0), value=[0, 0, 0]):
        self.__heatmap[position] = value

    def get_heatmapValue(self, position=(0, 0)):
        return self.__heatmap[position]

    def get_size(self):
        return self.__size

    def get_windowSizes(self):
        return self.__windowSizes

    def process_image(self, image, network):
        width = self.__heatmap.shape[0]
        height = self.__heatmap.shape[1]

        start_point_w = max(self.__windowSizes)/2
        end_point_w = start_point_w + width
        start_point_h = max(self.__windowSizes) / 2
        end_point_h = start_point_h + height

        for i in range(width):
            for j in range(height):
                pass
