import cv2
from process import Loader
import tensorflow as tf
import numpy as np
from tensorflow import keras

ld = Loader("/home/andry/HACKHAHAHAHAH/plajta/dataset/lomy/stepnylom_jpg", "/home/andry/HACKHAHAHAHAH/plajta/dataset/lomy/tvarnylom_jpg")
ld.randomize()

howmanydata = 30

trainData = [[], []]
for i in range(howmanydata):
    trainData[0].append(ld.get(i)[0] / 255.0)
    trainData[1].append(ld.get(i)[1])
testData = [[], []]
for i in range(howmanydata, 2 * howmanydata):
    testData[0].append(ld.get(i)[0] / 255.0)
    testData[1].append(ld.get(i)[1])

y_tr = np.array(trainData[1])
y_test = np.array(testData[1])

x_train = np.array(trainData[0])
x_test = np.array(testData[0])

model = keras.Sequential()
model.add(keras.Input(shape=(480, 640, 1)))
model.add(keras.layers.Conv2D(8, 15, activation="relu"))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Conv2D(32, 5, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(x_train, y_tr, batch_size=16, epochs=10)
