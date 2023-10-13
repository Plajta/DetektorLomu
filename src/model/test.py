import cv2 
from process import Loader
import tensorflow as tf
import numpy as np
from tensorflow import keras

ld = Loader("/home/andry/HACKHAHAHAHAH/plajta/dataset/lomy/stepnylom_jpg","/home/andry/HACKHAHAHAHAH/plajta/dataset/lomy/tvarnylom_jpg")
ld.randomize()

trainData = [[],[]]
for i in range(40):
    trainData[0].append(ld.get(i)[0]/255.0)
    trainData[1].append(ld.get(i)[1])
testData = [[],[]]
for i in range(40, 80):
    testData[0].append(ld.get(i)[0]/255.0)
    testData[1].append(ld.get(i)[1])

y_tr = np.array(trainData[1])
y_test = np.array(testData[1])

x_train = np.array(trainData[0])
x_train = x_train.reshape(20, 480*640)
x_test = np.array(testData[0])
x_test = x_test.reshape(20, 480*640)

model = keras.models.Sequential()
model.add(keras.layers.Dense(units=32, input_shape=(480*640,), activation='relu'))
model.add(keras.layers.Dense(units=32, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_tr, epochs=10, batch_size=4, validation_split=0.1, validation_data=(x_test,y_tr))
