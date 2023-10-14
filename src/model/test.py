import cv2
from process import Loader
import tensorflow as tf
import numpy as np
from tensorflow import keras

ld = Loader("/home/andry/HACKHAHAHAHAH/plajta/dataset/lomy/stepnylom_jpg", "/home/andry/HACKHAHAHAHAH/plajta/dataset/lomy/tvarnylom_jpg")

howmanydata = 60

trainData = ld.get_array(1)

testData = ld.get_array(2)

print(trainData)

y_tr = np.array(trainData[1])
y_test = np.array(testData[1])

x_train = np.array(trainData[0])
x_train = x_train.reshape(-1, 480 * 640)
x_test = np.array(testData[0])
x_test = x_test.reshape(-1, 480 * 640)

model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=(480 * 640)))
model.add(keras.layers.Dense(units=32, activation='relu'))
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(units=32, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(x_train, y_tr, batch_size=16,  # Using data augmentation
                    steps_per_epoch=len(x_train) / 1,  # Adjust this value
                    epochs=10,  # Increase the number of epochs
                    validation_data=(x_test, y_test))
