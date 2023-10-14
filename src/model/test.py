import cv2
from process import Loader
import tensorflow as tf
import numpy as np
from tensorflow import keras

ld = Loader("/home/andry/HACKHAHAHAHAH/plajta/dataset/lomy/stepnylom_jpg", "/home/andry/HACKHAHAHAHAH/plajta/dataset/lomy/tvarnylom_jpg")

howmanydata = 128

trainData = [[], []]
for i in range(howmanydata):
    img, label = ld.get(i)  # Get image and label from Loader
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    trainData[0].append((thresh1 / 255).reshape(480 * 640, ))  # Reshape the data
    trainData[1].append(label)

# Assuming ld.get_length() is the total number of samples
for i in range(howmanydata, ld.get_length()):  # Start from howmanydata
    img, label = ld.get(i)
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    trainData[0].append((thresh1 / 255).reshape(480 * 640, ))
    trainData[1].append(label)



testData = [[], []]
for i in range(howmanydata, 2*howmanydata):
    img, label = ld.get(i)  # Get image and label from Loader
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    testData[0].append((thresh1 / 255).reshape(480 * 640, ))  # Reshape the data
    testData[1].append(label)

# Assuming ld.get_length() is the total number of samples
for i in range(ld.get_length()-(2*howmanydata), ld.get_length()-howmanydata):  # Start from howmanydata
    img, label = ld.get(i)
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    testData[0].append((thresh1 / 255).reshape(480 * 640, ))
    testData[1].append(label)

# Convert trainData to numpy arrays
trainData[0] = np.array(trainData[0])
trainData[1] = np.array(trainData[1])

testData[0] = np.array(testData[0])
testData[1] = np.array(testData[1])

model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=(480 * 640,)))
model.add(keras.layers.Dense(units=64, activation='relu'))
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(units=64, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(trainData[0], 
                    trainData[1], 
                    batch_size=16, 
                    steps_per_epoch=len(trainData[0]) // 16,  
                    epochs=10, 
                    validation_data=(testData[0],testData[1])
                    )
model.save('my_model.keras')