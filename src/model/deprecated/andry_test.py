import cv2
from processing import Loader
import tensorflow as tf
import numpy as np
from tensorflow import keras

ld = Loader("/home/andry/HACKHAHAHAHAH/plajta/dataset/lomy/stepnylom_jpg", "/home/andry/HACKHAHAHAHAH/plajta/dataset/lomy/tvarnylom_jpg")
ld.generate_dataset(60)



trainData = [[], []]
for item in ld.get_array(1):
    img, label = item
    trainData[0].append(img / 255)
    trainData[1].append(label)

testData = [[], []]
for item in ld.get_array(2):
    img, label = item 
    testData[0].append(img / 255)
    testData[1].append(label)


trainData[0] = np.array(trainData[0])
trainData[1] = np.array(trainData[1])

testData[0] = np.array(testData[0])
testData[1] = np.array(testData[1])


model = keras.models.Sequential()
model.add(keras.layers.Input(shape=(480, 640, 1)))
model.add(keras.layers.Conv2D(16, kernel_size=(2, 2), activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(keras.layers.Flatten())  # Flatten the output
model.add(keras.layers.Dense(1, activation="sigmoid"))  # Output a single value

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(trainData[0], 
                    trainData[1], 
                    batch_size=64, 
                    steps_per_epoch=len(trainData[0]) // 16,  
                    epochs=10, 
                    #validation_data=(testData[0],testData[1])
                    )

test_loss, test_accuracy = model.evaluate(testData[0], testData[1])

# Print the test loss and accuracy
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


#model.save('my_model.keras')