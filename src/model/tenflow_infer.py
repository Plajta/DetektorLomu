import tensorflow as tf
from tensorflow import keras
from process import Loader
import cv2

new_model = keras.models.load_model('src/model/saved/NeuralNet/cnn.keras')

load = Loader("dataset/lomy/stepnylom_jpg","dataset/lomy/tvarnylom_jpg")

for i in range(20):
    image = load.get(i)[0]
    xtrain = tf.expand_dims(image/255, axis=-1)
    xtrain = tf.expand_dims(xtrain, axis=0)

    print(new_model.predict(xtrain))

    cv2.imshow('AMOGIS',image)
    cv2.waitKey()