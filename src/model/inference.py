import pickle
import tensorflow as tf
from tensorflow import keras
import cv2

import sklearn
from skimage.transform import resize

def infer_SVM(X):
    # load model
    model = pickle.load(open("src/model/saved/SVM/SVC.pickle", "rb"))

    X = resize(X, (150, 150, 1))
    X = X / 255
    X = X.flatten()
    X = X.reshape(1, -1)

    # you can use loaded model to compute predictions
    y_hat = model.predict(X)[0]

    if y_hat == 0:
        return "štěpný lom"
    elif y_hat == 1:
        return "tvárný lom"

def infer_CNN(X):
    model = keras.models.load_model('src/model/saved/NeuralNet/cnn_best.keras')

    X = tf.expand_dims(X/255, axis=-1)
    X = tf.expand_dims(X, axis=0)

    y_hat = round(model.predict(X)[0][0])

    if y_hat == 0:
        return "štěpný lom"
    elif y_hat == 1:
        return "tvárný lom"
    
def infer_ensemble_CNN(X):
    pass

if __name__ == "__main__":
    pass
