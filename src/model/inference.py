import pickle
import tensorflow as tf
from tensorflow import keras
import cv2

import sklearn
from skimage.transform import resize

#KNN
def image_to_feature_vector(image, size=(64, 48)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()


def extract_color_histogram(image, bins=(8, 8, 8)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    gray = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(gray, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 128, 0, 256, 0, 256])
    # handle normalizing the histogram if we are using OpenCV > 2.4.X
    cv2.normalize(hist, hist)
    return hist.flatten()

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
    
def infer_KNN_hist(X):
    # load model
    model = pickle.load(open("src/model/saved/KNN/KNNh.pickle", "rb"))

    X = extract_color_histogram(X)
    #X = X.reshape(-1, 1)
    y_hat = model.predict([X])[0]

    if y_hat == 0:
        return "štěpný lom"
    elif y_hat == 1:
        return "tvárný lom"

def infer_KNN_raw(X):
    # load model
    model = pickle.load(open("src/model/saved/KNN/KNNr.pickle", "rb"))

    X = image_to_feature_vector(X)
    y_hat = model.predict([X])[0]

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
    model = keras.models.load_model('src/model/saved/NeuralNet/cnn_best.keras')
