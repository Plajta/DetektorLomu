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

def infer(data, method, model):
    # Make data a list so it still works with passing only one image
    if type(data) is not list: data = [data]
    match method:
        case "CNN":
            out = infer_CNN(data, model)
        case "KNNh":
            out = infer_KNN_hist(data, model)
        case "KNNr":
            out = infer_KNN_raw(data, model)
        case "SVM":
            out = infer_SVM(data, model)
    return out

def infer_SVM(data, model_path):
    # load model
    model = pickle.load(open(model_path, "rb"))

    y_hat = []

    for X in data:
        # convert image to the model input
        X = resize(X, (150, 150, 1))
        X = X / 255
        X = X.flatten()
        X = X.reshape(1, -1)

        # run the model
        y_hat.append(model.predict(X)[0])

    return y_hat
    
def infer_KNN_hist(data, model_path):
    # load model
    model = pickle.load(open(model_path, "rb"))

    y_hat = []

    for X in data:
        # convert image to the model input
        X = extract_color_histogram(X)

        # run the model
        y_hat.append(model.predict([X])[0])

    return y_hat

def infer_KNN_raw(data, model_path):
    # load model
    model = pickle.load(open(model_path, "rb"))
    
    y_hat = []

    for X in data:
        # convert image to the model input
        X = image_to_feature_vector(X)

        # run the model
        y_hat.append(model.predict([X])[0])

    return y_hat

def infer_CNN(data, model_path):  # TODO make a batch inference command
    # load model
    model = keras.models.load_model(model_path)

    y_hat = []

    for X in data:
        # convert image to the model input
        X = tf.expand_dims(X/255, axis=-1)
        X = tf.expand_dims(X, axis=0)

        # run the model
        y_hat.append(round(model.predict(X)[0][0]))

    return y_hat