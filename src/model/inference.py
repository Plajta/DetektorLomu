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

def infer(X, method, model):
    print(method,model)
    match method:
        case "CNN":
            out = infer_CNN(X, model)
        case "KNNh":
            out = infer_KNN_hist(X, model)
        case "KNNr":
            out = infer_KNN_raw(X, model)
        case "SVM":
            out = infer_SVM(X, model)
    return out

def infer_SVM(X, model_path):
    # load model
    model = pickle.load(open(model_path, "rb"))

    # convert image to the model input
    X = resize(X, (150, 150, 1))
    X = X / 255
    X = X.flatten()
    X = X.reshape(1, -1)

    # run the model
    y_hat = model.predict(X)[0]

    return y_hat
    
def infer_KNN_hist(X, model_path):
    # load model
    model = pickle.load(open(model_path, "rb"))

    # convert image to the model input
    X = extract_color_histogram(X)

    # run the model
    y_hat = model.predict([X])[0]

    return y_hat

def infer_KNN_raw(X, model_path):
    # load model
    model = pickle.load(open(model_path, "rb"))

    # convert image to the model input
    X = image_to_feature_vector(X)

    # run the model
    y_hat = model.predict([X])[0]

    return y_hat

def infer_CNN(X, model_path):
    # load model
    model = keras.models.load_model(model_path)

    # convert image to the model input
    X = tf.expand_dims(X/255, axis=-1)
    X = tf.expand_dims(X, axis=0)

    # run the model
    y_hat = round(model.predict(X)[0][0])

    return y_hat