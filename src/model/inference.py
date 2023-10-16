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
    
def infer_KNN_hist(X, model_path):
    # load model
    model = pickle.load(open(model_path, "rb"))

    X = extract_color_histogram(X)
    #X = X.reshape(-1, 1)
    y_hat = model.predict([X])[0]

    if y_hat == 0:
        return "štěpný lom"
    elif y_hat == 1:
        return "tvárný lom"

def infer_KNN_raw(X, model_path):
    # load model
    model = pickle.load(open(model_path, "rb"))

    X = image_to_feature_vector(X)
    y_hat = model.predict([X])[0]

    if y_hat == 0:
        return "štěpný lom"
    elif y_hat == 1:
        return "tvárný lom"

def infer_CNN(X, model_path):
    model = keras.models.load_model(model_path)

    X = tf.expand_dims(X/255, axis=-1)
    X = tf.expand_dims(X, axis=0)

    y_hat = round(model.predict(X)[0][0])
    print(y_hat)
    if y_hat == 0:
        return "štěpný lom"
    elif y_hat == 1:
        return "tvárný lom"

# loader = Loader("dataset/lomy/stepnylom_jpg","dataset/lomy/tvarnylom_jpg")
# loader.generate_dataset(400)
# chyby = 0
# for i in range (100):
#     imag,label = loader.get(i,2)
#     print(label)
#     y_hat = infer_CNN(imag)
#     cv2.imwrite(f'test/{i}_{label}_{y_hat}.jpg',imag)
#     if label!=y_hat: chyby+=1
# print(chyby)
