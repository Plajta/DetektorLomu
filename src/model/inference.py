import pickle
import tensorflow as tf
from tensorflow import keras
import cv2

def infer_SVM(X):
    # load model
    model = pickle.load(open("/saved/SVM/SVC.pickle", "rb"))

    # you can use loaded model to compute predictions
    y_hat = model.predict(X)

def infer_CNN(X):
    model = keras.models.load_model('src/model/saved/NeuralNet/cnn_unvalidated.keras')

    X = tf.expand_dims(X/255, axis=-1)
    X = tf.expand_dims(X, axis=0)

    y_hat = model.predict(X)
    print(y_hat)
    return round(y_hat)

loader = Loader("dataset/test/stepnylom","dataset/test/tvarnylom")
loader.generate_dataset(0)
for i in range (10):
    imag,label = loader.get(i,2)
    print(label)
    y_hat = infer_CNN(imag)
    cv2.imwrite(f'test/{i}_{label}_{y_hat}.jpg',imag)
    if y_hat == 0:
        return "štěpný lom"
    elif y_hat == 1:
        return "tvárný lom"
