
# import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle
import cv2
from processing import Loader


Categories = ['stepnylom_jpg', 'tvarnylom_jpg']
flat_data_arr = []  # input array
target_arr = []  # output array
datadir = '../dataset/lomy/'
neighbors = 5
jobs = 1


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


print("[INFO] loading data set...")
ld = Loader(datadir+Categories[0], datadir+Categories[1])
rawImages = []
features = []
labels = []
print("[INFO] preparing data...")

for i in ld.get_array():
    image = i[0]
    label = i[1]

    pixels = image_to_feature_vector(image)
    hist = extract_color_histogram(image)

    rawImages.append(pixels)
    features.append(hist)
    labels.append(label)

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
(trainRI, testRI, trainRL, testRL) = train_test_split(
    rawImages, labels, test_size=0.20)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
    features, labels, test_size=0.20)

# train and evaluate a k-NN classifer on the raw pixel intensities
print("[INFO] evaluating raw pixel accuracy...")
modelR = KNeighborsClassifier(n_neighbors=neighbors, n_jobs=jobs)
modelR.fit(trainRI, trainRL)
predRL = modelR.predict(testRI)
acc = accuracy_score(predRL, testRL)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

print(classification_report(testRL, predRL, target_names=Categories))

# train and evaluate a k-NN classifer on the histogram representations
print("[INFO] evaluating histogram accuracy...")
modelH = KNeighborsClassifier(n_neighbors=neighbors, n_jobs=jobs)
modelH.fit(trainFeat, trainLabels)
predLabels = modelH.predict(testFeat)
acc = accuracy_score(predLabels, testLabels)
print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))

print(classification_report(testLabels, predLabels, target_names=Categories))

pickle.dump(modelR, open("model/saved/KNN/KNNr.pickle", "wb"))
pickle.dump(modelH, open("model/saved/KNN/KNNh.pickle", "wb"))

