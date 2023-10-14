import cv2
import numpy as np
from tensorflow import keras
from process import Loader
ld = Loader("/home/andry/HACKHAHAHAHAH/plajta/dataset/lomy/stepnylom_jpg", "/home/andry/HACKHAHAHAHAH/plajta/dataset/lomy/tvarnylom_jpg")

# Завантажте модель
model = keras.models.load_model('plajta/src/model/my_model.keras')

for i in range(ld.get_length()):
    img, label = ld.get(i)  # Get image and label from Loader
    print("waiting data:", label)
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    data = (thresh1 / 255).reshape(1, 480 * 640)  # Reshape the data

    datatoload = np.array(data)

    predictions = model.predict(datatoload)
    # Визначте клас з найвищою ймовірністю
    predicted_class = np.argmax(predictions)

    print("result:", predicted_class)
