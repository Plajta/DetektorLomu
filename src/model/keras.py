import tensorflow as tf
from tensorflow import keras
import numpy as np
from process import Loader
import matplotlib.pyplot as plt


print("Loading....")
loader = Loader("dataset/test")

loader.generate_dataset(round(loader.get_length()*0.8))
all_data = loader.get_array(0)
train_data = loader.get_array(1)
test_data = loader.get_array(2)
print("Loading done")

print(loader.get(0)[1])

train_dataset = tf.data.Dataset.from_tensor_slices((train_data[:][0], train_data[:][1]))
test_dataset = tf.data.Dataset.from_tensor_slices((train_data[:][0], test_data[:][1]))

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")