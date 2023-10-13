import tensorflow as tf
from tensorflow import keras
import numpy as np
from process import Loader
import matplotlib.pyplot as plt


print("Loading....")        
loader = Loader("dataset/test")

loader.generate_dataset(round(loader.get_length()*0.8))

train_data = loader.get_array(1)
test_data = loader.get_array(2)

print("Loading done")

train_dataset = tf.data.Dataset.from_tensor_slices(([i[0]/255 for i in train_data], [i[1] for i in train_data]))
test_dataset = tf.data.Dataset.from_tensor_slices(([i[0]/255 for i in test_data], [i[1] for i in test_data]))

BATCH_SIZE = 8

train_dataset = train_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model = keras.models.Sequential()
model.add(keras.layers.Input(batch_size=BATCH_SIZE,shape=(640,480,1)))
model.add(keras.layers.Conv2D(64, kernel_size=(5, 5), activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(128, kernel_size=(5, 5), activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1, activation="sigmoid"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

EPOCHS = 10
history = model.fit(train_dataset, epochs=EPOCHS)

# plt.figure(figsize=(10, 10))
# for images, labels in train_dataset.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(int(labels[i]))
#         plt.axis("off")