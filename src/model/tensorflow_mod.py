import tensorflow as tf
from tensorflow import keras
from processing import Loader
from keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

BATCH_SIZE = 32
EPOCHS = 20

SLICE_FACTOR = 4

print("Loading....")        
loader = Loader("dataset/lomy/stepnylom_jpg","dataset/lomy/tvarnylom_jpg")

loader.generate_dataset(400)

train_data = loader.get_array(1)
test_data = loader.get_array(2)

print("Loading done")

test_x = [i[0]/255 for i in test_data]
test_y = [i[1] for i in test_data]

train_x = [i[0]/255 for i in train_data]
train_y = [i[1] for i in train_data]

fraction_train_x = []
fraction_train_y = []

fraction_test_x = []
fraction_test_y = []

#reformat to ensemble

num_rows = 4
num_cols = 4

for i in range(len(train_x)):
    img = train_x[i]
    height, width = img.shape

    sub_img_height = height // num_rows
    sub_img_width = width // num_cols

    for i in range(num_rows):
        for j in range(num_cols):
            y_start = i * sub_img_height
            y_end = (i + 1) * sub_img_height
            x_start = j * sub_img_width
            x_end = (j + 1) * sub_img_width

            sub_image = img[y_start:y_end, x_start:x_end]

            fraction_train_x.append(sub_image)
            fraction_train_y.append(train_y[i])

for i in range(len(test_x)):
    img = test_x[i]
    height, width = img.shape

    sub_img_height = height // num_rows
    sub_img_width = width // num_cols

    for i in range(num_rows):
        for j in range(num_cols):
            y_start = i * sub_img_height
            y_end = (i + 1) * sub_img_height
            x_start = j * sub_img_width
            x_end = (j + 1) * sub_img_width

            sub_image = img[y_start:y_end, x_start:x_end]

            fraction_test_x.append(sub_image)
            fraction_test_y.append(test_y[i])

print(len(fraction_test_x))
print(len(fraction_train_x))

train_dataset = tf.data.Dataset.from_tensor_slices((fraction_train_x, fraction_train_y))
test_dataset = tf.data.Dataset.from_tensor_slices((fraction_test_x, fraction_test_y))

train_dataset = train_dataset.shuffle(69).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(69).batch(BATCH_SIZE)

model = keras.models.Sequential()
model.add(keras.layers.Input(batch_size=BATCH_SIZE,shape=(120,160,1)))
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(32, (3, 3), activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(1024, activation="relu"))
model.add(tf.keras.layers.BatchNormalization())

model.add(keras.layers.Dense(256, activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(train_dataset, epochs=EPOCHS, use_multiprocessing=False, validation_data=test_dataset)
model.evaluate(test_dataset)

model.save("src/model/saved/NeuralNet/cnn_small.keras")
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

print("model saved!")

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.show()