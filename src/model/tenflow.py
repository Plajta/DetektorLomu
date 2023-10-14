import tensorflow as tf
from tensorflow import keras
from process import Loader
from wandb.keras import WandbMetricsLogger
from keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

BATCH_SIZE = 8
EPOCHS = 10


print("Loading....")        
loader = Loader("dataset/lomy/stepnylom_jpg","dataset/lomy/tvarnylom_jpg")
loader.randomize()

loader.generate_dataset(500)

train_data = loader.get_array(1)
test_data = loader.get_array(2)

print("Loading done")

test_x = [i[0]/255 for i in test_data]
test_y = [i[1] for i in test_data]

train_dataset = tf.data.Dataset.from_tensor_slices(([i[0]/255 for i in train_data], [i[1] for i in train_data]))
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))

train_dataset = train_dataset.shuffle(69).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(69).batch(BATCH_SIZE)

model = keras.models.Sequential()
model.add(keras.layers.Input(batch_size=BATCH_SIZE,shape=(480,640,1)))
model.add(keras.layers.Conv2D(32, kernel_size=(5, 5), activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(64, kernel_size=(5, 5), activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(128, (3, 3), activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(128, (3, 3), activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
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

history = model.fit(train_dataset, epochs=EPOCHS, use_multiprocessing=False)

model.evaluate(test_dataset)

model.save("src/model/saved/NeuralNet/cnn.keras")
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

print("model saved!")


axis[0].plot(history.history['loss'], label='loss- training data')
#axis[0].plot(history.history['val_accuracy'], label='accuracy - validating data')
axis[0].grid()
axis[0].set_title('Loss')
axis[0].legend()



axis[1].plot(history.history['accuracy'], label='accuracy - training data')
#axis[0].plot(history.history['val_accuracy'], label='accuracy - validating data')
axis[1].grid()
axis[1].set_title('Accuracy')
axis[1].legend()

plt.show()
