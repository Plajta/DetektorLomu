import tensorflow as tf
from tensorflow import keras
from processing import Loader
from keras.utils import plot_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['ROCM_PATH'] = '/opt/rocm' 
#import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use("TkAgg")

BATCH_SIZE = 8
EPOCHS = 20
EARLY_STOPPING = True # Set to False to disable early stopping

print("Loading....")        
loader = Loader("dataset/lomy/stepnylom_jpg","dataset/lomy/tvarnylom_jpg")
loader.randomize()

loader.generate_dataset(500)

train_data = loader.get_array(1)
test_data = loader.get_array(2)

print("Loading done")

test_x = [i[0]/255 for i in test_data]
test_y = [i[1] for i in test_data]

train_x = [i[0]/255 for i in train_data]
train_y = [i[1] for i in train_data]

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))

train_dataset = train_dataset.shuffle(69).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(69).batch(BATCH_SIZE)

#Tensorflow callbacks
filepath = "src/model/saved/CNN/checkpoints/cnn-checkpoint-{epoch:02d}-{val_accuracy:.2f}.keras"
checkpoint_save = tf.keras.callbacks.ModelCheckpoint(filepath, save_weights_only=False, verbose=1, mode='auto', period=1)
if EARLY_STOPPING: early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

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

callback_list = [checkpoint_save,early_stopping] if EARLY_STOPPING else [checkpoint_save]

history = model.fit(train_dataset, epochs=EPOCHS, use_multiprocessing=False, validation_data=test_dataset,
                    callbacks=callback_list)
model.evaluate(test_dataset)

model.save("src/model/saved/CNN/cnn.keras")
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

print("model saved!")

# summarize history for accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()
# summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()

# plt.show()
