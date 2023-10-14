import tensorflow as tf
from tensorflow import keras
from process import Loader
import wandb
from wandb.keras import WandbMetricsLogger

BATCH_SIZE = 8
EPOCHS = 2

wandb.init(
    # set the wandb project where this run will be logged
    project="LomyDetection",

    # track hyperparameters and run metadata with wandb.config
    config={
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "metric": "accuracy",
        "epoch": EPOCHS,
        "batch_size": BATCH_SIZE
    }
)


print("Loading....")        
loader = Loader("dataset/lomy/stepnylom_jpg","dataset/lomy/tvarnylom_jpg")

#loader.generate_dataset(220)

train_data = loader.get_array(0)[0:500]
test_data = loader.get_array(0)[501:521]

print("Loading done")

test_x = [i[0]/255 for i in test_data]
test_y = [i[1] for i in test_data]

train_dataset = tf.data.Dataset.from_tensor_slices(([i[0]/255 for i in train_data], [i[1] for i in train_data]))
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))

train_dataset = train_dataset.shuffle(8).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(8).batch(BATCH_SIZE)

model = keras.models.Sequential()
model.add(keras.layers.Input(batch_size=BATCH_SIZE,shape=(480,640,1)))
model.add(keras.layers.Conv2D(64, kernel_size=(5, 5), activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(128, kernel_size=(5, 5), activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_dataset, epochs=EPOCHS, use_multiprocessing=False, callbacks=[WandbMetricsLogger()])

model.evaluate(test_dataset, callbacks=[WandbMetricsLogger()])

model.save("src/model/saved/NeuralNet/cnn.keras")
print("model saved!")

wandb.finish()

# plt.figure(figsize=(10, 10))
# for images, labels in train_dataset.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(int(labels[i]))
#         plt.axis("off")