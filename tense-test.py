from datetime import datetime

import keras as keras
import tensorflow as tf

print("is gpu available ",tf.config.list_physical_devices('GPU'))
print("List : ",tf.config.experimental.list_physical_devices())
print("is with Cuda " ,tf.test.is_built_with_cuda())
print("test gpu device name ", tf.test.gpu_device_name())


from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# checking images shape
print(X_train.shape, X_test.shape)

# display single image shape
print(X_train[0].shape)

# checking labels
print(y_train[:5])

# time
import time
start_time = time.time()

# scaling image values between 0-1
X_train_scaled = X_train/255
X_test_scaled = X_test/255

# one hot encoding labels
y_train_encoded = keras.utils.to_categorical(y_train, num_classes = 10, dtype = 'float32')
y_test_encoded = keras.utils.to_categorical(y_test, num_classes = 10, dtype = 'float32')

def get_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(32,32,3)),
        keras.layers.Dense(3000, activation='relu'),
        keras.layers.Dense(1000, activation='relu'),
        keras.layers.Dense(10, activation='sigmoid')
    ])
    model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

get_model().summary()

# model training
model = get_model()

# Create a TensorBoard callback
logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '500,520')


model.fit(X_train_scaled, y_train_encoded, epochs=10, batch_size=32, callbacks = [tboard_callback], validation_data=(X_test_scaled, y_test_encoded))


# model evaluation
test_loss, test_acc = model.evaluate(X_test_scaled, y_test_encoded)
print('Test accuracy:', test_acc)


# model prediction
predictions = model.predict(X_test_scaled)
print(predictions[0])
print(np.argmax(predictions[0]))
print(y_test[0])


# model saving
model.save('model.h5')


end_time = time.time()
print("Time taken to build model : ", end_time - start_time)
