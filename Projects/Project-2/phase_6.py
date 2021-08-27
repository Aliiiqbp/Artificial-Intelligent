import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential


noise = 0.5
# for i in range(6):
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))
x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))

x_train_noisy = x_train + noise * np.random.normal(0.0, noise, x_train.shape)
x_test_noisy = x_test + noise * np.random.normal(0.0, noise, x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0, 1)
x_test_noisy = np.clip(x_test_noisy, 0, 1)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train_noisy, x_train, epochs=1, batch_size=20, shuffle=True, validation_data=(x_test_noisy, x_test))
model.evaluate(x_test_noisy, x_test)
no_noise_img = model.predict(x_test_noisy)

plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(3, 10, i + 1)
    plt.imshow(x_test[i**3].reshape(28, 28), cmap="binary")
    plt.subplot(3, 10, 10 + i + 1)
    plt.imshow(x_test_noisy[i**3].reshape(28, 28), cmap="binary")
    plt.subplot(3, 10, 20 + i + 1)
    plt.imshow(no_noise_img[i**3].reshape(28, 28), cmap="binary")

plt.show()
# noise += 0.5
