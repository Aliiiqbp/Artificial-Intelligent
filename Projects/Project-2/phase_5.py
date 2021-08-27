import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))

train_size = 0.8
# for i in range(10):
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print("\n")
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

model = Sequential()
model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10, activation=tf.nn.softmax))
print("########## AdaDelta Optimizer ##########\n")

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x=x_train, y=y_train, epochs=1)
model.evaluate(x_test, y_test)

# print('################## Train size was %f ##################' % train_size)
# train_size -= 0.05

image_index = 1
for i in range(1, 4, 1):
    plt.imshow(x_test[image_index * i].reshape(28, 28), cmap='Greys')
    plt.show()
    pred = model.predict(x_test[image_index * i].reshape(1, 28, 28, 1))
    print("The digit is: ", pred.argmax())
