import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


test_size = 0.25
hidden_layer_sizes = [30, 40, 80, 120, 80, 60]
iteration = 2000

start_time = time.time()
x_val = np.array([0.8, 0.8, 1.2, 2.2, 3.8, 4.6, 5.6, 6.1, 6.4, 6.9, 7.5, 8.9, 10.6, 11.7, 12.9, 14.1, 14.8, 15.7, 16.4, 16.8, 17.6, 18.4, 18.9, 19.9, 21.1, 22.0, 22.9, 23.4, 24.1, 24.6, 25.2, 25.9, 26.4, 27.2, 27.5, 28.0, 28.7, 29.7, 30.7, 33.3, 36.7, 39.9, 40.6, 40.6, 40.6, 40.6, 40.6, 40.6, 40.8, 42.0, 43.1, 44.7, 46.3, 47.9, 49.0, 50.4, 51.2, 52.1, 53.0, 53.8, 54.3, 54.8, 55.3, 55.6, 56.0, 56.7, 57.1, 57.1, 57.2, 57.7, 58.7, 59.3, 59.9, 60.3, 60.7, 61.1, 61.8, 62.6, 63.3, 63.6, 64.1, 65.0, 65.8, 66.3, 67.0, 67.3, 67.6, 67.8, 68.1, 68.3])
y_val = np.array([-0.0, -0.3, -1.1, -2.4, -4.8, -5.8, -6.5, -6.5, -6.5, -6.1, -4.7, -0.6, 3.9, 6.7, 9.4, 10.7, 11.3, 11.7, 11.8, 11.8, 11.8, 11.8, 11.8, 11.0, 9.6, 7.8, 4.5, 2.5, -1.5, -5.9, -9.4, -12.0, -13.2, -14.1, -14.3, -14.3, -14.3, -14.3, -13.8, -11.6, -7.5, -3.1, 0.6, 2.8, 6.5, 10.0, 13.1, 13.2, 12.9, 11.4, 9.2, 6.3, 3.7, 1.2, -0.0, -0.8, -1.2, -1.8, -2.7, -3.8, -4.6, -5.4, -6.0, -6.4, -6.9, -8.1, -8.5, -8.5, -8.5, -8.5, -8.8, -9.1, -9.4, -9.6, -9.8, -10.0, -10.2, -10.2, -10.2, -10.0, -9.4, -8.4, -7.4, -6.8, -6.1, -5.5, -5.2, -5.0, -4.8, -4.8])

x_train, x_test, y_train, y_test = train_test_split(x_val, y_val, test_size=test_size)
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
y_train = y_train + np.random.normal(0, 0, y_train.shape)

mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=iteration)
mlp.fit(x_train, y_train)
predictions = mlp.predict(x_test)

print(mean_squared_error(y_test, predictions))
print(mlp.score(x_test, y_test))
print("--- %s seconds ---" % (time.time() - start_time))

plt.figure()
plt.plot(x_train, y_train, 'o')
plt.plot(x_test, y_test, 'x')
plt.show()

plt.plot(x_test, y_test, 'x')
plt.plot(x_test, predictions, 'ro')
plt.show()
