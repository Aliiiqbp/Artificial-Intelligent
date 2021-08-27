
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
import time

# gm = 0.1
# div = 0.1
# train_ac, test_ac, tim = [], [], []
# for i in range(10):
start_time = time.time()
vehicle_registration_plate = pd.read_csv("file.csv")
x = vehicle_registration_plate.drop('y', axis=1).values.astype('float32') / 255
y = vehicle_registration_plate.y.values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=5)
model = SVC(kernel='rbf', C=20, gamma=0.1)
model.fit(x_train, y_train)
y_pred_train = model.predict(x_train)
y_pred = model.predict(x_test)
print("training accuracy: ", metrics.accuracy_score(y_train, y_pred_train))
print("Test accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("--- %s seconds ---" % (time.time() - start_time))


# div += 0.05
# gm += 0.1
# train_ac.append(metrics.accuracy_score(y_train, y_pred_train))
# test_ac.append(metrics.accuracy_score(y_test, y_pred))
# tim.append(time.time() - start_time)


# cnt_tr = [0.1 * i for i in range(1, 11)]
# plt.figure()
# plt.plot(cnt_tr, train_ac, 'x')
# plt.xlabel("gamma")
# plt.ylabel("train accuracy")
# plt.show()
#
# plt.figure()
# plt.plot(cnt_tr, test_ac, 'x')
# plt.xlabel("gamma")
# plt.ylabel("test accuracy")
# plt.show()
#
# plt.figure()
# plt.plot(cnt_tr, tim, 'x')
# plt.xlabel("gamma")
# plt.ylabel("time")
# plt.show()
#
