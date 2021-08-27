import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
from tensorflow.keras.datasets import mnist
from sklearn.multiclass import OneVsRestClassifier
import time

train_ac, test_ac, tim = [], [], []
cnt_tr, cnt_ts = 500, 50
for i in range(10):
    start_time = time.time()
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    start = time.time()
    x_train = x_train[:cnt_tr]
    y_train = y_train[:cnt_tr]
    x_test = x_test[:cnt_ts]
    y_test = y_test[:cnt_ts]
    model = OneVsRestClassifier(SVC(kernel='sigmoid', C=1, gamma='scale'), n_jobs=10)
    x_test = x_test.reshape(cnt_ts, 28 * 28)
    x_train = x_train.reshape(cnt_tr, 28 * 28)
    model.fit(x_train, y_train)

    y_pred_train = model.predict(x_train)
    print("train accuracy: ", metrics.accuracy_score(y_train, y_pred_train))
    y_pred = model.predict(x_test)
    print("test accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("--- %s seconds ---" % (time.time() - start_time))

    cnt_tr += 500
    cnt_ts += 50

    train_ac.append(metrics.accuracy_score(y_train, y_pred_train))
    test_ac.append(metrics.accuracy_score(y_test, y_pred))
    tim.append(time.time() - start_time)


cnt_tr = [500 * i for i in range(1, 11)]
plt.figure()
plt.plot(cnt_tr, train_ac, 'x')
plt.xlabel("number of train")
plt.ylabel("train accuracy")
plt.show()

plt.figure()
plt.plot(cnt_tr, test_ac, 'x')
plt.xlabel("number of train")
plt.ylabel("test accuracy")
plt.show()

plt.figure()
plt.plot(cnt_tr, tim, 'x')
plt.xlabel("number of train")
plt.ylabel("time")
plt.show()
