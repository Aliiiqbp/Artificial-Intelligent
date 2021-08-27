import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from tkinter import *
from tkinter.ttk import *

#################################################


def gui():
    window = Tk()
    window.geometry("600x400")

    window.title("GUI")

    def first_button():
        global equation
        global start_domain, end_domain
        global step
        global test_size
        global hidden_layer_sizes
        global iteration
        global noise

        start_domain = float(start_point_entry.get())
        end_domain = float(end_point_entry.get())
        step = float(step_entry.get())
        noise = float(noise_entry.get())
        iteration = int(epochs_entry.get())
        equation = str(combo.get())
        test_size = float(test_size_entry.get())
        hidden_layer_sizes = str(hidden_layer_sizes_entry.get()).split(" ")
        hidden_layer_sizes = list(map(int, hidden_layer_sizes))

        window.destroy()

    button = Button(window, text="Run", width=10, command=first_button)
    button.pack()
    button.place(x=200, y=360)

    start_point_default = Label(window, text="Default")
    start_point_default.pack()
    start_point_default.place(x=450, y=10)

    start_point_entry = Entry(window)
    start_point_entry.pack()
    start_point_entry.place(x=200, y=80)
    start_point_label = Label(window, text="Start Domain : ")
    start_point_label.pack()
    start_point_label.place(x=50, y=80)
    start_point_default = Label(window, text="-1")
    start_point_default.pack()
    start_point_default.place(x=450, y=80)

    end_point_entry = Entry(window)
    end_point_entry.pack()
    end_point_entry.place(x=200, y=120)
    end_point_label = Label(window, text="End Domain : ")
    end_point_label.pack()
    end_point_label.place(x=50, y=120)
    end_point_default = Label(window, text="1")
    end_point_default.pack()
    end_point_default.place(x=450, y=120)

    step_entry = Entry(window)
    step_entry.pack()
    step_entry.place(x=200, y=160)
    step_label = Label(window, text="Step : ")
    step_label.pack()
    step_label.place(x=50, y=160)
    step_label_default = Label(window, text="0.001")
    step_label_default.pack()
    step_label_default.place(x=450, y=160)

    noise_entry = Entry(window)
    noise_entry.pack()
    noise_entry.place(x=200, y=200)
    noise_entry_label = Label(window, text="Noise : ")
    noise_entry_label.pack()
    noise_entry_label.place(x=50, y=200)
    noise_entry_default = Label(window, text="0")
    noise_entry_default.pack()
    noise_entry_default.place(x=450, y=200)

    epochs_entry = Entry(window)
    epochs_entry.pack()
    epochs_entry.place(x=200, y=240)
    epochs_entry_label = Label(window, text="Iteration : ")
    epochs_entry_label.pack()
    epochs_entry_label.place(x=50, y=240)
    epochs_entry_default = Label(window, text="2000")
    epochs_entry_default.pack()
    epochs_entry_default.place(x=450, y=240)

    hidden_layer_sizes_entry = Entry(window)
    hidden_layer_sizes_entry.pack()
    hidden_layer_sizes_entry.place(x=200, y=280)
    hidden_layer_sizes_entry_label = Label(window, text="Hidden Layer List : ")
    hidden_layer_sizes_entry_label.pack()
    hidden_layer_sizes_entry_label.place(x=50, y=280)
    hidden_layer_sizes_entry_default = Label(window, text="20 40 50 30")
    hidden_layer_sizes_entry_default.pack()
    hidden_layer_sizes_entry_default.place(x=450, y=280)

    test_size_entry = Entry(window)
    test_size_entry.pack()
    test_size_entry.place(x=200, y=320)
    batch_size_label = Label(window, text="test_size : ")
    batch_size_label.pack()
    batch_size_label.place(x=50, y=320)
    batch_size_default = Label(window, text="0.2")
    batch_size_default.pack()
    batch_size_default.place(x=450, y=320)

    combo = Combobox(window)
    combo['values'] = ('', "x ** 2 - 2 * x + 1", 'np.sin(np.pi * x)', "np.sin(2*np.pi*x) + np.sin(5*np.pi*x) + x")
    combo.current(0)
    combo.pack()
    combo.place(x=200, y=40)
    function_entry_label = Label(window, text="Choose Function : ")
    function_entry_label.pack()
    function_entry_label.place(x=50, y=40)
    function_entry_default = Label(window, text="x ** 2")
    function_entry_default.pack()
    function_entry_default.place(x=450, y=40)
    mainloop()

# ls = [5, 10, 10, 5]
# er, score, tim = [], [], []
# for i in range(30):
# noise = 0
# cnt = []


equation = 'x ** 2'
start_domain, end_domain = -1, 1
step = 0.001
test_size = 0.2
hidden_layer_sizes = [20, 40, 50, 30]
iteration = 2000
noise = 0

gui()
f = eval('lambda x: ' + equation)
#################################################

start_time = time.time()
x_val = np.arange(start_domain, end_domain, step)
y_val = f(x_val)

x_train, x_test, y_train, y_test = train_test_split(x_val, y_val, test_size=test_size)
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
y_train = y_train + np.random.normal(0, noise, y_train.shape)

mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=iteration)
mlp.fit(x_train, y_train)
predictions = mlp.predict(x_test)

print(mean_squared_error(y_test, predictions))
print(mlp.score(x_test, y_test))
print("--- %s seconds ---" % (time.time() - start_time))


# tim.append(time.time() - start_time)
# er.append(mean_squared_error(y_test, predictions))
# score.append(mlp.score(x_test, y_test))
# noise += 0.1

# print(sum(er) / len(er))
# print(sum(score) / len(score))
# print(sum(tim) / len(tim))

#################################################

plt.figure()
plt.plot(x_train, y_train, 'o')
plt.plot(x_test, y_test, 'x')
plt.show()

plt.plot(x_test, y_test, 'x')
plt.plot(x_test, predictions, 'ro')
plt.show()

# tim.append(time.time() - start_time)
# cnt.append(tmp)
# ac.append(mlp.score(x_test, y_test))
# er.append(mean_squared_error(y_test, predictions))
# tmp += 0.01
#
# cnt = [i for i in range(0, 20)]
# plt.figure()
# plt.plot(cnt, tim, 'o')
# plt.xlabel("noise")
# plt.ylabel("time")
# plt.show()
#
# plt.figure()
# plt.plot(cnt, er, 'x')
# plt.xlabel("noise")
# plt.ylabel("error")
# plt.show()
#
# plt.figure()
# plt.plot(cnt, score, 'ro')
# plt.xlabel("noise")
# plt.ylabel("score")
# plt.show()
