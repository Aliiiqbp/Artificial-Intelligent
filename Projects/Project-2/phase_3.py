import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from tkinter import *
from tkinter.ttk import *


def z_1(x, y):
    return np.exp(-(np.square(x) + np.square(y))/4)


def z_2(x, y):
    return 3 * x + x * y - 5 * y


def z_3(x, y):
    return x * y + np.e ** x - np.sin((np.pi / 2) * y) + np.tanh(np.pi * x)


def gui():
    window = Tk()
    window.geometry("400x400")

    window.title("GUI")

    def first_button():
        global equation
        global start_domain, end_domain
        global step
        # global test_size
        global hidden_layer_sizes
        global iteration
        # global noise

        start_domain = float(start_point_entry.get())
        end_domain = float(end_point_entry.get())
        step = float(step_entry.get())
        # noise = float(noise_entry.get())
        iteration = int(epochs_entry.get())
        equation = str(combo.get())
        # test_size = float(test_size_entry.get())
        hidden_layer_sizes = str(hidden_layer_sizes_entry.get()).split(" ")
        hidden_layer_sizes = list(map(int, hidden_layer_sizes))

        window.destroy()

    button = Button(window, text="Run", width=10, command=first_button)
    button.pack()
    button.place(x=200, y=360)

    # start_point_default = Label(window, text="Default")
    # start_point_default.pack()
    # start_point_default.place(x=450, y=10)

    start_point_entry = Entry(window)
    start_point_entry.pack()
    start_point_entry.place(x=200, y=80)
    start_point_label = Label(window, text="Start Domain : ")
    start_point_label.pack()
    start_point_label.place(x=50, y=80)
    # start_point_default = Label(window, text="-1")
    # start_point_default.pack()
    # start_point_default.place(x=450, y=80)

    end_point_entry = Entry(window)
    end_point_entry.pack()
    end_point_entry.place(x=200, y=120)
    end_point_label = Label(window, text="End Domain : ")
    end_point_label.pack()
    end_point_label.place(x=50, y=120)
    # end_point_default = Label(window, text="1")
    # end_point_default.pack()
    # end_point_default.place(x=450, y=120)

    step_entry = Entry(window)
    step_entry.pack()
    step_entry.place(x=200, y=160)
    step_label = Label(window, text="Step : ")
    step_label.pack()
    step_label.place(x=50, y=160)
    # step_label_default = Label(window, text="0.001")
    # step_label_default.pack()
    # step_label_default.place(x=450, y=160)

    # noise_entry = Entry(window)
    # noise_entry.pack()
    # noise_entry.place(x=200, y=200)
    # noise_entry_label = Label(window, text="Noise : ")
    # noise_entry_label.pack()
    # noise_entry_label.place(x=50, y=200)
    # noise_entry_default = Label(window, text="0")
    # noise_entry_default.pack()
    # noise_entry_default.place(x=450, y=200)

    epochs_entry = Entry(window)
    epochs_entry.pack()
    epochs_entry.place(x=200, y=240)
    epochs_entry_label = Label(window, text="Iteration : ")
    epochs_entry_label.pack()
    epochs_entry_label.place(x=50, y=240)
    # epochs_entry_default = Label(window, text="2000")
    # epochs_entry_default.pack()
    # epochs_entry_default.place(x=450, y=240)

    hidden_layer_sizes_entry = Entry(window)
    hidden_layer_sizes_entry.pack()
    hidden_layer_sizes_entry.place(x=200, y=280)
    hidden_layer_sizes_entry_label = Label(window, text="Hidden Layer List : ")
    hidden_layer_sizes_entry_label.pack()
    hidden_layer_sizes_entry_label.place(x=50, y=280)
    # hidden_layer_sizes_entry_default = Label(window, text="20 40 50 30")
    # hidden_layer_sizes_entry_default.pack()
    # hidden_layer_sizes_entry_default.place(x=450, y=280)

    # test_size_entry = Entry(window)
    # test_size_entry.pack()
    # test_size_entry.place(x=200, y=320)
    # batch_size_label = Label(window, text="test_size : ")
    # batch_size_label.pack()
    # batch_size_label.place(x=50, y=320)
    # batch_size_default = Label(window, text="0.2")
    # batch_size_default.pack()
    # batch_size_default.place(x=450, y=320)

    combo = Combobox(window)
    combo['values'] = ('', "np.exp(-(np.square(x) + np.square(y))/4)", "3 * x + x * y - 5 * y",
                       'x * y + np.e ** x - np.sin((np.pi / 2) * y) + np.tanh(np.pi * x)')
    combo.current(0)
    combo.pack()
    combo.place(x=200, y=40)
    function_entry_label = Label(window, text="Choose Function : ")
    function_entry_label.pack()
    function_entry_label.place(x=50, y=40)
    # function_entry_default = Label(window, text="x ** 2")
    # function_entry_default.pack()
    # function_entry_default.place(x=450, y=40)
    mainloop()


equation = 'x + y'
start_time = time.time()
start_domain, end_domain = -1, 1
step = 0.005
hidden_layer_sizes = [10, 20, 20, 20, 20, 10]
iteration = 2000

gui()
f = eval('lambda x, y: ' + equation)

x = np.arange(start_domain, end_domain, step)
xy = [(j, k) for j in x for k in x]
out = [f(p[0], p[1]) for p in xy]
x_train, x_test, y_train, y_test = train_test_split(xy, out)

fig = plt.figure()
ax = fig.gca(projection='3d')

# plot train data points
x1_vals = np.array([p[0] for p in x_train])
x2_vals = np.array([p[1] for p in x_train])

ax.scatter(x1_vals, x2_vals, y_train)
# plt.show()

# plot test data points
x1_val = np.array([p[0] for p in x_test])
x2_val = np.array([p[1] for p in x_test])

ax.scatter(x1_val, x2_val, y_test, marker='x')
# plt.show()

mlp = MLPRegressor(
    hidden_layer_sizes=hidden_layer_sizes,
    max_iter=iteration,
    tol=0,
)

mlp.fit(x_train, y_train)
predictions = mlp.predict(x_test)

ax.scatter(x1_val, x2_val, predictions, c='black')
plt.show()

print(mean_squared_error(y_test, predictions))
print(mlp.score(x_test, y_test))
print("--- %s seconds ---" % (time.time() - start_time))
