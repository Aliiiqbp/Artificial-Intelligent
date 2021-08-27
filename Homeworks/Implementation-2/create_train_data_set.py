import math
import random
import numpy as np

data_set_1 = {-1: [], 1: []}
data_set_2 = {-1: [], 1: []}
data_set_3 = {-1: [], 1: []}

while len(data_set_1[-1]) + len(data_set_1[1]) != 100:
    x, y = random.uniform(0, 10), random.uniform(0, 10)
    if x + y >= 5.5:
        data_set_1[1].append(["%.3f" % x, "%.3f" % y])
    elif x + y <= 4.5:
        data_set_1[-1].append(["%.3f" % x, "%.3f" % y])


while len(data_set_2[-1]) + len(data_set_2[1]) != 100:
    x, y = random.uniform(-10, 10), random.uniform(-10, 10)
    if 16 <= (x ** 2) + (y ** 2) <= 100:
        data_set_2[1].append(["%.3f" % x, "%.3f" % y])
    elif (x ** 2) + (y ** 2) <= 9:
        data_set_2[-1].append(["%.3f" % x, "%.3f" % y])

while len(data_set_3[-1]) + len(data_set_3[1]) != 100:
    x, y = random.uniform(-10, 10), random.uniform(-10, 10)
    if 16 <= (x ** 2) + (y ** 2) <= 100:
        data_set_2[1].append(["%.3f" % x, "%.3f" % y])
    elif (x ** 2) + (y ** 2) <= 9:
        data_set_2[-1].append(["%.3f" % x, "%.3f" % y])


print(data_set_1)
print(data_set_2)
