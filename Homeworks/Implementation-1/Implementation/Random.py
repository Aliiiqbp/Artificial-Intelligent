import numpy as np
import time


def ans_generator(m, min_value, max_value):
    np.random.seed(int(time.time()))
    out = np.random.rand(m)
    for i in range(len(out)):
        out[i] = int(out[i] * (max_value - min_value) + min_value)
    return out
