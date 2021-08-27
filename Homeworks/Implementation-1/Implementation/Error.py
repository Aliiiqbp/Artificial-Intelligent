import numpy as np


def calculate(multiplier_matrix, b, ans):
    tmp = 0
    for i in range(len(multiplier_matrix)):
        tmp += (np.dot(multiplier_matrix[i], ans) - b[i]) ** 2
    return tmp
