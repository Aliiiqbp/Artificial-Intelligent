import Error
import Random
import time
import math

print("Enter h:")
h = float(input())
print("Enter region of answers like this: min, max")
min_value, max_value = map(int, input().split(","))
print("Enter \"K\",number of iterations:")
k = int(input())
start_time = time.time()
multipliers_matrix, b = [], []
with open("example.txt", 'r') as f:
    for line in f:
        tmp = list(map(float, line.split(",")))
        multipliers_matrix.append(tmp[0:-1])
        b.append(tmp[-1])
n, m = len(b), len(multipliers_matrix[0])


final_answer, final_error = [], math.inf
for j in range(k):
    ans = list(Random.ans_generator(m, min_value, max_value))
    er = Error.calculate(multipliers_matrix, b, ans)

    cnt = 0
    while True:

        if cnt == 1000:
            h = 1
        elif cnt == 8000:
            h = 0.5
        elif cnt == 9500:
            h = 0.25
        elif cnt == 9900:
            h = 0.125
        elif cnt == 9990:
            h = 0.05

        neighbors = [ans.copy() for i in range(m * 2)]
        for i in range(m):
            if neighbors[i][i] + h <= max_value:
                neighbors[i][i] += h
            if neighbors[i + m][i] - h >= min_value:
                neighbors[i + m][i] -= h

        flag = False
        for x in neighbors:
            if Error.calculate(multipliers_matrix, b, x) < er:
                ans = x.copy()
                er = Error.calculate(multipliers_matrix, b, x)
                flag = True
        cnt += 1
        if cnt == 10000:
            break

    if er < final_error:
        final_error = er
        final_answer = ans


print("final answer and final error:\n", final_answer, final_error)
print("--- %s seconds ---" % (time.time() - start_time))
