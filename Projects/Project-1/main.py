import time
import create_chromosome
import next_generation
import list_to_string


t0 = time.time()
x, y = [], []
f = open('File/input.txt', '+r')
f.readline()
for i in range(5):
    tmp = list(f.readline().split(','))
    x.append(int(tmp[0]))
    y.append(int(tmp[1]))

functions = []
for i in range(100):
    functions.append(create_chromosome.chromosome())
for i in range(100):
    functions = next_generation.generate(functions, x, y)

t1 = time.time()
t = t1 - t0
print(t, '_____seconds_____')
for x in functions:
    print(list_to_string.list_to_str(x))
