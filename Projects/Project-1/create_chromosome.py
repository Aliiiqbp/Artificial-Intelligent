import numpy.random as random


def chromosome():
    chromosome = [0]
    op = random.randint(0, 5, 7)
    for i in range(7):
        if op[i] == 0:
            chromosome.append('+')
        elif op[i] == 1:
            chromosome.append('-')
        elif op[i] == 2:
            chromosome.append('*')
        elif op[i] == 3:
            chromosome.append('/')
        else:
            chromosome.append('**')

    op = random.randint(0, 3, 8)
    for i in range(8):
        if op[i] == 0 or op[i] == 1:
            tmp = round((random.rand() * 20) - 10, 5)
            if -0.00001 < tmp < 0.000001:
                chromosome.append(1)
            else:
                chromosome.append(str(tmp))
        else:
            chromosome.append('x')

    return chromosome
