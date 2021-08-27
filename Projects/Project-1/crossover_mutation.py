import numpy.random as random


def crossover(a, b):
    ans = []
    for i in range(1, len(a)):
        x = random.randint(0, 2)
        if x == 0:
            ans.append(a[i])
        else:
            ans.append(b[i])
    return ans


def mutation(a):
    for i in range(1, len(a)):
        if random.randint(0, 10) == 0:
            if i <= 7:
                x = random.randint(0, 5)
                if x == 0:
                    a[i] = '+'
                elif x == 1:
                    a[i] = '-'
                elif x == 2:
                    a[i] = '*'
                elif x == 3:
                    a[i] = '/'
                else:
                    a[i] = '**'
            else:
                x = random.randint(0, 3)
                if x == 0 or x == 1:
                    a[i] = str(random.randint(-10, 10))
                else:
                    a[i] = 'x'
