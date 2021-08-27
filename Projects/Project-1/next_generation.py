import numpy.random as random
import crossover_mutation
import list_to_string


def generate(functions, a, b):
    str_functions = []
    for x in functions:
        temp = []
        list_to_string.pre(x, 1, temp)
        str_functions.append(list_to_string.list_to_str(temp))

    ans, tmp, errors = [], [], []
    for i in range(len(str_functions)):
        tmp.append((lambda x: eval(str_functions[i])))

    for i in range(len(tmp)):
        error = 0
        for j in range(len(a)):
            error += (b[j] - tmp[i](a[j])) ** 2
        errors.append(error)

    for i in range(len(errors)):
        for j in range(len(errors) - 1):
            if errors[j] > errors[j + 1]:
                errors[j], errors[j + 1] = errors[j + 1], errors[j]
                functions[j], functions[j + 1] = functions[j + 1], functions[j]

    for i in range(10):
        ans.append(functions[i])

    for i in range(90):
        j, k = random.randint(10, 51), random.randint(10, 51)
        a = crossover_mutation.crossover(functions[j], functions[k])
        crossover_mutation.mutation(a)
        ans.append(a)

    return ans
