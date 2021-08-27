def pre(arr, i, ans):
    if i >= len(arr) / 2:
        ans.append(arr[i])
    else:
        pre(arr, 2 * i, ans)
        ans.append(arr[i])
        pre(arr, 2 * i + 1, ans)


def list_to_str(lst):
    s = ''
    for i in range(len(lst)):
        if i % 8 == 0:
            s += '('
        if i % 4 == 0:
            s += '('

        s += lst[i]

        if i % 4 == 2:
            s += ')'
        if i % 8 == 6:
            s += ')'
    return s
