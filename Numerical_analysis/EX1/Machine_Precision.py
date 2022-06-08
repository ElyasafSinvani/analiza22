def eps_test(a):
    for i in range(1, 100):
        if a == a + (1 / 2 ** i):
            return 1 / 2 ** (i - 1)


def ABS(a):
    if a + eps_test(1) == int(a) or a - eps_test(1) == int(a):
        return abs(int(a))
    return abs(a)


print(ABS(3.0 * (4 / 3.0 - 1) - 1))