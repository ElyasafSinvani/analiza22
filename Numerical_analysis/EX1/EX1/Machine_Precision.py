def eps_test(a):
<<<<<<< HEAD
    for i in range(1, 2000):
        if a == a + (2**0 / 2 ** i):
            return 2**0 / 2 ** i

print(eps_test(1))
b = 1 + 1 / 10 ** 16
c = 1 + 1 / 10 ** 15
print(b)
print(c)
=======
    for i in range(1, 100):
        if a == a + (1 / 2 ** i):
            return 1 / 2 ** (i - 1)


def ABS(a):
    if a + eps_test(1) == int(a) or a - eps_test(1) == int(a):
        return abs(int(a))
    return abs(a)


print(ABS(3.0 * (4 / 3.0 - 1) - 1))
>>>>>>> 8f124c386f506d8d6d403ba061906790cfdd2683
