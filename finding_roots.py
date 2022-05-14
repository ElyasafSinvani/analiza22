import sympy as sp
from sympy.utilities.lambdify import lambdify

x = sp.symbols('x')


def Bisection_Method(f, a, b, epsilon=10 ** -15):
    f_tag = sp.diff(f, x)
    f_tag = lambdify(x, f_tag)
    f = lambdify(x, f)
    temp = []
    result = 0
    if f(a) * f(b) < 0:
        while b - a > epsilon:
            print('[ %1.6f' % (a) + ', %1.6f ]' % (b))
            result = result + 1
            m = a + (b - a) / 2
            if f(a) * f(m) > 0:
                a = m
            else:
                b = m
        return [(a + b) / 2, result]
    elif f_tag(a) * f_tag(b) < 0:
        while b - a > epsilon:
            temp.append('[ %1.6f' % (a) + ', %1.6f ]' % (b))
            result = result + 1
            m = a + (b - a) / 2
            if f_tag(a) * f_tag(m) > 0:
                a = m
            else:
                b = m
        if abs(f((a + b) / 2)) < epsilon:
            for i in temp:
                print(i)
            return [(a + b) / 2, result]


def Newton_Raphson(f, a, b, epsilon=10 ** -15):
    f_tag = sp.diff(f, x)
    f_tag = lambdify(x, f_tag)
    f = lambdify(x, f)
    temp = []
    result = 0
    if f(a) * f(b) < 0:
        m1 = (a + b) / 2
        m2 = m1 - f(m1) / f_tag(m1)
        while abs(m2 - m1) > epsilon:
            print('[ %1.6f' % (m1) + ', %1.6f ]' % (m2))
            result = result + 1
            m1 = m2
            m2 = m1 - f(m1) / f_tag(m1)
        return [m2, result]
    elif f_tag(a) * f_tag(b) < 0:
        m1 = (a + b) / 2
        m2 = m1 - f(m1) / f_tag(m1)
        while abs(m2 - m1) > epsilon:
            temp.append('[ %1.6f' % (m1) + ', %1.6f ]' % (m2))
            result = result + 1
            m1 = m2
            m2 = m1 - f(m1) / f_tag(m1)
        if abs(f(m2)) < epsilon and (m2 >= a + epsilon or m2 >= a - epsilon) and (
                m2 <= b + epsilon or m2 <= b + epsilon):
            for i in temp:
                print(i)
            return [m2, result]


def secant_method(f, a, b, epsilon=10 ** -15):
    f_tag = sp.diff(f, x)
    f_tag = lambdify(x, f_tag)
    f = lambdify(x, f)
    temp1 = []
    result = 0
    if f(a) * f(b) < 0:
        m1 = a
        m2 = b
        while abs(m2 - m1) > epsilon:
            print('[ %1.6f' % (m1) + ', %1.6f ]' % (m2))
            temp = m2
            result = result + 1
            m2 = (m1 * f(m2) - m2 * f(m1)) / (f(m2) - f(m1))
            m1 = temp
        return [m2, result]
    elif f_tag(a) * f_tag(b) < 0:
        m1 = a
        m2 = b
        while abs(m2 - m1) > epsilon:
            temp1.append('[ %1.6f' % (m1) + ', %1.6f ]' % (m2))
            temp = m2
            result = result + 1
            m2 = (m1 * f(m2) - m2 * f(m1)) / (f(m2) - f(m1))
            m1 = temp
        if abs(f(m2)) < epsilon and (m2 >= a + epsilon or m2 >= a - epsilon) and (
                m2 <= b + epsilon or m2 <= b + epsilon):
            for i in temp1:
                print(i)
            return [m2, result]


def main():
    my_f = x**3-x-1
    f_tag = sp.diff(my_f, x)
    f_tag = lambdify(x, f_tag)
    f = lambdify(x, my_f)
    start_point = -100
    end_point = 100
    step = 0.1
    epsilon = 10 ** -6
    print('Please select the method you want to find the roots of the equation : \n')
    choice = input('1 - Bisection Method\n2 - Newton Raphson\n3 - secant method\n')
    while choice != '3' and choice != '2' and choice != '1':
        print('Error input please try again...\n')
        choice = input('1 - Bisection Method\n2 - Newton Raphson\n3 - secant method\n')
    if choice == '1':
        while end_point > start_point:
            if f(start_point) * f(start_point + step) < 0 or f_tag(start_point) * f_tag(start_point + step) < 0:
                temp = Bisection_Method(my_f, start_point, start_point + step, epsilon)
                if temp is not None:
                    print(f'The number of iterations to find the root is {temp[1]}')
                    print(f'The root of the function ==> %1.6f' % (temp[0]))
            start_point = start_point + step
    elif choice == '2':
        while end_point > start_point:
            if f(start_point) * f(start_point + step) < 0 or f_tag(start_point) * f_tag(start_point + step) < 0:
                temp = Newton_Raphson(my_f, start_point, start_point + step, epsilon)
                if temp is not None:
                    print(f'The number of iterations to find the root is {temp[1]}')
                    print(f'The root of the function ==> %1.6f' % (temp[0]))
            start_point = start_point + step
    elif choice == '3':
        while end_point > start_point:
            if f(start_point) * f(start_point + step) < 0 or f_tag(start_point) * f_tag(start_point + step) < 0:
                temp = secant_method(my_f, start_point, start_point + step, epsilon)
                if temp is not None:
                    print(f'The number of iterations to find the root is {temp[1]}')
                    print(f'The root of the function ==> %1.6f' % (temp[0]))
            start_point = start_point + step


main()
