from math import *
import sympy as sp
from sympy.utilities.lambdify import lambdify
from sympy import *
import math

x = symbols('x')


def function(x):
    function = (x * (e ** (-x)) + ln(x ** 2, e)) * (2 * x ** 3 + 2 * x ** 2 - 3 * x - 5)  # input your function
    return (function)


# Input your upper,lower and interval variables
upper = 1
lower = 0.5
intervals = 10
h = (upper - lower) / intervals  # The value of each step
z = intervals / 2  # The number of even functions(z+1 is the number of odd functions)
ans = 0


def simpsonsrule():
    rule = 0
    x = lower + h  # Calculates for all odd valures
    for i in range(1, int(z + 1)):
        print(rule)
        rule += 4 * function(x)  # adds all calculations to variable rule
        x += 2 * h

    x = lower + 2 * h  # Calculates for all even values
    for i in range(1, int(z)):
        print(rule)
        rule += 2 * function(x)  # adds all calculations to variable rule
        x += 2 * h
    ans = (h / 3) * (function(lower) + function(upper) + rule)  # This is carrying out the formula
    return (ans)


def trapeziumrule():
    rule = 0
    x = lower
    rule += function(x)  # adds lower value to rule
    x = upper
    rule += function(x)  # adds upper value to rule
    for i in range(1, intervals):  # all values inbetween
        x = lower + h * i
        print(f'rule = {rule}')
        rule += 2 * (function(x))  # adds these to the rule
    ans = (h / 2) * rule  # completes the function
    return (ans)


#print(f'Trapezoidal integral method {trapeziumrule()}\n')
print(f'Simpson integral method:\n'
      f'the func is : x*(e**(-(x**2)+(5*x)-3))*((x**2)+(3*x)-5)\n'
      f'from starts in x = {lower}\n'
      f'to ends in x= {upper}\n'
      f'The size between each step (like 1/4,1/2,3/4) {h}\n'
      f'the result is: {simpsonsrule()}\n'
      f'the intervals number is: {intervals}\n'
      f'we can see that if we upper the number of intervals the result get better\n')
