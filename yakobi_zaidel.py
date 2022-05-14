from sympy.utilities.lambdify import lambdify
import sympy as sp
x =sp.symbols('x')
f = x**3+2*x+5
f_prime = f.diff(x)
print("f : ",f)
print("f' : ",f_prime)
f = lambdify(x, f)
f_prime = lambdify(x, f_prime)
print("f(1):",f(1))
print("f'(1):",f_prime(1))
