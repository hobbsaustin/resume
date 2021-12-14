import math
from scipy.misc import derivative

p0 = float(input('initial approximation: '))
TOL = float(input('tolerance: '))
N = float(input('number of iterations: '))

def function(x):
    return math.cos(x) - x

i = 1
counter = 1

while i <= N:
    p = p0 - (function(p0)/derivative(function, p0))
    print(f'p_{counter} = {p}')
    if abs(p - p0) < TOL:
        print(p)
        break

    i += 1
    counter += 1
    p0 = p

