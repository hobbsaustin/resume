import math

p0 = float(input('P0: '))
p1 = float(input('P1: '))
TOL = float(input('TOL: '))
N = int(input('N of iterations: '))
i = 2


def f(x):
    return math.cos(x)


q0 = f(p0)
q1 = f(p1)
counter = 0

while i <= N:
    p = p1 - q1*(p1 - p0)/(q1 - q0)

    if abs(p - p1) < TOL:
        print(p)
        break
    print(f'p_{counter} = {p}')
    i += 1
    counter += 1
    p0 = p1
    q0 = q1
    p1 = p
    q1 = f(p)

