

def f(x):
    top = 10
    bottom = x + 4
    frac = top / bottom
    return frac**(1/2)

p0 = 1.5
TOL = 10e-5
N = 100

i = 1

while i <= N:

    p1 = f(p0)
    p2 = f(p1)
    p = p0 - (p1 - p0)**2 / (p2 - 2 * p1 + p0)

    if abs(p - p0):
        print(p)
        break

    i += 1
    p0 = p

