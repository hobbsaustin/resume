

p0 = float(input('initial approximation: '))
TOL = float(input('TOL: '))
N = int(input('number of iterations: '))
i = 1
counter = 0

def f(x):
    return 1/2*(10-x**3)**(1/2)

while i <= N:
    p = f(p0)
    if abs(p-p0)< TOL:
        print(p)
        break
    print(f'p_{counter} = {p}')
    i += 1
    counter += 1
    p0 = p
