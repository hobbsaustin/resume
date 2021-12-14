
a = float(input('A: '))
b = float(input('B: '))

TOL = float(input('TOL: '))
N = int(input('Number of iterations: '))
i = 1

def f(x):
    return x**3 + 4*x**2 - 10

FA = f(a)
counter = 0


while i <= N:
    p = a + (b-a)/2
    FP = f(p)
    if FP == 0 or (b-a)/2 < TOL:
        print(p)
        break
    print(f'n={i}   a={a}    b={b}    p={p}    f(p)={FP}')

    i += 1
    counter += 1

    if FA * FP > 0:
        a = p
        FA = f(p)

    else:
        b = p

