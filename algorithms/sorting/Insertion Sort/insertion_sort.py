import numpy as np
import time

A = np.array(np.random.randint(10**2, size=10**2))
B = np.array(np.random.randint(10**3, size=10**3))
C = np.array(np.random.randint(10**4, size=10**4))



def sort(A):
    # copy list to not effect og list
    temp = A

    counter = 0
    start = time.time()
    for j in range(1, len(temp)):
        key = temp[j]
        i = j - 1
        while i >= 0 and temp[i] > key:
            temp[i+1] = temp[i]
            i -= 1
            counter += 1
        temp[i+1] = key
    end = time.time() - start
    return temp, counter, end

one = sort(A)
two = sort(B)
three = sort(C)

print(f'\nn={len(A)} counter={one[1]} theoretical={len(A)**2} time elapsed={one[2]} sec')
print(f'n={len(B)} counter={two[1]} theoretical={len(B)**2} time elapsed={two[2]} sec')
print(f'n={len(C)} counter={three[1]} theoretical={len(C)**2} time elapsed={three[2]} sec \n')

