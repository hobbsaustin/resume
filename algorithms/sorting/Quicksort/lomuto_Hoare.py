import numpy as np

# Using a class to keep track of all the recursive actions taken
class MainCounter:
    def __init__(self):
        self._total = 0

    def add(self):
        self._total += 1

    def clear(self):
        self._total = 0

    def display(self):
        return self._total

# Lomuto Partition
def lomuto(A, p, r):
    x = A[r]
    i = p - 1
    for j in range(p, r):
        if A[j] <= x:
            i += 1
            A[i], A[j] = A[j], A[i]
        Counter.add()
    A[i+1], A[r] = A[r], A[i+1]
    Counter.add()
    return i + 1

# hoare partition
def hoare(A, p, r):
    x = A[p]
    i = p
    j = r
    while True:
        while A[i] < x:
            i += 1
            Counter.add()
        while A[j] > x:
            j -= 1
            Counter.add()
        if i >= j:
            Counter.add()
            return j
        A[i], A[j] = A[j], A[i]
        Counter.add()

# lomuto quick implementation
def quicksort(A, p, r):
    if p < r:
        Counter.add()
        q = lomuto(A, p, r)
        quicksort(A, p, q-1)
        quicksort(A, q+1, r)

# hoare quick impolementation
def hoarequicksort(A,p,r):
    if p < r:
        q = hoare(A, p, r)
        hoarequicksort(A, p, q-1)
        hoarequicksort(A, q+1, r)

# for list one i create a new list with the median of the numbers and append rest
# side note, since quicksort is a inplace sort i need to copy each list for a true case study
# list_l will be for lomuto
first = [12]
for i in [x for x in range(25) if x != 12]:
    first.append(i)

# copy list using numpy's copy function
first_l = np.copy(first)

# same process as above but append the median last
second = []
for i in [x for x in range(25) if x != 12]:
    second.append(i)
second.append(12)
second_l = np.copy(second)

# create list with 1 - 25 elements excluding the median
three = []
for i in [x for x in range(25) if x != 12]:
    three.append(i)
# shuffle the numbers to randomly sort them and insert the median to index 0
np.random.shuffle(three)
three.insert(0, 12)
three_l = np.copy(three)

# same process as above but append our median last
four = []
for i in [x for x in range(25) if x != 12]:
    four.append(i)
np.random.shuffle(four)
four.append(12)
four_l = np.copy(four)

#testing out the sorts with the counter class
print('-----------')
print('First array')

Counter = MainCounter()
hoarequicksort(first, 0, len(first)-1)
print(Counter.display(), 'hoare Counter')


Counter.clear()
quicksort(first_l, 0, len(first)-1)
print(Counter.display(), 'lomuto Counter')

print('-----------')
print('Second array')

Counter.clear()
hoarequicksort(second, 0, len(first)-1)
print(Counter.display(), 'hoare Counter')

Counter.clear()
quicksort(second_l, 0, len(first)-1)
print(Counter.display(), 'lomuto Counter')

print('-----------')
print('Three array')

Counter.clear()
hoarequicksort(three, 0, len(first)-1)
print(Counter.display(), 'hoare Counter')

Counter.clear()
quicksort(three_l, 0, len(first)-1)
print(Counter.display(), 'lomuto Counter')

print('----------')
print('Four array')

Counter.clear()
hoarequicksort(four, 0, len(first)-1)
print(Counter.display(), 'hoare Counter')

Counter.clear()
quicksort(four_l, 0, len(first)-1)
print(Counter.display(), 'lomuto Counter')