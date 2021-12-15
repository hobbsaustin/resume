import pandas as pd
import linkedlist
from datetime import datetime


# class for Baby names csv using __hash__ to give ord()
class Person:
    def __init__(self, name, year, gender, count):
        self.name = name
        self.year = year
        self.gender = gender
        self.count = count

    def __str__(self):
        return f'{self.name}, {self.year}, {self.gender}, {self.count}'

    def __hash__(self):
        return self.give_ord()

    def give_ord(self):
        return sum([ord(i) for i in self.name])**2 + (ord(self.gender)**2) * int(self.year)


# table class for storing baby names
class Table:
    def __init__(self, pdf, key):
        self.key = key
        self.file = pd.read_csv(pdf)
        self.data = self.data = [[] for _ in range(key)]
        self._insert_data()

    # helper function for inserting data into table
    def _insert_data(self):
        count = 0
        for index, row in self.file.iterrows():
            temp = Person(row[1], row[2], row[3], row[4])
            # using own hash function with ord
            hash_value = hash(temp)
            place = hash_value % self.key
            if len(self.data[place]) == 0:
                self.data[place] = linkedlist.LinkedList()
                self.data[place].AtBegining(temp)
            else:
                self.data[place].AtEnd(temp)
            count += 1
        print(f'steps taken to insert: {count}')

    # search method for table requires name, year, gender since thats how the class hashes
    def Search(self, name, year, gender):
        place = self.HASH_FUNCTION(name, year, gender)
        if len(self.data[place]) == 0:
            return 'Not Found'
        return self.data[place].FindVal(name, year, gender)

    def Search_complexity(self, name, year, gender):
        place = self.HASH_FUNCTION(name, year, gender)
        if len(self.data[place]) == 0:
            return 'Not Found'
        return self.data[place.FindVal_complexity(name, year, gender)]

    # required HASH_FUNCTION for class uses ord and adds name, gender and year
    def HASH_FUNCTION(self, name, year, gender):
        a = sum([ord(i) for i in name])**2 + (ord(gender)**2) * int(year)
        place = a % self.key
        return place

    # returns load of table
    def MAX_TABLE_LOAD(self):
        m = len(self.data)
        n = len(self.file)
        return n/m


    # inserts another class if needed
    def Insert_Table(self, person):
        temp = person
        # using own hash function with ord
        hash_value = hash(temp)
        place = hash_value % self.key
        if len(self.data[place]) == 0:
            self.data[place] = linkedlist.LinkedList()
            self.data[place].AtBegining(temp)
        else:
            self.data[place].AtEnd(temp)

# ---------TEST SECTION-----------

# table take around 1:21 to init
start = datetime.now()
print('init table')
Table = Table('NationalNames.csv', 179999)
print(datetime.now() - start)
# checking which node has the highest collisions
highest = 0
dex = 0
for index, value in enumerate(Table.data):
    if len(Table.data[index]):
        a = value.CountNodes()
        if a > highest:
            highest = a
            dex = index

print(highest, 'longest chain')
print(f'at index {dex}')
print('----------------')

# inserts person class into table
print('insert person')
temp = Person('Austin', 1996, 'M', 3)
Table.Insert_Table(temp)
print('done')
print('----------------')

# Checking the csv with the hash table if something comes up not found a flag will break loop
flag = True
print('checking search')
df = pd.read_csv('NationalNames.csv')
for index, row in df.iterrows():
    if not Table.Search(row[1], row[2], row[3]):
        print('Not Found')
        flag = False
        break

if flag:
    print('All Found')
