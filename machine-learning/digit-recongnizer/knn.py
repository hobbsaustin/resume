from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


def sklearn_knn(test_x, test_y, train_x, train_y):
    # add models and fit
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)

    # print score
    print(accuracy_score(test_y, prediction))

def Kmodel(df_x, df_y, image, k=7):
    temp = []
    for i in range(len(df_x)):
        coord = df_x.iloc[i].to_numpy()
        distance = np.sqrt(((coord-image)**2).sum())
        temp.append(distance)

    counter = Counter()
    for i in range(k):
        arg = np.argmin(temp)
        counter[str(df_y.iloc[arg])] += 1
        temp.pop(arg)
    return int(counter.most_common(1)[0][0])

def score(train_x, train_y, test_x, test_y):
    acc = 0
    print(len(test_x))
    for i in range(len(test_x)):
        pred = Kmodel(train_x, train_y, test_x.iloc[i].to_numpy())
        if pred == test_y.iloc[i]:
            acc += 1
        if i != 0:
            print(acc/i)

    return acc / len(test_x)


def main():
    # load data and split
    df = pd.read_csv('data/train.csv')
    test, train = train_test_split(df, test_size=0.3)
    train_x = train.copy().drop('label', axis=1)
    train_y = train.label
    test_x = test.copy().drop('label', axis=1)
    test_y = test.label
    image = test_x.iloc[1].to_numpy()


    #sklearn_knn(test_x, test_y, train_x, train_y)
    print(score(train_x, train_y, test_x, test_y))



main()
