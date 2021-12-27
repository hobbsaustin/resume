import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class RegressionModel:
    def __init__(self, data):
        self.df = pd.read_csv(data)
        self.bias = np.array([np.random.random()])
        self.weights = np.array([np.random.random()])
        self.learning_rate = 1e-8
        self.saved = ([],[])
        self._low_cost = np.float('inf')
        self.loss = []

    def forward(self, x):
        return (self.weights * x) + self.bias

    def squarederror(self):
        total = 0
        for i in range(len(self.df)):
            obs = self.df.iloc[i]
            output = self.forward(obs['bmi'])
            total += (output - obs['charges'])**2
        return total * 1/(2*len(self.df))

    def _antitheta0(self):
        total = 0
        for i in range(len(self.df)):
            obs = self.df.iloc[i]
            output = self.forward(obs['bmi'])
            total += (output - obs['charges']) * 1
        total = total * (self.learning_rate/len(self.df))
        return total

    def _antitheta1(self):
        total = 0
        for i in range(len(self.df)):
            obs = self.df.iloc[i]
            output = self.forward(obs['bmi'])
            total += (output - obs['charges']) * obs['charges']

        total = total * (self.learning_rate / len(self.df))
        return total

    def backprop(self):
        new_anti0 = self._antitheta0()
        new_anti1 = self._antitheta1()
        self.bias = self.bias - new_anti0
        self.weights = self.weights - new_anti1

    def training(self):
        for i in range(300):
            self.backprop()
            cost = self.squarederror()
            print(cost)
            self.loss.append(cost)
            if cost < self._low_cost:
                self.saved = (self.weights, self.bias)
                self._low_cost = cost
                print('Saved weights')
            #self.graph_line_int()
        self.weights = self.saved[0]
        self.bias = self.saved[1]

    def graph_line_int(self):
        x1 = np.arange(len(self.df))
        y1 = self.forward(x1)
        x = self.df['bmi']
        y = self.df['charges']
        plt.ylim(0, 60000)
        plt.xlim(0, 60)
        plt.scatter(x,y)
        plt.plot(x1, y1, 'r')
        plt.show(block=False)
        plt.pause(.25)
        plt.close()


model = RegressionModel('data.csv')

model.training()
print(model.squarederror(), 'final cost')

plt.plot(np.arange(len(model.loss)), model.loss)
plt.show()