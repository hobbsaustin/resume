import torch
import pandas as pd
import torch.nn as nn
from data.dataset import MNISTDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

EPOCHS = 20

class Linear_forward(nn.Module):
    def __init__(self):
        super(Linear_forward, self).__init__()
        self.PATH = 'weights/linear_weights.pt'

        self.linear = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def load(self):
        self.load_state_dict(torch.load(self.PATH))

    def save(self):
        torch.save(self.state_dict(), self.PATH)

def train():
    read = pd.read_csv('data/train.csv')
    train_split = read.iloc[:33600]
    test_split = read.iloc[33600:]
    Model = Linear_forward()
    Model.load()
    train_data = MNISTDataset(train_split)
    test_data = MNISTDataset(test_split)
    trainloader = DataLoader(train_data, batch_size=128, shuffle=True)
    testloader = DataLoader(test_data, batch_size=128, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Model.parameters())

    temp = []
    for e in range(EPOCHS):

        running_loss = 0
        for images, labels in trainloader:
            images = Variable(images)
            labels = Variable(labels)
            optimizer.zero_grad()
            output = Model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()


        print("Epoch {} - Training loss: {}".format(e, running_loss / len(trainloader)))
        temp.append(running_loss/len(trainloader))

    Model.save()
    x = np.arange(len(temp))
    print(temp, 'temp')
    print(x, 'x')
    plt.plot(x,temp)
    plt.show()


def test():
    df = pd.read_csv('data/train.csv')
    train_df, test_df = train_test_split(df, test_size=.3)
    test_data = MNISTDataset(test_df)
    testloader = DataLoader(test_data, batch_size=128, shuffle=False)
    model = Linear_forward()
    model.load()
    acc = 0
    with torch.no_grad():
        model.eval()
        for images, labels in testloader:
            images = Variable(images).float()
            output = model(images)
            predicted = torch.max(output, 1)[1]
            acc += (predicted == labels).sum()
    print(float(acc)/(len(testloader)*128))

train()
test()
