import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from data.dataset import MNISTDataset
import pandas as pd


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(.25)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(.25)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(.25)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x
def main():
    read = pd.read_csv('data/train.csv')
    train_split = read.iloc[:33600]
    test_split = read.iloc[33600:]
    model = Network()
    train_data = MNISTDataset(train_split)
    test_data = MNISTDataset(test_split)
    trainloader = DataLoader(train_data, batch_size=128, shuffle=True)
    testloader = DataLoader(test_data, batch_size=128, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())


    #model.load_state_dict(torch.load('weights/state_dict_model.pt'))
    y = []

    epochs = 25
    for e in range(epochs):

        running_loss = 0
        for images, labels in trainloader:
            images = Variable(images)
            labels = Variable(labels)
            optimizer.zero_grad()
            output = model(images)
            print(output)
            print(labels)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            model.eval()
            for images, labels in testloader:
                images = Variable(images).float()
                output = model(images)
                predicted = torch.max(output, 1)[1]
                accuracy += (predicted == labels).sum()

        model.train()

        print("Epoch {} - Training loss: {} - Accuracy: {}".format(e, running_loss / len(trainloader), float(accuracy) / (len(testloader)*128)))
        y.append(running_loss)

    torch.save(model.state_dict(), 'weights/state_dict_model_v2.pt')

    x = np.arange(len(y))
    plt.plot(x, y)
    plt.show()

main()
