# source: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from . import get_model
from ..data.dataset import DataSet
from ..data import datapath


class Trainer(object):
    '''
    Trainer.
        - init
        - (load)
        - train
        - save
    '''
    model = None

    def __init__(self, config):
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.model = get_model(config)(pretrained=True)
        self.trainset = DataSet(self.config, train=True)
        self.testset = DataSet(self.config, train=False)
        learn_rate = self.config.get()['learn_rate']
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=learn_rate, momentum=0.9)

    def load(self):
        root = datapath(self.config.current_name())
        modelfile = os.path.join(root, self.config.get()['modelfile'])
        if os.path.isfile(modelfile):
            checkpoint = torch.load(modelfile)
            self.model.load_state_dict(checkpoint['model_state_dict'])

    def save(self):
        root = datapath(self.config.current_name())
        modelfile = os.path.join(root, self.config.get()['modelfile'])
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, modelfile)

    def train(self):
        epochs = self.config.get()['epochs']
        batch_size = self.config.get()['batch_size']

        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')

    def performance(self):
        batch_size = self.config.get()['batch_size']
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = self.model(images)
                _, predicted = torch.Tensor.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        performance_ = 100 * correct / total
        return performance_