# coding=utf-8
import os
import numpy as np
from ..config import Config
from .fileset import FileSet
from . import get_dataset
from . import datapath
import torch
import torchvision.transforms as transforms

class DataSet():
    data = {}

    def __init__(self, config: Config, train=True):
        self.config = config
        self.root = datapath(self.config.current_name())
        self.train = train
        self.loadDataset()

    def loadDataset(self):
        dataset = self.config.get()['dataset']
        f = os.path.join(self.root, dataset)
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        if os.path.exists(f):
            self.dataset = FileSet(
                self.config, train=self.train, transform=transform)
        else:
            Dataset = get_dataset(dataset)
            self.dataset = Dataset(
                self.root, download=True, train=self.train, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
