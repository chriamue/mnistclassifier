# coding=utf-8
import os
import yaml
import numpy as np
from PIL import Image
from ..config import Config
from . import get_dataset
from . import datapath


class FileSet():
    dataset = []
    classes = []

    def __init__(self, config: Config, transform, train=True):
        self.config = config
        self.transform = transform
        self.train = train
        self.root = datapath(self.config.current_name())
        self.loadDataset()

    def loadDataset(self):
        dataset = os.path.join(self.root, self.config.get()['dataset'])
        with open(dataset) as file:
            mode = 'test'
            if self.train:
                mode = 'train'
            self.datasetfile = yaml.load(file)
            self.classes = self.datasetfile['classes']
            for label in self.datasetfile[mode]:
                for imgfile in self.datasetfile[mode][label]:
                    self.dataset.append((imgfile, self.classes.index(label)))
    
    def loadImage(self, image_tuple):
        filepath, target = image_tuple
        image = Image.open(os.path.join(self.root, filepath))
        image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.loadImage(self.dataset[index])
