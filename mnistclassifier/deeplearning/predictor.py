# source: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import os
import torch
from . import get_model
from ..data import datapath

class Predictor(object):

    model = None

    def __init__(self, config):
        self.config = config
        self.model = get_model(config)(pretrained=True)
        self.load()

    def load(self):
        root = datapath(self.config.current_name())
        modelfile = os.path.join(root, self.config.get()['modelfile'])
        if os.path.isfile(modelfile):
            checkpoint = torch.load(modelfile)
            self.model.load_state_dict(checkpoint['model_state_dict'])

    def predict(self, images):
        outputs = self.model(images)
        _, predicted = torch.Tensor.max(outputs, 1)
        return predicted
