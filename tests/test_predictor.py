import os
import numpy as np
import pytest

from PIL import Image
import torch
import torchvision.transforms as transforms

from mnistclassifier.config import Config
from mnistclassifier.deeplearning.predictor import Predictor

config = Config({"first": {}})
config.get()['modelfile'] = 'tmp.model.h5'


def test_predict_mnist():
    config.get()['model'] = 'cifar10'
    config.get()['modelfile'] = 'tmp.model.h5'
    config.get()['dataset'] = 'cifar10'
    config.get()['batch_size'] = 1000
    config.get()['epochs'] = 1
    predictor = Predictor(config)
    predictor.load()

    preprocess = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = Image.open('tests/res/images/1.png')
    img = preprocess(img)

    print(predictor.predict(img.unsqueeze_(0))[0].numpy())
