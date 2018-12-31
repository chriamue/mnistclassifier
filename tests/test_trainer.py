import os
import numpy as np
import pytest

from mnistclassifier.config import Config
from mnistclassifier.deeplearning.trainer import Trainer

config = Config({"first": {}})
config.get()['modelfile'] = 'tmp.model.h5'

def test_train_mnist():
    config.get()['model'] = 'cifar10'
    config.get()['modelfile'] = 'tmp.model.h5'
    config.get()['dataset'] = 'cifar10'
    config.get()['batch_size'] = 1000
    config.get()['epochs'] = 1
    trainer = Trainer(config)
    trainer.train()
    trainer.save()
    trainer.load()
    assert trainer.performance() > 0
    print(trainer.trainset[0][0].size())