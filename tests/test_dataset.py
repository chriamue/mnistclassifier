import os
import pytest
from datetime import datetime
from mnistclassifier.config import Config
from mnistclassifier.data import get_dataset
from mnistclassifier.data.dataset import DataSet

config = Config({"first": {}})
#dataset = DataSet(config)


def test_load():
    cwd = os.getcwd()
    print(cwd)
#    dataset.loadData()


def test_mnist_len():
    config.get()['dataset'] = 'mnist'
    dataset = DataSet(config)
    assert len(dataset) >= 0

def test_mnist():
    Dataset = get_dataset('mnist')
    dataset = Dataset('tmp', download=True)

def test_fileset():
    config.get()['dataset'] = '../../tests/res/dataset.yml'
    dataset = DataSet(config)
    print(dataset[0][0].size())
    print(len(dataset))