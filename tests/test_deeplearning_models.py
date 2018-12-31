import pytest
from mnistclassifier.deeplearning import get_model
from mnistclassifier.config import Config

config = Config({"first":{}})

def test_resnet_model():
    Model = get_model('resnet18')
    model = Model(pretrained=True)