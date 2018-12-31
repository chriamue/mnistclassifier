import torchvision.models as models
from .cifar10 import Cifar10

model_class = {
    'resnet18': models.resnet18,
    'vgg16': models.vgg16,
    'cifar10': Cifar10
}


def available():
    return list(model_class.keys())


def get_model(name):
    return model_class.get(name)
