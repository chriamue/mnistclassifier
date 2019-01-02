from importlib import import_module
import os
import importlib.util
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
    if name in model_class.keys():
        return model_class.get(name)
    class_name = os.path.splitext(os.path.basename(name))[0]
    print(name, class_name)
    spec = importlib.util.spec_from_file_location(class_name, name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)
