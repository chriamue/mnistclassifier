import os
import torchvision.datasets as dset

dataset_class = {
    'fashion_mnist': dset.FashionMNIST,
    'mnist': dset.MNIST,
    'cifar10': dset.CIFAR10
}


def available():
    return list(dataset_class.keys())


def get_dataset(name):
    return dataset_class.get(name)


def datapath(configname, data='data'):
    p = os.path.join(data, configname)
    if not os.path.exists(p):
        os.makedirs(p)
    return p
