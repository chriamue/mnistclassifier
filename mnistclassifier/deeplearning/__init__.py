# coding=utf-8
from ..config import Config
from .models import get_model as _get_model


def get_model(config='vgg16'):
    if type(config) == Config:
        config = config.get()['model']
    return _get_model(config)
