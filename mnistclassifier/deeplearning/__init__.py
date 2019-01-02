# coding=utf-8
import os
from ..config import Config
from .models import get_model as _get_model
from ..data import datapath


def get_model(config='vgg16'):
    if type(config) == Config:
        root = datapath(config.current_name())
        config = config.get()['model']
    if '.' in config:
        return _get_model(os.path.join(root, config))
    return _get_model(config)
