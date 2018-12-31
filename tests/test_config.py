import pytest
from mnistclassifier.config import Config

def test_add_defaults():
    config = Config({"first":{}})
    assert config[0]['batch_size'] == 1
    assert config[0]['learn_rate'] == 1
    assert 'first' == config.current_name()
