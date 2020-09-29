# Will search current folder and subfolders for any files that starts with test_<something>.py
# Will then run and methods or classes that start with test_<something> or Test
# Run: pytest

import pytest

import numpy as np
import tensorflow as tf
import cnn_model as cnn

### Setup and toy problem for testing ###
@pytest.fixture(scope="module")
def mytoymnist():
    return cnn.load_data()


### TESTS ###

def test_tf_version():
    assert int(tf.__version__.split('.')[0]) >= 2

def test_train_data(mytoymnist):
    assert isinstance(mytoymnist[0], tf.data.Dataset)

def test_test_data(mytoymnist):
    assert isinstance(mytoymnist[1], tf.data.Dataset)

