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

@pytest.fixture(scope="module")
def mytoymodel():
    tf.random.set_seed(322)
    return cnn.build_model((28,28,1))


### TESTS ###

def test_tf_version():
    assert int(tf.__version__.split('.')[0]) >= 2

def test_train_data(mytoymnist):
    assert isinstance(mytoymnist[0], tf.data.Dataset)

def test_test_data(mytoymnist):
    assert isinstance(mytoymnist[1], tf.data.Dataset)

def test_cnn_model_output(mytoymodel):
    np.random.seed(322)
    assert np.sum(mytoymodel(np.random.uniform(size=(1,28,28,1)).astype(np.float32))) == 1.0

def test_cnn_model_layers(mytoymodel):
    assert len(mytoymodel.layers) == 13