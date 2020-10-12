# Will search current folder and subfolders for any files that starts with test_<something>.py
# Will then run and methods or classes that start with test_<something> or Test
# Run: pytest

import pytest

import numpy as np

import tensorflow as tf
from tensorflow import keras

from tf_training import OneCycleCallback, MomRangeTestCallback, LrRangeTestCallback

### Setup and toy problem for testing ###
def gen_value():
    for i in range(10):
        yield i

@pytest.fixture()
def myonecycle():
    return OneCycleCallback(stepsize=10, max_lr=10, min_lr=1, max_mom=10, min_mom=1)

@pytest.fixture()
def mymomrange():
    return MomRangeTestCallback(gen_value=gen_value())

@pytest.fixture()
def mylrrange():
    return LrRangeTestCallback(gen_value=gen_value())

@pytest.fixture()
def mysimplemodel():
    model = keras.Sequential()
    model.add(keras.layers.Dense(8))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def teardown_module(module):
    pass

### TESTS ###

def test_mom_value(mymomrange, mysimplemodel):
    mymomrange.model = mysimplemodel
    for _ in range(2):
        mymomrange.on_train_batch_begin(batch=1)
    assert mymomrange.model.optimizer.beta_1 == 1

def test_lr_value(mylrrange, mysimplemodel):
    mylrrange.model = mysimplemodel
    for _ in range(2):
        mylrrange.on_train_batch_begin(batch=1)
    assert mylrrange.model.optimizer.lr == 1

def test_oc_gen_value(myonecycle, mysimplemodel):
    lr, mom = zip(*[(i,j) for i,j in myonecycle._gen_value])
    assert 209 == np.sum(np.concatenate((lr,mom)))

def test_stop_condition_false(myonecycle, mysimplemodel):
    myonecycle.model = mysimplemodel
    for _ in range(21):
        myonecycle.on_epoch_begin(epoch=1)
    assert False == myonecycle._cycle

def test_stop_condition_true(myonecycle, mysimplemodel):
    myonecycle.model = mysimplemodel
    for _ in range(19):
        myonecycle.on_epoch_begin(epoch=1)
    assert True == myonecycle._cycle

def test_stop_condition_true(myonecycle, mysimplemodel):
    myonecycle.model = mysimplemodel
    for _ in range(2):
        myonecycle.on_epoch_begin(epoch=1)
    assert 11 == keras.backend.get_value(myonecycle.model.optimizer.lr) + keras.backend.get_value(myonecycle.model.optimizer.beta_1)