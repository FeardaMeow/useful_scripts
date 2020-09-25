# Will search current folder and subfolders for any files that starts with test_<something>.py
# Will then run and methods or classes that start with test_<something> or Test
# Run: pytest

import pytest

import numpy as np
import shotgun_trainer as st

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


### Setup and toy problem for testing ###
size = 10000

cv_params_single = {
    "cv":10,
    "scoring":"neg_mean_squared_error",
    "return_train_score":True
}

cv_params_multiple = {
    "cv":10,
    "scoring":('r2', 'neg_mean_squared_error'),
    "return_train_score":True
}

def mytoydata():
    np.random.seed(12359780)
    X = np.random.uniform(low=-10,high=10,size=size)
    y = np.power(X,2) + np.random.randn(size)
    return  np.reshape(X,(-1,1)), y

@pytest.fixture(scope="module")
def mymodelsingle():
    X, y = mytoydata()
    models = [LinearRegression(),make_pipeline(PolynomialFeatures(2),LinearRegression())]
    trainer = st.try_all_models(models, cv_params_single)
    trainer.fit(X,y)
    return trainer

@pytest.fixture(scope="module")
def mymodelmultiple():
    X, y = mytoydata()
    models = [LinearRegression(),make_pipeline(PolynomialFeatures(2),LinearRegression())]
    trainer = st.try_all_models(models, cv_params_multiple)
    trainer.fit(X,y)
    return trainer

### TESTS ###

def test_list_size(mymodelsingle):
    assert len(mymodelsingle.data_dict["score"]) == 40

def test_dict_size(mymodelsingle):
    assert len(mymodelsingle.data_dict) == 4

def test_nan(mymodelsingle):
    assert np.all(np.isnan(mymodelsingle.data_dict["score"])) == False

def test_list_size_multiple(mymodelmultiple):
    assert len(mymodelmultiple.data_dict["score"]) == 80

def test_dict_size_multiple(mymodelmultiple):
    assert len(mymodelmultiple.data_dict) == 4

def test_nan_multiple(mymodelmultiple):
    assert np.all(np.isnan(mymodelmultiple.data_dict["score"])) == False

