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

cv_params = {
    "cv":10,
    "scoring":"neg_mean_squared_error",
    "return_train_score":True
}

def mytoydata():
    np.random.seed(12359780)
    X = np.random.uniform(low=-10,high=10,size=size)
    y = np.power(X,2) + np.random.randn(size)
    return  np.reshape(X,(-1,1)), y

@pytest.fixture(scope="module")
def mytoymodel():
    X, y = mytoydata()
    models = [LinearRegression(),make_pipeline(PolynomialFeatures(2),LinearRegression())]
    trainer = st.try_all_models(models, cv_params)
    trainer.fit(X,y)
    return trainer

### TESTS ###

def test_list_size(mytoymodel):
    assert len(mytoymodel.data_dict["score"]) == 40

def test_dict_size(mytoymodel):
    assert len(mytoymodel.data_dict) == 3

def test_nan(mytoymodel):
    assert np.all(np.isnan(mytoymodel.data_dict["score"])) == False

