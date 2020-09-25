import tensorflow as tf
from tensorflow import keras

import tarfile

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import reciprocal

import numpy as np


def main():
    np.random.seed(31415)

    with tarfile.open(mode="r:gz", name=".\\tf_scripts\\cal_housing.tgz") as f:
        housing = np.loadtxt(f.extractfile('CaliforniaHousing/cal_housing.data'), delimiter=',')
        housing = housing[:,[7, 8, 2, 3, 4, 5, 6, 1, 0]]


    X_train_full, X_test, y_train_full, y_test = train_test_split(housing[:,1:], housing[:,:1], test_size=0.2)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

    param_distribs = {
        "n_hidden": [1,2,3,4,5],
        "n_neurons": [10,20,30,40,50,60,70,80,100],
        "learning_rate": [1e-10,1e-8,1e-6,1e-4,1e-3,1e-2]
    }

    rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
    rnd_search_cv.fit(X_train, y_train, epochs=100, 
                        validation_data=(X_valid,y_valid),
                        callbacks=[ keras.callbacks.EarlyStopping(patience=10) ] )


def build_model(n_hidden=1, n_neurons=30, learning_rate=1e-3, input_shape=[8], loss="mse"):
    input_ = keras.layers.Input(shape=input_shape)
    x = input_
    for layer in range(n_hidden):
        try:
            x = keras.layers.Dense(n_neurons[layer] ,activation="relu", name=layer)(x)
        except:
            x = keras.layers.Dense(n_neurons ,activation="relu", name=layer)(x)
    output_ = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=[input_], outputs=[output_])
    model.compile(loss=loss, optimizer=keras.optimizers.SGD(lr=learning_rate))
    return model

if __name__ == "__main__":
    main()

