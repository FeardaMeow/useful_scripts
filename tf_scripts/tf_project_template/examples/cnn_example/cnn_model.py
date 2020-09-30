import tensorflow as tf
from tensorflow import keras
import os

'''

TODO: Finish training model test cases
TODO: Add in final model assessment
TODO: Add in logging
TODO: Add in cleanup after test case checking for files
TODO: Integrate in hyperband
'''

# In terminal to start tensorboard, call
# >tensorboard --logdir=./my_logs --port=6006
root_logdir = os.path.join(os.curdir, "my_logs")

def build_model(input_shape):
    input_ = keras.layers.Input(shape=input_shape)

    results = keras.layers.Conv2D(64,7, activation="relu", padding="same")(input_)
    results = keras.layers.MaxPool2D(2)(results)
    results = keras.layers.Conv2D(128,3, activation="relu", padding="same")(results)
    results = keras.layers.Conv2D(128,3, activation="relu", padding="same")(results)
    results = keras.layers.MaxPool2D(2)(results)
    results = keras.layers.Conv2D(256,3, activation="relu", padding="same")(results)
    results = keras.layers.Conv2D(256,3, activation="relu", padding="same")(results)
    results = keras.layers.MaxPool2D(2)(results)
    results = keras.layers.Flatten()(results)
    results = keras.layers.Dense(128, activation="relu")(results)
    results = keras.layers.Dropout(0.5)(results)
    results = keras.layers.Dense(10, activation="softmax")(results)

    return keras.Model(inputs=[input_], outputs=[results])

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

def load_data():
    import tensorflow_datasets as tfds

    datasets = tfds.load(name="mnist")
    mnist_train, mnist_test = datasets["train"], datasets["test"]

    return mnist_train, mnist_test

def main():
    mnist_train, mnist_test = load_data()
    mnist_train = mnist_train.shuffle(10000).batch(64)
    mnist_train = mnist_train.map(lambda items: (items["image"], items["label"]))
    mnist_train = mnist_train.prefetch(1)

    model = build_model((28, 28, 1))

    model_compile_params = {
        "loss":"sparse_categorical_crossentropy",
        "optimizer":keras.optimizers.SGD(lr=1e-3),
        "metrics":[]
    }
    model.compile(**model_compile_params)


    # Train data, epochs, validation data, etc.
    checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model_best.h5")
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    run_logdir = get_run_logdir()
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    model_fit_params = {
        "batch_size":64, 
        "epochs":15, 
        "callbacks":[checkpoint_cb, early_stopping_cb]
    }
    model.fit(mnist_train, **model_fit_params)

    # Save final model
    model.save("my_keras_model.h5")

if __name__ == "__main__":
    main()