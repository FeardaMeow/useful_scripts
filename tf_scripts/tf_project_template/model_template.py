import tensorflow as tf
from tensorflow import keras
import os

# In terminal to start tensorboard, call
# >tensorboard --logdir=./my_logs --port=6006
root_logdir = os.path.join(os.curdir, "my_logs")

def build_model(input_shape):
    input_ = keras.layers.Input(shape=input_shape)
    results = keras.layers.Dense(30, activation="relu")(input_)
    results = keras.layers.Dense(1)(results)
    results = keras.Model(inputs=[input_], outputs=[results])

    return results

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

def main():
    model = build_model(10)

    model_compile_params = {
        "loss":"mse",
        "optimizer":keras.optimizers.SGD(lr=1e-3)
    }
    model.compile(**model_compile_params)


    # Train data, epochs, validation data, etc.
    checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model_best.h5")
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    run_logdir = get_run_logdir()
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    model_fit_params = {
        "x":None, "y":None, 
        "batch_size":None, 
        "epochs":1, 
        "callbacks":None,
        "validation_data":None
    }
    model.fit(**model_fit_params)

    # Save final model
    model.save("my_keras_model.h5")

if __name__ == "__main__":
    pass