import tensorflow as tf
from tensorflow import keras
import numpy as np

class LrRangeTestCallback(keras.callbacks.Callback):
    '''
    TODO: Log losses for each learning rate
    TODO: Build Test Cases
    '''
    def __init__(self, gen_value):
        super(LrRangeTestCallback, self).__init__()
        self.gen_value = gen_value
        self._losses = []
        self._lr = []

    def on_train_batch_begin(self, batch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Set the value back to the optimizer before this epoch starts
        try:
            tf.keras.backend.set_value(self.model.optimizer.lr, next(self.gen_value))
            self._lr.append(float(keras.backend.get_value(self.model.optimizer.learning_rate)))
            print("\nBatch %05d: Learning rate is %6.4f." % (batch, float(keras.backend.get_value(self.model.optimizer.learning_rate))))
        except StopIteration:
            self.model.stop_training = True
            print("End of range test, training is stopping.")

    def on_train_batch_end(self, batch, logs=None):
        self._losses.append(logs["loss"])

    def on_train_end(self, logs=None):
        '''
        TODO: Write losses to file
        '''
        pass

class MomRangeTestCallback(keras.callbacks.Callback):
    
    def __init__(self,gen_value):
        super(MomRangeTestCallback, self).__init__()
        self.gen_value = gen_value
        self._losses = []
        self._mom = []

    def on_train_batch_begin(self, batch, logs=None):
        # Set the value back to the optimizer before this epoch starts
        if not hasattr(self.model.optimizer, "beta_1"):
            raise ValueError('Optimizer must have a "beta_1" attribute.')
        try:
            tf.keras.backend.set_value(self.model.optimizer.beta_1, next(self.gen_value))
            self._mom.append(float(keras.backend.get_value(self.model.optimizer.beta_1)))
            print("\nBatch %05d: Momentum (beta 1) is %6.4f." % (batch, float(keras.backend.get_value(self.model.optimizer.beta_1))))
        except StopIteration:
            self.model.stop_training = True
            print("End of range test, training is stopping.")

    def on_train_batch_end(self, batch, logs=None):
        self._losses.append(logs["loss"])

    def on_train_end(self, logs=None):
        '''
        TODO: Write losses to file
        '''
        pass

class OneCycleCallback(keras.callbacks.Callback):

    def __init__(self, stepsize=20, max_lr=0.01, min_lr=None, max_mom=0.9, min_mom=0.8):
        super(OneCycleCallback, self).__init__()
        self.stepsize = stepsize
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.max_mom = max_mom
        self.min_mom = min_mom
        self._cycle = True
        self._gen_value = self._make_gen()

    def _make_gen(self):
        if self.min_lr == None:
            self.min_lr = self.max_lr*(1./10)

        if self._cycle:
            lr_schedule = np.concatenate((np.linspace(self.min_lr, self.max_lr, self.stepsize), np.linspace(self.max_lr, self.min_lr, self.stepsize)[1:]))
            mom_schedule = np.concatenate((np.linspace(self.max_mom, self.min_mom, self.stepsize), np.linspace(self.min_mom, self.max_mom, self.stepsize)[1:]))
        else:
            lr_schedule = np.linspace(self.min_lr, self.max_lr, self.stepsize)
            mom_schedule = np.linspace(self.max_mom, self.min_mom, self.stepsize)

        for lr_i, mom_i in zip(lr_schedule, mom_schedule):
            yield lr_i, mom_i

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "beta_1"):
            raise ValueError('Optimizer must have a "beta_1" attribute.')
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')

        try:
            lr,beta_1 = next(self._gen_value)
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
            tf.keras.backend.set_value(self.model.optimizer.beta_1, beta_1)
        except StopIteration:
            self._cycle = False
            self.max_lr = float(keras.backend.get_value(self.model.optimizer.beta_1))
            self.min_lr = None
            self.min_mom = self.max_mom

            self._gen_value = self._make_gen()

            lr,beta_1 = next(self._gen_value)
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
            tf.keras.backend.set_value(self.model.optimizer.beta_1, beta_1)
