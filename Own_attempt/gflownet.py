import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
tfd = tfp.distributions

class GFlowNet:
    def __init__(self, env, n_layers=2, n_hidden=32, gamma=0.5, epochs=100, lr=0.005, decay_steps=10000, decay_rate=0.8):
        self.env = env
        self.dim = env.dim
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.gamma = gamma
        self.epochs = epochs
        self.lr = lr
        lr_schedule = ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )
        self.get_model()

    def get_model(self):
        # GO BACK AND REVIEW
        input_ = Input(shape=(self.dim, self.env.size), name='Input')
        layers = [Dense(units=self.n_hidden, activation='relu')(input_)]
        for _ in range(self.n_layers):
            layers.append(Dense(units=self.n_hidden, activation='relu')(layers[-1]))
        fpm = Dense(units=self.dim + 1, activation='log_softmax', name='forward_policy')(layers[-1])
        bpm = Dense(units=self.dim, activation='log_softmax', name='backward_policy')(layers[-1])
        self.z0 = tf.Variable(0., name='z0', trainable=True)
        self.model = Model(input_, [fpm, bpm])
        self.unif = tfd.Uniform(low=[0]*(self.dim + 1), high=[1]*(self.dim + 1))


    def mask_forward_actions(self, position_batch):


