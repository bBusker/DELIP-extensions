import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
import robot_doors_experiment as exp

tf.enable_eager_execution()
tfe = tf.contrib.eager

def generate_data(steps):
    experiment = exp.RobotDoorsExperiment()
    trajectory = []
    for i in range(steps):
        curr_action = np.random.choice([0,1,2])
        experiment.take_action(curr_action)
        res = experiment.get_observation_discrete() + (experiment.get_reward(),) + (curr_action,)
        trajectory.append(res)
    return trajectory


class DELIP_model(tf.keras.Model):
    def __init__(self, _latent_dim):
        super(DELIP_model, self).__init__()
        self.latent_dim = _latent_dim
        self.posterior_model = tf.keras.Sequential(
            [
                tf.keras.layers.LSTM(units=self.latent_dim, input_shape=(2,5))
            ]
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation=tf.nn.relu),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation=tf.nn.relu),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
            ]
        )

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random_normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random_normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits



if __name__ == "__main__":
    print("Using Tensorflow Version: {}".format(tf.VERSION))
    print("Using Keras Version: {}".format(tf.keras.__version__))
    data = generate_data(100)
    print(data)
    model = DELIP_model(10)
    print(model.posterior_model.summary())