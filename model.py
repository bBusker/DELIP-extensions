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

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)

  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)

def compute_gradients(model, x):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  return tape.gradient(loss, model.trainable_variables), loss

optimizer = tf.train.AdamOptimizer(1e-4)
def apply_gradients(optimizer, gradients, variables, global_step=None):
  optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

class DELIP_model(tf.keras.Model):
    def __init__(self, _latent_dim):
        super(DELIP_model, self).__init__()
        self.latent_dim = _latent_dim
        self.posterior_model = tf.keras.Sequential(
            [
                tf.keras.layers.LSTM(units=self.latent_dim*2, input_shape=(2,5))
            ]
        )

        self.state_generator = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim+1,)),
                tf.keras.layers.Dense(units=1, activation='linear')
            ]
        )

        self.observations_generator = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.layers.Dense(units=100, activation='relu'),
                tf.layers.Dense(units=100, activation='relu'),
                tf.layers.Dense(units=100, activation='relu'),
                tf.layers.Dense(units=6, activation='relu'),
            ]
        )

        self.rewards_generator = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.layers.Dense(units=100, activation='relu'),
                tf.layers.Dense(units=100, activation='relu'),
                tf.layers.Dense(units=100, activation='relu'),
                tf.layers.Dense(units=2, activation='relu'),
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
    print(model.state_generator.summary())
    print(model.observations_generator.summary())
    print(model.rewards_generator.summary())
    res = model.
