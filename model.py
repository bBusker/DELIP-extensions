import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
import robot_doors_experiment as exp

tf.enable_eager_execution()
tfe = tf.contrib.eager

def generate_data(steps=10, episodes=10):
    data = []
    for i in range(episodes):
        experiment = exp.RobotDoorsExperiment()
        trajectory = []
        for j in range(steps):
            curr_action = np.random.choice([0,1,2])
            experiment.take_action(curr_action)
            # TODO(slu): maybe need to standardize to discrete observations in generator for POMCP
            res = experiment.get_observation_discrete() + (experiment.get_reward(),) + (curr_action,)
            trajectory.append(res)
        data.append(trajectory)
    data = tf.constant(data, dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    return dataset

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def compute_loss(model, x):
  mean, logvar = model.infer(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.generate(z)

  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)

def compute_loss_combined(model, x, next_state):
    pass

def compute_gradients(model, x):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  return tape.gradient(loss, model.trainable_variables), loss

optimizer = tf.train.AdamOptimizer(1e-4)
def apply_gradients(optimizer, gradients, variables, global_step=None):
  optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

class DELIP_model(tf.keras.Model):
    def __init__(self, latent_dim=4): # Eliminate magic numbers
        super(DELIP_model, self).__init__()
        self.latent_dim = latent_dim

        # Use functional interface to define posterior model.
        # Inputs: observation and previous state, Outputs: current state as variational posterior (mean and var)
        observation = tf.keras.Input(shape=(1,5), name='observation')
        prev_state = tf.keras.Input(shape=(1,), name='prev_state')
        # latent_state = tf.keras.layers.Bidirectional( # TODO(slu): check bidirectional functionality, bidirectional or single direction and add s_t-1?
        #     tf.keras.layers.LSTM(units=10, input_shape=(1,5)), name='latent_state'
        # )(observation)
        latent_state = tf.keras.layers.LSTM(units=10, input_shape=(1, 5), name = 'latent_state')(observation)
        latent_prev_states = tf.keras.layers.concatenate([latent_state, prev_state])
        curr_state = tf.keras.layers.Dense(units=self.latent_dim*2, activation='relu', name='curr_state')(latent_prev_states)
        self.posterior_model = tf.keras.Model(inputs=[observation, prev_state], outputs=curr_state)

        self.next_state_model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.keras.layers.Dense(units=self.latent_dim, activation='linear')
            ]
        )

        self.observations_model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.layers.Dense(units=100, activation='relu'),
                tf.layers.Dense(units=100, activation='relu'),
                tf.layers.Dense(units=100, activation='relu'),
                tf.layers.Dense(units=3, activation='relu'),
            ]
        )

        self.rewards_model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.layers.Dense(units=100, activation='relu'),
                tf.layers.Dense(units=100, activation='relu'),
                tf.layers.Dense(units=100, activation='relu'),
                tf.layers.Dense(units=1, activation='sigmoid'),
            ]
        )

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random_normal(shape=(100, self.latent_dim))
        return self.generate(eps, apply_sigmoid=True)

    def infer(self, x):
        mean, logvar = tf.split(self.posterior_model(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random_normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    # For robot experiment, generate all 3 inference networks at once
    def generate(self, z, apply_sigmoid=False):
        state_logits = self.next_state_model(z)
        observations_logits = self.observations_model(z)
        rewards_logits = self.rewards_model(z)

        if apply_sigmoid:
            state_probs = tf.sigmoid(state_logits)
            observation_probs = tf.sigmoid(observations_logits)  # TODO(slu): need discrete observations?
            reward_probs = tf.sigmoid(rewards_logits)
            return state_probs, observation_probs, reward_probs

        return state_logits, observations_logits, rewards_logits


if __name__ == "__main__":
    print("Using Tensorflow Version: {}".format(tf.VERSION))
    print("Using Keras Version: {}".format(tf.keras.__version__))
    dataset = generate_data(10)
    model = DELIP_model(latent_dim=4)
    print(model.posterior_model.summary())
    # print(model.state_generator.summary())
    # print(model.observations_generator.summary())
    # print(model.rewards_generator.summary())

    # TODO(slu): Dummy data, remove later
    dataset_it = dataset.make_one_shot_iterator()
    temp = dataset_it.get_next()
    data_tensor = dataset_it.get_next()[0,]
    data_tensor = tf.reshape(data_tensor, (1,1,5))
    state_tensor = tf.constant([[1]], dtype=tf.float32)

    gaussians = model.infer([data_tensor, state_tensor])
    print("Gaussian Params: {}".format(gaussians))
    samples = model.reparameterize(gaussians[0], gaussians[1])
    print("Samples: {}".format(samples))
    state, observations, rewards = model.generate(samples)
    print("State: {}".format(state))
    print("Observations: {}".format(observations))
    print("Rewards: {}".format(rewards))


