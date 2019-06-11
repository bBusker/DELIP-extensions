import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
import robot_doors_experiment as exp
import time

tf.enable_eager_execution()
tfe = tf.contrib.eager
tfd = tfp.distributions


class DELIP_model(tf.keras.Model):
    def __init__(self, latent_dim=4): # Eliminate magic numbers
        super(DELIP_model, self).__init__()
        self.latent_dim = latent_dim
        self.last_latent_state = None

        # Use functional interface to define posterior model.
        # Inputs: observation and previous state, Outputs: current state as variational posterior (mean and var)
        observation = tf.keras.Input(shape=(1,5), name='observation')
        prev_state = tf.keras.Input(shape=(self.latent_dim,), name='prev_state')
        # latent_state = tf.keras.layers.Bidirectional( # TODO(slu): check bidirectional functionality, bidirectional or single direction and add s_t-1?
        #     tf.keras.layers.LSTM(units=10, input_shape=(1,5)), name='latent_state'
        # )(observation)
        latent_state = tf.keras.layers.LSTM(units=10, input_shape=(1, 5), name = 'latent_state')(observation)
        latent_prev_states = tf.keras.layers.concatenate([latent_state, prev_state])
        curr_state = tf.keras.layers.Dense(units=self.latent_dim*2, activation='softplus', name='curr_state')(latent_prev_states)
        # TODO(slu): softplus? check danijar
        self.posterior_model = tf.keras.Model(inputs=[observation, prev_state], outputs=curr_state)

        self.next_state_model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.keras.layers.Dense(units=self.latent_dim*2, activation='linear')
            ]
        )

        self.observations_model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.layers.Dense(units=100, activation='relu'),
                tf.layers.Dense(units=100, activation='relu'),
                tf.layers.Dense(units=100, activation='relu'),
                tf.layers.Dense(units=3*2, activation='softplus'),
            ]
        )

        self.rewards_model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.layers.Dense(units=100, activation='relu'),
                tf.layers.Dense(units=100, activation='relu'),
                tf.layers.Dense(units=100, activation='relu'),
                tf.layers.Dense(units=1*2, activation='sigmoid'),
            ]
        )

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random_normal(shape=(100, self.latent_dim))
        return self.generate(eps, apply_sigmoid=True)

    def infer(self, x):
        self.last_latent_state = self.posterior_model(x)
        mean, logvar = tf.split(self.last_latent_state, num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random_normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    # For robot experiment, mebe generate all 3 inference networks at once
    def generate_nextstate(self, code):
        state_logits = self.next_state_model(code)
        state_mean, state_logvar = tf.split(state_logits, num_or_size_splits=2, axis=1)
        return state_mean, state_logvar

    def generate_observations(self, code):
        observations_logits = self.observations_model(code)
        observations_mean, observations_logvar = tf.split(observations_logits, num_or_size_splits=2, axis=1) # TODO(slu): need discrete observations?
        return observations_mean, observations_logvar

    def generate_rewards(self, code):
        rewards_logits = self.rewards_model(code)
        rewards_mean, rewards_logvar = tf.split(rewards_logits, num_or_size_splits=2, axis=1)
        return rewards_mean, rewards_logvar


def generate_dataset(steps=10, episodes=10):
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


def compute_loss_combined(model, x, prev_state):
    prior = tfd.Normal(loc=tf.zeros(model.latent_dim), scale=tf.ones(model.latent_dim))
    mean, log_var = model.infer([x, prev_state])
    posterior = tfd.Normal(loc=mean, scale=log_var, allow_nan_stats=False)  # TODO(slu): log_var or var
    latent_sample = model.reparameterize(mean, log_var)

    state_loss = tf.constant([0], dtype=tf.float32)  # TODO(slu): compare to next or current state?

    observations_loc, observations_scale = model.generate_observations(latent_sample)
    observations_d = tfd.Normal(loc=observations_loc, scale=observations_scale, allow_nan_stats=False)
    observations_loss = observations_d.log_prob(x[0,0,0:3])

    rewards_loc, rewards_scale = model.generate_rewards(latent_sample)
    rewards_d = tfd.Normal(loc=rewards_loc, scale=rewards_scale, allow_nan_stats=False)
    rewards_loss = rewards_d.log_prob(x[0,0,3:4])

    kl_divergence = tfd.kl_divergence(posterior, prior, allow_nan_stats=False)

    return tf.reduce_mean(state_loss) + tf.reduce_mean(observations_loss) + tf.reduce_mean(rewards_loss) - tf.reduce_mean(kl_divergence)  # TODO(slu): reduce_mean or indiviually?


def compute_gradients(model, x, prev_state):
    with tf.GradientTape() as tape:
        loss = compute_loss_combined(model, x, prev_state)
    return tape.gradient(loss, model.trainable_variables), loss


def training():
    model = DELIP_model(latent_dim=4)
    optimizer = tf.train.AdamOptimizer(1e-4)
    train_dataset = generate_dataset(steps=10, episodes=10)
    test_dataset = generate_dataset(steps=10, episodes=10)
    epochs = 100

    prev_state = [[0]*model.latent_dim]

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_dataset:
            gradients, loss = compute_gradients(model, train_x, prev_state)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables), global_step=None)
            prev_state = model.last_latent_state
        end_time = time.time()

        if epoch % 1 == 0:
            loss = tfe.metrics.Mean()
            for test_x in test_dataset:
                loss(compute_loss(model, test_x))
            elbo = -loss.result()
            print('Epoch: {}, Test set ELBO: {}, '
                  'time elapse for current epoch {}'.format(epoch,
                                                            elbo,
                                                            end_time - start_time))

    return model


def testing():
    dataset = generate_dataset(10)
    model = DELIP_model(latent_dim=4)
    print(model.posterior_model.summary())
    # print(model.state_generator.summary())
    # print(model.observations_generator.summary())
    # print(model.rewards_generator.summary())

    # TODO(slu): Dummy data, remove later
    dataset_it = dataset.make_one_shot_iterator()
    temp = dataset_it.get_next()
    data_tensor = dataset_it.get_next()[0,]
    data_tensor = tf.reshape(data_tensor, (1, 1, 5))
    state_tensor = tf.constant([[1]], dtype=tf.float32)

    gaussians = model.infer([data_tensor, state_tensor])
    print("Gaussian Params: {}".format(gaussians))
    samples = model.reparameterize(gaussians[0], gaussians[1])
    print("Samples: {}".format(samples))
    # state, observations, rewards = model.generate(samples)
    # print("State: {}".format(state))
    # print("Observations: {}".format(observations))
    # print("Rewards: {}".format(rewards))

    print(compute_loss_combined(model, data_tensor, state_tensor))
    print("Done")


if __name__ == "__main__":
    print("Using Tensorflow Version: {}".format(tf.VERSION))
    print("Using Keras Version: {}".format(tf.keras.__version__))
    testing()


