import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras as K
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Bidirectional, Dense, Lambda, LSTM, TimeDistributed
import robot_doors_experiment as exp
import time

tf.enable_eager_execution()
tfe = tf.contrib.eager
tfd = tfp.distributions

def create_qRNN_model(latent_dim, input_timesteps):
    # Use functional interface to define posterior model.
    # Inputs: observation and previous state, Outputs: current state as variational posterior (mean and var)
    observation = Input(shape=(input_timesteps, 5), name='observation')
    prev_state = Input(shape=(latent_dim,), name='prev_state')
    latent_state = LSTM(units=10, input_shape=(input_timesteps, 5), name = 'latent_state')(observation)
    latent_prev_states = K.layers.concatenate([latent_state, prev_state])
    curr_state = Dense(units=latent_dim * 2, activation='softplus', name='curr_state')(
        latent_prev_states)
    posterior_model = Model(inputs=[observation, prev_state], outputs=curr_state)
    return posterior_model

def create_qBRNN_model(latent_dim, input_timesteps):
    # Use functional interface to define posterior model.
    # Inputs: observation and previous state, Outputs: current state as variational posterior (mean and var)
    observation = Input(shape=(input_timesteps, 5), name='observation')
    rnn_state = Bidirectional(
        LSTM(units=10, input_shape=(input_timesteps, 5)), name='rnn_state'
    )(observation)
    latent_state = Dense(units=latent_dim * 2, activation='softplus', name='latent_state')(rnn_state)
    latent_sample = K.layers.Lambda(reparameterize, output_shape=(latent_dim,), name='latent_sample')(latent_state)
    observations = Dense(units=100, activation='relu', name='obs1')(latent_sample)
    tf.layers.Dense(units=100, activation='relu'),
    tf.layers.Dense(units=100, activation='relu'),
    tf.layers.Dense(units=100, activation='relu'),
    tf.layers.Dense(units=3 * 2, activation='softplus'),


    posterior_model = Model(inputs=observation, outputs=curr_state)
    decoder_model = Model(inputs=latent_state, outputs=[observations, rewards])
    return posterior_model


class DELIP_model(Model):
    def __init__(self, latent_dim, input_timesteps): # TODO(slu): Eliminate magic numbers
        super(DELIP_model, self).__init__()

        # Configuration Parameters
        self.input_timesteps = input_timesteps
        self.latent_dim = latent_dim
        self.last_latent_state = None

        # Create NN Graph
        timestep_data = Input(shape=(self.input_timesteps, 5), name='timestep_data')
        rnn_state = Bidirectional(LSTM(units=10, input_shape=(self.input_timesteps, 5), return_sequences=True),name='rnn_state')(timestep_data)
        latent_state = TimeDistributed(Dense(units=self.latent_dim*2, activation='softplus', bias_initializer=K.initializers.zeros, kernel_initializer=K.initializers.zeros),name='latent_state')(rnn_state)
        latent_sample = TimeDistributed(Lambda(self.reparameterize_layer, output_shape=(self.latent_dim,)),name='latent_sample')(latent_state)

        decoder_in = K.layers.Input(shape=(self.input_timesteps, self.latent_dim,), name='latent_sample_in')
        observations = TimeDistributed(Dense(units=100, activation='relu'),name='obs1')(decoder_in)
        observations = TimeDistributed(Dense(units=100, activation='relu'),name='obs2')(observations)
        observations = TimeDistributed(Dense(units=100, activation='relu'),name='obs3')(observations)
        observations = TimeDistributed(Dense(units=3 * 2, activation='softplus'),name='obs_out')(observations)

        rewards = TimeDistributed(Dense(units=100, activation='relu'),name='rew1')(decoder_in)
        rewards = TimeDistributed(Dense(units=100, activation='relu'),name='rew2')(rewards)
        rewards = TimeDistributed(Dense(units=100, activation='relu'),name='rew3')(rewards)
        rewards = TimeDistributed(Dense(units=1 * 2, activation='sigmoid'),name='rew_out')(rewards)

        # Create Posterior and Decoder Models
        self.posterior_model = Model(inputs=timestep_data, outputs=latent_sample, name='encoder')
        self.decoder_model = Model(inputs=decoder_in, outputs=[observations, rewards], name='decoder')

        # Create VAE Model
        vae = self.decoder_model(self.posterior_model(timestep_data))
        self.vae_model = Model(inputs=timestep_data, outputs=vae+[latent_state])

        # # Declare Generator Layers
        # observations0 = TimeDistributed(Dense(units=100, activation='relu'), name='obs0')
        # observations1 = TimeDistributed(Dense(units=100, activation='relu'), name='obs1')
        # observations2 = TimeDistributed(Dense(units=100, activation='relu'), name='obs2')
        # observations_out = TimeDistributed(Dense(units=3*2, activation='softplus'), name='obs_out')
        # 
        # rewards0 = TimeDistributed(Dense(units=100, activation='relu'), name='rew0')
        # rewards1 = TimeDistributed(Dense(units=100, activation='relu'), name='rew1')
        # rewards2 = TimeDistributed(Dense(units=100, activation='relu'), name='rew2')
        # rewards_out = TimeDistributed(Dense(units=1*2, activation='sigmoid'), name='rew_out')
        # 
        # # Create VAE Model
        # observations_vae = observations0(latent_sample)
        # observations_vae = observations1(observations_vae)
        # observations_vae = observations2(observations_vae)
        # observations_vae = observations_out(observations_vae)
        # rewards_vae = rewards0(latent_sample)
        # rewards_vae = rewards1(rewards_vae)
        # rewards_vae = rewards2(rewards_vae)
        # rewards_vae = rewards_out(rewards_vae)
        # self.vae_model = Model(inputs=timestep_data, outputs=[observations_vae, rewards_vae, latent_state], name='vae')
        # 
        # # Create Decoder Model
        # decoder_in = K.layers.Input(shape=(self.input_timesteps, self.latent_dim,), name='latent_sample_in')
        # observations_dec = observations0(decoder_in)
        # observations_dec = observations1(observations_dec)
        # observations_dec = observations2(observations_dec)
        # observations_dec = observations_out(observations_dec)
        # rewards_dec = rewards0(decoder_in)
        # rewards_dec = rewards1(rewards_dec)
        # rewards_dec = rewards2(rewards_dec)
        # rewards_dec = rewards_out(rewards_dec)
        # self.decoder_model = Model(inputs=decoder_in, outputs=[observations_dec, rewards_dec], name='decoder')

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

    def reparameterize_layer(self, input):
        mean, logvar = tf.split(input, num_or_size_splits=2, axis=1)
        eps = tf.random_normal(shape=tf.shape(mean))
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


def generate_dataset(steps=10, episodes=10, return_data=False):
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
    if return_data:
        return data
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

def compute_loss_combined(model, x):
    # TODO(slu): state loss
    prior = tfd.Normal(loc=tf.zeros(model.latent_dim), scale=tf.ones(model.latent_dim))

    latent_state = model.posterior_model(x)
    latent_mean, latent_logvar = tf.split(latent_state, num_or_size_splits=2, axis=2)
    latent_sd = tf.exp(latent_logvar*0.5)
    posterior = tfd.Normal(loc=latent_mean, scale=latent_sd, allow_nan_stats=False)  # TODO(slu): log_var or var NEW: LOG_VAR IS MAKING LOSS HORRIBLE??

    observations, rewards = model.decoder_model(latent_state)

    observations_mean, observations_logvar = tf.split(observations, num_or_size_splits=2, axis=2)
    observations_sd = tf.exp(observations_logvar*0.5)
    observations_d = tfd.Normal(loc=observations_mean, scale=observations_logvar, allow_nan_stats=False)
    observations_prob = observations_d.log_prob(x[:,:,0:3])

    rewards_mean, rewards_logvar = tf.split(rewards, num_or_size_splits=2, axis=2)
    rewards_sd = tf.exp(rewards_logvar*0.5)
    rewards_d = tfd.Normal(loc=rewards_mean, scale=rewards_logvar, allow_nan_stats=False)
    rewards_prob = rewards_d.log_prob(x[:,:,3:4])

    kl_divergence = tfd.kl_divergence(posterior, prior, allow_nan_stats=False)

    return -(tf.reduce_mean(observations_prob) + tf.reduce_mean(rewards_prob) - tf.reduce_mean(kl_divergence))  # TODO(slu): reduce_mean or indiviually?


def compute_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = compute_loss_combined(model, x)
    return tape.gradient(loss, model.trainable_variables), loss


def custom_loss_obs(y_true, y_pred):
    observations_mean, observations_logvar = tf.split(y_pred, num_or_size_splits=2, axis=2)
    observations_sd = tf.exp(observations_logvar * 0.5)
    observations_d = tfd.Normal(loc=observations_mean, scale=observations_logvar, allow_nan_stats=False)
    observations_prob = observations_d.log_prob(y_true)
    return -tf.reduce_mean(observations_prob)


def custom_loss_rew(y_true, y_pred):
    rewards_mean, rewards_logvar = tf.split(y_pred, num_or_size_splits=2, axis=2)
    rewards_sd = tf.exp(rewards_logvar * 0.5)
    rewards_d = tfd.Normal(loc=rewards_mean, scale=rewards_logvar, allow_nan_stats=False)
    rewards_prob = rewards_d.log_prob(y_true)
    return -tf.reduce_mean(rewards_prob)

def custom_loss_latent(y_true, y_pred):
    prior = tfd.Normal(loc=tf.zeros(4), scale=tf.ones(4))

    latent_state = y_pred
    latent_mean, latent_logvar = tf.split(latent_state, num_or_size_splits=2, axis=2)
    latent_sd = tf.exp(latent_logvar * 0.5)
    posterior = tfd.Normal(loc=latent_mean, scale=latent_sd, allow_nan_stats=False)
    kl_divergence = tfd.kl_divergence(posterior, prior, allow_nan_stats=False)
    return tf.reduce_mean(kl_divergence)

def training():
    trajectory_timesteps = 100
    total_batch_size = 1
    epochs = 10000
    adam_lr = 1e-4

    model = DELIP_model(latent_dim=4, input_timesteps=trajectory_timesteps)

    optimizer = tf.train.AdamOptimizer(adam_lr)
    test_dataset = generate_dataset(steps=trajectory_timesteps, episodes=total_batch_size)
    train_dataset = generate_dataset(steps=trajectory_timesteps, episodes=total_batch_size)

    try:
        print("beginning training")
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            for train_x in train_dataset.batch(100):
                gradients, loss = compute_gradients(model, train_x)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables), global_step=None)
                model.reset_states()
            end_time = time.time()
            print("loss: {}".format(loss))
            # if epoch % 1 == 0:
            #     loss = tfe.metrics.Mean()
            #     for test_x in test_dataset:
            #         loss(compute_loss(model, test_x))
            #     elbo = -loss.result()
            #     print('Epoch: {}, Test set ELBO: {}, '
            #           'time elapse for current epoch {}'.format(epoch,
            #                                                     elbo,
            #                                                     end_time - start_time))
    except KeyboardInterrupt:
        model.vae_model.save("DELIP_model_vae")
        model.posterior_model.save("DELIP_model_posterior")
        model.decoder_model.save("DELIP_model_decoder")
        print("Saved models")

    return model


def training2():
    trajectory_timesteps = 100
    total_batch_size = 500
    epochs = 10000
    adam_lr = 1e-4

    model = DELIP_model(latent_dim=4, input_timesteps=trajectory_timesteps)
    model.vae_model.compile(optimizer='adam',
                            loss=[custom_loss_obs, custom_loss_rew, custom_loss_latent],
                            loss_weights=[1,1,1])
    train_data = generate_dataset(steps=trajectory_timesteps, episodes=total_batch_size, return_data=True)

    model.vae_model.fit(x=train_data,
                        y=[train_data[:,:,0:3], train_data[:,:,3:4], train_data[:,:,:-1]],
                        batch_size=100,
                        steps_per_epoch=epochs)


def testing():
    trajectory_timesteps = 1
    total_batch_size = 1

    model = DELIP_model(latent_dim=4, input_timesteps=trajectory_timesteps)
    # print(model.vae_model.summary())
    # K.utils.plot_model(model.vae_model, to_file='vae.png', show_shapes=True)
    # print(model.posterior_model.summary())
    # K.utils.plot_model(model.decoder_model, to_file='vae_encoder.png', show_shapes=True)
    # print(model.decoder_model.summary())
    # K.utils.plot_model(model.posterior_model, to_file='vae_decoder.png', show_shapes=True)

    dataset = generate_dataset(steps=trajectory_timesteps, episodes=total_batch_size)
    for item in dataset.batch(1):
        res = model.posterior_model(item)
        print(res)
        res = model.vae_model(item)
        print(res)
        print(compute_loss_combined(model, item))


if __name__ == "__main__":
    print("Using Tensorflow Version: {}".format(tf.VERSION))
    print("Using Keras Version: {}".format(K.__version__))
    print(tf.test.gpu_device_name())
    print(tf.test.is_gpu_available())
    training2()


