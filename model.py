import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras as K
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Bidirectional, Dense, Lambda, LSTM, TimeDistributed, CuDNNLSTM, Concatenate
import robot_doors_experiment as exp
import time
import random

tf.enable_eager_execution()
tfe = tf.contrib.eager
tfd = tfp.distributions

def create_vae(input_timesteps, latent_dim, reparameterize_layer):
    # Create NN Graph
    timestep_data = Input(shape=(input_timesteps, 5), name='timestep_data')
    # action_data = Input(shape=(input_timesteps, 1), name='action_data')
    action_data = Lambda(lambda x: x[:, :, -1:], output_shape=(1,), name='action_data')(timestep_data)

    #combined_data = Concatenate()([timestep_data, action_data])

    rnn_state = Bidirectional(CuDNNLSTM(units=10, input_shape=(input_timesteps, 5), return_sequences=True),name='rnn_state')(timestep_data)
    latent_state = TimeDistributed(Dense(units=latent_dim * 2, bias_initializer=K.initializers.zeros(),kernel_initializer=K.initializers.zeros(), kernel_constraint=K.constraints.max_norm(0.5)), name='latent_state')(rnn_state)
    latent_sample = TimeDistributed(Lambda(reparameterize_layer, output_shape=(latent_dim,)),name='latent_sample')(latent_state)

    observations = TimeDistributed(Dense(units=100, activation='relu'), name='obs1')(latent_sample)
    observations = TimeDistributed(Dense(units=100, activation='relu'), name='obs2')(observations)
    observations = TimeDistributed(Dense(units=100, activation='relu'), name='obs3')(observations)
    observations = TimeDistributed(Dense(units=3 * 2), name='obs_out')(observations)

    rewards = TimeDistributed(Dense(units=100, activation='relu'), name='rew1')(latent_sample)
    rewards = TimeDistributed(Dense(units=100, activation='relu'), name='rew2')(rewards)
    rewards = TimeDistributed(Dense(units=100, activation='relu'), name='rew3')(rewards)
    rewards = TimeDistributed(Dense(units=1 * 2), name='rew_out')(rewards)

    state_action = Concatenate()([latent_sample, action_data])
    next_state = TimeDistributed(Dense(units=100, activation='relu'), name='ns1')(state_action)
    #next_state = TimeDistributed(Dense(units=100, activation='relu'), name='ns2')(next_state)
    #next_state = TimeDistributed(Dense(units=100, activation='relu'), name='ns3')(next_state)
    next_state = TimeDistributed(Dense(units=latent_dim * 2), name='next_state')(next_state)

    # Create VAE Model
    return Model(inputs=timestep_data,
                 outputs=[observations, rewards, next_state, latent_state, latent_sample])


def create_decoder(input_timesteps, latent_dim):
    decoder_in = K.layers.Input(shape=(input_timesteps, latent_dim,), name='latent_sample_in')
    action_data = Input(shape=(input_timesteps, 1), name='action_data')
    observations = TimeDistributed(Dense(units=100, activation='relu'), name='obs1')(decoder_in)
    observations = TimeDistributed(Dense(units=100, activation='relu'), name='obs2')(observations)
    observations = TimeDistributed(Dense(units=100, activation='relu'), name='obs3')(observations)
    observations = TimeDistributed(Dense(units=3 * 2), name='obs_out')(observations)

    rewards = TimeDistributed(Dense(units=100, activation='relu'), name='rew1')(decoder_in)
    rewards = TimeDistributed(Dense(units=100, activation='relu'), name='rew2')(rewards)
    rewards = TimeDistributed(Dense(units=100, activation='relu'), name='rew3')(rewards)
    rewards = TimeDistributed(Dense(units=1 * 2), name='rew_out')(rewards)

    state_action = Concatenate()([decoder_in, action_data])
    next_state = TimeDistributed(Dense(units=100, activation='relu'), name='ns1')(state_action)
    #next_state = TimeDistributed(Dense(units=100, activation='relu'), name='ns2')(next_state)
    #next_state = TimeDistributed(Dense(units=100, activation='relu'), name='ns3')(next_state)
    next_state = TimeDistributed(Dense(units=latent_dim * 2), name='next_state')(next_state)

    return Model(inputs=[decoder_in, action_data],
                 outputs=[observations, rewards, next_state])


class DELIP_model(Model):
    def __init__(self, latent_dim, input_timesteps): # TODO(slu): Eliminate magic numbers
        super(DELIP_model, self).__init__()

        # Configuration Parameters
        self.input_timesteps = input_timesteps
        self.latent_dim = latent_dim

        # # Create NN Graph
        # timestep_data = Input(shape=(self.input_timesteps, 4), name='timestep_data')
        # action_data = Input(shape=(self.input_timesteps, 1), name='action_data')
        # combined_data = Concatenate()([timestep_data, action_data])
        #
        # rnn_state = Bidirectional(CuDNNLSTM(units=10, input_shape=(self.input_timesteps, 5), return_sequences=True),name='rnn_state')(combined_data)
        # latent_state = TimeDistributed(Dense(units=self.latent_dim*2, bias_initializer=K.initializers.zeros(), kernel_initializer=K.initializers.zeros(),  kernel_constraint=K.constraints.max_norm(0.5)), name='latent_state')(rnn_state)
        # latent_sample = TimeDistributed(Lambda(self.reparameterize_layer, output_shape=(self.latent_dim,)),name='latent_sample')(latent_state)
        #
        # observations = TimeDistributed(Dense(units=100, activation='relu'),name='obs1')(latent_sample)
        # observations = TimeDistributed(Dense(units=100, activation='relu'),name='obs2')(observations)
        # observations = TimeDistributed(Dense(units=100, activation='relu'),name='obs3')(observations)
        # observations = TimeDistributed(Dense(units=3 * 2),name='obs_out')(observations)
        #
        # rewards = TimeDistributed(Dense(units=100, activation='relu'),name='rew1')(latent_sample)
        # rewards = TimeDistributed(Dense(units=100, activation='relu'),name='rew2')(rewards)
        # rewards = TimeDistributed(Dense(units=100, activation='relu'),name='rew3')(rewards)
        # rewards = TimeDistributed(Dense(units=1 * 2),name='rew_out')(rewards)
        #
        # state_action = Concatenate()([latent_sample, action_data])
        # next_state = TimeDistributed(Dense(units=self.latent_dim*2 + 1, activation='linear'), name='next_state')(state_action)
        #
        # # Create Posterior and Decoder Models
        # # self.posterior_model = Model(inputs=[timestep_data, action_data], outputs=[latent_sample, action_data], name='encoder')
        # # self.decoder_model = Model(inputs=[decoder_in, decoder_action_in], outputs=[observations, rewards, next_state], name='decoder')
        #
        # # Create VAE Model
        # # vae = self.decoder_model(self.posterior_model(timestep_data))
        # # self.vae_model = Model(inputs=timestep_data, outputs=vae+[latent_state]+[latent_sample])
        # self.vae_model = Model(inputs=[timestep_data, action_data], outputs=[observations, rewards, next_state, latent_state, latent_sample])
        #
        # decoder_in = K.layers.Input(shape=(self.input_timesteps, self.latent_dim,), name='latent_sample_in')
        # action_data = Input(shape=(self.input_timesteps, 1), name='action_data')
        # observations = TimeDistributed(Dense(units=100, activation='relu'), name='obs1')(decoder_in)
        # observations = TimeDistributed(Dense(units=100, activation='relu'), name='obs2')(observations)
        # observations = TimeDistributed(Dense(units=100, activation='relu'), name='obs3')(observations)
        # observations = TimeDistributed(Dense(units=3 * 2), name='obs_out')(observations)
        #
        # rewards = TimeDistributed(Dense(units=100, activation='relu'), name='rew1')(decoder_in)
        # rewards = TimeDistributed(Dense(units=100, activation='relu'), name='rew2')(rewards)
        # rewards = TimeDistributed(Dense(units=100, activation='relu'), name='rew3')(rewards)
        # rewards = TimeDistributed(Dense(units=1 * 2), name='rew_out')(rewards)
        #
        # state_action = Concatenate()([decoder_in, action_data])
        # next_state = TimeDistributed(Dense(units=self.latent_dim * 2 + 1, activation='linear'), name='next_state')(
        #     state_action)

        self.vae_model = create_vae(self.input_timesteps, self.latent_dim, self.reparameterize_layer)
        self.decoder_model = create_decoder(self.input_timesteps, self.latent_dim)

    def reparameterize_w_logvar(self, mean, logvar):
        eps = tf.random_normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean

    def reparameterize_w_sd(self, mean, sd):
        eps = tf.random_normal(shape=tf.shape(mean))
        return eps * sd + mean

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


def test_model(filepath="trained_model.h5"):
    timesteps = 100
    model = DELIP_model(latent_dim=4, input_timesteps=None)
    model.vae_model.load_weights(filepath, by_name=True)
    test_dataset = generate_dataset(steps=timesteps, episodes=10)
    for trajectory in test_dataset.batch(1):
        observations, rewards, latents = model.vae_model(trajectory)
        observations_sample = sd_sampling(tf.reshape(observations,(timesteps,6)))
        rewards_sample = sd_sampling(tf.reshape(rewards,(timesteps,2)))
        for i in range(100):
            print("model: {}".format(observations_sample[i]))
            print("actual: {}".format(trajectory[0][i][0:3]))
        print("done")
    print("hello")


def testing():
    print(tf.test.gpu_device_name())
    print(tf.test.is_gpu_available())

    trajectory_timesteps = 10
    total_batch_size = 100

    model = DELIP_model(latent_dim=4, input_timesteps=trajectory_timesteps)
    print(model.vae_model.summary())
    K.utils.plot_model(model.vae_model, to_file='vae.png', show_shapes=True)
    print(model.decoder_model.summary())
    K.utils.plot_model(model.decoder_model, to_file='vae_decoder.png', show_shapes=True)
    # print(model.posterior_model.summary())
    # K.utils.plot_model(model.posterior_model, to_file='vae_decoder.png', show_shapes=True)

    #model.vae_model.load_weights("./trained_models/DELIP_vaemodel_ep:30_loss:-3.31.hdf5", by_name=True)
    print("hello")

    # dataset = generate_dataset(steps=trajectory_timesteps, episodes=total_batch_size, return_data=True)
    # temp = generate_batches(dataset, batch_size=5)
    # for item in temp:
    #     print(item)


if __name__ == "__main__":
    print("Using Tensorflow Version: {}".format(tf.VERSION))
    print("Using Keras Version: {}".format(K.__version__))
    #test_model("trained_model_vae.h5")
    testing()


