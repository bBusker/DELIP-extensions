import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras as K
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Bidirectional, Dense, Lambda, LSTM, TimeDistributed
import robot_doors_experiment as exp
import time
import random

#tf.enable_eager_execution()
tfe = tf.contrib.eager
tfd = tfp.distributions

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
        latent_state = TimeDistributed(Dense(units=self.latent_dim*2, activation='softplus', bias_initializer=K.initializers.zeros(), kernel_initializer=K.initializers.zeros(),  kernel_constraint=K.constraints.max_norm(0.5)), name='latent_state')(rnn_state)
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
    print("GENERATE_DATASET IS CURRENTLY MAKING NP ARRAYS LISTS FOR RETURN_DATA")
    data = np.array(data, dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    if return_data:
        return data
    return dataset


def generate_batches(trajectories, batch_size):
    # 5 Buckets
    cum_rew_per_traj = np.sum(trajectories, axis=1)[:,3]
    sorted_data = [trajectory for cum_rew, trajectory in sorted(zip(cum_rew_per_traj, trajectories), key=lambda x: x[0])]
    buckets = []
    for i in range(5):
        buckets.append(sorted_data[len(sorted_data)//5*i:len(sorted_data)//5*(i+1)])

    while True:
        train_data = []
        for i in range(batch_size//5):
            for bucket in buckets:
                train_data.append(random.choice(bucket))
        train_data = np.stack(train_data)
        target_data = [train_data[:, :, 0:3], train_data[:, :, 3:4], train_data[:, :, :-1]]
        yield (train_data, target_data)

def sd_sampling(input):
    mean, sd = tf.split(input, num_or_size_splits=2, axis=1)
    eps = tf.random_normal(shape=tf.shape(mean))
    return eps * sd + mean


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
    total_batch_size = 2000
    batch_size = 100
    epochs = 10000

    adam_lr = 1e-3
    adam_b1 = 0.9
    adam_b2 = 0.999
    adam_epsilon = 1e-8

    model = DELIP_model(latent_dim=4, input_timesteps=trajectory_timesteps)
    train_data = generate_dataset(steps=trajectory_timesteps, episodes=total_batch_size, return_data=True)
    adam_optimizer = K.optimizers.Adam(lr=adam_lr, beta_1=adam_b1, beta_2=adam_b2, epsilon=adam_epsilon)
    checkpointer = K.callbacks.ModelCheckpoint(filepath='./trained_models/DELIP_vaemodel_ep:{epoch:02d}_loss:{loss:.2f}.hdf5',
                                               verbose=1,
                                               save_weights_only=True,
                                               period=5)

    model.vae_model.compile(optimizer=adam_optimizer,
                            loss=[custom_loss_obs, custom_loss_rew, custom_loss_latent],
                            loss_weights=[1,1,1])
    # model.vae_model.fit(x=train_data,
    #                     y=[train_data[:,:,0:3], train_data[:,:,3:4], train_data[:,:,:-1]],
    #                     steps_per_epoch=total_batch_size//batch_size,
    #                     batch_size=batch_size,
    #                     epochs=epochs,
    #                     callbacks=[checkpointer])

    model.vae_model.fit_generator(generator=generate_batches(train_data, batch_size),
                                  steps_per_epoch=total_batch_size // batch_size,
                                  epochs=epochs,
                                  callbacks=[checkpointer])

    model.vae_model.save("trained_model_vae.h5", include_optimizer=False)


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
    print(model.posterior_model.summary())
    K.utils.plot_model(model.decoder_model, to_file='vae_encoder.png', show_shapes=True)
    print(model.decoder_model.summary())
    K.utils.plot_model(model.posterior_model, to_file='vae_decoder.png', show_shapes=True)

    model.vae_model.load_weights("./trained_models/DELIP_vaemodel_ep:30_loss:-3.31.hdf5", by_name=True)
    print("hello")

    # dataset = generate_dataset(steps=trajectory_timesteps, episodes=total_batch_size, return_data=True)
    # temp = generate_batches(dataset, batch_size=5)
    # for item in temp:
    #     print(item)


if __name__ == "__main__":
    print("Using Tensorflow Version: {}".format(tf.VERSION))
    print("Using Keras Version: {}".format(K.__version__))
    #test_model("trained_model_vae.h5")
    training2()


