import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras as K
import robot_doors_experiment as exp
import time
import random

import model as DELIP

tf.enable_eager_execution()
tfe = tf.contrib.eager
tfd = tfp.distributions


def generate_dataset(steps=10, episodes=10, numpy=False):
    data = []
    for i in range(episodes):
        experiment = exp.RobotDoorsExperiment()
        trajectory = []
        for j in range(steps):
            curr_action = np.random.choice([0,1,2])
            experiment.take_action(curr_action)
            # TODO(slu): standardize to discrete observations in generator for POMCP
            res = experiment.get_observation_discrete() + (experiment.get_reward(),) + (curr_action,)
            trajectory.append(res)
        data.append(trajectory)
    # dataset = tf.data.Dataset.from_tensor_slices(data)
    if numpy:
        return np.array(data, dtype=np.float32)
    return tf.constant(data)


def generate_batches_train2(trajectories, batch_size):
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
    total_batch_size = 2000
    batch_size = 100
    epochs = 10000

    adam_lr = 1e-3
    adam_b1 = 0.9
    adam_b2 = 0.999
    adam_epsilon = 1e-8

    model = DELIP.DELIP_model(latent_dim=4, input_timesteps=trajectory_timesteps)
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

    model.vae_model.fit_generator(generator=generate_batches_train2(train_data, batch_size),
                                  steps_per_epoch=total_batch_size // batch_size,
                                  epochs=epochs,
                                  callbacks=[checkpointer])

    model.vae_model.save("trained_model_vae.h5", include_optimizer=False)


if __name__ == "__main__":
    training()