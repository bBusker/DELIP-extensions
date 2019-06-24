import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import robot_doors_experiment as exp
import time
import random

import model as DELIP

tf.enable_eager_execution()
tfe = tf.contrib.eager
tfd = tfp.distributions


def generate_bucketed_data(batch_size, steps, episodes):
    # 5 Buckets
    num_buckets=5

    trajectories = exp.generate_data(steps=steps, episodes=episodes)
    trajectories = np.array(trajectories, dtype=np.float32)
    cum_rew_per_traj = np.sum(trajectories, axis=1)[:,3]
    sorted_data = [trajectory for cum_rew, trajectory in sorted(zip(cum_rew_per_traj, trajectories), key=lambda x: x[0])]
    buckets = []
    for i in range(num_buckets):
        buckets.append(tf.data.Dataset.from_tensor_slices(sorted_data[len(sorted_data)//num_buckets*i:
                                                                      len(sorted_data)//num_buckets*(i+1)]))

    # train_buckets = []
    # for bucket in buckets:
    #     train_buckets.append(bucket.batch(batch_size=batch_size//num_buckets))
    return buckets


def compute_loss_combined(model, x):
    # TODO(slu): state loss
    #prior = tfd.Normal(loc=tf.zeros(model.latent_dim), scale=tf.ones(model.latent_dim))

    observations, rewards, latent_state = model.vae_model(x)

    latent_mean, latent_logvar = tf.split(latent_state, num_or_size_splits=2, axis=2)
    #print("latent_mean:{}, latent_logvar:{}".format(latent_mean[0][0], latent_logvar[0][0]))
    #latent_sd = tf.exp(latent_logvar*0.5)
    #posterior = tfd.Normal(loc=latent_mean, scale=latent_sd, allow_nan_stats=False)

    observations_mean, observations_logvar = tf.split(observations, num_or_size_splits=2, axis=2)
    observations_sd = tf.exp(observations_logvar*0.5)
    #print("actual obs: {}".format(x[0][0][0:3]))
    #print("obs_mean: {}, obs_sd: {}".format(observations_mean[0][0], observations_sd[0][0]))
    observations_d = tfd.Normal(loc=observations_mean, scale=observations_sd, allow_nan_stats=False)
    observations_prob = observations_d.log_prob(x[:,:,0:3])

    rewards_mean, rewards_logvar = tf.split(rewards, num_or_size_splits=2, axis=2)
    rewards_sd = tf.exp(rewards_logvar*0.5)
    # print("rew_mean: {}, rew_sd: {}".format(rewards_mean[0][0], rewards_sd[0][0]))
    rewards_d = tfd.Normal(loc=rewards_mean, scale=rewards_sd, allow_nan_stats=False)
    rewards_prob = rewards_d.log_prob(x[:,:,3:4])

    # kl_divergence = tfd.kl_divergence(posterior, prior, allow_nan_stats=False)

    kl_divergence = -0.5 * tf.reduce_sum(1 + 2 * latent_logvar - latent_mean**2 - tf.exp(2 * latent_logvar), 2)

    # return -(tf.reduce_mean(observations_prob) + tf.reduce_mean(rewards_prob) - tf.reduce_mean(kl_divergence))  # TODO(slu): reduce_mean or indiviually?
    return tf.reduce_mean(observations_prob), tf.reduce_mean(rewards_prob), tf.reduce_mean(kl_divergence)


def compute_gradients(model, x):
    with tf.GradientTape() as tape:
        obs_prob, rew_prob, kl_loss = compute_loss_combined(model, x)
        total_loss = -obs_prob - rew_prob + kl_loss
    return tape.gradient(total_loss, model.trainable_variables), obs_prob, rew_prob, kl_loss


def training():
    trajectory_timesteps = 100
    total_batch_size = 1000
    epochs = 10000
    batch_size = 100
    adam_lr = 1e-3

    model = DELIP.DELIP_model(latent_dim=4, input_timesteps=trajectory_timesteps)

    optimizer = tf.train.AdamOptimizer(adam_lr)
    # test_dataset = generate_dataset(steps=trajectory_timesteps, episodes=total_batch_size)
    train_buckets = generate_bucketed_data(batch_size=batch_size, episodes=total_batch_size, steps=trajectory_timesteps)

    try:
        print("beginning training")
        for epoch in range(1, epochs + 1):
            bucket_iters = []
            for bucket in train_buckets:
                bucket.shuffle(total_batch_size//5)
                bucket_iters.append(bucket.batch(batch_size//5).make_one_shot_iterator())
            for i in range(total_batch_size//batch_size):
                train_batch = tf.concat([bucket_iters[0].get_next(), bucket_iters[1].get_next(), bucket_iters[2].get_next(), bucket_iters[3].get_next(), bucket_iters[4].get_next()], axis=0)
                gradients, obs_prob, rew_prob, kl_loss = compute_gradients(model.vae_model, train_batch)
                optimizer.apply_gradients(zip(gradients, model.vae_model.trainable_variables), global_step=None)  # TODO(slu): use model or model.vae_model???
                model.reset_states()
            print("epoch {} | obs_loss: {}, rew_loss: {}, kl_loss: {}".format(epoch, -obs_prob, -rew_prob, kl_loss))
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


if __name__ == "__main__":
    training()