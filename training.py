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
    return buckets


def compute_loss_combined(model, x):
    observations, rewards, next_state, latent_state, latent_sample = model.vae_model(x)

    observations_mean, observations_logvar = tf.split(observations, num_or_size_splits=2, axis=2)
    observations_sd = tf.exp(observations_logvar*0.5)
    observations_d = tfd.Normal(loc=observations_mean, scale=observations_sd, allow_nan_stats=False)
    observations_prob = observations_d.log_prob(x[:,:,0:3])

    rewards_mean, rewards_logvar = tf.split(rewards, num_or_size_splits=2, axis=2)
    rewards_sd = tf.exp(rewards_logvar*0.5)
    rewards_d = tfd.Normal(loc=rewards_mean, scale=rewards_sd, allow_nan_stats=False)
    rewards_prob = rewards_d.log_prob(x[:,:,3:4])

    next_state_mean, next_state_logvar = tf.split(next_state[:,:-1,:], num_or_size_splits=2, axis=2)
    next_state_sd = tf.exp(next_state_logvar*0.5)
    next_state_d = tfd.Normal(loc=next_state_mean, scale=next_state_sd, allow_nan_stats=False)
    next_state_prob = next_state_d.log_prob(latent_sample[:,1:,:])

    latent_mean, latent_logvar = tf.split(latent_state, num_or_size_splits=2, axis=2)
    # latent_sd = tf.exp(latent_logvar*0.5)
    # prior = tfd.Normal(loc=tf.zeros(model.latent_dim), scale=tf.ones(model.latent_dim))
    # posterior = tfd.Normal(loc=latent_mean, scale=latent_sd, allow_nan_stats=False)
    # kl_divergence = tfd.kl_divergence(posterior, prior, allow_nan_stats=False)
    kl_divergence = -0.5 * tf.reduce_sum(1 + 2 * latent_logvar - latent_mean**2 - tf.exp(2 * latent_logvar), 2)

    # return -(tf.reduce_mean(observations_prob) + tf.reduce_mean(rewards_prob) - tf.reduce_mean(kl_divergence))
    return tf.reduce_mean(observations_prob), tf.reduce_mean(rewards_prob), tf.reduce_mean(next_state_prob), tf.reduce_mean(kl_divergence)


def compute_gradients(model, x):
    with tf.GradientTape() as tape:
        obs_prob, rew_prob, ns_prob, kl_loss = compute_loss_combined(model, x)
        total_loss = -obs_prob - rew_prob - ns_prob + kl_loss
    return tape.gradient(total_loss, model.trainable_variables), obs_prob, rew_prob, ns_prob, kl_loss


def training():
    trajectory_timesteps = 100
    total_batch_size = 1000
    epochs = 10000
    batch_size = 100
    adam_lr = 1e-3

    model = DELIP.DELIP_model(latent_dim=4, input_timesteps=trajectory_timesteps)

    optimizer = tf.train.AdamOptimizer(adam_lr)
    # test_data = exp.generate_data(steps=trajectory_timesteps, episodes=total_batch_size)
    # test_data = tf.constant(test_data, dtype=tf.float32)
    # test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
    train_buckets = generate_bucketed_data(batch_size=batch_size, episodes=total_batch_size, steps=trajectory_timesteps)

    try:
        print("beginning training")
        # for epoch in range(1, epochs+1):
        #     for batch in test_dataset.batch(batch_size):
        #         gradients, obs_prob, rew_prob, state_prob, kl_loss = compute_gradients(model, batch)
        #         optimizer.apply_gradients(zip(gradients, model.trainable_variables),
        #                                   global_step=None)  # TODO(slu): use model or model.vae_model???
        #         model.reset_states()
        #     print("epoch {} | obs_loss: {}, rew_loss: {}, ns_loss: {}, kl_loss: {}".format(epoch, -obs_prob, -rew_prob, -state_prob, kl_loss))


        for epoch in range(1, epochs + 1):
            bucket_iters = []
            gradients, obs_prob, rew_prob, ns_prob, kl_loss = None, 0, 0, 0, 0
            for bucket in train_buckets:
                bucket.shuffle(total_batch_size//5)
                bucket_iters.append(bucket.batch(batch_size//5).make_one_shot_iterator())
            for i in range(total_batch_size//batch_size):
                train_batch = tf.concat([bucket_iters[0].get_next(), bucket_iters[1].get_next(), bucket_iters[2].get_next(), bucket_iters[3].get_next(), bucket_iters[4].get_next()], axis=0)
                gradients, obs_prob, rew_prob, ns_prob, kl_loss = compute_gradients(model, train_batch)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables), global_step=None)  # TODO(slu): use model or model.vae_model???
                model.reset_states()
            print("epoch {} | obs_loss: {}, rew_loss: {}, ns_loss: {}, kl_loss: {}".format(epoch, -obs_prob, -rew_prob, -ns_prob, kl_loss))
            if epoch % 5 == 0:
                model.vae_model.save("./trained_models/DELIP_model_vae_ep{}_loss{}".format(epoch, -obs_prob - rew_prob - ns_prob + kl_loss))
    except KeyboardInterrupt:
        model.vae_model.save("DELIP_model_vae")
        model.posterior_model.save("DELIP_model_posterior")
        model.decoder_model.save("DELIP_model_decoder")
        print("Saved models")

    return model


if __name__ == "__main__":
    training()