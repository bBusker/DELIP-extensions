import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import robot_doors_experiment as exp
import time
import random
import pomcp

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


def compute_loss_combined(model, x, train_next_state):
    observations, rewards, next_state, latent_state, latent_sample = model.vae_model(x)

    observations_mean, observations_logvar = tf.split(observations, num_or_size_splits=2, axis=2)
    observations_sd = tf.exp(observations_logvar*0.5)
    observations_d = tfd.Normal(loc=observations_mean, scale=observations_sd, allow_nan_stats=False)
    observations_prob = observations_d.log_prob(x[:,:,0:3])

    rewards_mean, rewards_logvar = tf.split(rewards, num_or_size_splits=2, axis=2)
    rewards_sd = tf.exp(rewards_logvar*0.5)
    rewards_d = tfd.Normal(loc=rewards_mean, scale=rewards_sd, allow_nan_stats=False)
    rewards_prob = rewards_d.log_prob(x[:,:,3:4])

    if train_next_state:
        next_state_mean, next_state_logvar = tf.split(next_state[:,:-1,:], num_or_size_splits=2, axis=2)
        next_state_sd = tf.exp(next_state_logvar*0.5)
        next_state_d = tfd.Normal(loc=next_state_mean, scale=next_state_sd, allow_nan_stats=False)
        next_state_prob = next_state_d.log_prob(latent_sample[:,1:,:])

        if np.random.rand() > 0.9:
            print("mean:    {}".format(next_state_mean[0][0]))
            print("sd:      {}".format(next_state_sd[0][0]))
            print("actual1: {}".format(latent_sample[:, 1:, :][0][0]))
            print("actual0: {}".format(latent_sample[:, 0:, :][0][0]))
            print("action:  {}".format(x[:, :, -1:][0][0]))
    else:
        next_state_prob = tf.constant([0], dtype=tf.float32)

    latent_mean, latent_logvar = tf.split(latent_state, num_or_size_splits=2, axis=2)
    # latent_sd = tf.exp(latent_logvar*0.5)
    # prior = tfd.Normal(loc=tf.zeros(model.latent_dim), scale=tf.ones(model.latent_dim))
    # posterior = tfd.Normal(loc=latent_mean, scale=latent_sd, allow_nan_stats=False)
    # kl_divergence = tfd.kl_divergence(posterior, prior, allow_nan_stats=False)
    kl_divergence = -0.5 * tf.reduce_sum(1 + 2 * latent_logvar - latent_mean**2 - tf.exp(2 * latent_logvar), 2)

    # return -(tf.reduce_mean(observations_prob) + tf.reduce_mean(rewards_prob) - tf.reduce_mean(kl_divergence))
    return tf.reduce_mean(observations_prob), tf.reduce_mean(rewards_prob), tf.reduce_mean(next_state_prob), tf.reduce_mean(kl_divergence)


def compute_gradients(model, x, train_next_state):
    with tf.GradientTape() as tape:
        obs_prob, rew_prob, ns_prob, kl_loss = compute_loss_combined(model, x, train_next_state)
        total_loss = -obs_prob - rew_prob - ns_prob + kl_loss
    return tape.gradient(total_loss, model.trainable_variables), obs_prob, rew_prob, ns_prob, kl_loss


def training():
    trajectory_timesteps = 100
    total_batch_size = 300
    epochs = 10000
    batch_size = 100
    adam_lr = 1e-3

    model = DELIP.DELIP_model(latent_dim=2, input_timesteps=trajectory_timesteps)

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

        for epoch in range(1, 1000):
            bucket_iters = []
            for bucket in train_buckets:
                bucket.shuffle(total_batch_size//5)
                bucket_iters.append(bucket.batch(batch_size//5).make_one_shot_iterator())
            for i in range(total_batch_size//batch_size):
                train_batch = tf.concat([bucket_iters[0].get_next(), bucket_iters[1].get_next(), bucket_iters[2].get_next(), bucket_iters[3].get_next(), bucket_iters[4].get_next()], axis=0)
                gradients, obs_prob, rew_prob, ns_prob, kl_loss = compute_gradients(model, train_batch, True) # TODO:TRUEEEEEEEEEE
                optimizer.apply_gradients(zip(gradients, model.trainable_variables), global_step=None)  # TODO(slu): use model or model.vae_model???
                model.reset_states()
            print("epoch {} | obs_loss: {:.5f}, rew_loss: {:.5f}, ns_loss: {:.5f}, kl_loss: {:.5f}".format(epoch, -obs_prob, -rew_prob, -ns_prob, kl_loss))
            if epoch % 5 == 0:
                model.vae_model.save("./trained_models/DELIP_model_vae_ep{}_loss{:.2f}.hdf5".format(epoch, -obs_prob - rew_prob - ns_prob + kl_loss))

        for epoch in range(1000, epochs+1):
            bucket_iters = []
            for bucket in train_buckets:
                bucket.shuffle(total_batch_size//5)
                bucket_iters.append(bucket.batch(batch_size//5).make_one_shot_iterator())
            for i in range(total_batch_size//batch_size):
                train_batch = tf.concat([bucket_iters[0].get_next(), bucket_iters[1].get_next(), bucket_iters[2].get_next(), bucket_iters[3].get_next(), bucket_iters[4].get_next()], axis=0)
                gradients, obs_prob, rew_prob, ns_prob, kl_loss = compute_gradients(model, train_batch, True)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables), global_step=None)  # TODO(slu): use model or model.vae_model???
                model.reset_states()
            print("epoch {} | obs_loss: {:.5f}, rew_loss: {:.5f}, ns_loss: {:.5f}, kl_loss: {:.5f}".format(epoch, -obs_prob, -rew_prob, -ns_prob, kl_loss))
            if epoch % 5 == 0:
                model.vae_model.save("./trained_models/DELIP_model_vae_ep{}_loss{:.2f}.hdf5".format(epoch, -obs_prob - rew_prob - ns_prob + kl_loss))

    except KeyboardInterrupt:
        model.vae_model.save("DELIP_model_vae.hdf5")
        print("Saved models")

    return model


def test_model():
    timesteps = 50
    pomcp_timeout = 50
    model_filepath = './trained_models/DELIP_model_vae_ep7725_loss-4.92.hdf5'

    experiment = exp.RobotDoorsExperiment()
    experiment.load_model(model_filepath)

    planner = pomcp.POMCP(experiment.generate_step_DELIP, timeout=pomcp_timeout)
    planner.initialize(experiment.state_space, experiment.action_space, experiment.obs_space)

    cum_rew = 0
    print("Starting robot experiment with model {}".format(model_filepath))
    for i in range(timesteps):
        action = planner.Search()
        print(planner.tree.nodes[-1][:])
        print(action)
        experiment.take_action(action)
        cum_rew += experiment.get_reward()
        observation = experiment.get_observation_discrete()
        planner.tree.prune_after_action(action, observation)
        planner.UpdateBelief(action, observation)
    print("Model took {} timesteps with pomcp simtime {}. Got cumulative reward {}!".format(timesteps, pomcp_timeout, cum_rew))

def testing():
    timesteps = 100
    episodes = 10
    model_filepath = './trained_models/with_next_state_1layerNN/DELIP_model_vae_ep2645_loss-4.63.hdf5'

    #experiment = exp.RobotDoorsExperiment()
    data = exp.generate_data(timesteps, episodes)
    data = tf.constant(data, dtype=tf.float32)
    data_t = tf.data.Dataset.from_tensor_slices(data)

    model = DELIP.DELIP_model(latent_dim=4, input_timesteps=timesteps)
    model.vae_model.load_weights(model_filepath)

    for temp in data_t.batch(1):
        print(temp)
        obs, rew, ns, lst, lsa = model.vae_model(temp)
        obs_s = model.reparameterize_layer(obs[0]).numpy().astype(np.float64).round(1).tolist()
        rew_s = model.reparameterize_layer(rew[0]).numpy().astype(np.float64).round(1).tolist()
        print(obs_s)


    print("testing")


if __name__ == "__main__":
    testing()