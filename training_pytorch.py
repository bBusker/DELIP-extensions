from model_pytorch import DELIP_model
import torch
from torch import nn
from torch import optim
import robot_doors_experiment as exp
import numpy as np

def generate_bucketed_data(steps, episodes):
    # 5 Buckets
    num_buckets=5

    trajectories = exp.generate_data(steps=steps, episodes=episodes)
    trajectories = np.array(trajectories, dtype=np.float32)
    cum_rew_per_traj = np.sum(trajectories, axis=1)[:,3]
    sorted_data = [torch.from_numpy(trajectory) for cum_rew, trajectory in sorted(zip(cum_rew_per_traj, trajectories), key=lambda x: x[0])]
    # buckets = []
    # for i in range(num_buckets):
    #     buckets.append(torch.from_numpy(sorted_data[len(sorted_data)//num_buckets*i:len(sorted_data)//num_buckets*(i+1)][0]))
    return sorted_data #buckets

def training():
    trajectory_timesteps = 100
    total_batch_size = 100
    epochs = 10000
    batch_size = 100
    adam_lr = 1e-3

    model = DELIP_model()

    optimizer = optim.Adam(model.parameters(), lr=adam_lr)
    loss_func = nn.MSELoss()
    train_buckets = generate_bucketed_data(episodes=total_batch_size, steps=trajectory_timesteps)

    try:
        print("beginning training")

        model.train()
        model.cuda()
        train_loss = 0
        # for batch_idx, (data, _) in enumerate(train_loader):
        for data in train_buckets:
            data = data.cuda().unsqueeze(0)
            optimizer.zero_grad()
            next_state, obs, rew, mu, logvar, sample = model((data[:,:,:4], data[:,:,4].unsqueeze(-1)))

            obs_loss = loss_func(obs, data[:,:,:3])
            rew_loss = loss_func(rew, data[:,:,3,].unsqueeze(-1))
            next_state_loss = loss_func(next_state[:,:-1,:], sample[:,1:,:])
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            total_loss = obs_loss + rew_loss + next_state_loss + KLD
            total_loss.backward()
            optimizer.step()
            print("Obs_loss: {} Rew_loss: {} Next_state_loss: {} KLD_loss: {}".format(obs_loss, rew_loss, next_state_loss, KLD))


    except KeyboardInterrupt:
        model.vae_model.save("DELIP_model_vae.hdf5")
        print("Saved models")

    return model

if __name__ == "__main__":
    training()
    print("hello")