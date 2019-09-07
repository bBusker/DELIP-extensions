from model_pytorch import DELIP_model
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from robot_doors_dataset import RobotDoorsDataset

def training():
    trajectory_timesteps = 150
    total_episodes = 2000
    epochs = 10000
    batch_size = 20
    adam_lr = 1e-3
    print_freq = 25

    model = DELIP_model()

    optimizer = optim.Adam(model.parameters(), lr=adam_lr)
    loss_func = nn.MSELoss()

    print("initializing dataset")
    train_dataset = RobotDoorsDataset(total_episodes, trajectory_timesteps)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, collate_fn=cat_collate)

    try:
        print("beginning training")

        model.train()
        model.cuda()

        # for batch_idx, (data, _) in enumerate(train_loader):
        for i in range(1, epochs+1):
            obs_loss_total = 0
            rew_loss_total = 0
            next_state_loss_total = 0
            KLD_loss_total = 0
            data_iter = iter(train_dataloader)

            for data in data_iter:
                optimizer.zero_grad()
                next_state, obs, rew, mu, logvar, sample = model((data[:,:,:4], data[:,:,4].unsqueeze(-1)))

                obs_loss = loss_func(obs, data[:,:,:3])
                rew_loss = loss_func(rew, data[:,:,3,].unsqueeze(-1))
                next_state_loss = 10*loss_func(next_state[:,:-1,:], sample[:,1:,:])
                KLD_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                obs_loss_total += obs_loss.detach().item()
                rew_loss_total += rew_loss.detach().item()
                next_state_loss_total += next_state_loss.detach().item()
                KLD_loss_total += KLD_loss.detach().item()

                total_loss = obs_loss + rew_loss + next_state_loss + KLD_loss
                total_loss.backward()
                optimizer.step()

            if i%print_freq == 0:
                print("Epoch: {} Obs_loss: {} Rew_loss: {} Next_state_loss: {} KLD_loss: {}".format(i, obs_loss_total, rew_loss_total, next_state_loss_total, KLD_loss_total))
                print(next_state[0][5][:].detach().cpu().numpy())
                print(sample[0][6][:].detach().cpu().numpy())


    except KeyboardInterrupt:
        pass

    return model

def cat_collate(batch):
    return torch.cat(batch, axis=0)


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    training()
    print("hello")