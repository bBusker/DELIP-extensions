from model_pytorch import DELIP_model
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from robot_doors_dataset import RobotDoorsDataset
import sys
import pickle
import utils


def training():
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():  # If debugging
        trajectory_timesteps = 100
        total_episodes = 20
        epochs = 10000
        batch_size = 10
        adam_lr = 1e-3
        print_freq = 1
    else:
        trajectory_timesteps = 100
        total_episodes = 1000
        epochs = 15000
        batch_size = 20
        adam_lr = 5e-4
        print_freq = 25

    model = DELIP_model()

    optimizer = optim.Adam(model.parameters(), lr=adam_lr)
    loss_func = nn.MSELoss()
    kld_loss = nn.KLDivLoss()

    print("initializing dataset")
    train_dataset = RobotDoorsDataset(total_episodes, trajectory_timesteps)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, collate_fn=cat_collate, shuffle=True)

    plotter = utils.VisdomLinePlotter()

    try:
        print("beginning training")
        print("dataset size: {}".format(len(train_dataset)))

        model.train()
        model.cuda()

        # for batch_idx, (data, _) in enumerate(train_loader):
        for epoch in range(1, epochs+1):
            obs_loss_total = 0
            rew_loss_total = 0
            next_state_loss_total = 0
            KLD_loss_total = 0
            data_iter = iter(train_dataloader)

            for data in data_iter:
                optimizer.zero_grad()
                last_data = data
                next_state, obs, rew, mu, logvar, sample = model((data[:,:,:4], data[:,:,4].unsqueeze(-1)))

                obs_loss = loss_func(obs, data[:,:,:3])
                rew_loss = loss_func(rew, data[:,:,3,].unsqueeze(-1))
                next_state_loss = loss_func(next_state[:,:-1,:], sample[:,1:,:])
                KLD_loss = -0.0 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                obs_loss_total += obs_loss.detach().item()
                rew_loss_total += rew_loss.detach().item()
                next_state_loss_total += next_state_loss.detach().item()
                KLD_loss_total += KLD_loss.detach().item()

                total_loss = obs_loss + rew_loss + next_state_loss # + KLD_loss
                total_loss.backward()
                optimizer.step()

            if epoch%print_freq == 0:
                print("Epoch: {} Obs_loss: {} Rew_loss: {} Next_state_loss: {} KLD_loss: {}".format(epoch, obs_loss_total, rew_loss_total, next_state_loss_total, KLD_loss_total))
                print("Sample next_state")
                print(next_state[0][10][:].detach().cpu().numpy())
                print(sample[0][6][:].detach().cpu().numpy())
                print("Sample obs")
                print(obs[0][10][:].detach().cpu().numpy())
                print(last_data[0][10][:3].detach().cpu().numpy())
                print("Sample rew")
                print(rew[0][10][:].detach().cpu().numpy())
                print(last_data[0][10][3].detach().cpu().numpy())
                print("Logvar")
                print(logvar[0][10][:].detach().cpu().numpy())

                plotter.plot("obs_loss", epoch, obs_loss_total)
                plotter.plot("rew_loss", epoch, rew_loss_total)
                plotter.plot("next_state_loss", epoch, next_state_loss_total)
                plotter.plot("KLD_loss", epoch, KLD_loss_total, title="KLD_losses")

                for param_group in optimizer.param_groups:
                    param_group['lr'] = adam_lr*(1-(epoch/epochs))

    except KeyboardInterrupt:
        print("Saving model")
        torch.save(model.state_dict(), "./delip_model")

    torch.save(model.state_dict(), "./delip_model")
    return model

def generate_initial_beliefs():
    trajectory_timesteps = 100
    total_episodes = 1000
    epochs = 10000
    batch_size = 20
    adam_lr = 1e-3
    print_freq = 25
    model = DELIP_model()
    model.load_state_dict(torch.load("./delip_model"))
    model.eval()
    model.cuda()

    print("initializing dataset for initial belief generation")
    train_dataset = RobotDoorsDataset(total_episodes, trajectory_timesteps)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, collate_fn=cat_collate)

    dataloader = iter(train_dataloader)
    first_states = ()
    round_func = lambda x: round(x, 3)
    with torch.no_grad():
        for data in dataloader:
            next_state, obs, rew, mu, logvar, sample = model((data[:, :, :4], data[:, :, 4].unsqueeze(-1)))
            unrounded_states = sample[:,1,:].detach().cpu().tolist()
            rounded_states = tuple([tuple(list(map(round_func, i))) for i in unrounded_states])
            first_states += rounded_states
    with open("initial_states.pkl", "wb") as f:
        pickle.dump(first_states, f)

def cat_collate(batch):
    return torch.cat(batch, axis=0)


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    training()
    generate_initial_beliefs()
    print("hello")