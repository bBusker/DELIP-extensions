import torch
from torch import nn


class DELIP_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn_layer = nn.LSTM(input_size=4, hidden_size=10, bidirectional=True, batch_first=True)
        self.prev_rnn_states = None
        self.latent_layer = nn.Linear(20, 8)
        self.obs_out = nn.Sequential(
            nn.Linear(4,100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100,3)
        )
        self.rew_out = nn.Sequential(
            nn.Linear(4,100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100,1)
        )
        self.state_out = nn.Sequential(
            nn.Linear(5, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 4)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, input):
        hidden, rnn_states = self.rnn_layer(input[0])
        # self.prev_rnn_states = rnn_states

        latent = self.latent_layer(hidden)
        mu = latent[:,:,:4]
        logvar = latent[:,:,4:]
        sample = self.reparameterize(mu, logvar)

        state_out = self.state_out(torch.cat((sample, input[1]), axis=2))
        obs_out = self.obs_out(sample)
        rew_out = self.rew_out(sample)

        return state_out, obs_out, rew_out, mu, logvar, sample

    def decode(self, latent, action):
        state_out = self.state_out(latent)
        obs_out = self.obs_out(latent)
        rew_out = self.rew_out(latent)

        return state_out, obs_out, rew_out

if __name__ == "__main__":
    model = DELIP_model()
    observations = torch.rand(1,10,5)
    actions = torch.rand(1,10,1)
    res = model((observations, actions))
    print("hello")