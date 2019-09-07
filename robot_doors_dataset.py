from torch.utils.data import Dataset, DataLoader
import torch
import robot_doors_experiment as exp
import numpy as np
import random


class RobotDoorsDataset(Dataset):
    def __init__(self, episodes, steps):
        self.experiment = exp.RobotDoorsExperiment()

        # 5 Buckets
        self.num_buckets = 5

        trajectories = exp.generate_data(episodes=episodes, steps=steps)
        trajectories = np.array(trajectories, dtype=np.float32)
        cum_rew_per_traj = np.sum(trajectories, axis=1)[:, 3]
        sorted_data = [trajectory for cum_rew, trajectory in sorted(zip(cum_rew_per_traj, trajectories), key=lambda x: x[0])]
        self.buckets = []
        for i in range(self.num_buckets):
            bucket_data = sorted_data[len(sorted_data)//self.num_buckets*i:len(sorted_data)//self.num_buckets*(i+1)]
            self.buckets.append(torch.stack([torch.Tensor(t) for t in bucket_data]).cuda())

    def __len__(self):
        return self.buckets[0].shape[0]

    def __getitem__(self, item):
        res = torch.stack([self.buckets[i][item] for i in range(self.num_buckets)])
        return res