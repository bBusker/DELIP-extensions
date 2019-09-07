from torch.utils.data import Dataset, DataLoader
import torch
import robot_doors_experiment as exp
import numpy as np


class RobotDoorsDataset(Dataset):
    def __init__(self, episodes, steps):
        self.experiment = exp.RobotDoorsExperiment()

        # 5 Buckets
        num_buckets = 5

        trajectories = exp.generate_data(episodes=episodes, steps=steps)
        trajectories = np.array(trajectories, dtype=np.float32)
        cum_rew_per_traj = np.sum(trajectories, axis=1)[:, 3]
        sorted_data = [trajectory for cum_rew, trajectory in sorted(zip(cum_rew_per_traj, trajectories), key=lambda x: x[0])]
        self.buckets = []
        for i in range(num_buckets):
            self.buckets.append(torch.from_numpy(sorted_data[len(sorted_data)//num_buckets*i:len(sorted_data)//num_buckets*(i+1)][0]))

    def __len__(self):
        return len(self.buckets[0].shape[0])

    def __getitem__(self, item):
        return self.sorted_data[item]

if __name__ == "__main__":
    t = RobotDoorsDataset(10, 20)