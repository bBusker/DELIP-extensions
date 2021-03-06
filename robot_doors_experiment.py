import numpy as np
from matplotlib import pyplot as plt
import utils
import pomcp
import model
import model_pytorch
import torch

class RobotDoorsExperiment():
    def __init__(self):
        self.door_locations = [-12, -6, 6, 12]
        self.goal_door = 2
        self.goal_door_open = False

        self.action_space = ["left", "open", "right"]
        self.state_space = np.linspace(-15, 15, 300).tolist()
        self.obs_space = [self.get_observation_discrete(x) for x in self.state_space]

        self.initial_robot_state = np.clip(np.random.normal(0, 5), -15, 15)
        self.robot_state = self.state_space[np.searchsorted(self.state_space, self.initial_robot_state)]
        self.open_action = False

        self.DELIP_model = None

    # Load a DELIP model
    def load_model(self, filepath):
        self.DELIP_model = model_pytorch.DELIP_model()
        self.DELIP_model.load_state_dict(torch.load(filepath))
        self.DELIP_model.eval()

    # Takes in state and action values
    # Returns value of next_state, observation, and reward
    def generate_step_oracle(self, state, action):
        mag_action = self.action_space.index(action)-1

        next_state = state + mag_action
        closest_in_statespace = np.searchsorted(self.state_space, next_state)
        next_state_clipped = self.state_space[np.clip(closest_in_statespace, 0, len(self.state_space) - 1)]

        observation = self.get_observation_discrete(next_state_clipped)  # TODO: should this be observation indicies?
        reward = self.get_reward(next_state, mag_action == 0)
        return next_state_clipped, observation, reward

    # Takes in state and action values
    # Returns value of next_state, observation, and reward
    def generate_step_DELIP(self, state, action):
        with torch.no_grad():
            assert self.DELIP_model is not None
            state = torch.Tensor(state)
            action = torch.Tensor([self.action_space.index(action)-1])

            next_state, observation, reward = self.DELIP_model.decode(state, action)
            next_state = tuple([round(i,3) for i in next_state.tolist()])
            observation = tuple([round(i,2) for i in observation.tolist()])
            reward = tuple([round(i,2) for i in reward.tolist()])[0]

            return next_state, observation, reward

    # Takes in position of robot
    # Returns calculated observation
    def get_observation_continuous(self, curr_pos=None):
        if isinstance(curr_pos, np.ndarray):
            curr_pos = curr_pos[0]
        elif curr_pos == None: # Default to state of current instance
            curr_pos = self.robot_state

        door = 0
        for door_loc in self.door_locations:
            door += 0.6*utils.gaussian(curr_pos, door_loc, 0.5)  # Doors
        left = (-np.tanh(5 * (curr_pos + 13)) + 1) / 2  # Left wall
        right = (np.tanh(5 * (curr_pos - 13)) + 1) / 2  # Right wall
        return (door, left, right)

    def get_observation_discrete(self, curr_pos=None):
        continuous_obs = self.get_observation_continuous(curr_pos)
        return (round(continuous_obs[0], 2), round(continuous_obs[1], 2), round(continuous_obs[2], 2))

    # Takes in position of robot
    # Returns closest observation from obs_space
    def get_observation_closest_state(self, curr_pos=None):
        if isinstance(curr_pos, np.ndarray):
            curr_pos = curr_pos[0]
        elif curr_pos == None: # Default
            curr_pos = self.robot_state

        closest_state = np.searchsorted(self.state_space, curr_pos)
        return self.obs_space[closest_state]

    def get_reward(self, curr_pos=None, open=None):
        if curr_pos == None:
            curr_pos = self.robot_state
        if open == None:
            open = self.open_action

        res = 0
        if open:
            target_loc = self.door_locations[self.goal_door]
            if target_loc-1.1 <= curr_pos <= target_loc+1.1:  # Correct door
                res += 1
        res += (np.tanh(5*(curr_pos+13)) - 1)/2  # Left wall
        res += (-np.tanh(5*(curr_pos-13)) - 1)/2  # Right wall
        return res

    def get_state(self):
        return self.robot_state

    def take_action(self, action):
        if isinstance(action, str):
            mag_action = self.action_space.index(action)-1
        else:
            mag_action = action
        # i_state = self.state_space.index(self.robot_state)

        # If the robot tries to open door, check for it in reward function
        if mag_action == 0:
            self.open_action = True
        else:
            self.open_action = False

        # Take the action, return
        # i_next = np.clip(i_state + mag_action, 0, len(self.state_space)-1)
        next_state = self.robot_state + mag_action
        closest_in_statespace = np.searchsorted(self.state_space, next_state)
        next_state_clipped = self.state_space[np.clip(closest_in_statespace, 0, len(self.state_space)-1)]
        self.robot_state = next_state_clipped
        return next_state_clipped

    def is_terminal(self, state):
        target_loc = self.door_locations[self.goal_door]
        return (target_loc - 1 <= state <= target_loc + 1) and self.goal_door_open

    def plot_signals(self):
        x_values = np.linspace(-15, 15, 300)
        door_values = [self.get_observation_continuous(x)[0] for x in x_values]
        left_values = [self.get_observation_continuous(x)[1] for x in x_values]
        right_values = [self.get_observation_continuous(x)[2] for x in x_values]
        plt.plot(x_values, door_values)
        plt.plot(x_values, left_values)
        plt.plot(x_values, right_values)
        plt.show()

    def plot_rewards(self):
        x_values = np.linspace(-15, 15, 300)
        reward_values = [self.get_reward(x) for x in x_values]
        plt.plot(x_values, reward_values)
        plt.show()

def generate_data(episodes=10, steps=10):
    data = []
    for i in range(episodes):
        experiment = RobotDoorsExperiment()
        trajectory = []
        for j in range(steps):
            curr_action = np.random.choice([-1,0,1])
            experiment.take_action(curr_action)
            # TODO(slu): standardize to discrete observations in generator for POMCP
            res = experiment.get_observation_discrete() + (experiment.get_reward(),) + (curr_action,)
            trajectory.append(res)
        data.append(trajectory)
    return data

if __name__ == "__main__":
    experiment = RobotDoorsExperiment()
    planner = pomcp.POMCP(experiment.generate_step_oracle, timeout=100)

    planner.initialize(experiment.state_space, experiment.action_space, experiment.obs_space)

    for i in range(100):
        action = planner.Search()
        print(planner.tree.nodes[-1][:])
        print(action)
        experiment.take_action(action)
        observation = experiment.get_observation_discrete()
        #i_action = experiment.action_space.index(action)
        #i_observation = experiment.obs_space.index(observation)
        planner.tree.prune_after_action(action, observation)
        planner.UpdateBelief(action, observation)


