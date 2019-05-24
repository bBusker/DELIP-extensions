import numpy as np
from matplotlib import pyplot as plt
import utils
import pomcp

#TODO: discretize observations!!!!
class RobotDoorsExperiment():
    def __init__(self):
        self.action_space = ["left", "open", "right"]
        self.state_space = np.linspace(-15, 15, 300)
        self.obs_space = [experiment.get_observation(x) for x in self.state_space]
        self.initial_robot_state = np.clip(np.random.normal(0, 5), -15, 15)
        self.robot_state = np.searchsorted(self.state_space, self.initial_robot_state)
        self.open_action = False

        self.door_locations = [-12, -6, 6, 12]
        self.goal_door = 2
        self.goal_door_open = False

    def generate_step_oracle(self, state, action):
        if action is None:
            print("Tried to generate a step with a null action")
            return None

        i_action = action_space.index(action)
        i_state = np.where(state_space == state)[0]

        i_next = np.clip(i_state+i_action-1, 0, self.state_space.size-1)
        next_state = self.state_space[i_next]
        observation = self.get_observation(next_state)  # TODO: should this be observation indicies?
        reward = self.get_reward(next_state, action == 1)
        return next_state, observation, reward

    def generate_step_vi(self, state, action):
        if action is None:
            print("Tried to generate a step with a null action")
            return None

        observation = self.get_observation(self.robot_state)
        reward = self.get_reward(self.robot_state, open)
        next_state = self.take_action(action)
        is_terminal = self.is_terminal
        return next_state, observation, reward

    def get_observation_continuous(self, curr_pos=None):
        if isinstance(curr_pos, np.ndarray):
            curr_pos = curr_pos[0]
        elif curr_pos == None: # Default
            curr_pos = self.robot_state

        door = 0
        for door_loc in self.door_locations:
            door += 0.6*utils.gaussian(curr_pos, door_loc, 0.5)  # Doors
        left = (-np.tanh(5 * (curr_pos + 13)) + 1) / 2  # Left wall
        right = (np.tanh(5 * (curr_pos - 13)) + 1) / 2  # Right wall
        return (door, left, right)

    def get_observation_discrete(self, curr_pos=None):
        if isinstance(curr_pos, np.ndarray):
            curr_pos = curr_pos[0]
        elif curr_pos == None: # Default
            curr_pos = self.robot_state

        return self.obs_space

    def get_reward(self, curr_pos=None, open=None):
        if curr_pos == None:
            curr_pos = self.robot_state
        if open == None:
            open = self.open_action

        res = 0
        if open:
            target_loc = self.door_locations[self.goal_door]
            if target_loc-1 <= curr_pos <= target_loc+1:  # Correct door
                res += 1
        res += (np.tanh(5*(curr_pos+13)) - 1)/2  # Left wall
        res += (-np.tanh(5*(curr_pos-13)) - 1)/2  # Right wall
        return res

    def get_state(self):
        return self.robot_state

    def take_action(self, action):
        if isinstance(action, str):
            i_action = self.action_space.index(action)
        i_state = np.where(state_space == self.robot_state)[0]

        # If the robot tries to open door, check for it in reward function
        if i_action == 1:
            self.open_action = True
        else:
            self.open_action = False

        # Take the action, return
        i_next = np.clip(i_state + i_action - 1, 0, self.state_space.size - 1)
        next_state = self.state_space[i_next]
        return next_state

    def is_terminal(self, state):
        target_loc = self.door_locations[self.goal_door]
        return (target_loc - 1 <= state <= target_loc + 1) and self.goal_door_open

    def plot_signals(self):
        x_values = np.linspace(-15, 15, 300)
        door_values = [self.get_observation(x)[0] for x in x_values]
        left_values = [self.get_observation(x)[1] for x in x_values]
        right_values = [self.get_observation(x)[2] for x in x_values]
        plt.plot(x_values, door_values)
        plt.plot(x_values, left_values)
        plt.plot(x_values, right_values)
        plt.show()

    def plot_rewards(self):
        x_values = np.linspace(-15, 15, 300)
        reward_values = [self.get_reward(x) for x in x_values]
        plt.plot(x_values, reward_values)
        plt.show()

if __name__ == "__main__":
    experiment = RobotDoorsExperiment()
    planner = pomcp.POMCP(experiment.generate_step_oracle, timeout=10)

    state_space = np.linspace(-15, 15, 300)
    action_space = ["left", "open", "right"]
    obs_space = [experiment.get_observation(x) for x in state_space]

    planner.initialize(state_space, action_space, obs_space)

    for i in range(10):
        action = planner.Search()
        print(planner.tree.nodes[-1][:4])
        print(action)
        experiment.take_action(action)
        observation = experiment.get_observation()
        i_action = action_space.index(action)
        i_observation = obs_space.index(observation)
        planner.tree.prune_after_action(i_action, i_observation)
        planner.UpdateBelief(action, observation)


