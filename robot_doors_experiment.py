import numpy as np
from matplotlib import pyplot as plt
from delip import utils
from pomdpy.pomdp import Model, StepResult


class RobotDoorsExperiment(Model):
    def __init__(self):
        self.initial_robot_state = np.clip(np.random.normal(0, 5), -15, 15)
        self.robot_state = self.initial_robot_state
        self.door_locations = [-12, -6, 6, 12]
        self.goal_door = 2
        self.goal_door_open = False

    def reset_for_simulation(self):
        self.robot_state = self.initial_robot_state

    def reset_for_epoch(self):
        self.robot_state = self.initial_robot_state

    def update(self, sim_data):
        pass

    def generate_step(self, state, action):
        if action is None:
            print("Tried to generate a step with a null action")
            return None

        res = StepResult()
        res.action = action
        res.observation = self.get_observation(self.robot_state)
        res.reward = self.get_reward(self.robot_state, open)
        res.next_state = self.take_action(action)
        res.is_terminal = self.is_terminal

    def sample_an_init_state(self):
        return self.sample_state_uninformed()

    def sample_state_uninformed(self):
        return np.random.uniform(-15, 15)

    def sample_state_informed(self, belief):
        pass

    

    def get_observation(self, curr_pos):
        door = 0
        for door_loc in self.door_locations:
            door += 0.6*utils.gaussian(curr_pos, door_loc, 0.5)  # Doors
        left = (-np.tanh(5 * (curr_pos + 13)) + 1) / 2  # Left wall
        right = (np.tanh(5 * (curr_pos - 13)) + 1) / 2  # Right wall
        return [door, left, right]

    def get_reward(self, curr_pos, open):
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
        # If the robot tries to open door, check for it in reward function
        if action == 0:
            open = True
        else:
            open = False

        # Take the action, return
        curr_state = self.robot_state
        next_state = np.clip(curr_state + action, -15, 15)
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
    exp = RobotDoorsExperiment()
    exp.plot_signals()

