import torch
import robot_doors_experiment as exp
import pomcp
import pickle
import numpy as np

def test_model():
    timesteps = 100
    pomcp_timeout = 50
    model_filepath = 'delip_model'

    experiment = exp.RobotDoorsExperiment()
    experiment.load_model(model_filepath)

    # planner = pomcp.POMCP(experiment.generate_step_DELIP, timeout=pomcp_timeout, gamma=0.8, no_particles=2000, threshold=0.001)
    # with open("./initial_states.pkl", "rb") as f:
    #     initial_belief = pickle.load(f)
    # planner.initialize(experiment.state_space, experiment.action_space, experiment.obs_space, initial_belief)

    planner = pomcp.POMCP(experiment.generate_step_oracle, timeout=pomcp_timeout, gamma=0.8, no_particles=2000, threshold=0.001)
    state_space = np.linspace(-13, 13, 300).tolist()
    planner.initialize(experiment.state_space, experiment.action_space, experiment.obs_space, state_space)

    cum_rew = 0
    print("Starting robot experiment with model {}".format(model_filepath))
    for i in range(timesteps):
        action = planner.Search()
        print(planner.tree.nodes[-1][:4])
        print(action)
        experiment.take_action(action)
        print("Robot state: {}, Reward: {}".format(experiment.get_state(), experiment.get_reward()))
        cum_rew += experiment.get_reward()
        observation = experiment.get_observation_discrete()
        planner.tree.prune_after_action(action, observation)
        planner.UpdateBelief(action, observation)
        # experiment.take_action(action)
        # planner.tree.prune_after_action(action, observation)
    print("Model took {} timesteps with pomcp simtime {}. Got cumulative reward {}!".format(timesteps, pomcp_timeout, cum_rew))


if __name__ == "__main__":
    test_model()
