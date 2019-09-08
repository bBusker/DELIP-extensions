import torch
import robot_doors_experiment as exp
import pomcp

def test_model():
    timesteps = 1000
    pomcp_timeout = 50
    model_filepath = 'delip_model'

    experiment = exp.RobotDoorsExperiment()
    experiment.load_model(model_filepath)

    planner = pomcp.POMCP(experiment.generate_step_DELIP, timeout=pomcp_timeout)
    planner.initialize(experiment.state_space, experiment.action_space, experiment.obs_space)

    cum_rew = 0
    print("Starting robot experiment with model {}".format(model_filepath))
    for i in range(timesteps):
        action = planner.Search()
        print(planner.tree.nodes[-1][:])
        print(action)
        experiment.take_action(action)
        print("Robot state: {}".format(experiment.get_state()))
        cum_rew += experiment.get_reward()
        observation = experiment.get_observation_discrete()
        planner.tree.prune_after_action(action, observation)
        planner.UpdateBelief(action, observation)
    print("Model took {} timesteps with pomcp simtime {}. Got cumulative reward {}!".format(timesteps, pomcp_timeout, cum_rew))


if __name__ == "__main__":
    test_model()
