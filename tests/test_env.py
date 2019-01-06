from copy import deepcopy

import numpy as np
import gym

from mctslearn.env import EnvDynamics


def test_env_step():

    env_fn = lambda: gym.make('CartPole-v0')
    state_variable_names = [
        ('env', 'state'),
        '_elapsed_steps',
    ]

    dynamics = EnvDynamics(env_fn, state_variable_names)
    obs, state = dynamics.reset()

    env = deepcopy(dynamics.env)

    actions = [0, 0, 1, 1, 0, 1]
    for action in actions:
        env_obs, env_reward, env_terminal, env_info = env.step(action)
        obs, reward, terminal, info, state = dynamics.step(state, action)

        assert np.array_equal(env_obs, obs)
        assert env_reward == reward
        assert env_terminal == terminal
        assert env_info == info

if __name__ == '__main__':
    test_env_step()
