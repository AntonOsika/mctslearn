import gym
from mctslearn.dynamics import EnvDynamics

# TODO: Include some "has been reset flag" to get rid of warning
cartpole_dynamics = EnvDynamics(
    env_fn=lambda: gym.make('CartPole-v0'),
    state_variable_names=[
        ('env', 'state'),
        '_elapsed_steps',
    ])
