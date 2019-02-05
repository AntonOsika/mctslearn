from copy import deepcopy

import gym
from mctslearn import mcts
from mctslearn.dynamics_registry import cartpole_dynamics

dynamics = cartpole_dynamics

env = gym.make('CartPole-v0')

obs = env.reset()
state = dynamics.env_to_state(env)
state = deepcopy(state)

episodes = 1

agent = mcts.Agent(dynamics=dynamics, n_simulations=50)
agent.set_start_state(obs, state)

N = 100
rs = []
ts = []
ss = []
for _ in range(N):
    a = agent.act(obs, obs)
    s, r, t, info = env.step(a)
    rs.append(r)
    ts.append(t)
    ss.append(s)

print(r)
