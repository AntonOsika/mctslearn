import gym
from mctslearn import mcts

env = gym.make('CartPole-v0')

episodes = 1
state = env.reset()

agent = mcts.Agent(env, n_simulations=50)
agent.set_start_state(state, env)

N = 100
rs = []
ts = []
ss = []
for _ in range(N):
    a = agent.act(state, env)
    s, r, t, info = env.step(a)
    rs.append(r)
    ts.append(t)
    ss.append(s)

print(r)
