"""
Notes:
    Where do we to store reward? 
    - In tree. When doing simulations, we will go to edge, copy the state, and then step.
    How do we identify if a state was seen before?
    - Observations for now, is not enough for POMDP (needs inferred state).
    Do we reason about choices or hash of states?
    - A combination, hash is nice since the agent should support any change in the 
      state between acting.
    What is the interface of an agent?
    - We initialise it with the dynamics, and can 'set start state'

TODO:
    Rename state -> obs to support POMDP
    Add 'valid actions mask'

"""

from copy import deepcopy
import numpy as np

from mctslearn.tree import Node
from mctslearn.dynamics import EnvDynamics

# make hashing of arrays reasonably fast:
np.set_printoptions(precision=8)


def puct_agz(prior, w, n, puct_const):
    """ PUCT equation used by deepmind for AGZ """
    q = np.nan_to_num(w / n)
    exploration = puct_const * prior * np.sqrt(n.sum()) / (1 + n)
    return q + exploration


class Agent:
    """
    Creates agent for an environment.
    set_start_state() needs to be set before starting.
    Observations are hashed with str().
    The tree is persisted between episodes.
    """

    def __init__(
            self,
            dynamics: EnvDynamics,
            deterministic_env=True,
            fully_observable_env=True,
            single_player_env=True,
            puct_equation=puct_agz,
            n_simulations=1600,
            puct_const=1.0,
    ):
        assert deterministic_env
        # If not true, we need to use root or previous observations to infer
        # hidden state.
        assert fully_observable_env
        assert single_player_env

        self.dynamics = dynamics
        self.root = None
        self.nodes = {}  # Map from state to node

        # TODO: Handle continuous envs
        self.actions = np.arange(dynamics.action_space.n)

        self.puct_equation = puct_equation
        self.n_simulations = n_simulations
        self.puct_const = puct_const

        self.n_restarts = 0

    def set_start_state(self, obs, state):
        """ Sets start state, called at start of episode """
        # TODO: Decide on strategy for clearing old nodes
        # TODO: try removing this and use logic in act instead

        self.root = Node(state=state, actions=self.actions)
        self.nodes[str(obs)] = self.root

    def act(self, obs, state):
        if str(obs) not in self.nodes:
            self.n_restarts += 1
            self.set_start_state(obs, state)

        for _ in range(self.n_simulations):
            self.expand(obs, state)
        return self.select(obs)

    def select(self, obs, deterministic=True):
        # TODO valid moves
        node = self.nodes[str(obs)]
        if deterministic:
            return np.argmax(node.n)
        else:
            return np.random.choice(
                np.arange(len(node.n)), p=node.n / node.n.sum())

    def expand(self, obs, state):
        obs_hash = str(obs)

        start_node = self.nodes[obs_hash]

        # Select the best path through tree until we find a leaf
        node = start_node
        action = self.puct_select(node)
        while action in node.children:
            node = node.children[action]
            action = self.puct_select(node)

        # We now have an action that has not been expanded before
        # This could be because the game ended in this state

        # If episode is over we only backpropagate:
        if not node.terminal_state:
            node = self.next_node(node, action)

        # Backpropagate the future reward from this state
        # If an action results in terminal state.
        # Then we probably get a reward for this action.

        node.backpropagate(root=start_node)

        # TODO: add the new node while checking that it doesn't already exist
        # TODO: Check  if the state is not already visited

    def next_node(self, parent, action):
        obs, reward, terminal, info, state = self.dynamics.step(state=parent.state, action=action)

        value = 0  # TODO: call a value function here
        prior = 1  # TODO: call a policy function here
        node = Node(
            state=state,
            actions=self.actions,
            parent=parent,
            reward=reward,
            value=value,
            prior=prior,
            result_of_action=action,
            terminal_state=terminal,
        )
        # TODO: if node exists, create multiple parents to that node here 
        # This would imply that we go from tree -> graph
        self.nodes[str(obs)] = node
        return node

    def puct_select(self, node):
        scores = self.puct_equation(
            node.prior,
            node.w,
            node.n,
            puct_const=self.puct_const,
        )
        return np.argmax(scores)
