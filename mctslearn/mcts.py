"""
Notes:
    Where do we need to store reward? In tree?
    When doing simulations, we will go to edge, copy the env, and then
    Do we reason about choices or hash of states?

TODO:
    Valid actions mask
    Two player score negation etc

"""

from copy import deepcopy
import numpy as np

# make hashing of arrays reasonably fast:
np.set_printoptions(precision=8)


def puct_agz(prior, w, n, puct_const):
    """ PUCT equation used by deepmind for AGZ """
    q = np.nan_to_num(w / n)
    exploration = puct_const * prior * np.sqrt(n.sum()) / (1 + n)
    return q + exploration


class Node:
    """ Node in the tree we are searching """

    # FIXME: update arguments and set them
    def __init__(
            self,
            env,
            actions,
            parent=None,
            reward=0,
            value=0,
            prior=1,
            result_of_action=None,
            terminal_state=False,
    ):
        self.children = {}  # map from action to node

        self.env = env
        self.actions = actions
        self.reward = reward
        self.parent = parent
        self.prior = prior
        self.value = value
        self.result_of_action = result_of_action
        self.terminal_state = terminal_state

        # Sum of reward for each action
        self.w = np.zeros(len(self.actions))
        # Number of attempts for each action
        self.n = np.zeros(len(self.actions))

        self.move_number = 0

        if parent:
            self.parent.children[result_of_action] = self
            self.move_number = parent.move_number + 1

    def backpropagate(self, gamma=1.0, root=None):
        """
        Backpropagate value estimate for the state.
        Doesn't backpropagate past node `root`
        """
        node = self
        value = self.value
        while node.parent is not None and node != root:
            value = gamma * value + self.reward
            node.parent.w[node.result_of_action] += value
            node.parent.n[node.result_of_action] += 1
            node = node.parent


class Agent:
    """
    Creates agent for an environment.
    set_start_state() needs to be set before starting.
    States are hashed with str().
    The tree is persisted between episodes.
    """

    def __init__(
            self,
            env,  # Warning – env must be deepcopyable
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

        self.env = env
        self.root = None
        self.nodes = {}  # Map from state to node

        # TODO: Handle continuous envs
        self.actions = np.arange(env.action_space.n)

        self.puct_equation = puct_equation
        self.n_simulations = n_simulations
        self.puct_const = puct_const

    def set_start_state(self, state, env):
        """ Deepcopies env, called at start of episode """
        env = deepcopy(env)
        self.root = Node(env=env, actions=self.actions)
        self.nodes[str(state)] = self.root

    def act(self, state, env):
        if str(state) not in self.nodes:
            self.set_start_state(state, env)

        for _ in range(self.n_simulations):
            self.expand(state)
        return self.select(state)

    def select(self, state, deterministic=True):
        # TODO valid moves
        node = self.nodes[str(state)]
        if deterministic:
            return np.argmax(node.n)
        else:
            return np.random.choice(
                np.arange(len(node.n)), p=node.n / node.n.sum())

    def expand(self, state):
        state_hash = str(state)

        start_node = self.nodes[state_hash]

        # Select the best path through tree until we find a leaf
        node = start_node
        action = self.puct_select(node)
        while action in node.children:
            node = node.children[action]
            action = self.puct_select(node)

        # We now have an action that has not been expanded before
        # This could be because the game ended in this state

        # If episode is over we just backpropagate:
        if not node.terminal_state:
            node = self.next_node(node, action)

        # Backpropagate the future reward from this state
        # If an action results in terminal state.
        # Then we probably get a reward for this action.

        node.backpropagate(root=start_node)

        # TODO: add the new node while checking that it doesn't already exist
        # TODO: Check  if the state is not already visited

    def next_node(self, parent, action):
        env = deepcopy(parent.env)
        state, reward, terminal, info = env.step(action)

        value = 0  # TODO: call a value function here
        prior = 1  # TODO: call a policy function here
        node = Node(
            env=env,
            actions=self.actions,
            parent=parent,
            reward=reward,
            value=value,
            prior=prior,
            result_of_action=action,
            terminal_state=terminal,
        )
        # TODO: if it exists, create multiple parents to that node here!
        self.nodes[str(state)] = node
        return node

    def puct_select(self, node):
        scores = self.puct_equation(
            node.prior,
            node.w,
            node.n,
            puct_const=self.puct_const,
        )
        return np.argmax(scores)