import numpy as np

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
