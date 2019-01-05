import numpy as np


def get_best_states(node):
    res = []
    while node.children:
        res.append([node.env.env.state, 
            np.argmax(node.n),
            np.max(node.n),
            ])
        node = node.children[np.argmax(node.n)]

    return res
