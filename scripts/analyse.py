import numpy as np


def get_best_states(node):
    res = []
    while node.children:
        a = np.argmax(node.n)
        res.append([
            node.env.env.state,
            a,
            node.w[a] / node.n[a],
            np.max(node.n),
        ])
        node = node.children[np.argmax(node.n)]

    return res
