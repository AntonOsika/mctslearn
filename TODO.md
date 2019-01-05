TODO
====

- Debug by checking if states of nodes and actual states are the same

- Prune tree (evaluate performance gain from this)
- Multiple parents if tree merges back
- Handle that different continuous states are grouped together

Best Ideas:
=========

- Keep track on Walkers that walk down the tree
- Compare MCTS, swarmwave, and combinations of them

Weird ideas
===========

- Train q learning with data from tree
- Learn how much information will be gained from expanding in a certain direction

Thoughts:
=========

Maybe we want our own env representation that takes care of copying,
hashing states, getting valid acitons and so on.
Probably not yet.

