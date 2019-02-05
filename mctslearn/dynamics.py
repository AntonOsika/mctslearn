from copy import deepcopy

# Question (2018-12-28):
# Are env states modified inplace or overwritten?
# Lets assume inplace.
# In such case we have to do a deepcopy before we step
# After we have stepped, we can be sure that the we will never modify the leftover env?


class EnvDynamics:
    """
    env wrapper that statelessly processes states with step() and returns the step.

    We refer to the `state` as everything that determines the outcome of actions,
    and the `observation` as what is returned from `env.step(action).

    Internally, `step()` will set the state of its internal _env and call
    `_env.step(action)`

    TODO(ao): Handle random seeds
    """

    def __init__(self, env_fn, state_variable_names):
        """
        Args
            env_fn: constructor of the openai env (only called once)
            state_variable_names: list of variable names in the env that
                are part of the state. Each variable name is a tuple of strings,
                so that "recursive" lookup of member variables is possible.
        """
        self.env = env_fn()
        self.state_variable_names = state_variable_names

        # Makes sure we can step with the dynamics-env:
        self.env.reset()

        # Unwrap untill we get the action_space
        unwrapped_env = self.env
        while not hasattr(unwrapped_env, 'action_space'):
            assert hasattr(unwrapped_env, 'env')
            unwrapped_env = unwrapped_env.env

        self.action_space = unwrapped_env.action_space

    def reset(self):
        """ Starts a new episode and returns state and observation """
        observation = self.env.reset()
        state = self.env_to_state(self.env)
        # Note: Since we return the current state, internal state of `env` is
        # never manipulated before deepcopying
        return observation, state

    def step(self, state, action):
        state = deepcopy(state)
        self.assign_state_to_env(self.env, state)

        observation, reward, terminal, info = self.env.step(action)
        state = self.env_to_state(self.env)

        return observation, reward, terminal, info, state

    def _get_env_attr(self, env, key):
        """ Get value of (recursive list of) env variable name """
        if isinstance(key, (list, tuple)):
            value = env
            for k in key:
                value = getattr(value, k)
            return value
        return getattr(env, key)

    def _set_env_attr(self, env, key, value):
        """ Assign value to (recursive list of) env variable name """
        if isinstance(key, (list, tuple)):
            container = env
            for k in key[:-1]:
                container = getattr(container, k)
            setattr(container, key[-1], value)
        else:
            setattr(env, key, value)

    def assign_state_to_env(self, env, state):
        """ Assign state to provided env """
        for k, v in state.items():
            self._set_env_attr(env, k, v)

    def env_to_state(self, env):
        """ Extract state from env """
        state = {}
        for key in self.state_variable_names:
            state[key] = self._get_env_attr(env, key)
        return state
