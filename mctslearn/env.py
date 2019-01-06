from copy import deepcopy

# Question:
# Are env states modified inplace or overwritten?
# Lets assume inplace.
# In such case we have to do a deepcopy before we step
# After we have stepped, we can be sure that the we will never

class EnvDynamics:
    """ 
    env wrapper that statelessly processes states and returns them.

    We refer to the `state` as everything that determines the, 
    and the `observation` as what is returned from `env.step(action).

    Args:
        env_fn: env factory, will only be called once.

    TODO(ao): Handle random seeds
    """
    def __init__(self, env_fn, state_variable_names):
        self.env = env_fn()
        self.state_variable_names = state_variable_names

    def reset(self):
        """ Starts a new episode and returns state and observation """
        observation = self.env.reset()
        state = self.get_state(self.env)
        # Since we return the current state, internal state of `env` should never
        # be manipulated unless explicit
        return observation, state

    def step(self, state, action):
        state = deepcopy(state)
        self.set_state(self.env, state)

        observation, reward, terminal, info = self.env.step(action)
        state = self.get_state(self.env)

        return observation, reward, terminal, info, state

    def get_env_attr(self, env, key):
        """ Get value of (recursive list of) env variable name """
        if isinstance(key, (list, tuple)):
            value = env
            for k in key:
                value = getattr(value, k)
            return value
        return getattr(env, key)

    def set_env_attr(self, env, key, value):
        """ Assign value to (recursive list of) env variable name """
        if isinstance(key, (list, tuple)):
            container = env
            for k in key[:-1]:
                container = getattr(container, k)
            setattr(container, key[-1], value)
        else:
            setattr(env, key, value)

    def set_state(self, env, state):
        for k, v in state.items():
            self.set_env_attr(env, k, v)

    def get_state(self, env):
        state = {}
        for key in self.state_variable_names:
            state[key] = self.get_env_attr(env, key)
        return state
