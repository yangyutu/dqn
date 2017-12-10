import numpy as np


class Episode:
    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []


class ReplayMemory:
    def __init__(self, config, obs_shape, recurrent_mode, forced_capacity=None):
        self.recurrent_mode = recurrent_mode
        #self.agent_history_length = int(config.get('agent', 'history_length'))
        self.batch_size = int(config.get('replay_memory', 'batch_size'))
        self.capacity = forced_capacity if forced_capacity else int(config.get('replay_memory', 'capacity'))
        self.obs_dtype = config.get('replay_memory', 'obs_dtype')

        self.size = 0                   # Number of experiences currently in the memory
        self.insertion_ptr = 0          # Index of the next experience to be overwritten
        self.ready_for_outcome = False  # Ensures observation's outcome is saved before incrementing the insertion pointer

        # Pre-allocate the replay memory; this has better performance and we'll also know immediately if it's too big
        self.episodes = [Episode() for i in range(self.capacity)]
        #self.obs = np.zeros([self.capacity] + list(obs_shape), dtype=getattr(np, self.obs_dtype))
        #self.action = np.zeros([self.capacity], dtype=np.int32)
        #self.reward = np.zeros([self.capacity], dtype=np.float32)
        #self.terminal = np.zeros([self.capacity], dtype=np.bool)

    def save_obs(self, obs):
        assert not self.ready_for_outcome
        self.ready_for_outcome = True

        self.episodes[self.insertion_ptr].obs = obs

    def save_outcome(self, action, reward, terminal):
        assert self.ready_for_outcome
        self.ready_for_outcome = False

        self.episodes[self.insertion_ptr].action = action
        self.episodes[self.insertion_ptr].reward = reward
        self.episodes[self.insertion_ptr].terminal = terminal

        if terminal:
            self._update_insertion_ptr()
            self.episodes[self.insertion_ptr] = Episode()

    def _update_insertion_ptr(self):
        # Increment size unless capacity is reached
        if self.size < self.capacity:
            self.size += 1

        # Increment the insertion pointer, wrapping around if the end is reached
        self.insertion_ptr = (self.insertion_ptr + 1) % self.capacity

    def append_history_to_obs(self, obs):
        '''
        starting_index = self.insertion_ptr - self.agent_history_length
        offsets = range(self.agent_history_length)

        obs = np.concatenate([self.obs[starting_index+j] for j in offsets], axis=-1)
        return obs
        '''
        raise NotImplementedError

    def sample_minibatch(self):
        '''
        starting_indices = np.random.choice(self.size - self.agent_history_length, self.batch_size)
        offsets = range(self.agent_history_length)

        obs_batch = np.stack([np.stack([self.obs[i+j] for i in starting_indices]) for j in offsets])
        action_batch = np.stack([self.action[i+j] for i in starting_indices for j in offsets])
        reward_batch = np.stack([self.reward[i+j] for i in starting_indices for j in offsets])
        next_obs_batch = np.stack([np.stack([self.obs[i+j+1] for i in starting_indices]) for j in offsets])
        non_terminal_multiplier = 1 - np.stack([self.terminal[i+j] for i in starting_indices for j in offsets])

        axis = 0 if self.recurrent_mode else -1
        obs_batch = np.concatenate(obs_batch, axis)
        next_obs_batch = np.concatenate(next_obs_batch, axis)
        '''

        i = np.random.choice(self.size)
        obs_batch = np.array(self.episodes[i].obs)
        action_batch = np.array(self.episodes[i].action)
        reward_batch = np.array(self.episodes[i].reward)
        next_obs = np.array(self.episodes[i].next_obs)
        non_terminal_multiplier = 1 - np.array(self.episodes[i].terminal)

        return obs_batch, action_batch, reward_batch, next_obs_batch, non_terminal_multiplier
