import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TigerProblemEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(2)

        self._seed()

        self.t = None
        self.state = None

        self.HORIZON = 10
        self.OBS_RELIABILITY = 0.85

        self.STATE_TIGER_LEFT = -1
        self.STATE_TIGER_RIGHT = 1

        self.ACTION_OPEN_LEFT_DOOR = 0
        self.ACTION_OPEN_RIGHT_DOOR = 1
        self.ACTION_LISTEN = 2

        self.REWARD_TIGER = -100.
        self.REWARD_NO_TIGER = 10.
        self.REWARD_LISTEN = -1.

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.t += 1

        if action == self.ACTION_OPEN_LEFT_DOOR:
            obs = self._random_state()
            reward = self.REWARD_TIGER if self.state == self.STATE_TIGER_LEFT else self.REWARD_NO_TIGER
            self.state = self._random_state()

        elif action == self.ACTION_OPEN_RIGHT_DOOR:
            obs = self._random_state()
            reward = self.REWARD_TIGER if self.state == self.STATE_TIGER_RIGHT else self.REWARD_NO_TIGER
            self.state = self._random_state()

        elif action == self.ACTION_LISTEN:
            obs = self._listen()
            reward = self.REWARD_LISTEN

        #done = (self.t >= self.HORIZON)
        done = (action == self.ACTION_OPEN_LEFT_DOOR or action == self.ACTION_OPEN_RIGHT_DOOR) or self.t >= self.HORIZON

        if self.t == self.HORIZON + 1:
            logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")

        return np.array([obs]), reward, done, {}

    def _reset(self):
        self.t = 0
        obs = self._random_state()
        self.state = self._random_state()

        return np.array([obs])

    def _render(self, mode='human', close=False):
        # TODO: how should we render the environment?
        pass

    def _random_state(self):
        return self.np_random.choice([self.STATE_TIGER_LEFT, self.STATE_TIGER_RIGHT])

    def _listen(self):
        if self.state == self.STATE_TIGER_LEFT:
            obs = self.STATE_TIGER_LEFT if self.np_random.random_sample() < self.OBS_RELIABILITY else self.STATE_TIGER_RIGHT

        elif self.state == self.STATE_TIGER_RIGHT:
            obs = self.STATE_TIGER_RIGHT if self.np_random.random_sample() < self.OBS_RELIABILITY else self.STATE_TIGER_LEFT

        return obs
