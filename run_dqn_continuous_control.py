import gym
import tensorflow as tf
import numpy as np

import dqn
import utils
from q_functions import *
from replay_memory import NStepReplayMemory


class PulseWidthModulatedEnv(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.action_space, gym.spaces.Box)
        self.old_action_space = self.action_space
        print(self.old_action_space)

        self.action_space = gym.spaces.Discrete(n=3)

    def action(self, action):
        return {
            0: self.old_action_space.low,
            1: np.zeros(self.old_action_space.shape),
            2: self.old_action_space.high,
        }[action]


def main():
    env = gym.make('Pendulum-v0')
    env = PulseWidthModulatedEnv(env)
    env = gym.wrappers.Monitor(env, 'videos/', force=True)

    seed = 0
    utils.set_global_seeds(seed)
    env.seed(seed)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

    n_timesteps = 2000000
    exploration_schedule = utils.PiecewiseSchedule(
                               [(0, 1.0), (1e6, 0.1)],
                               outside_value=0.1,
                           )

    replay_memory = NStepReplayMemory(
                        size=1000000,
                        history_len=1,
                        discount=0.99,
                        nsteps=3,
                    )

    dqn.learn(
        env,
        CartPoleNet(),
        replay_memory,
        optimizer=optimizer,
        exploration=exploration_schedule,
        max_timesteps=n_timesteps,
        batch_size=32,
        learning_starts=10000,
        learning_freq=4,
        target_update_freq=250,
        log_every_n_steps=25000,
    )
    env.close()


if __name__ == '__main__':
    main()
