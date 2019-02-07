import gym
import tensorflow as tf
import argparse

import dqn
import utils
import atari_wrappers
from q_functions import *
from replay_memory import NStepReplayMemory


def make_atari_env(name):
    from gym.wrappers.monitor import Monitor
    from gym.envs.atari.atari_env import AtariEnv
    env = AtariEnv(game=name, frameskip=4, obs_type='image')
    env = Monitor(env, 'videos/', force=True, video_callable=lambda e: False)
    env = atari_wrappers.wrap_deepmind(env)
    return env


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',         type=str, default='pong')
    parser.add_argument('--nsteps',      type=int, default=1)
    parser.add_argument('--history-len', type=int, default=4)
    parser.add_argument('--seed',        type=int, default=0)
    parser.add_argument('--recurrent',   action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    env = make_atari_env(args.env)

    utils.set_global_seeds(args.seed)
    env.seed(args.seed)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-4)

    n_timesteps = 10000000
    exploration_schedule = utils.PiecewiseSchedule(
                               [(0, 1.0), (1e6, 0.1), (n_timesteps, 0.05)],
                               outside_value=0.05,
                           )

    replay_memory = NStepReplayMemory(
                        size=1000000,
                        history_len=args.history_len,
                        discount=0.99,
                        nsteps=args.nsteps,
                    )

    dqn.learn(
        env,
        AtariRecurrentConvNet if args.recurrent else AtariConvNet,
        replay_memory,
        optimizer=optimizer,
        exploration=exploration_schedule,
        max_timesteps=n_timesteps,
        batch_size=32,
        learning_starts=50000,
        learning_freq=4,
        target_update_freq=10000,
        grad_clip=40.,
        log_every_n_steps=250000,
    )
    env.close()


if __name__ == '__main__':
    main()
