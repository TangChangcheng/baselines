#!/usr/bin/env python3
import argparse
import importlib

import tensorflow as tf
import matplotlib.pyplot as plt

from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger
from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MlpPolicy

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.ppo2.hyperparams import *


def train(env_id, num_timesteps, seed):
    ncpu = 8
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    def make_env():
        EnvModule = importlib.import_module(env_id)
        env = getattr(EnvModule, env_id)(**env_config)
        env = bench.Monitor(env, 'log/convex')
        return env
    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = MlpPolicy
    ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=3e-4,
        cliprange=0.2,
        total_timesteps=num_timesteps)


def test(env, epochs=5):
  """
  {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
  :param epochs:
  :return:
  """
  pi = policy_fn('pi', env.observation_space, env.action_space)
  pi.load(model_dir)
  gen = pposgd_simple.traj_segment_generator(pi, env, env.max_steps, False)
  
  c = 5
  r = int(np.ceil(epochs / 5))
  fig = plt.figure()
  ax = [fig.add_subplot(int('%d%d%d' % (r, c, i))) for i in range(1, epochs + 1)]
  samples = []
  for i in range(epochs):
    sample = gen.__next__()
    samples.append(sample)
    ax[i].plot(sample['vtrue'])
  
  with open('test.json', 'w') as fw:
    import json
    samples = [
      {k: np.array(v).tolist() for k, v in sample.items()}
      for sample in samples
    ]
    json.dump(samples, fw)


def main():
    args = mujoco_arg_parser().parse_args()
    logger.configure()
    train(env_id, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
