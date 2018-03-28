#!/usr/bin/env python3
import importlib
import sys

import tensorflow as tf
import matplotlib.pyplot as plt
import functools

from baselines import bench, logger
from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MlpPolicy

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.hyperparams import *

sys.path.append('baselines/Environments')

ncpu = 8
config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=ncpu,
                        inter_op_parallelism_threads=ncpu)


def make_env(log=True):
  EnvModule = importlib.import_module(env_id)
  env = getattr(EnvModule, env_id)(**env_config)
  if log:
    filename = 'log/{dir}'.format(dir=env_id)
  else:
    filename = None
  env = bench.Monitor(env, filename, info_keywords=('vtrue',), allow_early_resets=True)
  return env

policy = functools.partial(MlpPolicy, **mlp_config)
env = DummyVecEnv([make_env])
env = VecNormalize(env, ob=False, ret=False)

def train(num_timesteps, seed, load=False):
    
    set_global_seeds(seed)
    model = ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=3e-4,
        cliprange=0.2,
        total_timesteps=num_timesteps, save_interval=0, load=load)
    model.save(model_dir)
    return model


def test(model=None, epochs=5):
  """
  {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
  :param epochs:
  :return:
  """
  model = model or ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=3e-4,
        cliprange=0.2,
        total_timesteps=0)
  #model.load(model_dir)
  runner = ppo2.Runner(env=env, model=model, nsteps=env_config['max_steps'], gamma=0.99, lam=0.95)
  def Gen():
    while True:
      obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()
      yield {
        'obs': obs,
        'returns': returns,
        'vtrue': epinfos[0]['vtrue'],
      }
  gen = Gen()
  
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
    logger.configure(dir='/home/tangcc/_march/Documents/Project/baselines/baselines/ppo2/log')
    with tf.Session(config=config):
      model = train(num_timesteps=3e5, seed=0, load=False)
    
    tf.reset_default_graph()
    with tf.Session(config=config):
      test(model, epochs=5)


if __name__ == '__main__':
    main()
    plt.show()
