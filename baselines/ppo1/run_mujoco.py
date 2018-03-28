#!/usr/bin/env python3
import matplotlib.pyplot as plt
import importlib
import sys

from baselines.common.cmd_util import mujoco_arg_parser
from baselines.common import tf_util as U
from baselines import logger
from baselines.bench.monitor import Monitor

from baselines.hyperparams import *

from baselines.ppo1 import mlp_policy, pposgd_simple, cnn_policy
import os


EnvModule = importlib.import_module(env_id)
env = getattr(EnvModule, env_id)(**env_config)
U.make_session(num_cpu=8).__enter__()
def policy_fn(name, ob_space, ac_space):
  return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                              **mlp_config)
  # return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
  #                             **cnn_config)

def train(env, num_timesteps, seed):
    wrapped_env = Monitor(env, 'log/{}'.format(env_id))
    pposgd_simple.learn(wrapped_env, policy_fn,
            max_timesteps=num_timesteps,
            # max_timesteps=1E5,
            timesteps_per_actorbatch=env.max_steps * 10,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    wrapped_env.close()
    
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
      {k: np.array(v).tolist() for k,v in sample.items()}
      for sample in samples
    ]
    json.dump(samples, fw)


def main():
    args = mujoco_arg_parser().parse_args()
    logger.configure(dir='log', format_strs=['stdout'])
    train(env, num_timesteps=5e4, seed=args.seed)

if __name__ == '__main__':
    main()
    test(env)
    plt.show()
