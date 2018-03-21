#!/usr/bin/env python3
import numpy  as np
import pandas as pd

from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common import tf_util as U
from baselines import logger
from baselines.hyperparams import *

from Environment import Convex
from baselines.bench.monitor import Monitor

nb_status = 10
params = None

from baselines.ppo1 import mlp_policy, pposgd_simple
U.make_session(num_cpu=1).__enter__()

def policy_fn(name, ob_space, ac_space):
  return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                              hid_size=64, num_hid_layers=2)

def train(env_id, num_timesteps, seed):

    # env = make_mujoco_env(env_id, seed)
    env = Convex(nb_status=nb_status, params=params, H=1)
    wrapped_env = Monitor(env, 'log/convex')
    pposgd_simple.learn(wrapped_env, policy_fn,
            max_timesteps=num_timesteps,
            # max_timesteps=1E5,
            timesteps_per_actorbatch=env.max_steps * 10,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    wrapped_env.close()
    
def test(epochs=5):
  """
  {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
  :param epochs:
  :return:
  """
  env = Convex(nb_status=nb_status, params=params)
  pi = policy_fn('pi', env.observation_space, env.action_space)
  pi.load(model_dir)
  gen = pposgd_simple.traj_segment_generator(pi, env, env.max_steps, False)
  samples = []
  for i in range(epochs):
    sample = gen.__next__()
    samples.append(sample)
  with open('test.json', 'w') as fw:
    import json
    samples = [
      {k: np.array(v).tolist() for k,v in sample.items()}
      for sample in samples
    ]
    json.dump(samples, fw)
    
  


def main():
    args = mujoco_arg_parser().parse_args()
    logger.configure(dir='log', format_strs=['csv', 'stdout'])
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)
    test()


if __name__ == '__main__':
    main()
