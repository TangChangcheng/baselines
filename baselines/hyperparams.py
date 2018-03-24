# coding=utf-8
import numpy as np

env_id = 'Convex'

model_dir = 'model'
obs = 'grad'
ln = False
observe_params = False
reward_normalize = True
random_fn = np.random.uniform

env_config = {
  'nb_status': 6,
  'max_steps': 8,
  'params': None,
  'params_mean': 0,
  'params_std': 2,
  'status_mean': 0,
  'status_std': 2,
  'H': 8,
}

mlp_config={
  'hid_size': 128,
  'num_hid_layers': 3,
  'load': False,
}


