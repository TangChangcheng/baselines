# coding=utf-8
import numpy as np

env_id = 'Convex'

model_dir = 'model'
obs = 'grad_val'
ln = False
observe_params = False
reward_normalize = True
random_fn = np.random.uniform

env_config = {
  'nb_status': 6,
  'max_steps': 16,
  'params': None,
  'params_mean': 0,
  'params_std': 2,
  'status_mean': 0,
  'status_std': 2,
  'H': 5,
}

mlp_config={
  'hid_size': 64,
  'num_hid_layers': 2,
  'load': False,
}


