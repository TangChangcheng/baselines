# coding=utf-8
import numpy as np

env_id = 'OPA'

model_dir = 'model'
param_path = '/home/tangcc/_march/Documents/Project/baselines/baselines/Environments/param/'

H = 16
max_steps = 16
nb_status = 7
nb_params = 1

nb_group = 1

delta_w = 5e-8
delta_bias = 1e-6
EPS = 1e-3


sensor_dims = {
  'cur_status': (nb_status, ),
  'grad': (nb_status,),
  'val': (1, ),
  'params': (nb_params, ),
  # 'bias': (nb_status, ),
  # 'actions': (nb_group, nb_status, ),
  # 'group': (nb_group, nb_status, ),
  # 'p_best': (nb_group, nb_status, ),
  # 'g_best': (nb_group, nb_status, ),
  # 'vals': (nb_group, ),
}

obs_include = ['val', 'grad']
meta_include = []

ln = False
reward_normalize = True
random_fn = np.random.uniform

status_min = np.array([1.0, 0.65, 1.0, 0.65, 0.65] + [5e-7, 1e-6])
status_max = np.array([1.2, 0.85, 1.2, 0.85, 0.85] + [5e-6, 1e-5])
status_range = (status_min, status_max)

params_min = np.array([6e-7] * nb_params)
params_max = np.array([2e-5] * nb_params)
params_range = (params_min, params_max)

def transform(val):
  # return np.log10(val)
  return val

def detransform(val):
  # return np.power(10, val)
  return val

def encode(val, min, max):
  return (transform(val) - transform(min)) / (transform(max) - transform(min))

def decode(val, min, max):
  return detransform(val * (transform(max) - transform(min)) + transform(min))

env_config = {
  'nb_status': nb_status,
  'nb_params': nb_params,
  'max_steps': max_steps,
  'params': None,
  'params_mean': 0,
  'params_std': 2,
  'status_mean': 0,
  'status_std': 2,
  'H': H,
  'nb_group': nb_group,
  'status_range': status_range,
  'params_range': params_range,
  'encode_fn':encode,
  'decode_fn':decode,
  'penlaty_factor': 20
  
}

mlp_config={
  'hid_size': 128,
  'num_hid_layers': 3,
  'load': False,
}

cnn_config={
  'hid_size': 64,
  'num_hid_layers': 2,
}


