from gym import spaces
from copy import deepcopy
from gym.core import Env
from baselines.hyperparams import *

import functools

import matplotlib.pyplot as plt

plt.style.use('ggplot')


class ackley(Env):
  id = 'ackley'
  
  def __init__(self, nb_status=8, max_steps=10, params=None, status=None, H=10, params_mean=0, params_std=1,
               status_mean=0, status_std=1., nb_group=10, **kwargs):
    super(ackley, self).__init__()
    
    self.action_shape = (nb_status,)
    self.action_space = spaces.Box(-1., 1., shape=self.action_shape)
    self.action = self.action_space.sample()
    
    self.H = H
    self.params_mean = params_mean
    self.params_std = params_std
    self.status_mean = status_mean
    self.status_std = status_std
    self.nb_group = nb_group

    
    self._seed = 0
    
    self.reward = 0
    
    self.max_steps = max_steps
    self.reward_range = (-np.inf, 0)
    
    if params is None:
      self.is_params_setted = False
    else:
      self.coefs = np.array(params[0])
      self.bias = np.array(params[1])
      self.is_params_setted = True
      
    if status is None:
      self.is_status_setted = False
    else:
      self.status = np.array(status)
      self.setted_status = np.array(status)
      self.is_status_setted = True
    
    
    reduce_mul = lambda seq: functools.reduce(lambda x, y: x * y, seq)

    obs_shape = [(H,) + sensor_dims[obs] for obs in obs_include]
    meta_shape = [sensor_dims[meta] for meta in meta_include]

    self.observation_space = spaces.Box(-10, 10, shape=(sum([reduce_mul(tup) for tup in obs_shape + meta_shape]),))
    # self.observation_space = spaces.Box(-10, 10, shape=(nb_group, 3 * nb_status, H))
    
    self.observations = {
      obs: np.zeros((0,) + sensor_dims[obs])
      for obs in obs_include
    }
    self.metas = {
      meta: np.zeros_like(sensor_dims[meta])
      for meta in meta_include
    }

    self.actions = np.zeros(sensor_dims['actions'])
    self.g_best = np.zeros(sensor_dims['g_best'])
    
    self.g_best_val = np.inf

    self.reset()
  
  def get_states(self, group):
    # vals = [self.foo(status) for status in group]
    # p_idx = np.argmax(vals)
    # p_best_val = vals[p_idx]
    # p_best = group[p_idx]
    # if p_best_val < self.g_best_val:
    #   self.g_best_val = p_best_val
    #   self.g_best = p_best
    #
    # p_best = np.tile(p_best, [self.nb_group, 1]) - group
    # g_best = np.tile(self.g_best, [self.nb_group, 1]) - group
    #
    # return {
    #   'actions': self.actions,
    #   'p_best': p_best,
    #   'g_best': g_best,
    #   'params': self.coefs,
    #   'bias': self.bias,
    # }
    val = self.foo(group)
    if self.g_best_val > val:
      self.g_best_val = val
    return {
      'grad': self.gradient(group),
      'val': [val],
      'params': self.coefs,
      'bias': self.bias,
    }
  
  def set_observations(self, states):
    for obs in obs_include:
      self.observations[obs] = np.append(self.observations[obs], [states[obs]], axis=0)
  
  def set_metas(self, states):
    for meta in meta_include:
      self.metas[meta] = np.tile(states[meta], [nb_group, 0, H])
  
  def foo(self, x):
    coef = self.coefs
    y = -20 * np.exp(-0.2 * np.sqrt(np.sum(np.power(np.matmul(coef, x) - self.bias, 2), axis=-1))) - np.exp(
      np.sum(np.cos(2 * np.pi * (np.matmul(coef, x) - self.bias)), axis=-1)) + 20 + np.e
    return y
  
  def gradient(self, x):
    convex = np.matmul(self.coefs, x) - self.bias
    y = 8 * np.matmul(self.coefs.T, convex) * np.exp(-0.2 * np.sqrt(np.sum(np.power(convex, 2), axis=-1))) /\
        np.sqrt(np.sum(np.power(convex, 2))) + \
        2 * np.pi * np.matmul(self.coefs.T, np.sin(convex)) * np.exp(
      np.sum(np.cos(2 * np.pi * (np.matmul(self.coefs, x) - self.bias)), axis=-1)) + 20 + np.e
    return y
  
  
  def get_loss(self):
    return self.foo(self.status)

  
  def reset(self, status=None):
    # print('\n--------------------------------------------------------------------------------')
    # self.coefs = np.random.uniform(0, 1, self.coefs.shape)
    
    if status is None and not self.is_status_setted:
      self.status = random_fn(self.status_mean, self.status_std, size=sensor_dims['cur_status'])
    elif status is not None:
      self.status = np.array(status)
    elif self.is_status_setted:
      self.status = deepcopy(self.setted_status)
    
    if not self.is_params_setted:
      self.coefs = random_fn(self.params_mean, self.params_std, size=sensor_dims['params'])
      self.bias = random_fn(self.params_mean, self.params_std, size=sensor_dims['bias'])
    self.init_status = deepcopy(self.status)
    states = self.get_states(self.status)
    self.set_observations(states)
    self.set_metas(states)
    
    self.nb_step = 0
    
    self.init_loss = self.g_best_val
    self.loss = self.init_loss
    self.info = {'vtrue': [self.init_loss]}
    
    # print('init_loss = ', self.loss)
    return self.observe()
  
  def seed(self, _int):
    np.random.seed(_int)
  
  def observe(self):
    # observations = []
    # for obs in obs_include:
    #   v = self.observations[obs]
    #   # shape (H,  nb_group, d))
    #   v = np.concatenate([
    #     np.zeros((max(0, self.H - len(v)),) + sensor_dims[obs]),
    #     v[-self.H:]
    #   ], axis=0)
    #   v = v.transpose([1, 2, 0])
    #   observations.append(v)
    #
    # for meta in meta_include:
    #   observations.append(self.metas[meta].flatten())
    # return np.concatenate(observations, axis=1)

    observations = []
    for obs in obs_include:
      v = self.observations[obs]
      if obs == 'val':
        v = self.loss - v
      v = np.concatenate([
        np.zeros((max(0, self.H - len(v)),) + sensor_dims[obs]),
        v[-self.H:]
      ], axis=0).flatten()
      observations.append(v)

    for meta in meta_include:
      observations.append(self.metas[meta].flatten())
    return np.concatenate(observations)
  
  def step(self, action):
    """

    :param action:
    :return:
        observation (object):
        reward (float): sum of rewards
        done (bool): whether to reset environment or not
        info (dict): for debug only
    """
    # print(self.status, action)
    self.nb_step += 1
    
    self.status += action.flatten()
    self.actions = action
    states = self.get_states(self.status)
    # tmp = self.g_best_val
    tmp = states['val'][0]
    
    self.last_reward = self.reward
    self.reward = -tmp
    self.loss = tmp
    
    self.set_observations(states)
    observation = self.observe()
    
    done = self.nb_step >= self.max_steps
    
    self.info['vtrue'].append(self.loss)
    
    if reward_normalize:
      self.reward /= self.init_loss
    
    return observation, self.reward, done, self.info
