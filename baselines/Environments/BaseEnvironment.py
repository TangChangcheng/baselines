
import numpy as np
import copy
from gym import spaces

import matplotlib.pyplot as plt

plt.style.use('ggplot')

from gym.core import Env


from util import *
from abc import abstractmethod
from easydict import EasyDict

from gym import spaces
from copy import deepcopy
from gym.core import Env
from baselines.hyperparams import *

import functools

import matplotlib.pyplot as plt
from abc import abstractmethod

plt.style.use('ggplot')


class BaseEnv(Env):
  # 用_status（有下划线）访问归一化后的状态，status（无下划线）访问原始值，params同理
  # _status中min，max，range分别保存没有归一化的最小、最大值及其范围
  def __init__(self, nb_status, nb_params, max_steps, status=None, params=None, factor=1, encode_fn=lambda x, *y: x, decode_fn=lambda x, *y: x, H=1, **kwargvs):
    super(BaseEnv, self).__init__()
    
    self.action_shape = (nb_status,)
    self.action_space = spaces.Box(-1, 1, shape=self.action_shape)
    self.actions = self.action_space.sample()

    self.status_shape = (nb_status,)
    self.params_shape = (nb_params,)

    if 'status_range' not in kwargvs.keys() or 'params_range' not in kwargvs.keys():
      raise ValueError('The range of status and params must be provided.')

    self._status = {
      'val': np.array([0] * nb_status),
      'min': np.array(kwargvs['status_range'][0]),
      'max': np.array(kwargvs['status_range'][1]),
    }

    self._params = {
      'val': np.array([0] * nb_params),
      'min': np.array(kwargvs['params_range'][0]),
      'max': np.array(kwargvs['params_range'][1])
    }

    self.max_steps = max_steps

    self.reward = 0
    self.penalty = 0
    self.last_reward = 0
    self.min_reward = 0
    self._seed = 0
    self.factor = factor
    self.encode_fn = encode_fn
    self.decode_fn = decode_fn
    self.H = H

    if params is None:
      self.is_params_setted = False
    else:
      self.params = params
      self.is_params_setted = True

    reduce_mul = lambda seq: functools.reduce(lambda x, y: x * y, seq)
    obs_shape = [(H,) + sensor_dims[obs] for obs in obs_include]
    meta_shape = [sensor_dims[meta] for meta in meta_include]

    self.observation_space = spaces.Box(-10, 10, shape=(sum([reduce_mul(tup) for tup in obs_shape + meta_shape]),))

    self.observations = {
      obs: np.zeros((0,) + sensor_dims[obs])
      for obs in obs_include
    }
    self.metas = {
      meta: np.zeros_like(sensor_dims[meta])
      for meta in meta_include
    }
    
    self.cross_bound = np.zeros([nb_status, ])

    self.reset(status)

  def seed(self, seed=None):
    np.random.seed(seed)
    return

  def get_states(self, status):
    raise NotImplemented

  def set_observations(self, states):
    for obs in obs_include:
      self.observations[obs] = np.append(self.observations[obs], [states[obs]], axis=0)

  def set_metas(self, states):
    for meta in meta_include:
      self.metas[meta] = states[meta]

  @property
  def status(self):
    return self.status_decode(self._status['val'])

  @status.setter
  def status(self, _status):
    self._status['val'] = self.status_encode(_status)

  @property
  def params(self):
    return self.params_decode(self._params['val'])

  @params.setter
  def params(self, _params):
    self._params['val'] = self.params_encode(_params)
    
  def params_encode(self, val):
    return self.encode_fn(val, self._params['min'], self._params['max']) * self.factor

  def params_decode(self, val):
    return self.decode_fn(val / self.factor, self._params['min'], self._params['max'])

  def status_encode(self, val):
    return self.encode_fn(val, self._status['min'], self._status['max']) * self.factor

  def status_decode(self, val):
    return self.decode_fn(val / self.factor, self._status['min'], self._status['max'])

  def reset(self, status=None):
    if not self.is_params_setted:
      self.params = random_fn(self._params['min'], self._params['max'])
      self.set_params(self.params)

    if status is None:
      self.status_init = self.init_status()
      self._status['val'] = copy.deepcopy(self.status_init)
    else:
      self.status = np.array(status)
      
    self.cliped_status = self.status
    self.nb_step = 0
    states = self.get_states(self.cliped_status)
    self.set_observations(states)
    self.set_metas(states)

    self.init_loss = states['val'][0]
    self.loss = self.init_loss
    self.info = {'vtrue': [self.init_loss]}

    return self.observe()


  @abstractmethod
  def init_status(self):
    raise NotImplementedError

  def clip_status(self, _status):
    clipped_status = copy.deepcopy(_status)
    status_penalty = add_penalty(_status, self.factor) + add_penalty(0, _status)
    self.cross_bound[_status > self.factor] = 1.
    self.cross_bound[_status < 0] = -1.
    clipped_status[_status > self.factor] = self.factor - 0.01
    clipped_status[_status < 0] = 0.01
    return clipped_status, status_penalty

  def update_status(self, action):
    self._status['val'] += action
    self.cliped_status, p = self.clip_status(self._status['val'])
    self.cliped_status = self.status_decode(self.cliped_status)
    self.penalty += np.sum(p)

  def set_params(self, params):
    self.params = params
    if len(params) <= 0:
      return
    update_parameter('outparameter', self.params)

  @abstractmethod
  def get_loss(self, status):
    raise NotImplementedError

  def get_reward(self, loss):
    return - loss
  
  def acting(self, action):
    # self.action, p = self.clip_action(action)
    # self.penalty += p
    self.update_status(action)
    states = self.get_states(self.cliped_status)
    states['val'] += self.penalty
    states['grad'][self.cross_bound > 0] = self.factor
    states['grad'][self.cross_bound < 0] = -self.factor
    self.set_observations(states)
    self.cross_bound = np.zeros_like(self.cross_bound)
    self.penalty = 0
    
    reward = self.get_reward(states['val'][0])
    self.loss = states['val'][0]
    return reward

  def observe(self):
    raise NotImplemented

  def clip_action(self, action):
    clipped_action = copy.deepcopy(action)
    action_penalty = (sum(add_penalty(action, self.action_space.high)) + sum(add_penalty(self.action_space.low, action)))
    clipped_action[action > self.action_space.high] = self.action_space.high[action > self.action_space.high]
    clipped_action[action < self.action_space.low] = self.action_space.low[action < self.action_space.low]
    return clipped_action, action_penalty

  def step(self, action):
    """
    :param action:
    :return:
      observation (object):
      reward (float): sum of rewards
      done (bool): whether to reset environment or not
      info (dict): for debug only
    """
    self.actions = action
    self.nb_step += 1
    self.min_reward = min(self.min_reward, self.reward)
    try:
    # if True:
      self.reward = self.acting(self.actions)
      # self.reward -= self.penalty
      observation = self.observe()

      done = self.nb_step >= self.max_steps# or status_cond
    except ValueError as e:
      print(e)
      print('ValueError catched')
      observation = self.observe()
      done = True
      self.reward = self.min_reward - 1
    
    if reward_normalize:
      self.reward /= self.init_loss

    self.info['vtrue'].append(self.loss)
    return observation, self.reward, done, self.info

