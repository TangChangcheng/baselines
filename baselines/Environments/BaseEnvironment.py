
import numpy as np
import copy
from gym import spaces

import matplotlib.pyplot as plt

plt.style.use('ggplot')

from gym.core import Env


from util import *
from abc import abstractmethod
from easydict import EasyDict

class BaseEnv(Env):
  # 用_status（有下划线）访问归一化后的状态，status（无下划线）访问原始值，params同理
  # _status中min，max，range分别保存没有归一化的最小、最大值及其范围
  def __init__(self, nb_status, nb_params, max_step, status=None, params=None, encode_factor=10, **kwargvs):
    super(BaseEnv, self).__init__()
    
    self.action_shape = (nb_status,)
    self.action_space = spaces.Box(-1, 1, shape=self.action_shape)
    self.action = self.action_space.sample()

    self.status_shape = (nb_status,)
    self.params_shape = (nb_params,)

    if 'status_range' not in kwargvs.keys() or 'params_range' not in kwargvs.keys():
      raise ValueError('The range of status and params must be provided.')

    self._status = EasyDict({
      'val': np.array([0] * nb_status),
      'min': np.array(kwargvs['status_range'][0]),
      'max': np.array(kwargvs['status_range'][1]),
    })
    self._status.range = transform(self._status.max) - transform(self._status.min)

    self._params = EasyDict({
      'val': np.array([0] * nb_params),
      'min': np.array(kwargvs['params_range'][0]),
      'max': np.array(kwargvs['params_range'][1])
    })
    self._params.range = transform(self._params.max) - transform(self._params.min)

    self.observation_shape = (nb_status + nb_params,)
    self.observation_space = spaces.Box(-1., 1., shape=self.observation_shape)

    self.max_step = max_step

    self.reward = 0
    self.penalty = 0
    self.last_reward = 0
    self.min_reward = 0
    self._seed = 0
    self.is_training = True
    self.factor = encode_factor

    if params is None:
      self.is_params_setted = False
    else:
      self.set_params(self.params_encode(np.array(params)))
      self.is_params_setted = True

    self.reset(status)

  def seed(self, _int):
    np.random.seed(_int)

  @property
  def status(self):
    return self.status_decode(self._status.val, factor=self.factor)

  @status.setter
  def status(self, _status):
    self._status.val = self.status_encode(_status, factor=self.factor)

  @property
  def params(self):
    return self.params_decode(self._params.val, factor=self.factor)

  @params.setter
  def params(self, _params):
    self._params.val = self.params_encode(_params, factor=self.factor)

  def reset(self, status=None):
    print('\n--------------------------------------------------------------------------------')

    if not self.is_params_setted:
      self._params.val = np.array([np.random.uniform(0., self.factor)] * self.params_shape[0])
      self.set_params(self.params)

    # p = 0
    if status is None:
      self.status_init = self.init_status()
      self._status.val = copy.deepcopy(self.status_init)
      # self._status.val, p = self.clip_status(self._status.val)

    else:
      self.status = np.array(status)

    self.nb_step = 0
    self.loss = self.get_loss(self.status)

    self.init_loss = copy.deepcopy(self.loss)
    self.losses = [self.init_loss]
    # self.penalty += p

    return self.observe()

  def params_encode(self, val, factor=10):
    val = np.array(val)
    return factor * (transform(val) - transform(self._params.min)) / self._params.range

  def params_decode(self, val, factor=10):
    val = np.array(val)
    return detransform(val / factor * self._params.range + transform(self._params.min))

  def status_encode(self, val, factor=10):
    val = np.array(val)
    return factor * (transform(val) - transform(self._status.min)) / self._status.range

  def status_decode(self, val, factor=10):
    val = np.array(val)
    return detransform((val / factor) * self._status.range + transform(self._status.min))

  @abstractmethod
  def init_status(self):
    raise NotImplementedError

  def clip_status(self, _status):
    clipped_status = copy.deepcopy(_status)
    status_penalty = self.relu(_status - self.factor).sum() + \
                     self.relu(0 - _status).sum()
    clipped_status[_status > self.factor] = self.factor - 0.01
    clipped_status[_status < 0] = 0.01
    return clipped_status, status_penalty

  def update_status(self, action):
    self._status.val += action
    self._status.val, p = self.clip_status(self._status.val)
    self.penalty += p

  def set_params(self, params):
    self.params = params
    if len(params) <= 0:
      return
    update_parameter('outparameter', self.params)

  def relu(self, x):
    return x * (x > 0)

  @abstractmethod
  def get_loss(self, status):
    raise NotImplementedError

  def acting(self, action):
    self.action, p = self.clip_action(action)
    self.penalty += p
    self.update_status(action)
    new_loss = self.get_loss(self.status)
    reward = self.get_reward(new_loss)
    self.loss = new_loss
    return reward

  def get_reward(self, loss):
    return self.loss - loss

  def observe(self):
    return np.concatenate([np.array(self.status).reshape(-1), np.array(self.params).reshape(-1)])

  def clip_action(self, action):
    clipped_action = copy.deepcopy(action)
    action_penalty = (sum(self.relu(action - self.action_space.high)) + sum(self.relu(self.action_space.low - action)))
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
    # print(self.status, action)
    self.action = action
    self.nb_step += 1
    self.last_reward = self.reward
    self.min_reward = min(self.min_reward, self.reward)
    try:
    # if True:
      self.reward = self.acting(self.action)
      self.reward -= self.penalty
      self.penalty = 0
      observation = self.observe()

      status_cond = np.any(self.status > self._status.max) or np.any(self.status < self._status.min)
      if self.is_training:
        done = self.nb_step >= self.max_step# or status_cond
      else:
        done = self.nb_step >= 2 * self.max_step# or status_cond

      self.losses.append(self.loss)
    except ValueError as e:
      print(e)
      print('ValueError catched')
      observation = self.observe()
      done = True
      self.reward = self.min_reward - 1

    info = {}
    return observation, self.reward, done, info

  def render(self, mode='human', close=False):
    # print('\ninit: ', self.init_status)
    print('Step: ', self.nb_step)
    print('init loss: ', self.init_loss)
    print('params: ', self.params)
    print('reward: ', self.reward)
    print('action: ', self.action)
    print('loss: ', self.loss)
    print('status: ', self._status.val)

