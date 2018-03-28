from gym import spaces
from copy import deepcopy
from gym.core import Env
from baselines.hyperparams import *

import functools

import matplotlib.pyplot as plt
plt.style.use('ggplot')

class Convex(Env):
    id = 'Convex'
    def __init__(self, nb_status=8, max_steps=10, params=None, status=None, H=10, params_mean=0, params_std=1, status_mean=0, status_std=1., **kwargs):
        super(Convex, self).__init__()
        
        self.action_shape = (nb_status, )
        self.action_space = spaces.Box(-1., 1., shape=self.action_shape)
        self.action = self.action_space.sample()

        self.H = H

        self.params_mean = params_mean
        self.params_std = params_std
        self.status_mean = status_mean
        self.status_std = status_std

        self.coefs = np.ones(self.action_shape * 2)
        self.bias = np.zeros(self.action_shape)

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
        
        reduce_mul = lambda seq: functools.reduce(lambda x, y: x * y, seq)
        
        obs_shape = [(H, ) + sensor_dims[obs] for obs in obs_include]
        meta_shape = [sensor_dims[meta] for meta in meta_include]

        self.observation_space = spaces.Box(-10, 10,shape=(sum([reduce_mul(tup) for tup in obs_shape + meta_shape]), ))
        
        self.observations = {
          obs: np.zeros((0, ) + sensor_dims[obs])
          for obs in obs_include
        }
        self.metas = {
          meta: np.zeros_like(sensor_dims[meta])
          for meta in meta_include
        }

        if status is None:
            self.is_status_setted = False
        else:
            self.status = np.array(status)
            self.setted_status = np.array(status)
            self.is_status_setted = True

        self.reset()
      
    def get_states(self, status):
      return {
        'grad': self.gradient(status),
        'val': [self.foo(status)],
        'params': self. coefs,
        'bias': self.bias,
      }
    
    def set_observations(self, states):
      for obs in obs_include:
        self.observations[obs] = np.append(self.observations[obs], [states[obs]], axis=0)
    
    def set_metas(self, states):
      for meta in meta_include:
        self.metas[meta] = states[meta]
    
    
    def foo(self, x):
        coefs = self.coefs
        y = np.sum(np.power(np.matmul(coefs, x) - self.bias, 2))
        if ln:
            y = np.log(y + np.e)
        return y

    def get_loss(self):
        return self.foo(self.status)

    def gradient(self, x):
        grad = 2 * np.matmul(np.transpose(self.coefs), (np.matmul(self.coefs, x) - self.bias))
        if ln:
            grad = grad / (np.sum(np.power(np.matmul(self.coefs, x) - self.bias, 2)) + np.e)
        return grad

    def reset(self, status=None):
        # print('\n--------------------------------------------------------------------------------')
        # self.coefs = np.random.uniform(0, 1, self.coefs.shape)

        if status is None and not self.is_status_setted:
            self.status = random_fn(self.status_mean, self.status_std, size=self.action_shape)
        elif status is not None:
            self.status = np.array(status)
        elif self.is_status_setted:
            self.status = deepcopy(self.setted_status)

        if not self.is_params_setted:
            self.coefs = random_fn(self.params_mean, self.params_std, size=np.shape(self.coefs))
            self.bias = random_fn(self.params_mean, self.params_std, size=np.shape(self.bias))
        self.init_status = deepcopy(self.status)
        states = self.get_states(self.status)
        self.set_observations(states)
        self.set_metas(states)
        
        self.nb_step = 0

        self.init_loss = states['val'][0]
        self.loss = self.init_loss
        self.info = {'vtrue': [self.init_loss]}

        # print('init_loss = ', self.loss)
        return self.observe()

    def seed(self, _int):
        np.random.seed(_int)

    def observe(self):
        observations = []
        for obs in obs_include:
          v = self.observations[obs]
          if obs == 'val':
            v = self.loss - v
          v = np.concatenate([
                              np.zeros((max(0, self.H - len(v)), ) + sensor_dims[obs]),
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

        self.status += action
        self.action = action
        states = self.get_states(self.status)
        tmp = float(states['val'][0])

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

    def render(self, mode='human', close=False):
        # print('\ninit: ', self.init_status)
        # print('coefs: ', self.coefs)
        print('reward: ', self.reward)
        # print('action: ', self.action)
        print('init_loss', self.init_loss)
        print('loss: ', self.loss)
        print('status: ', self.status)
        # print('solution', self.solution())
        # print('delta', self.status - self.solution())
        # print('bias: ', self.bias)

        if self.is_ploting:
            self.ax.plot([self.i, self.i + 1], self.losses[-2:], 'r')
            plt.pause(0.001)
            self.i += 1

    def solution(self):
        return -self.coefs[1]/self.coefs[0]/2.0

y = -20 * np.exp(-0.2 * np.sqrt(np.sum(np.power(np.matmul(coef, x), 2), axis=-1))) - np.exp(
            np.sum(np.cos(2 * np.pi * (np.matmul(coef, x))) * coef[2], axis=-1)) + 20 + np.e