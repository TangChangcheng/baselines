from gym import spaces
from copy import deepcopy
from gym.core import Env
from baselines.hyperparams import *

import matplotlib.pyplot as plt
plt.style.use('ggplot')

interval = 0.05
status = np.arange(-3, 3, interval)
actions = np.arange(-6, 6, interval)


class Convex(Env):
    id = 'Convex'
    def __init__(self, nb_status=8, max_steps=10, params=None, status=None, H=10, params_mean=0, params_std=1, status_mean=0, status_std=1.):
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
        self.last_reward = 0
        self.nb_plot = 0
        self.is_training = True
        self.is_ploting = False
        self.plt = plt
        self.plot_row = 1
        self.plot_col = 1

        self.max_steps = max_steps
        self.reward_range = (-np.inf, 0)

        if params is None:
            self.is_params_setted = False
        else:
            self.coefs = np.array(params[0])
            self.bias = np.array(params[1])
            self.is_params_setted = True

        shape_status = nb_status
        shape_past_val = H
        shape_past_grad = H * nb_status
        shape_w = self.coefs.flatten().shape[0]
        shape_bias = self.bias.shape[0]
        
        if obs == 'grad':
            shapes = [shape_past_grad]
        elif obs == 'grad_val':
            shapes = [shape_past_grad, shape_past_val]
        else:
            shapes = [shape_status]
        
        if observe_params:
            shapes.extend([shape_w, shape_bias])

        self.observation_space = spaces.Box(-10, 10, shape=(sum(shapes), ))

        if status is None:
            self.is_status_setted = False
        else:
            self.status = np.array(status)
            self.setted_status = np.array(status)
            self.is_status_setted = True

        self.reset()

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
        self.loss = np.sum(self.foo(self.status))
        self.init_loss = deepcopy(self.loss)
        self.nb_step = 0

        self.losses = [self.init_loss]

        self.history_observation = {
            'val': [self.init_loss],
            'gradient': [[0] * self.action_shape[0]] * self.H
        }

        self.history_observation['gradient'][-1] = self.gradient(self.status)
        
        self.info = {'vtrue': []}

        # print('init_loss = ', self.loss)
        return self.observe(self.loss)

    def seed(self, _int):
        np.random.seed(_int)

    def observe(self, current_loss):
        # return np.concatenate([np.array(self.status), self.coefs.flatten(), self.bias])
        res = []
        if obs == 'grad':
            res.append(self.observe_grad(current_loss))
        elif obs == 'grad_val':
            res.extend([self.observe_val(current_loss), self.observe_grad(current_loss)])
        else:
            res.append(self.status)
        
        if observe_params:
            res.extend([self.coefs.flatten(), self.bias])
        
        return np.concatenate(res)
    
    
    def observe_grad(self, current_loss=None):
        return np.array(self.history_observation['gradient']).flatten()

    def observe_val(self, current_loss):
        if len(self.history_observation['val']) < self.H:
            self.history_observation['val'] = [0] * (self.H - len(self.history_observation['val'])) + \
                                              [current_loss - k for k in self.history_observation['val']]
        else:
            self.history_observation['val'] = self.history_observation['val'][-self.H:]
        return np.array(self.history_observation['val'])


    def observe3(self, current_loss):
        return np.concatenate([np.array(self.history_observation['gradient']).flatten(),
                              self.coefs.flatten(),
                              self.bias])


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

        # action[action < self.action_space.low] = self.action_space.low[action < self.action_space.low]
        # action[action > self.action_space.high] = self.action_space.low[action > self.action_space.high]


        self.status += action

        self.action = action
        tmp = self.foo(self.status)
        # self.history_observation['val'].pop(0)
        self.history_observation['val'].append(tmp)
        grad = self.gradient(self.status)
        self.history_observation['gradient'].pop(0)
        self.history_observation['gradient'].append(grad)

        observation = self.observe(tmp)
        self.last_reward = self.reward
        # self.reward = self.loss - tmp
        self.reward = -tmp
        self.loss = tmp

        if self.is_training:
            # done = np.any(action <= self.action_space.low) or np.any(action >= self.action_space.high) or self.loss > 1000 or self.nb_step > self.max_steps
            done = self.nb_step >= self.max_steps
        else:
            done = self.loss > 1000 or self.nb_step >= self.max_steps
        
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
