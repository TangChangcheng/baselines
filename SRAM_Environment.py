
import subprocess
import numpy as np
import re
import copy
from gym import spaces

import matplotlib.pyplot as plt
plt.style.use('ggplot')

EPS = 1e-3

WNM_SHELL = 'observe.sh'
UPDATE_PARAMS = 'update_parameter.py'
FILE_PATH = 'smic40sram/'
# FILE_PATH = ''

inparams_means = np.array([ -3.541455e+00,
														-1.592342e-01,
														-4.096521e-02,
														3.736432e+00,
														1.009334e+00,
														-3.842951e-02,
														1.425451e-01,
														6.664235e-01,
														9.204750e-02,
														-4.316191e+00,
														-3.016369e-01,
														-9.477469e-03,
														-1.970786e+00,
														3.191499e-01,
														1.614013e-01,
														5.245454e-02,
														-7.042912e-01,
														4.578746e-01])
outparams_means = np.array([-2.000000e-01,
														4.500000e-01,
														4.500000e-01])



def update_parameter(filename, params):
	filename += '{}.ptr'
	with open(FILE_PATH + filename.format('_template'), 'r') as ft, open(FILE_PATH + filename.format(''), 'w') as fw:
		content = ft.read()
		for i, p in enumerate(params, 1):
			content = re.sub(r'(\${})\n'.format(i), str(p) + '\n', content)
		if '$' in content:
			raise ValueError('got wrong number of parameters.')
		fw.write(content)

class Env(object):
	def __init__(self, nb_status, nb_params, max_step, params=None, status=None, t=1, gamma=0.05):
		self.action_shape = (nb_status, )
		self.action_space = spaces.Box(-1, 1, shape=self.action_shape)
		self.action = self.action_space.sample()

		self.status_shape = (nb_status, )
		self.params_shape = (nb_params, )		

		self.observation_shape = (nb_status + nb_params, )
		self.observation_space = spaces.Box(-0.5, 0.5, shape=self.observation_shape)

		self.reward = 0
		self.min_reward = 0
		self.last_reward = 0

		self._seed = 0

		self.max_step = max_step
		self.t = t
		self.gamma = gamma

		if params is None:
			self.set_params(np.random.random(nb_params))
			self.is_params_setted = False
		else:
			self.set_params(np.array(params))
			self.is_params_setted = True

		if status is None:
			self.status = np.random.uniform(-1.5, 1.5, self.status_shape)
			self.is_status_setted = False
		else:
			self.status = np.array(status)
			self.is_status_setted = True

		self.nb_plot = 0
		self.is_training = True
		self.is_ploting = False
		self.plt = plt
		self.plot_row = 1
		self.plot_col = 1

		self.reset(status)

	def reset(self, status=None):
		print('\n\n--------------------------------------------------------------------------------')
		while True:
			try:
				if status is None and not self.is_status_setted:
					self.status = np.random.uniform(-1.5, 1.5, self.status_shape)
				elif status is not None:
					self.status = np.array(status)
	
				if not self.is_params_setted:
					self.set_params(np.random.uniform(1-0.2, 1+0.2, self.params_shape))

				self.update_status(0)

				self.init_status = copy.deepcopy(self.status)
				self.nb_step = 0
		
				self.loss = self.get_loss(self.status)
				break
			except ValueError:
				continue
		self.init_loss = copy.deepcopy(self.loss)
		self.losses = [self.init_loss]

		if self.is_ploting:
			self.i = 0
			self.nb_plot += 1
			self.fig = plt.figure(0)
			self.ax = self.fig.add_subplot(self.plot_row, self.plot_col, self.nb_plot)
			plt.ion()

		# print('init_loss = ', self.loss)
		return self.observe()

	def update_status(self, action):
		self.status += action

	def set_params(self, params):
		self.params = params
		update_parameter('outparameter', params * outparams_means)

	def seed(self, _int):
		np.random.seed(_int)

	def foo(self, status):
		# status *= inparams_means
		update_parameter('inparameter', status)
		output = subprocess.check_output('./{}'.format(WNM_SHELL), shell=True)
		output = output.decode('utf-8').strip().split('\n')[-1].strip().split(' ')[0]
		return float(output)

	def relu(self, x):
		return x * (x > 0)

	def get_loss(self, status):
		# return np.sum(np.power(status, 2)) * self.t - np.log(self.gamma - self.foo(status))
		g = self.relu(self.foo(status) - self.gamma)
		
		# return np.mean(np.power(status * inparams_means, 2)) + self.t * g / (EPS + g)
		return np.sum(np.power(status, 2)) + self.t * g

	def acting(self, action):
		# self.update_status(self.status + action)
		self.update_status(action)
		new_loss = self.get_loss(self.status)
		reward = self.get_reward(new_loss)
		self.loss = new_loss
		return reward

	def get_reward(self, loss):
		reward = self.loss - loss
		self.min_reward = min(self.min_reward, reward)
		return reward
		# return -loss

	def observe(self):
		return np.concatenate([np.array(self.status), np.array(self.params)])

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
		self.action = action
		self.last_reward = self.reward
		
		try:	
			self.reward = self.acting(action)
			observation = self.observe()
			# done = np.abs(actions[action]) < 1e-4 or self.loss > 100 or self.nb_step >= self.max_step
			if self.is_training:
				done = np.any(action <= self.action_space.low) or np.any(action >= self.action_space.high) or self.loss > 1000 or self.nb_step > self.max_step
				#done = self.loss > 1000 or self.nb_step > self.max_step
			else:
				done = self.loss > 1000 or self.nb_step > 2 * self.max_step

			self.losses.append(self.loss)
		except ValueError:
			observation = self.observe()
			done = True
			self.reward = 0
		info = {}
		return observation, self.reward, done, info

	def render(self, mode='human', close=False):
		print('\ninit: ', self.init_status)
		print('init loss: ', self.init_loss)
		print('params: ', self.params)
		print('reward: ', self.reward)
		print('action: ', self.action)
		print('loss: ', self.loss)
		print('status: ', self.status)
		print('observe', np.sum(np.power(self.status, 2)))
		print('restrict', self.loss - np.sum(np.power(self.status, 2)))

		if self.is_ploting:
			self.ax.plot([self.i, self.i + 1], [self.last_reward, self.reward], 'r')
			plt.pause(0.001)
			self.i += 1

if __name__ == '__main__':
	update_parameter('inparameter', [2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.])
