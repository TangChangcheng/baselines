
import subprocess
import numpy as np
import re
import copy
from gym import spaces

from util import *
from BaseEnvironment import BaseEnv

SPICE_SCRIPT = 'hspice result/amp.sp'
EPS = 1e-3

num_status = 2
status_min = np.array([1.0, 0.65, 1.0, 0.65, 0.65] + [5e-7] * num_status)
status_max = np.array([1.2, 0.85, 1.2, 0.85, 0.85] + [9e-4] * num_status)

status_range = (status_min, status_max)

params_min = np.array([6e-7] * num_status)
params_max = np.array([2e-5] * num_status)
params_range = (params_min, params_max)

def get_band(string):
  try:
    match = re.compile(r'\bgbw=\s?([\d+.\-\w]+)\s').findall(string)[0]
    val = np.log10(float(match)) - 5
  except ValueError:
    print('Measure band failed.')
    val = 0
  return val

def get_power(string):
  #match = re.compile(r'\bpower\s+[\d+.\-e]+\s*[\d+.\-e]+\s*([\d+.\-e]+)\s?').findall(string)[0]
  match = re.compile(r'total voltage source power dissipation=\s*([\d+.\-e]+\s*)').findall(string)[0]
  val = np.log10(float(match))
  return val


def get_gain(string):
  match = re.compile(r'\bdcgain=\s?([\d+.\-eE]*)\s').findall(string)[0]
  val = float(match)
  return val


def call(status):
  res = subprocess.check_output(SPICE_SCRIPT, shell=True)
  res = res.decode('utf8')
  return res


class OPAEnv(BaseEnv):

  def __init__(self, nb_status, nb_params, max_step, status=None, params=None, encode_factor=10, penalty_factor=1, **kwargvs):
    self.Gain = OptElement(0, 70, threshold=30, positive=True)
    self.Band = OptElement(8, 11, threshold=9, positive=True)
    self.Power = OptElement(-5, 2, threshold=-4.5, positive=False)
    self.penalty_factor = penalty_factor

    super().__init__(nb_status, nb_params, max_step, status, params, encode_factor, **kwargvs)



  def init_status(self):
    return np.zeros(self.status_shape)

  def get_opt_elements(self, string):
    # the first element is the primary optimization object
    # others are constrain subjection
    try:
      self.Gain.val = get_gain(string)
      # self.Band.val = get_band(string)
      self.Power.val = get_power(string)
      # print(self.Gain._val, ' ', self.Band._val, ' ', self.Power._val)
      return self.Gain, (self.Power, )

    except ValueError as e:
      print(e)
      raise ValueError(str(e))

  def get_loss(self, status):
    update_parameter('inparameter', status)
    res = call(status)

    primary, others = self.get_opt_elements(res)
    if others is None:
      others = ()

    if isinstance(self.penalty_factor, int) or \
        isinstance(self.penalty_factor, float):
      self.penalty_factor = [self.penalty_factor] * (len(others) + 1)
    elif len(self.penalty_factor) == 1:
      self.penalty_factor = [self.penalty_factor[0]] * (len(others) + 1)
    elif len(self.penalty_factor) != len(others) + 1:
      raise ValueError('the number of penlaty factor must be equal to optElements')

    loss = primary.val * self.penalty_factor[0] + sum(
      [elem.penalty * pf for elem, pf in zip(others, self.penalty_factor[1:])]
    )

    print('primary_val: ', primary.val * self.penalty_factor[0])
    for elem, pf in zip(others, self.penalty_factor[1:]):
      print('other penalty: ', elem.penalty * pf)
    return loss




if __name__ == '__main__':
  env = OPAEnv(7, 2, 2, status_range=status_range, params_range=params_range)
