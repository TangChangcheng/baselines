
import subprocess
import numpy as np
import re
import copy
from gym import spaces

from util import *
from baselines.hyperparams import *
from BaseEnvironment import BaseEnv

SPICE_SCRIPT = 'hspice /home/tangcc/_march/Documents/Project/baselines/baselines/Environments/result/amp.sp'





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


primary = OptElement(0, 70, threshold=30, positive=True)

class OPA(BaseEnv):

  def __init__(self, nb_status, nb_params, max_steps, status=None, params=None, factor=10, encode_fn=lambda x, *y: x, decode_fn=lambda x, *y: x, H=1, penalty_factor=1, **kwargvs):
    self.Gain = OptElement(0, 70, threshold=30, positive=True)
    self.Band = OptElement(8, 11, threshold=9, positive=True)
    self.Power = OptElement(-5, 2, threshold=-4.5, positive=False)
    self.penalty_factor = penalty_factor

    super(OPA, self).__init__(nb_status, nb_params, max_steps, status, params, factor,encode_fn, decode_fn, H, **kwargvs)

  def init_status(self):
    return np.zeros(self.status_shape)
  
  def get_states(self, status):
    loss, grad = self.get_loss(status)
    return {
      'val': [loss],
      'grad': grad,
      'params': self.params,
      'status': self.status,
    }
  

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
    
  def gradient(self, loss, delta_loss):
    delta = delta_loss - loss
    grad = np.concatenate([delta[:5] / delta_bias, delta[5:]/delta_w], axis=0)
    return grad_normalize(grad)
  
  def observe(self):
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

  def get_loss(self, status):
    update_parameter('inparameter', status)
    call(status)
    
    with open('amp.ma0', 'r') as fr:
      a = fr.read().strip().split('\n')[-nb_status - 1:]
    
    a = [float(re.match(r'\s*([+\-.\d]+)', line).group()) for line in a]
    

    vals = [1 - primary.normalize(l) for l in a]
    grad = self.gradient(vals[0], np.array(vals[1:]))
    others = None
    
    # primary, others = self.get_opt_elements(res)
    if others is None:
      others = ()

    if isinstance(self.penalty_factor, int) or \
        isinstance(self.penalty_factor, float):
      self.penalty_factor = [self.penalty_factor] * (len(others) + 1)
    elif len(self.penalty_factor) == 1:
      self.penalty_factor = [self.penalty_factor[0]] * (len(others) + 1)
    elif len(self.penalty_factor) != len(others) + 1:
      raise ValueError('the number of penlaty factor must be equal to optElements')

    loss = vals[0] * self.penalty_factor[0] + sum(
      [elem.penalty * pf for elem, pf in zip(others, self.penalty_factor[1:])]
    )

    print('primary_val: ', vals[0] * self.penalty_factor[0])
    for elem, pf in zip(others, self.penalty_factor[1:]):
      print('other penalty: ', elem.penalty * pf)
    return loss, grad




if __name__ == '__main__':
  env = OPAEnv(7, 2, 2, status_range=status_range, params_range=params_range)
