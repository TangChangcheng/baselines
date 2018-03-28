import numpy as np
import re
from baselines.hyperparams import delta_bias, delta_w, EPS,param_path




def add_penalty(f, constrain):
  res = f - constrain
  return np.max([np.zeros_like(res), f - constrain], axis=0)

def update_parameter(filename, params):
  filename += '{}'
  with open(param_path + filename.format('_template'), 'r') as ft, open(param_path + filename.format(''), 'w') as fw:
    content = ft.read()
    content = content.format(*params, delta_bias=delta_bias, delta_w=delta_w)
    fw.write(content)

def grad_normalize(grad):
  return grad

class OptElement(object):
  def __init__(self, _min, _max, threshold=None, positive=False):
    # if positive is True, the penalty is added when value is less than threshold
    self.min = _min
    self.max = _max
    self.range = _max - _min
    self.threshold = threshold
    self.positive = positive
    self.normed_threshold = (threshold - _min) / self.range
    if self.positive:
      self.normed_threshold *= -1

    self._val = 0

  def normalize(self, val):
    normed_val = (val - self.min) / self.range
    return normed_val

  @property
  def val(self):
    normed_val = self.normalize(self._val)
    if self.positive:
      normed_val = -normed_val
    return normed_val

  @val.setter
  def val(self, val):
    self._val = val

  @property
  def penalty(self):
    # when positive is false, penalty is added when val > T
    # when positive is true, penalty is added when -val > -T => val < T
    diff = (self.val - self.normed_threshold) / (abs(self.normed_threshold) + EPS)
    diff[diff < 0] = 0
    return diff
