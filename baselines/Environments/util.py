import numpy as np
import re
EPS = 1e-3


FILE_PATH = 'param/'

def add_penalty(f, constrain):
  return np.max([np.zeros_like(f), f - constrain], )
  return x * (x > 0)

def transform(val):
  return np.log10(val)
  # return val

def detransform(val):
  return np.power(10, val)
  # return val

def update_parameter(filename, params):
  filename += '{}'
  with open(FILE_PATH + filename.format('_template'), 'r') as ft, open(FILE_PATH + filename.format(''), 'w') as fw:
    content = ft.read()
    for i, p in enumerate(params, 1):
      content = re.sub(r'(\${})\s'.format(i), str(p) + '\n', content)
    if '$' in content:
      raise ValueError('got wrong number of parameters.')
    fw.write(content)


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
    return relu(diff)
