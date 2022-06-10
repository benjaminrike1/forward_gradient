import torch
import functorch as ft
from functools import partial
import numpy as np


class forwardSGD():
  def __init__(self, function, criterion, params, lr=2e-4, momentum=False, learning=True, decay = 1e-4):
    self.lr = lr
    self.original_lr = lr
    self.momentum = False
    self.function = function
    self.params = params
    self.criterion = criterion
    self.learning = learning
    self.steps = 0
    self.decay = decay

  def step(self, image=None, label=None):
    self.steps+=1
    with torch.no_grad():
      tangents = tuple([torch.randn_like(param) for param in self.params])
      # Calculate f and jvp
      f = partial(
            self.criterion,
            fmodel=self.function,
            x=image,
            t=label
            )
      f_t, jvp = ft.jvp(f, (self.params, ), (tangents, ))
      gradients = [jvp.mul(t) for t in tangents]
      nye = []
      for g, param in zip(gradients, self.params):
        new_params = param - g*self.lr
        nye.append(new_params)
      self.params = tuple(nye)
      self.lr = self.original_lr*np.exp(-self.steps*self.decay)

      return self.params, f_t