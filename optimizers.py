import torch
import functorch as ft
from functools import partial
import numpy as np

# Numerical stabilizer
DELTA = 1e-8

class ForwardSGD():
  def __init__(self, fmodel, criterion, params, lr=2e-4, momentum=0, nesterov=False, decay = 1e-4):
    self.lr = lr
    self.original_lr = lr
    self.momentum = momentum
    self.function = fmodel
    self.params = params
    self.criterion = criterion
    self.steps = 0
    self.decay = decay
    self.nesterov = nesterov
    if momentum > 0:
      self.velocities = tuple([torch.zeros_like(param) for param in self.params])

  def step(self, image, label):
    self.steps+=1
    with torch.no_grad():

      tangents = tuple([torch.randn_like(param) for param in self.params])

      # Calculate f and jvp
      f = partial(
            self.criterion,
            fmodel=self.fmodel,
            x=image,
            t=label
      )

      if not self.nesterov:
        loss, jvp = ft.jvp(f, (self.params, ), (tangents, ))
      else :
        # Calculate gradient from position obtained by taking momentum step
        look_ahead_params = tuple([param - self.momentum * prev_v for param, prev_v in zip(self.params, self.velocities)])
        loss, jvp = ft.jvp(f, (look_ahead_params, ), (tangents, ))
      
      gradients = [jvp * t for t in tangents]

      # Velocity update
      if self.momentum > 0:
        self.velocities = tuple([self.momentum * v + g for v, g in zip(self.velocities, gradients)])
        gradients = self.velocities
      
      # Param + LR update
      self.params = tuple([param - g * self.lr for param, g in zip(self.params, gradients)])
      self.lr = self.original_lr*np.exp(-self.steps*self.decay)

      return self.params, loss


class ForwardRMSprop():
  def __init__(self, fmodel, criterion, params, lr=2e-4, alpha=0.99, momentum=0, decay=1e-4):
    self.fmodel = fmodel
    self.criterion = criterion
    self.params = params
    self.lr0 = lr
    self.lr = lr
    self.alpha = alpha
    self.momentum = momentum
    self.decay = decay
    self.steps = 0
    # Squared gradient average
    self.squared_grads = tuple([torch.zeros_like(param) for param in self.params])
    if momentum > 0:
      self.velocities = tuple([torch.zeros_like(param) for param in self.params])

  def step(self, image, label):
    self.steps += 1
    with torch.no_grad():
      # Calculate f and gradients
      f = partial(
            self.criterion,
            fmodel=self.fmodel,
            x=image,
            t=label
      )
      tangents = tuple([torch.randn_like(param) for param in self.params])
      loss, jvp = ft.jvp(f, (self.params, ), (tangents, ))
      gradients = [jvp * t for t in tangents]


      # Accumulated squared gradient (RMSProp with Nesterov Momentum)
      self.squared_grads = tuple([self.alpha * s_g + (1 - self.alpha) * g * g for s_g, g in zip(self.squared_grads, gradients)])

      delta = tuple([g / torch.sqrt(s_g + DELTA) for g, s_g in zip(gradients, self.squared_grads)])

      if self.momentum > 0:
        self.velocities = tuple([self.momentum * v + d for v, d in zip(self.velocities, delta)])
        self.params = tuple([param - self.lr * v for param, g in zip(self.params, self.velocities)])
      else:
        self.params = tuple([param - self.lr * d for param, d in zip(self.params, delta)])

      self.lr = self.lr0*np.exp(-self.steps*self.decay)
      
      return self.params, loss

class ForwardAdam():
  def __init__(self, fmodel, criterion, params, lr=0.001, betas=(0.9, 0.999), decay=1e-4):
    self.fmodel = fmodel
    self.criterion = criterion
    self.params = params
    self.lr0 = lr
    self.lr = lr
    self.b1 = betas[0]
    self.b2 = betas[1]
    self.decay = decay
    self.steps = 0
    self.moment1 = tuple([torch.zeros_like(p) for p in self.params])
    self.moment2 = tuple([torch.zeros_like(p) for p in self.params])

  def step(self, image, label):
    self.steps += 1
    with torch.no_grad():
      tangents = tuple([torch.randn_like(p) for p in self.params])

      f = partial(
          self.criterion, 
          fmodel=self.fmodel,
          x=image,
          t=label
      )
      loss, jvp = ft.jvp(f, (self.params, ), (tangents, ))

      gradients = tuple([jvp * t for t in tangents])
      
      # Update moments, and normalize
      self.moment1 = tuple([self.b1 * m1 + (1 - self.b1) * g for m1, g in zip(self.moment1, gradients)])
      self.moment2 = tuple([self.b2 * m2 + (1 - self.b2) * g * g for m2, g in zip(self.moment2, gradients)])
      normalized_m1 = tuple([m1 / (1 - self.b1 ** self.steps) for m1 in self.moment1])
      normalized_m2 = tuple([m2 / (1 - self.b2 ** self.steps) for m2 in self.moment2])

      self.params = tuple([
        p - self.lr * m1 / torch.sqrt(m2 + DELTA) for p, m1, m2 in zip(*[self.params, normalized_m1, normalized_m2])
      ])
      self.lr = self.lr0*np.exp(-self.steps*self.decay)

      return self.params, loss
