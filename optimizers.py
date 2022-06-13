"""The forward gradient optimizers implemented in the project.
 All the optimizers takes as input a model to optimize, a criterion to optimize on,
 and a set of parameters to optimize. The optimizers then make one optimization step,
 and returns the new parameters and the loss."""

import torch
import functorch as ft
from functools import partial
import numpy as np  

# Numerical stabilizer
DELTA = 1e-8

class ForwardSGD():
  """Class which implements SGD and SGD with nesterov momentum for forward gradient"""
  def __init__(self, fmodel, criterion, params, lr=2e-4, momentum=0, nesterov=False, decay = 1e-6, clip = False):
    """Initializing optimizer.
    Args:
        fmodel: The network you are optimizing made functional in functorch
        criterion: The objective function
        params: The parameters of the network
        lr: the step size
        momentum: momentum parameter
        nesterov: use nesterov momentum if True
        decay: the learning rate decay used
        clip: The maximum norm of the Jacobian-vector product, used for gradient clipping
    """
    self.lr = lr
    self.original_lr = lr
    self.momentum = momentum
    self.fmodel = fmodel
    self.params = params
    self.criterion = criterion
    self.steps = 0
    self.decay = decay
    self.clip = clip
    self.nesterov = nesterov
    if momentum > 0:
      # saving velocities for momentum
      self.velocities = tuple([torch.zeros_like(param) for param in self.params])

  def step(self, image, label):
    """Making one optimizer step.
    Args:
        image: The input image
        label: The label
    """
    self.steps+=1
    with torch.no_grad():
      
      # sampling perturbation vectors form a multivariate Gaussian with mean 0 and unit variance
      tangents = tuple([torch.randn_like(param) for param in self.params])

      # making criterion a partial function such it can be called with only params as parameter
      f = partial(
            self.criterion,
            fmodel=self.fmodel,
            x=image,
            t=label
      )

      if not self.nesterov:
        # calculating the Jacobian-vector product
        loss, jvp = ft.jvp(f, (self.params, ), (tangents, ))
      else :
        # Calculate gradient from position obtained by taking momentum step
        look_ahead_params = tuple([param + self.momentum * prev_v for param, prev_v in zip(self.params, self.velocities)])
        loss, jvp = ft.jvp(f, (look_ahead_params, ), (tangents, ))
      
      # gradient clipping
      if self.clip and (self.clip<torch.abs(jvp)):
        if self.clip < jvp:
          jvp = self.clip
        else:
          jvp = -self.clip
      # calculating the forward gradient    
      gradients = [jvp * t for t in tangents]

      # Parameter updates
      if self.momentum > 0:
        self.velocities = tuple([self.momentum * v - self.lr*g for v, g in zip(self.velocities, gradients)])
        self.params = tuple([param + v for param, v in zip(self.params, self.velocities)])
      else:
        self.params = tuple([param - g * self.lr for param, g in zip(self.params, gradients)])
      
      # decaying learning rate
      self.lr = self.original_lr*np.exp(-self.steps*self.decay)

      return self.params, loss, jvp


class ForwardAdam():
  """Forward gradient implementation of the Adam optimizer"""
  def __init__(self, fmodel, criterion, params, lr=0.001, betas=(0.9, 0.999), decay=1e-4, clip = False):
    """Initializing optimizer.
    Args:
        fmodel: The network you are optimizing made functional in functorch
        criterion: The objective function
        params: The parameters of the network
        lr: the step size
        beta: tuple of decay parameters. The first element should be the decay rate for the first moment,
         and the second the decay rate for the second moment.
        decay: the learning rate decay used
        clip: The maximum norm of the Jacobian-vector product, used for gradient clipping
    """
    self.fmodel = fmodel
    self.criterion = criterion
    self.params = params
    self.lr0 = lr
    self.lr = lr
    self.b1 = betas[0]
    self.b2 = betas[1]
    self.decay = decay
    self.steps = 0
    self.clip = clip

    # keeping track of moment estimates
    self.moment1 = tuple([torch.zeros_like(p) for p in self.params])
    self.moment2 = tuple([torch.zeros_like(p) for p in self.params])

  def step(self, image, label):
    """Making one optimizer step.
    Args:
        image: The input image
        label: The label
    """
    self.steps += 1
    with torch.no_grad():
      # sampling perturbation vectors form a multivariate Gaussian with mean 0 and unit variance
      tangents = tuple([torch.randn_like(p) for p in self.params])

      # turing criterion into a partial function
      f = partial(
          self.criterion, 
          fmodel=self.fmodel,
          x=image,
          t=label
      )

      # calculating loss and Jacobian-vector product
      loss, jvp = ft.jvp(f, (self.params, ), (tangents, ))

      # gradient clipping
      if self.clip and (self.clip<torch.abs(jvp)):
        if self.clip < jvp:
          jvp = self.clip
        else:
          jvp = -self.clip

      # calculating forward gradient
      gradients = tuple([jvp * t for t in tangents])
      
      # Update moments, and normalize
      self.moment1 = tuple([self.b1 * m1 + (1 - self.b1) * g for m1, g in zip(self.moment1, gradients)])
      self.moment2 = tuple([self.b2 * m2 + (1 - self.b2) * g * g for m2, g in zip(self.moment2, gradients)])
      normalized_m1 = tuple([m1 / (1 - self.b1 ** self.steps) for m1 in self.moment1])
      normalized_m2 = tuple([m2 / (1 - self.b2 ** self.steps) for m2 in self.moment2])

      # calculating parameter update
      self.params = tuple([
        p - self.lr * m1 / torch.sqrt(m2 + DELTA) for p, m1, m2 in zip(*[self.params, normalized_m1, normalized_m2])
      ])

      # decaying learning rate
      self.lr = self.lr0*np.exp(-self.steps*self.decay)

      return self.params, loss