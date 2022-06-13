#@title ## Imports
"""Helper function for optimizing optimization test functions."""

import os
import numpy as np
import math
import sys
import time
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import functorch as ft

def optimize(loss_func, params, steps, optimizer=None, lr = 3e-4):
    """ Takes as input a loss function and parameters and optimizes the function."""
    losses = []
    grads = []
    parameters = []
    if optimizer=="SGD":
      optimizer = torch.optim.SGD(params, lr)
    for ii in range(steps):
        if optimizer is not None:
          parameters.append(tuple([params[i].item() for i in range(len(params))]))
          optimizer.zero_grad()
          loss = loss_func(params)
          loss.backward()
          optimizer.step()
          for i in range(len(params)):
            grads.append(params[i].grad.item())

          losses.append(loss.item())
        else:
          parameters.append(params)
          tangents = []
          for i in range(len(params)):
            tangents.append(torch.randn(1))
          tangents = tuple(tangents)
          f, jvp = ft.jvp(loss_func, params, tangents)
          losses.append(f.item())

          gradients = [jvp.mul(tangent) for tangent in tangents]
          grads.append(gradients)
          with torch.no_grad():
            new_params = []
            for g, param in zip(gradients, params):
              new_param = param - lr*g
              new_params.append(new_param)
            params = tuple(new_params)
    return losses, [element.item() for element in np.asarray(grads).ravel()], [element.item() for element in np.asarray(parameters).ravel()]

