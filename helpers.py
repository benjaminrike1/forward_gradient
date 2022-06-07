#@title ## Imports
import os
import numpy as np
import math
import sys
import time
import matplotlib.pyplot as plt
import torch.autograd.forward_ad as fwAD
import torch
import seaborn as sns

def optimize(optimizer, loss_func, params, steps):
    print('Initial_loss: %.2f'%(loss_func(params)))
    losses = []
    grads = []
    for ii in range(steps):
        optimizer.zero_grad()
        loss = loss_func(params)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    plot_loss(losses, steps)
    return losses, grads

def plot_loss(losses, steps):
    fig, ax = plt.subplots(1,1, figsize=(15, 8))
    X = [i for i in range(steps)]
    ax.set_title("f(x) by steps")
    ax.set_xlabel('Steps')
    ax.set_ylabel('f(x)')
    sns.lineplot(x=X, y=losses, ax=ax)