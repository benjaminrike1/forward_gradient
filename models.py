"""The models implemented in the project."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  """The vanilla neural network used for the MNIST dataset."""

  def __init__(self):
      super(Net, self).__init__()
      # defining layers
      self.flatten = nn.Flatten()
      self.fc1 = nn.Linear(28*28, 1024)
      self.fc2 = nn.Linear(1024, 1024)
      self.activ = nn.ReLU()
      self.fc3 = nn.Linear(1024,10)

  def forward(self, x):
      # makes one forward pass thorugh the network
      x = self.flatten(x)
      x = self.activ(self.fc1(x))
      x = self.activ(self.fc2(x))
      logits = self.fc3(x)
      return logits

class ConvNet(nn.Module):
  """The convolutional neural network used for the MNIST dataset."""
  def __init__(self):
      super(ConvNet, self).__init__()
      # defining layers
      self.conv1 = nn.Conv2d(1, 64, 3)
      self.pool = nn.MaxPool2d(2, 2)
      self.conv2 = nn.Conv2d(64, 64, 3)
      self.conv3 = nn.Conv2d(64, 64, 3)
      self.conv4 = nn.Conv2d(64, 64, 3)
      self.flatten = nn.Flatten()
      self.fc1 = nn.Linear(1024, 1024)
      self.fc2 = nn.Linear(1024, 10)
      self.activation = nn.ReLU()

  def forward(self, x):
    # making one forward pass thorugh the network
    x = self.activation(self.conv1(x))
    x = self.pool(self.activation(self.conv2(x)))
    x = self.activation(self.conv3(x))
    x = self.pool(self.activation(self.conv4(x)))
    x = self.flatten(x)
    x = self.activation(self.fc1(x))
    x = self.fc2(x)
    # returning the logits, i.e. before softmax
    return x



class CifarNet(nn.Module):
  """The convolutional neural network used for the CIFAR-10 dataset.
  The network follows the VGG architecture, but with only three convolutional blocks and 
  two dense layers."""
  def __init__(self):
      super(CifarNet, self).__init__()
      # defining the three convolutional blocks
      self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
      self.conv2 = nn.Conv2d(64, 64, 3,  padding=1)
      self.pool1 = nn.MaxPool2d(2, 2)
      self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
      self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
      self.pool2 = nn.MaxPool2d(2, 2)
      self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
      self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
      self.pool3 = nn.MaxPool2d(2, 2)

      # the dense part of the network
      self.flatten = nn.Flatten()
      self.fc1 = nn.Linear(4096, 4096)
      self.fc2 = nn.Linear(4096, 10)

      # using ReLU as activation function
      self.activation = nn.ReLU()

  def forward(self, x):
    # making one forward pass thorugh the network
    x = self.activation(self.conv1(x))
    x = self.pool1(self.activation(self.conv2(x)))
    x = self.activation(self.conv3(x))
    x = self.pool2(self.activation(self.conv4(x)))
    x = self.activation(self.conv5(x))
    x = self.pool3(self.activation(self.conv6(x)))
    x = self.flatten(x)
    x = self.activation(self.fc1(x))
    x = self.fc2(x)
    # returning the logits
    return x
