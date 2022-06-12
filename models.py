import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
      super(Net, self).__init__()
      self.flatten = nn.Flatten()
      self.fc1 = nn.Linear(28*28, 1024)
      self.fc2 = nn.Linear(1024, 1024)
      self.activ = nn.ReLU()
      self.fc3 = nn.Linear(1024,10)

  def forward(self, x):
      x = self.flatten(x)
      x = self.activ(self.fc1(x))
      x = self.activ(self.fc2(x))
      logits = self.fc3(x)
      return logits

class ConvNet(nn.Module):
  def __init__(self):
      super(ConvNet, self).__init__()
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
    x = self.activation(self.conv1(x))
    x = self.pool(self.activation(self.conv2(x)))
    x = self.activation(self.conv3(x))
    x = self.pool(self.activation(self.conv4(x)))
    x = self.flatten(x)
    x = self.activation(self.fc1(x))
    x = self.fc2(x)
    return x

class LogisticRegression(nn.Module):

  def __init__(self, input_dim, output_dim):
    super(LogisticRegression, self).__init__()
    self.linear = nn.Linear(input_dim, output_dim)

  def forward(self, x):
    outputs = torch.sigmoid(self.linear(x.double()))
    return outputs


class CifarNet(nn.Module):
  def __init__(self):
      super(ConvNet, self).__init__()
      self.conv1 = nn.Conv2d(3, 32, 3)
      self.conv2 = nn.Conv2d(32, 32, 3)
      self.pool1 = nn.MaxPool2d(2, 2)
      self.conv3 = nn.Conv2d(32, 64, 3)
      self.conv4 = nn.Conv2d(64, 64, 3)
      self.pool2 = nn.MaxPool2d(2, 2)
      self.conv5 = nn.Conv2d(64, 128, 3)
      self.conv6 = nn.Conv2d(128, 128, 3)
      self.pool3 = nn.MaxPool2d(2, 2)
      self.flatten = nn.Flatten()
      self.fc1 = nn.Linear(1024, 1024)
      self.fc2 = nn.Linear(1024, 10)
      self.activation = nn.ReLU()

  def forward(self, x):
    x = self.activation(self.conv1(x))
    x = self.pool1(self.activation(self.conv2(x)))
    x = self.activation(self.conv3(x))
    x = self.pool2(self.activation(self.conv4(x)))
    x = self.activation(self.conv5(x))
    x = self.pool3(self.activation(self.conv6(x)))
    x = self.flatten(x)
    x = self.activation(self.fc1(x))
    x = self.fc2(x)
    return x
