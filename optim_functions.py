import torch

def beale(*args):
  if len(args)==1:
    args = args[0]
  x, y = args[0], args[1]
  return ((torch.tensor([1.5]) - x + x*y)**2 + 
         (torch.tensor([2.25]) - x + x*y*y)**2 + 
         (torch.tensor([2.625]) - x + x*y**3)**2)

def rosenbrock(*args):
  if len(args)==1:
    args = args[0]
  x, y = args[0], args[1]
  a, b = torch.tensor([1]), torch.tensor([100])
  return (a - x)**2 + b * (y - x*x)**2

