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
      try:
        tangents = tuple([torch.randn_like(param) for param in self.params])
      except:
        tangents = tuple([torch.randn(1)])
      # Calculate f and jvp
      if not self.learning:
        f = self.function
      else:
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

criterion = None

opt = forwardSGD(lambda x: x[0]**2, None, (torch.tensor([4]), ), lr=0.1, learning=False)    

#def criterion(params, fmodel, input, target):
#    y = fmodel(params, buffers, input)
#    return _xent(y, target)

print(opt.step())
for i in range(20):
  opt.step()
opt.step()