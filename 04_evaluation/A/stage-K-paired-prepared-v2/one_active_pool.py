import torch
class FullSwapPool:
 def __init__(self,dense,candidate):self.models={'dense':dense,'candidate':candidate};self.active=None
 def activate(self,name):
  if self.active!=name:
   if self.active is not None:self.models[self.active].to('cpu')
   torch.cuda.empty_cache();self.models[name].to('cuda:0');self.active=name
  return self.models[name]
 def release(self):
  if self.active is not None:self.models[self.active].to('cpu')
  self.active=None;torch.cuda.empty_cache()
