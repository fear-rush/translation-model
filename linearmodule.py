import torch
import torch.nn as nn

class MyModule(nn.Module):
  
  def __init__(self):
    super().__init__()
    self.linears = nn.ModuleList([nn.Linear(10, 10) for _ in range(2)])
    
  def forward(self, x):
    for i, l in enumerate(self.linears):
      x = self.linears[i // 2](x) + l(x)
    
    return x
  

x = torch.randn(3, 10)
module = MyModule()
print(module(x))