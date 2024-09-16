import torch
import math

# pe = torch.zeros(2, 3)
# print(pe)

# position = torch.arange(0, 4, dtype=torch.float).unsqueeze(1)
# print(position.size())

# sp = torch.randn(4);
# print(sp)

# div_term = torch.exp(torch.arange(0, 4, 2).float()) * (-math.log(100000.0) / 4)
# print(div_term.size())

pe = torch.zeros(8, 4)
position = torch.arange(0, 8, dtype=torch.float).unsqueeze(1)
div_term = torch.exp(torch.arange(0, 4, 2).float()) * (-math.log(100000.0) / 4)
div_term = div_term.unsqueeze(0)
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)

print(f'position = {position}')
print(f'pe = {pe}')
print(position.size())
print(div_term)
print(pe.size())

arr = [[0, 0],
       [-2.8782, -21.2674],
       [-5.7564, -42.5348],
       [-8.6346, -63.8022],
       [-11.5128,	-85.0696],
      [-14.391,	-106.337],
      [-17.2692,	-127.6044],
      [-20.1474,	-148.8718]]

tnsr = torch.FloatTensor(arr)

print(torch.sin(tnsr))

# class Test:
  
#   def __init__(self):
#     super().__init__()
#     self.num = 1
#     pe = 2
  
#   def func(self):
#     print(self.pe)
    

# obj = Test()
# obj.func()