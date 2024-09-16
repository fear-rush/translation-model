import torch
from torch.utils.data import DataLoader

dataset = torch.randn(5, 4)
print(dataset)

test_3d_matrix = [
                  [[1, 2, 3], [4, 5, 6]], 
                  
                  [[7, 8, 9], [10, 11, 12]]
                ]

# print(test_3d_matrix[0][1][2])