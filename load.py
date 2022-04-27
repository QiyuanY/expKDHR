
import numpy as np
import torch

input_data = np.load(r"./data/hh_graph.npy")
data = input_data.tolist()

hh_edge_index = torch.tensor(data, dtype=torch.long) - 390  # 边索引需要减去390
hh_x = torch.tensor([[i] for i in range(390, 1195)])

print(hh_x)

