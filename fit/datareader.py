# -*- coding: utf-8 -*-
"""
@Time   : 2024/3/24 17:16
@Author : Yichao Li
"""
import torch
import pandas as pd
import os

current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "PropData.csv")
data = pd.read_csv(file_path, header=None)
tri_data = data.iloc[:, :3].values
T = data.iloc[:, 3].values.reshape(-1,1)
Q = data.iloc[:, 3].values.reshape(-1,1)

train_input = torch.tensor(tri_data[:80])
test_input = torch.tensor(tri_data[80:])
train_T = torch.tensor(T[:80])
test_T = torch.tensor(T[80:])
train_Q = torch.tensor(Q[:80])
test_Q = torch.tensor(Q[80:])
