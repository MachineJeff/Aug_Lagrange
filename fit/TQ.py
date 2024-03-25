# -*- coding: utf-8 -*-
"""
@Time   : 2024/3/24 17:30
@Author : Yichao Li
"""
import torch
import numpy as np
import os
from fit.fit_T import TwoHiddenLayerNet
from fit.fit_Q import ThreeHiddenLayerNet

current_dir = os.path.dirname(__file__)
T_path = os.path.join(current_dir, 'model_T.pth')
Q_path = os.path.join(current_dir, 'model_Q.pth')
model_T = torch.load(T_path)
model_Q = torch.load(Q_path)

def ensure_array(input_data):
    if not isinstance(input_data, np.ndarray):
        input_data = np.array(input_data)
    return input_data

def T(input_data):
    input_data = ensure_array(input_data)
    input_data = input_data.astype(np.float64)
    input_data = torch.tensor(input_data)
    model_T.eval()
    with torch.no_grad():
        output = model_T(input_data)
    return output.item()

def Q(input_data):
    input_data = ensure_array(input_data)
    input_data = input_data.astype(np.float64)
    input_data = torch.tensor(input_data)
    model_Q.eval()
    with torch.no_grad():
        output = model_Q(input_data)
    return output.item()

if __name__ == '__main__':
    t = T([1.1,7.25,2.99]) # real T = 6.3
    q = Q([1.1,7.25,2.99]) # real Q = 14.8
    print(np.round(t, 1))
    print(np.round(q, 1))

