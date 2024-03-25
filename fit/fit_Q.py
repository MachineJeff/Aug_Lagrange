# -*- coding: utf-8 -*-
"""
@Time   : 2024/3/23 20:01
@Author : Yichao Li
"""
import torch
import torch.nn as nn
import torch.optim as optim
from fit.datareader import train_input, test_input, train_Q, test_Q

class ThreeHiddenLayerNet(nn.Module):
    def __init__(self):
        super(ThreeHiddenLayerNet, self).__init__()
        self.hidden1 = nn.Linear(3, 10)
        self.hidden2 = nn.Linear(10, 10)
        self.hidden3 = nn.Linear(10, 5)
        self.output = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.sigmoid(self.hidden1(x))
        x = torch.sigmoid(self.hidden2(x))
        x = torch.sigmoid(self.hidden3(x))
        x = self.output(x)
        return x

if __name__ == "__main__":
    epoch = 0
    error_p = 10e10
    model_Q = ThreeHiddenLayerNet().double()
    criterion = nn.MSELoss()
    optimizer_Q = optim.SGD(model_Q.parameters(), lr=0.01)
    while True:
        epoch += 1
        outputs = model_Q(train_input)
        loss = criterion(outputs, train_Q)

        optimizer_Q.zero_grad()
        loss.backward()
        optimizer_Q.step()

        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
            predicted_values = model_Q(test_input)
            error_n = criterion(predicted_values, test_Q)
            print(f'Test error: {error_n.item():.4f}')
            if error_n > error_p:
                torch.save(model_Q, 'model_Q.pth')
                break
            error_p = error_n
