# -*- coding: utf-8 -*-
"""
@Time   : 2024/3/24 19:18
@Author : Yichao Li
"""
from fit.fit_T import TwoHiddenLayerNet
from fit.fit_Q import ThreeHiddenLayerNet
from fit.TQ import T, Q
from tool import Aug_Lagrange_Function, armijo_backtracking
import numpy as np

T0 = 0.1

def obj_f(x):
    return x[0]*x[2]*Q([x[0],x[1],x[2]])

def c1(x):
    return x[0] + x[3]**2 - 4

def c2(x):
    return x[1] + x[4]**2 - 10

def c3(x):
    return x[2] + x[5]**2 - 7

def c4(x):
    return -x[0] + x[6]**2 + 1

def c5(x):
    return -x[1] + x[7]**2

def c6(x):
    return -x[2] + x[8]**2 +1

def c7(x):
    return T([x[0], x[1], x[2]]) - T0

eqc_list = [c1, c2, c3, c4, c5, c6, c7]

def ALF(x, p, nargout=0):
    lambd = p[0]
    mu = p[1]
    f = Aug_Lagrange_Function(x, obj_f, eqc_list, lambd, mu)

    if nargout == 0:
        return f
    else:
        n = len(x)
        e = np.sqrt(np.finfo(np.float64).eps)
        df = np.zeros(n)
        for k in range(n):
            y = x.copy()
            y[k] += e
            df[k] = ALF(y, p)
        g = (df - f) / e
        return f, g

param_this = np.array([1.,0.,1.,np.sqrt(3),np.sqrt(10),np.sqrt(6),0.,0.,0.]).astype(np.float64)
lambd = np.ones(7)
mu = 2
epoch = 0
for epoch in range(10):
    count = 0
    p = [lambd, mu]
    while True:
        count += 1
        if count == 1:
            loss_this, gradient_this = ALF(param_this, p, 2)
        else:
            loss_this = loss_next
            gradient_this = gradient_next
        search_direction = -gradient_this
        step, param_next, loss_next, gradient_next = armijo_backtracking(ALF, param_this, loss_this, gradient_this,
                                                                         search_direction, 1, p)
        param_this = param_next
        if abs(loss_this - loss_next) < 10e-6 or abs(np.linalg.norm(gradient_next)) < 0.01:
            # print(f'Current epoch {epoch+1}')
            # print(f'Loss {loss_next:.5f}')
            break
    if abs(np.linalg.norm(gradient_next)) < 0.01:
        break

    lambd = lambd - np.array([mu*func(param_next) for func in eqc_list])
    mu *= 2

print(f'For T0 = {T0:.2f}, solution:')
print(np.round(param_next[:3], 5))
print('Object funtion value: {}'.format(np.round(obj_f(param_next), 5)))



