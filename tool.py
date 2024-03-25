# -*- coding: utf-8 -*-
"""
@Time   : 2024/3/24 19:48
@Author : Yichao Li
"""
import numpy as np

def armijo_backtracking(function, x, loss, gradient, direction, step, data):
    alpha = step
    loss_this = loss
    gradient_this = gradient
    search_direction = direction
    gradient_projection_this = np.dot(search_direction.T, gradient_this)
    scale_gradient_projection_this = 0.01 * gradient_projection_this
    for _ in range(10):
        x_next = x + alpha * search_direction
        loss_next = function(x_next, data)
        if loss_next > loss_this + alpha * scale_gradient_projection_this:
            alpha *= 0.01
        else:
            loss_next, gradient_next = function(x_next, data, 2)
            break
    return alpha, x_next, loss_next, gradient_next

def Aug_Lagrange_Function(x, f, eqc_list, lambd, mu):
    lagrange_term = sum(eqc_list[i](x) * lambd[i] for i in range(len(eqc_list)))
    penality_term = sum(func(x) ** 2 for func in eqc_list)
    result = np.sqrt(np.square(f(x))) - lagrange_term + mu/2.0 * penality_term
    return result