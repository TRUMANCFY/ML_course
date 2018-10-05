# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    n = np.shape(x)[0]
    print(n)
    res = np.zeros((n, degree+1))
    for i in range(n):
        for j in range(degree+1):
            res[i][j] = np.power(x[i], j)
    return res