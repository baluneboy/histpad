#!/usr/bin/env python

import sys
import numpy as np

# Return 2d numpy array of float32's read from input file.
def padread(filename, columns=4, out_dtype=np.float32):
    """Return 2d numpy array of float32's read from input file.
    
    >>> pad_file = '/data/pad/year2016/month05/day09/sams2_accel_121f04/2016_05_09_10_00_00.007-2016_05_09_10_10_00.026.121f04'
    >>> m = padread(pad_file)
    >>> print m
    [[  0.00000000e+00  -1.71904601e-02   4.74499539e-02  -2.38601220e-04]
     [  2.00000009e-03   2.60950569e-02  -3.70590352e-02   4.20072023e-03]
     [  4.00000019e-03   2.15624515e-02  -4.94106971e-02   2.37301015e-03]
     ..., 
     [  6.00015381e+02   1.92782432e-02  -4.79847938e-02   3.17850709e-03]
     [  6.00017334e+02  -2.36788839e-02   3.52317020e-02  -9.14235832e-04]
     [  6.00019348e+02  -2.67130807e-02   5.85177876e-02  -7.78640970e-04]]
    """
    with open(filename, "rb") as f: 
        A = np.fromfile(f, dtype=np.float32) # accel file: 32-bit float "singles"
    B = np.reshape(A, (-1, columns))
    if B.dtype == out_dtype:
        return B
    return B.astype(out_dtype)

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)