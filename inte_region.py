# -*- coding: utf-8 -*-

import numpy as np

# --- integration region description --- #

"""
    points inside integration region
    return 1, otherwise return -1e150
"""

def inte_region(x,a): # a is extra arguments passed by N_ndim_nwalkers[0]

    if x[0] < 0 or x[1] < 0 or x[2] < 0:
        return -1.e150
    if x[0]**2 + x[1]**2 + x[2]**2 < a**2 and x[2]**2 > x[0]**2 + x[1]**2:
        return 1.
    else:
        return -1.e150

## --- end --- ##

