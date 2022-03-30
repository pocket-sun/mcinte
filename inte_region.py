# -*- coding: utf-8 -*-

import numpy as np

EXCLUDE = -1e150

# --- integration region description --- #

"""
    points inside integration region
    return 1, otherwise return EXCLUDE
    argument must be changed below, not in region
"""
a = 1.5 # hypersphere and round
#a = 2 # sphere sliced_round
#a = 1 # drill plane vol
argument = a

def region(x,a=argument): # a is extra arguments passed by N_ndim_nwalkers[0]
    """mathematical description of integral region

    modification of parameter a must be passed to arguments above
    """

#    if x[0] < 0 or x[1] < 0 or x[2] < 0:
#        return -1.e150
#    if x[0]**2 + x[1]**2 + x[2]**2 < a**2 and x[2]**2 > x[0]**2 + x[1]**2:
#        return 1.
#    else:
#        return EXCLUDE

    if x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2<=a**2:
        return 1.
    else:
        return EXCLUDE

#    if x[0]**2+x[1]**2+x[2]**2<=a**2:
#        return 1.
#    else:
#        return EXCLUDE

#    if x[0]**2+x[1]**2<=a**2:
#        return 1.
#    else:
#        return EXCLUDE

#    if np.sum(x**2) < a**2 and -a/2 < x[0] < a/2:
#        return 1.
#    else:
#        return EXCLUDE

#    if x[0] > 0 and x[1] > 0 and x[2] > 0 and \
#        np.sum(x) < a:
#        return 1.
#    else:
#        return EXCLUDE
    
    if np.sum(x**2) < a**2 and x[0]**2 + x[1]**2 < a*x[0]:
        return 1.
    else:
        return EXCLUDE

## --- end --- ##

