import mcinte as mc
import numpy as np
import time
import matplotlib.pyplot as plt

filename = "hypersphere.h5"

a = 1.5

ndim = 5
nwalkers = 50

p0 = np.random.random([nwalkers,ndim]) - a/2


N_ndim_nwalkers = (a, ndim, nwalkers)

#mc.run(filename, N_ndim_nwalkers, p0, 10000)
#mc.continue_run(filename, N_ndim_nwalkers, 3)
sampler = mc.read(filename)
#mc.cor_diag(sampler, labels)
#mc.line_diag(sampler, labels)
#mc.hist_diag(sampler, labels[0], 0)

beg = time.time_ns()
res=mc.get_vol(sampler, N_ndim_nwalkers, 0.01, 0.05)
print(res)
print((time.time_ns()-beg)/1e9)
