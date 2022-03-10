from . import inte_region as pos
import numpy as np
import emcee
from multiprocessing import Pool

def run(filename, N_ndim_nwalkers, p0, number, thread = None):
    # data fibrication
    N = N_ndim_nwalkers[0]
    ndim = N_ndim_nwalkers[1]
    nwalkers = N_ndim_nwalkers[2]

    # prepare for data saving
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    # parallelization
    with Pool(thread) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, pos.inte_region,
                                        pool=pool, # parallelization
                                        args=[N], # inte_region(x, args=[...])
                                        backend=backend) # backup data on the fly
        state = sampler.run_mcmc(p0, # initial values for x are p0[walker_number] for each walker
                                 number, # (number) of steps nwalkers move on, s.t. nwalkers*(number) obtained
                                 progress=True) # show progress
