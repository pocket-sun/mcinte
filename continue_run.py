from . import inte_region as pos
import emcee
from multiprocessing import Pool

def continue_run(filename, N_ndim_nwalkers, continue_num , thread = None):
    # retrieve data stored in #filename
    reader = emcee.backends.HDFBackend(filename)
    N = N_ndim_nwalkers[0]
    ndim = N_ndim_nwalkers[1]
    nwalkers = N_ndim_nwalkers[2]
    # parallelization
    with Pool(thread) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, pos.inte_region,
                                        pool=pool, # parallelization
                                        args=[N], # inte_region(x, args=[...])
                                        backend=reader) # backup data on the fly
        sampler.run_mcmc(None,
                         continue_num, # (number) of steps nwalkers move on, s.t. nwalkers*(number) obtained
                         progress=True) # show progress bar
