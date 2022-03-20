import numpy as np
import emcee
from os import cpu_count
from multiprocessing import Process, Queue

# para sum
que = Queue()
def para_sum(rst, low, high, fcn):
    sm = 0.
    for k in range(low,high):
        sm += fcn(rst[k,:])
    que.put(sm)
    return

# integration of fcn over inte_region, normalized by 1/V, V = get_vol
def get_inte(sampler, fcn, norm=False, nthread=None):
    """calculate integration of fcn under vol descripted by inte_region

    norm = True, 1/V normalization is used, otherwise nonorm = (True, arguments)
    """

    cor_t = int(np.ceil(sampler.get_autocorr_time().max()))
    rst = sampler.get_chain(flat=True, discard=3*cor_t)#, thin=int(0.1*cor_t))
    ndim = rst.shape[1]
    nnum = rst.shape[0]
    if ndim < 2:
        print("too low dimensional case is not welcome.")
        return -1
    
    inte = 0.
    if nthread == None:
        nthread = cpu_count()
    num_sec = nnum // nthread
    p = []
    for k in range(nthread):
        low = k * num_sec ; high = (k+1) * num_sec
        if k == nthread-1:
            high = nnum
        p = p + [Process(target=para_sum, args=(rst,low,high,fcn))]
        p[k].start()
    for k in range(nthread):
        p[k].join()
    while not que.empty():
        inte += que.get()
    
    if not norm:
        from mcinte.convex_vol import convex_vol
        from mcinte.inte_region import argument
        vol = convex_vol(sampler)
        return vol * inte/nnum
    return inte/nnum
