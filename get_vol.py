import numpy as np
import emcee
import time
from os import cpu_count
from multiprocessing import Process, Queue
from itertools import product 
from . import inte_region as pos

# return sorted np.array (along first column)
#def sort_kpor(arr): # keep order
#    rst = np.zeros(arr.shape)
#    sort_index_of_first_col = np.argsort(arr,axis=0)[:,0]
#    row_new = 0
#    for row_old in sort_index_of_first_col:
#        rst[row_new, :] = arr[row_old, :]
#        row_new += 1
#    return rst # np.array is ptr not the whole array!

# return point list inside the square
cmp = lambda x: x[0]
def sort_and_find(lst, cor, r, ndim):
    for d in range(1, ndim-1):
        if lst.shape[0] == 0:
            return 0.
        tmp = np.array(sorted(lst, key = cmp))
        low_ind = np.searchsorted(tmp[:,0], cor[d]-r)
        high_ind = np.searchsorted(tmp[:,0], cor[d]+r)
        lst = np.delete(tmp[low_ind:high_ind,:],0,axis=1)
    if lst.shape[0] == 0:
        return 0.
    tmp = np.array(sorted(lst, key = cmp))
    low_ind = np.searchsorted(tmp[:,0], cor[ndim-1]-r)
    high_ind = np.searchsorted(tmp[:,0], cor[ndim-1]+r)
    return high_ind-low_ind


# is all points of the square inside inte_region?
def is_in(x, sign, a):
    toggle = True
    for i in sign:
        toggle = toggle and (pos.region(x + i, a) > 0)
    return toggle

# is two seed points too close?
def not_too_close(seed_set,cor0,direc,r,ndim):
    toggle = True
    cri_distance2 = 4*ndim*r**2
    for seed in seed_set:
        b_m_a = seed - cor0
        direc2 = np.dot(direc, direc)
        l = b_m_a - direc * np.dot(b_m_a,direc)/direc2
        toggle = toggle and (np.dot(l,l) > cri_distance2)
        if not toggle:
            return toggle
    return toggle
        

que = Queue()
# multiprocess
def fill_square(nnum,cnum,rst,a,r,ndim,sign,direc):
    cnt_list = np.array([0, 0]) # pcnt, sqrcnt
    np.random.seed(time.time_ns() % 2**18)
    trigger = False
    cor0 = rst[np.random.randint(nnum),:]
    cor = cor0
    seed_set = [cor0]
    while cnt_list[0] <= cnum:
        if is_in(cor+direc,sign,a):
            cor += direc
        elif trigger: 
            cor0 = rst[np.random.randint(nnum),:]
            while (not is_in(cor0,sign,a)) or (not not_too_close(seed_set,cor0,direc,r,ndim)):
                cor0 = rst[np.random.randint(nnum),:]
            cor = cor0
            seed_set.append(cor0)
        else: 
            cor = cor0 + direc # cover cor0 itselt
            direc *= -1
            trigger = True
        cnt_list[1] += 1
        tmp = rst;
        low_ind = np.searchsorted(rst[:,0], cor[0]-r)
        high_ind = np.searchsorted(rst[:,0], cor[0]+r)
        lst = np.delete(tmp[low_ind:high_ind,:],0,axis=1)
        cnt_list[0] += sort_and_find(lst, cor, r, ndim)
#        lower = np.searchsorted(rst_1st_col, cor[0]-r)
#        upper = np.searchsorted(rst_1st_col, cor[0]+r)
#        for i in range(lower, upper):
#            dcnt = 0
#            for d in range(1, ndim):
#                if rst[i,d] >= cor[d]-r and rst[i,d] <= cor[d]+r:
#                    dcnt += 1 # [1, ndim) => ndim-1 times
#            if dcnt == ndim-1:
#               cnt_list[0] += 1
    que.put(cnt_list)
    return

# random direction (not from me and it is not uniform in sphere)
def sample_spherical(ndim):
    vec = np.random.randn(ndim)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

# return the volume descripted by inte_region
def get_vol(sampler, N_ndim_nwalkers, rate = 0.1, r = -1, nthread = None):

    # preparation
    cor_t = int(np.ceil(sampler.get_autocorr_time().max()))
    rst = sampler.get_chain(flat=True, discard=3*cor_t)#, thin=int(0.1*cor_t))
    rst = np.array(sorted(rst, key = cmp))
    ndim = rst.shape[1]
    nnum = rst.shape[0]
    a = N_ndim_nwalkers[0]
    cnum = int(nnum * rate) # critical number
    if ndim < 2:
        print("too low dimensional case is not welcome.")
        return -1

    # square size
    if r <= 0:
        dim_width = np.empty(ndim)
        for d in range(ndim):
            dim_width[d] = rst[:,d].max() - rst[:,d].min()
        min_wid = dim_width.min()
        r = 0.03 * min_wid
    square_vol = (2*r)**ndim
    sign_raw = product([r,-r],repeat=ndim)
    sign = []
    for i in sign_raw:
        sign = sign + [np.array(i)]

    # fill with square
    if nthread == None:
        nthread = cpu_count()
    max_batch = 15
    cnum_batch = cnum//max_batch
    pcnt_sqrcnt = np.array([0, 0])
    np.random.seed(time.time_ns() % 2**16)
    while pcnt_sqrcnt[0] < cnum:
        p = []
        for k in range(nthread):
            direc = 2 * r * (ndim) ** (1/2) * sample_spherical(ndim)
            p = p + [Process(target=fill_square, args=(nnum,cnum_batch,rst,a,r,ndim,
                                                       sign,direc))]
            p[k].start()
        for k in range(nthread):
            p[k].join()
        while not que.empty():
            pcnt_sqrcnt += que.get()

    print("sqr_vol=%f,sqr_num=%d,cnt=%d" % (square_vol, pcnt_sqrcnt[1],pcnt_sqrcnt[0]))
    vol = square_vol * pcnt_sqrcnt[1] * nnum / pcnt_sqrcnt[0]
    return vol
    

        
            
            
        
        
        
        

        
