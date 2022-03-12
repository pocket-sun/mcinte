import numpy as np
import emcee
import time
from os import cpu_count
from multiprocessing import Process, Queue
from itertools import product 
from . import inte_region as pos

# global variables
global ndim, nnum, test_direc, ntdirec, rst, rmax # in get_vol
global existed_balls = [] # store tuple (cor, r), cor is ptr

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
    #print("b",x)
    return toggle

# 
def expand_ball(cor):
    rupper = rmax
    rlower = 0
    # if r too large:
    #   rupper = r
    # else
    #   rlower = r
    # r = (rupper+rlower)/2
    # till rupper-rlower<critical_diff
    
        

que = Queue()
# multiprocess
def fill_ball(rst, rmax, seed = None):

    global existed_balls

    if seed == None:
        seed = rst[np.random.randint(nnum),:]
    expand_ball(seed, rmax)
    
    # create a ball with [seed], expand utill it touches the boundary

    # get a random [cor] inside rst, [rst] is points set excluding existed balls

    # construct a ball centered at cor, with given small radius [rmin]

    # expand the ball with bisec method

    # if the ball touches one of [existed_balls], push away the ball along radical direction

    # the ball must touches only one of [existed_balls], shrink the radius if not

    # if the ball touches the boundary and does not touch any one of [existed_balls], 
    # push away the ball along radical direction

    # if the ball touches both boundary and [existed_balls], shrink

    # continue untill [radius_diff] < [critical_diff]

    # continue untill required points [cnum] are obtained
    cnt_list = np.array([0, 0]) # pcnt, sqrcnt
    trigger = False
    cor0 = rst[np.random.randint(nnum),:]
    cor = cor0.copy()
    seed_set = [cor0]
    while cnt_list[0] <= cnum:
        cor += direc
        if not is_in(cor,sign,a):
            if trigger: 
                cor0 = rst[np.random.randint(nnum),:]
                while (not is_in(cor0,sign,a)) or (not not_too_close(seed_set,cor0,direc,r,ndim)):
                    cor0 = rst[np.random.randint(nnum),:]
                cor = cor0.copy()
                seed_set.append(cor0)
                trigger = False
                continue
            else:  
                cor = cor0 + direc # cover cor0 itselt
                direc *= -1
                trigger = True
                continue
        #print("fff===")
        cnt_list[1] += 1
        low_ind = np.searchsorted(rst[:,0], cor[0]-r)
        high_ind = np.searchsorted(rst[:,0], cor[0]+r)
        lst = np.delete(rst[low_ind:high_ind,:],0,axis=1)
        cnt_list[0] += sort_and_find(lst, cor, r, ndim)
    print((cnt_list[0], cnum, len(seed_set)))
    que.put(cnt_list)
    return

# random direction (not from me and it is not uniform in sphere)
def sample_spherical(npoints, ndim):
    vec = np.random.randn(npoints, ndim)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

# return the volume descripted by inte_region
def get_vol(sampler, a, rate=0.1, max_batch=20, nthread=None):

    # global def here
    global ndim, nnum, test_direc, ntdirec, rst, rmax

    # preparation
    cor_t = int(np.ceil(sampler.get_autocorr_time().max()))
    rst = sampler.get_chain(flat=True, discard=3*cor_t)#, thin=int(0.1*cor_t))
    ndim = rst.shape[1]
    nnum = rst.shape[0]
    cnum = int(nnum * rate) # critical number
    np.random.seed(time.time_ns() % 2**10)
    if ndim < 2:
        print("too low dimention")
        return -1
    for d in range(ndim):
        dim_width[d] = rst[:,d].max() - rst[:,d].min()
    rmax = 10*dim_width.max()

    # test direction
    test_direc = sample_spherical(2**(ndim+1), ndim) # [npoints, ndim]
    ntdirec = test_direc.shape[0]

    # fill with ball
    if nthread == None:
        nthread = cpu_count()
    cnum_batch = cnum//max_batch
    pcnt_sqrcnt = np.array([0, 0])
    np.random.seed(time.time_ns() % 2**10)
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
#        print("sqr_vol=%E, nnum=%d" % (square_vol,nnum))
#        print("pcnt=%-8d, sqrcnt=%-8d" % (pcnt_sqrcnt[0],pcnt_sqrcnt[1]))
#        print("density=", pcnt_sqrcnt[1]/pcnt_sqrcnt[0])

#    print("sqr_vol=%E,sqr_num=%d,cnt=%d" % (square_vol, pcnt_sqrcnt[1],pcnt_sqrcnt[0]))
    vol = square_vol * pcnt_sqrcnt[1] * nnum / pcnt_sqrcnt[0]
    return vol
