import numpy as np
import time
import itertools
from os import cpu_count
from multiprocessing import Process, Queue
from .inte_region import region
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from mcinte.inte_region import argument

# global variables
global ndim, nnum, rst, a
global hull_ptrs, hull_points
scale = 20

aaaaa

# x[0] = cos[phi0], x[1] = sin[phi0]cos[phi1], ... ,
# x[ndim-1] = sin[phi0]...sin[phi_{ndim-2}]cos[phi_{ndim-1}]
# x[ndim]   = sin[phi0]...sin[phi_{ndim-2}]sin[phi_{ndim-1}]
# phi0~phi_{ndim-2} in [0,pi], phi_{ndim-1} in [0, 2pi]
def uni_spherical(phi, ndim):
    if ndim != 2:
        if len(phi) != ndim-1:
            print("phi and ndim mismatch") 
            return -1.  
        x = [np.cos(phi[0])]
        for i in range(1, ndim-1):
            x.append(np.prod(np.sin(phi[0:i])) * np.cos(phi[i]))
        x.append(np.prod(np.sin(phi[0:i])) * np.sin(phi[i]))
        return np.array([x])
    elif ndim == 2:
        x = [np.cos(phi), np.sin(phi)]
        return np.array([x])
    else:
        print("low dimension")
        return -1.

def sample_spherical(npoints, ndim):
    vec = np.random.randn(npoints, ndim)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

# instantiate hull
def find_and_construct_hull(center, test_direc, rmax):
    global hull_ptrs, hull_points
    r0max = rmax
    rc = (rmax / scale) * 1e-12
    hull_ptrs = []
    for ray in test_direc:
        rmin = 0 ; rmax = r0max ; r = (rmax+rmin)/2
        while np.abs(rmax - rmin) > rc:
            if region(center + ray * r, a) < 0: # outside
                rmax = r
            else: # inside
                rmin = r
            r = (rmin + rmax)/2
        hull_ptrs = hull_ptrs + [center + ray * r]
    hull_points = hull_ptrs
    if not isinstance(hull_ptrs,Delaunay):
        hull_ptrs = Delaunay(hull_ptrs)


# count num of points inside polytope
que = Queue()
def count_in_polytope(low, high):
    que.put ( 
             np.count_nonzero(
                 hull_ptrs.find_simplex( rst[low:high,:] )>=0 ) )


# return the volume descripted by inte_region
# r>0 size=r; r<0 size=-r*min_wid
def convex_vol(sampler, angle_num=0, nthread=None):

    global ndim, nnum, rst, a
    a = argument

    # preparation
    cor_t = int(np.ceil(sampler.get_autocorr_time().max()))
    rst = sampler.get_chain(flat=True, discard=3*cor_t)#, thin=int(0.1*cor_t))
    ndim = rst.shape[1]
    nnum = rst.shape[0]
    np.random.seed(time.time_ns() % 2**10)
    if ndim < 2:
        print("too low dimension!")
        return -1
    dim_center = [] ; dim_width = 0
    for d in range(ndim):
        tmax = rst[:,d].max()
        tmin = rst[:,d].min()
        dim_width = np.max([dim_width, tmax - tmin])
        dim_center = dim_center + [(tmax+tmin)/2]
    rmax = scale * dim_width
    dim_center = np.array(dim_center)
    if region(dim_center, a) < 0:
        dim_center = rst[np.random.randint(nnum),:]

    # test direction
    print("generating angles...")
    if angle_num == 0:
        angle_num = 2**(ndim)
    theta = np.linspace(0, np.pi, angle_num+2) ; theta = theta[1:-1]
    phi = np.linspace(0, 2*np.pi, angle_num+1) ; phi = phi[:-1]
    if ndim == 2:
        angles = phi
    else:
        angles = []
        thete_iter = itertools.product(theta, repeat=ndim-2)
        for i in thete_iter:
            for j in phi:
                tmp = list(i) ; tmp.append(j)
                angles = angles + [tmp]
    test_direc = []
    print("generating directions...")
    for ang in angles:
        test_direc = test_direc + [uni_spherical(ang, ndim)]
    test_direc = np.concatenate(test_direc, axis=0)
    test_direc[np.abs(test_direc)<1e-10] = 0

    # find polytope boundary
    find_and_construct_hull(dim_center, test_direc, rmax)

    # divide points
    if nthread == None:
        nthread = cpu_count()
    nnum_batch = nnum//nthread
    p = [] ; cnt = 0
    print("multiprocessing...")
    for k in range(nthread-1):
        p = p + [Process( target=count_in_polytope,
                          args=(k*nnum_batch, (k+1)*nnum_batch) ) ]
        p[k].start()
    p = p + [Process( target=count_in_polytope,
                          args=((nthread-1)*nnum_batch, nnum) ) ]
    p[nthread-1].start()
    for k in range(nthread):
        p[k].join()
    while not que.empty():
        cnt += que.get()

    polytope_vol = ConvexHull(hull_points).volume

    vol = polytope_vol * nnum / cnt
    return vol
