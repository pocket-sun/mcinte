import mcinte as mc
import numpy as np
import time
import matplotlib.pyplot as plt




#filename = "drill.h5"
filename = "hypersphere.h5"
#filename = "sphere.h5"
#filename = "round.h5"

a = 1.5 # hypersphere and round
#a = 2 # sphere
#a = 1 # drill

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

#lst = sampler.get_chain(flat=True, discard=100)
#cnt = 0
#rad = 1.06
#for p in lst:
#    toggle = True
#    for d in range(ndim):
#        toggle = toggle and (-rad<p[d]<rad)
#    if toggle:
#        cnt += 1
#vol = (2*rad)**ndim
#print(lst.shape[0]*vol/cnt)


beg = time.time_ns()
res=mc.convex_vol(sampler, a)
#res=mc.ld_vol(sampler, a)
print(res)
print((time.time_ns()-beg)/1e9)

#from mpl_toolkits.mplot3d import axes3d
#phi = np.linspace(0, np.pi, 20)
#theta = np.linspace(0, 2 * np.pi, 40)
#x = a * np.outer(np.sin(theta), np.cos(phi))
#y = a * np.outer(np.sin(theta), np.sin(phi))
#z = a * np.outer(np.cos(theta), np.ones_like(phi))
#xi, yi, zi = sample_spherical(100)
#fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal'})
#ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1)
#ax.scatter(xi, yi, zi, s=100, c='r', zorder=10)
