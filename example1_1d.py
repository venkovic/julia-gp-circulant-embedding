import numpy as np
import pylab as pl

pl.rcParams['text.usetex'] = True
params = {'text.latex.preamble' : [r'\usepackage{amssymb}',
                                   r'\usepackage{amsmath}']}
pl.rcParams.update(params)
pl.rcParams['axes.labelsize'] = 20
pl.rcParams['axes.titlesize'] = 20
pl.rcParams['legend.fontsize'] = 20
pl.rcParams['xtick.labelsize'] = 20
pl.rcParams['ytick.labelsize'] = 20
pl.rcParams['legend.numpoints'] = 1

n = 1000
nsmp = 5

Q = np.zeros((n, nsmp))
for ismp in range(nsmp):
  Q[:, ismp] = np.load('data/real%d.npy' % (ismp + 1))

xvals = np.linspace(0, 1, n)

fig, ax = pl.subplots()
for ismp in range(nsmp):
  ax.plot(xvals, Q[:, ismp], lw=.2)
pl.xlabel('x')
pl.show()

evals = np.load('data/evals.npy')
mrksize = .5
fig, ax = pl.subplots()
for ismp in range(nsmp):
  ax.semilogy(evals, '.', lw=0, color='k', markersize=mrksize)
pl.xlabel(r'$k$')
pl.ylabel(r'$\lambda_k$')
pl.title(r'$\mathrm{Eigvals\ of\ circulant\ embedding}\; \mathbf{C}$')
ax.grid(linestyle="-.")
pl.show()
