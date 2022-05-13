## This document serves to review how to generate new potential points from old ones in 1-D.

import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import matplotlib.pyplot as plt



def generateV(s=1, V0=1, noise=1e-10, points=100):
    
    fig, ax = plt.subplots(figsize=(4,4))
    fig2, ax2 = plt.subplots(figsize=(12,6))
    phi = np.linspace(0,20,points)*s
    vcov = np.zeros((points,points))
    for i in range(points):
        for j in range(i+1):
            vcov[i,j] = V0**2*np.exp(-(phi[i]-phi[j])**2/(2*s**2))
            vcov[j,i] = vcov[i,j]
    half = np.ceil(points/2).astype(np.int64)
    vcov = vcov + np.identity(points)*noise
    
    L = sp.linalg.cholesky(vcov, lower=True)
    y = np.random.normal(loc=0, scale=1, size=points)
    V = L @ y
    
    ax.plot(phi[:half], V[:half], 'o--', c='darkgrey')
    ax2.plot(phi[:half], V[:half], 'o--', c='darkgrey')
    ax2.plot(phi[half:], V[half:], 'o--', c='orange')
    ax.set_xlabel(r"$\phi^{(0)}/s$", fontsize=20)
    ax.set_ylabel(r"$V/V_0$", fontsize=20)
    ax2.set_xlabel(r"$\phi^{(0)}/s$", fontsize=20)
    ax2.set_ylabel(r"$V/V_0$", fontsize=20)
    ax.grid()
    ax2.grid()
    
    return



def plotMuPlusSkew(end=2, points=100, figres=1):
    
    def mu(f1, x): return f1*np.exp(-x**2/2)
    def skew(x): return np.sqrt(1 - np.exp(-x**2))
    
    fig, ax = plt.subplots(1, 3, figsize=(30*figres,9*figres), sharex=True, sharey=True)
    
    x = np.linspace(-end,end,points)
    mus0 = mu(0, x)
    mus1 = mu(1, x)
    mus2 = mu(2, x)
    skews = skew(x)
    
    ax[0].plot(x, mus0, 'k', label=r'$\mu$')
    ax[0].fill_between(x, mus0+skews, mus0-skews, color='b', alpha=0.4, label=r'$|y|<1$')
    ax[0].fill_between(x, mus0+2*skews, mus0-2*skews, color='b', alpha=0.2, label=r'$|y|<2$')
    ax[0].set_xlabel(r'$f_2$', fontsize=30*figres)
    ax[0].set_ylabel(r'$V/V_{\star}$', fontsize=30*figres)
    ax[0].set_title(r'$f_1=0$', fontsize=20*figres)
    ax[0].grid()
    ax[0].legend(fontsize=25*figres)
    ax[0].tick_params(labelsize=15*figres)
    
    ax[1].plot(x, mus1, 'k', label=r'$\mu$')
    ax[1].fill_between(x, mus1+skews, mus1-skews, color='b', alpha=0.4, label=r'$|y|<1$')
    ax[1].fill_between(x, mus1+2*skews, mus1-2*skews, color='b', alpha=0.2, label=r'$|y|<2$')
    ax[1].set_xlabel(r'$f_2$', fontsize=30*figres)
    ax[1].set_ylabel(r'$V/V_{\star}$', fontsize=30*figres)
    ax[1].grid()
    ax[1].legend(fontsize=25*figres)
    ax[1].set_title(r'$f_1=1$', fontsize=20*figres)
    ax[1].tick_params(labelsize=15*figres)
    
    ax[2].plot(x, mus2, 'k', label=r'$\mu$')
    ax[2].fill_between(x, mus2+skews, mus2-skews, color='b', alpha=0.4, label=r'$|y|<1$')
    ax[2].fill_between(x, mus2+2*skews, mus2-2*skews, color='b', alpha=0.2, label=r'$|y|<2$')
    ax[2].set_xlabel(r'$f_2$', fontsize=30*figres)
    ax[2].set_ylabel(r'$V/V_{\star}$', fontsize=30*figres)
    ax[2].grid()
    ax[2].legend(fontsize=25*figres)
    ax[2].set_title(r'$f_1=2$', fontsize=20*figres)
    ax[2].tick_params(labelsize=15*figres)
    
    
    return



def plotNumNewEntries(Ns = np.arange(1,10), Ts = np.arange(2000)):
    
    fig, ax = plt.subplots(figsize = np.array((6,4))*0.8, dpi=300)
    [ax.plot(Ts, (N+1)**2*(2*Ts+1), c=(N/max(Ns),(1 - N/max(Ns)),0.2), label=rf"$N={N}$") for N in Ns]
    ax.legend()
    ax.set_xlabel("Time Step")
    ax.set_ylabel(r"# of New Entries in $\Gamma$")
    ax.grid()
    
    return