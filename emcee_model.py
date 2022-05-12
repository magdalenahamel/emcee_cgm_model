import os
import Sample as sample
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp

from scipy.spatial.distance import pdist, cdist
from scipy.stats import kstwobign, pearsonr
from scipy.stats import genextreme

import emcee
from astropy import constants as const
from numpy import random
import math as m

from scipy.optimize import curve_fit
from matplotlib.ticker import ScalarFormatter
from scipy.stats import chisquare
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import scipy.optimize as optimization
import matplotlib as mpl
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import SkyCoord
import corner 
import telepot
import matplotlib
matplotlib.use('Agg')

bot = telepot.Bot('5106282512:AAFwfJ144PNtf9LwOP_o7Qmc6qrLNH8qEM8')
bot.sendMessage(2079147193, 'Empezó codico MCMC')

def prob_hit_log_lin(r, r_vir, a, b, por_r_vir = 0.5):
    r_t = r/r_vir
    return(np.exp(a)*(np.exp(-b*r_t)))

churchill_iso = pd.read_csv('Churchill_iso_full.txt', error_bad_lines=False, delim_whitespace=True)
W_r_churchill_iso = churchill_iso['Wr'].to_numpy()
D_R_vir_churchill_iso = churchill_iso['etav'].to_numpy()
e_Wr = churchill_iso['eWr'].to_numpy()

con_upper = (e_Wr == -1.)
W_r_churchill_upper = W_r_churchill_iso[con_upper]
D_R_vir_churchill_upper = D_R_vir_churchill_iso[con_upper]
e_Wr_upper = e_Wr[con_upper]

W_r_churchill_no_upper = W_r_churchill_iso[~con_upper]
D_R_vir_churchill_no_upper = D_R_vir_churchill_iso[~con_upper]
e_Wr_no_upper = e_Wr[~con_upper]

all_d = np.concatenate((D_R_vir_churchill_no_upper, D_R_vir_churchill_upper))

#####for runing 4 params #####
def EW_D_model(params):
    bs = params[0] # characteristic radius of the exponential function (it is accually a porcentage of Rvir) in log scale to make the range more homogeneous in lin scale
    csize = params[1] #poner en escala mas separada
    hs = params[2] #bajar un poco para que no sea un  1,10,20
    hvt = params[3] #bajar maximo a 100
    hv = 10**(hvt)
    zabs = 0.656
    lam0 = 2796.35

    vel_min = -1500
    vel_max = 1500
    lam_min = ((vel_min/const.c.to('km/s').value)+1)*(lam0*(1+zabs))
    lam_max = ((vel_max/const.c.to('km/s').value)+1)*(lam0*(1+zabs))

    w_spectral = 0.03

    wave = np.arange(lam_min,lam_max+w_spectral, w_spectral)
    vels_wave = (const.c.to('km/s').value * ((wave/ (lam0 * (1 + zabs))) - 1))

### run the model in the parameter grid

    results_Wr = []
    results_D = []
    results_R_vir = []
    results_specs = []
    results_tpcf_minor = []
    results_tpcf_major = []

    exp_fac = sample.Sample(prob_hit_log_lin,200,sample_size=200, csize=csize, h=hs, hv=hv)
    exp_results_1 = exp_fac.Nielsen_sample(np.log(100),bs,0.2)

    E_W_r = exp_results_1[8]
    D = exp_results_1[3]
    R_vir = exp_results_1[7]

    X1 = D/R_vir
    Y1 = E_W_r
    condY1 = Y1 > 0.03
    X1 = X1[condY1]
    Y1 = Y1[condY1]

    return(X1,Y1)



#define likelihood function
### 2D KS test ###

def ks2d2s(x1, y1, x2, y2, nboot=None, extra=False):
    '''Two-dimensional Kolmogorov-Smirnov test on two samples.
    Parameters
    ----------
    x1, y1 : ndarray, shape (n1, )
        Data of sample 1.
    x2, y2 : ndarray, shape (n2, )
        Data of sample 2. Size of two samples can be different.
    extra: bool, optional
        If True, KS statistic is also returned. Default is False.
    Returns
    -------
    p : float
        Two-tailed p-value.
    D : float, optional
        KS statistic. Returned if keyword `extra` is True.
    Notes
    -----
    This is the two-sided K-S test. Small p-values means that the two samples are significantly different. Note that the p-value is only an approximation as the analytic distribution is unkonwn. The approximation is accurate enough when N > ~20 and p-value < ~0.20 or so. When p-value > 0.20, the value may not be accurate, but it certainly implies that the two samples are not significantly different. (cf. Press 2007)
    References
    ----------
    Peacock, J.A. 1983, Two-Dimensional Goodness-of-Fit Testing in Astronomy, Monthly Notices of the Royal Astronomical Society, vol. 202, pp. 615-627
    Fasano, G. and Franceschini, A. 1987, A Multidimensional Version of the Kolmogorov-Smirnov Test, Monthly Notices of the Royal Astronomical Society, vol. 225, pp. 155-170
    Press, W.H. et al. 2007, Numerical Recipes, section 14.8
    '''
    assert (len(x1) == len(y1)) and (len(x2) == len(y2))
    n1, n2 = len(x1), len(x2)
    D = avgmaxdist(x1, y1, x2, y2)

    if nboot is None:
        sqen = np.sqrt(n1 * n2 / (n1 + n2))
        r1 = pearsonr(x1, y1)[0]
        r2 = pearsonr(x2, y2)[0]
        r = np.sqrt(1 - 0.5 * (r1**2 + r2**2))
        d = D * sqen / (1 + r * (0.25 - 0.75 / sqen))
        p = kstwobign.sf(d)
    else:
        n = n1 + n2
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])
        d = np.empty(nboot, 'f')
        for i in range(nboot):
            idx = random.choice(n, n, replace=True)
            ix1, ix2 = idx[:n1], idx[n1:]
            #ix1 = random.choice(n, n1, replace=True)
            #ix2 = random.choice(n, n2, replace=True)
            d[i] = avgmaxdist(x[ix1], y[ix1], x[ix2], y[ix2])
        p = np.sum(d > D).astype('f') / nboot
    if extra:
        return p, D
    else:
        return p


def avgmaxdist(x1, y1, x2, y2):
    D1 = maxdist(x1, y1, x2, y2)
    D2 = maxdist(x2, y2, x1, y1)
    return (D1 + D2) / 2


def maxdist(x1, y1, x2, y2):
    n1 = len(x1)
    D1 = np.empty((n1, 4))
    for i in range(n1):
        a1, b1, c1, d1 = quadct(x1[i], y1[i], x1, y1)
        a2, b2, c2, d2 = quadct(x1[i], y1[i], x2, y2)
        D1[i] = [a1 - a2, b1 - b2, c1 - c2, d1 - d2]

    # re-assign the point to maximize difference,
    # the discrepancy is significant for N < ~50
    D1[:, 0] -= 1 / n1

    dmin, dmax = -D1.min(), D1.max() + 1 / n1
    return max(dmin, dmax)


def quadct(x, y, xx, yy):
    n = len(xx)
    ix1, ix2 = xx <= x, yy <= y
    a = np.sum(ix1 & ix2) / n
    b = np.sum(ix1 & ~ix2) / n
    c = np.sum(~ix1 & ix2) / n
    d = 1 - a - b - c
    return a, b, c, d


def estat2d(x1, y1, x2, y2, **kwds):
    return estat(np.c_[x1, y1], np.c_[x2, y2], **kwds)


def estat(x, y, nboot=1000, replace=False, method='log', fitting=False):
    '''
    Energy distance statistics test.
    Reference
    ---------
    Aslan, B, Zech, G (2005) Statistical energy as a tool for binning-free
      multivariate goodness-of-fit tests, two-sample comparison and unfolding.
      Nuc Instr and Meth in Phys Res A 537: 626-636
    Szekely, G, Rizzo, M (2014) Energy statistics: A class of statistics
      based on distances. J Stat Planning & Infer 143: 1249-1272
    Brian Lau, multdist, https://github.com/brian-lau/multdist
    '''
    n, N = len(x), len(x) + len(y)
    stack = np.vstack([x, y])
    stack = (stack - stack.mean(0)) / stack.std(0)
    if replace:
        rand = lambda x: random.randint(x, size=x)
    else:
        rand = random.permutation

    en = energy(stack[:n], stack[n:], method)
    en_boot = np.zeros(nboot, 'f')
    for i in range(nboot):
        idx = rand(N)
        en_boot[i] = energy(stack[idx[:n]], stack[idx[n:]], method)

    if fitting:
        param = genextreme.fit(en_boot)
        p = genextreme.sf(en, *param)
        return p, en, param
    else:
        p = (en_boot >= en).sum() / nboot
        return p, en, en_boot


def energy(x, y, method='log'):
    dx, dy, dxy = pdist(x), pdist(y), cdist(x, y)
    n, m = len(x), len(y)
    if method == 'log':
        dx, dy, dxy = np.log(dx), np.log(dy), np.log(dxy)
    elif method == 'gaussian':
        raise NotImplementedError
    elif method == 'linear':
        pass
    else:
        raise ValueError
    z = dxy.sum() / (n * m) - dx.sum() / n**2 - dy.sum() / m**2
    # z = ((n*m)/(n+m)) * z # ref. SR
    return z



def boot_sample(params):
    model_D_R_vir = params[0]
    model_Wr = params[1]
    no_upper_sample = random.normal(loc=W_r_churchill_no_upper, scale=e_Wr_no_upper, size=None)
    upper_sample = random.uniform(low=0.0, high=W_r_churchill_upper, size=None)
    all_sample = np.concatenate((no_upper_sample, upper_sample))
    p = ks2d2s(all_d,all_sample,model_D_R_vir, model_Wr)
    return(p)

xdata = D_R_vir_churchill_iso  #array with (x,y) data coordinates (degrees)
ydata = W_r_churchill_iso

###### for running for parameters ############
paramnames = ['bs', 'cs','h', 'hv'] # Define the labels for each parameter (make sure they are in the same order as parammins/parammaxs)
parammins =  [0.5, 0.01,1,-4] #Define the minimum values for each parameter
parammaxs = [10, 10, 50, 5] #Define the maximum values for each parameter


#Define the properties of the MCMC sampler/modelling
ndim = len(paramnames) #Number of model parameters
nwalkers = 200 # Number of walkers
nsteps = 8000 #Number of steps each walker takes
#Define a burn-in; i.e. the first nburn steps to ignore
nburn=1


filename = "try_19.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

#############################
#Define the important functions

def tophatPrior(x, min, max):
    '''
    Function for probability of a tophat probability distribution function
    Parameters
    ----------
    x (float) - The data which to evaluate
    min (float) - Minimum value (inclusive) in which the tophat PDF starts
    max (float) - Maximum value (inclusive) in which the tophat PDF ends

    Returns
    -------
    probability (float) that value x is in the tophat PDF

    '''

    #If value within min/max values, return 1.0 (i.e. within tophat)
    if (x>=min)*(x<=max):
        return 1.0
    #Otherwise, it is not within the tophat PDF, return 0.0
    else:
        return 0


def totalprior(params, parammins, parammaxs):
    #Function that creates a total prior for a list of parameters with a flat/top-hat distribution
    totprior  =  1.0 #total prior to return
    #Loop through each parameter
    for ii, param in enumerate(params):
        totprior *= tophatPrior(param, parammins[ii], parammaxs[ii])
    #print('cccccc', params, totprior)
    return totprior


#
def loglikelihood(params):

    totprior  =  1.0
    for ii, param in enumerate(params):
        totprior *= tophatPrior(param, parammins[ii], parammaxs[ii])
        if totprior == 0:
            return(0)
    D_W = EW_D_model(params)
    model_D_R_vir = D_W[0]
    model_Wr = D_W[1]
    if len(model_D_R_vir)< 10:
        return(0)
    #print('W,D', model_Wr,model_D_R_vir )
    no_upper_sample = random.normal(loc=W_r_churchill_no_upper, scale=e_Wr_no_upper, size=None)
    upper_sample = random.uniform(low=0.0, high=W_r_churchill_upper, size=None)
    all_sample = np.concatenate((no_upper_sample, upper_sample))
    ps = []
    for i in range(100):
        pi = boot_sample(D_W)
        ps.append(pi)

    p = np.mean(ps)
    #print('like', np.log(p))
    return(np.log(p))


def logPosterior(params):
    #The  function MCMC evaluates; Bayes theorem in log space
    #p(model params | data) * p(data)
  
    return loglikelihood(params) + np.log(totalprior(params, parammins, parammaxs))


######################
#Setup the MCMC sampler and run it!

from multiprocessing import Pool



#Define the starting position of each walker, for each parameter
init_guess = np.zeros((nwalkers,ndim))
for ii, pmin in enumerate(parammins):
    init_guess[:,ii] = np.random.uniform(pmin, parammaxs[ii], size=nwalkers)



#Run the mcmc sampler!
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logPosterior, backend=backend,pool=pool)
    sampler.run_mcmc(init_guess, nsteps, progress=True)



##############################
#Playing with the output of the sampler

#Chains is the cube of data of dimensions [walker #, step #, parameter #]
chains = sampler.chain

#Distribution for parameter pp
#1d array of all the values of parameter pp (i.e. for a histogram)
#pdata = chains[:,nburn:,pp].flatten()

#Chain for walker ww, for param pp
#wchain = chains[ww, :, pp]

fig, axs = plt.subplots(ndim,1)
for pp in range(len(parammins)):
    for ww in range(nwalkers):
        axs[pp].plot(np.arange(0, nsteps, 1.0), chains[ww, :, pp], rasterized=True)

fig.savefig('mcmc_chains_19.pdf')


#Make a corner plot (how each parameter scales with another)
#Configure the data for the corner package to understand, removing the burn-in values
data = chains[:, nburn:, :]

#Make the corner plot
fig1= corner.corner(data.reshape(data.shape[0]*data.shape[1], data.shape[2]), labels=paramnames)
fig1.savefig('mcmc_corner_19.pdf')
bot.sendMessage(2079147193, 'Ternimo codico MCMC')

