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
from astropy.convolution import convolve, Gaussian1DKernel
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
bot.sendMessage(2079147193, 'Empezó codico MCMC TPCF')

zabs = 0.656
lam0 = 2796.35

vel_min = -1500
vel_max = 1500
lam_min = ((vel_min/const.c.to('km/s').value)+1)*(lam0*(1+zabs))
lam_max = ((vel_max/const.c.to('km/s').value)+1)*(lam0*(1+zabs))

w_spectral = 0.03
wave = np.arange(lam_min,lam_max+w_spectral, w_spectral)
vels_wave = (const.c.to('km/s').value * ((wave/ (lam0 * (1 + zabs))) - 1))

def filtrogauss(R, spec_res, lam_0, flux):
    del_lam = lam_0/R
    del_lam_pix = del_lam/spec_res
    gauss_kernel = (Gaussian1DKernel(del_lam_pix))
    gausflux = convolve(flux, gauss_kernel)
    return(gausflux)

def prob_hit_log_lin(r, r_vir, a, b, por_r_vir = 0.5):
    r_t = r/r_vir
    return(np.exp(a)*(np.exp(-b*r_t)))

minor_tpcf = pd.read_csv('2minor.txt', delimiter='     ', engine='python')
major_tpcf = pd.read_csv('2major.txt', delimiter='     ', engine='python')

face_tpcf = pd.read_csv('2face.txt', delimiter='     ', engine='python')
edge_tpcf = pd.read_csv('2edge.txt', delimiter='     ', engine='python')

minor_vel = minor_tpcf['vel'].to_numpy()
minor_bins = minor_vel - 5
minor_bins = np.append(minor_bins, minor_bins[-1] +10)
minor_tpcf_val = minor_tpcf['TPCF'].to_numpy()
minor_error = np.abs(minor_tpcf['minus_error'].to_numpy() - minor_tpcf['plus_error'].to_numpy())

major_vel = major_tpcf['vel'].to_numpy()
major_bins = major_vel - 5
major_bins = np.append(major_bins, major_bins[-1] +10)
major_tpcf_val = major_tpcf['TPCF'].to_numpy()
major_error = np.abs(major_tpcf['minus_error'].to_numpy() - major_tpcf['plus_error'].to_numpy())

ydata = major_tpcf_val
sigma = major_error

face_vel = face_tpcf['vel'].to_numpy()
face_bins = face_vel - 5
face_bins = np.append(face_bins, face_bins[-1] +10)
face_tpcf_val = face_tpcf['TPCF'].to_numpy()
face_error = np.abs(face_tpcf['minus_error'].to_numpy() - face_tpcf['plus_error'].to_numpy())

edge_vel = edge_tpcf['vel'].to_numpy()
edge_bins = edge_vel - 5
edge_bins = np.append(edge_bins, edge_bins[-1] +10)
edge_tpcf_val = edge_tpcf['TPCF'].to_numpy()
edge_error = np.abs(edge_tpcf['minus_error'].to_numpy() - edge_tpcf['plus_error'].to_numpy())

#####for runing 4 params #####
from itertools import combinations

def TPCF_aux(params):
    speci_empty_t = params[0]
    pos_alpha = params[1]
    print('TPCF', pos_alpha)
    #cond = np.asarray(nr_clouds) == 0
    #if len(speci_empty_t) == 0:
        #return(np.zero(len(major_vel)))

    if len(speci_empty_t)==0 and pos_alpha =='minor':
            return(np.zeros(len(minor_vel)))
    elif len(speci_empty_t)==0 and pos_alpha =='major':
            return(np.zeros(len(major_vel)))
    gauss_specs = []
    abs_specs = []
    vels_abs = []
    #speci_empty_t = np.asarray(speci_empty)[~cond]
    print('how many specs', len(speci_empty_t))

    for m in range(len(speci_empty_t)):

        gauss_specj = filtrogauss(45000,0.03,2796.35,speci_empty_t[m])
        gauss_specs.append(gauss_specj)
        zabs=0.656

        cond_abs1 = gauss_specj < 0.98
        cond_abs2 = np.abs(vels_wave) < 800
        abs_gauss_spec_major = vels_wave[cond_abs1 & cond_abs2]
        abs_specs.append(abs_gauss_spec_major)

    # Convert input list to a numpy array
    abs_specs_f = np.concatenate(np.asarray(abs_specs))
   # print('start tpcf')
    comb = combinations(abs_specs_f, 2)
    '''with concurrent.futures.ProcessPoolExecutor() as executor:
        result = [executor.submit(absdif, co) for co in comb]
        print('finish tpcf')
        # bla = [abs(a -b) for a, b in combinations(abs_specs_f, 2)]
        if pos_alpha == 'minor':
           bla2 = np.histogram(result,bins=minor_vel)
        elif pos_alpha == 'major':
           bla2 = np.histogram(result,bins=major_vel)
        bla_t = bla2[0]/len(result)
        return(bla_t)'''
    results = [absdif(co, pos_alpha) for co in comb]
    if pos_alpha == 'minor':
       bla2 = np.histogram(results,bins=minor_bins)
    elif pos_alpha == 'major':
       bla2 = np.histogram(results,bins=major_bins)

    if pos_alpha == 'face':
       bla2 = np.histogram(results,bins=minor_bins)
    elif pos_alpha == 'edge':
       bla2 = np.histogram(results,bins=major_bins)
    bla_t = bla2[0]/len(results)
    print(' end TPCF', pos_alpha)
    return(bla_t)

def absdif(bla, bla2):
    #print('absdif',bla, bla2)
    a = bla[0]
    b = bla[1]
    return(abs(a -b))

def TPCF(params):
    bs = params[0] # characteristic radius of the exponential function (it is accually a porcentage of Rvir) in log scale to make the range more homogeneous in lin scale
    csize = params[1] #poner en escala mas separada
    hs = params[2] #bajar un poco para que no sea un  1,10,20
    hv = params[3] #bajar maximo a 100

    zabs = 0.656
    lam0 = 2796.35

    vel_min = -1500
    vel_max = 1500
    lam_min = ((vel_min/const.c.to('km/s').value)+1)*(lam0*(1+zabs))
    lam_max = ((vel_max/const.c.to('km/s').value)+1)*(lam0*(1+zabs))

    w_spectral = 0.03

    wave = np.arange(lam_min,lam_max+w_spectral, w_spectral)
    vels_wave = (const.c.to('km/s').value * ((wave/ (lam0 * (1 + zabs))) - 1))

    results_tpcf_minor_major = []
    results_tpcf_face_edge = []

    exp_fill_fac = sample.Sample(prob_hit_log_lin,200,sample_size=200, csize=csize, h=hs, hv=hv)
    e3_a_1 = exp_fill_fac.Nielsen_sample(np.log(100),bs,0.2)
    cond_spec = e3_a_1[0] == 0
    spec_abs = e3_a_1[1][~cond_spec]
    alphas_abs = e3_a_1[2][~cond_spec]
    inclis_abs = e3_a_1[6][~cond_spec]
    cond_major = alphas_abs < 45
    #cond_minor = alphas_abs > 45

    #cond_face = inclis_abs < 57
    #cond_edge = inclis_abs > 57

    #spec_minor = spec_abs[cond_minor]
    specs_major = spec_abs[cond_major]

    #spec_face = spec_abs[cond_face]
    #spec_edge = spec_abs[cond_edge]

    specs_tot = (specs_major,'major')
    #specs_tot_i = [(spec_face,'face'), (spec_edge, 'edge')]


    print('empieza TPCF')
    results = TPCF_aux(specs_tot)
    #results = map(TPCF, specs_major)
    #list_res = list(results)

    #results_tpcf_minor_major.append(list_res[0])
    #results_tpcf_minor_major.append(list_res[1])

    #results_i = map(TPCF, specs_tot_i)
    #list_res_i = list(results_i)

    #results_tpcf_face_edge.append(list_res_i[0])
    #results_tpcf_face_edge.append(list_res_i[1])



    #results_nr_clouds.append(e3_a_1[0])
    #results_specs.append(e3_a_1[1])
    #results_alphas.append(e3_a_1[2])
    #results_D.append(e3_a_1[3])
    #results_vels.append(e3_a_1[4])
    #results_b.append(e3_a_1[5])
    #results_inclis.append(e3_a_1[6])
    #results_R_vir.append(e3_a_1[7])
    #results_Wr.append(e3_a_1[8])

    return(results)


#xdata = D_R_vir_churchill_iso  #array with (x,y) data coordinates (degrees)
#ydata = W_r_churchill_iso
#sigma = out_l[:,5][~np.isnan(out_l[:,4])]
#
#
#xdata = None #Define the x-value array of your data
#ydata = None #Define the y-value array of your data (same shape as xdata)
#sigma = None #Define the error in the ydata (float or same shape as ydata)

###### for running for parameters ############
paramnames = ['bs', 'cs','h', 'hv'] # Define the labels for each parameter (make sure they are in the same order as parammins/parammaxs)
parammins =  [0.01, 0.01,1,0.01] #Define the minimum values for each parameter
parammaxs = [10, 10, 50, 100] #Define the maximum values for each parameter

####### for running two parameters ###########
#paramnames = ['bs', 'cs'] # Define the labels for each parameter (make sure they are in the same order as parammins/parammaxs)
#parammins =  [0.01, 0.01] #Define the minimum values for each parameter
#parammaxs = [10, 10] #Define the maximum values for each parameter


#Define the properties of the MCMC sampler/modelling
ndim = len(paramnames) #Number of model parameters
nwalkers = 5 # Number of walkers
nsteps = 700 #Number of steps each walker takes
#Define a burn-in; i.e. the first nburn steps to ignore
nburn=20


filename = "try_4_TPCF.h5"
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
    print('cccccc', params, totprior)
    return totprior

from scipy.stats.distributions import chi2

#
def loglikelihood(params):

    totprior  =  1.0
    for ii, param in enumerate(params):
        totprior *= tophatPrior(param, parammins[ii], parammaxs[ii])
        if totprior == 0:
            return(0)
    y = TPCF(params)
    #print('W,D', model_Wr,model_D_R_vir )
    #print('like', np.log(p))
    deg_of_free = len(y) - 4
    p=np.sum((ydata-y)**2 / sigma**2)
    p_t = chi2.sf(p,deg_of_free)
    print(p_t)
    return(p_t)


'''def loglikelihood(params):
    #Define the probability that the model, given a set of parameters, is a good fit.
    #Get the associated y-value for the given the model parameters
    #y = get_v(params)
    print('aaaa', params)
    y = EW_D_model(params)
    #Calculate the log likelihood and return
    #Sigma is the error on the y-data
    #return np.nansum(-1.0*(ydata-y)**2 / sigma**2) #Gaussian function is likelihood for least squares fitting
    return(loglikelihood(y))'''

def logPosterior(params):
    #The  function MCMC evaluates; Bayes theorem in log space
    #p(model params | data) * p(data)

    return loglikelihood(params) + np.log(totalprior(params, parammins, parammaxs))


######################
#Setup the MCMC sampler and run it!

sampler = emcee.EnsembleSampler(nwalkers, ndim, logPosterior, backend=backend)

#Define the starting position of each walker, for each parameter
init_guess = np.zeros((nwalkers,ndim))
for ii, pmin in enumerate(parammins):
    init_guess[:,ii] = np.random.uniform(pmin, parammaxs[ii], size=nwalkers)



#Run the mcmc sampler!
sampler.run_mcmc(init_guess, nsteps, progress=True)

bot.sendMessage(2079147193, 'mcmc listo')

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

fig.savefig('mcmc_chains_4_TPCF.pdf')


#Make a corner plot (how each parameter scales with another)
#Configure the data for the corner package to understand, removing the burn-in values
data = chains[:, nburn:, :]

#Make the corner plot
fig1= corner.corner(data.reshape(data.shape[0]*data.shape[1], data.shape[2]), labels=paramnames)
fig1.savefig('mcmc_corner_4_TPCF.pdf')

bot.sendMessage(2079147193, 'Codigo listo TPCF:)')
