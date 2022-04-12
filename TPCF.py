from __future__ import division


from numpy import random
from scipy.spatial.distance import pdist, cdist
from scipy.stats import kstwobign, pearsonr
from scipy.stats import genextreme

import concurrent.futures
import functools

from tqdm.notebook import tqdm, trange
import time
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import Sample
#import cgmspec.Sample as sample

import concurrent.futures
import itertools

from astropy import constants as const
from astropy.convolution import convolve, Gaussian1DKernel

import time
import os
import Sample as sample
import telepot

bot = telepot.Bot('5106282512:AAFwfJ144PNtf9LwOP_o7Qmc6qrLNH8qEM8')
bot.sendMessage(2079147193, 'Empezó codico TPCF')

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


from itertools import combinations

def TPCF(params):
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



def prob_hit_log_lin(r, r_vir, a, b, por_r_vir = 0.5):
    r_t = r/r_vir
    return(np.exp(a)*(np.exp(-b*r_t)))

def TPCF_f(bs, csize, hs, hv):

#params = [bs,csize]

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

    results_nr_clouds = []
    results_specs = []
    results_alphas = []
    results_D = []
    results_vels = []
    results_b = []
    results_inclis = []
    results_R_vir = []
    results_Wr = []

    results_tpcf_minor_major = []
    results_tpcf_face_edge = []

    exp_fill_fac = sample.Sample(prob_hit_log_lin,200,sample_size=200, csize=csize, h=hs, hv=hv)
    e3_a_1 = exp_fill_fac.Nielsen_sample(np.log(100),bs,0.2)
    cond_spec = e3_a_1[0] == 0
    spec_abs = e3_a_1[1][~cond_spec]
    alphas_abs = e3_a_1[2][~cond_spec]
    inclis_abs = e3_a_1[6][~cond_spec]
    cond_major = alphas_abs < 45
    cond_minor = alphas_abs > 45

    cond_face = inclis_abs < 57
    cond_edge = inclis_abs > 57

    spec_minor = spec_abs[cond_minor]
    spec_major = spec_abs[cond_major]

    spec_face = spec_abs[cond_face]
    spec_edge = spec_abs[cond_edge]

    specs_tot = [(spec_minor,'minor'), (spec_major, 'major')]
    specs_tot_i = [(spec_face,'face'), (spec_edge, 'edge')]


    print('empieza TPCF')
    results = map(TPCF, specs_tot)
    list_res = list(results)

    results_tpcf_minor_major.append(list_res[0])
    results_tpcf_minor_major.append(list_res[1])

    results_i = map(TPCF, specs_tot_i)
    list_res_i = list(results_i)

    results_tpcf_face_edge.append(list_res_i[0])
    results_tpcf_face_edge.append(list_res_i[1])



    results_nr_clouds.append(e3_a_1[0])
    results_specs.append(e3_a_1[1])
    results_alphas.append(e3_a_1[2])
    results_D.append(e3_a_1[3])
    results_vels.append(e3_a_1[4])
    results_b.append(e3_a_1[5])
    results_inclis.append(e3_a_1[6])
    results_R_vir.append(e3_a_1[7])
    results_Wr.append(e3_a_1[8])

    return(results_tpcf_minor_major, results_tpcf_face_edge, results_nr_clouds, results_specs, results_alphas,results_D,results_vels, results_b,results_inclis,results_R_vir,results_Wr)



bs_n = 4
csize_n = 1
h_n = 10
hv_n = 10

bs_s = [0.01, 10]
csize_s =  [0.01, 10]
h_s = [1, 30]
hv_s = [0.001, 100]

def run_TPCF(bs, csize, h, hv, param_name):
    if param_name == 'bs':
        results_s = TPCF_f(bs[0], csize, h, hv)
        results_b = TPCF_f(bs[1], csize, h, hv)
    elif param_name == 'csize':
        results_s = TPCF_f(bs, csize[0], h, hv)
        results_b = TPCF_f(bs, csize[1], h, hv)
    elif param_name == 'h':
        results_s = TPCF_f(bs, csize, h[0], hv)
        results_b = TPCF_f(bs, csize, h[1], hv)
    elif param_name == 'hv':
        results_s = TPCF_f(bs, csize, h, hv[0])
        results_b = TPCF_f(bs, csize, h, hv[1])
    dirName = 'TPCF_param_exp/' + param_name
    os.makedirs(dirName)
    np.save(dirName + '/tpcf_minor_major_s',results_s[0])
    np.save(dirName + '/tpcf_face_edge_s',results_s[1])

    np.save(dirName+'/nr_clouds_s', results_s[2])
    np.save(dirName+'/specs_s', results_s[3])
    np.save(dirName+'/alphas_s',results_s[4])
    np.save( dirName+'/D_s', results_s[5])
    np.save(dirName+ '/vels_s', results_s[6])
    np.save(dirName+'/b_s', results_s[7])
    np.save(dirName+'/inclis_s', results_s[8])
    np.save(dirName+'/R_vir_s', results_s[9])
    np.save(dirName+'/Wr_s', results_s[10])
    

    np.save(dirName + '/tpcf_minor_major_b',results_b[0])
    np.save(dirName + '/tpcf_face_edge_b',results_b[1])

    np.save(dirName+'/nr_clouds_b', results_b[2])
    np.save(dirName+'/specs_b', results_b[3])
    np.save(dirName+'/alphas_b',results_b[4])
    np.save( dirName+'/D_b', results_b[5])
    np.save(dirName+ '/vels_b', results_b[6])
    np.save(dirName+'/b_b', results_b[7])
    np.save(dirName+'/inclis_b', results_b[8])
    np.save(dirName+'/R_vir_b', results_b[9])
    np.save(dirName+'/Wr_b', results_b[10])
    
run_TPCF(bs_s, csize_n, h_n, hv_n, 'bs')
run_TPCF(bs_n, csize_s, h_n, hv_n, 'csize')
run_TPCF(bs_n, csize_n, h_s, hv_n, 'h')
run_TPCF(bs_n, csize_n, h_n, hv_s, 'hv')
bot.sendMessage(2079147193, 'TPCF listo :)')
