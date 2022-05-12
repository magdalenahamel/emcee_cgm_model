import utils as csu
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const

import astropy.units as u
from astropy.coordinates import SkyCoord

from sampledist import RanDist

from scipy.special import wofz
from scipy.special import gamma

from scipy import stats
import random
from astropy.cosmology import FlatLambdaCDM

import importlib
from numpy import float32

import concurrent.futures
import itertools
import functools
import datetime

from disco import Disco

dat_N_b = pd.read_csv('table4-zlim1_12842.ascii', delim_whitespace=True)
N_churchill = dat_N_b['N-MgII'].to_numpy()
b_churchill = dat_N_b['b-MgII'].to_numpy()
z_churchill = dat_N_b['zabs'].to_numpy()

#### Define the spectral resolution ####


zabs_ = 0.656
lam0 = 2796.35

vel_min = -1500
vel_max = 1500
lam_min = ((vel_min/const.c.to('km/s').value)+1)*(lam0*(1+zabs_)) 
lam_max = ((vel_max/const.c.to('km/s').value)+1)*(lam0*(1+zabs_)) 

w_spectral = 0.03

wave = np.arange(lam_min,lam_max+w_spectral, w_spectral)
vels_wave = (const.c.to('km/s').value * ((wave/ (lam0 * (1 + zabs_))) - 1))

##### Data ####

magiicat_iso = data_r_vir = pd.read_csv('magiicat_isolated.txt', on_bad_lines=False, delim_whitespace=True)
D_magiicat = magiicat_iso['D'].to_numpy()
R_vir_magiicat = D_magiicat/magiicat_iso['D/R_vir'].to_numpy()
v_magiicat = magiicat_iso['V_circ'].to_numpy()
z_gal_magiicat = magiicat_iso['z_gal'].to_numpy()

D_R_v_magiicat = np.array((D_magiicat,R_vir_magiicat, v_magiicat, z_gal_magiicat)).T

churchill_iso = data_r_vir = pd.read_csv('Churchill_iso_full.txt', on_bad_lines=False, delim_whitespace=True, dtype=object)
D_R_vir_churchill_iso = churchill_iso['etav'].to_numpy()
W_r_churchill_iso = churchill_iso['Wr'].to_numpy()
D_churchill_iso = churchill_iso['D'].to_numpy()

#cfchen_iso = pd.read_csv('chen_data.txt', error_bad_lines=False, delim_whitespace=True)
#D_chen = chen_iso['rho'].to_numpy()



#### Parameters distributions ####

D_dist_magii = np.histogram(D_churchill_iso,100,(np.min(D_churchill_iso),np.max(D_churchill_iso)))[0]
D_vals_magii = np.linspace(np.min(D_churchill_iso),np.max(D_churchill_iso),100)
f_D_C = RanDist(D_vals_magii, D_dist_magii)

'''Inclination distribution'''
def sin_i_dist(y, ymin):
    Ay = 1/np.sqrt(1-ymin**2)
    return(Ay * y / np.sqrt(1-(y**2)))

sinivals = np.linspace(np.sin(np.radians(5.7)),0.99,100)
f_D_i = RanDist(sinivals, sin_i_dist(sinivals,np.radians(5.7)))
print('exi', f_D_i.random(10))

'''Doppler parameter distribution'''

df = 6.71701 # parametro de forma.
chi2 = stats.chi2(df)
x = np.linspace(0,20)
fp = chi2.pdf(x) # Función de Probabilidad

bvals = np.linspace(0,20,100)
fNb = RanDist(bvals, chi2.pdf(bvals))

'''v_max distribution'''

def rot_vel_dist(v, phi_c, v_c, alpha, beta):
    a = phi_c*((v/v_c)**alpha)
    b = np.exp(-(v/v_c)**beta)
    c = beta/gamma(alpha/beta)
    return(a*b)

v_dist_magii = np.histogram(v_magiicat,100,(np.min(v_magiicat),np.max(v_magiicat)))[0]

v_vals = np.linspace(np.min(v_magiicat),np.max(v_magiicat),100)
f_v = RanDist(v_vals, v_dist_magii)


'''N distribution'''

def ndist(n, beta = 1.5 ):
    return n**-beta

nvals = np.logspace(11, 16, 1000)
#fN = RanDist(nvals, ndist(nvals))
N_dist_churchill = np.histogram(N_churchill,bins=np.logspace(np.log10(np.min(N_churchill)),np.log10(15), 51))[0]
                              
N_vals = np.logspace(np.log10(np.min(N_churchill)),np.log10(15),50)
fN = RanDist(N_vals, N_dist_churchill)


#### Model fuctions ####

def get_clouds(ypos,zpos,probs,velos):
    randomnum = np.random.uniform(0, 100, len(probs))
    selected = probs >= randomnum
    return(velos[selected])

def averagelos(model, D, alpha, lam, iter,X, z, grid_size, b, r_0, v_max, h_v, v_inf, results):
    h = model.h
    incli = model.incl

        #list of len(iter) velocities satisfying prob_hit
    results = np.asarray(results)



    fluxes = [0]*iter
    fluxtoaver = [losspec(model, lam,results[x],X,b,z) for x in fluxes]
    fluxtoaver = np.asarray(fluxtoaver)
    totflux = np.median(fluxtoaver, axis=0)

    return(totflux)

def ndist(n, beta = 1.5 ):
    return n**-beta

def losspec(model,lam,velos,X, b,z):
    Nst = fN.random(len(velos))
    
    Ns = 10**(Nst)
    
    N = np.empty([len(velos), 1])
    for i in range(len(Ns)):
        N[i,0]=Ns[i]
    taus = Tau(lam,velos,X,N,b,z)
    tottau = np.sum(taus,axis=0)
    return(np.exp(-tottau))

def Tau(lam,vel,X,N, b,z):
    if X ==1:
        lam0 = [2796.35]
        f = [0.6155]
    if X ==2:
        lam0 = [2803.53]
        f = [0.3054]
    if X ==12:

        lam0 = [2796.35, 2803.53]
        f = [0.6155, 0.3054]

    gamma, mass = [2.68e8, 24.305]
    c  = const.c.to('cm/s').value
    sigma0 = 0.0263
    taus = []
    for i in range(len(lam0)):

        lamc = ((vel[:,None]/const.c.to('km/s').value)+1)*((lam0[i]))
            #print('lamc', lamc)
        nu = c/(lam*1e-8)
        nu0 = c/(lamc*1e-8)

        dnu = nu - (nu0/(1+z))
        dnud = (b[:, np.newaxis]*100000)*nu/c
       # print(len(dnud))

        x = dnu/dnud
        y = gamma/(4*np.pi*dnud)
        zi = x + 1j*y
        v = np.asarray(np.real(wofz(zi)/(np.sqrt(np.pi)*dnud)))

        #print('N', N)
        #print('v', v)

        taut =N * sigma0*f[i] * v

        taus.append(taut)

    taus = np.asarray(taus)
    taust = taus.sum(axis=0)
    #print(taust)
    return(taust)

def get_cells(model,D,alpha,size,r_0,p_r_0, vR,hv,prob_func,  rmax, por_r_vir):
    #print('get_cells', vR,hv)
    h = model.h
    incli = model.incl

    m = -np.tan(np.radians(90-incli))

    x0 = D * np.cos(np.radians(alpha))
    y0 = D*np.sin(np.radians(alpha))/np.cos(np.radians(incli))
    n = -m*y0
    #print('y0,alpha,incli',y0,alpha,incli)
    y1 = ((h/2)-n)/m
    y2 = (-(h/2)-n)/m
    #print('hnm', h,n,m)
    mindis = np.sqrt(2*(size**2))/2
    z1 = h/2
    z2 = -h/2
    b = -1
    zgrid = np.arange((-h/2) + (size/2), (h/2) + (size/2), size)
    #print('y1', y1)
    #print('size',size)
    ymin = int(y1/size) * size + (size/2)
    ymax = int(y2/size)*size +(size/2)



        #print('yminmax', ymin,ymax)
    ygrid = np.arange(ymin,ymax,size)
    points = abs((m * ygrid + b * zgrid[:,None] + n)) / (np.sqrt(m * m + b * b))
    selected = points <= mindis
    yv, zv = np.meshgrid(ygrid, zgrid)
    ypos = yv[selected]
    zpos = zv[selected]
        #print('yposis', ypos)
    radios = np.sqrt((x0**2)+ypos**2)
    probs = prob_func(radios, r_0, p_r_0, rmax, por_r_vir)
    velos = los_vel(model, ypos, D, alpha, vR, hv)
    return(ypos,zpos, probs, velos)

def los_vel(model, y, D, alpha, vR, hv, v_inf=0):
    #print('los_vel', vR, hv)
    v_los_inf = (v_inf * np.sin(model.incl_rad)) * (y/(np.sqrt((y**2) + D**2)))
    al_rad = np.radians(alpha)

    R = D * np.sqrt(1+(np.sin(al_rad)**2)*np.tan(model.incl_rad)**2)
    vrot = (2/np.pi)*np.arctan2(R,1)

    x0 = D * np.cos(al_rad)  # this is p in Ho et al. 2019, fig 10.
        #print('al,x0',al_rad,x0)
    y0 = D * np.sin(al_rad) / np.cos(model.incl_rad)  # this is y0 in the same fig.
    if x0>=0:
        a = np.sin(model.incl_rad) / np.sqrt(1 + (y/x0)**2)
    else:
        a = -np.sin(model.incl_rad) / np.sqrt(1 + (y/x0)**2)
    if hv == 0:
        vr = (vR*vrot*a) + v_los_inf
    b = np.exp(-np.fabs(y - y0) / hv * np.tan(model.incl_rad))
        #print(b)
    vr = (vR*vrot*a*b) + v_los_inf

        #print('vel', vr)
    return(vr)

xs = np.linspace(-200,200,2*200)
ys = np.linspace(-200,200,2*200)
x, y = np.meshgrid(xs, ys)
#print('before csu')
d_alpha_t = csu.xy2alpha(x, y)
#print('after csu')
ds = []
alphas = []
for i in range(len(d_alpha_t[0])):
    for j in range(len(d_alpha_t[0][0])):
        if d_alpha_t[0][i][j]>200:
           pass
        else:
           ds.append(d_alpha_t[0][i][j])
           alphas.append(d_alpha_t[1][i][j])
            
def get_nielsen_sample(prob_r_cs,csize,hv, filling_factor,rmax,por_r_vir, zabs,h, wave, vels_wave, w_pix, par_param):
    #print('running get_nielsen_sample')
    #print('pars, prob_r_cs',par_param,  prob_r_cs) 
    # print('loop',prob_r_cs,csize,hv)
    d = par_param[0]
    #print('defi primer param')
    alpha = par_param[1]
    random_inclis_i = par_param[2] 
    random_r_vir_i = par_param[3] 
    random_vels_i = par_param[4]
   # print('defi seg param')
    model = Disco(h, random_inclis_i, Rcore=0.1)
            #print('loop',bs,csize,h,hv)
    #print('defi primer model')
    cells = get_cells(model,d,alpha,csize, random_r_vir_i,prob_r_cs,random_vels_i,hv, filling_factor,  rmax, por_r_vir)
    #print('get_cells_runed')
    results = [0]*1
    results = [get_clouds(cells[0],cells[1],cells[2],cells[3]) for x in results]
    #print('get_clouds_runed')
    results_nr = csu.nr_clouds(results, 6.6)
    b = fNb.random(len(results[0]))
    speci = averagelos(model, d, alpha, wave, 1,1, zabs, csize, b, 0, random_vels_i, hv, 0, results)
    #print('averagelos_runed')
    equi_wid_i = csu.eq_w(speci, vels_wave, random_vels_i+20, zabs,  w_pix)
    return(equi_wid_i, results_nr[0], speci)
   

#print('initial alpha', alphas)
"""This Class Sample represents a sample of MgII absorbers from Galaxies with the model Disco"""


class Sample:
    """Represents a sample of MgII absorbers"""

    def __init__(self, filling_factor, dmax, h=10, w_pix = 0.03, zabs=0.656, csize=1, hv=10, sample_size=2000):
        """c"""
        self.filling_factor  = filling_factor
        self.dmax = dmax
        self.h = h
        self.w_pix = w_pix
        self.zabs=zabs
        self.csize=csize
        self.hv=hv
        self.sample_size=sample_size
        
    def get_spec_TPCF(self,  prob_r_cs, rmax, por_r_vir, incli, alpha, size_i = 50):
        dmax = self.dmax
        filling_factor = self.filling_factor
        dmax = self.dmax
        h = self.h
        w_pix = self.w_pix
        zabs = self.zabs
        csize = self.csize
        hv = self.hv
        sample_size = self.sample_size

        inclis_s = incli[0]
        inclis_b = incli[1]
        sin_incli_s = np.arcsin(np.radians(inclis_s))
        sin_incli_b = np.arcsin(np.radians(inclis_b))
        alpha_s = alpha[0]
        alpha_b = alpha[1]

        i = 0
        xs = np.linspace(-dmax,dmax,2*dmax)
        ys = np.linspace(-dmax,dmax,2*dmax)
        x, y = np.meshgrid(xs, ys)
        d_alpha_t = csu.xy2alpha(x, y)

        ds = []
        alphas = []
        for i in range(len(d_alpha_t[0])):
            for j in range(len(d_alpha_t[0][0])):
                if d_alpha_t[0][i][j]>dmax:
                    pass
                else:
                    ds.append(d_alpha_t[0][i][j])
                    alphas.append(d_alpha_t[1][i][j])


        z_median = np.median(z_gal_magiicat)
        R_vir_min = np.min(R_vir_magiicat)
        R_vir_max = np.max(R_vir_magiicat)

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        H = cosmo.H(z_median)
        vel_min = R_vir_min * u.kpc * H / 0.1
        vel_min = vel_min.to(u.km/u.second).value

        vel_max = R_vir_max * u.kpc * H / 0.1
        vel_max = vel_max.to(u.km/u.second).value

        vels = np.linspace(vel_min,vel_max,1000)

        vels_dist = rot_vel_dist(vels,0.061,10**2.06, 0.66, 2.10)




        random_nr_clouds = []
        random_specs = []
        random_alphas = []
        random_im_par = []
        random_vels = []
        random_b = []
        random_inclis = []
        random_r_vir = []
        random_equi_wid = []

        i = 0
        while i < size_i:
            alpha_i = np.random.uniform(alpha_s, alpha_b)
            d_i = f_D_C.random(1)
            random_vels_i = f_v.random(1)
            random_r_vir_i = (random_vels_i * u.km /u.second)*0.1/H
            random_r_vir_i = random_r_vir_i.to(u.kpc).value

            random_inclis_i = f_D_i.random(1)
            #print('inclis rad', random_inclis_i )
            random_inclis_i = np.degrees(np.arcsin(random_inclis_i))
  
            model = Disco(h, random_inclis_i, Rcore=0.1)
            cells = get_cells(model,d_i,alpha_i,csize, random_r_vir_i,prob_r_cs,random_vels_i,hv,self.filling_factor,  rmax, por_r_vir)
            results = [0]*1
            results = [get_clouds(cells[0],cells[1],cells[2],cells[3]) for x in results]
            results_nr = len(results[0])
            if results_nr > 0:
                b = fNb.random(len(results[0]))
                speci = averagelos(model, d_i, alpha_i, wave, 1,1, zabs, csize, b, 0, random_vels_i, hv, 0, results)
                random_specs.append(speci)
                random_nr_clouds.append(results_nr)
                #print('nr. clouds', results_nr)
                equi_wid_i = csu.eq_w(speci, vels_wave, random_vels_i, zabs,  w_pix)
                random_equi_wid.append(equi_wid_i)
                i = i+1
            else:
                pass
            #print(i)

        return(np.asarray([np.asarray(random_nr_clouds),
        np.asarray(random_specs),
        np.asarray(alpha_i),
        np.asarray(d_i),
        np.asarray(random_vels_i),
        np.asarray(random_b),
        np.asarray(random_inclis_i),
        np.asarray(random_r_vir_i),
        np.asarray(random_equi_wid)]))


    def Nielsen_sample(self, prob_r_cs, rmax, por_r_vir):
        begin_time = datetime.datetime.now()
        #print('runing Nielsen_sample')
        dmax = self.dmax
        filling_factor = self.filling_factor
        dmax = self.dmax
        h = self.h
        w_pix = self.w_pix
        zabs = self.zabs
        csize = self.csize
        hv = self.hv
        sample_size = self.sample_size

        #wave = np.arange(4849.58349609375,5098.33349609375+0.125, w_pix)
        #vels_wave = (const.c.to('km/s').value * ((wave/ (2796.35 * (1 + zabs))) - 1))

        z_median = np.median(z_gal_magiicat)
        R_vir_min = np.min(R_vir_magiicat)
        R_vir_max = np.max(R_vir_magiicat)
        #print('1')
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        H = cosmo.H(z_median)
        vel_min = R_vir_min * u.kpc * H / 0.1
        vel_min = vel_min.to(u.km/u.second).value
        #print('2')
        vel_max = R_vir_max * u.kpc * H / 0.1
        vel_max = vel_max.to(u.km/u.second).value

        vels = np.linspace(vel_min,vel_max,1000)

        vels_dist = rot_vel_dist(vels,0.061,10**2.06, 0.66, 2.10)
        fN_v = RanDist(vels, vels_dist)
        
        #print('3')
        d_alpha = list(zip(ds,alphas))

        random_nr_clouds = []
        random_specs = []
        random_alphas = []
        random_im_par = []
        random_vels = []
        random_b = []
        random_inclis = []
        random_r_vir = []
        random_equi_wid = []
        alpha_i = random.choices(alphas, k=sample_size)
        #print('selected alpha', alpha_i)
        #d_i = random.choices(ds, k=sample_size)
        #print('5')
        d_i = f_D_C.random(sample_size)

        #print('6')
        random_vels_i = f_v.random(sample_size)
        #print('6a')
        #random_vels_i = fN_v.random(sample_size)
        random_r_vir_i = (random_vels_i * u.km /u.second)*0.1/H
        #print('6b')
        random_r_vir_i = random_r_vir_i.to(u.kpc).value

        #print('7')
        random_inclis_i = f_D_i.random(sample_size)
        #print('inclis rad', random_inclis_i )
        random_inclis_i = np.degrees(np.arcsin(random_inclis_i))
        #print('inclis deg', random_inclis_i )
        #print('8')
        random_nr_clouds_pow_i = []
        random_specs_pow_i = []
        random_equi_wid_pow_i =[]
        #print('before loop')
        partial_params = [[d_i[i], alpha_i[i], random_inclis_i[i], random_r_vir_i[i], random_vels_i[i]] for i in range(sample_size)]
        
        #print('defined partial_params')

        
        partial_get_niel_samp = functools.partial(get_nielsen_sample,  prob_r_cs,csize,hv, filling_factor,rmax,por_r_vir, zabs,h, wave, vels_wave, w_pix)
        
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(partial_get_niel_samp, partial_params))
            #result_list= list(results)
       
        #print('executor.map ready')
        #print('result_list',results)
        random_equi_wid = [r[0] for r in results]
        random_nr_clouds = [r[1] for r in results]
        random_specs = [r[2] for r in results]
        
       # (equi_wid_i, results_nr[0], speci)
        
        print(datetime.datetime.now() - begin_time)
        return(np.asarray([np.asarray(random_nr_clouds),
        np.asarray(random_specs),
        np.asarray(alpha_i),
        np.asarray(d_i),
        np.asarray(random_vels_i),
        np.asarray(random_b),
        np.asarray(random_inclis_i),
        np.asarray(random_r_vir_i),
        np.asarray(random_equi_wid)]))


'''for i in range(sample_size):
            print('running samile ', i)
            print('loop',prob_r_cs,csize,hv)
            d = d_i[i]
            alpha = alpha_i[i]
            model = cgm.Disco(h, random_inclis_i[i], Rcore=0.1)
            #print('loop',bs,csize,h,hv)
            cells = get_cells(model,d,alpha,csize, random_r_vir_i[i],prob_r_cs,random_vels_i[i],hv,self.filling_factor,  rmax, por_r_vir)
            results = [0]*1
            results = [get_clouds(cells[0],cells[1],cells[2],cells[3]) for x in results]
            results_nr = csu.nr_clouds(results, 6.6)
            b = fNb.random(len(results[0]))
            speci = averagelos(model, d, alpha, wave, 1,1, zabs, csize, b, 0, random_vels_i[i], hv, 0, results)
            random_specs.append(speci)
            random_nr_clouds.append(results_nr[0])
            equi_wid_i = csu.eq_w(speci, vels_wave, random_vels_i[i]+20, zabs,  w_pix)
            random_equi_wid.append(equi_wid_i)
            #print(i)'''
