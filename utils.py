import numpy as np
import math
from math import log
from astropy import constants as const
from scipy.special import wofz
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.convolution import convolve, Gaussian1DKernel

"""Utilities for cgmspec"""


def ndist(n, beta = 1.5 ):
   return n**-beta

def wavelenght_to_rfvel(lam, lam0, z):
    return (const.c.to('km/s').value * ((lam / (lamc *(1 + z)))-1))

def get_alpha(radec, pos_gal, PA_gal):
    ra_dec = np.hsplit(radec,2)
    pos = SkyCoord(ra_dec[0]*u.degree,ra_dec[1]*u.degree,frame='icrs')
    PA_pix = pos_gal.position_angle(pos).degree
    alpha = PA_gal +180 -PA_pix
    return (alpha)

def get_impactparam(radec, pos_gal, scale):
    ra_dec = np.hsplit(radec,2)
    pos = SkyCoord(ra_dec[0]*u.degree,ra_dec[1]*u.degree,frame='icrs')
    D = scale * pos_gal.separation(pos).arcsec
    return D

def prob_hit(r, rmin, r_0, prob_rmin=100):
    """
    Probability of hitting a cloud at distance r in the plane xy of a disc of radius rmax
    :param r: np.array, array with distances to the center of disc in plane xy in kpc
    :param rmin: float, minimum radius of the disc in plane xy in kpc, below this value probability is set by prob_rmin
    :param rmax: float, radius of the disc in plane xy in kpc, above this value prob is set to zero
    :param prob_rmin: float, probability at rmin or below
    :param prob_rmax: float, probability at rmax
    :return: float, probability of hitting a cloud
    """

    ind = np.log(prob_rmin)/(np.log(rmin)-np.log(r_0))
    A = r/r_0
    prob = A**(ind)
    prob = np.where(r<=rmin, prob_rmin, prob)

    '''b = np.log10(prob_rmax / prob_rmin) / np.log10(rmax)
    prob = prob_rmin * (r ** b)
    prob = np.where(r>rmax, 0., prob)
    prob = np.where(r<=rmin, prob_rmin, prob)'''
    return prob


def los_disc_intersect(D, i, alpha, R, h):
    """For a given set of (i, D, alpha) + (R, h), this function return (x0, yt1, yt2, zt1, zt2)
    If the sightline does not intersect the disk, it returns False
    """

    radio = R
    hdis = h
    incli = i
    incli_rad = np.radians(incli)

    al_rad = np.radians(alpha)
    x0 = D * np.cos(al_rad)
    y0 = D * np.sin(al_rad) / np.cos(incli_rad)
    #print('x0',x0)
    # check if sightline intersects the disk
    if np.sqrt((x0**2) + (y0**2)) > radio:
        return False

    if incli == 90:  # edge-on
        z0 = D * np.sin(al_rad)
        if np.fabs(z0) > hdis / 2.:
            return False  # outside projected disk
        else:
            yt = [-np.sqrt((radio ** 2) - x0 ** 2), np.sqrt((radio ** 2) - x0 ** 2)]
            zt = [D * np.sin(al_rad), D * np.sin(al_rad)]

    elif incli == 0.0:  # face-on
        yt = [D * np.sin(al_rad), D * np.sin(al_rad)]
        zt = [-hdis / 2, hdis / 2]
        incli_rad = np.radians(incli)
        # print(yt, zt)

    else:
        incli_rad = np.radians(incli)
        y0 = D * np.sin(al_rad) / np.cos(incli_rad)
        #print('y0',y0)

        z0 = 0

        radio1 = np.sqrt((radio ** 2) - x0 ** 2)
        #print('radii',radio1)
        ymin = y0 - (hdis / 2) * np.tan(incli_rad)
        ymax = y0 + (hdis / 2) * np.tan(incli_rad)
        zmin = -(np.sqrt((-radio1 - y0) ** 2) / np.tan(incli_rad))
        zmax = (np.sqrt((radio1 - y0) ** 2) / np.tan(incli_rad))
        ys = [ymin, ymax, -radio1, radio1]
        zs = [-hdis / 2, hdis / 2, zmin, zmax]
        yt = []
        zt = []
        #print('AAAAA',ymin, ymax, zmin, zmax)

        for i in range(len(ys)):
            if abs(ys[i]) < radio1:
                #           print(ys[i], zs[i])
                yt.append(ys[i])
                zt.append(zs[i])
            else:
                pass
        for i in range(len(zs)):
            if abs(zs[i]) < hdis / 2:
                 #           print(ys[i], zs[i])
                yt.append(ys[i])
                zt.append(zs[i])
            else:
                pass

        yt = np.sort(yt)
        zt = np.sort(zt)


    if len(yt)==0 or len(zt)==0:
        return(False)
    return x0, yt[0], yt[1], zt[0], zt[1]



def Tau(lam,vel,X,N, b,z):
    '''
    lam: array of wavelenghts for out spectrum
    vel: array with velos of the get_clouds
    N: array with logN of the clouds, len(N )=lem(Vel)
    Returns: Taus for all clouds in a LOS, len(vel) = nr clouds
    '''
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
        dnud = (b*100000)*nu/c

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


def normflux(taun):
    taun = np.asarray(taun)
    return(np.exp(-taun))


def sumtau(taus):
    tautosum =  []
    for i in range(len(taus)):
        tautosum.append(taus[i][0])
    tautosum = np.asarray(tautosum)
    #print(tautosum)

    tausum = []
    for i in range(len(tautosum[0])):
        tot = 0
        for j in range(len(tautosum)):
            tot = tot + tautosum[j][i]
        tausum.append(tot)
    #print(tausum)

    return(tausum)


####################################################################


def xy2alpha(x,y):
    alpha = np.arctan2(y, x)
    alpha_deg = np.degrees(alpha)
    d = np.sqrt((x**2) + y**2)
    return(d,alpha_deg)

def nr_clouds(vels, d_vel):
    vels_detected = []
    if len(vels[0]) == 0:
        return(0, None, None)
    vels_t = np.sort(vels)
    nr = 0
    vels_t = vels_t[0]
    vel0 = vels_t[0]
    for i in range(len(vels_t)):
        if i == 0:
            nr = nr + 1
            vels_detected.append(vels_t[i])
        elif abs(vel0 - vels_t[i]) > d_vel:
            nr = nr + 1
            vel0 = vels_t[i]
            vels_detected.append(vels_t[i])
    #print(nr)
    return(nr, vels_detected, vels)

def eq_w(spec, vel, rang,z, w_pix):
    cond = np.abs(vel)<rang
    flux = spec[cond]
    a = (np.sum(1-flux))*w_pix
    #print(a)
    w = a/(1+z)
    return(w)


def filtrogauss(R, spec_res, lam_0, flux):
    del_lam = lam_0/R
    del_lam_pix = del_lam/spec_res
    gauss_kernel = (Gaussian1DKernel(del_lam_pix))
    gausflux = convolve(flux, gauss_kernel)
    return(np.asarray(gausflux))
