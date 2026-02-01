#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, rc, ticker
import os, sys

def do_norm(buf):

    return buf / np.linalg.norm(buf)

class Config(object):

    def __init__(self, **kw):

        np.random.seed(2025)

        self.N      =   kw['N']
        self.BW     =   kw['BW']
        self.Fs     =   self.BW * 2
        self.Ts     =   1. / self.Fs

def find_max_sigma(x, y):

    # Fit a quadratic
    p, cov = np.polyfit(x, y, 2, cov=True)
    a, b, c = p
    
    # --- xmax ---
    xmax = -b / (2*a)
    
    sigma_a2 = cov[0,0]
    sigma_b2 = cov[1,1]
    sigma_ab = cov[0,1]
    
    dx_da =  b/(2*a**2)
    dx_db = -1/(2*a)
    
    sigma_xmax = np.sqrt(dx_da**2*sigma_a2 +
                         dx_db**2*sigma_b2 +
                         2*dx_da*dx_db*sigma_ab)
    
    # --- ymax ---
    ymax = c - b**2/(4*a)
    
    sigma_c2 = cov[2,2]
    sigma_ac = cov[0,2]
    sigma_bc = cov[1,2]
    
    dy_da =  b**2/(4*a**2)
    dy_db = -b/(2*a)
    dy_dc = 1
    
    sigma_ymax = np.sqrt(
        dy_da**2*sigma_a2 +
        dy_db**2*sigma_b2 +
        dy_dc**2*sigma_c2 +
        2*dy_da*dy_db*sigma_ab +
        2*dy_da*dy_dc*sigma_ac +
        2*dy_db*dy_dc*sigma_bc
    )
    
    print("xmax =", xmax, "+/-", sigma_xmax)
    print("ymax =", ymax, "+/-", sigma_ymax)

    return xmax, sigma_xmax, ymax, sigma_ymax

def fit_tau(taus, amps):

    idmax   =   np.argmax(amps)

    id0     =   idmax - 2
    id1     =   idmax + 2
    x   =   taus[id0:id1+1]
    y   =   amps[id0:id1+1]

#    p   =   np.polyfit(x, y, deg=2) 
    return find_max_sigma(x, y)

def make_sum(buf, nsum):

    buf =   buf.reshape((-1, nsum))
    buf =   buf.mean(axis=-1)
    return buf

def gen_raw(N, BW, df_fr, tau_frac, tau_res):

    cfg =   Config(N=N, BW=BW)

    cfg.df_fr       =   df_fr
    cfg.tau_frac    =   tau_frac

    raw_name    =   'raw.npy'

    if os.path.exists(raw_name):
        buf1, buf2 = np.load(raw_name, allow_pickle=True)
        return buf1, buf2, cfg

def do_fit(cfg, taus, vsums, name = ''):

    amps    =   np.abs(vsums)

    xmax, xsig, ymax, ysig  =   fit_tau(taus, amps)

    if name == '':
        return xmax, xsig, ymax, ysig

    plt.clf()
    plt.plot(taus, amps, ls='-', c='steelblue', marker='s', mec='none')
    plt.xlabel('Delay [$u$s]')
    plt.savefig('fit_%s.png' % (name))

    return

def plot_spec(cfg, spec, name):

    N   =   cfg.N
    BW  =   cfg.BW
    Fs  =   cfg.BW * 2

    df  =   Fs / N
    
    fx  =   np.arange(N) * df

    nsum    =   1
    spec    =   make_sum(spec, nsum)
    fx      =   make_sum(fx, nsum)

    plt.clf()
    rc('font', size=14)
    fig =   plt.figure()
    fig.set_figwidth(6)
    fig.set_figheight(4)
    ax  =   fig.add_subplot(111)
    plt.subplots_adjust(left=0.16, right=0.96, top=0.97, bottom=0.14)
    plt.plot(fx/1E3, np.angle(spec)/np.pi*180., 'rs', ms=3)
    plt.xlim(0, BW*2/1E3)
#    plt.ylim(-np.pi, np.pi)
    plt.ylim(-180., 180.)
    plt.xlabel('Frequency [kHz]')
    plt.ylabel('Phase [degree]')

    ax.yaxis.set_major_locator(ticker.MultipleLocator(60))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
 
#    ax.twinx()
#    amp =   np.abs(spec)
#    plt.plot(fx/1E3, 20.*np.log10(amp), 'b-', lw=2)
#    plt.ylabel('Amplitude [dB]', color='b')

    plt.savefig('spec_%s.png' % (name))
    plt.savefig('spec_%s.eps' % (name))

