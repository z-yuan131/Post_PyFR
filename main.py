# -*- coding: utf-8 -*-
import sys
from argparse import ArgumentParser, FileType
from collections import defaultdict

from feature.region import Region

from pyfr.inifile import Inifile


def config_file():
    # This is configuration part to pass parameters to the main part of the code.
    cfg = Inifile.load(sys.argv[1])
    arg = {}

    # Directory that stores mesh and solution
    # For mesh, give full directory and mesh name
    # For solution, give full directory and prefix of solution, i.e.
    dir = cfg.get('directory','dir')
    mname = cfg.get('directory','mesh_name')
    sheader = cfg.get('directory','soln_header')
    odir = cfg.get('directory','outdir')

    arg['mesh'] = f'{dir}/{mname}'
    arg['soln'] = f'{dir}/{sheader}'
    arg['odir'] = f'{odir}'

    # Time series (if single snapshot, keep start time equals end time)
    tstart = cfg.getfloat('time-series','tstart')
    tend = cfg.getfloat('time-series','tend')
    dt = cfg.getfloat('time-series','dt')
    fmat = cfg.get('time-series','fmat')
    arg['series_time'] = [tstart, tend, dt, fmat]

    return arg, cfg

def main():

    arg, cfg = config_file()

    if 'func-spanavg' in cfg.sections():
        from feature.spanavg import SpanavgBase
        fname = 'func-spanavg'
        SpanavgBase(arg, cfg, fname).main_proc()

    if 'func-probes' in cfg.sections():
        from feature.probes import Probes
        fname = 'func-probes'
        Probes(arg, cfg, fname).main_proc()

    if 'func-gradient' in cfg.sections():
        from feature.grad import Gradient
        fname = 'func-gradient'
        Gradient(arg, cfg, fname).gradproc()

    if 'feature-bl' in cfg.sections():
        raise NotImplementedError
        from functions.bl import BL, BL_Coeff, BL_Coeff_hotmap, Drag
        fname = 'feature-bl'
        #Duplicate_Pts(arg).main_proc()
        #BL(arg, cfg, fname).main_proc()
        #BL_Coeff(arg, cfg, fname).main_proc()
        #BL_Coeff_hotmap(arg, cfg, fname).main_proc()
        #Drag(arg, cfg, fname).main_proc()

    if 'func-Q-criterion' in cfg.sections():
        from feature.grad import Gradient
        from feature.Q_criterion import Q_criterion
        fname = 'func-Q-criterion'
        Q_criterion(arg, cfg, fname).main_proc()

    if 'func-gu' in cfg.sections():
        from feature.gatherup import DUP
        fname = 'func-gu'
        DUP(arg, cfg, fname).main_proc()

    if 'feature-spod' in cfg.sections():
        from functions.spod import SPOD
        fname = 'feature-spod'
        SPOD(arg, cfg, fname).main_proc()


if __name__ == "__main__":
    main()
