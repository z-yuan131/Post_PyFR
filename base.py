# -*- coding: utf-8 -*-
from pyfr.inifile import Inifile
from pyfr.readers.native import NativeReader
from pyfr.util import memoize, subclass_where
from pyfr.shapes import BaseShape
from pyfr.quadrules import get_quadrule

import numpy as np
import h5py
from collections import defaultdict


class Base(object):
    def __init__(self, avg):

        # Time series
        self.get_time_series(avg['series_time'])

        # Define mesh name and solution name
        self.name = name = [avg['mesh'],f"{avg['soln']}{self.time[0]}.pyfrs"]

        # Output directory
        try:
            self.dir = avg['odir']
        except KeyError:
            self.dir = '.'

        # solution dirctory
        self.solndir = avg['soln']

        self.mesh = NativeReader(name[0])
        self.soln = NativeReader(name[1])

        # Check solution and mesh are compatible
        if self.mesh['mesh_uuid'] != self.soln['mesh_uuid']:
            raise RuntimeError('Solution "%s" was not computed on mesh "%s"' %
                               (name[0], name[1]))

        # Load the configuration and stats files
        self.cfg = Inifile(self.soln['config'])
        self.stats = Inifile(self.soln['stats'])
        self.order = self.cfg.getint('solver','order')
        self.dtype = np.dtype(self.cfg.get('backend','precision')).type

        # Data file prefix (defaults to soln for backwards compatibility)
        self.dataprefix = self.stats.get('data', 'prefix', 'soln')

        # Get element types and array shapes
        self.mesh_inf = self.mesh.array_info('spt')
        self.soln_inf = self.soln.array_info(self.dataprefix)

        # Get the number of elements of each type in each partition
        self.mesh_part = self.mesh.partition_info('spt')

        # Dimensions
        self.ndims = next(iter(self.mesh_inf.values()))[1][2]
        self.nvars = next(iter(self.soln_inf.values()))[1][1]

        # Mesh order
        self.meshord = self._get_mesh_order()

        # Constants
        self._constants = self.cfg.items_as('constants', float)

        # If using Sutherland's Law as viscous correction
        self._viscorr = self.cfg.get('solver', 'viscosity-correction', 'none')

    def get_time_series(self, time):
        self.tst = time[0]
        self.ted = time[1]
        self.dt = time[2]

        Ndt = int((time[1] - time[0])/time[2] + 1)
        tt = np.linspace(time[0], time[1], Ndt, endpoint=True)
        self.time = list()
        for i in range(len(tt)):
            self.time.append(f"{time[-1]}".format(tt[i]))

    def get_time_series_mpi(self, rank, size):
        time = []
        irank = np.arange(size)
        from itertools import cycle
        for r, t in zip(cycle(irank), self.time):
            if r == rank:
                time.append(t)
        return time

    # Operators
    def _get_mesh_op(self, etype, nspts):
        svpts = self._get_std_ele(etype, nspts, self.order)
        mesh_op = self._get_mesh_op_vis(etype, nspts, svpts)
        return mesh_op

    def _get_ops_interp(self, nspts, etype, upts, nupts, order):
        svpts = self._get_std_ele(etype, nspts, order)
        mesh_op = self._get_mesh_op_gll(etype, nspts, upts)
        return mesh_op

    def _get_vis_op(self, nspts, etype, order):
        svpts = self._get_std_ele(etype, nspts, order)
        mesh_op = self._get_mesh_op_vis(etype, nspts, svpts)
        return mesh_op

    def _get_shape(self, name, nspts, cfg):
        shapecls = subclass_where(BaseShape, name=name)
        return shapecls(nspts, cfg)

    def _get_std_ele(self, name, nspts, order):
        return self._get_shape(name, nspts, self.cfg).std_ele(order)

    def _get_mesh_op_vis(self, name, nspts, svpts):
        shape = self._get_shape(name, nspts, self.cfg)
        return shape.sbasis.nodal_basis_at(svpts).astype(self.dtype)

    def _get_mesh_op_gll(self, name, nspts, gllpts):
        shape = self._get_shape(name, nspts, self.cfg)
        return shape.sbasis.nodal_basis_at(gllpts).astype(self.dtype)

    def _get_mesh_op_qr(self, name, nspts, qrpts):
        shape = self._get_shape(name, nspts, self.cfg)
        return shape.sbasis.nodal_basis_at(qrpts).astype(self.dtype)

    def _get_mesh_op_sln(self, name, nspts, upts):
        shape = self._get_shape(name, nspts, self.cfg)
        return shape.sbasis.nodal_basis_at(upts).astype(self.dtype)

    def _get_npts(self, name, order):
        return self._get_shape(name, 0, self.cfg).nspts_from_order(order)

    def _get_order(self, name, nspts):
        return self._get_shape(name, nspts, self.cfg).order_from_nspts(nspts)

    def _get_soln_op(self, name, nspts):
        shape = self._get_shape(name, nspts, self.cfg)
        svpts = self._get_std_ele(name, nspts, self.order)
        return shape.ubasis.nodal_basis_at(svpts).astype(self.dtype)

    def _get_mesh_order(self):
        for key in self.mesh_inf:
            etype = key.split('_')[1]
            npts = self.mesh[key].shape[0]
            return self._get_order(etype, npts) - 1
