from feature.region import Region

import numpy as np
from collections import defaultdict
import h5py

from pyfr.readers.native import NativeReader
from pyfr.quadrules import get_quadrule
from pyfr.util import subclass_where, subclasses
from pyfr.shapes import BaseShape

class Gradient(Region):
    def __init__(self, argv, icfg, fname):
        super().__init__(argv, icfg, fname)

        from pyfr.solvers.base import BaseSystem
        # System and elements classes
        self.systemscls = subclass_where(
            BaseSystem, name=self.cfg.get('solver', 'system')
        )
        self.elementscls = self.systemscls.elementscls

        self.bl_support = icfg.get(fname, 'blsupport', True)

    def _get_eles(self):
        _, mesh_wall, fids = self.get_boundary()
        return mesh_wall, fids

    def gradproc(self):
        # Prepare for MPI process
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            mesh_wall, fids = self._get_eles()

            mesh = self._pre_proc_mesh(mesh_wall, fids)

        else:
            mesh = None
            mesh_wall = None


        # Boardcast pts and eles information
        mesh = comm.bcast(mesh, root=0)
        mesh_wall = comm.bcast(mesh_wall, root=0)

        # Get time series
        time = self.get_time_series_mpi(rank, size)

        soln_op = self._get_op_sln(mesh_wall)

        if self.bl_support:
            bl_op = self._get_op_bl(mesh)
            for t in time:
                self._proc_soln(t, mesh_wall, soln_op, mesh, bl_op)
        else:
            for t in time:
                self._proc_soln(t, mesh_wall, soln_op, mesh)

        if rank == 0:
            if self.bl_support:
                for etype in mesh:
                    if etype in self.suffix:
                        mesh[etype] = self._post_proc_fields(mesh[etype], bl_op[etype])
            self._flash_to_disk(mesh)


    def _proc_soln(self, time, mesh_wall, soln_op, mesh, bl_op = []):
        soln = f'{self.solndir}{time}.pyfrs'
        soln = self._load_snapshot(soln, mesh_wall, soln_op)

        for etype in soln:
            soln[etype] = self._pre_proc_fields_grad(etype, mesh[etype], soln[etype])
            if self.bl_support:
                soln[etype] = self._post_proc_fields(soln[etype].swapaxes(1,-1), bl_op[etype])

        self._flash_to_disk(soln, time)

    def _post_proc_fields(self, vars, m0):
        #ovars = []
        ovars, ndims = defaultdict(list), vars.shape[-1]
        for id, fid in enumerate(m0):
            temp = fid @ vars[:,id]
            ovars[len(temp)].append(temp)
            """
            if len(vars) != 125:
                if len(temp) != 25:
                    ovars.append(temp)
            else:
                ovars.append(temp)
            """
        ovars = {k: np.array(v).reshape(-1,ndims) for k,v in ovars.items()}
        return np.concatenate([v for k, v in ovars.items()], axis = 0)
        #return np.array(ovars).swapaxes(0,1)

    def _flash_to_disk(self, array, t = []):
        if t:
            f = h5py.File(f'{self.dir}/grad_{t}.s', 'w')
            for etype in array:
                f[f'{etype}'] = array[etype]
            f.close()
        else:
            f = h5py.File(f'{self.dir}/grad_mesh.s', 'w')
            for etype in array:
                f[f'{etype}'] = array[etype]
            f.close()


    def _load_snapshot(self, name, mesh_wall, soln_op = []):
        soln = defaultdict()
        f = h5py.File(name,'r')
        for k in mesh_wall:
            _, etype, part = k.split('_')
            name = f'{self.dataprefix}_{etype}_{part}'
            sln = np.array(f[name])[...,mesh_wall[k]]
            if len(soln_op) > 0:
                sln = np.einsum('ij, jkl -> ikl',soln_op[etype],sln)
            sln = self._pre_proc_fields_soln(sln.swapaxes(0,1)).swapaxes(0,1)
            try:
                soln[etype] = np.append(soln[etype], sln, axis = -1)
            except KeyError:
                soln[etype] = sln
        f.close()
        return soln

    def _get_op_sln(self, mesh_wall):
        soln_op = {}
        for key in mesh_wall:
            _, etype, part = key.split('_')
            # Operator
            if etype not in soln_op:
                name = f'{self.dataprefix}_{etype}_{part}'
                nspts = self.soln[name].shape[0]
                soln_op[etype] = self._get_soln_op(etype, nspts)
        return soln_op

    def _get_op_bl(self, mesh):
        basismap = {b.name: b for b in subclasses(BaseShape, just_leaf=True)}
        bl_op = defaultdict(list)
        m0, fpts = {}, {}
        for key in mesh:
            if key in self.suffix:
                msh = mesh[key]
                fid = mesh[f'{key}_fid']

                if key not in m0:
                    basis = basismap[key](msh.shape[0], self.cfg)
                    fpts[key] = basis.facefpts
                    m0[key] = self._get_interp_mat(basis, key)

                # Get surface mesh of hex type of elements
                for id in fid:
                    bl_op[key].append(m0[key][fpts[key][id]])
        return bl_op

    def _get_interp_mat(self, basis, etype):
        iqrule = self._get_std_ele(etype, basis.nupts, self.order)
        iqrule = np.array(iqrule)

        # Use strict solution points for quad and pri or line
        if self.ndims == 3:
            self.cfg.set('solver-elements-tri','soln-pts','williams-shunn')
            self.cfg.set('solver-elements-quad','soln-pts','gauss-legendre')
        else:
            self.cfg.set('solver-elements-line','soln-pts','gauss-legendre')

        m = []
        for id0, (kind, proj, norm) in enumerate(basis.faces):
            npts = basis.npts_for_face[kind](self.order)

            qrule = self._get_std_ele(kind, npts, self.order)
            qrule = np.array(qrule)

            pts = self._proj_pts(proj, qrule)

            # Search for closest point
            m0 = np.zeros([npts, basis.nupts])
            for id, pt in enumerate(pts):
                idx = np.argsort(np.linalg.norm(iqrule - pt, axis = 1))
                m0[id, idx[0]] = 1
            m.append(m0)

        return np.vstack(m)

    def _proj_pts(self, projector, pts):
        pts = np.atleast_2d(pts.T)
        return np.vstack(np.broadcast_arrays(*projector(*pts))).T

    def _pre_proc_mesh(self, mesh_wall, fids):
        mesh = {}
        for key, eles in mesh_wall.items():
            _, etype, part = key.split('_')
            nspts = self.mesh[key].shape[0]
            # Operator
            mesh_op_vis = self._get_vis_op(nspts, etype, self.order)
            msh = self.mesh[key][:,mesh_wall[key]]
            msh = np.einsum('ij, jkl -> ikl', mesh_op_vis, msh)

            try:
                mesh[etype] = np.append(mesh[etype], msh, axis = 1)
                mesh[f'{etype}_fid'] += fids[key]
            except KeyError:
                mesh[etype] = msh
                mesh[f'{etype}_fid'] = fids[key]

        return mesh

    def _pre_proc_fields_soln(self, soln):
        # Convert from conservative to primitive variables
        return np.array(self.elementscls.con_to_pri(soln, self.cfg))

    def _pre_proc_fields_grad(self, name, mesh, soln_original):
        # Reduce solution size since only velocity gradient is interested
        soln = soln_original[:,1:self.ndims+1]

        # Dimensions
        nupts, nvars = soln.shape[:2]

        # Get the shape class
        basiscls = subclass_where(BaseShape, name=name)

        # Construct an instance of the relevant elements class
        eles = self.elementscls(basiscls, mesh, self.cfg)

        # Get the smats and |J|^-1 to untransform the gradient
        smat = eles.smat_at_np('upts').transpose(2, 0, 1, 3)
        rcpdjac = eles.rcpdjac_at_np('upts')

        # Gradient operator
        gradop = eles.basis.m4.astype(self.dtype)

        # Evaluate the transformed gradient of the solution
        gradsoln = gradop @ soln.reshape(nupts, -1)
        gradsoln = gradsoln.reshape(self.ndims, nupts, nvars, -1)

        # Untransform
        gradsoln = np.einsum('ijkl,jkml->mikl', smat*rcpdjac, gradsoln,
                             dtype=self.dtype, casting='same_kind')
        gradsoln = gradsoln.reshape(nvars*self.ndims, nupts, -1)

        return np.append(soln_original, gradsoln.swapaxes(0,1), axis = 1)
