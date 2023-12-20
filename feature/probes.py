from base import Base

import numpy as np
from collections import defaultdict
import h5py

from pyfr.readers.native import NativeReader
from pyfr.quadrules import get_quadrule
from pyfr.util import subclasses, subclass_where
from pyfr.shapes import BaseShape
from pyfr.mpiutil import get_comm_rank_root
from pyfr.polys import get_polybasis



import time

class Probes(Base):
    name = None

    qrule_map_gll = {
        'quad': 'gauss-legendre-lobatto',
        'tri': 'alpha-opt',
        'hex': 'gauss-legendre-lobatto',
        'pri': 'alpha-opt~gauss-legendre-lobatto',
        'pyr': 'gauss-legendre-lobatto',
        'tet': 'alpha-opt'
    }

    def __init__(self, argv, icfg, fname):
        super().__init__(argv)
        self._argproc(icfg, fname)
        self.dty = np.float64

    def _argproc(self, icfg, fname):
        self.exactloc = icfg.getbool(fname, 'exactloc', True)
        # List of points to be sampled and format
        self.fmt = icfg.get(fname, 'format', 'primitive')
        self.mode = icfg.get(fname, 'mode', 'mesh')
        self.porder = icfg.getint(fname, 'porder', self.order)
        self.nmsh_dir = icfg.get(fname, 'new-mesh-dir', None)

        if not self.nmsh_dir:
            raise ValueError('Directory and file name is needed for the target mesh')

        if self.exactloc == False:
            raise NotImplementedError('not finished yet, use exact location instead.')

        print('mode: ', self.mode)
        print('format: ', self.fmt)
        print('poly order: ', self.porder)


    def preprocpts_elewise(self):
        # Load new mesh and interpolate to solution points
        f = h5py.File(self.nmsh_dir, 'r')
        self.kshp, mesh, nele = [], [], []
        comm, rank, root = get_comm_rank_root()

        # Collect data
        for k,v in f.items():
            if 'spt' in k.split('_'):
                if f'p{rank}' == k.split('_')[-1]:
                    dim = v.shape[-1]
                    # Raise mesh to the aimed polynomial order and interpolate it to upts
                    etype = k.split('_')[1]
                    nspts = v.shape[0]
                    nupts = self._get_npts(etype, self.porder+1)

                    # Quadrature rule is consistent with original setups
                    quadr = self.cfg.get(f'solver-elements-{etype}', 'soln-pts')
                    upts = get_quadrule(etype, quadr, nupts).pts
                    mesh_op = self._get_mesh_op_qr(etype, nspts, upts)
                    vv = np.einsum('ij, jkl -> ikl',mesh_op,v)

                    # Interpolation is done in pointwise operations
                    try:
                        self.pts = np.append(vv.reshape(-1, dim), self.pts, axis = 0)
                    except:
                        self.pts = vv.reshape(-1, dim)
                    self.kshp.append((k, vv.shape))

        f.close()
        print('No. of mpi rank: ', rank, ', Number of points', len(self.pts))

    def preprocpts_ptswise(self):
        comm, rank, root = get_comm_rank_root()
        if rank == 0:
            f = h5py.File(f'{self.nmsh_dir}','r')
            pts = np.array(f['mesh'])
            f.close()

        else:
            pts = None

        # Boardcast mm array
        pts = comm.bcast(pts, root=0)

        size = comm.Get_size()
        # Let each rank get a batch of points
        npts_rank = len(pts) // size
        if rank == size - 1:
            self.pts = pts[rank*npts_rank:]
        else:
            self.pts = pts[rank*npts_rank:(rank+1)*npts_rank]

        self.kshp = list([('spt_pts_p0', (len(pts),2,self.ndims))])
        print('No. of mpi rank: ', rank, ', Number of points', len(self.pts))


    def main_proc(self):
        # Prepare for MPI process
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if self.mode == 'mesh':
            # Allocation of points to mpi ranks
            if 'pyfrm' in self.nmsh_dir.split('.'):
                self.preprocpts_elewise()
            else:
                self.preprocpts_ptswise()

            # Get bounding boxes and process points
            lookup, gbox, lbox = self.preprocmesh()
            self.procpts(lookup, gbox, lbox)
            return 0

        else:
            if 'pyfrm' in self.nmsh_dir.split('.'):
                # Load interpolation info
                ptsinfo, lookup = self.load_ptsinfo()

                # Process information
                eidinfo, intopinfo, soln_op, plocs, pide = self.procinf(ptsinfo, lookup)


                for t in self.time:
                    print(rank, t)
                    self.procsoln_elewise(t, eidinfo, intopinfo, lookup,
                                                        soln_op, plocs, pide)
            else:
                if self.method == 'Standard':
                    # Get time series for each rank
                    time = self.get_time_series_mpi(rank, size)
                    print(rank, time)
                    if len(time) == 0:
                        comm.Abort(0)


                    # Load interpolation info
                    ptsinfo, lookup = self.load_ptsinfo_ptswise()

                    # Process information
                    eidinfo, intopinfo, soln_op, plocs, pide = self.procinf(ptsinfo, lookup)

                    if rank == 0:
                        self.dump_to_h5_ptswise(plocs)

                    osoln = {}
                    for t in time:
                        self.procsoln_ptswise(t, eidinfo, intopinfo,
                                            lookup, soln_op, plocs, pide)
                    #self.dump_to_h5_ptswise(osoln, time)
                elif self.method == 'lowRAM':
                    # Load interpolation info
                    ptsinfo, lookup = self.load_ptsinfo_ptswise_serial_time()

                    # Process information
                    eidinfo, intopinfo, soln_op, plocs, pide = self.procinf(ptsinfo, lookup)

                    self.dump_to_h5_ptswise_serial_time(np.array(plocs), 'mesh')

                    # Do some sortings to make sure mesh and solution are paried.
                    pp = []
                    for erank, pid in pide.items():
                        pp += pid
                    osoln = {}
                    for t in self.time:
                        if rank == 0:
                            print(t, flush = 'true')
                        self.procsoln_ptswise_serial_time(t, eidinfo, intopinfo,
                                            lookup, soln_op, plocs, pide, pp)


    def procsoln_ptswise_serial_time(self, time, eidinfo, intops, lookup, soln_op, plocs, pide, pp):
        soln = f'{self.solndir}{time}.pyfrs'
        soln = self._load_snapshot(soln, eidinfo, lookup, soln_op)

        # Do interpolations
        sln = [intops[k][id] @ soln[k][...,id] for k in intops
                                            for id in range(len(intops[k]))]

        sln = [(pid, ss) for pid, ss in zip(pp, sln)]
        sln.sort()
        sln = [ss for pid, ss in sln]

        # Make it primitive varibles
        if self.fmt == 'primitive':
            sln = np.array(sln).swapaxes(0,-1)
            sln = np.array(self._con_to_pri(sln)).swapaxes(0,-1)

        self.dump_to_h5_ptswise_serial_time(sln, time)


    def procsoln_ptswise(self, time, eidinfo, intops, lookup, soln_op, plocs, pide):
        soln = f'{self.solndir}{time}.pyfrs'
        soln = self._load_snapshot(soln, eidinfo, lookup, soln_op)

        # Do interpolations
        sln = [intops[k][id] @ soln[k][...,id] for k in intops
                                            for id in range(len(intops[k]))]

        # Do some sortings to make sure mesh and solution are paried.
        pp = []
        for erank, pid in pide.items():
            pp += pid

        sln = [(pid, ss) for pid, ss in zip(pp, sln)]
        sln.sort()
        sln = [ss for pid, ss in sln]

        # Make it primitive varibles
        if self.fmt == 'primitive':
            sln = np.array(sln).swapaxes(0,-1)
            sln = np.array(self._con_to_pri(sln)).swapaxes(0,-1)

        self.dump_to_h5_ptswise(sln, time)

    def procsoln_elewise(self, time, eidinfo, intops, lookup, soln_op, plocs, pide):
        soln = f'{self.solndir}{time}.pyfrs'
        soln = self._load_snapshot(soln, eidinfo, lookup, soln_op)

        # Do interpolations
        sln = [intops[k][id] @ soln[k][...,id] for k in intops
                                            for id in range(len(intops[k]))]

        # Do some sortings to make sure mesh and solution are paried.
        pp = []
        for erank, pid in pide.items():
            pp += pid

        sln = [(pid, ss) for pid, ss in zip(pp, sln)]
        sln.sort()
        sln = [ss for pid, ss in sln]


        # Reshape to original shape
        tNpt, soln = 0, {}
        for k, shp in self.kshp:
            Npt = shp[0] * shp[1]
            soln[k] = np.array(sln)[tNpt:tNpt+Npt].reshape(shp[0], shp[1], self.nvars)
            tNpt += Npt

        # Flash to disk
        self.dump_to_h5_elewise(soln, time)


    def _load_snapshot(self, name, eidinfo, lookup, soln_op):
        soln = defaultdict(list)
        f = h5py.File(name,'r')
        for k in soln_op:
            etype,part = lookup[k].split('_')[1:]
            key = f'{self.dataprefix}_{etype}_{part}'
            sln = np.array(f[key])[...,eidinfo[k]]
            sln = np.einsum('ij, jkl -> ikl',soln_op[k],sln)
            soln[k].append(sln)
        f.close()
        return {k: np.concatenate(v, axis = -1) for k,v in soln.items()}

    def dump_to_h5_ptswise(self, var, time = []):
        if len(time) == 0:
            f = h5py.File(f'{self.dir}/interp_mesh.pyfrs','w')
            f['mesh'] = var
            f.close()
        else:
            f = h5py.File(f'{self.dir}/interp_{time}.pyfrs','w')
            f[f'soln'] = var
            f.close()

        #raise RuntimeError

    def dump_to_h5_ptswise_serial_time(self, var, t):
        comm, rank, root = get_comm_rank_root()
        size = comm.Get_size()

        na = 'mesh' if t == 'mesh' else 'soln'

        # Try to use parallel HDF5
        try:
            f = h5py.File(f'{self.dir}/interp_{t}.pyfrs', 'w', driver='mpio', comm=comm)
            # Parallel HDF5 requires that data sets be created collectively
            f.create_dataset(na, (self.npts,var.shape[-1]), dtype = self.dty)
            fna = f[na]

            if rank == 0:
                fna[:self.ginfo[rank]] = var
            else:
                fna[self.ginfo[rank-1]:self.ginfo[rank]] = var
            comm.Barrier()
            f.close()
        except:
            print(var.shape, self.ginfo)
            for erank in range(size):
                if erank == rank:
                    if rank == 0:
                        f = h5py.File(f'{self.dir}/interp_{t}.pyfrs', 'w')
                        f.create_dataset(na, (self.npts,var.shape[-1]), dtype = self.dty)
                        fna = f[na]
                        fna[:self.ginfo[rank]] = var
                    else:
                        f = h5py.File(f'{self.dir}/interp_{t}.pyfrs', 'a')
                        fna = f[na]
                        fna[self.ginfo[rank-1]:self.ginfo[rank]] = var
                    f.close()
                comm.Barrier()

    def dump_to_h5_elewise(self, soln, time):
        # Get mpi info
        comm, rank, root = get_comm_rank_root()
        # Load new mesh informations
        if rank == 0:
            nmesh = NativeReader(self.nmsh_dir)

            f = h5py.File(f'{self.dir}/interp_{time}.pyfrs','w')
            f['mesh_uuid'] = nmesh['mesh_uuid']
            self.cfg.set('solver','order',self.porder)
            f['config'] = self.cfg.tostr()
            f['stats'] = self.stats.tostr()
            f.close()

        for i in range(comm.Get_size()):
            if i == rank:
                f = h5py.File(f'{self.dir}/interp_{time}.pyfrs','a')
                for k, v in soln.items():
                    prefix, etype, part = k.split('_')
                    f[f'soln_{etype}_{part}'] = v.swapaxes(1,-1)
                f.close()
            comm.Barrier()


    def _con_to_pri(self, cons):
        rho, E = cons[0], cons[-1]

        # Divide momentum components by rho
        vs = [rhov/rho for rhov in cons[1:-1]]

        # Compute the pressure
        gamma = self.cfg.getfloat('constants', 'gamma')
        p = (gamma - 1)*(E - 0.5*rho*sum(v*v for v in vs))

        return [rho] + vs + [p]


    def procinf(self, ptsinfo, lookup):
        """Performance improvement: Get pid into consideration in case
        resort is need in the interpolation stage"""
        # Process information
        eranks, eids, plocs, ntlocs = zip(*ptsinfo)
        pids = [*range(len(ntlocs))]
        eidinfo = defaultdict(list)
        intopinfo, pide = defaultdict(list), defaultdict(list)
        for pid, erank, eid, ntloc in zip(pids, eranks, eids, ntlocs):
            eidinfo[erank].append(eid)
            intopinfo[erank].append(ntloc)
            pide[erank].append(pid)

        intopinfo, soln_op = self._get_op_interp(intopinfo, lookup)
        return eidinfo, intopinfo, soln_op, plocs, pide


    def _get_op_interp(self, intopinfo, lookup):
        # Get basis class map
        basismap = {b.name: b for b in subclasses(BaseShape, just_leaf=True)}

        info, soln_op = {}, {}
        # Loop over all the points for each element type
        for erank, tlocs in intopinfo.items():

            _, etype, part = lookup[erank].split('_')

            npts = len(self.soln[f'soln_{etype}_{part}'])
            #quadr = self.cfg.get(f'solver-elements-{etype}', 'soln-pts')
            #upts = get_quadrule(etype, quadr, npts).pts
            upts = get_quadrule(etype, self.qrule_map_gll[etype], npts).pts

            ubasis = get_polybasis(etype, self.order + 1, upts)

            # Form the corresponding interpolation operators
            ops = ubasis.nodal_basis_at(tlocs).astype(self.dtype)

            info[erank] = ops

            basis = basismap[etype](len(upts), self.cfg)
            ops = basis.ubasis.nodal_basis_at(upts).astype(self.dtype)

            soln_op[erank] = ops
        return info, soln_op




    def load_ptsinfo(self):
        # Get mpi info
        comm, rank, root = get_comm_rank_root()
        f = h5py.File(f'{self.dir}/probes.m','r')

        lookup, ptsinfo = [], []
        for i in f:
            if i == f'{rank}':
                if self.ndims == 3:
                    for key, eid, ploc, ntloc in np.array(f[i])[['f0','f1','f2','f3']].astype('U14, i4, (3,)f8, (3,)f8'):
                        if key not in lookup:
                            lookup.append(key)
                        # Create a dictionary containing everything
                        ptsinfo.append((lookup.index(key),eid, ploc, ntloc))
                elif self.ndims == 2:
                    for key, eid, ploc, ntloc in np.array(f[i])[['f0','f1','f2','f3']].astype('U14, i4, (2,)f8, (2,)f8'):
                        if key not in lookup:
                            lookup.append(key)
                        # Create a dictionary containing everything
                        ptsinfo.append((lookup.index(key),eid, ploc, ntloc))
                self.kshp = np.array(f[f'{i}_info'])[['f0','f1']].astype('U14, (3,)i4').tolist()
        f.close()

        return ptsinfo, [str(key) for key in lookup]


    def load_ptsinfo_ptswise_serial_time(self):
        # Get mpi info
        comm, rank, root = get_comm_rank_root()
        f = h5py.File(f'{self.dir}/probes.m','r')

        lookup, ptsinfo, rrank = [], [], []
        npts = 0
        for i in f:
            if len(i.split('_')) == 1:
                rrank.append((int(i),i))
                npts += len(np.array(f[i]))
        rrank.sort()

        # Divide npts into each rank
        size = comm.Get_size()
        npts_rank = npts // size

        self.npts = npts
        ginfo = np.ones(size)*npts_rank
        ginfo[-1] = int(npts - npts_rank*(size - 1))
        self.ginfo = np.cumsum(ginfo).astype(int)

        # Load data to each rank
        id = 0
        for ii,i in rrank:
            if self.ndims == 3:
                for key, eid, ploc, ntloc in np.array(f[i])[['f0','f1','f2','f3']].astype('U14, i4, (3,)f8, (3,)f8'):
                    if id >= npts_rank * size and rank == size - 1:
                        if key not in lookup:
                            lookup.append(key)
                        # Create a dictionary containing everything
                        ptsinfo.append((lookup.index(key),eid, ploc, ntloc))
                    elif rank == id // npts_rank:
                        if key not in lookup:
                            lookup.append(key)
                        # Create a dictionary containing everything
                        ptsinfo.append((lookup.index(key),eid, ploc, ntloc))
                    id += 1
            elif self.ndims == 2:
                for key, eid, ploc, ntloc in np.array(f[i])[['f0','f1','f2','f3']].astype('U14, i4, (2,)f8, (2,)f8'):
                    if id >= npts_rank * size and rank == size - 1:
                        if key not in lookup:
                            lookup.append(key)
                        # Create a dictionary containing everything
                        ptsinfo.append((lookup.index(key),eid, ploc, ntloc))
                    elif rank == id // npts_rank:
                        if key not in lookup:
                            lookup.append(key)
                        # Create a dictionary containing everything
                        ptsinfo.append((lookup.index(key),eid, ploc, ntloc))
                    id += 1
            self.kshp = np.array(f[i])[['f0','f1']].astype('U14, (3,)i4').tolist()
        f.close()
        return ptsinfo, [str(key) for key in lookup]


    def load_ptsinfo_ptswise(self):
        # Get mpi info
        comm, rank, root = get_comm_rank_root()
        f = h5py.File(f'{self.dir}/probes.m','r')

        lookup, ptsinfo, rrank = [], [], []
        for i in f:
            if len(i.split('_')) == 1:
                rrank.append((int(i),i))
        rrank.sort()
        #print(rrank)
        for ii,i in rrank:
            print(i)
            if self.ndims == 3:
                for key, eid, ploc, ntloc in np.array(f[i])[['f0','f1','f2','f3']].astype('U14, i4, (3,)f8, (3,)f8'):
                    if key not in lookup:
                        lookup.append(key)
                    # Create a dictionary containing everything
                    ptsinfo.append((lookup.index(key),eid, ploc, ntloc))
            elif self.ndims == 2:
                for key, eid, ploc, ntloc in np.array(f[i])[['f0','f1','f2','f3']].astype('U14, i4, (2,)f8, (2,)f8'):
                    if key not in lookup:
                        lookup.append(key)
                    # Create a dictionary containing everything
                    ptsinfo.append((lookup.index(key),eid, ploc, ntloc))
            self.kshp = np.array(f[i])[['f0','f1']].astype('U14, (3,)i4').tolist()
        f.close()
        return ptsinfo, [str(key) for key in lookup]

    def preprocmesh(self):
        # In this section, bounding boxes will be created
        lookup, gbox, lbox = [], [], []
        for k, v in self.mesh.items():
            if 'spt' in k.split('_'):
                lookup.append(k)
                gbox.append([(np.min(v[...,i]),np.max(v[...,i])) for i in range(self.ndims)])
                lbox.append([(np.min(v[...,i], axis = 0),np.max(v[...,i], axis = 0)) for i in range(self.ndims)])

        return lookup, gbox, lbox

    def procpts(self, lookup, gbox, lbox):
        # Get mpi info
        comm, rank, root = get_comm_rank_root()

        # Get global location
        pid, gid = self._global_check(gbox)

        # Get local location
        eleinfo = self._local_check(pid, lbox, lookup)

        # Load corresponding mesh into memory
        ptsinfo, mesh_trans, lookup = self.data_process(eleinfo, lookup)

        # Input to algorithm to do finding the ownership
        ptsinfo = self._owner_check(ptsinfo, lookup, mesh_trans)

        # Newton iteration to find the exact reference coordinate locations
        ptsinfo = self._refine_pts(ptsinfo, lookup)

        """Performance improvement: Maybe not necessary?"""
        # Write interpolation information to disk
        self.dump_to_file(ptsinfo, lookup)



    def dump_to_file(self, ptsinfo, lookup):
        # Get mpi info
        comm, rank, root = get_comm_rank_root()
        pts = []
        for erank, eid, ploc, ntloc in ptsinfo:
            pts.append((lookup[erank], eid, ploc, ntloc))

        if self.ndims == 3:
            pts = np.array(pts, dtype='S14, i4, (3,)f8, (3,)f8')
        else:
            pts = np.array(pts, dtype='S14, i4, (2,)f8, (2,)f8')

        for i in range(comm.Get_size()):
            if i == rank:
                if i == 0:
                    f = h5py.File(f'{self.dir}/probes.m','w')
                else:
                    f = h5py.File(f'{self.dir}/probes.m','a')
                f[f'{rank}'] = pts
                f[f'{rank}_info'] = np.array(self.kshp, dtype='S14, (3,)i4')
                f.close()
            comm.barrier()

    def _refine_pts(self, info, lookup):
        # Get basis class map
        basismap = {b.name: b for b in subclasses(BaseShape, just_leaf=True)}

        ptsinfo = []
        # Loop over all the points for each element type
        for pid, erank, spts, tlocs, plocs, iplocs, eids in info:

            _, etype, part = lookup[erank].split('_')

            upts = get_quadrule(etype, self.qrule_map_gll[etype], len(spts)).pts

            ubasis = get_polybasis(etype, self.order + 1, upts)
            ntlocs, nplocs = self._plocs_to_tlocs(ubasis, spts, plocs,
                                                                tlocs, iplocs)

            # Append to the point info list
            ptsinfo.extend(
                (*info, erank) for info in zip(pid, eids, nplocs, ntlocs)
            )

        # Resort to the original index
        ptsinfo.sort()

        # Strip the index, move etype to the front, and return
        return [(erank, *info) for pid, *info, erank in ptsinfo]

    def _plocs_to_tlocs(self, ubasis, spts, plocs, itlocs, iplocs):
        # Set current tolerance
        tol = 1e-7#5e-7

        # Iterates
        kplocs, ktlocs = iplocs.copy(), itlocs.copy()

        # Output array
        oplocs, otlocs = iplocs.copy(), itlocs.copy()

        """Performance improvement: does not need to calculate full point list
        for each iteration."""
        indexp = [*range(len(kplocs))]
        # Apply maximum ten iterations of Newton's method
        for k in range(10):
            # Get Jacobian operators
            jac_ops = ubasis.jac_nodal_basis_at(ktlocs)
            # Solve from ploc to tloc
            kjplocs = np.einsum('ijk,jkl->kli', jac_ops, spts)
            ktlocs -= np.linalg.solve(kjplocs, kplocs - plocs)
            # Transform back to ploc
            ops = ubasis.nodal_basis_at(ktlocs)
            np.einsum('ij,jik->ik', ops, spts, out=kplocs)

            # Apply check routine
            kdists = np.linalg.norm(plocs - kplocs, axis=1)

            # Get the points satisfied criterion
            index = np.where(kdists < tol)[0]
            index = [id for id in index if id in indexp]
            indexp = list(set(indexp) - set(index))

            print('iteration number:', k, ', Num pts left:', len(indexp),
                                            ', max distance:', np.max(kdists))

            if len(index) != 0:
                oplocs[index], otlocs[index] = kplocs[index], ktlocs[index]
            if len(indexp) == 0:
                return otlocs, oplocs
            if k == 19:
                print(plocs[indexp])
                raise RuntimeError('Newton iteration does not converge to tol.')


    def _owner_check(self, pts, lookup, mesh_trans):
        # Get basis class map
        basismap = {b.name: b for b in subclasses(BaseShape, just_leaf=True)}

        ptsinfo, eleinfo, pidinfo = {}, defaultdict(list), defaultdict(list)
        ppeinfo = {}
        # Classify it with erank
        for pid, info in pts.items():
            p = self.pts[pid]
            for erank, e, eidx in info:

                # Compute the distances between each point and p
                dists = np.linalg.norm(e - p, axis=-1)
                for ide, dist in enumerate(dists.T):

                    # Get the index of the closest point to p for each element
                    idp = np.unravel_index(np.argmin(dist), dist.shape)
                    spts, tloc = e[:,ide], mesh_trans[erank][idp]

                    # Gather up
                    eleinfo[erank].append((spts, tloc, p, spts[idp]))
                    pidinfo[erank].append((pid, eidx[ide]))

            # Label points exist in multiple eranks
            if len(info) > 1:
                erank, *e =  zip(*info)
                ppeinfo[pid] = erank

        # Sort by element type to increase stability
        eidsort = {'hex': 0, 'pyr': 1, 'pri': 2, 'tet': 3, 'quad': 0, 'tri': 1}
        eidrank = [(eidsort[lookup[erank].split('_')[1]], erank) for erank in eleinfo]
        eidrank.sort()
        eleinfo = {erank: eleinfo[erank] for id, erank in eidrank}

        pt_temp = {}
        for erank, info in eleinfo.items():
            spts, tlocs, ipts, iplocs = zip(*info)

            spts, iplocs = np.array(spts).swapaxes(0,1), np.array(iplocs)
            ipts = np.array(ipts)

            # Relevent classes and basises
            etype = lookup[erank].split('_')[1]
            basis = basismap[etype](len(spts), self.cfg)

            # Apply newton check to cut down possible owner elements.
            inner, dp = self._newton_check(spts, tlocs, ipts, iplocs, basis)

            # Optimized approach for selecting point IDs
            pid, eid = zip(*pidinfo[erank])
            if basis.name in ['hex', 'quad']:
                tol = 1e-10
            else:
                tol = 1e-8
            idi, pinfo = {}, {}

            for id, (p, val, d) in enumerate(zip(pid, inner, dp)):
                if p not in pinfo:
                    pinfo[p] = {'tol': val < tol, 'dp': d}
                    idi[p] = id
                else:
                    # Update if tolerence is meeted
                    if pinfo[p]['tol'] != val < tol:
                        pinfo[p]['tol'] = True
                        pinfo[p]['dp'] = d
                        idi[p] = id

                    # Update the minimum physical distance
                    if pinfo[p]['tol']:
                        if val < tol and d < pinfo[p]['dp']:
                            pinfo[p]['dp'] = d
                            idi[p] = id
                    elif d < pinfo[p]['dp']:
                        pinfo[p]['dp'] = d
                        idi[p] = id

            idi = np.array([v for k, v in idi.items()])

            pid = np.array(pid)
            ptsinfo[erank] = (pid[idi], idi)
            pt_temp[erank] = {k: (v['dp'], v['tol']) for k, v in pinfo.items()}

        # Determine which point at erank boundaries to keep
        erdel = defaultdict(list)
        for p, er in ppeinfo.items():
            pinfo = []
            for e in er:
                info = pt_temp[e][p]
                pinfo.append((*info, e))
            pinfo.sort()
            index = np.arange(len(pinfo))
            dp, fl, e = zip(*pinfo)

            try:
                idx = fl.index(True)
                index = np.delete(index, idx)
            except ValueError:
                index = index[1:]

            for ee in index:
                erdel[e[ee]].append(p)


        for erank, p in erdel.items():
            info = ptsinfo[erank]
            index = [id for id, pp in enumerate(info[0]) if pp not in p]
            ptsinfo[erank] = (info[0][index], info[1][index])

        pinfo = []
        for erank, inf in ptsinfo.items():
            if len(inf) > 0:
                pid, idi = inf
                spts, tlocs, ipts, iplocs = zip(*eleinfo[erank])
                spts, iplocs = np.array(spts).swapaxes(0,1), np.array(iplocs)
                ipts = np.array(ipts)

                _, eid = zip(*pidinfo[erank])
                pinfo.append((pid, erank, spts[:,idi], np.array(tlocs)[idi],
                                ipts[idi], iplocs[idi], np.array(eid)[idi]))
        return pinfo

    def _newton_check(self, spts, tlocs, pts, iplocs, basis):
        # Get gll points quadrature rule
        etype = basis.name
        upts = get_quadrule(etype, self.qrule_map_gll[etype], len(spts)).pts
        sbasis = get_polybasis(etype, self.order + 1, upts)

        ktlocs, kplocs = np.array(tlocs).copy(), np.array(iplocs).copy()

        if etype in ['hex', 'quad']:
            nit = 1
        else:
            nit = 2
        for n in range(nit):
            # Get Jacobian operators
            jac_ops = sbasis.jac_nodal_basis_at(ktlocs)

            # Solve from ploc to tloc
            kjplocs = np.einsum('ijk,jkl->kli', jac_ops, spts)
            ktlocs -= np.linalg.solve(kjplocs, kplocs - pts)

            ops = sbasis.nodal_basis_at(ktlocs)
            np.einsum('ij,jik->ik', ops, spts, out=kplocs)

        # Apply check routine: all points out of bounding box will be thrown away
        return self.std_ele_box(ktlocs, basis, pts - kplocs)

    def std_ele_box(self, tlocs, basis, dp):
        if basis.name in ['hex', 'quad']:
            # For boundary points, this should be relaxed
            tol = 1e-10
        else:
            tol = 1e-8
        # Get a point on the surface
        cls = subclass_where(Probes, name=basis.name)
        fc, norm = cls.face_center()

        tlocs = tlocs[:,None,:] - fc
        tlocs = tlocs/np.linalg.norm(tlocs, axis = -1)[:,:,None]
        inner = np.einsum('ijk,jk -> ij', tlocs, np.array(norm))

        return np.max(inner, axis = 1), np.max(abs(dp), axis = 1)

    def _local_check(self, pid, lbox, lookup):
        # Get the local bounding box for each point
        eleinfo = defaultdict(list)
        for id, gid in pid.items():
            for gidx in gid:
                llbox = lbox[gidx]
                index = np.arange(len(llbox[0][0]))
                for did, (amin, amax) in enumerate(llbox):
                    idx = np.argsort(amin[index])
                    idxm = np.searchsorted(amin[index], self.pts[id,did], sorter = idx, side = 'right')
                    idx = idx[:idxm]
                    idy = np.argsort(amax[index[idx]])
                    idxm = np.searchsorted(amax[index[idx]], self.pts[id,did], sorter = idy, side = 'left')
                    idy = idy[idxm:]
                    index = index[idx[idy]]

                    if len(index) == 0:
                        break

                if len(index) != 0:
                    ptsinfo = id
                    eleinfo[gidx].append((id,index))

            # Check if all points are in local boundaing boxes
            if id != ptsinfo:
                print('Failed to find point %f', self.pts[id])
                raise RuntimeError('Failed to find point %f', self.pts[id])

        return eleinfo



    def _global_check(self, gbox):
        # Get the global bounding box for each point
        """Find a way to make it more efficient"""
        pid, gid = defaultdict(list), []
        if self.ndims == 3:
            for id, p in enumerate(self.pts):
                for idg, rg in enumerate(gbox):
                    if p[0] >= rg[0][0] and p[0] <= rg[0][1]:
                        if p[1] >= rg[1][0] and p[1] <= rg[1][1]:
                            if p[2] >= rg[2][0] and p[2] <= rg[2][1]:
                                pid[id].append(idg)
                                if idg not in gid:
                                    gid.append(idg)
        else:
            for id, p in enumerate(self.pts):
                for idg, rg in enumerate(gbox):
                    if p[0] >= rg[0][0] and p[0] <= rg[0][1]:
                        if p[1] >= rg[1][0] and p[1] <= rg[1][1]:
                            pid[id].append(idg)
                            if idg not in gid:
                                gid.append(idg)
        return pid, gid


    def data_process(self, eleinfo, lookup, op='bg'):
        # Process original mesh to gll points in plocs and tlocs
        mesh, mesh_trans, lookup_update = [], [], []
        for erank, index in eleinfo.items():
            key = lookup[erank]
            _, etype, part = key.split('_')

            # Get Operators
            soln_name = f'{self.dataprefix}_{etype}_{part}'
            nupts = self.soln[soln_name].shape[0]
            nspts = self.mesh[key].shape[0]
            upts = get_quadrule(etype, self.qrule_map_gll[etype], nupts).pts
            mesh_op = self._get_ops_interp(nspts, etype, upts, nupts, self.order)

            # Lift to high order and project to gll points
            mesh_temp = np.einsum('ij, jkl -> ikl',mesh_op,self.mesh[key])
            mesh.append([(id, mesh_temp[:,idx], idx) for id, idx in index])
            # Collect quadrature points
            mesh_trans.append(upts)
            # Update lookup dictionary
            lookup_update.append(lookup[erank])


        # Form a new dictionary with pids
        ptsinfo = defaultdict(list)
        for erankid, info in enumerate(mesh):
            for id, msh, idx in info:
                ptsinfo[id].append((erankid, msh, idx))

        return ptsinfo, mesh_trans, lookup_update

class HexShape(Probes):
    name = 'hex'
    def face_center():
        fc = np.array([[0,0,-1],[0,-1,0],[1,0,0],[0,1,0],[-1,0,0],[0,0,1]])
        norm = np.array([[0,0,-1],[0,-1,0],[1,0,0],[0,1,0],[-1,0,0],[0,0,1]])
        return fc, norm

class QuadShape(Probes):
    name = 'quad'
    def face_center():
        fc = np.array([[0,-1],[1,0],[0,1],[-1,0]])
        norm = np.array([[0,-1],[1,0],[0,1],[-1,0]])
        return fc, norm

class TetShape(Probes):
    name = 'tet'
    def face_center():
        fc = np.array([[-1,-1,-1],[1,-1,-1],[-1,-1,-1],[-1,-1,1]])
        norm = np.array([[0,0,-1],[0,-1,0],[-1,0,0], \
                        [ 0.57735027,0.57735027,0.57735027]])
        return fc, norm

class TriShape(Probes):
    name = 'tri'
    def face_center():
        fc = np.array([[0,-1],[1,-1],[-1,0]])
        norm = np.array([[0,-1],[0.70710678,-0.70710678,0],[-1,0]])
        return fc, norm

class PyrShape(Probes):
    name = 'pyr'
    def face_center():
        fc = np.array([[0,0,-1],[1,-1,-1],[1,-1,-1],[-1,1,-1],[-1,1,-1]])
        norm = np.array([[0,0,-1],[ 0,-0.89442719,0.4472136], \
                        [0.89442719,0,0.4472136],[0,0.89442719,0.4472136], \
                        [-0.89442719,0,0.4472136]])
        return fc, norm

class PriShape(Probes):
    name = 'pri'
    def face_center():
        fc = np.array([[-1,-1,-1],[-1,-1,1],[0,-1,0],[1,-1,-1],[-1,0,0]])
        norm = np.array([[0,0,-1],[0,0,1],[0,-1,0],[0.70710678,0.70710678,0],\
                        [-1,0,0]])
        return fc, norm
