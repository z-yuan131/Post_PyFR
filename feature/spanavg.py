from feature.region import Region

import numpy as np
from collections import defaultdict
import h5py

from pyfr.readers.native import NativeReader
from pyfr.quadrules import get_quadrule
from pyfr.mpiutil import get_comm_rank_root


class SpanavgBase(Region):
    def __init__(self, argv, icfg, fname):
        super().__init__(argv, icfg, fname)
        self.mode = icfg.get(fname, 'mode', 'mesh')
        self.method = icfg.get(fname, 'method', 'Standard')
        self.tol = icfg.getfloat(fname, 'tol', 1e-6)
        self.outfreq = icfg.getint(fname, 'outfreq', 1)
        nfft = icfg.getint(fname, 'nfft', 10)
        self._linmap = lambda n: {'hex': np.array([0, n-1, n**2-n, n**2-1,
                                n**2*(n-1), (n-1)*(n**2+1), n**3-n, n**3-1])}

        # Get indices of results we need
        if nfft <= 0:
            raise RuntimeError('nfft has to be equal or larger than 1.')
        self.nfft = np.arange(nfft)

        # The box that elements are in
        self.box = icfg.getliteral(fname, 'box', None)

        self.emap = lambda x: {(x+1)**2: 'quad', (x+1)*(x+2)/2: 'tri'}

    def _get_box(self):
        if self.box and self.layers > 0:
            import warnings
            warnings.warn(f'Use the box region instead of layers '+
                                          f'from the bc {self.boundary_name}.')

        mesh, lookup = [], []
        # Get all elements inside the box that to be averaged
        for k, v in self.mesh.items():
            if 'spt' in k.split('_'):
                _, etype , part = k.split('_')
                if etype in self.suffix:
                    if self.box:
                        amin, amax = np.min(v, axis = 0),np.max(v, axis = 0)
                        idxmin = np.where(amax[:,0] > self.box[0][0])[0]
                        idxmax = np.where(amin[idxmin,0] < self.box[1][0])[0]
                        index = idxmin[idxmax]

                        idxmin = np.where(amax[index,1] > self.box[0][1])[0]
                        idxmax = np.where(amin[index[idxmin],1] < self.box[1][1])[0]
                        index = index[idxmin[idxmax]]

                        if len(index) > 0:
                            # Raise to current polynomial order
                            mesh_op = self._get_mesh_op(etype, len(v))
                            mesh.append(np.einsum('ij, jkl -> ikl', mesh_op, v[:,index]))
                            lookup.append((k,index))
                    else:
                        _, mesh_wall = self.get_wall_O_grid()
                        for k, index in mesh_wall.items():
                            prefix, etype, _ = k.split('_')

                            # Raise to current polynomial order
                            v = self.mesh[k][:,index]
                            mesh_op = self._get_mesh_op(etype, len(v))
                            mesh.append(np.einsum('ij, jkl -> ikl', mesh_op, v))
                            lookup.append((k,index))
        return mesh, lookup

    def main_proc(self):

        if self.mode == 'mesh':
            mesh, lookup = self._get_box()

            gbox, lbox = self._gen_bounding_box(mesh)

            cmesh = self._get_center(mesh, lookup)

            ptsinfo = self._resort_mesh(mesh, gbox, lbox, cmesh)

            self._cachedfile(ptsinfo, lookup)
            return 0
        else:
            # Get mpi info
            comm, rank, root = get_comm_rank_root()
            size = comm.Get_size()

            if self.method == 'Standard':
                ptsinfo, lookup = self._cachedfile()

                if rank == 0:
                    mesh = self._load_mesh(lookup)
                    if np.max(self.nfft) == 1:
                        spts = self._avg_proc(ptsinfo, mesh, self.ndims)
                    else:
                        spts, index = self._fft_proc(ptsinfo, mesh, self.ndims)

                    self._flash_to_disk(spts, 'mesh')
                    del mesh, spts
                else:
                    index = None

                if np.max(self.nfft) != 1:
                    index = comm.bcast(index, root = 0)

                soln_op = self._get_op_soln(lookup)
                # Get time series for each rank
                time = self.get_time_series_mpi(rank, size)
                for t in time:
                    soln = self._load_soln(t, lookup, soln_op)
                    if np.max(self.nfft) == 1:
                        # Average subroutine
                        soln = self._avg_proc(ptsinfo, soln, self.nvars - 1)
                    else:
                        # FFT subroutine
                        soln, _ = self._fft_proc(ptsinfo, soln, self.nvars - 1, index)
                        # Average boundary surface and do FFT
                        soln = self.ebcavgfft(soln)

                    self._flash_to_disk(soln, t)

            elif self.method == 'lowRAM':
                ptsinfo, lookup = self._cachedfile_parallel()

                mesh = self._load_mesh(lookup)

                if np.max(self.nfft) == 1:
                    spts = self._avg_proc(ptsinfo, mesh, self.ndims)
                else:
                    spts, index = self._fft_proc(ptsinfo, mesh, self.ndims)

                self._get_parallel_info(spts)
                self._get_array_shape()
                self._flash_to_disk_parallel(spts, 'mesh')
                del mesh, spts

                soln_op = self._get_op_soln(lookup)
                for t in self.time:
                    soln = self._load_soln(t, lookup, soln_op)
                    if np.max(self.nfft) == 1:
                        # Average subroutine
                        soln = self._avg_proc(ptsinfo, soln, self.nvars - 1)
                    else:
                        # FFT subroutine
                        soln, _ = self._fft_proc(ptsinfo, soln, self.nvars - 1, index)
                        # Average boundary surface and do FFT
                        soln = self.ebcavgfft(soln)

                        if t == self.time[0]:
                            self._get_parallel_info(soln)
                            self._get_array_shape(t)

                    self._flash_to_disk_parallel(soln, t)

    def _cachedfile(self, ptsinfo = [], lookup = []):
        if lookup != []:
            f = h5py.File(f'{self.dir}/spanavg.m','w')
            for id, einfo in ptsinfo.items():
                for erank, idx, ids in einfo:
                    f[f'{id}/{erank}/idx'] = idx
                    f[f'{id}/{erank}/ids'] = ids
            for erank, k in enumerate(lookup):
                f[f'{k[0]}_{erank}'] = k[1]
            f.close()
        else:
            ptsinfo, lookup = defaultdict(list), []
            f = h5py.File(f'{self.dir}/spanavg.m','r')
            for id in f:
                if 'spt' in id.split('_'):
                    prefix, etype, part, erank = id.split('_')
                    lookup.append((int(erank), f'{prefix}_{etype}_{part}', np.array(f[id])))
                else:
                    for erank in f[id]:
                        idx = np.array(f[f'{id}/{erank}/idx'])
                        ids = np.array(f[f'{id}/{erank}/ids'])
                        ptsinfo[id].append((int(erank), idx, ids))
            f.close()

            lookup.sort()
            return ptsinfo, [(key, idx) for erank, key, idx in lookup]

    def _cachedfile_parallel(self):
        # Get MPI info
        comm, rank, root = get_comm_rank_root()
        size = comm.Get_size()

        ptsinfo, lookup = defaultdict(list), []
        f = h5py.File(f'{self.dir}/spanavg.m','r')
        for id in f:
            if 'spt' in id.split('_'):
                prefix, etype, part, erank = id.split('_')
                lookup.append((int(erank), f'{prefix}_{etype}_{part}', np.array(f[id])))
            else:
                for erank in f[id]:
                    idx = np.array(f[f'{id}/{erank}/idx'])
                    ids = np.array(f[f'{id}/{erank}/ids'])
                    ptsinfo[id].append((int(erank), idx, ids))

        f.close()
        lookup.sort()

        # Let each rank get a batch of elements to process
        npts_rank = len(ptsinfo) // size
        if rank == size - 1:
            rpts = np.arange(rank * npts_rank, len(ptsinfo))
        else:
            rpts = np.arange(rank * npts_rank, (rank + 1) * npts_rank)

        ptsinfo = {int(k): v for k, v in ptsinfo.items() if int(k) in rpts}

        # Get real indices of elements within the MPI rank
        info = defaultdict(list)
        for k,v in ptsinfo.items():
            for er, ix, _ in v:
                info[er].append(ix)
        info = {er: np.unique(np.concatenate(ix)) for er, ix in info.items()}

        # Update lookup dictionary
        lookup2 = [(erank, key, idx[info[erank]]) for erank, key, idx in lookup if erank in info]

        # Update erank infomation
        eranks = [erank for erank, *v in lookup2]

        # Update index array in ptsinfo
        info = defaultdict(list)
        for id, v in ptsinfo.items():
            for erank, idx, _ in v:
                eridx = np.searchsorted(eranks, erank)

                # Map indices from old lookup table to the new one
                idxn = np.searchsorted(lookup2[eridx][2], lookup[erank][2][idx])
                info[id].append((eridx, idxn, _))

        # Format lookup and return
        return info, [v for erank, *v in lookup2]

    def _get_parallel_info(self, array):
        comm, rank, root = get_comm_rank_root()

        info = [(etype, soln.shape) for etype, soln in array.items()]

        # Distribute the data info to all ranks
        ginfo = comm.allgather(info)
        minfo = defaultdict(list)
        for info in ginfo:
            for etype, shape in info:
                minfo[etype].append(shape[1])

        self.ginfo = {k: np.cumsum(v) for k, v in minfo.items()}

    def _get_array_shape(self, t = []):
        n = self.order + 1
        npts = {'quad': n**2, 'tri': int(n*(n+1)/2)}

        self.shape = {}
        for k in self.ginfo:
            if t:
                self.shape[k] = (len(self.nfft), self.ginfo[k][-1], npts[k], self.nvars-1)
            else:
                self.shape[k] = (npts[k], self.ginfo[k][-1], self.ndims)

        if np.max(self.nfft) == 1 or not t:
            self.dty = np.float64
        else:
            self.dty = np.complex128

    def _flash_to_disk(self, array, t):
        f = h5py.File(f'{self.dir}/spanproc_{t}.s', 'w')
        for etype, soln in array.items():
            f[f'{etype}'] = soln
        f.close()

    def _flash_to_disk_parallel(self, array, t):
        comm, rank, root = get_comm_rank_root()
        size = comm.Get_size()

        # Try to use parallel HDF5
        try:
            f = h5py.File(f'{self.dir}/spanproc_{t}.s', 'w', driver='mpio', comm=comm)
            # Parallel HDF5 requires that data sets be created collectively
            for name in self.ginfo:
                f.create_dataset(name, self.shape[name], dtype = self.dty)

            for etype, soln in array.items():
                fna = f[etype]
                if rank == 0:
                    fna[:,:self.ginfo[etype][rank]] = soln
                else:
                    fna[:,self.ginfo[etype][rank-1]:self.ginfo[etype][rank]] = soln
            comm.Barrier()
            f.close()
        except:
            for erank in range(size):
                if erank == rank:
                    if rank == 0:
                        f = h5py.File(f'{self.dir}/spanproc_{t}.s', 'w')
                        for name in self.ginfo:
                            f.create_dataset(name, self.shape[name], dtype = self.dty)
                    else:
                        f = h5py.File(f'{self.dir}/spanproc_{t}.s', 'a')

                    for etype, soln in array.items():
                        fna = f[etype]
                        if rank == 0:
                            fna[:,:self.ginfo[etype][rank]] = soln
                        else:
                            fna[:,self.ginfo[etype][rank-1]:self.ginfo[etype][rank]] = soln
                    f.close()
                comm.Barrier()

    def _get_op_soln(self, lookup):
        soln_op = {}
        for k, idx in lookup:
            _, etype, part = k.split('_')
            # Operator
            if etype not in soln_op:
                name = f'{self.dataprefix}_{etype}_{part}'
                nspts = self.soln[name].shape[0]
                soln_op[etype] = self._get_soln_op(etype, nspts)
        return soln_op

    def _con_to_pri(self, cons):
        rho, E = cons[0], cons[-1]

        # Divide momentum components by rho
        vs = [rhov/rho for rhov in cons[1:-1]]

        # Compute the pressure
        gamma = self.cfg.getfloat('constants', 'gamma')
        p = (gamma - 1)*(E - 0.5*rho*sum(v*v for v in vs))

        return [rho] + vs + [p]


    def _load_soln(self, time, lookup, soln_op):
        soln = []
        f = h5py.File(f'{self.solndir}{time}.pyfrs', 'r')
        for k, idx in lookup:
            _, etype, part = k.split('_')
            kk = f'{self.dataprefix}_{etype}_{part}'

            sln = np.array(f[kk])[...,idx]
            sln = np.einsum('ij, jkl -> ilk', soln_op[etype], sln[:,[0,1,2,4]])
            sln = np.array(self._con_to_pri(sln.swapaxes(0,-1))).swapaxes(0,-1)
            soln.append(sln)
        f.close()
        return soln

    def _load_mesh(self, lookup):
        mesh = []
        for k, idx in lookup:
            msh = self.mesh[k][:,idx]
            etype = k.split('_')[1]
            mesh_op = self._get_mesh_op(etype, len(msh))
            msh = np.einsum('ij, jkl -> ikl', mesh_op, msh)
            mesh.append(msh)
        return mesh


    def _fft_proc(self, ptsinfo, arr, dim, index = {}):
        spts = defaultdict(list)
        for id, einfo in ptsinfo.items():
            pts = []
            for erank, idx, ids in einfo:
                # Idx sorted out all elements respect to this reference surf
                var = arr[erank][:,idx].reshape(-1,dim)
                Nptz = len(idx)*(self.order + 1)
                # Ids sorted out all pts in z direction respect to the reference surf
                var = var[ids].reshape(Nptz, -1, dim, order = 'F')
                pts.append(var)

            # Unlike average proc, fft proc needs to deal with elements boundary surfaces
            pts = np.concatenate(pts, axis = 0)

            # If dealing with mesh, get the ordering index
            if dim == self.ndims:
                index[id] = np.argsort(pts[:,0,-1])
                # Just average mesh
                pts = np.mean(pts, axis = 0)
                etype_2d = self.emap(self.order).get(len(pts))
                spts[etype_2d].append(pts)

            else:
                pts = pts[index[id]]

                etype_2d = self.emap(self.order).get(pts.shape[1])
                # In case the number of element is different in z direction
                spts[f'{etype_2d}_{len(pts)}'].append(pts)

        return {k: np.array(v).swapaxes(0,1) for k,v in spts.items()}, index

    def ebcavgfft(self, pts):
        pinfo = defaultdict(list)
        Ndz = len(pts) - 1
        d1 = np.arange(self.order+1,Ndz,self.order+1)
        d2 = np.arange(self.order+2,Ndz,self.order+1)

        for k, v in pts.items():
            sl = []
            # Average elements boundaries
            for id, pt in enumerate(v):
                if id in d1:
                    pinfo.append(0.5*(pt + pts[id+1]))
                elif id in d2:
                    continue
                else:
                    sl.append(pt)
            # Do FFT
            sl = np.fft.fft(np.array(sl), axis = 0)[self.nfft] / len(sl)

            kk = k.split('_')[0]
            if kk in pinfo:
                pinfo[kk] = np.append(pinfo[kk], sl, axis = 1)
            else:
                pinfo[kk] = sl
        return pinfo

    def _avg_proc(self, ptsinfo, arr, dim):
        spts = defaultdict(list)
        for id, einfo in ptsinfo.items():
            pts = []
            for erank, idx, ids in einfo:
                var = arr[erank][:,idx].reshape(-1,dim)
                Nptz = len(idx)*(self.order + 1)
                var = var[ids].reshape(Nptz, -1, dim, order = 'F')
                pts.append(var)
            pts = np.mean(np.concatenate(pts, axis = 0), axis = 0)
            etype_2d = self.emap(self.order).get(len(pts))
            spts[etype_2d].append(pts)

        return {k: np.array(v).swapaxes(0,1) for k,v in spts.items()}

    def _avg_mesh(self, ptsinfo, mesh):
        spts = defaultdict(list)
        for etype, einfo in enumerate(ptsinfo):
            for id, info in einfo.items():
                ele = []
                for inf in info:
                    pt = []
                    for erank, idx, ids in inf:
                        msh = mesh[erank][:,idx].reshape(-1,self.ndims)
                        pt.append(msh[ids])
                    #print(len(np.concatenate(pt, axis = 0)))
                    pt = np.mean(np.concatenate(pt, axis = 0), axis = 0)
                    ele.append(pt)
                spts[etype].append(np.array(ele))

        return [np.array(v).swapaxes(0,1) for k,v in spts.items()]

    def _avg_soln(self, ptsinfo, soln):
        spts = defaultdict(list)
        for etype, einfo in enumerate(ptsinfo):
            for id, info in einfo.items():
                ele = []
                for inf in info:
                    pt = []
                    for erank, idx, ids in inf:
                        msh = soln[erank][:,idx].reshape(-1,self.nvars)
                        pt.append(msh[ids])
                    #print(len(np.concatenate(pt, axis = 0)))
                    pt = np.mean(np.concatenate(pt, axis = 0), axis = 0)
                    ele.append(pt)
                spts[etype].append(np.array(ele))

        return [np.array(v).swapaxes(0,1) for k,v in spts.items()]


    def _gen_bounding_box(self, mesh):
        # In this section, bounding boxes will be created
        gbox, lbox = [], []
        for v in mesh:
            gbox.append([(np.min(v[...,i])-self.tol,np.max(v[...,i])+self.tol) for i in range(self.ndims-1)])
            lbox.append([(np.min(v[...,i], axis = 0)-self.tol,np.max(v[...,i]+self.tol, axis = 0)) for i in range(self.ndims-1)])

        return gbox, lbox

    def _get_center(self, mesh, lookup):
        cmesh = []
        for lp, msh in zip(lookup, mesh):
            etype = lp[0].split('_')[1]
            idx = self._linmap(self.order + 1).get(etype)
            cmesh.append(np.mean(msh[idx], axis = 0))

        return np.concatenate(cmesh, axis = 0)

    def _resort_mesh(self, mesh, gbox, lbox, cmesh):
        # Collect one periodic boundary

        amin = np.min(cmesh[:,-1])

        index = np.where(abs(cmesh[:,-1] - amin) < self.tol)[0]
        pele = cmesh[index]

        # take each 2-D element as a unit to search through all bounding boxes
        eid = self._global_check(gbox, pele)
        ptsinfo = self._local_check(lbox, pele, eid)
        ptsinfo = self._refine_pts(ptsinfo, mesh, pele)

        return ptsinfo

    def _refine_pts(self, ptsinfo, mesh, pts):
        oinfo, ref_face = defaultdict(list), {}
        for id, inf in ptsinfo.items():
            pt = pts[id]
            for erank, idx in inf:
                msh = mesh[erank][:,idx]

                # Get center of mesh for refinement
                ids = self._linmap(self.order + 1).get('hex')
                cmesh = np.mean(msh[ids], axis = 0)
                dist = np.linalg.norm(pt[:2] - cmesh[:,:2], axis = -1)
                ids = np.where(dist < self.tol)[0]
                idx, msh = idx[ids], msh[:,ids]

                if len(idx) > 0:
                    # Reorder element points
                    npts, neles, ndims = msh.shape
                    msh = msh.reshape(-1, self.ndims)

                    if id not in ref_face:
                        Nptz = int(len(idx)*(self.order + 1))
                        Nptf = int(len(msh)/Nptz)
                        ids = np.argsort(msh[:,-1])[:Nptf]
                        fpts = sorted(msh[ids,:2], key=lambda k: [k[1], k[0]])
                        ref_face[id] = fpts
                    else:
                        fpts = ref_face[id]

                    ids = []
                    for fpt in fpts:
                        tol, idd, Nptz = 1e-4, [], neles*(self.order + 1)
                        while len(idd) != Nptz:
                            tol += 1e-5
                            idd = np.where(np.linalg.norm(fpt - msh[:,:2], axis = 1) < tol)[0]

                            if tol > 1.5e-2 or len(idd) > Nptz:
                                print(tol, len(idd), Nptz)
                                import matplotlib.pyplot as plt
                                plt.figure()
                                plt.plot(msh[:,0],msh[:,1],'.')
                                afpts = np.array(fpts)
                                plt.plot(afpts[:,0],afpts[:,1],'.')
                                plt.show()
                                raise RuntimeError
                        ids.append(idd)
                    ids = np.concatenate(ids, axis = 0)

                    oinfo[id].append((erank, idx, ids))
        return oinfo




    def _local_check(self, lbox, pts, eid):
        ptsinfo = defaultdict(list)
        for id, gid in eid.items():
            pt = pts[id]
            for gidx in gid:
                llbox = lbox[gidx]
                index = np.arange(len(llbox[0][0]))
                for dim, (amin, amax) in enumerate(llbox):
                    idx = np.argsort(amin[index])
                    idxm = np.searchsorted(amin[index], pt[dim], sorter = idx, side = 'right')
                    idx = idx[:idxm]
                    idy = np.argsort(amax[index[idx]])
                    idxm = np.searchsorted(amax[index[idx]], pt[dim], sorter = idy, side = 'left')
                    idy = idy[idxm:]
                    index = index[idx[idy]]

                    if len(index) == 0:
                        break

                if len(index) != 0:
                    ptsinfo[id].append((gidx,index))
            if id not in ptsinfo:
                raise RuntimeError
        return ptsinfo


    def _global_check(self, gbox, pts):
        eid = defaultdict(list)

        for id in range(len(pts)):
            p = pts[id]
            for idg, rg in enumerate(gbox):
                if p[0] >= rg[0][0] and p[0] <= rg[0][1]:
                    if p[1] >= rg[1][0] and p[1] <= rg[1][1]:
                        eid[id].append(idg)
        return eid
