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
        self.tol = icfg.getfloat(fname, 'tol', 1e-6)
        self.outfreq = icfg.getint(fname, 'outfreq', 1)
        nfft = icfg.getint(fname, 'nfft', 10)
        self._linmap = lambda n: {'hex': np.array([0, n-1, n**2-n, n**2-1,
                                n**2*(n-1), (n-1)*(n**2+1), n**3-n, n**3-1])}

        # Get indices of results we need
        #self.nfft = np.append(np.arange(0,nfft),np.arange(-nfft+1,0))
        self.nfft = np.arange(nfft)

        # The box that elements are in
        self.box = icfg.getliteral(fname, 'box', None)

        self.emap = lambda x: {(x+1)**2: 'quad', (x+1)*(x+2)/2: 'tri'}

    def rot_map(self):
        from math import pi
        self.AoA = 3
        rot_map = np.array([[np.cos(self.AoA/180*pi),np.sin(self.AoA/180*pi),0],
                [-np.sin(self.AoA/180*pi), np.cos(self.AoA/180*pi), 0],
                [0,0,1]])
        return rot_map[:self.ndims,:self.ndims]

    def _get_box(self):
        mesh, lookup = [], []
        # Get all elements inside the box that to be averaged
        for k, v in self.mesh.items():
            if 'spt' in k.split('_'):
                _, etype , part = k.split('_')
                if etype in self.suffix:
                    if self.box:
                        amin, amax = np.min(v, axis = 0),np.max(v, axis = 0)

                        # For NACA0012 Re2e5 case
                        amin = amin @ self.rot_map()
                        amax = amax @ self.rot_map()
                        #end

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
                            #mesh.append(v[:,index])
                            lookup.append((k,index))
        return mesh, lookup

    def main_proc(self):

        if self.mode == 'mesh':
            mesh, lookup = self._get_box()

            gbox, lbox = self._gen_bounding_box(mesh)

            cmesh = self._get_center(mesh, lookup)

            """
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(cmesh[:,0],cmesh[:,1],'.')
            plt.show()
            """
            ptsinfo = self._resort_mesh(mesh, lookup, gbox, lbox, cmesh)


            #"""
            import matplotlib.pyplot as plt
            plt.figure()
            for id, info in ptsinfo.items():
                pts_t = []
                for erank, idx, ids in info:
                    pts = mesh[erank][:,idx].reshape(-1, self.ndims)
                    Nptz = len(idx)*(self.order + 1)
                    #print(len(ids)/Nptz)
                    pts = pts[ids].reshape(Nptz, -1, self.ndims, order = 'F')
                    pts_t.append(pts)
                pts = np.mean(np.concatenate(pts_t, axis = 0),axis = 0)
                #pts = np.concatenate(pts_t, axis = 1)
                plt.plot(pts[:,0],pts[:,1],'.')
            plt.figure()
            for id, info in ptsinfo.items():
                pts_t = []
                for erank, idx, ids in info:
                    pts = mesh[erank][:,idx].reshape(-1, self.ndims)
                    Nptz = len(idx)*(self.order + 1)
                    #print(len(ids)/Nptz)
                    pts = pts[ids].reshape(Nptz, -1, self.ndims, order = 'F')
                    pts_t.append(pts)
                pts = np.mean(np.concatenate(pts_t, axis = 0),axis = 0)
                #pts = np.concatenate(pts_t, axis = 1)
                plt.plot(pts[:,0],pts[:,-1],'.')
            plt.show()
            print(np.max(pts[:,-1]),np.min(pts[:,-1]))
            #"""

            self._cachedfile(ptsinfo, lookup)
            return 0
        else:
            # Get mpi info
            comm, rank, root = get_comm_rank_root()
            size = comm.Get_size()

            for i in range(size):
                if i == rank:
                    ptsinfo, lookup = self._cachedfile()


            if rank == 0:
                mesh = self._load_mesh(lookup)
                if np.max(self.nfft) == 0:
                    spts = self._avg_proc(ptsinfo, mesh, self.ndims)
                else:
                    spts, index = self._fft_proc(ptsinfo, mesh, self.ndims)

                self._flash_to_disk(spts, 'mesh')
                del mesh, spts
            else:
                index = None

            if np.max(self.nfft) != 0:
                index = comm.bcast(index, root = 0)

            """
            import matplotlib.pyplot as plt
            plt.figure()
            for k, pts in spts.items():
                print(pts.shape)
                plt.plot(pts[...,0],pts[...,1],'.')
            plt.show()

            raise RuntimeError
            """

            soln_op = self._get_op_soln(lookup)
            # Get time series for each rank
            time = self.get_time_series_mpi(rank, size)
            print(rank, time)
            for t in time:
                soln = self._load_soln(t, lookup, soln_op)
                if np.max(self.nfft) == 0:
                    # Average subroutine
                    soln = self._avg_proc(ptsinfo, soln, self.nvars)
                else:
                    # FFT subroutine
                    soln, _ = self._fft_proc(ptsinfo, soln, self.nvars, index)
                    # Average boundary surface and do FFT
                    soln = self.ebcavgfft(soln)

                self._flash_to_disk(soln, t)

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

    def _cachedfile2(self, ptsinfo = [], lookup = []):
        if lookup != []:
            f = h5py.File(f'{self.dir}/spanavg.m','w')
            for etype, einfo in enumerate(ptsinfo):
                for id, info in einfo.items():
                    pid = 0
                    for inf in info:
                        for erank, idx, ids in inf:
                            f[f'{etype}/{id}/{pid}/{erank}/idx'] = idx
                            f[f'{etype}/{id}/{pid}/{erank}/ids'] = ids
                        pid += 1

            for k, idx in lookup:
                f[k] = idx
            f.close()
        else:
            ptsinfo, lookup, info = {}, [], []
            f = h5py.File(f'{self.dir}/spanavg.m','r')
            for etype in f:
                if 'spt' in etype.split('_'):
                    lookup.append((etype, np.array(f[etype])))
                else:
                    for id in f[etype]:
                        ele = []
                        for pid in f[f'{etype}/{id}']:
                            p = []
                            for erank in f[f'{etype}/{id}/{pid}']:
                                idx = np.array(f[f'{etype}/{id}/{pid}/{erank}/idx'])
                                ids = np.array(f[f'{etype}/{id}/{pid}/{erank}/ids'])
                                p.append((int(erank), idx, ids))
                            ele.append(p)
                        ptsinfo[id] = ele
                    info.append(ptsinfo)

            f.close()
            return info, lookup




    def _flash_to_disk(self, array, t):
        f = h5py.File(f'{self.dir}/spanproc_{t}.s', 'w')
        for etype, soln in array.items():
            f[f'{etype}'] = soln
        f.close()

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
            sln = np.einsum('ij, jkl -> ilk', soln_op[etype], sln)
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

    def _resort_mesh(self, mesh, lookup, gbox, lbox, cmesh):
        # Collect one periodic boundary

        amin = np.min(cmesh[:,-1])

        index = np.where(abs(cmesh[:,-1] - amin) < self.tol)[0]
        pele = cmesh[index]

        # wake Region
        #"""
        if any(cmesh[:,0] - 108 > 0):
            index = np.where(cmesh[:,0] - 108 > 0)[0]
            amin = np.min(cmesh[index,-1])
            index = np.where(abs(cmesh[:,-1] - amin) < self.tol)[0]
            pele = np.concatenate((pele, cmesh[index]), axis = 0)
        """
        index = np.where(cmesh[:,0] - 10 < 0)[0]
        amin = np.min(cmesh[index,-1])
        index = np.where(abs(cmesh[:,-1] - amin) < self.tol)[0]
        pele = np.concatenate((pele, cmesh[index]), axis = 0)
        #"""

        #"""
        import matplotlib.pyplot as plt
        plt.plot(pele[:,0],pele[:,1],'.')
        plt.show()
        #raise RuntimeError
        #"""



        # take each 2-D element as a unit to search through all bounding boxes
        eid = self._global_check(gbox, pele)
        ptsinfo = self._local_check(lbox, pele, eid)
        ptsinfo = self._refine_pts(ptsinfo, mesh, pele)

        return ptsinfo

    def _refine_pts(self, ptsinfo, mesh, pts):
        #import matplotlib.pyplot as plt
        #plt.figure()
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

                    # Reoder points inside one element
                    #plt.plot(msh[:,0],msh[:,-1],'.')


                    """
                    msht = np.array([(msh[id,0],msh[id,1],msh[id,2]) for id in range(len(msh))],
                        dtype=[('x', np.float64), ('y', np.float64), ('z', np.float64)])

                    Nptz = int(len(idx)*(self.order + 1))
                    Nptxy = int(len(msh)/Nptz)
                    ids = np.argsort(msht, order=('z'))
                    ids = ids.reshape(-1, Nptz, order = 'F').T

                    idss = []
                    for sid, idd in enumerate(ids):
                        if (sid+1) % (self.order+1) == 0 and sid+1 != Nptz:
                            idd = np.append(idd, ids[sid+1], axis = 0)
                            stid = np.argsort(msht[idd], order=('x','y'))
                            stid = stid.reshape(2, Nptxy)
                            for tid in stid:
                                idss.append(idd[tid])

                        elif sid % (self.order+1) == 0 and sid != 0:
                            continue
                        else:
                            idd = idd[np.argsort(msht[idd], order=('x','y'))]
                            idss.append(idd)

                    #for ids in idss:
                    #    plt.plot(msh[ids,0],msh[ids,-1],'.')
                    #plt.show()

                    ids = np.array(idss).reshape(-1, order = 'F')
                    #raise RuntimeError
                    #"""


                    #ids = sorted(msh , key=lambda k: [k[2], k[1], k[0]])
                    #msh = np.array([(msh[id,0],msh[id,1],msh[id,2]) for id in range(len(msh))],
                    #    dtype=[('x', np.float64), ('y', np.float64), ('z', np.float64)])
                    #ids = np.argsort(msh, order=('y','x','z'))
                    #r = np.linalg.norm(msh[:,:2], axis = 1)
                    #msh = np.array([(r[id],msh[id,2]) for id in range(len(msh))],
                    #    dtype=[('x', np.float64), ('z', np.float64)])
                    #ids = np.argsort(msh, order=('x','z'))

                    if id not in ref_face:
                        Nptz = int(len(idx)*(self.order + 1))
                        Nptf = int(len(msh)/Nptz)
                        ids = np.argsort(msh[:,-1])[:Nptf]
                        fpts = sorted(msh[ids,:2], key=lambda k: [k[1], k[0]])
                        ref_face[id] = fpts

                    else:
                        fpts = ref_face[id]

                    #"""
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
                    #"""


                    """
                    tol = 1e-4
                    while len(ids) != len(msh):
                        tol += 1e-5
                        ids = [np.where(np.linalg.norm(pt - msh[:,:2], axis = 1) < tol)[0] for pt in fpts]
                        #print([len(idd) for idd in ids], len(msh)/25)
                        ids = np.concatenate(ids, axis = 0)
                        #print(tol, len(ids), len(msh))
                        if tol > 1.5e-2:
                            print(tol, len(ids), Nptz*Nptf)
                            import matplotlib.pyplot as plt
                            plt.figure()
                            plt.plot(msh[:,0],msh[:,1],'.')
                            afpts = np.array(fpts)
                            plt.plot(afpts[:,0],afpts[:,1],'.')
                            plt.show()
                            raise RuntimeError
                    #"""
                    oinfo[id].append((erank, idx, ids))
        #plt.show()
        #raise RuntimeError
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

            # For checking purpose
            #if id == 200:
            #    break
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
