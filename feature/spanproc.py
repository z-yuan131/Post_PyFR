from feature.region import Region

import numpy as np
from collections import defaultdict
import h5py

from pyfr.readers.native import NativeReader
from pyfr.quadrules import get_quadrule

"""
Idea here:
Use Region to extrat the region that we interested. Note here only mesh_wall
is kind of usedful since it has all information about element ids we need.

second, use the function in the Spanavg ( change the name) to re order the
elements relationship that we need for doing span average or FFT in span. Note,
avgspan and fftspan could be two distinct useful functions. More importantly,
it would be better to find a way to output the results from reordering.
Otherwise it could be a problem in loading snapshot if mpi failed.

third, mpi process for loading and process all snapshots. The key point here is
to evenly distribute all snapshots that needs to output.

After all of these, the total memory requirement should be much much smaller
and could be able to process locally for plotting etc.
"""


class SpanBase(Region):
    def __init__(self, argv, icfg, fname):
        super().__init__(argv, icfg, fname)
        self.mode = icfg.get(fname, 'mode', 'mesh')
        self.tol = icfg.getfloat(fname, 'tol', 1e-6)
        self.outfreq = icfg.getint(fname, 'outfreq', 1)
        nfft = icfg.getint(fname, 'nfft', 10)
        self._linmap = lambda n: {'hex': np.array([0, n-1, n**2-n, n**2-1,
                                n**2*(n-1), (n-1)*(n**2+1), n**3-n, n**3-1])}

        # Get indices of results we need
        self.nfft = np.append(np.arange(0,nfft),np.arange(-nfft+1,0))

    def _get_eles(self):
        f = h5py.File(f'{self.dir}/region.m','r')
        mesh_wall = defaultdict()
        for key in f:
            mesh_wall[key] = list(f[key])
        f.close()
        return mesh_wall

    def _ele_reorder(self):
        mesh_wall = self._get_eles()

        # Pre-process mesh
        mesh, lookup = self.mesh_proc(mesh_wall)

        # Collect one periodic boundary
        amin = min([np.min(msh[:,-1]) for msh in mesh])
        index = [np.where(abs(msh[:,-1] - amin) < self.tol)[0] for msh in mesh]

        #peid = []
        """
        for id, eids in enumerate(index):
            if len(eids) > 0:
                _, etype, part = lookup[id].split('_')

                con, _ = self.load_connectivity(self, part)

                uid = mesh_wall[lookup[id]][eids]
                uid = zip(etype, uid)

                for l, r in con:
                    if r in uid and l in uid:
                        con_new.append()
        """

        npeid = np.sum([len(index[id]) for id in range(len(mesh)) if len(index[id]) > 0])

        # Find adjacent elements in the periodic direction
        zeleid = defaultdict(list)
        n = 0
        for idp, eids in enumerate(index):
            if len(eids) > 0:
                for idm, msh in enumerate(mesh):
                    pts = mesh[idp][eids]
                    dists = [np.linalg.norm(msh[:,:2] - pt[:2], axis=1)
                                                                for pt in pts]
                    for peidx,dist in enumerate(dists):
                        iidx = np.where(dist < self.tol)[0]
                        if len(iidx) > 0:
                            zeleid[n+peidx].append((idm, iidx))

                n += len(eids)

        return zeleid, mesh_wall, lookup

    def spanfft(self):
        # Prepare for MPI process
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if self.mode == 'mesh':
            if rank == 0:
                zeleid, mesh_wall, lookup = self._ele_reorder()
                self.mesh_avg(zeleid, mesh_wall, lookup)
            return 0
        else:
            if rank == 0:
                zeleid, idlist, index, sortid, mesh_wall, lookup = self.loadcahcedfile()
            else:
                zeleid, idlist, index, sortid, mesh_wall, lookup = [None]*6

            # Boardcast pts and eles information
            zeleid = comm.bcast(zeleid, root=0)
            idlist = comm.bcast(idlist, root=0)
            index = comm.bcast(index, root=0)
            sortid = comm.bcast(sortid, root=0)
            mesh_wall = comm.bcast(mesh_wall, root=0)
            lookup = comm.bcast(lookup, root=0)

            # Get time series for each rank
            time = self.get_time_series_mpi(rank, size)

            if len(time) > 0:
                for id, item in index.items():
                    index[id] = [self.highorderproc(idx, self.order)
                                                            for idx in item]
                for t in time:
                    if self.method == 'FFT':
                        self.soln_fft(zeleid, mesh_wall, idlist, index,
                                                            sortid, t, lookup)
                    elif self.method == 'AVG':
                        self.soln_avg(zeleid, mesh_wall, idlist, index,
                                                            sortid, t, lookup)

    def soln_fft(self, zeleid, mesh_wall, idlist, index, sortid, time, lookup):
        ele_type = {'hex': 'quad'}
        soln_pts = {'quad': 'gauss-legendre'}

        # Solution average
        soln = f'{self.solndir}{time}.pyfrs'
        #soln = f'/Users/yuanzhenyang/compute/pyfr/Naca0012trip/Re200k/run/mesh.pyfrm'
        soln = self._load_snapshot(soln, mesh_wall, lookup)

        n = self.order + 1
        soln_fft = np.zeros([n**2,len(self.nfft),self.nvars,len(zeleid)],
                                                        dtype = np.complex_)

        for id, item in zeleid.items():
            for idx, (key, eid) in enumerate(item):
                if idx == 0:
                    sln = soln[key][:,eid]
                else:
                    sln = np.append(sln, soln[key][:,eid], axis = 1)

            if id in idlist:
                idx = index[id]
                for k, v in enumerate(idlist[id]):
                    mt = sln[:,v]
                    sln[:,v] = mt[idx[k]]

            # Elimate common surface boundary
            sln = self.comfb_fft(sln, n, sortid[id])
            soln_fft[...,id] = sln

        self._flash_to_disk(soln_fft, time, 'Imag')

    def soln_avg(self, zeleid, mesh_wall, idlist, index, sortid, time, lookup):
        ele_type = {'hex': 'quad'}
        soln_pts = {'quad': 'gauss-legendre'}

        # Solution average
        soln = f'{self.solndir}{time}.pyfrs'
        #soln = f'/Users/yuanzhenyang/compute/pyfr/Naca0012trip/Re200k/run/mesh.pyfrm'
        soln = self._load_snapshot(soln, mesh_wall, lookup)


        n = self.order + 1
        soln_avg = np.zeros([n**2,self.nvars,len(zeleid)])

        for id, item in zeleid.items():
            for idx, (key, eid) in enumerate(item):
                if idx == 0:
                    sln = soln[key][:,eid]
                else:
                    sln = np.append(sln, soln[key][:,eid], axis = 1)

            if id in idlist:
                idx = index[id]
                for k, v in enumerate(idlist[id]):
                    mt = sln[:,v]
                    sln[:,v] = mt[idx[k]]

            # Elimate common surface boundary
            sln = sln.reshape(n**2, -1, nvars, order= 'F')
            sln = np.mean(sln[:,sortid[id]], axis = 1)
            soln_avg[...,id] = sln

        self._flash_to_disk(soln_avg, time, 'Real')

    def _flash_to_disk(self, array, t, dtype):
        """Possibly there're much efficient ways to save complex numbers."""
        #f.create_dataset('soln', array, dtype='complex')
        if dtype == 'Real':
            f = h5py.File(f'{self.dir}/spanavg_avg_{t}.s', 'w')
            f['soln'] = array
        else:
            f = h5py.File(f'{self.dir}/spanavg_fft_{t}.s', 'w')
            f['soln_real'] = array.real
            f['soln_imag'] = array.imag
        f.close()



    def comfb_fft(self, sln, npts, sortid):
        neles, nvars = sln.shape[1:]
        soln = np.zeros([npts**2, neles * (npts - 1) + 1, nvars])
        sln = sln.reshape(npts**2, -1, nvars, order= 'F')
        sln = sln[:,sortid]

        j = 0
        for i in range(sln.shape[1]):
            if i != 0 and i % npts == 0:
                soln[:,j] = (sln[:,i] + sln[:,i-1])/2
                j += 1
            elif i % npts == npts - 1:
                continue
            else:
                soln[:,j] = sln[:,i]
                j += 1
        #print(soln.shape)
        return np.fft.fft(soln, axis = 1)[:,self.nfft] / soln.shape[1]


    def loadcahcedfile(self):
        f = h5py.File(f'{self.dir}/spanavg.m','r')
        zeleid = defaultdict(list)
        idlist = defaultdict(list)
        index = defaultdict(list)
        mesh_wall = {}
        lookup = {}
        sortid = []
        for i in f:
            if i == 'mesh':
                mesh = np.array(f[i])
            if i == 'zeleid':
                for id  in np.array(f[i]):
                    for kid in np.array(f[f'{i}/{id}']):
                        zeleid[int(id)].append((int(kid), np.array(f[f'{i}/{id}/{kid}'])))
            if i == 'sortid':
                for id  in np.array(f[i]):
                    sortid.append(np.array(f[f'{i}/{id}']))
            if i == 'idlist':
                for id  in np.array(f[i]):
                    for kid in np.array(f[f'{i}/{id}']):
                        idlist[int(id)].append(list(f[f'{i}/{id}/{kid}']))
            if i == 'index':
                for id  in np.array(f[i]):
                    for kid in np.array(f[f'{i}/{id}']):
                        index[int(id)].append(list(f[f'{i}/{id}/{kid}']))
            if i == 'mesh_wall':
                for key in np.array(f[i]):
                    mesh_wall[key] = np.array(f[f'{i}/{key}'])
            if i == 'lookup':
                for id in np.array(f[i]):
                    idx = int(np.array(f[f'{i}/{id}']))
                    lookup[idx] = id
        f.close()

        return zeleid, idlist, index, sortid, mesh_wall, lookup


    def mesh_avg(self, zeleid, mesh_wall, lookup):
        ele_type = {'hex': 'quad'}
        soln_pts = {'quad': 'gauss-legendre'}

        mesh = []
        idlist = {}
        index = {}
        indexx = {}
        sortid = []

        # Get operator
        n = self.meshord + 1
        mesh_op = self._get_mesh_op('hex', n**3)

        for key in lookup:
            msh = self.mesh[key][:,mesh_wall[key]]
            msh = np.einsum('ij,jkl -> ikl', mesh_op, msh)
            mesh.append(msh)

        n = self.order + 1
        mesh_avg = np.zeros([n**2,self.ndims,len(zeleid)])

        for id, item in zeleid.items():
            for idx, (kid, eids) in enumerate(item):
                if idx == 0:
                    msh = mesh[kid][:,eids]
                else:
                    msh = np.append(msh, mesh[kid][:,eids], axis = 1)
            if msh.shape[1] < 100:
                print(msh.shape)
            idxx, index1 = self.mesh_sort(msh)
            if len(idxx) > 0:
                idlist[id] = idxx
                index[id] = index1

        for id, item in index.items():
            indexx[id] = [self.highorderproc(idx, self.order) for idx in item]

        for id, item in zeleid.items():
            for idx, (key, eid) in enumerate(item):
                if idx == 0:
                    msh = mesh[key][:,eid]
                else:
                    msh = np.append(msh, mesh[key][:,eid], axis = 1)

            if id in idlist:
                idx = indexx[id]
                for k, v in enumerate(idlist[id]):
                    mt = msh[:,v]
                    msh[:,v] = mt[idx[k]]

            msh = msh.reshape(n**2,-1,self.ndims, order = 'F')
            sortid.append(np.argsort(np.mean(msh[:,:,-1],axis = 0)))

            # Average
            mesh_avg[:,:,id] = np.mean(msh, axis = 1)

        """
        # use strict soln pts set if not exit
        etype = ele_type['hex']
        try:
            self.cfg.get(f'solver-elements-{etype}', 'soln-pts')
        except:
            self.cfg.set(f'solver-elements-{etype}', 'soln-pts', soln_pts[etype])

        # Get operator
        mesh_op = self._get_mesh_op(etype, mesh_avg.shape[0])
        mesh_avg = np.einsum('ij,jkl -> ikl', mesh_op, mesh_avg)
        """

        # Flash ro disk
        #self._flash_to_disk(self.dir, mesh_avg)
        f = h5py.File(f'{self.dir}/spanavg.m', 'w')
        f['mesh'] = mesh_avg
        for id in zeleid:
            for kid, eids in zeleid[id]:
                f[f'zeleid/{id}/{kid}'] = eids

        for id in idlist:
            for k, v in enumerate(idlist[id]):
                f[f'idlist/{id}/{k}'] = v
            for k, v in enumerate(index[id]):
                f[f'index/{id}/{k}'] = v
        for id, v in enumerate(sortid):
            f[f'sortid/{id}'] = v

        for k in mesh_wall:
            f[f'mesh_wall/{k}'] = mesh_wall[k]
        for k, v in enumerate(lookup):
            f[f'lookup/{v}'] = k
        f.close()

        #return idlist, index1


    def mesh_sort(self, mesh):
        npts, neles = mesh.shape[:2]

        # Check if all elements in z direction are facing the same direction
        loc = [np.linalg.norm(mesh[0,0,:2] - mesh[0,i,:2], axis = 0) < self.tol
                                        for i in range(neles)]
        if not all(loc):
            idx = defaultdict(list)
            _map = self._linmap(self.meshord+1)['hex']
            mesh = mesh[_map]
            msh = mesh[:,0]
            index = []
            nid = -1
            for id, k in enumerate(loc):
                if not k:
                    msh[:,-1] += np.min(mesh[:,id,-1]) - np.min(msh[:,-1])
                    dlist = [np.linalg.norm(msh[j] - mesh[:,id], axis = 1)
                                        < self.tol for j in range(len(_map))]
                    dlist = [k for d in dlist for k, x in enumerate(d) if x]

                    if dlist not in index:
                        nid += 1
                        index.append(dlist)

                    idx[nid].append(id)

            return [v for k, v in idx.items()], index
        else:
            return [], []

    def highorderproc(self, index, order):
        """Just tabulate it until I found a way to write it efficiently"""
        n = order + 1
        for i in range(n**2):
            if i == 0:
                oindex = np.arange(n**3-n,n**3,1)
            else:
                oindex = np.append(oindex, np.arange(n**3-n*(i+1),n**3-n*i,1))
        return oindex


    def mesh_proc(self, mesh_wall):
        # Corner points map for hex type of elements
        n = self.meshord + 1
        _map = self._linmap(n)

        mesh = []
        lookup = []
        for k in mesh_wall:
            _, etype, part = k.split('_')
            msh = self.mesh[k][:,mesh_wall[k]]
            mesh.append(np.mean(msh[_map[etype]],axis = 0))
            lookup.append(k)
        return mesh, lookup

    def _load_snapshot(self, name, region, lookup):
        soln = []
        f = h5py.File(name, 'r')
        for kid, key in lookup.items():
            _, etype, part = key.split('_')
            kk = f'{self.dataprefix}_{etype}_{part}'
            sln = np.array(f[kk])[...,region[key]].swapaxes(1,-1)
            npts, nele, nvars = sln.shape
            sln = np.array(self._con_to_pri(sln.reshape(-1,nvars).T)).T
            soln.append(sln.reshape(npts, nele, nvars))
        f.close()
        return soln

    def _con_to_pri(self, cons):
        rho, E = cons[0], cons[-1]

        # Divide momentum components by rho
        vs = [rhov/rho for rhov in cons[1:-1]]

        # Compute the pressure
        gamma = self.cfg.getfloat('constants', 'gamma')
        p = (gamma - 1)*(E - 0.5*rho*sum(v*v for v in vs))

        return [rho] + vs + [p]
