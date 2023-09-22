from base import Base

import numpy as np
from collections import defaultdict
import h5py

"""
This function does nothing but gather time series into one file. It make a lot
of sense if those files are small but many. This will be replaced by some other
functions that could be integrated to solver directly.
"""


class DUP(Base):
    def __init__(self, argv, icfg, fname):
        super().__init__(argv)
        self.prefix = icfg.get(fname, 'prefix')
        self.dimension = icfg.get(fname, 'dim','3D')

    def main_proc(self):
        # A potential good algorithm for getting rid of duplicated points

        # Load mesh
        f = h5py.File(f'{self.dir}/{self.prefix}_mesh.s','r')
        mesh, lookup = [], []
        for etype in f:
            if len(etype.split('_')) == 1:
                mesh.append(np.array(f[etype]).reshape(-1, self.ndims))
                lookup.append(etype)
        f.close()

        mesh = np.concatenate(mesh, axis = 0)

        # limit region
        #mesh = mesh @ self.rot_map()
        #region_limit = np.where(mesh[:,1]>0)[0]

        if self.dimension == '2D':
            mesh_dp,b,c,d =  np.unique(mesh[:,:2],return_index=True, return_inverse=True,return_counts=True, axis = 0)
        else:
            mesh_dp,b,c,d =  np.unique(mesh,return_index=True, return_inverse=True,return_counts=True, axis = 0)

        # resort to classify data into several dicts
        index = np.argsort(c)
        n, idx = 0, defaultdict(list)
        for k in d:
            idx[k].append(index[n:n+k])
            n+=k

        # Collect all time series
        for t in self.time:
            print(t)
            #self._load_snapshot(t, idx, lookup, region_limit)
            self._load_snapshot(t, idx, lookup)

        # Write time series into one file
        f = h5py.File(f'{self.dir}/{self.prefix}_timeseries.s','r')
        soln = []
        for i in f:
            soln.append((i, np.array(f[i])))
        f.close()

        soln.sort()
        soln = [v for t, v in soln]
        soln = np.stack(soln)

        print(soln.shape)

        # average
        mesh_dp = []
        for k,v in idx.items():
            v = np.array(v)
            temp = mesh[v.reshape(-1)].reshape(v.shape[0],v.shape[1],-1)
            mesh_dp.append(np.mean(temp, axis = 1))

        mesh_dp = np.concatenate(mesh_dp, axis = 0)

        f = h5py.File(f'{self.dir}/{self.prefix}_timeseries.s','w')
        f['mesh'] = mesh_dp
        f['soln'] = soln
        f.close()

        f = h5py.File(f'{self.dir}/{self.prefix}_mean.s','w')
        f['mesh'] = mesh_dp
        f['soln'] = np.mean(soln, axis = 0)
        f.close()

    def _load_snapshot(self, t, idx, lookup, region_limit=[]):
        # open HDF5-files
        data = h5py.File(f'{self.dir}/{self.prefix}_{t}.s','r')

        soln = []
        for path in lookup:
            nvars = np.array(data[path]).shape[-1]
            soln.append(np.array(data[path]).reshape(-1, nvars))
        data.close()

        soln = np.concatenate(soln, axis = 0)

        # limit region
        #soln = soln[region_limit]

        # average
        nsoln = []
        for k,v in idx.items():
            v = np.array(v)
            temp = soln[v.reshape(-1)].reshape(v.shape[0],v.shape[1],-1)
            nsoln.append(np.mean(temp, axis = 1))

        soln = np.concatenate(nsoln, axis = 0)

        if t == self.time[0]:
            new_data = h5py.File(f'{self.dir}/{self.prefix}_timeseries.s','w')
        else:
            new_data = h5py.File(f'{self.dir}/{self.prefix}_timeseries.s','a')
        new_data[f'{t}'] = soln
        new_data.close()
