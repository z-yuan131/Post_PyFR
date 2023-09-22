import numpy as np
from collections import defaultdict
import h5py

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import ticker
from scipy.signal import welch, get_window

import functions.spod as sd
from functions.bl import BL_base
"""
This code is for processing spod modes. SPOD code is from spod.py. This class is
just a function to call and to post treatment.
"""

class SPOD(BL_base):
    def __init__(self, argv, icfg, fname):
        super().__init__(argv, icfg, fname)
        self.mode = icfg.get(fname, 'mode')
        self.fna = icfg.get(fname,'fname')

    def _con_to_pri(self, cons):
        rho, E = cons[0], cons[-1]

        # Divide momentum components by rho
        vs = [rhov/rho for rhov in cons[1:-1]]

        # Compute the pressure
        gamma = self.cfg.getfloat('constants', 'gamma')
        p = (gamma - 1)*(E - 0.5*rho*sum(v*v for v in vs))

        return [rho] + vs + [p]

    def sort_station(self, mesh, soln):

        Ntime, npts, nvars = soln.shape
        Ny = len(np.where(abs(mesh[:,-1] - np.min(mesh[:,-1])) < 1e-3)[0])
        idy = np.argsort(mesh[:,-1])
        mesh, soln = mesh[idy], soln[:,idy]
        print(mesh.shape, soln.shape)
        Nz = int(len(idy)/Ny)
        mesh = mesh.reshape(Ny, Nz, 3, order = 'F')
        soln = soln.reshape(Ntime, Ny, Nz, nvars, order = 'F')
        for i in range(mesh.shape[1]):
            id0 = np.argsort(mesh[:,i,1])
            mesh[:,i], soln[:,:,i] = mesh[id0,i], soln[:,id0,i]
        mesh, soln = mesh.reshape(-1, 3, order = 'F'), soln.reshape(Ntime, -1, nvars, order = 'F')
        return mesh, soln


    def _load_mesh_soln(self):

        if self.mode == 'genSPOD':
            # Load field data
            f = h5py.File(f'{self.dir}/{self.fna}','r')
            mesh = np.array(f['mesh'])
            soln = np.array(f['soln'])
            f.close()

            print(mesh.shape, soln.shape)

            # Calculate weight
            dy = mesh[0,1:,1] - mesh[0,:-1,1]
            dz = abs(mesh[1,0,-1] - mesh[0,0,-1])

            aweight = np.zeros(mesh.shape[:2])
            for ny in range(len(dy)):
                if ny == 0:
                    aweight[:,ny] = (dy[ny]*dz)/2
                    #aweight[[0,-1],ny] -= (dy[ny]*dz)/2
                if ny != 0:
                    aweight[:,ny] = (dy[ny-1]*dz)/2 + (dy[ny]*dz)/2
                if ny == len(dy) - 1:
                    aweight[:,ny+1] = (dy[ny]*dz)/2
                    #aweight[[0,-1],ny+1] -= (dy[ny]*dz)/2

            Ntime, Nz, Ny, Nvar = soln.shape
            mesh_plot, soln_plot, aweight = mesh.reshape(-1, 3), soln.reshape(Ntime, -1, Nvar), aweight.reshape(-1)
            mesh_plot[:,1] -= np.min(mesh_plot[:,1])

            triangle=tri.Triangulation(mesh_plot[:,-1],mesh_plot[:,1])

            # Plot mean flow
            for id in range(3):
                plt.figure()
                #var = soln[...,id]
                var = soln_plot[...,id]
                levels = np.linspace(np.min(var),np.max(var),40)
                #levels = np.linspace(0,0.35)
                plt.tricontourf(triangle, np.mean(var, axis = 0),levels = levels,cmap = 'coolwarm')
                plt.colorbar()
                if id == 0:
                    plt.tricontour(triangle, np.mean(var, axis = 0), levels= 0, linestyles='dashed', colors='black')
                plt.xlabel("$z-z^'$")
                plt.ylabel("$y-y_w$")
                #plt.savefig(f'{self.dir}/figs/{self.fna}_meanflow_{varmap[1]}.png')

                if id == 2:
                    plt.figure()
                    plt.tricontourf(triangle, np.sqrt(aweight),levels = 200,cmap = 'coolwarm')
                    plt.colorbar()

                    index = np.where(abs(mesh_plot[:,-1] + 12.365) < 1e-3)[0]
                    plt.figure()
                    plt.plot(mesh_plot[index,1],aweight[index],'.')
                    #plt.plot(mesh_plot[:,-1],mesh_plot[:,1],'.')
            plt.show()


            # Apply weight to solution
            soln_plot = soln_plot * np.sqrt(aweight)[None,:,None]

            #raise RuntimeError
            """ End """

            weights=np.ones(soln_plot.shape[1:])
            return mesh_plot[:,[1,2]], soln_plot, weights


        else:
            f = h5py.File(f'{self.dir}/{self.fna}','r')
            mesh = np.array(f['mesh']).reshape(-1,3)
            f.close()

            return mesh


    def _cal_spod(self, soln, dt, ndims, nvars, ndft, overlap, nmodes, weights, dir):
        """
        required:
          - time_step   : 1
          - n_space_dims: 2
          - n_variables : 1
          - n_dft       : 64

        optional:
          - overlap          : 50
          - mean_type        : 'longtime'
          - normalize_weights: False
          - normalize_data   : False
          - n_modes_save     : 40
          - conf_level       : 0.95
          - reuse_blocks     : False
          - savefft          : False
          - dtype            : 'double'
          - savedir          : 'spod_results'
          - fullspectrum     : False
        """

        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank

        from pyspod.spod.standard  import Standard  as spod_standard
        import pyspod.spod.utils     as utils_spod

        params = {}
        params['time_step']      = dt
        params['n_space_dims']   = ndims
        params['n_variables']    = nvars
        params['n_dft']          = ndft
        params['overlap']        = overlap
        params['n_modes_save']   = nmodes
        params['savedir']        = dir

        params['mean_type']         = 'longtime'
        params['normalize_weights'] = False
        params['normalize_data']    = False
        params['conf_level']        = 0.95
        params['reuse_blocks']      = False
        params['savefft']           = False
        params['dtype']             = 'double'
        params['fullspectrum']      = False


        ## compute spod modes and check orthogonality
        standard  = spod_standard(params=params, weights=weights, comm=comm)
        #streaming = spod_streaming(params=params, comm=comm)
        spod = standard.fit(data_list=soln)
        results_dir = spod.savedir_sim
        flag, ortho = utils_spod.check_orthogonality(
            results_dir=results_dir, mode_idx1=[1],
            mode_idx2=[0], freq_idx=[5], dtype='double',
            comm=comm)
        print(f'flag = {flag},  ortho = {ortho}')



    def main_proc(self):
        if self.mode == 'genSPOD':
            # Load field coordinate
            mesh, soln, weights = self._load_mesh_soln()

            print(mesh.shape, soln.shape)

            if len(soln.shape) == 3:
                ntime, npts, nvar  = soln.shape
                #soln = soln.reshape(ntime, -1)
            elif len(soln.shape) == 2:
                ntime, npts  = soln.shape
            else:
                raise RuntimeError('manupulate data shape in inport section')

            print(mesh.shape, soln.shape)

            # Get fluctuation
            soln = soln - np.mean(soln, axis = 0)[None,:,:]

            print(mesh.shape, soln.shape)

            weights = {'weights': weights, 'weights_name':'User_defined'}

            # soln, dt, ndims, nvars, ndft, overlap, nmodes, dir
            self._cal_spod(soln, self.dt*0.3/100, 2, 3, 128, 75, 40, weights, self.dir)

            return 0

        elif self.mode == 'postSPOD':
            print('pos_SPOD')

            mode = 2
            if mode == 1:
                self.station, self.Re = 5, '100k'

                dir = f'nfft128_novlp96_nblks172_station_{self.station}'

                ef = np.load(f'{self.dir}/{dir}/eigs_freq.npz')
                self.SPOD_mode_energy(ef['freq'], ef['eigs'])

                # Get the frequency that we wanted
                index = np.argsort(abs(ef['freq'] - 10.9744))[0]   #8.95
                print(index, ef['freq'][index])

                # Change a format of numbering
                index = "{:08.0f}".format(index)

                # Get SPOD mode shape
                mode = np.load(f'{self.dir}/{dir}/modes/freq_idx_{index}.npy')

                mesh = self._load_mesh_soln()

                self.SPOD_mode_field_mode_shape_new2(mesh, abs(mode))
            else:
                self.station, self.Re = [1,2,3,4,5], '100k'
                mm, ss = [], []
                for id in range(5):
                    dir = f'nfft128_novlp96_nblks172_station_{self.station[id]}'

                    ef = np.load(f'{self.dir}/{dir}/eigs_freq.npz')
                    #self.SPOD_mode_energy(ef['freq'], ef['eigs'])

                    # Get the frequency that we wanted
                    index = np.argsort(abs(ef['freq'] - 10.9744))[0]   #8.95
                    print(index, ef['freq'][index])

                    # Change a format of numbering
                    index = "{:08.0f}".format(index)

                    # Get SPOD mode shape
                    mode = np.load(f'{self.dir}/{dir}/modes/freq_idx_{index}.npy')

                    mesh = self._load_mesh_soln()

                    mm.append(mesh)
                    ss.append(abs(mode))

                    print(mode.shape)

                self.SPOD_mode_field_mode_shape_new3(mm, ss)

    def SPOD_mode_field_mode_shape_new3(self, mm, ss):
        fig, ax = plt.subplots(nrows=1, ncols=5,layout='constrained', sharey=True,sharex=True,figsize=(22.5, 4.5))

        s = np.array(ss)
        levels = np.linspace(0,np.max(s[:,:,0,0]),400)
        levels = np.linspace(0,0.05,400)
        print(np.max(levels))

        for mesh, soln, col in zip(mm, ss, ax):
            # Play around with mesh
            mesh[:,1] -= np.min(mesh[:,1])
            idx = np.where(mesh[:,1] < 3.5)
            msh, sln = mesh[idx], soln[idx]
            sln = sln[:,0,0]


            triangle = tri.Triangulation(msh[:,-1],msh[:,1])
            im = col.tricontourf(triangle, sln, levels,extend='both',cmap = 'coolwarm')

        # Set common labels
        fig.supxlabel("$z-z'$")
        fig.supylabel("$y-y_w$")
        plt.savefig(f'figs_2023_09_21/{self.Re}_station_{self.station}_abs_modes.png')
        plt.show()


    def SPOD_mode_field_mode_shape_new2(self, mesh, soln_abs):
        varmap = ['rho','u','v','w','p']

        # Play around with mesh
        mesh[:,1] -= np.min(mesh[:,1])
        idx = np.where(mesh[:,1] < 3)
        mesh, soln_abs = mesh[idx], soln_abs[idx]

        mode = []   #[sln[:,1] for sln in soln_abs]
        for nmode in range(soln_abs.shape[-1]):
            for nvar in range(soln_abs.shape[1]):
                mode.append(soln_abs[:,nvar,nmode])

        fig, ax = plt.subplots(nrows=1, ncols=3,layout='constrained', sharey=True,sharex=True,figsize=(12, 4.5))


        triangle = tri.Triangulation(mesh[:,-1],mesh[:,1])
        for i,row in zip(np.arange(1),[ax]):
            for j,col in zip(np.arange(3),row):
                sln = mode[i*3+j].copy()
                #levels = np.linspace(-np.max(sln),np.max(sln),40)
                levels = np.linspace(0,np.max(sln),400)
                print(sln.shape, mesh.shape)
                #im = col.tricontourf(triangle, sln, levels, locator=ticker.LogLocator(),extend='both',cmap = 'coolwarm')
                im = col.tricontourf(triangle, sln, levels,extend='both',cmap = 'coolwarm')


        # marks
        for i, j in zip([1,2,3],['u','v','w']):
            print(i)
            plt.figtext((i-1)*0.32+0.3, 0.9, fr"$\tilde {j}$",fontsize = 25)
            #plt.figtext((i-1)*0.32+0.25, 0.9, fr"$mode {i}$",fontsize = 16)


        # Set common labels
        fig.supxlabel("$z-z'$")
        fig.supylabel("$y-y_w$")

        #cbar = fig.colorbar(im, orientation='horizontal')
        """
        cbar = fig.colorbar(im)
        cbar.formatter.set_powerlimits((0, 0))
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()
        #"""
        #plt.tight_layout()
        #plt.savefig(f'{self.dir}/figs/100k_station_1_abs_{freq}.png')
        plt.savefig(f'figs_2023_09_21/{self.Re}_station_{self.station}_abs_modes.png')
        plt.show()


    def SPOD_mode_energy(self, freq, eig2, mode_num = 5):

        fig = plt.figure()
        # loop over each mode
        for imode in range(eig2.shape[1]):
            if imode < mode_num:  # highlight modes with colors
                plt.loglog(freq, eig2[:,imode], '.-',label='Mode '+str(imode+1)) # truncate last frequency
            elif imode == eig2.shape[1]-1:
                plt.loglog(freq, eig2[:,imode], '.-',color='lightgrey',label='Others')
            else:
                plt.loglog(freq, eig2[:,imode], '.-',color='lightgrey',label='')

        # figure format
        plt.xlabel('$St$')
        plt.ylabel('SPOD mode energy')
        #plt.legend(loc='best')
        plt.legend(loc='lower left')
        #plt.savefig(f'{self.dir}/figs/{self.fna}_SPOD_mode_energy.png ')
        plt.savefig(f'figs_2023_09_21/{self.Re}_station_{self.station}_energy.png')
        plt.show()
