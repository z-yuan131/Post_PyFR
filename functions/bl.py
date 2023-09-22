import numpy as np
from collections import defaultdict
import h5py

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import ticker
from matplotlib import colors
from scipy.signal import welch, get_window

plt.rcParams.update({'font.size': 12})

from base import Base
from pyfr.util import subclasses
from pyfr.shapes import BaseShape

"""
This code is for processing boundary layer properties
The code can be divided into two parts:
the first part is designed to focus on the boundary layer profiles
the second part is focus on the surface pressure and stress distribution
the first part will use data from the spanfft
the second part will use data from the grad
"""

class BL_base(Base):
    def __init__(self, argv, icfg, fname):
        super().__init__(argv)
        self.AoA = icfg.getfloat(fname, 'AoA', 0)
        self._trip_loc = icfg.getfloat(fname, 'trip-loc', None)
        self.fmt = icfg.get(fname, 'format', 'primitive')
        self.tol = 1e-5
        self.etype = ['hex']

        self.L = 100

        print(self._constants)
        self._rho = rho = self._constants['rhoInf']
        self._Uinf = Uinf = self._constants['uInf']
        self._pinf = pinf = self._constants['pInf']
        self._dyna_p = 0.5*rho*Uinf**2

        self.enpts = {}
        for k in self.mesh_inf:
            if k.split('_')[1] in self.etype:
                self.enpts[k.split('_')[1]] = self.mesh_inf[k][1][0]

    def _load_preproc_wall_mesh(self):
        # Get wall mesh from pyfrm file
        mid = defaultdict(list)
        for k in self.mesh:
            if 'bcon' in k.split('_') and 'wall' in k.split('_'):
                for etype, eid, fid, pid in self.mesh[k][['f0','f1','f2','f3']].astype('U4,i4,i1,i2'):
                    if etype in self.etype:
                        part = k.split('_')[-1]
                        mid[f'spt_{etype}_{part}'].append((eid, fid))

        basismap = {b.name: b for b in subclasses(BaseShape, just_leaf=True)}

        fpts, m0, mesh_op = {}, {}, {}
        for etype in self.etype:
            basis = basismap[etype](self.enpts[etype], self.cfg)
            fpts[etype] = basis.facefpts
            m0[etype] = self._get_interp_mat(basis, etype)
            mesh_op[etype] = self._get_mesh_op(etype, self.enpts[etype])

        for k in mid:
            eid, fid = zip(*mid[k])
            etype = k.split('_')[1]
            # Get elements on the surface
            msh = self.mesh[k][:,list(eid)]
            # Interpolation to same order of solution
            msh = np.einsum('ij, jkl -> ikl',mesh_op[etype],msh)
            # Get face points based on etype
            msh = np.array([m0[etype][fpts[etype][fid[id]]] @ msh[:,id] for id in range(len(fid))])

            try:
                mesh = np.append(mesh, msh, axis = 0)
            except UnboundLocalError:
                mesh = msh

        # Get a slice of mesh and get rid of all duplicated points
        # Set the first point as reference
        mesh = mesh.reshape(-1, self.ndims)
        msh = mesh[0]
        # Get a slice
        index = np.where(abs(mesh[:,-1] - msh[-1]) < self.tol)[0]
        mesh = mesh[index]
        # Use Pandas to get rid of duplicated points
        mesh = self._duplpts_pd(mesh)
        """
        plt.figure()
        plt.plot(mesh[:,0],mesh[:,1],'.')
        plt.show()
        """
        return mesh @ self.rot_map()

    def _ortho_mesh(self, mesh):
        # Split mesh into suction and pressure side
        index = np.where(mesh[:,1] >= 0)[0]
        idx = np.argsort(mesh[index, 0])
        meshu = mesh[index[idx]]

        index = np.where(mesh[:,1] <= 0)[0]
        idx = np.argsort(mesh[index, 0])
        meshl = mesh[index[idx]]

        vect = np.diff(meshu, axis = 0)
        # Normalise tangential vector
        vectu = vect/np.linalg.norm(vect, axis = -1)[:,None]
        # Calculate normal vector
        vecnu = np.cross(vectu,np.array([0,0,-1]))

        vect = np.diff(meshl, axis = 0)
        # Normalise tangential vector
        vectl = vect/np.linalg.norm(vect, axis = -1)[:,None]
        # Calculate normal vector
        vecnl = np.cross(vectl,np.array([0,0,1]))

        mesh = np.append(meshu[:-1], meshl[1:], axis = 0)
        vect = np.append(vectu, vectl, axis = 0)
        vecn = np.append(vecnu, vecnl, axis = 0)

        # Create the mesh by growing out via Chebychev points
        from math import pi
        L = 10*(1-np.cos(np.linspace(0, pi/2, 200)))
        mesh = mesh[...,None] + np.einsum('ij,k ->ijk', vecn, L)

        # Angle vector from cartitian to local coordinate
        xt = vect @ np.array([1,0,0])
        yt = vect @ np.array([0,1,0])

        xn = vecn @ np.array([1,0,0])
        yn = vecn @ np.array([0,1,0])

        return mesh.swapaxes(1,-1), np.append(xt[:,None],yt[:,None], axis = -1).T

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

    def _duplpts_pd(self, mesh, subset=['x','y']):
        # Use panda to fast drop duplicated points
        import pandas as pd
        df = pd.DataFrame({'x':mesh[:,0], 'y':mesh[:,1], 'z':mesh[:,2]})
        return df.drop_duplicates(subset=subset).values

    def rot_map(self):
        from math import pi
        rot_map = np.array([[np.cos(self.AoA/180*pi),np.sin(self.AoA/180*pi),0],
                [-np.sin(self.AoA/180*pi), np.cos(self.AoA/180*pi), 0],
                [0,0,1]])
        return rot_map[:self.ndims,:self.ndims]

    def stress_tensor(self, du, u):
        c = self._constants

        # Density, pressure
        rho, p = u[0], u[-1]

        # Gradient of velocity
        gradu = du.reshape(self.ndims, self.ndims, -1)

        # Bulk tensor
        bulk = np.eye(self.ndims)[:, :, None]*np.trace(gradu)

        # Viscosity
        mu = c['mu']

        if self._viscorr == 'sutherland':
            cpT = c['gamma']*p/rho/(c['gamma'] - 1)
            Trat = cpT/c['cpTref']
            mu *= (c['cpTref'] + c['cpTs'])*Trat**1.5 / (cpT + c['cpTs'])

        return mu*(gradu + gradu.swapaxes(0, 1) - 2/3*bulk)

    def _reorder_pts(self, mesh, soln):
        #msh = self._duplpts_pd(mesh, ['x','y'])
        index = np.where(abs(mesh[:,-1] - np.max(mesh[:,-1])) < 1e-5)
        msh = mesh[index]

        omesh, osoln = [], []
        for pt in msh:
            index = np.where(np.linalg.norm(pt[:2] - mesh[:,:2], axis = 1) < 1e-3)[0]
            #index2 = np.argsort(mesh[index1,-1])
            #index = index1[index2]
            #print(len(index))
            if len(index) > 30:
                omesh.append(np.mean(mesh[index], axis = 0))
                osoln.append(np.mean(soln[index], axis = 0))
        return np.array(omesh), np.array(osoln)

    def _get_rid_of_dp_pts_cf(self, mesh, soln, index = []):
        mn, sn = [], []
        nvars = soln.shape[-1]
        mesh = mesh.reshape(-1, self.ndims)
        soln = soln.reshape(-1, nvars)
        for k, idx in enumerate(index):
            msh = np.mean(mesh[idx], axis = 0)
            sln = np.mean(soln[idx], axis = 0)

            mn.append(msh)
            sn.append(sln)

        msh = np.stack(mn)
        sln = np.stack(sn)
        return msh, sln

    def _con_to_pri(self, cons):
        rho, E = cons[0], cons[-1]

        # Divide momentum components by rho
        vs = [rhov/rho for rhov in cons[1:-1]]

        # Compute the pressure
        gamma = self.cfg.getfloat('constants', 'gamma')
        p = (gamma - 1)*(E - 0.5*rho*sum(v*v for v in vs))

        return [rho] + vs + [p]


class BL_wall_unit(BL_base):
    def __init__(self, argv, icfg, fname):
        super().__init__(argv, icfg, fname)

    def main_proc(self):

        print('This function is for wall normal velocity')
        dir = './data/probes_Re200k_refined'

        f = h5py.File(f'{dir}/interp_wall.pyfrs','r')
        meshw = np.array(f['mesh'])
        solnw = np.array(f['soln'])
        f.close()


        f = h5py.File(f'{dir}/grad_mean.s','r')
        meshg = np.array(f['mesh'])
        solng = np.array(f['soln'])
        f.close()

        plt.figure()
        name = ['0.55','0.65','0.85','0.95']
        for id in range(4):
            mesh, soln = meshw[id], np.mean(solnw[id], axis = 0)

            index = np.where(np.linalg.norm(meshg[:,:2] - mesh[0,:2],axis = -1)<1e-2)[0]
            mg, sg = meshg[index], solng[index]

            mg, sg = mg[0], np.mean(sg, axis = 0)


            du = sg.T[self.nvars:]
            u = sg.T[:self.nvars]
            tau = self.stress_tensor(du, u)

            # Tan aerofoil
            dy = lambda x: 0.6*(0.2969*x**(-0.5)*0.5 - 0.1260*x**0 - 0.3516*x**1*2 + 0.2843*x**2*3 - 0.1015*x**3*4)

            tan = dy(mg[0]/100)
            angle = np.angle(tan)
            # \tau'\!_{xy}  =  (\sigma_{yy} - \sigma_{xx}) sin(theta)*cos(theta) + \tau_{xy} * (cos^2(theta) - sin^2(theta))
            tau = (tau[1,1] - tau[0,0])*np.sin(angle)*np.cos(angle) + tau[0,1]*np.cos(2*angle)

            vect = np.array([1,tan])
            vect = vect / np.linalg.norm(vect)

            u = soln.T
            uu = u[1] * vect[0] + u[2] * vect[1]




            # Viscosity
            c = self._constants
            mu = c['mu']

            print(tau,du[1], mu,du[1]* mu)
            tau = tau*2/3

            if self._viscorr == 'sutherland':
                rho,p = u[0], u[-1]
                cpT = c['gamma']*p/rho/(c['gamma'] - 1)
                Trat = cpT/c['cpTref']
                mu *= (c['cpTref'] + c['cpTs'])*Trat**1.5 / (cpT + c['cpTs'])

            #y = np.linalg.norm(mesh - mesh[0], axis = -1)
            y = mesh - mesh[0]
            y = np.sqrt(y[:,0]**2 + y[:,1]**2)

            tau = (uu[1] - uu[0])/(y[1] - y[0])
            tau = tau*0.00015


            ut = np.sqrt(tau/rho)
            nu = mu/rho
            yplus = y*ut/nu
            uplus = uu/ut

            print(yplus.shape, u.shape)

            plt.plot(yplus, uplus, label = f'x/c = {name[id]}')

        kappa, B = 0.41, 5.2
        index = np.where(yplus > 5)[0]
        plt.plot(yplus[index], (1/kappa)*np.log(yplus[index]) + B, '--', label = 'Log law: $\kappa, B = 0.41, 5.2$')
        # Linear law
        index = np.where(yplus < 10)[0]
        plt.plot(yplus[index],yplus[index],'.-',label = 'Linear law')
        plt.xscale('log')

        plt.legend()
        plt.xlabel('$y^+$')
        plt.xlabel('$u^+$')
        plt.savefig('figs_2023_09_21/Re2e5_wall_unit.eps')


        plt.figure()
        plt.plot(mesh[:,0],mesh[:,1],'.')
        plt.plot(mg[0],mg[1],'.')


        plt.figure()
        plt.plot( soln[:,1], mesh[:,1],'.')
        plt.show()


class BL(BL_base):
    def __init__(self, argv, icfg, fname):
        super().__init__(argv, icfg, fname)


    def _load_mesh_soln(self):
        f = h5py.File(f'{self.dir}/spanavg_mean.s','r')
        #f = h5py.File(f'{self.dir}/slice2d_mean_80k_0.12.s','r')
        mesh = np.array(f['mesh'])
        soln = np.array(f['soln'])
        f.close()

        """ Limit the region for plotting and calculation """
        index = np.where(mesh[:,0] < 105)
        mesh, soln = mesh[index], soln[index]
        """ END """

        print(self.fmt)
        if self.fmt != 'primitive':
            soln = np.array(self._con_to_pri(soln.T)).T

        return mesh @ self.rot_map(), soln


    def main_proc(self):
        # Create new mesh which is orthogonal to wall mesh
        mesh_wall = self._load_preproc_wall_mesh()
        # Load original mesh and soln
        mesh, soln = self._load_mesh_soln()

        print(mesh.shape, soln.shape, self._trip_loc)



        #raise RuntimeError
        """ Runtime cp-cf first to get tau"""

        meshw, vecm = self._ortho_mesh(mesh_wall)

        xmesh = []
        # Get location of profiles are plotted
        #xloc = [0.62, 0.75, 0.85, 0.94]
        xloc = np.linspace(0.55, 0.95, 5)
        for x in xloc:
            idx0 = np.where(meshw[:,0,1] > 0)[0]
            idx1 = np.where(meshw[:,0,0] > x*np.max(meshw[:,0,0]))[0][:1]
            xmesh.append(np.mean(meshw[idx0[idx1]], axis = 0))
        xmesh = np.stack(xmesh)
        print(meshw.shape, xmesh.shape)


        xsoln = self._interpolation(mesh, soln, xmesh)
        solnw = self._interpolation(mesh, soln, meshw)
        self._sample_location(meshw, solnw, xmesh, xsoln)
        bledge = self._plot_session_bl_profile(xmesh, xsoln, xloc)

        #f = h5py.File('bledge.h5','a')
        #f['lower'] = np.array(bledge)
        #f.close()

        # Plot field varibles
        self._plot_session_field(mesh, soln, mesh_wall, bledge)

        """
        # Split the region at the tripping location
        mm, info = [], {}
        if self._trip_loc:
            index = np.where(mesh_wall[:,0] < self._trip_loc)[0]
            mm.append(mesh_wall[index])
            index = np.where(mesh_wall[:,0] > self._trip_loc)[0]
            mm.append(mesh_wall[index])

            for id, msh in enumerate(mm):
                meshw, vecm = self._ortho_mesh(msh)
                info[id] = self._interpolation(mesh, soln, meshw, vecm)
        """

        #self._plot_session(info)

    def _plot_session_field(self, mesh, soln, meshw, bledge):
        # Bad points on the wall
        index = [np.argsort(np.linalg.norm(mesh[:,:2] - msh[:2] , axis = 1))[0] for msh in meshw]

        # Normalise stuff with chord length
        mesh, bledge, self._trip_loc = mesh/100, bledge/100, self._trip_loc/100

        # Seperation bubble
        """
        v = np.linalg.norm(soln[:,[1,2]], axis = 1)
        idx = np.where(v < 1e-3)
        mesh_bubble = mesh[idx]

        sln_amp = np.linalg.norm(soln[:,[1,2,3]], axis = 1)
        soln = np.append(soln, sln_amp[:,None], axis = 1)
        """
        varname = ['rho','u','v','w','p']
        for i in [0,1,2,4]:
            var = soln[:,i].copy()
            bubble = soln[:,1].copy()
            levels = np.linspace(np.min(var),np.max(var),40)

            # Wall points
            var[index] = np.NaN
            bubble[index] = np.NaN
            bubble = np.ma.masked_invalid(bubble)
            bubble = bubble.filled(fill_value=0.0)

            plt.figure(figsize=(20,3))
            if self._trip_loc:
                for j in range(2):
                    sln = var.copy()
                    if j == 0:
                        idx = np.where(mesh[:,0] > self._trip_loc)[0]
                    else:
                        idx = np.where(mesh[:,0] < self._trip_loc)[0]
                    sln[idx] = np.NaN

                    sln = np.ma.masked_invalid(sln)
                    sln = sln.filled(fill_value=-999)

                    triangle = tri.Triangulation(mesh[:,0],mesh[:,1])
                    im = plt.tricontourf(triangle, sln.real, levels ,cmap = 'coolwarm') # coldwarm jets

                    iidd = np.where(mesh[:,0] > 0.6)[0]
                    triangle = tri.Triangulation(mesh[iidd,0],mesh[iidd,1])
                    plt.tricontour(triangle, bubble[iidd], levels=np.array([0.0]),colors = 'magenta',linestyles='dashed')
                    #plt.tricontour(triangle, bubble, levels=np.array([-999]),colors = 'black',linestyles='solid')
            else:
                sln = var.copy()
                sln = np.ma.masked_invalid(sln)
                sln = sln.filled(fill_value=-999)
                triangle = tri.Triangulation(mesh[:,0],mesh[:,1])
                plt.tricontourf(triangle, sln.real, levels ,cmap = 'coolwarm') # coldwarm jets

            cbar = plt.colorbar(im)

            # Plot boundary layer edge
            if self._trip_loc:
                idx = np.where(bledge[:,0] > self._trip_loc)[0]
                plt.plot(bledge[idx,0],bledge[idx,1],'k--')
                idx = np.where(bledge[:,0] < self._trip_loc)[0]
                plt.plot(bledge[idx,0],bledge[idx,1],'k--')
            else:
                plt.plot(bledge[idx,0],bledge[idx,1],'k--')

            plt.ylim([-0.15,0.15])
            plt.xlim([-0.047,1.02])
            plt.tight_layout()
            plt.savefig(f'{self.dir}/figs/{varname[i]}_meanflow.png')
        plt.show()


    def _sample_location(self, mesh, soln, xmesh, xsoln):
        mesh, xmesh = mesh/100, xmesh/100
        levels = np.linspace(0.0,0.45,30)
        # Mask invalid location, i.e. wall
        soln[:,0] = np.NaN

        # Trip location
        plt.figure()
        if self._trip_loc:
            for i in range(2):
                if i==0:
                    index = np.where(mesh[:,0,0] > self._trip_loc/100)[0]
                else:
                    index = np.where(mesh[:,0,0] < self._trip_loc/100)[0]
                sln = soln.copy()
                sln[index] = np.NaN
                sln = np.ma.masked_invalid(sln)
                sln = sln.filled(fill_value=-999)

                msh = mesh.reshape(-1, self.ndims)
                sln = sln.reshape(-1, self.nvars)
                triangle = tri.Triangulation(msh[:,0],msh[:,1])
                im = plt.tricontourf(triangle, sln[:,1], levels ,cmap = 'coolwarm') # coldwarm jets
            cbar = plt.colorbar(im)
        # Probes location
        for i in range(xmesh.shape[0]):
            # Invalid position due to interpolation
            index = np.where(np.isnan(xsoln[i,:,0]) == False)[0]
            plt.plot(xmesh[i,index,0],xmesh[i,index,1],'r--')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.show()


    def _get_stress_tensor(self, msh):
        # Load stress tensor from surface stress calculation
        try:
            f = h5py.File(f'{self.dir}/tau.s','r')
            mesh = np.array(f['mesh'])
            tau = np.array(f['tau'])
            f.close()
        except:
            raise RuntimeError('Run cp cf alogrithm first and copy tau to this directory')

        #tau = np.sum(tau.reshape(9,-1, order = 'F'), axis = 0)
        #tau = tau[0,1]
        tau = tau

        # Do interpolation
        return np.interp(msh[:,0], mesh[:,0]*100, abs(tau))

    def _plot_session_bl_profile(self, mesh, soln, xloc):
        npts, nlayers, nvars = soln.shape
        vtot = np.linalg.norm(soln[...,1:self.ndims], axis = -1)
        rho = soln[...,0]
        y = np.linalg.norm(mesh[...,:2] - mesh[:,0,:2][:,None], axis = -1)


        tau = self._get_stress_tensor(mesh[:,0,:2])

        # Viscosity
        c = self._constants
        mu = c['mu']

        if self._viscorr == 'sutherland':
            p = soln[...,-1]
            cpT = c['gamma']*p/rho/(c['gamma'] - 1)
            Trat = cpT/c['cpTref']
            mu *= (c['cpTref'] + c['cpTs'])*Trat**1.5 / (cpT + c['cpTs'])

        # Wall units
        plt.figure()
        for i in range(npts):
            # Drop nan due to interpolation
            index = np.where(np.isnan(vtot[i]) == False)[0]
            ut = np.sqrt(tau[i]/rho[i,index])
            nu = mu[i,index]/rho[i,index]
            delta_nu = nu/ut
            yplus = y[i,index]/delta_nu
            uplus = vtot[i,index]/ut
            plt.plot(yplus, uplus, label = f'x/c = {xloc[i]}')

        # Log law
        kappa, B = 0.41, 5.0
        index = np.where(yplus > 5)[0]
        plt.plot(yplus[index], (1/kappa)*np.log(yplus[index]) + B, '--', label = 'Log law')
        # Linear law
        index = np.where(yplus < 10)[0]
        plt.plot(yplus[index],yplus[index],'.-',label = 'Linear law')

        plt.legend()
        plt.xscale('log')
        plt.xlabel('$y^+$')
        plt.ylabel('$u^+$')
        plt.tight_layout()
        plt.savefig(f'{self.dir}/figs/velocity_profile_inner_scale.eps')
        plt.show()
        #"""

        # Mean profile scaled in outer units
        idx = []
        plt.figure()
        for i in range(npts):
            # Drop nan due to interpolation
            index = np.where(np.isnan(vtot[i]) == False)[0]

            deltas = np.trapz(1 - rho[i,index]*vtot[i,index]/(rho[i,index[-1]]*vtot[i,index[-1]]), y[i, index])
            plt.plot(vtot[i,index]/vtot[i,index[-5]], y[i,index]/deltas,label = f'x/c = {xloc[i]}')

            # Get boundary layer edge position
            vedge = vtot[i,index[-5]]
            for j in range(len(y[i])-3):
                if vtot[i,j] > 0.99*vedge:
                    idx.append(mesh[i,j])
                    break
        #plt.legend()
        #plt.xlabel('$\lVert u/U_{\infty} \rVert$')
        plt.xlabel('$||u/U_e||$')
        plt.ylabel('$y/\delta^*$')
        plt.ylim([0,8])
        plt.savefig(f'{self.dir}/figs/velocity_profile.eps')
        plt.show()
        raise RuntimeError
        return np.array(idx)



    def _plot_session_bl_profile_berlin(self, mesh, soln, xloc):
        npts, nlayers, nvars = soln.shape
        vtot = np.linalg.norm(soln[...,1:self.ndims], axis = -1)
        rho = soln[...,0]
        y = np.linalg.norm(mesh[...,:2] - mesh[:,0,:2][:,None], axis = -1)

        tau = self._get_stress_tensor(mesh[:,0,:2])

        # Wall units
        plt.figure()
        for i in range(npts):
            # Drop nan due to interpolation
            index = np.where(np.isnan(vtot[i]) == False)[0]
            ut = np.sqrt(np.sum(tau[i])/rho[i,index])
            nu = self._constants['mu']/rho[i,index]
            delta_nu = nu/ut
            yplus = y[i,index]/delta_nu
            uplus = vtot[i,index]/ut
            plt.plot(yplus, uplus, label = f'x/c = {xloc[i]}')

        # Log law
        kappa, B = 0.41, 5.2
        index = np.where(yplus > 5)[0]
        plt.plot(yplus[index], (1/kappa)*np.log(yplus[index]) + B, '--', label = 'Log law')
        # Linear law
        index = np.where(yplus < 10)[0]
        plt.plot(yplus[index],yplus[index],'.-',label = 'Linear law')

        plt.legend()
        plt.xscale('log')
        plt.xlabel('$y^+$')
        plt.ylabel('$u^+$')



        # Mean profile
        plt.figure()
        for i in range(npts):
            plt.plot(vtot[i]/self._constants['uInf'], y[i]/np.max(mesh[:,0,0]),label = f'x/c = {xloc[i]}')
        plt.legend()
        #plt.xlabel('$\lVert u/U_{\infty} \rVert$')
        plt.xlabel('$||u/U_{\infty}||$')
        plt.ylabel('$y/c$')

        # Mean profile scaled in outer units
        plt.figure()
        #deltas = [np.trapz(1 - vtot[i]/np.mean(vtot[i,-5:], axis = -1), y[i]) for i in range(npts)]
        for i in range(npts):
            #plt.plot(vtot[i]/np.mean(vtot[i,-5:], axis = -1), y[i]/deltas[i],label = f'x/c = {xloc[i]}')
            # Drop nan due to interpolation
            index = np.where(np.isnan(vtot[i]) == False)[0]

            deltas = np.trapz(1 - rho[i,index]*vtot[i,index]/(rho[i,index[-1]]*vtot[i,index[-1]]), y[i, index])
            print(deltas)
            plt.plot(vtot[i,index]/vtot[i,index[-1]], y[i,index]/deltas,label = f'x/c = {xloc[i]}')
        plt.legend()
        #plt.xlabel('$\lVert u/U_{\infty} \rVert$')
        plt.xlabel('$||u/U_e||$')
        plt.ylabel('$y/\delta^*$')




        # External results
        import scipy.io
        mat = scipy.io.loadmat(f'{self.dir}/data_BLexp_U30_AoAeff3_xdc94_tripzz0d4.mat')
        print(mat.keys())
        mat['uBL'] = mat['uBL'].reshape(-1)
        mat['dist2wall'] = mat['dist2wall'].reshape(-1)

        plt.figure()
        i = -1
        index = np.where(np.isnan(vtot[i]) == False)[0]
        deltas = np.trapz(1 - rho[i,index]*vtot[i,index]/(rho[i,index[-1]]*vtot[i,index[-1]]), y[i, index])
        plt.plot(vtot[i,index]/vtot[i,index[-1]], y[i,index]/deltas,label = f'LES')
        deltas = np.trapz(1 - mat['uBL']/mat['uBL'][-1], mat['dist2wall'])
        print(mat['dist2wall'],mat['uBL'],deltas)
        plt.plot(mat['uBL']/mat['uBL'][-1], mat['dist2wall']/deltas,'.', label = 'Exp')
        plt.legend()
        #plt.xlabel('$\lVert u/U_{\infty} \rVert$')
        plt.xlabel('$||u/U_e||$')
        plt.ylabel('$y/\delta^*$')


        plt.figure()
        i = -1
        index = np.where(np.isnan(vtot[i]) == False)[0]
        plt.plot(vtot[i,index]/vtot[i,index[-1]], y[i,index]/100,label = f'LES')
        plt.plot(mat['uBL'][1:]/mat['uBL'][-1], mat['dist2wall'][1:]/100 + 0.005,'.', label = 'Exp')
        plt.legend()
        #plt.xlabel('$\lVert u/U_{\infty} \rVert$')
        plt.xlabel('$||u/U_e||$')
        plt.ylabel('$y/c$')


        plt.figure()
        # Y is corrected by density
        i = -1
        index = np.where(np.isnan(vtot[i]) == False)[0]

        Y = [np.trapz(rho[i,index[:id]] / rho[i,index[-1]],y[i,index[:id]]/100) for id in range(1,len(index))]
        Y = np.array([0] + Y)
        plt.plot(vtot[i,index]/vtot[i,index[-1]], Y,label = f'LES')
        plt.plot(mat['uBL'][1:]/mat['uBL'][-1], mat['dist2wall'][1:]/100 + 0.005,'.', label = 'Exp')
        plt.legend()
        #plt.xlabel('$\lVert u/U_{\infty} \rVert$')
        plt.xlabel('$||u/U_e||$')
        plt.ylabel('$\eta$')
        plt.show()


    def _plot_session(self, info):

        for id, inf in info.items():
            if id == 0:
                mesh, soln, meshw = inf['mesh'], inf['soln'], inf['intmesh']
                levels = [np.linspace(np.min(soln[:,i]),np.max(soln[:,i]),50) for i in range(self.nvars)]
            else:
                meshw = inf['intmesh']

            # Bad points on the wall
            index = [np.argsort(np.linalg.norm(mesh[:,:2] - msh[:2] , axis = 1))[0] for msh in meshw[:,0]]
            #plt.figure()
            #plt.plot(mesh[index,0],mesh[index,1],'.')
            soln[index] = np.NaN
            soln = np.ma.masked_invalid(soln)
            soln = soln.filled(fill_value=-999)
            # Bad points on the tripping boundary
            if self._trip_loc:
                id1 = np.where(meshw[:,0,1] > 0)[0]
                id11 = np.argsort(abs(meshw[id1,0,0] - self._trip_loc))[0]
                id2 = np.where(meshw[:,0,1] < 0)[0]
                id22 = np.argsort(abs(meshw[id2,0,0] - self._trip_loc))[0]
                index = [id1[id11],id2[id22]]
                index = [np.argsort(np.linalg.norm(mesh[:,:2] - msh[:2] , axis = 1))[0]  for idx in index for msh in meshw[idx]]
                soln[index] = np.NaN
                soln = np.ma.masked_invalid(soln)
                soln = soln.filled(fill_value=-999)

        plt.figure()
        triangle = tri.Triangulation(mesh[:,0],mesh[:,1])
        plt.tricontourf(triangle, soln[:,0], levels[0] ,cmap = 'coolwarm') # coldwarm jets
        cbar = plt.colorbar()
        plt.show()

    def _interpolation(self, mesh, soln, meshw):
        # Interpolation function
        print(mesh.shape, soln.shape)
        # Use scipy to do linear 2D interpolation
        from scipy.interpolate import LinearNDInterpolator
        interp = LinearNDInterpolator(mesh[:,:2], soln)

        npts, nlayer, ndims = meshw.shape
        msh = meshw.reshape(-1, ndims)
        return interp(msh[:,:2]).reshape(npts, nlayer, soln.shape[-1])




    def _interpolation2(self, mesh, soln, meshw, vecm):
        # Interpolation function
        print(mesh.shape, soln.shape, vecm.shape)
        # Use scipy to do linear 2D interpolation
        from scipy.interpolate import LinearNDInterpolator
        interpU = LinearNDInterpolator(mesh[:,:2], soln[:,1]/soln[:,0])
        interpV = LinearNDInterpolator(mesh[:,:2], soln[:,2]/soln[:,0])

        npts, nlayer, ndims = meshw.shape
        msh = meshw.reshape(-1, ndims)
        U = interpU(msh[:,:2]).reshape(npts, nlayer)
        V = interpV(msh[:,:2]).reshape(npts, nlayer)
        u_t = np.einsum('ij, i -> ij',U,vecm[0]) + np.einsum('ij, i -> ij',V,vecm[1])
        #u_n = np.einsum('ij, i -> ij',U,vecm[2]) + np.einsum('ij, i -> ij',V,vecm[3])

        # Creat a dictionary to store all useful variables
        return {'mesh': mesh, 'soln': soln, 'intut': u_t, 'intmesh':meshw}

        id = 10
        y = ((meshw[id,:,0] - meshw[id,0,0])**2 + (meshw[id,:,1] - meshw[id,0,1])**2)**0.5
        plt.figure()
        plt.plot(u_t[id],y,'.')
        #return u_t, u_n


        index = np.where(msh[:,0] > 0.2)[0]

        levels = np.linspace(-0.01,0.5,60)

        # Get bad points
        u_t[:,0] = np.NaN
        u_t = np.ma.masked_invalid(u_t)
        u_t = u_t.filled(fill_value=-999)
        # On the wall it should be zeros
        #u_t[:,0] = -999
        sln = u_t.reshape(-1)



        plt.figure()
        triangle = tri.Triangulation(msh[:,0],msh[:,1])
        plt.tricontourf(triangle, sln,levels,cmap = 'coolwarm') # coldwarm jets
        cbar = plt.colorbar()

        plt.plot(meshw[:,0,0],meshw[:,0,1],'.')
        plt.plot(meshw[id,:,0],meshw[id,:,1],'.')


        # BL thickness
        blthick = {}
        for ly in range(u_t.shape[1]-1):
            index = np.where(u_t[:,ly]/u_t[:,ly+1] > 0.991)[0]
            for id in index:
                if id not in blthick:
                    blthick[id] = ly
        bl = []
        for id, ly in blthick.items():
            bl.append((meshw[id,0], y[ly]))



        plt.figure()
        triangle = tri.Triangulation(msh[:,0],msh[:,1])
        plt.tricontourf(triangle, sln,levels,cmap = 'coolwarm') # coldwarm jets
        cbar = plt.colorbar()

        plt.plot(meshw[:,0,0],meshw[:,0,1],'.')
        for id, ly in blthick.items():
            plt.plot(meshw[id,ly,0],meshw[id,ly,1],'r.')


        # Original
        plt.figure()
        triangle = tri.Triangulation(mesh[:,0],mesh[:,1])
        plt.tricontourf(triangle, soln[:,1]/soln[:,0],levels,cmap = 'coolwarm') # coldwarm jets
        cbar = plt.colorbar()
        plt.plot(meshw[:,0,0],meshw[:,0,1],'.')

        plt.show()

class BL_Coeff(BL_base):
    def __init__(self, argv, icfg, fname):
        super().__init__(argv, icfg, fname)

    def _load_mesh_soln(self):
        # A strict rule about element type here
        f = h5py.File(f'{self.dir}/grad_mean.s','r')
        mesh = np.array(f['mesh'])
        soln = np.array(f['soln'])
        f.close()

        mesh = mesh /self.L
        self._trip_loc = self._trip_loc / self.L

        return mesh, soln

    def main_proc(self):
        # Load averaged soln and mesh file
        mesh, soln = self._load_mesh_soln()
        print(mesh.shape, soln.shape)

        # Reorder points
        mesh, soln = self._reorder_pts(mesh, soln)

        mesh_0 = mesh @ self.rot_map()


        cp, cf, mp = defaultdict(list), defaultdict(list), defaultdict(list)
        tau = defaultdict(list)
        for i in range(2):
            # seperate the upper and lower side
            if i == 0:
                index = np.where(mesh_0[:,1] > 0)[0]
            else:
                index = np.where(mesh_0[:,1] < 0)[0]
            msh = mesh[index]
            sln = soln[index]

            # Sort along x-axis
            index = np.argsort(msh[:,0])
            msh, sln = msh[index], sln[index]

            # Splite the region before and after the tripping location
            mm, sm = [], []
            if self._trip_loc:
                index = np.where(msh[:,0] > self._trip_loc)[0]
                mm.append(msh[0:index[0]])
                mm.append(msh[index[0]:])
                sm.append(sln[0:index[0]])
                sm.append(sln[index[0]:])
            else:
                mm.append(msh)
                sm.append(sln)


            for msh, sln in zip(mm, sm):
                if len(msh) == 0:
                    continue
                if i == 0:
                    side = 1
                else:
                    side = -1
                _msh, _cp, _cf, _tau = self.cal_cp_cf(msh, sln, side)
                cp[i].append(_cp)
                cf[i].append(_cf)
                tau[i].append(_tau)
                mp[i].append(_msh)

        self._plot_session(mp, cp, cf, tau)

    def _plot_session(self, mesh, cp, cf, tau):
        # Line plots Cp
        plt.figure()
        for k, v in cp.items():
            for msh, _cp in zip(mesh[k],v):
                if k == 0:
                    plt.plot(msh[:,0],_cp,'r')
                else:
                    plt.plot(msh[:,0],_cp,'k-.')
        plt.xlabel('$x/c$')
        plt.ylabel('$-C_p$')
        plt.tight_layout()
        plt.savefig(f'{self.dir}/figs/cp.eps')
        plt.figure()
        for k, v in cf.items():
            for msh, _cf in zip(mesh[k],v):
                if k == 0:
                    plt.plot(msh[:,0],_cf,'r')
                else:
                    plt.plot(msh[:,0],_cf,'k-.')
            plt.axhline(y = 0.0, color = 'b', linestyle = '--')
        plt.xlabel('$x/c$')
        plt.ylabel('$C_f$')

        # Insert text
        #plt.figtext(0.75, 0.9, 'Re80k rough')

        # Have a small plot inside
        """
        # location for the zoomed portion
        sub_axes = plt.axes([.4, .2, .45, .35])

        # plot the zoomed portion
        for k, v in cf.items():
            for msh, _cf in zip(mesh[k],v):
                idx = np.where(msh[:,0] > 0.6528)[0]  # rough 0.7
                idx1 = np.where(msh[idx,0] < 0.97739)[0] # rough 0.943
                idx = idx[idx1]
                if k == 0:
                    sub_axes.plot(msh[idx,0], _cf[idx], 'r')
                else:
                    sub_axes.plot(msh[idx,0], _cf[idx], 'k-.')
            plt.axhline(y = 0.0, color = 'b', linestyle = '--')
        #"""



        plt.tight_layout()
        plt.savefig(f'{self.dir}/figs/cf_Re80k_rough.eps')
        plt.show()

        #"""
        f = h5py.File(f'{self.dir}/tau.s','w')
        for k, v in tau.items():
            tt = []
            for msh, _tau in zip(mesh[k],v):
                if tt == []:
                    tt = _tau
                    mmsh = msh
                else:
                    tt = np.append(tt, _tau, axis = -1)
                    mmsh = np.append(mmsh, msh, axis = 0)
            f[f'tau'] = tt
            f[f'mesh'] = mmsh
            break
        f.close()
        #"""

    def cal_cp_cf(self, mesh, soln, side):
        mesh[:,-1] = 0

        # Get normal direction
        vect = mesh[1:] - mesh[:-1]
        vect = np.append(vect, vect[-1][None,:], axis = 0)

        # Be sure that no duplicated points
        index = np.where(np.linalg.norm(vect, axis=1) > 1e-10)[0]
        mesh, soln, vect = mesh[index], soln[index], vect[index]

        # Reshape vectors
        npts, nvars = soln.shape

        # Normalise tangential vector
        vect = vect/np.linalg.norm(vect, axis = -1)[:,None]
        # Calculate normal vector
        vecn = np.cross(vect,np.array([0,0,1]))

        # Calculate stress
        du = soln.T[self.nvars:]
        u = soln.T[:self.nvars]
        tau = self.stress_tensor(du, u)

        print(tau.shape, vecn.shape)

        # Calculate cf
        #tau = np.einsum('ik, jki -> jki', vecn, tau)
        # Angle between wall norm and cartician norm y
        angle = np.einsum('ik, k -> i', vect, np.array([1,0,0]))
        angle = np.arccos(angle)
        print(angle.shape)
        #tau = -0.5*(tau[0,0] -tau[1,1])*np.sin(2*angle) + tau[0,1]*np.cos(2*angle)
        tau = -0.5*(tau[0,0] -tau[1,1])*(2*np.sqrt(1-angle**2)*angle) + tau[0,1]*(2*angle**2 - 1)
        #cf = tau[0,1]/self._dyna_p
        cf = tau/self._dyna_p
        # Get wall friction coefficient
        #cf = cf[1,0] * side

        # Calculate cp
        cp = (u[-1] - self._pinf)/self._dyna_p

        # Reshape
        return mesh.reshape(npts, self.ndims), cp, cf, tau


    def _duplpts_pd(self, mesh, subset):
        # Use panda to fast drop duplicated points
        import pandas as pd
        df = pd.DataFrame({'x':mesh[:,0], 'y':mesh[:,1], 'z':mesh[:,2]})
        return df.drop_duplicates(subset=subset).values



class Drag(BL_base):
    def __init__(self, argv, icfg, fname):
        super().__init__(argv, icfg, fname)

        self.Re = '100k'
        self.type = 'rough_u'

    def _load_mesh(self):
        #f = h5py.File(f'{self.dir}/grad_timesereis2.s','r')
        f = h5py.File(f'{self.dir}/grad_mean.s','r')
        mesh = np.array(f['mesh'])
        #soln = np.array(f['soln'])[:200].swapaxes(0,1)
        soln = np.array(f['soln'])
        f.close()

        soln = np.append(soln[:,None,:],soln[:,None,:], axis = 1)

        print(mesh.shape, soln.shape)
        return mesh, soln

    def main_proc(self):
        """
        Idea of drag calculation is done through triangulization points array
        to triangles. Area is calculated for each triangle and integration is
        done inside each triangle. To do this, grad data is used. Note that use
        grad data after dropping all duplicated points.
        Delaunay will output triangles only when the mesh is 2D. So some
        deformation is needed to mesh file.
        And Delaunay will process convex geometry, so tolerence correction to
        mesh file is needed.
        """
        mesh, soln = self._load_mesh()
        mesh, soln, mesh_rough = self._pre_proc_field(mesh, soln)

        #plt.figure()
        #plt.plot(mesh[:,0],mesh[:,-1],'.')
        #plt.show()

        mesh, soln, area, norm = self._cal_area_norm(mesh, soln, mesh_rough)

        """
        # Write area information into files
        f = h5py.File('./figs_2023_09_21/drag.s','a')
        try:
            print('area', np.sum(area))
            f[f'{self.Re}_{self.type}_A'] = np.sum(area)
        except:
            for k in f:
                if 'A' in k.split('_'):
                    a = f[f'{self.Re}_{self.type}_A']
                    a[...] = np.sum(area)
        f.close()
        """

        #raise RuntimeError

        #self._cal_forces(mesh, soln, area, norm)
        print(soln.shape)
        ntime = soln.shape[2]
        fp, fx, fz = [], [], []
        for t in range(ntime):
            _fp, _fx, _fz = self._cal_forces(mesh, soln[:,:,t], area, norm)
            fp.append(_fp)
            fx.append(_fx)
            fz.append(_fz)

        plt.figure()
        plt.plot(np.arange(ntime),fp,'-')
        plt.plot(np.arange(ntime),fx,'--')
        plt.plot(np.arange(ntime),fz,':')
        plt.plot(np.arange(ntime),np.array(fp)+np.array(fx),'-.')
        plt.show()


        """
        f = h5py.File('./figs_2023_09_21/drag.s','a')
        try:
            f[f'{self.Re}_{self.type}_fp'] = np.array(fp)
            f[f'{self.Re}_{self.type}_fx'] = np.array(fx)
            f[f'{self.Re}_{self.type}_fz'] = np.array(fz)
        except:
            for k in f:
                if 'fp' in k.split('_'):
                    a = f[f'{self.Re}_{self.type}_fp']
                    a[...] = np.array(fp)
                elif 'fx' in k.split('_'):
                    a = f[f'{self.Re}_{self.type}_fx']
                    a[...] = np.array(fx)
                elif 'fz' in k.split('_'):
                    a = f[f'{self.Re}_{self.type}_fz']
                    a[...] = np.array(fz)

        f.close()
        """



    def _cal_forces(self, mesh, soln, area, norm):
        nele, npt, nvars = soln.shape
        area = np.ones((nele, npt))*area[:,None]
        area = area.reshape(-1)
        print(soln.shape, norm.shape)
        norm = np.ones((nele, npt))[:,:,None, None]*norm[:,None,:,:]
        norm = norm.reshape(-1,3,3)

        # Get stress tensor
        ss = soln.reshape(-1, nvars).T
        u = ss[:self.nvars]
        du = ss[self.nvars:]
        tau = self.stress_tensor(du, u)
        #tau = tau.reshape(self.ndims, self.ndims, nele*npt, ntime)

        # Calculate stress force inside each element
        #f = tau*area[None, None,:,None]
        f = tau*area[None, None,:]

        print(f.shape,'qqqqq')

        # Force projection
        cart = np.array([[1,0,0],[0,1,0],[0,0,1]])

        alpha = np.einsum('lj,kij -> lik', cart, norm)

        #f = np.sum(np.einsum('ilk, lk -> imk', f, alpha[:,1]), axis = 0)
        #f = np.sum(f, axis = 0)
        #f = np.sum(np.einsum('ik, ilk -> ilk', f, alpha), axis = 1)
        #fx = f[1] * norm[:,1,0] + f[2] * norm[:,2,0]
        #fz = f[1] * norm[:,1,-1] + f[2] * norm[:,2,-1]
        #    np.einsum('ilk, lmk -> mk', f, alpha[1,1]) + \
        #    np.einsum('ilk, lmk -> mk', f, alpha[2,1]) )


        f = np.einsum('ik, lik -> ilk', f[:,0], alpha) + \
            np.einsum('ik, lik -> ilk', f[:,1], alpha) + \
            np.einsum('ik, lik -> ilk', f[:,2], alpha)

        #f = f.reshape(self.ndims*self.ndims,-1)
        #f = np.sum([np.einsum('k, ilk -> ilk', ff, alpha) for ff in f], axis = 0)
        #fx = np.sum(f[:,1], axis = 0)*norm[:,1,0] + np.sum(f[:,2], axis = 0)*norm[:,2,0]
        #fz = np.sum(f[:,1], axis = 0)*norm[:,2,-1] + np.sum(f[:,2], axis = 0)*norm[:,2,-1]
        #print(f.shape, norm.shape, cart.shape)

        # Get force in dreestream direction
        fx, fz = f[0,1],f[1,2]

        #"""
        plt.figure()
        mesh = mesh.reshape(-1,3)
        triangle = tri.Triangulation(mesh[:,0],mesh[:,-1])
        plt.tricontourf(triangle, np.log10(area), levels = 50)
        plt.colorbar()
        plt.show()
        #"""
        fx, fz = np.sum(fx), np.sum(fz)

        # Pressure force
        """
        plt.figure()
        mesh = mesh.reshape(-1,3)
        a = mesh + norm
        plt.plot(mesh[:,0],mesh[:,1],'.')
        plt.plot(a[:,0],a[:,1],'.')
        plt.show()
        """
        fp = u[-1]*area*norm[:,0,0]*(-1)
        fp = np.sum(fp)
        print(fp, fx, fz)

        return fp, fx, fz

    def _cal_area_norm(self, mesh, soln, mesh_rough):
        # Use Delaunay to create triangles
        from scipy.spatial import Delaunay
        tri = Delaunay(mesh_rough[:,[0,2]])

        """
        print(tri.simplices.shape, tri.simplices)
        print(mesh[tri.simplices[0]])

        plt.figure()
        plt.triplot(mesh[:,0],mesh[:,-1],tri.simplices)
        plt.plot(mesh[:,0], mesh[:,-1], 'o')
        plt.figure()
        plt.plot(mesh[tri.simplices[0],0],mesh[tri.simplices[0],1],'.')
        plt.show()

        plt.figure()
        plt.triplot(mesh[:,0], mesh[:,-1], tri.simplices.copy())
        plt.plot(mesh[tri.convex_hull][...,0],mesh[tri.convex_hull][...,-1],'or')
        plt.show()
        raise RuntimeError

        #"""

        # Divie mesh file into triangles
        mesh = np.array([mesh[index] for index in  tri.simplices])
        soln = np.array([soln[index] for index in  tri.simplices])

        # Calculate face normal and area,
        # 2D Delaunay triangles points arrangement is counter-clockwise
        vvect = np.cross(mesh[:,0]-mesh[:,1], mesh[:,2] - mesh[:,1], axis = -1)
        area = np.linalg.norm(vvect, axis = -1)
        norm = vvect/area[:,None]
        area /= 2*3

        # Calculate surface tangential vectors
        tan1 = (mesh[:,0]-mesh[:,1]) / np.linalg.norm(mesh[:,0]-mesh[:,1],axis = -1)[:,None]
        print(np.max(np.linalg.norm(tan1, axis = -1)),np.min(np.linalg.norm(tan1, axis = -1)),np.any(np.isnan(np.linalg.norm(tan1, axis = -1))))
        tan2 = np.cross(norm, tan1, axis = -1)
        print(np.max(np.linalg.norm(tan2, axis = -1)),np.min(np.linalg.norm(tan2, axis = -1)),np.any(np.isnan(np.linalg.norm(tan2, axis = -1))))

        norm = np.concatenate([norm[:,None,:],tan1[:,None,:],tan2[:,None,:]], axis = 1)

        """
        print(np.where(abs(np.linalg.norm(norm, axis = -1) - 1) > 1e-6))
        index = np.where(norm[:,-1] > 1e-4)[0]
        print(norm[index])
        print(norm.shape, mesh.shape)
        """

        # Check if norm is working well
        if self.type.split('_')[-1] == 'l':
            mesh[...,1] *= -1
            norm[:,0,1] *= -1
        plt.figure()
        a = mesh + norm[:,0,:][:,None,:]
        plt.plot(mesh[:,1,0],mesh[:,1,1],'.')
        plt.plot(a[:,1,0],a[:,1,1],'.')
        #plt.show()

        return mesh, soln, area, norm

    def _pre_proc_field(self, mesh, soln):
        mm, ss = [], []
        # Split mesh into half along y axis
        if self.type.split('_')[-1] == 'l':
            mesh[:,1] *= -1
        index = np.where(mesh[:,1] >= -0.0001)[0]
        #idx = np.where(mesh[index,0] > 60)[0]
        #idx1 = np.where(mesh[index[idx],0] <90)[0]
        #index = index[idx[idx1]]
        mesh, soln = mesh[index], soln[index]

        # Correct mesh periodic bc to make sure it is convex
        for i in [0,-1]:
            for j in [np.min(mesh[:,i]),np.max(mesh[:,i])]:
                index = np.where(abs(mesh[:,i] - j)< 1e-4)[0]
                mesh[index,i] = j

        mesh_rough = mesh.copy()

        # Deal with 3D geometry on the aerofoil surface
        if self.type.split('_')[0] == 'rough':
            id0 = np.where(mesh[:,0] > 51)[0]
            id1 = np.where(mesh[id0,0] < 53)[0]
            id2 = np.where(abs(mesh[id0[id1],1]) > 5.4)[0]
            idx = id0[id1[id2]]
            for id in range(2):
                if id == 0:
                    id2 = np.where(mesh[idx,-1] > -9)[0]
                else:
                    id2 = np.where(mesh[idx,-1] < -9)[0]
                index = idx[id2]
                cmesh = np.mean(mesh[index], axis = 0)
                mesh_rough[index] = (mesh[index] - cmesh)*0.8 + cmesh

        return mesh, soln, mesh_rough



    def _cal_forces2(self, mesh, soln, area, norm):
        nele, npt, nvars = soln.shape

        # Get stress tensor
        ss = soln.reshape(-1, nvars).T
        u = ss[:self.nvars]
        du = ss[self.nvars:]
        tau = self.stress_tensor(du, u)
        tau = tau.reshape(self.ndims, self.ndims, nele, npt)

        # Rotate stress tensor
        alpha = np.einsum('ij,j -> i', norm[:,1], np.array([1,0,0]))
        beta = np.einsum('ij,j -> i', norm[:,0], np.array([0,1,0]))
        gamma = np.einsum('ij,j -> i', norm[:,2], np.array([0,0,1]))
        angle = []
        for id, i in zip([0,1,2],[alpha, beta, gamma]):
            if 1 < np.max(abs(i)) and 1 + 1e-5 > np.max(abs(i)) :
                index = np.where(abs(i) - 1 > 0)[0]
                i[index] = 1*i[index]/abs(i[index])
                angle.append(np.arccos(i))
            elif 1 >= np.max(abs(i)):
                angle.append(np.arccos(i))
                continue
            else:
                print(id, np.max(i),np.min(i))
                raise RuntimeError

        alpha, beta, gamma = angle[0],angle[1],angle[2]

        R = np.zeros([len(norm),3,3])
        R[:,0,0] = np.cos(alpha)*np.cos(beta)
        R[:,0,1] = np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma)
        R[:,0,2] = np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma)
        R[:,1,0] = np.sin(alpha)*np.cos(beta)
        R[:,1,1] = np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma)
        R[:,1,2] = np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma)
        R[:,2,0] = -np.sin(beta)
        R[:,2,1] = np.cos(beta)*np.sin(gamma)
        R[:,2,2] = np.cos(beta)*np.cos(gamma)

        tau = np.einsum('kij, ijkl -> ijkl', R, tau)
        #tau = np.einsum('ijkl, kij -> ijkl', tau, np.linalg.inv(R))
        tau = np.einsum('ijkl, kji -> ijkl', tau, R)

        print(np.max(tau[0,2]),np.min(tau[0,2]))
        print(np.max(tau[0,1]),np.min(tau[0,1]))
        print(np.max(tau[1,-1]),np.min(tau[1,-1]))
        """
        # \sigma'\!_{xx}  =  \sigma_{xx} * cos^2(theta) + \sigma_{yy} * sin^2(theta) + 2 * \tau_{xy} * sin(theta)*cos(theta)
        # \sigma'\!_{yy}  =  \sigma_{xx} * sin^2(theta) + \sigma_{yy} * cos^2(theta) - 2 * \tau_{xy} * sin(theta)*cos(theta)
        # \tau'\!_{xy}  =  (\sigma_{yy} - \sigma_{xx}) sin(theta)*cos(theta) + \tau_{xy} * (cos^2(theta) - sin^2(theta))
        theta = np.arccos(np.einsum('ij,j -> i', norm, np.array([0,1,0])))
        theta = np.concatenate([theta[:,None], theta[:,None], theta[:,None]], axis = -1)

        print(theta.shape)
        txy = (tau[1,1] - tau[0,0]) * np.sin(theta)*np.cos(theta) + tau[0,1] * np.cos(2*theta)

        print(np.max(tau[:,:,0,2]),np.min(tau[:,:,0,2]))
        print(np.max(tau[:,:,0,1]),np.min(tau[:,:,0,1]))
        print(np.max(txy),np.min(txy))
        print(np.max(tau[:,:,1,-1]),np.min(tau[:,:,1,-1]))
        """

        # Calculate skin friction force
        skin = (tau[0,1][:,:,None] * norm[:,1][:,None,:] + tau[1,2][:,:,None] * norm[:,2][:,None,:]) * area[:, None, None] / 3
        pressure = soln[:,:,self.nvars][:,:,None] * norm[:,0,:][:,None,:] * area[:, None, None] / 3

        skin = np.linalg.norm(skin, axis = -1)
        pressure = np.linalg.norm(pressure, axis = -1)
        print(skin.shape, pressure.shape, mesh.shape)

        plt.figure()
        mm = mesh.reshape(-1, 3)
        ss = skin.reshape(-1) + pressure.reshape(-1)
        triangle = tri.Triangulation(mm[:,0],mm[:,-1])
        plt.tricontourf(triangle, ss,levels=49,cmap = 'coolwarm')
        plt.colorbar()
        plt.show()

        print(np.sum(ss, axis = 0))



class BL_Coeff_hotmap(BL_base):
    def __init__(self, argv, icfg, fname):
        super().__init__(argv, icfg, fname)

    def _load_mesh_soln(self):
        # A strict rule about element type here
        f = h5py.File(f'{self.dir}/grad_mean.s','r')
        #f = h5py.File(f'{self.dir}/grad_timesereis2.s','r')
        mesh = np.array(f['mesh'])
        soln = np.array(f['soln'])
        #soln = np.array(f['soln'])[101]
        f.close()

        mesh = mesh /self.L
        self._trip_loc = self._trip_loc / self.L

        return mesh @ self.rot_map(), soln

    def main_proc(self):
        # Load averaged soln and mesh file
        mesh, soln = self._load_mesh_soln()
        print(mesh.shape, soln.shape)

        # Make sure NACA0012 part of aerofoil is used
        index = np.where(mesh[:,0] < 0.98)[0]
        mesh, soln = mesh[index], soln[index]

        mp, cp, cf = defaultdict(list), defaultdict(list), defaultdict(list)
        for i in range(2):
            if i == 0:
                index = np.where(mesh[:,1] > 0)[0]
                msh, sln = mesh[index], soln[index]
            else:
                index = np.where(mesh[:,1] > 0)[0]
                msh, sln = mesh[index], soln[index]

            # sort along x-axis
            index = np.argsort(msh[:,0])
            msh, sln = msh[index], sln[index]

            # Splite the region before and after the tripping location
            mm, sm = [], []
            if self._trip_loc:
                index = np.where(msh[:,0] > self._trip_loc)[0]
                mm.append(msh[0:index[0]])
                mm.append(msh[index[0]:])
                sm.append(sln[0:index[0]])
                sm.append(sln[index[0]:])
            else:
                mm.append(msh)
                sm.append(sln)

            for msh, sln in zip(mm, sm):
                if i == 0:
                    side = -1
                else:
                    side = 1
                _msh, _cp, _cf = self.cal_cp_cf(msh, sln, side)
                cp[i].append(_cp)
                cf[i].append(_cf)
                mp[i].append(_msh)

        self._plot_session(mp, cp, cf)

    def _plot_session(self, mesh, cp, cf):
        # Contour plot skin friction field
        side = ['upper','lower']
        for i in cf:
            # get levels
            abound = []
            for mm, _cf in zip(mesh[i], cf[i]):
                abound += [np.min(_cf),np.max(_cf)]
            levels = np.linspace(np.min(abound),np.max(abound),10)

            plt.figure(figsize=(20,4))
            for mm, _cf in zip(mesh[i], cf[i]):

                #levels = np.linspace(np.min(_cf),np.max(_cf),50)
                levels = np.linspace(-0.002,0.01,50)

                print(mm.shape, _cf.shape)
                triangle = tri.Triangulation(mm[:,0],mm[:,-1])
                divnorm=colors.TwoSlopeNorm(vmin=np.min(levels), vcenter=0., vmax=np.max(levels))
                plt.tricontourf(triangle, _cf,levels,cmap = 'coolwarm', norm=divnorm,boundaries=[-0.005] + levels + [0.01],
                                extend='both',
                                extendfrac='auto',) # coldwarm jets



            cbar = plt.colorbar()
            # plot zero contour line
            for mm, _cf in zip(mesh[i], cf[i]):

                # Only plot the region close to the trailing edge
                if any(mm[:,0] < 0.33):
                    print(np.max(mm[:,0]))
                    continue

                plt.tricontour(triangle, _cf,levels=np.array([0.0]),linestyles='dashed', colors='grey'
                )

            """
            # plot cylinders
            index = np.where(abs(mm[:,1])>0.056)[0]
            idx1 = np.where(mm[index,-1] < -0.09)[0]
            triangle = tri.Triangulation(mm[index[idx1],0],mm[index[idx1],-1])
            plt.tricontourf(triangle, np.zeros(len(index[idx1])), cmap = 'Greys')
            idx1 = np.where(mm[index,-1] > -0.09)[0]
            triangle = tri.Triangulation(mm[index[idx1],0],mm[index[idx1],-1])
            plt.tricontourf(triangle, np.zeros(len(index[idx1])), cmap = 'Greys')
            #"""

            plt.xlabel('$x/c$')
            plt.ylabel('$z/c$')

            plt.savefig(f'{self.dir}/figs/cf_hotmap_{side[i]}.png')
            plt.tight_layout()
            plt.show()




    def NACA0012_norm_dir(self, mesh, soln):
        #y= +- 0.6*[0.2969*sqrt(x) - 0.1260*x - 0.3516*x2 + 0.2843*x3 - 0.1015*x4]
        dx = np.min([0.01, np.max(mesh[:,0]/1000)])
        # Force mesh to be away from negative values
        index = np.where(abs(mesh[:,0]) < 1e-10)[0]
        mesh[index,0] = 0
        if any(mesh[:,1] > 0):
            y = lambda x: 0.6*(0.2969*x**(0.5) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
            dy = y(mesh[:,0] + dx) - y(mesh[:,0])
            vect = np.zeros([3,len(dy)])
            vect[0], vect[1], vect[2] = dx, dy, 0

            #index = np.where(np.linalg.norm(vect, axis = 1) > 1e-10)[0]
            #mesh, soln, vect = mesh[index], soln[index], vect[:,index]
            vect = vect / np.linalg.norm(vect, axis = 0)[None]

            # Calculate normal vector
            vecn = np.cross(vect.T,np.array([0,0,-1]))

        else:
            y = lambda x: -0.6*(0.2969*x**(0.5) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
            dy = y(mesh[:,0] + dx) - y(mesh[:,0])
            vect = np.zeros([3,len(dy)])
            vect[0], vect[1], vect[2] = dx, dy, 0

            #index = np.where(np.linalg.norm(vect, axis = 1) > 1e-10)[0]
            #mesh, soln, vect = mesh[index], soln[index], vect[:,index]

            vect = vect / np.linalg.norm(vect, axis = 0)[None]
            # Calculate normal vector
            vecn = np.cross(vect.T,np.array([0,0,1]))

        return vecn



    def cal_cp_cf(self, mesh, soln, side):
        # Get normal direction from NACA0012 profile
        vecn =  self.NACA0012_norm_dir(mesh, soln)

        # Calculate stress
        du = soln.T[self.nvars:]
        u = soln.T[:self.nvars]
        tau = self.stress_tensor(du, u)

        # Calculate cf
        cf = np.einsum('ij, jki -> jki', vecn, tau)/self._dyna_p

        # Calculate cp
        cp = (u[-1] - self._pinf)/self._dyna_p

        return mesh, cp, cf[1,0]

    def _get_rid_of_dp_pts_cf(self, mesh, soln, index = []):
        mn, sn = [], []
        nvars = soln.shape[-1]
        mesh = mesh.reshape(-1, self.ndims)
        soln = soln.reshape(-1, nvars)
        for k, idx in enumerate(index):
            msh = mesh[idx]
            sln = soln[idx]

            # Use pandas to drop all duplicated points
            mm = self._duplpts_pd(msh,['z'])

            # Average duplicate points
            sln = [np.mean(sln[np.where(np.linalg.norm(pt - msh, axis = -1) < self.tol)[0]], axis = 0) for pt in mm]

            mn.append(mm)
            sn.append(sln)

        return mn, sn

    def _duplpts_pd(self, mesh, subset):
        # Use panda to fast drop duplicated points
        import pandas as pd
        df = pd.DataFrame({'x':mesh[:,0], 'y':mesh[:,1], 'z':mesh[:,2]})
        return df.drop_duplicates(subset=subset).values


class BL_wavenumber_trans(BL_base):
    """
    To do wave number transform, one has to interpolate field alone x direction
    and z direction. The thing about z direction is that our mesh has some very
    small error but could be bad for fft and it can save effort to reorder
    points. Then fft in z direction, fft in x direction, psd in time, we can get
    wavenumber, frequency in x and time repesct to each wavenumber in span.
    Define, kx = alpha, kz = beta and frequency f. Notice here none of them is angular.
    """
    def __init__(self, argv, icfg, fname):
        super().__init__(argv, icfg, fname)

        n = self.order + 1
        self.ele_map = [0, n -1, n*(n-1), n**2 - 1]
        self.chord = 100

    def _pre_proc_mesh(self):
        # A strict rule about element type here
        #f = h5py.File(f'{self.dir}/grad.m','r')
        f = h5py.File(f'./grad.m','r')
        mesh = np.array(f['hex']) / self.chord
        f.close()

        mesh = mesh @ self.rot_map()

        # Center of a element
        elec = np.mean(mesh[self.ele_map], axis = 0)

        """
        # Slite upper and lower side
        mm, idx = [], []
        for i in range(2):
            if i == 0:
                index = np.where(elec[:,1] > 0)[0]
            else:
                index = np.where(elec[:,1] < 0)[0]
            if self._trip_loc:
                index1 = np.where(elec[index,0] > self._trip_loc)[0]

            idx0 = index[index1]
            msh = mesh[:, idx0]




            #mn.append()
            #mm.append(mesh[index[index1]])

        """
        mesh = mesh.reshape(-1, self.ndims)

        # Slite upper and lower side
        mm, idx, mn, sp = [], [], [], []
        for i in range(2):
            if i == 0:
                index = np.where(mesh[:,1] > 0)[0]
            else:
                index = np.where(mesh[:,1] < 0)[0]
            if self._trip_loc:
                index1 = np.where(mesh[index,0] > self._trip_loc/self.chord)[0]

            idx.append(index[index1])
            msh = mesh[index[index1]]
            #plt.figure()
            #plt.plot(msh[:,0],msh[:,1],'.')
            #plt.show()
            # Create a new mesh for linear interpolation
            x = np.linspace(np.min(msh[:,0]) + 1/100,np.max(msh[:,0]) - 1/100, 100)
            z = np.linspace(np.min(msh[:,-1]) + 0.1/100,np.max(msh[:,-1]) - 0.1/100, 100)
            xv, zv = np.meshgrid(x, z)
            mn.append(np.array([xv.reshape(-1), zv.reshape(-1)]).T)
            mm.append(msh)
            sp.append(xv.shape)

        return idx, mm, mn, sp


    def _pre_proc_soln(self, index, mesh, meshn, shape):
        f = h5py.File(f'{self.dir}/field_pressure_timeseries_15285.00_23720.00.s', 'r')
        p = []
        for i in range(2):
            p.append(np.array(f[f'{i}']))
        f.close()

        # Use scipy to do linear 2D interpolation
        from scipy.interpolate import LinearNDInterpolator

        pn = []
        for idx, mm, mn, sp, _p in zip(index, mesh, meshn, shape, p):
            #plt.figure()
            #plt.plot(mn[:,0],mn[:,1],'.')
            #plt.plot(mm[:,0],mm[:,-1],'.')
            #plt.show()
            pp = _p.T
            nt = pp.shape[-1]
            interp = LinearNDInterpolator(mm[:,[0,2]], pp)
            (nx, nz) = sp
            pp = interp(mn).reshape(nx,nz,nt)
            print(np.min(pp))

            pn.append(pp)

        return pn

    def _get_pressure(self, index, mesh):
        pp = defaultdict(list)
        for t in self.time:
            print(t)
            f = h5py.File(f'{self.dir}/grad_{t}.s', 'r')
            soln = np.array(f['hex'])
            f.close()

            nvars = soln.shape[-1]
            soln = soln.reshape(-1, nvars).T
            rho, rhou, E = soln[0], soln[1:self.ndims+1], soln[self.nvars-1]
            p = E - 0.5*np.sum([rhou[i]**2 for i in range(self.ndims)], axis = 0)/rho
            p = p*(self._constants['gamma'] - 1)

            for id, idx in enumerate(index):
                pp[id].append(p[idx])

        f = h5py.File(f'{self.dir}/field_pressure_timeseries_{self.time[0]}_{self.time[-1]}.s', 'w')
        for k, v in pp.items():
            print(k,np.stack(v).shape, mesh[k].shape)
            f[f'{k}'] = np.stack(v)
            f[f'mesh_{k}'] = mesh[k]
        f.close()

    def main_proc(self):
        # Load soln and mesh file
        idx, mm, mn, sp = self._pre_proc_mesh()

        # First, precess pressure data into a file
        #self._get_pressure(idx, mm)
        #return 0

        pn = self._pre_proc_soln(idx, mm, mn, sp)

        for msh, p, s in zip(mn, pn, sp):
            #self._post_proc_field_alpha_f(msh, p, s)
            self._post_proc_field_alpha_beta(msh, p, s)
        #plt.show()

    def _post_proc_field_alpha_beta(self, msh, p, shape):
        # FFT in span
        pp = np.fft.fft(p, axis = 1) #/ (0.5* self._constants['rhoInf']*self._constants['uInf'])
        # FFT in streamwise
        pp = np.fft.fft(pp, axis = 0)
        # PSD in time
        f, pxx = self.psd(pp.conj())

        alpha = np.fft.fftfreq(shape[0], msh[1,0] - msh[0,0])
        alpha = np.fft.fftshift(alpha)
        beta = np.fft.fftfreq(shape[1], msh[0,1] - msh[0,0])
        beta = np.fft.fftshift(beta)
        pxx = np.fft.fftshift(pxx).real

        for ff in range(10):
            pp = pxx[:,:,ff]

            m0, m1 = np.meshgrid(alpha, beta)
            plt.figure()
            levels = np.linspace(np.min(pp), np.max(pp), 10)
            plt.contourf(m0, m1, pp.T, levels, locator = ticker.LogLocator(), cmap = 'binary')

            #plt.plot(np.array([-50*0.3,0]),np.array([50,0]),'--')

            plt.xlabel('$k_x$')
            plt.ylabel('$k_z$')
            #plt.colorbar()
            #plt.clim([1e-6,1e-3])
            plt.title(f'$f = {f[ff]}$')
            #plt.ylim([0,np.max(f)])
            #plt.ylim([0,30])
        plt.show()


    def _post_proc_field_alpha_f(self, msh, p, shape):
        # FFT in span
        pp = np.fft.fft(p, axis = 1) #/ (0.5* self._constants['rhoInf']*self._constants['uInf'])
        # FFT in streamwise
        pp = np.fft.fft(pp, axis = 0)
        # PSD in time
        f, pxx = self.psd(pp.conj())

        alpha = np.fft.fftfreq(shape[0], msh[1,0] - msh[0,0])
        alpha = np.fft.fftshift(alpha)
        f = np.fft.fftshift(f)
        pxx = np.fft.fftshift(pxx).real

        for beta in range(10):
            pp = pxx[:,beta]

            m0, m1 = np.meshgrid(alpha, f)
            plt.figure()
            levels = np.linspace(np.min(pp), np.max(pp), 10)
            plt.contourf(m0, m1, pp.T, levels, locator = ticker.LogLocator(), cmap = 'binary')

            #plt.plot(np.array([-50*0.3,0]),np.array([50,0]),'--')

            plt.xlabel('$k_x$')
            plt.ylabel('$f$')
            #plt.colorbar()
            #plt.clim([1e-6,1e-3])
            plt.title(f'$beta = {beta}$')
            #plt.ylim([0,np.max(f)])
            #plt.ylim([0,30])
        plt.show()



    def psd(self, soln):
        fs = 1/self.dt*self.chord/self._constants['uInf']
        nperseg = 256
        noverlap = int(nperseg*0.75)
        window = get_window('hann', nperseg)
        f, PSD = welch(soln, fs, window, nperseg, noverlap, scaling='density', axis = -1)
        return f, PSD
