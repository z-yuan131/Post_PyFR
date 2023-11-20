import numpy as np
from collections import defaultdict
import h5py

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import ticker


from base import Base
from pyfr.util import subclass_where
from pyfr.shapes import BaseShape
from pyfr.writers.vtk import BaseShapeSubDiv, VTKWriter

from feature.grad import Gradient
from feature.region import Region
import re


class Q_criterion(Gradient):
    _nodemaps, vtk_types_ho = VTKWriter._nodemaps, VTKWriter.vtk_types_ho

    def __init__(self, argv, icfg, fname):
        super().__init__(argv, icfg, fname)

        # Save data space
        self.dtype = np.float32

        # The box that elements are in
        self.box = icfg.getliteral(fname, 'box', [])

    def _get_eles(self):
        if len(self.box) > 0:
            eles = self._box_region()
        else:
            _, eles = self.get_wall_O_grid()
        return eles

    def _box_region(self):
        eles = {}
        for key in self.mesh:
            if 'spt' in key.split('_'):
                cmesh = np.mean(self.mesh[key], axis = 0)
                idx0 = np.where(cmesh[:,0] > self.box[0][0])[0]
                idx1 = np.where(cmesh[idx0,0] < self.box[1][0])[0]
                idx2 = np.where(cmesh[idx0[idx1],1] > self.box[0][1])[0]
                idx3 = np.where(cmesh[idx0[idx1[idx2]],1] < self.box[1][1])[0]
                index = idx0[idx1[idx2[idx3]]].tolist()

                if len(index) > 0:
                    eles[key] = index
        return eles

    def main_proc(self):
        # Prepare for MPI process
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            eles = self._get_eles()
        else:
            eles = None

        eles = comm.bcast(eles, root=0)

        mesh = self._pre_proc_mesh_local(eles)

        # Get time series
        time = self.get_time_series_mpi(rank, size)
        for t in time:
            print(t)
            self._proc_soln(t, eles, mesh)

    def _pre_proc_mesh_local(self, meles, op = False):
        mesh = {}
        for key, eles in meles.items():
            _, etype, part = key.split('_')
            nspts = self.mesh[key].shape[0]
            # Operator
            msh = self.mesh[key][:,eles]
            if op:
                mesh_op_vis = self._get_vis_op(nspts, etype, self.order)
                msh = np.einsum('ij, jkl -> ikl', mesh_op_vis, msh)

            try:
                mesh[etype] = np.append(mesh[etype], msh, axis = 1)
            except KeyError:
                mesh[etype] = msh
        return mesh

    def _proc_soln(self, time, eles, mesh):
        soln = f'{self.solndir}{time}.pyfrs'
        soln = self._load_snapshot(soln, eles)

        Q = {}
        for etype, sln in soln.items():
            Q[etype] = self._proc_Q(etype, mesh[etype], sln)

        self._flash_to_vtk(mesh, Q, time)

    def _proc_Q(self, etype, mesh, soln):
        gradu = self._pre_proc_fields_grad(etype, mesh, soln)
        gradu = gradu.swapaxes(0,1)

        ndim2, nupts, neles = gradu.shape
        # Gradient of velocity
        gradu = gradu.reshape(self.ndims, self.ndims, nupts, neles)

        # strain rate tensor, rotation tensor
        S = 0.5 * (gradu + gradu.swapaxes(0, 1)).reshape(-1, nupts, neles)
        R = 0.5 * (gradu - gradu.swapaxes(0, 1)).reshape(-1, nupts, neles)

        S = np.einsum('ijk, ijk -> jk', S, S)
        R = np.einsum('ijk, ijk -> jk', R, R)

        # Q = 1/2 ( ||R||^2 – ||S||^2 )
        Q = 0.5 * (R - S)

        return np.append(Q[:,None], soln[:,1][:,None], axis = 1)

    def _pre_proc_fields_grad(self, name, mesh, soln):
        # Reduce solution size since only velocity gradient is interested
        soln = soln[:,1:self.ndims+1]

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

        return gradsoln.swapaxes(0,1)

    #@memoize
    def _get_mesh_vtu_op(self, name, nspts, svpts):
        shape = self._get_shape(name, nspts, self.cfg)
        return shape.sbasis.nodal_basis_at(svpts).astype(self.dtype)

    #@memoize
    def _get_soln_vtu_op(self, name, nspts, svpts):
        shape = self._get_shape(name, nspts, self.cfg)
        return shape.ubasis.nodal_basis_at(svpts).astype(self.dtype)

    def _flash_to_vtk(self, mesh, Q, time):
        # High order output adopted from pyfr vtkwriter
        parts = defaultdict(list)
        for etype, ms in mesh.items():
            parts[f'{self.dir}/{time}.vtu'].append((str(0), f'{etype}'))

        write_s_to_fh = lambda s: fh.write(s.encode())

        self.vtkfile_version = '2.1'
        for pfn, misil in parts.items():
            with open(pfn, 'wb') as fh:
                write_s_to_fh('<?xml version="1.0" ?>\n<VTKFile '
                              'byte_order="LittleEndian" '
                              'type="UnstructuredGrid" '
                              f'version="{self.vtkfile_version}">\n'
                              '<UnstructuredGrid>\n')

                self._write_time_value(write_s_to_fh, time)

                 # Running byte-offset for appended data
                off = 0

                # Header
                for pn, k in misil:
                    off = self._write_serial_header(fh, Q[k], off, k)

                write_s_to_fh('</UnstructuredGrid>\n'
                              '<AppendedData encoding="raw">\n_')

                # Data
                for pn, k in misil:
                    self._write_data(fh, pn, mesh[k], Q[k], k)

                write_s_to_fh('\n</AppendedData>\n</VTKFile>')

    def _write_data(self, vtuf, pn, mk, sk, name):
        mesh = mk.astype(self.dtype)
        soln = sk.astype(self.dtype)

        # Dimensions
        nspts, neles = mesh.shape[:2]

        # Sub divison points inside of a standard element
        svpts = self._get_std_ele(name, nspts, self.order)
        nsvpts = len(svpts)

        #if name != 'pyr' and self.ho_output:
        if name != 'pyr':
            svpts = [svpts[i] for i in self._nodemaps[name, nsvpts]]

        # Generate the operator matrices
        mesh_vtu_op = self._get_mesh_vtu_op(name, nspts, svpts)
        soln_vtu_op = self._get_soln_vtu_op(name, nspts, svpts)

        # Calculate node locations of VTU elements
        vpts = mesh_vtu_op @ mesh.reshape(nspts, -1)
        vpts = vpts.reshape(nsvpts, -1, self.ndims)

        # Interpolate the solution to the vis points
        vsoln = soln_vtu_op @ soln.reshape(len(soln), -1)
        vsoln = vsoln.reshape(nsvpts, -1, neles).swapaxes(0, 1)

        # Write element node locations to file
        self._write_darray(vpts.swapaxes(0, 1), vtuf, self.dtype)

        # Perform the sub division for pyr etype
        if name != 'pyr':
            nodes = np.arange(nsvpts)
            subcellsoff = nsvpts
            types = self.vtk_types_ho[name]
        else:
            subdvcls = subclass_where(BaseShapeSubDiv, name=name)
            nodes = subdvcls.subnodes(self.order)
            subcellsoff = subdvcls.subcelloffs(self.order)
            types = subdvcls.subcelltypes(self.order)

        # Prepare VTU cell arrays
        vtu_con = np.tile(nodes, (neles, 1))
        vtu_con += (np.arange(neles)*nsvpts)[:, None]

        # Generate offset into the connectivity array
        vtu_off = np.tile(subcellsoff, (neles, 1))
        vtu_off += (np.arange(neles)*len(nodes))[:, None]

        # Tile VTU cell type numbers
        vtu_typ = np.tile(types, neles)

        # Write VTU node connectivity, connectivity offsets and cell types
        self._write_darray(vtu_con, vtuf, np.int32)
        self._write_darray(vtu_off, vtuf, np.int32)
        self._write_darray(vtu_typ, vtuf, np.uint8)

        # Process and write out the Q-criterion fields
        for arr in vsoln:
            self._write_darray(arr.T, vtuf, self.dtype)

    def _get_array_attrs(self, sk, etype):
        dtype = 'Float32' if self.dtype == np.float32 else 'Float64'
        dsize = np.dtype(self.dtype).itemsize

        vvars = [('Q-criterion', ['Q']),('Velocity-U', ['U'])] # only Q and U are output here

        names = ['', 'connectivity', 'offsets', 'types']
        types = [dtype, 'Int32', 'Int32', 'UInt8']
        comps = ['3', '', '', '']

        for fname, varnames in vvars:
            names.append(fname.title())
            types.append(dtype)
            comps.append(str(len(varnames)))


        # If a solution has been given the compute the sizes
        if etype == 'pyr':
            # If not pyr: npts = nnodes = pts_per_cell*nele
            # If pyr: npts = pts_per_cell*nele, nnodes = len(subdvcls.subnodes(self.etypes_div[etype]))*neles
            #npts, ncells, nnodes = self._get_npts_ncells_nnodes(sk)
            neles = sk.shape[1]
            shapecls = subclass_where(BaseShape, name=etype)
            subdvcls = subclass_where(BaseShapeSubDiv, name=etype)
            # Number of vis points
            npts = shapecls.nspts_from_order(self.order + 1)*neles
            ncells = len(subdvcls.subcells(self.order))*neles
            nnodes = len(subdvcls.subnodes(self.order))*neles

            nb = npts*dsize
        else:
            npts, ncells = sk.shape[:2]
            nnodes = npts * ncells
            nb = nnodes * dsize

        sizes = [3*nb, 4*nnodes, 4*ncells, ncells]
        sizes.extend(len(varnames)*nb for fname, varnames in vvars)

        return names, types, comps, sizes

    def _write_serial_header(self, vtuf, sk, off, etype):
        sk = sk.swapaxes(1,-1)
        names, types, comps, sizes = self._get_array_attrs(sk, etype)
        npts, neles = sk.shape[:2]
        if etype == 'pyr':
            shapecls = subclass_where(BaseShape, name=etype)
            subdvcls = subclass_where(BaseShapeSubDiv, name=etype)
            # Number of sub cells and nodes
            npts = shapecls.nspts_from_order(self.order + 1)
            ncells = len(subdvcls.subcells(self.order))*neles
        else:
            ncells = neles

        write_s = lambda s: vtuf.write(s.encode())

        write_s(f'<Piece NumberOfPoints="{npts*neles}" NumberOfCells="{ncells}">\n'
                '<Points>\n')

        # Write VTK DataArray headers
        for i, (n, t, c, s) in enumerate(zip(names, types, comps, sizes)):
            write_s(f'<DataArray Name="{self._process_name(n)}" type="{t}" '
                    f'NumberOfComponents="{c}" '
                    f'format="appended" offset="{off}"/>\n')

            off += 4 + s
            # Points => Cells => CellData => PointData transition
            if i == 0:
                write_s('</Points>\n<Cells>\n')
            # No need for partition info
            #elif i == 3:
            #    write_s('</Cells>\n<CellData>\n')
            #elif i == 4:
            #    write_s('</CellData>\n<PointData>\n')
            elif i == 3:
                write_s('</Cells>\n<PointData>\n')

        # Close
        write_s('</PointData>\n</Piece>\n')

        # Return the current offset
        return off

    def _write_darray(self, array, vtuf, dtype):
        array = array.astype(dtype)

        np.uint32(array.nbytes).tofile(vtuf)
        array.tofile(vtuf)

    def _process_name(self, name):
        return re.sub(r'\W+', '_', name)

    def _write_time_value(self, write_s, time):
        write_s('<FieldData>\n'
                '<DataArray Name="TimeValue" type="Float64" '
                'NumberOfComponents="1" NumberOfTuples="1" format="ascii">\n'
                f'{time}\n'
                '</DataArray>\n</FieldData>\n')
