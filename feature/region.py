from base import Base

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

After all of these, the total memory requirement shoudl be much much smaller
and could be able to process locally for plotting etc.
"""

class Region(Base):
    def __init__(self, argv, icfg, fname):
        super().__init__(argv)
        self.layers = icfg.getint(fname, 'layers', 0)
        #self.suffix = ['hex'] #icfg.get(fname, 'etype')
        self.suffix = icfg.get(fname, 'etype')
        self.boundary_name = icfg.get(fname, 'bname', None)

        #if self.boundary_name == None:
        #    raise RuntimeError('Region has to be attached to a boundary.')

        if self.suffix == None:
            self.suffix = ['hex'] #['quad','tri','hex','tet','pri','pyr']
        else:
            self.suffix = self.suffix[1:-1].split(", ")
        self.boundary_name = self.boundary_name[1:-1].split(", ")

        mesh_part = self.mesh.partition_info('spt')

        # Use strict element type for O grids
        #self.suffix_parts = np.where(np.array(mesh_part['hex']) > 0)[0]
        #self.suffix_parts = [f'p{i}' for i in self.suffix_parts]
        parts = []
        for etype in self.suffix:
            parts.append(np.where(np.array(mesh_part[etype]) > 0)[0])
        parts = set(list(np.concatenate(parts)))
        self.suffix_parts = [f'p{i}' for i in parts]


    def get_boundary(self):
        mesh_wall = defaultdict(list)
        mesh_wall_fid = defaultdict(list)
        mesh_wall_tag = list()
        # Get first level wall mesh
        for key in self.mesh:
            keyword = key.split('_')
            for bname in self.boundary_name:
                if bname in keyword:

                    part = keyword[-1]
                    for etype, eid, fid, pid in self.mesh[key][['f0','f1','f2','f3']].astype('U4,i4,i1,i2'):
                        if etype in self.suffix:
                            mesh_wall[f'spt_{etype}_{part}'].append(eid)
                            mesh_wall_fid[f'spt_{etype}_{part}'].append(fid)

                            # Tag all elements in the set as belonging to the first layer
                            if isinstance(mesh_wall_tag, list):
                                mesh_wall_tag = {part: {(etype, eid): 0}}
                            elif part in mesh_wall_tag:
                                mesh_wall_tag[part].update({(etype, eid): 0})
                            else:
                                mesh_wall_tag.update({part: {(etype, eid): 0}})

        return mesh_wall_tag, mesh_wall, mesh_wall_fid



    def get_wall_O_grid(self):
        # Get O-grid meshes
        mesh_wall_tag, mesh_wall, _ = self.get_boundary()

        # For single rank process, we'd better to pre load connectivities
        con = defaultdict(list)
        con_new = defaultdict(list)
        pcon = {}
        for part in self.suffix_parts:

            cont, pcont = self.load_connectivity(part)

            con[part] = cont
            pcon.update(pcont)


        # For wall normal quantities, load the connectivities
        #keys = list(mesh_wall.keys())
        for i in range(self.layers):

            #for key in keys:
            for part in self.suffix_parts:
                #_, etype, part = key.split('_')
                if part not in mesh_wall_tag:
                    mesh_wall_tag[part] = {}

                # Exchange information about recent updates to our set
                if len(pcon) > 0:
                    for p, (pc, pcr, sb) in pcon[part].items():
                        sb[:] = [mesh_wall_tag[part].get(c, -1) == i for c in pc]

                # Growing out by considering inertial partition
                for l, r in con[part]:
                    # Exclude elements which are not interested
                    if not all([r[0] in self.suffix]) or not all([l[0] in self.suffix]):
                        continue
                    if mesh_wall_tag[part].get(l, -1) == i and r not in mesh_wall_tag[part]:
                        mesh_wall_tag[part].update({r: i + 1})
                        mesh_wall[f'spt_{r[0]}_{part}'].append(r[1])

                    elif mesh_wall_tag[part].get(r, -1) == i and l not in mesh_wall_tag[part]:
                        mesh_wall_tag[part].update({l: i + 1})
                        mesh_wall[f'spt_{l[0]}_{part}'].append(l[1])

                # Grow our element set by considering adjacent partitions
                for p, (pc, pcr, sb) in pcon[part].items():
                    for l, r, b in zip(pc, pcr, sb):
                        if not all([r[0] in self.suffix]):
                            continue
                        try:
                            if b and r not in mesh_wall_tag[f'{p}']:
                                mesh_wall_tag[f'{p}'].update({r: i + 1})
                                mesh_wall[f'spt_{r[0]}_{p}'].append(r[1])

                        except  KeyError:
                                mesh_wall_tag.update({f'{p}': {r: i + 1}})
                                mesh_wall[f'spt_{r[0]}_{p}'].append(r[1])

        self.write_to_disk(mesh_wall)

        return mesh_wall_tag, mesh_wall

    def write_to_disk(self, mesh_wall):
        f = h5py.File(f'{self.dir}/region.m','w')
        for key in mesh_wall:
            f[key] = mesh_wall[key]
        f.close()

        """
        plt.figure()
        for key in mesh_wall:
            mesh = self.mesh[key][0,mesh_wall[key]]
            plt.plot(mesh[:,0],mesh[:,1],'.')
        plt.show()



        f = h5py.File(f'{self.solndir}{self.time[0]}.pyfrs','r')
        soln = defaultdict()
        for key in mesh_wall:
            _,etype,part = key.split('_')
            name = f'{self.dataprefix}_{etype}_{part}'
            soln[name] = np.array(f[name])[:,:,mesh_wall[key]]
        f.close()

        f = h5py.File(f'./preproc/out.s','w')
        for k in soln:
            f[k] = soln[k]
        f.close()
        """

    def load_connectivity(self, part):

        # Load our inner connectivity arrays
        con = self.mesh[f'con_{part}'].T
        con = con[['f0', 'f1']].astype('U4,i4').tolist()

        pcon = {}
        # Load our partition boundary connectivity arrays
        for p in self.suffix_parts:
            try:
                pc = self.mesh[f'con_{part}{p}']
                pc = pc[['f0', 'f1']].astype('U4,i4').tolist()
                pcr = self.mesh[f'con_{p}{part}']
                pcr = pcr[['f0', 'f1']].astype('U4,i4').tolist()

            except KeyError:
                continue
            try:
                pcon[part].update({p: (pc, pcr, *np.empty((1, len(pc)), dtype=bool))})
            except KeyError:
                pcon.update({part: {p: (pc, pcr, *np.empty((1, len(pc)), dtype=bool))}})

        return con, pcon
