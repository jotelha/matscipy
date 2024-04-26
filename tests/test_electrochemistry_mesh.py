#
# Copyright 2021 Lars Pastewka (U. Freiburg)
#           2019-2021 Johannes Hoermann (U. Freiburg)
#
# matscipy - Materials science with Python at the atomic-scale
# https://github.com/libAtoms/matscipy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import sys
import matscipytest
import numpy as np
import os, os.path
import shutil
import stat
import tempfile
import unittest



from mpi4py import MPI

try:
    import gmsh
except ImportError:
    print("gmsh not found: skipping gmsh-dependent tests")

try:
    import dolfinx.io
    from dolfinx.io import XDMFFile, gmshio
except ImportError:
    print("dolfinx.io not found: skipping dolfinx.io-dependent tests")



# from looseversion import LooseVersion

# 10.36 [l] is equivalent to 100 nm at Debye length 9.65 nm.
def gmsh_square(model: gmsh.model, name: str, d=10.36) -> gmsh.model:
    """Create a Gmsh model of a square.

    Args:
        model: Gmsh model to add the mesh to.
        name: Name (identifier) of the mesh to add.
        d: edged length

    Returns:
        Gmsh model with a square.

    """

    x0 = 0
    x1 = d

    model.add(name)
    model.setCurrent(name)

    p1 = gmsh.model.occ.addPoint(x1, x0, 0)
    p2 = gmsh.model.occ.addPoint(x1, x1, 0)
    l1 = gmsh.model.occ.addLine(p1, p2)

    p3 = gmsh.model.occ.addPoint(x0, x1, 0)
    l2 = gmsh.model.occ.addLine(p2, p3)

    p4 = gmsh.model.occ.addPoint(x0, x0, 0)
    l3 = gmsh.model.occ.addLine(p3, p4)
    l4 = gmsh.model.occ.addLine(p4, p1)

    lines = [l1, l2, l3, l4]

    loop = gmsh.model.occ.addCurveLoop(lines)
    surface = gmsh.model.occ.addPlaneSurface([loop])

    model.occ.synchronize()

    # Define physical groups (optional)
    gmsh.model.addPhysicalGroup(1, [l1], tag=1, name="Right boundary")
    gmsh.model.addPhysicalGroup(1, [l2], tag=2, name="Upper boundary")
    gmsh.model.addPhysicalGroup(1, [l3], tag=3, name="Left boundary")
    gmsh.model.addPhysicalGroup(1, [l4], tag=4, name="Lower boundary")
    gmsh.model.addPhysicalGroup(2, [surface], tag=5, name="Domain")

    model.mesh.generate(dim=2)

    return model

@unittest.skipIf("gmsh" not in sys.modules, "gmsh required")
@unittest.skipIf("dolfinx" not in sys.modules, "dolfinx required")
class ElectrochemistryFEMMeshTest(matscipytest.MatSciPyTestCase):
    """Tests mesh processing functionality."""

    def setUp(self):
        """Reads reference data files"""
        self.test_path = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.test_path, 'electrochemistry_data')

        gmsh.initialize()

        model = gmsh.model()
        model = gmsh_square(model, "square", d=10.36)

        self.msh_path = os.path.join(self.data_path, 'square.msh')
        if not os.path.isfile(self.msh_path):
            gmsh.write(self.msh_path)

        self.xdmf_path = os.path.join(self.data_path, 'square.xdmf')

        if not os.path.isfile(self.xdmf_path):
            msh, ct, ft = gmshio.model_to_mesh(model, MPI.COMM_SELF, rank=0)
            msh.name = "square"
            ct.name = f"{msh.name}_cells"
            ft.name = f"{msh.name}_facets"

            with XDMFFile(msh.comm, self.xdmf_path, "w") as file:
                msh.topology.create_connectivity(1, 2)
                file.write_mesh(msh)
                file.write_meshtags(ct, msh.geometry,
                                    geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry")
                file.write_meshtags(ft, msh.geometry,
                                    geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry")

    def tearDown(self):
        pass
        # self.bin_path.cleanup()

    def test_electrochemistry_fem_mesh_input(self):
        """Read msh from file."""

        mesh, cell_markers, facet_markers = gmshio.read_from_msh(self.msh_path,
                                                                 MPI.COMM_WORLD,
                                                                 gdim=2)

        np.testing.assert_array_equal(np.unique(facet_markers.values), [1,2,3,4])