#
# Copyright 2019-2020 Johannes Hoermann (U. Freiburg)
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
import matscipytest
import numpy as np
import os.path
import sys
import unittest

try:
    import dolfinx
except ImportError:
    print("dolfinx not found: skipping fenicsx-dependent tests")


class PoissonNernstPlanckSolverFEniCSxTest(matscipytest.MatSciPyTestCase):

    def setUp(self):
        """Provides 0.1 mM NaCl solution at 0.05 V across 100 nm open half space reference data from binary npz file"""
        self.test_path = os.path.dirname(os.path.abspath(__file__))
        self.ref_data = np.load(
            os.path.join(self.test_path, 'electrochemistry_data',
            'NaCl_c_0.1_mM_0.1_mM_z_+1_-1_L_1e-7_u_0.05_V_seg_200_interface.npz') )

        self.mesh_file_path = os.path.join(self.test_path,
                                           'electrochemistry_data', 'square.msh')


    @unittest.skipIf("dolfinx" not in sys.modules,
                     "dolfinx required")
    def test_poisson_nernst_planck_solver_fenicsx_std_interface_bc(self):
        """Tests PNP solver against simple interfacial BC"""
        from matscipy.electrochemistry.poisson_nernst_planck_solver_fenicsx \
            import PoissonNernstPlanckSystemFEniCSx as PoissonNernstPlanckSystem

        pnp = PoissonNernstPlanckSystem(
            c=[0.1,0.1], z=[1,-1], L=1e-7, delta_u=0.05,
            N=200, e=1e-12, maxit=20)
        pnp.use_standard_interface_bc()
        pnp.solve()

        # Reference data has been generated with controlled-volume solver and
        # is slightly off the FEM results, hence the generous tolerances below.
        self.assertArrayAlmostEqual(pnp.grid, self.ref_data['x'])
        self.assertArrayAlmostEqual(pnp.potential, self.ref_data['u'], 1e-6)
        self.assertArrayAlmostEqual(pnp.concentration, self.ref_data['c'], 1e-5)

    def test_poisson_nernst_planck_solver_fenicsx_2d(self):
        """Tests 2d PNP solver on simple square domain against 1d solution"""
        from matscipy.electrochemistry.poisson_nernst_planck_solver_fenicsx \
            import PoissonNernstPlanckSystemFEniCSx

        from matscipy.electrochemistry.poisson_nernst_planck_solver_2d_fenicsx \
            import PoissonNernstPlanckSystemFEniCSx2d

        # c = [0.1, 0.1] # mM
        c = [1.0, 1.0]  # mM
        z = [1, -1] # number charge
        delta_u = 0.05  # V
        N = 200 # number of cells for the 1d interval

        # solve the 1d problem
        pnp1d = PoissonNernstPlanckSystemFEniCSx(c=c, z=z, L=1e-7,
                                                 delta_u=delta_u, N=N)

        pnp1d.use_standard_interface_bc()
        pnp1d.solve()

        pnp2d = PoissonNernstPlanckSystemFEniCSx2d(c=c, z=z,
                                                   mesh_file=self.mesh_file_path,
                                                   scale_mesh=False)

        # the 2d solver has no convenience interface for boundary conditions
        # instead, we apply them manually

        # the square's upper domain boundary is tagged "2"
        pnp2d.apply_concentration_dirichlet_bc(0, pnp2d.c_scaled[0], 2)
        pnp2d.apply_concentration_dirichlet_bc(1, pnp2d.c_scaled[1], 2)

        delta_u_scaled = delta_u / pnp2d.u_unit

        # the square's lower domain boundary is tagged "4"
        pnp2d.apply_potential_dirichlet_bc(delta_u_scaled, 4)
        pnp2d.apply_potential_dirichlet_bc(0.0, 2)

        pnp2d.solve()

        # evaluate the 2d solution on a regular grid

        Nx = 20
        Ny = N+1

        lower_left = np.min(pnp2d.mesh.geometry.x, 0)
        upper_right = np.max(pnp2d.mesh.geometry.x, 0)

        x = np.linspace(lower_left[0], upper_right[0], Nx)
        y = np.linspace(lower_left[1], upper_right[1], Ny)

        X, Y = np.meshgrid(x, y)
        xy = np.vstack([X.flatten(), Y.flatten()]).T

        w = pnp2d.evaluate_dimensionless(xy)
        u_dimensionless = w[:, 0]
        c_dimensionless = w[:, 1:]

        U = u_dimensionless.reshape(Ny, Nx) * pnp2d.u_unit
        C = c_dimensionless.reshape(Ny, Nx, 2) * pnp2d.c_unit

        # all cross-sections along y-direction must be equivalent to 1d solution
        # the tolerances are very generous, but the 2d mesh is coarse.
        for ix in range(Nx):
            # grid goes from 0 to ~10.36, tolerance is 0.01
            self.assertArrayAlmostEqual(pnp1d.grid_dimensionless, Y[:, ix], 1e-2)
            # potential is between 0.0 and 0.05, tolerance in difference is 0.0002
            self.assertArrayAlmostEqual(pnp1d.potential, U[:, ix], 2e-4)
            # cationic species, positive charge, thinned out at electrode
            # between 0.14 mM and 1 mM, tolerance in difference ~ 0.002
            self.assertArrayAlmostEqual(pnp1d.concentration[0,:], C[:, ix, 0], 2e-3)
            # anionic species, negative charge, highly concentrated at electrode
            # between 1mM and ~7mM, tolerance in difference at 0.2
            self.assertArrayAlmostEqual(pnp1d.concentration[1,:], C[:, ix, 1], 0.2)


if __name__ == '__main__':
    unittest.main()
