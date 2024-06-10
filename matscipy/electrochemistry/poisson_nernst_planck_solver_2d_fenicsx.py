#
# Copyright 2020 Johannes Hoermann (U. Freiburg)
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
""" Compute ion concentrations consistent with general
Poisson-Nernst-Planck (PNP) equations via FEniCS.

Copyright 2019 IMTEK Simulation
University of Freiburg

Authors:
  Johannes Hoermann <johannes.hoermann@imtek-uni-freiburg.de>
"""
import logging
import numpy as np

from mpi4py import MPI

import ufl
import basix
import dolfinx.log
import dolfinx.mesh
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import gmshio # XDMFFile

from matscipy.electrochemistry.poisson_nernst_planck_solver_base import PoissonNernstPlanckSystemABC


class PoissonNernstPlanckSystemFEniCSx2d(PoissonNernstPlanckSystemABC):
    """Describes and solves a 2D Poisson-Nernst-Planck system with FEniCSx FEM."""

    @property
    def grid(self):
        return self.grid_dimensionless * self.l_unit

    @property
    def potential(self):
        return self.dimensionless_potential_on_mesh_nodes * self.u_unit

    @property
    def concentration(self):
        return np.where(self.dimensionless_concentrations_on_mesh_nodes > np.finfo('float64').resolution,
                        self.dimensionless_concentrations_on_mesh_nodes * self.c_unit, 0.0)

    @property
    def charge_density(self):
        return np.sum(self.F * self.concentration.T * self.z, axis=1)

    @property
    def x1_scaled(self):
        return self.x0_scaled + self.L_scaled

    @property
    def grid_dimensionless(self):
        return self.mesh.geometry.x[:,:2]

    def evaluate_potential(self, x):
        """Evaluate SI unit potential on SI unit points x.
        
        Parameters
        ----------
        x: (number of points, 2) array, SI units (m)

        Returns
        -------
        u: (number of points) array, SI units (V)
            electrostatic potential
        """
        w = self.evaluate_dimensionless(x / self.l_unit)
        return w[:, 0]*self.u_unit

    def evaluate_concentration(self, x):
        """Evaluate SI unit concentrations on SI unit points x..

        Parameters
        ----------
        x: (number of points, 2) array, SI units (m)

        Returns
        -------
        c: (number of points, M) array, SI units (mol/m^3)
            concentrations of M species
        """
        w = self.evaluate_dimensionless(x / self.l_unit)
        return w[:, 1:] * self.c_unit

    def evaluate_dimensionless(self, x):
        """Evaluate solution on points x.

        Parameters
        ----------
        x: (number of points, 2) array

        Returns
        -------
        w: (number of points, M+1) array
            potential in first column, then concentrations of M species
        """
        # make single point a 1-line 2d array
        if len(x.shape) == 1:
            x = np.atleast_2d(x)

        # make 2d coordinates 3d with zero third component
        if x.shape[1] == 2:
            x = np.hstack([x, np.zeros((x.shape[0],1))])

        # compute bounding boxes of cells
        bb_tree = dolfinx.geometry.bb_tree(self.mesh, 2)

        # find cells which may encompass coordinate points of interest
        cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, x)
        cell_list = dolfinx.geometry.compute_colliding_cells(self.mesh, cell_candidates, x)

        # compute solution at all points
        w = np.array([self.w.eval(p, cell_list.links(i)[0]) for i, p in enumerate(x)]).T

        return w.T

    def solve(self):
        """Evoke FEniCS FEM solver.

        Returns
        -------
        dimensionless_potential_on_mesh_nodes : (Ni,) ndarray
            potential at Ni grid points
        dimensionless_concentrations_on_mesh_nodes : (M,Nij) ndarray
            concentrations of M species at Ni grid points
        dimensionless_lagrange_multipliers_on_mesh_nodes: (L,) ndarray
            value of L Lagrange multipliers
        """

        # weak form and FEM scheme:

        # in the weak form, u and v are the trial and test functions associated
        # with the Poisson part, p and q the trial and test functions associated
        # with the Nernst-Planck part. lam and mu are trial and test functions
        # associated to constraints introduced via Lagrange multipliers.
        # w is the whole set of trial functions [u,p,lam]
        # W is the space all w live in.
        rho = 0
        for i in range(self.M):
            rho += self.z[i] * self.p[i]

        source = - 0.5 * rho * self.v * ufl.dx

        laplace = ufl.dot(ufl.grad(self.u), ufl.grad(self.v)) * ufl.dx

        poisson = laplace + source

        nernst_planck = 0
        for i in range(self.M):
            nernst_planck += ufl.dot(
                - ufl.grad(self.p[i]) - self.z[i] * self.p[i] * ufl.grad(self.u),
                ufl.grad(self.q[i])
            ) * ufl.dx

        # constraints set up elsewhere
        F = poisson + nernst_planck + self.constraints

        problem = NonlinearProblem(F, self.w, bcs=self.boundary_conditions)

        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        solver.convergence_criterion = "incremental"
        solver.rtol = 1e-9
        solver.report = True

        # suggestions from fenicsx tutorial
        # ksp = solver.krylov_solver
        # opts = PETSc.Options()
        # option_prefix = ksp.getOptionsPrefix()
        # opts[f"{option_prefix}ksp_type"] = "cg"
        # opts[f"{option_prefix}pc_type"] = "gamg"
        # opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
        # ksp.setFromOptions()

        n, converged = solver.solve(self.w)

        self.logger.info("Converged: %s, %d iterations", converged, n)

        # compute solution on mesh nodes
        # bb_tree = dolfinx.geometry.bb_tree(self.mesh, 2)
        # cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, self.mesh.geometry.x)
        # cell_list = dolfinx.geometry.compute_colliding_cells(self.mesh, cell_candidates, self.mesh.geometry.x)
        # wij = np.array([self.w.eval(p, cell_list.links(i)[0]) for i, p in enumerate(self.mesh.geometry.x)]).T
        #
        # self.dimensionless_potential_on_mesh_nodes = wij[0, :]  # potential
        # self.dimensionless_concentrations_on_mesh_nodes = wij[1:(self.M + 1), :]  # concentrations
        # self.dimensionless_lagrange_multipliers_on_mesh_nodes = wij[(self.M + 1):, :]  # Lagrange multipliers
        #
        # return (self.dimensionless_potential_on_mesh_nodes,
        #         self.dimensionless_concentrations_on_mesh_nodes,
        #         self.dimensionless_lagrange_multipliers_on_mesh_nodes)

    def apply_potential_dirichlet_bc(self, u0, group):
        """FEniCS Dirichlet BC u0 for potential at boundary group.

        Parameters
        ----------
        u0: float
            dimensionless potential
        group: int
            boundary marker (tag)
        """
        self.boundary_conditions.extend([
            dolfinx.fem.dirichletbc(
                dolfinx.default_scalar_type(u0),
                self.boundary_dofs[group][0], self.W.sub(0))])

    def apply_concentration_dirichlet_bc(self, k, c0, group):
        """FEniCS Dirichlet BC c0 for k'th ion species at boundary group."""
        self.boundary_conditions.extend([
            dolfinx.fem.dirichletbc(
                dolfinx.default_scalar_type(c0),
                self.boundary_dofs[group][k+1], self.W.sub(k+1))])

    # LAGRANGE MULTIPLIERS NOT YET POSSIBLE WITH FENICSX
    # def apply_number_conservation_constraint(self, k, c0):
    #     """
    #     Enforce number conservation constraint via Lagrange multiplier.
    #     See https://fenicsproject.org/docs/dolfin/1.6.0/python/demo/documented/neumann-poisson/python/documentation.html
    #     """
    #     self.constraints += self.lam[k] * self.q[k] * ufl.dx \
    #                         + (self.p[k] - c0) * self.mu[k] * ufl.dx

    def read_mesh_from_file(self, filename, scale=True):
        """Read gmsh msh format mesh from file and scale by Debye length if desired"""
        self.mesh, self.cell_markers, self.facet_markers = gmshio.read_from_msh(
            filename, MPI.COMM_WORLD, gdim=2)

        min_x = np.min(self.mesh.geometry.x, axis=0)
        max_x = np.max(self.mesh.geometry.x, axis=0)
        mesh_dim = max_x - min_x

        if scale:
            self.logger.info("Read mesh of size %g x %g m (SI units)",
                             mesh_dim[0], mesh_dim[1])

            # self.mesh.scale(1./self.l_unit)
            self.mesh.geometry.x[:,:] /= self.l_unit

            min_x = np.min(self.mesh.geometry.x, axis=0)
            max_x = np.max(self.mesh.geometry.x, axis=0)
            mesh_dim = max_x - min_x
            self.logger.info("Scaled mesh by %g to dimensionless measures %g x %g",
                             self.l_unit, mesh_dim[0], mesh_dim[1])
        else:
            self.logger.info("Read mesh of size %g x %g (unitless), not scaled",
                             mesh_dim[0], mesh_dim[1])

    def discretize(self):
        """Builds function space, call again after introducing constraints"""
        # construct test and trial function space from elements
        # spanned by Lagrange polynomials for the physical variables of
        # potential and concentration and global elements with a single degree
        # of freedom ('Real') for constraints.

        P = basix.ufl.element('Lagrange', self.mesh.basix_cell(), 3)

        # No Real elements in fenicsx yet, see
        # https://fenicsproject.discourse.group/t/integral-constrains-in-fenicsx/11429
        # suggested the use of https://github.com/jorgensd/dolfinx_mpc
        # R = basix.ufl.element('Real', self.mesh.basix_cell(), 0)

        elements = [P] * (1 + self.M) # + [R] * self.K

        H = basix.ufl.mixed_element(elements)

        # mixed elements introduce artifacts in the solution, vector elements apparently work fine
        # ignore constraints for now
        # H = ufl.VectorElement("Lagrange", self.mesh.ufl_cell(), 3, dim=self.M+1)

        self.W = dolfinx.fem.FunctionSpace(self.mesh, H)

        # boundary degrees of freedom
        self.unique_facet_markers = np.unique(self.facet_markers.values)

        self.logger.info("Identified %d distinc boundaries.", len(self.unique_facet_markers))
        self.boundary_dofs = {}
        for facet_marker in self.unique_facet_markers:
            self.boundary_dofs[facet_marker] = []
            for k in range(0, self.M + 1):
                self.boundary_dofs[facet_marker].append(
                    dolfinx.fem.locate_dofs_topological(
                        V=self.W.sub(k),
                        entity_dim=self.mesh.topology.dim - 1,
                        entities=self.facet_markers.find(facet_marker)))

        # solution functions
        self.w = dolfinx.fem.Function(self.W)

        # set uniform initial values
        # if self.ui0 is not None:
        #     self.w.sub(0).interpolate(lambda x: self.ui0 * np.ones(x[0].shape))
        #
        # if self.ni0 is not None:
        #     for k in range(self.ni0.shape[0]):
        #         self.w.sub(k+1).interpolate(lambda x: self.ni0[k] * np.ones(x[0].shape))

        # self.w.x.scatter_forward()

        # u represents voltage , p concentrations
        # uplam = self.w.split() # # apparently causes issues within PETSc,
        # see https://fenicsproject.discourse.group/t/how-to-debug-the-petsc-error/13406
        uplam = ufl.split(self.w)
        self.u, self.p, self.lam = (
            uplam[0], [*uplam[1:(self.M + 1)]], [*uplam[(self.M + 1):]])

        # v, q and mu represent respective test functions
        vqmu = ufl.TestFunctions(self.W)
        self.v, self.q, self.mu = (
            vqmu[0], [*vqmu[1:(self.M + 1)]], [*vqmu[(self.M + 1):]])

    def init(self, **kwargs):
        """Initializes a 2D Poisson-Nernst-Planck system description."""
        self.logger = logging.getLogger(__name__)
        super().init(**kwargs)

        # default solver settings
        self.converged = False  # solver's convergence flag

    def __init__(self, *args, mesh_file=None, scale_mesh=True, **kwargs):
        """Same parameters as PoissonNernstPlanckSystem.init"""
        self.init(*args, **kwargs)

        self.K = 0  # number of Lagrange multipliers (constraints)
        self.constraints = 0  # holds constraint kernels
        self.read_mesh_from_file(mesh_file, scale=scale_mesh)
        self.discretize()
