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
# from petsc4py import PETSc

import ufl
import dolfinx.log
import dolfinx.mesh
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

from matscipy.electrochemistry.poisson_nernst_planck_solver_base import PoissonNernstPlanckSystemABC


class PoissonNernstPlanckSystemFEniCSx(PoissonNernstPlanckSystemABC):
    """Describes and solves a 1D Poisson-Nernst-Planck system with FEniCSx FEM."""

    @property
    def grid(self):
        return self.X * self.l_unit

    @property
    def potential(self):
        return self.uij * self.u_unit

    @property
    def concentration(self):
        return np.where(self.nij > np.finfo('float64').resolution,
                        self.nij * self.c_unit, 0.0)

    @property
    def charge_density(self):
        return np.sum(self.F * self.concentration.T * self.z, axis=1)

    @property
    def x1_scaled(self):
        return self.x0_scaled + self.L_scaled

    @property
    def X(self):
        local_cells = np.arange(
            self.mesh.topology.index_map(self.mesh.topology.dim).size_local, dtype=np.int32)
        midpoints = dolfinx.mesh.compute_midpoints(self.mesh, self.mesh.topology.dim, local_cells)
        return midpoints[:,0]

    def solve(self):
        """Evoke FEniCS FEM solver.

        Returns
        -------
        uij : (Ni,) ndarray
            potential at Ni grid points
        nij : (M,Nij) ndarray
            concentrations of M species at Ni grid points
        lamj: (L,) ndarray
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

        # store results:
        # see https://fenicsproject.discourse.group/t/evaluate-function-on-whole-mesh/12420/3
        # Interpolate solution DOFs to discontinuous lagrange of zero'th order
        dg0 = ufl.VectorElement("DG", self.mesh.ufl_cell(), 0, dim=self.M + self.K + 1)
        W0 = dolfinx.fem.FunctionSpace(self.mesh, dg0)

        w0 = dolfinx.fem.Function(W0, dtype=dolfinx.default_scalar_type)
        w0.interpolate(self.w)

        wij = w0.x.array.reshape(self.mesh.geometry.x.shape[0]-1, self.M + self.K + 1).T
        self.uij = wij[0, :]  # potential
        self.nij = wij[1:(self.M + 1), :]  # concentrations
        self.lamj = wij[(self.M + 1):, :]  # Lagrange multipliers

        return self.uij, self.nij, self.lamj

    def apply_left_potential_dirichlet_bc(self, u0):
        """FEniCS Dirichlet BC u0 for potential at left boundary."""
        self.boundary_conditions.extend([
            dolfinx.fem.dirichletbc(
                dolfinx.default_scalar_type(u0),
                self.left_boundary_dofs[0], self.W.sub(0))])

    def apply_right_potential_dirichlet_bc(self, u0):
        """FEniCS Dirichlet BC u0 for potential at right boundary."""
        self.boundary_conditions.extend([
            dolfinx.fem.dirichletbc(
                dolfinx.default_scalar_type(u0),
                self.right_boundary_dofs[0], self.W.sub(0))])

    def apply_left_concentration_dirichlet_bc(self, k, c0):
        """FEniCS Dirichlet BC c0 for k'th ion species at left boundary."""
        self.boundary_conditions.extend([
            dolfinx.fem.dirichletbc(
                self.mesh, dolfinx.default_scalar_type(c0),
                self.left_boundary_dofs[k+1], self.W.sub(k+1))])

    def apply_right_concentration_dirichlet_bc(self, k, c0):
        """FEniCS Dirichlet BC c0 for k'th ion species at right boundary."""
        self.boundary_conditions.extend([
            dolfinx.fem.dirichletbc(
                dolfinx.default_scalar_type(c0),
                self.right_boundary_dofs[k+1], self.W.sub(k+1))])

    # TODO: Robin BC!
    def apply_left_potential_robin_bc(self, u0, lam0):
        self.logger.warning("Not implemented!")

    def apply_right_potential_robin_bc(self, u0, lam0):
        self.logger.warning("Not implemented!")

    def apply_number_conservation_constraint(self, k, c0):
        """
        Enforce number conservation constraint via Lagrange multiplier.
        See https://fenicsproject.org/docs/dolfin/1.6.0/python/demo/documented/neumann-poisson/python/documentation.html
        """
        self.constraints += self.lam[k] * self.q[k] * ufl.dx \
                            + (self.p[k] - c0) * self.mu[k] * ufl.dx

    def apply_potential_dirichlet_bc(self, u0, u1):
        """Potential Dirichlet BC u0 and u1 on left and right boundary."""
        self.apply_left_potential_dirichlet_bc(u0)
        self.apply_right_potential_dirichlet_bc(u1)

    def apply_potential_robin_bc(self, u0, u1, lam0, lam1):
        """Potential Robin BC on left and right boundary."""
        self.apply_left_potential_robin_bc(u0, lam0)
        self.apply_right_potential_robin_bc(u1, lam1)

    def use_standard_interface_bc(self):
        """Interface at left hand side and open bulk at right hand side"""
        self.boundary_conditions = []

        # Potential Dirichlet BC
        self.u0 = self.delta_u_scaled
        self.u1 = 0

        self.logger.info('Left hand side Dirichlet boundary condition:  u0 = {:> 8.4g}'.format(self.u0))
        self.logger.info('Right hand side Dirichlet boundary condition: u1 = {:> 8.4g}'.format(self.u1))

        self.apply_potential_dirichlet_bc(self.u0, self.u1)

        for k in range(self.M):
            self.logger.info(('Ion species {:02d} right hand side concentration '
                              'Dirichlet boundary condition: c1 = {:> 8.4g}').format(k, self.c_scaled[k]))
            self.apply_right_concentration_dirichlet_bc(k, self.c_scaled[k])

    def use_standard_cell_bc(self):
        """
        Interfaces at left hand side and right hand side, species-wise
        number conservation within interval."""
        self.boundary_conditions = []

        # Introduce a Lagrange multiplier per species
        # rebuild discretization scheme (function spaces)
        self.K = self.M
        self.discretize()

        # Potential Dirichlet BC
        self.u0 = self.delta_u_scaled / 2.
        self.u1 = - self.delta_u_scaled / 2.
        self.logger.info('{:>{lwidth}s} u0 = {:< 8.4g}'.format(
            'Left hand side Dirichlet boundary condition', self.u0, lwidth=self.label_width))
        self.logger.info('{:>{lwidth}s} u1 = {:< 8.4g}'.format(
            'Right hand side Dirichlet boundary condition', self.u1, lwidth=self.label_width))

        self.apply_potential_dirichlet_bc(self.u0, self.u1)

        # Number conservation constraints
        self.constraints = 0
        N0 = self.L_scaled * self.c_scaled  # total amount of species in cell
        for k in range(self.M):
            self.logger.info('{:>{lwidth}s} N0 = {:<8.4g}'.format(
                'Ion species {:02d} number conservation constraint'.format(k),
                N0[k], lwidth=self.label_width))
            self.apply_number_conservation_constraint(k, self.c_scaled[k])

    def use_central_reference_concentration_based_cell_bc(self):
        """
        Interfaces at left hand side and right hand side, species-wise
        concentration fixed at cell center."""
        self.boundary_conditions = []

        # Introduce a Lagrange multiplier per species anderson
        # rebuild discretization scheme (function spaces)
        # self.K = self.M
        # self.discretize()

        # Potential Dirichlet BC
        self.u0 = self.delta_u_scaled / 2.
        self.u1 = - self.delta_u_scaled / 2.
        self.logger.info('{:>{lwidth}s} u0 = {:< 8.4g}'.format(
            'Left hand side Dirichlet boundary condition', self.u0, lwidth=self.label_width))
        self.logger.info('{:>{lwidth}s} u1 = {:< 8.4g}'.format(
            'Right hand side Dirichlet boundary condition', self.u1, lwidth=self.label_width))

        self.apply_potential_dirichlet_bc(self.u0, self.u1)

        for k in range(self.M):
            self.logger.info(
                'Ion species {:02d} reference concentration condition: c1 = {:> 8.4g} at cell center'.format(
                    k, self.c_scaled[k]))
            self.apply_central_reference_concentration_constraint(k, self.c_scaled[k])

    def use_stern_layer_cell_bc(self):
        """
        Interfaces at left hand side and right hand side, species-wise
        number conservation within interval."""
        self.boundary_conditions = []

        # Introduce a Lagrange multiplier per species anderson
        # rebuild discretization scheme (function spaces)
        self.constraints = 0
        self.K = self.M
        self.discretize()

        # Potential Dirichlet BC
        self.u0 = self.delta_u_scaled / 2.
        self.u1 = - self.delta_u_scaled / 2.

        boundary_markers = fn.MeshFunction(
            'size_t', self.mesh, self.mesh.topology().dim() - 1)

        bx = [
            Boundary(x0=self.x0_scaled, tol=self.bctol),
            Boundary(x0=self.x1_scaled, tol=self.bctol)]

        # Boundary.mark crashes the kernel if Boundary is internal class
        for i, b in enumerate(bx):
            b.mark(boundary_markers, i)

        boundary_conditions = {
            0: {'Robin': (1. / self.lambda_S_scaled, self.u0)},
            1: {'Robin': (1. / self.lambda_S_scaled, self.u1)},
        }

        ds = fn.Measure('ds', domain=self.mesh, subdomain_data=boundary_markers)

        integrals_R = []
        for i in boundary_conditions:
            if 'Robin' in boundary_conditions[i]:
                r, s = boundary_conditions[i]['Robin']
                integrals_R.append(r * (self.u - s) * self.v * ds(i))

        self.constraints += sum(integrals_R)

        # Number conservation constraints

        N0 = self.L_scaled * self.c_scaled  # total amount of species in cell
        for k in range(self.M):
            self.logger.info('{:>{lwidth}s} N0 = {:<8.4g}'.format(
                'Ion species {:02d} number conservation constraint'.format(k),
                N0[k], lwidth=self.label_width))
            self.apply_number_conservation_constraint(k, self.c_scaled[k])

    def discretize(self):
        """Builds function space, call again after introducing constraints"""
        # FEniCSx interface
        self.mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, self.N,
                                                 points=(self.x0_scaled, self.x1_scaled))

        self.left_boundary = dolfinx.mesh.locate_entities_boundary(
            self.mesh, dim=(self.mesh.topology.dim - 1),
            marker=lambda x: np.isclose(x[0], self.x0_scaled))
        self.right_boundary = dolfinx.mesh.locate_entities_boundary(
            self.mesh, dim=(self.mesh.topology.dim - 1),
            marker=lambda x: np.isclose(x[0], self.x1_scaled))

        # construct test and trial function space from elements
        # spanned by Lagrange polynomials for the physical variables of
        # potential and concentration and global elements with a single degree
        # of freedom ('Real') for constraints.

        # P = basix.ufl.element('Lagrange', self.mesh.basix_cell(), 3)

        # No Real elements in fenicsx yet, see
        # https://fenicsproject.discourse.group/t/integral-constrains-in-fenicsx/11429
        # suggested the use of https://github.com/jorgensd/dolfinx_mpc
        # R = basix.ufl.element('Real', self.mesh.basix_cell(), 0)

        # elements = [P] * (1 + self.M) + [R] * self.K

        # H = basix.ufl.mixed_element(elements)

        # mixed elements introduce artifacts in the solution, vector elements apparently work fine
        # ignore constraints for now
        H = ufl.VectorElement("Lagrange", self.mesh.ufl_cell(), 3, dim=self.M+1)

        self.W = dolfinx.fem.FunctionSpace(self.mesh, H)

        # boundary degrees of freedom
        self.left_boundary_dofs = []
        for k in range(0, self.M + 1):
            # Q, _ = self.W.sub(k).collapse()
            self.left_boundary_dofs.append(
                dolfinx.fem.locate_dofs_topological(
                    V=self.W.sub(k),
                    entity_dim=self.mesh.topology.dim - 1,
                    entities=self.left_boundary))

        self.right_boundary_dofs = []
        for k in range(0, self.M + 1):
            # Q, _ = self.W.sub(k).collapse()
            self.right_boundary_dofs.append(
                dolfinx.fem.locate_dofs_topological(
                    V=self.W.sub(k),
                    entity_dim=self.mesh.topology.dim - 1,
                    entities=self.right_boundary))

        # solution functions
        self.w = dolfinx.fem.Function(self.W)

        # set uniform initial values
        if self.ui0 is not None:
            self.w.sub(0).interpolate(lambda x: self.ui0 * np.ones(x[0].shape))

        if self.ni0 is not None:
            for k in range(self.ni0.shape[0]):
                self.w.sub(k+1).interpolate(lambda x: self.ni0[k] * np.ones(x[0].shape))

        self.w.x.scatter_forward()

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

    def init(self,
             L=100e-9,  # 100 nm
             lambda_S=0,  # Stern layer (compact layer) thickness
             x0=0,  # zero position
             delta_u=0.05,  # potential difference [V]
             relative_permittivity=79,
             N=200,  # number of grid segments, number of grid points Ni = N + 1
             e=1e-10,  # absolute tolerance, TODO: switch to standardized measure
             maxit=20,  # maximum number of Newton iterations
             solver=None,
             options=None,
             **kwargs):
        """Initializes a 1D Poisson-Nernst-Planck system description.

        Expects quantities in SI units per default.

        Parameters
        ----------
        c : (M,) ndarray, optional
            bulk concentrations of each ionic species [mol/m^3]
            (default: [ 0.1, 0.1 ])
        z : (M,) ndarray, optional
            charge of each ionic species [1] (default: [ +1, -1 ])
        x0 : float, optional
            left hand side reference position (default: 0)
        L : float, optional
            1D domain size [m] (default: 100e-9)
        lambda_S: float, optional
            Stern layer thickness in case of Robin BC [m] (default: 0)
        T : float, optional
            temperature of the solution [K] (default: 298.15)
        delta_u : float, optional
            potential drop across 1D cell [V] (default: 0.05)
        relative_permittivity: float, optional
            relative permittivity of the ionic solution [1] (default: 79)
        vacuum_permittivity: float, optional
            vacuum permittivity [F m^-1] (default: 8.854187817620389e-12 )
        R : float, optional
            molar gas constant [J mol^-1 K^-1] (default: 8.3144598)
        F : float, optional
            Faraday constant [C mol^-1] (default: 96485.33289)
        N : int, optional
            number of discretization grid segments (default: 200)
        e : float, optional
            absolute tolerance for Newton solver convergence (default: 1e-10)
        maxit : int, optional
            maximum number of Newton iterations (default: 20)
        solver: func( func(x), x0), optional
            solver to use (default: None, will use own simple Newton solver)
        potential0: (N+1,) ndarray, optional (default: None)
            potential initial values
        concentration0: (M,N+1) ndarray, optional (default: None)
            concentration initial values
        """
        self.logger = logging.getLogger(__name__)
        super().init(**kwargs)

        # default solver settings
        self.converged = False  # solver's convergence flag
        self.N = N  # discretization segments
        self.e = e  # Newton solver default tolerance
        self.maxit = maxit  # Newton solver maximum iterations

        self.L = L  # 1d domain size
        self.lambda_S = lambda_S  # Stern layer thickness
        self.x0 = x0  # reference position
        self.delta_u = delta_u  # potential difference

        # domain
        self.L_scaled = self.L / self.l_unit

        # compact layer
        self.lambda_S_scaled = self.lambda_S / self.l_unit

        # reference position
        self.x0_scaled = self.x0 / self.l_unit

        # potential difference
        self.delta_u_scaled = self.delta_u / self.u_unit

        # print scaled quantities to log
        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'reduced domain size L*', self.L_scaled, lwidth=self.label_width))
        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'reduced compact layer thickness lambda_S*', self.lambda_S_scaled, lwidth=self.label_width))
        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'reduced reference position x0*', self.x0_scaled, lwidth=self.label_width))
        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'reduced potential delta_u*', self.delta_u_scaled, lwidth=self.label_width))

    def __init__(self, *args, **kwargs):
        """Same parameters as PoissonNernstPlanckSystem.init

        Additional parameters:
        ----------------------
        bctol : float, optional
            tolerance for identifying domain boundaries
            (default: 1e-14)
        solver_parameters : dict, optional
            Additional solver parameters passed through to fenics solver
            (default: {})
        """
        self.init(*args, **kwargs)

        self.solver_parameters = {}
        if "solver_parameters" in kwargs:
            self.solver_parameters = kwargs["solver_parameters"]

        self.bctol = 1e-14  # tolerance for identifying domain boundaries
        if "bctol" in kwargs:
            self.bctol = kwargs["bctol"]

        self.K = 0  # number of Lagrange multipliers (constraints)
        self.constraints = 0  # holds constraint kernels
        self.discretize()
