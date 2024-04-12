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
"""
Compute ion concentrations with general Poisson-Nernst-Planck (PNP) equations.

Copyright 2019 IMTEK Simulation
University of Freiburg

Authors:
  Johannes Hoermann <johannes.hoermann@imtek-uni-freiburg.de>
"""
import logging
import time
import numpy as np
import scipy.constants as sc
import scipy.optimize

from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PoissonNernstPlanckSystemABC(ABC):
    """Describes and solves a 1D Poisson-Nernst-Planck system"""

    # properties "offer" the solution in physical units:
    @property
    @abstractmethod
    def grid(self):
        ...

    @property
    @abstractmethod
    def potential(self):
        ...

    @property
    @abstractmethod
    def concentration(self):
        ...

    @property
    @abstractmethod
    def charge_density(self):
        ...


    @abstractmethod
    def potential_dirichlet_bc(self, x, u0=0):
        ...

    @abstractmethod
    def left_dirichlet_bc(self, x, k, x0=0):
        """Construct Dirichlet BC at boundary"""
        ...

    @abstractmethod
    def potential_robin_bc(self, x, lam, u0=0):
        ...

    @abstractmethod
    def robin_bc(self, x, k, lam, x0=0):
        """
        Compute left hand side Robin (u + lam*dudx = u0 ) BC at in accord with
        2nd order finite difference scheme.

        Parameters
        ----------
        x : (Ni,) ndarray
            N-valued variable vector
        k : int
            ion species (-1 for potential)
        lam: float
            BC coefficient, corresponds to Stern layer thickness
            if applied to potential variable in PNP problem. Here, this steric
            layer is assumed to constitute a region of uniform charge density
            and thus linear potential drop across the interface.
        x0 : float
            right hand side value of BC, corresponds to potential beyond Stern
            layer if applied to potential variable in PNP system.

        Returns
        -------
        float: boundary condition residual
        """
        ...

    @abstractmethod
    def number_conservation_constraint(self, x, k, N0):
        """N0: total amount of species, k: ion species"""
        ...

    @property
    def ionic_strength(self):  # ionic strength
        """Compute the system's ionic strength from charges and concentrations.

        Returns
        -------
        ionic_strength : float
            ionic strength ( 1/2 * sum(z_i^2*c_i) )
            [concentration unit, i.e. mol m^-3]
        """
        return 0.5*np.sum(np.square(self.z) * self.c)

    @property
    def lambda_D(self):
        """Compute the system's Debye length.

        Returns
        -------
        lambda_D : float
            Debye length, sqrt( epsR*eps*R*T/(2*F^2*I) ) [length unit, i.e. m]
        """
        return np.sqrt(
            self.relative_permittivity*self.vacuum_permittivity*self.R*self.T/(
                2.0*self.F**2*self.ionic_strength))

    # default 0.1 mM (i.e. mol/m^3) NaCl aqueous solution
    def init(self,
             c=np.array([0.1, 0.1]),
             z=np.array([1, -1]),
             T=298.15,
             relative_permittivity=79,
             vacuum_permittivity=sc.epsilon_0,
             R=sc.value('molar gas constant'),
             F=sc.value('Faraday constant'),
             potential0=None,
             concentration0=None,
             **kwargs):
        """Initializes a Poisson-Nernst-Planck system description.

        Expects quantities in SI units per default.

        Parameters
        ----------
        c : (M,) ndarray, optional
            bulk concentrations of each ionic species [mol/m^3]
            (default: [ 0.1, 0.1 ])
        z : (M,) ndarray, optional
            charge of each ionic species [1] (default: [ +1, -1 ])
        T : float, optional
            temperature of the solution [K] (default: 298.15)
        relative_permittivity: float, optional
            relative permittivity of the ionic solution [1] (default: 79)
        vacuum_permittivity: float, optional
            vacuum permittivity [F m^-1] (default: 8.854187817620389e-12 )
        R : float, optional
            molar gas constant [J mol^-1 K^-1] (default: 8.3144598)
        F : float, optional
            Faraday constant [C mol^-1] (default: 96485.33289)
        potential0: (N+1,) ndarray, optional (default: None)
            potential initial values
        concentration0: (M,N+1) ndarray, optional (default: None)
            concentration initial values
        """
        assert len(c) == len(z), "Provide concentration AND charge for ALL ion species!"

        # default output settings
        self.label_width = 40  # character width of quantity labels in log

        # empty BC
        self.boundary_conditions = []
        # empty constraints
        self.g = []  # list of constrain functions, not fully implemented / tested

        # system parameters
        self.M = len(c)  # number of ion species

        self.c = c  # concentrations
        self.z = z  # number charges
        self.T = T  # temperature

        self.relative_permittivity = relative_permittivity
        self.vacuum_permittivity = vacuum_permittivity
        # R = N_A * k_B
        # (universal gas constant = Avogadro constant * Boltzmann constant)
        self.R = R
        self.F = F

        self.f = F / (R*T)  # for convenience

        # print all quantities to log
        for i, (c, z) in enumerate(zip(self.c, self.z)):
            self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
                "ion species {:02d} concentration c".format(i), c, lwidth=self.label_width))
            self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
                "ion species {:02d} number charge z".format(i), z, lwidth=self.label_width))

        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'temperature T', self.T, lwidth=self.label_width))
        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'relative permittivity eps_R', self.relative_permittivity, lwidth=self.label_width))
        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'vacuum permittivity eps_0', self.vacuum_permittivity, lwidth=self.label_width))
        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'universal gas constant R', self.R, lwidth=self.label_width))
        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'Faraday constant F', self.F, lwidth=self.label_width))
        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'f = F / (RT)', self.f, lwidth=self.label_width))

        # scaled units for dimensionless formulation

        # length unit chosen as Debye length lambda
        self.l_unit = self.lambda_D

        # concentration unit is ionic strength
        self.c_unit = self.ionic_strength

        # no time unit for now, only steady state
        # self.t_unit = self.l_unit**2 / self.Dn # fixes Dn_scaled = 1

        self.u_unit = self.R * self.T / self.F  # thermal voltage

        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'spatial unit [l]', self.l_unit, lwidth=self.label_width))
        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'concentration unit [c]', self.c_unit, lwidth=self.label_width))
        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'potential unit [u]', self.u_unit, lwidth=self.label_width))

        # domain
        # self.L_scaled = self.L / self.l_unit

        # bulk concentrations
        self.c_scaled = self.c / self.c_unit

        # print scaled quantities to log
        for i, c_scaled in enumerate(self.c_scaled):
            self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
                "ion species {:02d} reduced concentration c*".format(i),
                c_scaled, lwidth=self.label_width))

        # initialize initial value arrays
        if potential0 is not None:
            self.ui0 = potential0 / self.u_unit
        else:
            self.ui0 = None

        if concentration0 is not None:
            self.ni0 = concentration0 / self.c_unit
        else:
            self.ni0 = None

        self.zi0 = None

    def __init__(self, *args, **kwargs):
        """Constructor, see init doc string for arguments."""

        self.logger = logging.getLogger(__name__)

        # default output settings
        self.label_width = 40  # character width of quantity labels in log

        # empty BC
        self.boundary_conditions = []
        # empty constraints
        self.g = []  # list of constrain functions, not fully implemented / tested

        # system parameters
        self.M = 0  # number of ion species

        self.c = 0  # concentrations
        self.z = 0  # number charges
        self.T = 0  # temperature

        self.relative_permittivity = 0
        self.vacuum_permittivity = 0

        self.R = 0
        self.F = 0

        self.f = 0

        self.l_unit = 0
        self.c_unit = 0
        self.u_unit = 0

        self.c_scaled = 0

        self.ui0 = None
        self.ni0 = None
        self.zi0 = None

        self.init(*args, **kwargs)