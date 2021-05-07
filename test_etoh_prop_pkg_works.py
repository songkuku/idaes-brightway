##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2019, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes-pse".
##############################################################################
"""
Example ideal parameter block for the VLE calucations for a
Benzene-Toluene-o-Xylene system.
"""

# Import Python libraries
import logging

# Import Pyomo libraries
from pyomo.environ import Constraint, Expression, log, NonNegativeReals,\
    Var, Set, Param, sqrt, log10, units as pyunits
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.util.calc_var_value import calculate_variable_from_constraint

# Import IDAES cores
from idaes.core import (declare_process_block_class,
                        MaterialFlowBasis,
                        PhysicalParameterBlock,
                        StateBlockData,
                        StateBlock,
                        MaterialBalanceType,
                        EnergyBalanceType,
                        Component,
                        LiquidPhase,
                        VaporPhase)
from idaes.core.util.initialization import (fix_state_vars,
                                            revert_state_vars,
                                            solve_indexed_blocks)
from idaes.core.util.misc import add_object_reference
from idaes.core.util.model_statistics import degrees_of_freedom, \
                                             number_unfixed_variables
from idaes.core.util.misc import extract_data
import idaes.logger as idaeslog

# Set up logger
_log = idaeslog.getLogger(__name__)


@declare_process_block_class("ETOHParameterBlock")
class ETOHParameterData(PhysicalParameterBlock):
    CONFIG = PhysicalParameterBlock.CONFIG()

    def build(self):
        '''
        Callable method for Block construction.
        '''
        super(ETOHParameterData, self).build()

        self._state_block_class = IdealStateBlock

        self.ethanol = Component()
        self.water = Component()
        self.CO2 = Component()
        #self.hydrogen = Component()

        self.Liq = LiquidPhase()
        self.Vap = VaporPhase()

        # List of components in each phase (optional)
        self.phase_comp = {"Liq": self.component_list,
                           "Vap": self.component_list}

        # List of phase equilibrium index
        self.phase_equilibrium_idx = Set(initialize=[1, 2, 3])

        self.phase_equilibrium_list = \
            {1: ["ethanol", ("Vap", "Liq")], #
             2: ["water", ("Vap", "Liq")],
             3: ["CO2", ("Vap", "Liq")]}
             #4: ["methane", ("Vap", "Liq")],
             #5: ["diphenyl", ("Vap", "Liq")]}

        # Thermodynamic reference state
        self.pressure_ref = Param(mutable=True,
                                  default=101325,
                                  doc='Reference pressure [Pa]')
        self.temperature_ref = Param(mutable=True,
                                     default=298.15,
                                     doc='Reference temperature [K]')

        # Source: The Properties of Gases and Liquids (1987)
        # 4th edition, Chemical Engineering Series - Robert C. Reid
        pressure_crit_data = {'ethanol': 63e5,
                                'water': 220.5e5,
                                'CO2':  73.8e5
                                }

        self.pressure_crit = Param(
            self.component_list,
            within=NonNegativeReals,
            mutable=False,
            initialize=extract_data(pressure_crit_data),
            doc='Critical pressure [Pa]')

        # Source: The Properties of Gases and Liquids (1987)
        # 4th edition, Chemical Engineering Series - Robert C. Reid
        temperature_crit_data = {'ethanol': 515.15,
                                    'water': 647.15,
                                    'CO2': 304.34
                                 }

        self.temperature_crit = Param(
            self.component_list,
            within=NonNegativeReals,
            mutable=False,
            initialize=extract_data(temperature_crit_data),
            doc='Critical temperature [K]')

        # Gas Constant
        self.gas_const = Param(within=NonNegativeReals,
                               mutable=False,
                               default=8.314,
                               doc='Gas Constant [J/mol.K]')

        # Source: The Properties of Gases and Liquids (1987)
        # 4th edition, Chemical Engineering Series - Robert C. Reid
        mw_comp_data = {'ethanol': 46.07E-3,
                        'water': 18.02E-3,
                        'CO2': 44.01e-3
                        }

        self.mw_comp = Param(self.component_list,
                             mutable=False,
                             initialize=extract_data(mw_comp_data),
                             doc="molecular weight Kg/mol")

        # Constants for liquid densities
        # Source: Perry's Chemical Engineers Handbook
        #         - Robert H. Perry (Cp_liq)
        dens_liq_data = {('ethanol', '1'): 1.048,   #todo    
                         ('ethanol', '2'): 0.27627,
                         ('ethanol', '3'): 513.92,
                         ('ethanol', '4'): 0.2331,
                         ('water', '1'): 5.459,
                         ('water', '2'): 0.30542,
                         ('water', '3'): 647.13,
                         ('water', '4'): 0.081,
                         ('CO2', '1'): 2.768,
                         ('CO2', '2'): 0.26212,
                         ('CO2', '3'): 304.21,
                         ('CO2', '4'): 0.2908}

        self.dens_liq_params = Param(
                self.component_list,
                ['1', '2', '3', '4'],
                mutable=False,
                initialize=extract_data(dens_liq_data),
                doc="Parameters to compute liquid densities")

        # Boiling point at standard pressure
        # Source: Perry's Chemical Engineers Handbook
        #         - Robert H. Perry (Cp_liq)
        bp_data = {('ethanol'): 351.52,
                   ('water'): 373.15,
                   ('CO2'): 194.69}

        self.temperature_boil = Param(
                self.component_list,
                mutable=False,
                initialize=extract_data(bp_data),
                doc="Pure component boiling points at standard pressure [K]")

        # Constants for specific heat capacity, enthalpy
        # Sources: The Properties of Gases and Liquids (1987)
        #         4th edition, Chemical Engineering Series - Robert C. Reid
        #         Perry's Chemical Engineers Handbook
        #         - Robert H. Perry (Cp_liq)
        # Unit: J/kmol-K
        cp_ig_data = {('Liq', 'ethanol', '1'): 1.2064E5,  #todo
                      ('Liq', 'ethanol', '2'): -1.3963E2,
                      ('Liq', 'ethanol', '3'): -3.0341E-2,
                      ('Liq', 'ethanol', '4'): 2.0386E-3,
                      ('Liq', 'ethanol', '5'): 0,
                      ('Vap', 'ethanol', '1'): 36.548344E3,
                      ('Vap', 'ethanol', '2'): 5.221192,
                      ('Vap', 'ethanol', '3'): 0.46109444,
                      ('Vap', 'ethanol', '4'): -0.000583975,
                      ('Vap', 'ethanol', '5'): 2.20986E-07,
                      ('Liq', 'water', '1'): 2.7637E5,  #reference: toluene 1.40e5
                      ('Liq', 'water', '2'): -2.0901E3,
                      ('Liq', 'water', '3'): 8.1250,
                      ('Liq', 'water', '4'): -1.4116E-2,
                      ('Liq', 'water', '5'): 9.3701E-6,
                      ('Vap', 'water', '1'): 3.654E4,
                      ('Vap', 'water', '2'): -34.802404,
                      ('Vap', 'water', '3'): -0.1168117,
                      ('Vap', 'water', '4'): -0.000130031,
                      ('Vap', 'water', '5'): 5.25445E-08,
                      ('Liq', 'CO2', '1'): -8.3043E6,  # 6.6653e1,
                      ('Liq', 'CO2', '2'): 1.0437E5,  # 6.7659e3,
                      ('Liq', 'CO2', '3'): -4.3333E2,  # -1.2363e2,
                      ('Liq', 'CO2', '4'): 6.0042E-1,  # 4.7827e2, # Eqn 2
                      ('Liq', 'CO2', '5'): 0,
                      ('Vap', 'CO2', '1'): 27095.326,
                      ('Vap', 'CO2', '2'): 11.273784,
                      ('Vap', 'CO2', '3'): 0.12487628,
                      ('Vap', 'CO2', '4'): -0.000197374,
                      ('Vap', 'CO2', '5'): 8.77958E-08}




        self.cp_ig = Param(self.phase_list, self.component_list,
                           ['1', '2', '3', '4', '5'],
                           mutable=False,
                           initialize=extract_data(cp_ig_data),
                           doc="parameters to compute Cp_comp")

        # Source: NIST
        # fitted to Antoine form
        # Unit: Pvp [bar] -> unit conversion later
        pressure_sat_coeff_data = {('ethanol', 'A'): 5.24677,
                                   ('ethanol', 'B'): 1598.673,
                                   ('ethanol', 'C'): -46.424,
                                   ('water', 'A'): 5.40221,
                                   ('water', 'B'): 1838.675,
                                   ('water', 'C'): -31.737	,
                                   ('CO2', 'A'): 6.812,
                                   ('CO2', 'B'): 1302,
                                   ('CO2', 'C'): -3.494}

                                   		
        self.pressure_sat_coeff = Param(
            self.component_list,
            ['A', 'B', 'C'],
            mutable=False,
            initialize=extract_data(pressure_sat_coeff_data),
            doc="parameters to compute Cp_comp")

        # Source: The Properties of Gases and Liquids (1987)
        # 4th edition, Chemical Engineering Series - Robert C. Reid
        dh_vap = {'ethanol': 42.4e3, 'water': 43.86e3, 'CO2': 16.5e3}

        self.dh_vap = Param(self.component_list,
                            mutable=False,
                            initialize=extract_data(dh_vap),
                            doc="heat of vaporization")

    @classmethod
    def define_metadata(cls, obj):
        """Define properties supported and units."""
        obj.add_properties(
            {'flow_mol': {'method': None, 'units': 'mol/s'},
             'flow_mol_phase_comp': {'method': None, 'units': 'mol/s'},
             'mole_frac_comp': {'method': None, 'units': 'none'},
             'temperature': {'method': None, 'units': 'K'},
             'pressure': {'method': None, 'units': 'Pa'},
             'flow_mol_phase': {'method': None, 'units': 'mol/s'},
             'dens_mol_phase': {'method': '_dens_mol_phase',
                                'units': 'mol/m^3'},
             'pressure_sat': {'method': '_pressure_sat', 'units': 'Pa'},
             'mole_frac_phase_comp': {'method': '_mole_frac_phase',
                                      'units': 'no unit'},
             'energy_internal_mol_phase_comp': {
                     'method': '_energy_internal_mol_phase_comp',
                     'units': 'J/mol'},
             'energy_internal_mol_phase': {
                     'method': '_enenrgy_internal_mol_phase',
                     'units': 'J/mol'},
             'enth_mol_phase_comp': {'method': '_enth_mol_phase_comp',
                                     'units': 'J/mol'},
             'enth_mol_phase': {'method': '_enth_mol_phase',
                                'units': 'J/mol'},
             'entr_mol_phase_comp': {'method': '_entr_mol_phase_comp',
                                     'units': 'J/mol'},
             'entr_mol_phase': {'method': '_entr_mol_phase',
                                'units': 'J/mol'},
             'temperature_bubble': {'method': '_temperature_bubble',
                                    'units': 'K'},
             'temperature_dew': {'method': '_temperature_dew',
                                 'units': 'K'},
             'pressure_bubble': {'method': '_pressure_bubble',
                                 'units': 'Pa'},
             'pressure_dew': {'method': '_pressure_dew',
                              'units': 'Pa'},
             'fug_vap': {'method': '_fug_vap', 'units': 'Pa'},
             'fug_liq': {'method': '_fug_liq', 'units': 'Pa'},
             'dh_vap': {'method': '_dh_vap', 'units': 'J/mol'},
             'ds_vap': {'method': '_ds_vap', 'units': 'J/mol.K'}})

        obj.add_default_units({'time': pyunits.s,
                               'length': pyunits.m,
                               'mass': pyunits.g,
                               'amount': pyunits.mol,
                               'temperature': pyunits.K})


class _IdealStateBlock(StateBlock):
    """
    This Class contains methods which should be applied to Property Blocks as a
    whole, rather than individual elements of indexed Property Blocks.
    """

    def initialize(blk, state_args={}, state_vars_fixed=False,
                   hold_state=False, outlvl=idaeslog.NOTSET,
                   solver='ipopt', optarg={'tol': 1e-8}):
        """
        Initialization routine for property package.
        Keyword Arguments:
            state_args : Dictionary with initial guesses for the state vars
                         chosen. Note that if this method is triggered
                         through the control volume, and if initial guesses
                         were not provied at the unit model level, the
                         control volume passes the inlet values as initial
                         guess.The keys for the state_args dictionary are:

                         flow_mol_phase_comp : value at which to initialize
                                               phase component flows
                         pressure : value at which to initialize pressure
                         temperature : value at which to initialize temperature
            outlvl : sets output level of initialization routine
                     * 0 = no output (default)
                     * 1 = return solver state for each step in routine
                     * 2 = include solver output infomation (tee=True)
            optarg : solver options dictionary object (default=None)
            state_vars_fixed: Flag to denote if state vars have already been
                              fixed.
                              - True - states have already been fixed by the
                                       control volume 1D. Control volume 0D
                                       does not fix the state vars, so will
                                       be False if this state block is used
                                       with 0D blocks.
                             - False - states have not been fixed. The state
                                       block will deal with fixing/unfixing.
            solver : str indicating whcih solver to use during
                     initialization (default = 'ipopt')
            hold_state : flag indicating whether the initialization routine
                         should unfix any state variables fixed during
                         initialization (default=False).
                         - True - states varaibles are not unfixed, and
                                 a dict of returned containing flags for
                                 which states were fixed during
                                 initialization.
                        - False - state variables are unfixed after
                                 initialization by calling the
                                 relase_state method
        Returns:
            If hold_states is True, returns a dict containing flags for
            which states were fixed during initialization.
        """

        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="properties")
        solve_log = idaeslog.getSolveLogger(blk.name, outlvl, tag="properties")

        # Fix state variables if not already fixed
        if state_vars_fixed is False:
            flags = fix_state_vars(blk, state_args)

        else:
            # Check when the state vars are fixed already result in dof 0
            for k in blk.keys():
                if degrees_of_freedom(blk[k]) != 0:
                    raise Exception("State vars fixed but degrees of freedom "
                                    "for state block is not zero during "
                                    "initialization.")
        # Set solver options
        if optarg is None:
            sopt = {"tol": 1e-8}
        else:
            sopt = optarg

        opt = SolverFactory('ipopt')
        opt.options = sopt

        # ---------------------------------------------------------------------
        # If present, initialize bubble and dew point calculations
        for k in blk.keys():
            if hasattr(blk[k], "eq_temperature_dew"):
                calculate_variable_from_constraint(blk[k].temperature_dew,
                                                   blk[k].eq_temperature_dew)

            if hasattr(blk[k], "eq_pressure_dew"):
                calculate_variable_from_constraint(blk[k].pressure_dew,
                                                   blk[k].eq_pressure_dew)

        init_log.info_high("Initialization Step 1 - Dew and bubble points "
                      "calculation completed.")

        # ---------------------------------------------------------------------
        # If flash, initialize T1 and Teq
        for k in blk.keys():
            if (blk[k].config.has_phase_equilibrium and
                    not blk[k].config.defined_state):
                blk[k]._t1.value = max(blk[k].temperature.value,
                                       blk[k].temperature_bubble.value)
                blk[k]._teq.value = min(blk[k]._t1.value,
                                        blk[k].temperature_dew.value)

        init_log.info_high("Initialization Step 2 - Equilibrium temperature "
                           " calculation completed.")

        # ---------------------------------------------------------------------
        # Initialize flow rates and compositions
        # TODO : This will need to be generalised more when we move to a
        # modular implementation
        for k in blk.keys():
            # Deactivate equilibrium constraints, as state is fixed
            if hasattr(blk[k], 'equilibrium_constraint'):
                blk[k].equilibrium_constraint.deactivate()

        free_vars = 0
        for k in blk.keys():
            free_vars += number_unfixed_variables(blk[k])
        if free_vars > 0:
            try:
                with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                    res = solve_indexed_blocks(opt, [blk], tee=slc.tee)
            except:
                res = None
        else:
            res = None

        for k in blk.keys():
            # Reactivate equilibrium constraints
            if hasattr(blk[k], 'equilibrium_constraint'):
                blk[k].equilibrium_constraint.activate()

        # ---------------------------------------------------------------------
        # Return state to initial conditions
        if state_vars_fixed is False:
            if hold_state is True:
                return flags
            else:
                blk.release_state(flags)

        init_log.info("Initialization Complete")

    def release_state(blk, flags, outlvl=0):
        '''
        Method to relase state variables fixed during initialization.
        Keyword Arguments:
            flags : dict containing information of which state variables
                    were fixed during initialization, and should now be
                    unfixed. This dict is returned by initialize if
                    hold_state=True.
            outlvl : sets output level of of logging
        '''
        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="properties")
        if flags is None:
            init_log.debug("No flags passed to release_state().")
            return

        # Unfix state variables
        revert_state_vars(blk, flags)

        init_log.info_high("State Released.")


@declare_process_block_class("IdealStateBlock",
                             block_class=_IdealStateBlock)
class IdealStateBlockData(StateBlockData):
    """An example property package for ideal VLE."""

    def build(self):
        """Callable method for Block construction."""
        super(IdealStateBlockData, self).build()

        # Add state variables
        self.flow_mol_phase_comp = Var(
                self._params.phase_list,
                self._params.component_list,
                initialize=0.5,
                bounds=(1e-8, 100),
                doc='Phase-component molar flow rates [mol/s]')

        self.pressure = Var(initialize=101325,
                            bounds=(101325, 400000),
                            domain=NonNegativeReals,
                            doc='State pressure [Pa]')
        self.temperature = Var(initialize=298.15,
                               bounds=(298.15, 1000),
                               domain=NonNegativeReals,
                               doc='State temperature [K]')

        # Add supporting variables
        def flow_mol_phase(b, p):
            return sum(b.flow_mol_phase_comp[p, j]
                       for j in b._params.component_list)
        self.flow_mol_phase = Expression(self._params.phase_list,
                                         rule=flow_mol_phase,
                                         doc='Phase molar flow rates [mol/s]')

        def flow_mol(b):
            return sum(b.flow_mol_phase_comp[p, j]
                       for j in b._params.component_list
                       for p in b._params.phase_list)
        self.flow_mol = Expression(rule=flow_mol,
                                   doc='Total molar flowrate [mol/s]')

        def mole_frac_phase_comp(b, p, j):
            return b.flow_mol_phase_comp[p, j]/b.flow_mol_phase[p]
        self.mole_frac_phase_comp = Expression(
                self._params.phase_list,
                self._params.component_list,
                rule=mole_frac_phase_comp,
                doc='Phase mole fractions [-]')

        def mole_frac_comp(b, j):
            return (sum(b.flow_mol_phase_comp[p, j]
                        for p in b._params.phase_list) / b.flow_mol)
        self.mole_frac_comp = Expression(self._params.component_list,
                                         rule=mole_frac_comp,
                                         doc='Mixture mole fractions [-]')

        # Reaction Stoichiometry
        add_object_reference(self, "phase_equilibrium_list_ref",
                             self._params.phase_equilibrium_list)

        if (self.config.has_phase_equilibrium and
                self.config.defined_state is False):
            # Definition of equilibrium temperature for smooth VLE
            self._teq = Var(
                    initialize=self.temperature.value,
                    doc='Temperature for calculating phase equilibrium')
            self._t1 = Var(initialize=self.temperature.value,
                           doc='Intermediate temperature for calculating Teq')

            self.eps_1 = Param(default=0.01,
                               mutable=True,
                               doc='Smoothing parameter for Teq')
            self.eps_2 = Param(default=0.0005,
                               mutable=True,
                               doc='Smoothing parameter for Teq')

            # PSE paper Eqn 13
            def rule_t1(b):
                return b._t1 == 0.5*(
                        b.temperature + b.temperature_bubble +
                        sqrt((b.temperature-b.temperature_bubble)**2 +
                             b.eps_1**2))
            self._t1_constraint = Constraint(rule=rule_t1)

            # PSE paper Eqn 14
            # TODO : Add option for supercritical extension
            def rule_teq(b):
                return b._teq == 0.5*(b._t1 + b.temperature_dew -
                                      sqrt((b._t1-b.temperature_dew)**2 +
                                           b.eps_2**2))
            self._teq_constraint = Constraint(rule=rule_teq)

            def rule_tr_eq(b, i):
                return b._teq / b._params.temperature_crit[i]
            self._tr_eq = Expression(
                    self._params.component_list,
                    rule=rule_tr_eq,
                    doc='Component reduced temperatures [-]')

            def rule_equilibrium(b, i):
                return b.fug_vap[i] == b.fug_liq[i]
            self.equilibrium_constraint = Constraint(
                    self._params.component_list, rule=rule_equilibrium)

# -----------------------------------------------------------------------------
# Property Methods
    def _dens_mol_phase(self):
        self.dens_mol_phase = Var(self._params.phase_list,
                                  initialize=1.0,
                                  doc="Molar density [mol/m^3]")

        def rule_dens_mol_phase(b, p):
            if p == 'Vap':
                return b._dens_mol_vap()
            else:
                return b._dens_mol_liq()
        self.eq_dens_mol_phase = Constraint(self._params.phase_list,
                                            rule=rule_dens_mol_phase)

    def _energy_internal_mol_phase_comp(self):
        self.energy_internal_mol_phase_comp = Var(
                self._params.phase_list,
                self._params.component_list,
                doc="Phase-component molar specific internal energies [J/mol]")

        def rule_energy_internal_mol_phase_comp(b, p, j):
            if p == 'Vap':
                return b.energy_internal_mol_phase_comp[p, j] == \
                        b.enth_mol_phase_comp[p, j] - \
                        b._params.gas_const*(b.temperature -
                                             b._params.temeprature_ref)
            else:
                return b.energy_internal_mol_phase_comp[p, j] == \
                        b.enth_mol_phase_comp[p, j]
        self.eq_energy_internal_mol_phase_comp = Constraint(
            self._params.phase_list,
            self._params.component_list,
            rule=rule_energy_internal_mol_phase_comp)

    def _energy_internal_mol_phase(self):
        self.energy_internal_mol_phase = Var(
            self._params.phase_list,
            doc='Phase molar specific internal energies [J/mol]')

        def rule_energy_internal_mol_phase(b, p):
            return b.energy_internal_mol_phase[p] == sum(
                b.energy_internal_mol_phase_comp[p, i] *
                b.mole_frac_phase_comp[p, i]
                for i in b._params.component_list)
        self.eq_energy_internal_mol_phase = Constraint(
                self._params.phase_list,
                rule=rule_energy_internal_mol_phase)

    def _enth_mol_phase_comp(self):
        self.enth_mol_phase_comp = Var(
                self._params.phase_list,
                self._params.component_list,
                initialize=7e5,
                doc='Phase-component molar specific enthalpies [J/mol]')

        def rule_enth_mol_phase_comp(b, p, j):
            if p == 'Vap':
                return b._enth_mol_comp_vap(j)
            else:
                return b._enth_mol_comp_liq(j)
        self.eq_enth_mol_phase_comp = Constraint(
                self._params.phase_list,
                self._params.component_list,
                rule=rule_enth_mol_phase_comp)

    def _enth_mol_phase(self):
        self.enth_mol_phase = Var(
                self._params.phase_list,
                initialize=7e5,
                doc='Phase molar specific enthalpies [J/mol]')

        def rule_enth_mol_phase(b, p):
            return b.enth_mol_phase[p] == sum(
                    b.enth_mol_phase_comp[p, i] *
                    b.mole_frac_phase_comp[p, i]
                    for i in b._params.component_list)
        self.eq_enth_mol_phase = Constraint(self._params.phase_list,
                                            rule=rule_enth_mol_phase)

    def _entr_mol_phase_comp(self):
        self.entr_mol_phase_comp = Var(
                self._params.phase_list,
                self._params.component_list,
                doc='Phase-component molar specific entropies [J/mol.K]')

        def rule_entr_mol_phase_comp(b, p, j):
            if p == 'Vap':
                return b._entr_mol_comp_vap(j)
            else:
                return b._entr_mol_comp_liq(j)
        self.eq_entr_mol_phase_comp = Constraint(
                self._params.phase_list,
                self._params.component_list,
                rule=rule_entr_mol_phase_comp)

    def _entr_mol_phase(self):
        self.entr_mol_phase = Var(
                self._params.phase_list,
                doc='Phase molar specific enthropies [J/mol.K]')

        def rule_entr_mol_phase(b, p):
            return b.entr_mol_phase[p] == sum(
                    b.entr_mol_phase_comp[p, i] *
                    b.mole_frac_phase_comp[p, i]
                    for i in b._params.component_list)
        self.eq_entr_mol_phase = Constraint(self._params.phase_list,
                                            rule=rule_entr_mol_phase)

# -----------------------------------------------------------------------------
# General Methods
    def get_material_flow_terms(self, p, j):
        """Create material flow terms for control volume."""
        if j in self._params.component_list:
            return self.flow_mol_phase_comp[p, j]
        else:
            return 0

    def get_enthalpy_flow_terms(self, p):
        """Create enthalpy flow terms."""
        return self.flow_mol_phase[p] * self.enth_mol_phase[p]

    def get_material_density_terms(self, p, j):
        """Create material density terms."""
        if j in self._params.component_list:
            return self.dens_mol_phase[p] * self.mole_frac_phase_comp[p, j]
        else:
            return 0

    def get_enthalpy_density_terms(self, p):
        """Create enthalpy density terms."""
        return self.dens_mol_phase[p] * self.energy_internal_mol_phase[p]

    def default_material_balance_type(self):
        return MaterialBalanceType.componentPhase

    def default_energy_balance_type(self):
        return EnergyBalanceType.enthalpyTotal

    def get_material_flow_basis(b):
        return MaterialFlowBasis.molar

    def define_state_vars(self):
        """Define state vars."""
        return {"flow_mol_phase_comp": self.flow_mol_phase_comp,
                "temperature": self.temperature,
                "pressure": self.pressure}

    # Property package utility functions
    def calculate_bubble_point_temperature(self, clear_components=True):
        """"To compute the bubble point temperature of the mixture."""

        if hasattr(self, "eq_temperature_bubble"):
            # Do not delete components if the block already has the components
            clear_components = False

        calculate_variable_from_constraint(self.temperature_bubble,
                                           self.eq_temperature_bubble)

        return self.temperature_bubble.value

        if clear_components is True:
            self.del_component(self.eq_temperature_bubble)
            self.del_component(self._p_sat_bubbleT)
            self.del_component(self.temperature_bubble)

    def calculate_dew_point_temperature(self, clear_components=True):
        """"To compute the dew point temperature of the mixture."""

        if hasattr(self, "eq_temperature_dew"):
            # Do not delete components if the block already has the components
            clear_components = False

        calculate_variable_from_constraint(self.temperature_dew,
                                           self.eq_temperature_dew)

        return self.temperature_dew.value

        # Delete the var/constraint created in this method that are part of the
        # IdealStateBlock if the user desires
        if clear_components is True:
            self.del_component(self.eq_temperature_dew)
            self.del_component(self._p_sat_dewT)
            self.del_component(self.temperature_dew)

    def calculate_bubble_point_pressure(self, clear_components=True):
        """"To compute the bubble point pressure of the mixture."""

        if hasattr(self, "eq_pressure_bubble"):
            # Do not delete components if the block already has the components
            clear_components = False

        calculate_variable_from_constraint(self.pressure_bubble,
                                           self.eq_pressure_bubble)

        return self.pressure_bubble.value

        # Delete the var/constraint created in this method that are part of the
        # IdealStateBlock if the user desires
        if clear_components is True:
            self.del_component(self.eq_pressure_bubble)
            self.del_component(self._p_sat_bubbleP)
            self.del_component(self.pressure_bubble)

    def calculate_dew_point_pressure(self, clear_components=True):
        """"To compute the dew point pressure of the mixture."""

        if hasattr(self, "eq_pressure_dew"):
            # Do not delete components if the block already has the components
            clear_components = False

        calculate_variable_from_constraint(self.pressure_dew,
                                           self.eq_pressure_dew)

        return self.pressure_dew.value

        # Delete the var/constraint created in this method that are part of the
        # IdealStateBlock if the user desires
        if clear_components is True:
            self.del_component(self.eq_pressure_dew)
            self.del_component(self._p_sat_dewP)
            self.del_component(self.pressure_dew)

# -----------------------------------------------------------------------------
# Bubble and Dew Points
# Ideal-Ideal properties allow for the simplifications below
# Other methods require more complex equations with shadow compositions

# For future work, propose the following:
# Core class writes a set of constraints Phi_L_i == Phi_V_i
# Phi_L_i and Phi_V_i make calls to submethods which add shadow compositions
# as needed
    def _temperature_bubble(self):
        self.temperature_bubble = Param(initialize=33.0,
                                        doc="Bubble point temperature (K)")

    def _temperature_dew(self):

        self.temperature_dew = Var(initialize=298.15,
                                   doc="Dew point temperature (K)")

        def rule_psat_dew(b, j):
            return 1e5*10**(b._params.pressure_sat_coeff[j, 'A'] -
                            b._params.pressure_sat_coeff[j, 'B'] /
                            (b.temperature_dew +
                             b._params.pressure_sat_coeff[j, 'C']))

        try:
            # Try to build expression
            self._p_sat_dewT = Expression(self._params.component_list,
                                          rule=rule_psat_dew)

            def rule_temp_dew(b):
                return b.pressure * sum(b.mole_frac_comp[i] /
                                        b._p_sat_dewT[i]
                                        for i in ['ethanol', 'water']) \
                    - 1 == 0
            self.eq_temperature_dew = Constraint(rule=rule_temp_dew)
        except AttributeError:
            # If expression fails, clean up so that DAE can try again later
            # Deleting only var/expression as expression construction will fail
            # first; if it passes then constraint construction will not fail.
            self.del_component(self.temperature_dew)
            self.del_component(self._p_sat_dewT)

    def _pressure_bubble(self):
        self.pressure_bubble = Param(initialize=1e8,
                                     doc="Bubble point pressure (Pa)")

    def _pressure_dew(self):
        self.pressure_dew = Var(initialize=298.15,
                                doc="Dew point pressure (Pa)")

        def rule_psat_dew(b, j):
            return 1e5*10**(b._params.pressure_sat_coeff[j, 'A'] -
                            b._params.pressure_sat_coeff[j, 'B'] /
                            (b.temperature +
                             b._params.pressure_sat_coeff[j, 'C']))

        try:
            # Try to build expression
            self._p_sat_dewP = Expression(self._params.component_list,
                                          rule=rule_psat_dew)

            def rule_pressure_dew(b):
                return b.pressure_dew * \
                    sum(b.mole_frac_comp[i] / b._p_sat_dewP[i]
                        for i in ['ethanol', 'water']) \
                    - 1 == 0
            self.eq_pressure_dew = Constraint(rule=rule_pressure_dew)
        except AttributeError:
            # If expression fails, clean up so that DAE can try again later
            # Deleting only var/expression as expression construction will fail
            # first; if it passes then constraint construction will not fail.
            self.del_component(self.pressure_dew)
            self.del_component(self._p_sat_dewP)

# -----------------------------------------------------------------------------
# Liquid phase properties
    def _dens_mol_liq(b):
        return b.dens_mol_phase['Liq'] == 1e3*sum(
                b.mole_frac_phase_comp['Liq', j] *
                b._params.dens_liq_params[j, '1'] /
                b._params.dens_liq_params[j, '2'] **
                (1 + (1-b.temperature /
                      b._params.dens_liq_params[j, '3']) **
                 b._params.dens_liq_params[j, '4'])
                for j in ['water', 'ethanol'])                                # TODO: Need to include diphenyl here later

    def _fug_liq(self):
        def fug_liq_rule(b, i):
            if i in ['CO2']:
                return b.mole_frac_phase_comp['Liq', i]
            else:
                return b.pressure_sat[i] * b.mole_frac_phase_comp['Liq', i]
        self.fug_liq = Expression(self._params.component_list,
                                  rule=fug_liq_rule)

    def _pressure_sat(self):
        self.pressure_sat = Var(self._params.component_list,
                                initialize=101325,
                                doc="Vapor pressure [Pa]")

        def rule_P_sat(b, j):
            return ((log10(b.pressure_sat[j]*1e-5) -
                     b._params.pressure_sat_coeff[j, 'A']) *
                    (b._teq + b._params.pressure_sat_coeff[j, 'C'])) == \
                   -b._params.pressure_sat_coeff[j, 'B']
        self.eq_pressure_sat = Constraint(self._params.component_list,
                                          rule=rule_P_sat)

    def _enth_mol_comp_liq(b, j):  #todo
        return b.enth_mol_phase_comp['Liq', j] * 1E3 == \
                ((b._params.cp_ig['Liq', j, '5'] / 5) *
                    (b.temperature**5 - b._params.temperature_ref**5)
                    + (b._params.cp_ig['Liq', j, '4'] / 4) *
                      (b.temperature**4 - b._params.temperature_ref**4)
                    + (b._params.cp_ig['Liq', j, '3'] / 3) *
                      (b.temperature**3 - b._params.temperature_ref**3)
                    + (b._params.cp_ig['Liq', j, '2'] / 2) *
                      (b.temperature**2 - b._params.temperature_ref**2)
                    + b._params.cp_ig['Liq', j, '1'] *
                      (b.temperature - b._params.temperature_ref))

    def _entr_mol_comp_liq(b, j):
        return b.entr_mol_phase_comp['Liq', j] * 1E3 == (
                ((b._params.cp_ig['Liq', j, '5'] / 4) *
                    (b.temperature**4 - b._params.temperature_ref**4)
                    + (b._params.cp_ig['Liq', j, '4'] / 3) *
                      (b.temperature**3 - b._params.temperature_ref**3)
                    + (b._params.cp_ig['Liq', j, '3'] / 2) *
                      (b.temperature**2 - b._params.temperature_ref**2)
                    + b._params.cp_ig['Liq', j, '2'] *
                      (b.temperature - b._params.temperature_ref)
                    + b._params.cp_ig['Liq', j, '1'] *
                      log(b.temperature / b._params.temperature_ref)) -
                b._params.gas_const *
                log(b.mole_frac_phase_comp['Liq', j]*b.pressure /
                    b._params.pressure_ref))

# -----------------------------------------------------------------------------
# Vapour phase properties
    def _dens_mol_vap(b):
        return b.pressure == (b.dens_mol_phase['Vap'] *
                              b._params.gas_const *
                              b.temperature)

    def _fug_vap(self):
        def fug_vap_rule(b, i):
            if i in ['CO2']:
                return 1e-6
            else:
                return b.mole_frac_phase_comp['Vap', i] * b.pressure
        self.fug_vap = Expression(self._params.component_list,
                                  rule=fug_vap_rule)

    def _dh_vap(self):
        # heat of vaporization
        add_object_reference(self, "dh_vap",
                             self._params.dh_vap)

    def _ds_vap(self):
        # entropy of vaporization = dh_Vap/T_boil
        # TODO : something more rigorous would be nice
        self.ds_vap = Var(self._params.component_list,
                          initialize=86,
                          doc="Entropy of vaporization [J/mol.K]")

        def rule_ds_vap(b, j):
            return b.dh_vap[j] == (b.ds_vap[j] *
                                   b._params.temperature_boil[j])
        self.eq_ds_vap = Constraint(self._params.component_list,
                                    rule=rule_ds_vap)

    def _enth_mol_comp_vap(b, j):
        return b.enth_mol_phase_comp['Vap', j] == b.dh_vap[j] + \
                ((b._params.cp_ig['Vap', j, '5'] / 5) *
                    (b.temperature**5 - b._params.temperature_ref**5)
                    + (b._params.cp_ig['Vap', j, '4'] / 4) *
                      (b.temperature**4 - b._params.temperature_ref**4)
                    + (b._params.cp_ig['Vap', j, '3'] / 3) *
                      (b.temperature**3 - b._params.temperature_ref**3)
                    + (b._params.cp_ig['Vap', j, '2'] / 2) *
                      (b.temperature**2 - b._params.temperature_ref**2)
                    + b._params.cp_ig['Vap', j, '1'] *
                      (b.temperature - b._params.temperature_ref))

    def _entr_mol_comp_vap(b, j):
        return b.entr_mol_phase_comp['Vap', j] == (
                b.ds_vap[j] +
                ((b._params.cp_ig['Vap', j, '5'] / 4) *
                    (b.temperature**4 - b._params.temperature_ref**4)
                    + (b._params.cp_ig['Vap', j, '4'] / 3) *
                      (b.temperature**3 - b._params.temperature_ref**3)
                    + (b._params.cp_ig['Vap', j, '3'] / 2) *
                      (b.temperature**2 - b._params.temperature_ref**2)
                    + b._params.cp_ig['Vap', j, '2'] *
                      (b.temperature - b._params.temperature_ref)
                    + b._params.cp_ig['Vap', j, '1'] *
                      log(b.temperature / b._params.temperature_ref)) -
                b._params.gas_const *
                log(b.mole_frac_phase_comp['Vap', j]*b.pressure /
                    b._params.pressure_ref))
