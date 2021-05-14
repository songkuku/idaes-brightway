##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2020, by the
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
Air separation phase equilibrium package using Peng-Robinson EoS.
Example property package using the Generic Property Package Framework.
This example shows how to set up a property package to do air separation
phase equilibrium in the generic framework using Peng-Robinson equation
along with methods drawn from the pre-built IDAES property libraries.
The example includes two dictionaries.
1. The dictionary named configuration contains parameters obtained from
The Properties of Gases and Liquids (1987) 4th edition and NIST.
2. The dictionary named configuration_Dowling_2015 contains parameters used in
A framework for efficient large scale equation-oriented flowsheet optimization
(2015) Dowling. The parameters are extracted from Properties of Gases and
Liquids (1977) 3rd edition for Antoine's vapor equation and acentric factors
and converted values from the Properties of Gases and Liquids (1977)
3rd edition to j.
"""

# Import Python libraries
import logging

# Import Pyomo units
from pyomo.environ import units as pyunits

# Import IDAES cores
from idaes.core import LiquidPhase, VaporPhase, Component

from idaes.generic_models.properties.core.state_definitions import FTPx
from idaes.generic_models.properties.core.eos.ideal import Ideal
from idaes.generic_models.properties.core.phase_equil import SmoothVLE
from idaes.generic_models.properties.core.phase_equil.bubble_dew import \
        IdealBubbleDew
from idaes.generic_models.properties.core.pure import RPP4
from idaes.generic_models.properties.core.pure import NIST
from idaes.generic_models.properties.core.pure import RPP3

from idaes.core import LiquidPhase, VaporPhase, Component
from idaes.core.phases import PhaseType as PT

from idaes.generic_models.properties.core.state_definitions import FTPx
from idaes.generic_models.properties.core.eos.ideal import Ideal
from idaes.generic_models.properties.core.phase_equil import SmoothVLE
from idaes.generic_models.properties.core.phase_equil.bubble_dew import \
        IdealBubbleDew
from idaes.generic_models.properties.core.phase_equil.forms import fugacity

from idaes.generic_models.properties.core.pure.Perrys import Perrys
from idaes.generic_models.properties.core.pure.NIST import NIST


# Set up logger
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Configuration dictionary for a Peng-Robinson Oxygen-Argon-Nitrogen system

# Data Sources:
# [1] The Properties of Gases and Liquids (1987)
#     4th edition, Chemical Engineering Series - Robert C. Reid
# [2] NIST, https://webbook.nist.gov/
#     Retrieved 16th August, 2020
# [3] The Properties of Gases and Liquids (1987)
#     3rd edition, Chemical Engineering Series - Robert C. Reid
#     Cp parameters where converted to j in Dowling 2015
# [4] A framework for efficient large scale equation-oriented flowsheet optimization (2015)
#


configuration = {
    # Specifying components
    "components": {
        'ethanol': {"type": Component,
                    "dens_mol_liq_comp": Perrys,
                    "enth_mol_liq_comp": Perrys,
                    "enth_mol_ig_comp": RPP4,
                    "pressure_sat_comp": RPP4,
                    "phase_equilibrium_form": {("Vap", "Liq"): fugacity},
                    "parameter_data": {
                        "mw": (46.07E-3, pyunits.kg/pyunits.mol),  # [1]
                        "pressure_crit": (64e5, pyunits.Pa),  # [1]
                        "temperature_crit": (515.15, pyunits.K),  # [1]
                        "dens_mol_liq_comp_coeff": {
                            '1': (1.048, pyunits.kmol*pyunits.m**-3),  # [2] pg. 2-98
                            '2': (0.27627, None),
                            '3': (513.92, pyunits.K),
                            '4': (0.2331, None)},
                         "cp_mol_ig_comp_coeff": {
                             "A": (9.014,
                                   pyunits.J/pyunits.mol/pyunits.K),  # [1]
                             "B": (2.141E-1,
                                   pyunits.J/pyunits.mol/pyunits.K**2),
                             "C": (-8.39E-05,
                                   pyunits.J/pyunits.mol/pyunits.K**3),
                             "D": (1.37E-09,
                                   pyunits.J/pyunits.mol/pyunits.K**4)},
                        "cp_mol_liq_comp_coeff": {
                            '1': (1.2064E2, pyunits.J/pyunits.kmol/pyunits.K),  # [2]
                            '2': (-1.3963E-1, pyunits.J/pyunits.kmol/pyunits.K**2),
                            '3': (-3.0341E-5, pyunits.J/pyunits.kmol/pyunits.K**3),
                            '4': (2.0386E-6, pyunits.J/pyunits.kmol/pyunits.K**4),
                            '5': (0, pyunits.J/pyunits.kmol/pyunits.K**5)},
                        "enth_mol_form_liq_comp_ref": (
                            -277.6e3, pyunits.J/pyunits.mol),  # [3]
                        "enth_mol_form_vap_comp_ref": (
                            -234.8e3, pyunits.J/pyunits.mol),  # [3]
                        "pressure_sat_comp_coeff": {'A': (-8.51838, None),  # [1]
                                                    'B': (0.34163, None),
                                                    'C': (-5.73683, None),
                                                    'D': (8.32581, None)}}},

        'water': {"type": Component,
                "dens_mol_liq_comp": Perrys,
                "enth_mol_liq_comp": Perrys,
                "enth_mol_ig_comp": NIST,
                "pressure_sat_comp": NIST,
                "phase_equilibrium_form": {("Vap", "Liq"): fugacity},
                "parameter_data": {
                    "mw": (18.0153E-3, pyunits.kg/pyunits.mol),  # [1]
                    "pressure_crit": (220.64E5, pyunits.Pa),  # [1]
                    "temperature_crit": (647, pyunits.K),  # [1]
                    "dens_mol_liq_comp_coeff": {
                        '1': (5.459, pyunits.kmol*pyunits.m**-3),  # [2] pg. 2-98, temperature range 273.16 K - 333.15 K
                        '2': (0.30542, None),
                        '3': (647.13, pyunits.K),
                        '4': (0.081, None)},
                    "cp_mol_ig_comp_coeff": {
                        'A': (30.09200, pyunits.J/pyunits.mol/pyunits.K),  # [1] temperature range 500 K- 1700 K
                        'B': (6.832514, pyunits.J*pyunits.mol**-1*pyunits.K**-1*pyunits.kiloK**-1),
                        'C': (6.793435, pyunits.J*pyunits.mol**-1*pyunits.K**-1*pyunits.kiloK**-2),
                        'D': (-2.534480, pyunits.J*pyunits.mol**-1*pyunits.K**-1*pyunits.kiloK**-3),
                        'E': (0.082139, pyunits.J*pyunits.mol**-1*pyunits.K**-1*pyunits.kiloK**2),
                        'F': (-250.8810, pyunits.kJ/pyunits.mol),
                        'G': (223.3967, pyunits.J/pyunits.mol/pyunits.K),
                        'H': (0, pyunits.kJ/pyunits.mol)},
                    "cp_mol_liq_comp_coeff": {
                        '1': (2.7637E5, pyunits.J/pyunits.kmol/pyunits.K),  # [2] pg 2-174, temperature range 273.16 K - 533.15 K
                        '2': (-2.0901E3, pyunits.J/pyunits.kmol/pyunits.K**2),
                        '3': (8.125, pyunits.J/pyunits.kmol/pyunits.K**3),
                        '4': (-1.4116E-2, pyunits.J/pyunits.kmol/pyunits.K**4),
                        '5': (9.3701E-6, pyunits.J/pyunits.kmol/pyunits.K**5)},
                    "enth_mol_form_liq_comp_ref": (
                        -285.83E3, pyunits.J/pyunits.mol),  # [1]
                    "enth_mol_form_vap_comp_ref": (
                        0, pyunits.J/pyunits.mol),  # [1]
                    "pressure_sat_comp_coeff": {
                        'A': (4.6543, None),  # [1], temperature range 255.9 K - 373 K
                        'B': (1435.264, pyunits.K),
                        'C': (-64.848, pyunits.K)}}},

        # 'CO2': {"type": Component,
        #         "valid_phase_types": PT.vaporPhase,
        #         "enth_mol_ig_comp": NIST,
        #         "parameter_data": {
        #            "mw": (44.0095E-3, pyunits.kg/pyunits.mol),  # [1]
        #            "pressure_crit": (73.825E5, pyunits.Pa),  # [1]
        #            "temperature_crit": (304.23, pyunits.K),  # [1]
        #            "cp_mol_ig_comp_coeff": {                 # [1], temperature range 298 K - 1200 K
        #                'A': (24.99735, pyunits.J/pyunits.mol/pyunits.K),
        #                'B': (55.18696, pyunits.J*pyunits.mol**-1*pyunits.K**-1*pyunits.kiloK**-1),
        #                'C': (-33.69137, pyunits.J*pyunits.mol**-1*pyunits.K**-1*pyunits.kiloK**-2),
        #                'D': (7.948387, pyunits.J*pyunits.mol**-1*pyunits.K**-1*pyunits.kiloK**-3),
        #                'E': (-0.136638, pyunits.J*pyunits.mol**-1*pyunits.K**-1*pyunits.kiloK**2),
        #                'F': (-403.6075, pyunits.kJ/pyunits.mol),
        #                'G': (228.2431, pyunits.J/pyunits.mol/pyunits.K),
        #                'H': (0, pyunits.kJ/pyunits.mol)},
        #            "enth_mol_form_vap_comp_ref": (0, pyunits.J/pyunits.mol)# [1]
        #                  }}

        'CO2': {"type": Component,
                "dens_mol_liq_comp": Perrys,
                "enth_mol_liq_comp": Perrys,
                "enth_mol_ig_comp": NIST,
                "pressure_sat_comp": NIST,
                "phase_equilibrium_form": {("Vap", "Liq"): fugacity},
                "parameter_data": {
                    "mw": (44.0095E-3, pyunits.kg/pyunits.mol),  # [1]
                    "pressure_crit": (73.825E5, pyunits.Pa),  # [1]
                    "temperature_crit": (304.23, pyunits.K),  # [1]
                    "dens_mol_liq_comp_coeff": {
                        '1': (2.768, pyunits.kmol*pyunits.m**-3),  # [2] pg. 2-98, temperature range 273.16 K - 333.15 K
                        '2': (0.26212, None),
                        '3': (304.21, pyunits.K),
                        '4': (0.2908, None)},
                   "cp_mol_ig_comp_coeff": {                 # [1], temperature range 298 K - 1200 K
                       'A': (24.99735, pyunits.J/pyunits.mol/pyunits.K),
                       'B': (55.18696, pyunits.J*pyunits.mol**-1*pyunits.K**-1*pyunits.kiloK**-1),
                       'C': (-33.69137, pyunits.J*pyunits.mol**-1*pyunits.K**-1*pyunits.kiloK**-2),
                       'D': (7.948387, pyunits.J*pyunits.mol**-1*pyunits.K**-1*pyunits.kiloK**-3),
                       'E': (-0.136638, pyunits.J*pyunits.mol**-1*pyunits.K**-1*pyunits.kiloK**2),
                       'F': (-403.6075, pyunits.kJ/pyunits.mol),
                       'G': (228.2431, pyunits.J/pyunits.mol/pyunits.K),
                       'H': (0, pyunits.kJ/pyunits.mol)},
                    "cp_mol_liq_comp_coeff": { # J/kmol-K
                        '1': (-8.3043E3, pyunits.J/pyunits.kmol/pyunits.K),  # [2]
                        '2': (1.0437E2, pyunits.J/pyunits.kmol/pyunits.K**2),
                        '3': (-4.3333E-1, pyunits.J/pyunits.kmol/pyunits.K**3),
                        '4': (6.0042E-4, pyunits.J/pyunits.kmol/pyunits.K**4),
                        '5': (0, pyunits.J/pyunits.kmol/pyunits.K**5)},
                    "enth_mol_form_liq_comp_ref": (
                        -1000E3, pyunits.J/pyunits.mol),  # [1]
                    "enth_mol_form_vap_comp_ref": (
                        -393.52E3, pyunits.J/pyunits.mol),  # [1]
                    "pressure_sat_comp_coeff": {
                        'A': (5.24677, None),  # [1], temperature range 255.9 K - 373 K
                        'B': (1598.673, pyunits.K),
                        'C': (-46.424, pyunits.K)}}}
                        },


    # Specifying phases
    "phases":  {"Liq": {"type": LiquidPhase,
                        "equation_of_state": Ideal},
                "Vap": {"type": VaporPhase,
                        "equation_of_state": Ideal}},

    # Set base units of measurement
    "base_units": {"time": pyunits.s,
                   "length": pyunits.m,
                   "mass": pyunits.kg,
                   "amount": pyunits.mol,
                   "temperature": pyunits.K},

    # Specifying state definition
    "state_definition": FTPx,
    "state_bounds": {"flow_mol": (0, 10, 20, pyunits.mol/pyunits.s),
                     "temperature": (273.15, 323.15, 1000, pyunits.K),
                     "pressure": (5E4, 108900, 1e7, pyunits.Pa)},
                     #"mole_frac_comp": {"benzene":(0,0.5,1),"ethanol":(0,0.5,1),"water":(0,0.5,1),"CO2":(0,0.5,1)}},
    "pressure_ref": (101325, pyunits.Pa),
    "temperature_ref": (298.15, pyunits.K),

    # Defining phase equilibria
    "phases_in_equilibrium": [("Vap", "Liq")],
    "phase_equilibrium_state": {("Vap", "Liq"): SmoothVLE},
    "bubble_dew_method": IdealBubbleDew,}
