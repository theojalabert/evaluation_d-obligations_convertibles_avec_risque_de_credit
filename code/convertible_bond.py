###########################################
###      PROJET - RISQUE DE CRÉDIT      ###
### ALLERS - JALABERT - SIMONEAU-FRIGGI ###
###      file = convertible_bond.py     ###
###########################################

"""
Fusion des paramètres et fonctionnalités des obligations convertibles.
"""

from __future__ import absolute_import

import numpy as np
from model import WienerJumpProcess
from payoff import CallA, Stack, Time

# Classe pour simplifier la création de composants d'une obligation convertible
class Annuity:
    def __init__(self, maturity, payment_times, C, N, R):
        self.maturity = maturity
        self.payment_times = payment_times
        self.coupon = C
        self.nominal_value = N
        self.recovery_rate = R

class Call:
    def __init__(self, maturity, strike, annuity):
        self.maturity = maturity
        self.strike = strike
        self.annuity = annuity

class Put:
    def __init__(self, maturity, strike, annuity):
        self.maturity = maturity
        self.strike = strike
        self.annuity = annuity


class ConvertibleBondConfig:
    def __init__(self):
        self.T = 5  # Time till maturity = 5 years
        self.set_processes()
        self.set_payoff_components()

    def set_processes(self):
        # Configurations de différentes dynamiques de prix de l'action avec saut de défaut
        self.dS = WienerJumpProcess(r=0.05, sigma=0.2)
        self.dS_total = WienerJumpProcess(r=0.05, sigma=0.2, lambd_=0.02, eta=1)
        self.dS_typical = WienerJumpProcess(r=0.05, sigma=0.2, lambd_=0.02, eta=0.3)
        self.dS_partial = WienerJumpProcess(r=0.05, sigma=0.2, lambd_=0.02, eta=0)
        self.dS_var12 = WienerJumpProcess(r=0.05, sigma=0.2, lambd_=lambda S: 0.02 * (S / 100)**-1.2, eta=1)
        self.dS_var20 = WienerJumpProcess(r=0.05, sigma=0.2, lambd_=lambda S: 0.02 * (S / 100)**-2.0, eta=1)
        self.dS_mk12 = WienerJumpProcess(r=0.05, sigma=0.25, lambd_=0.062, eta=1)
        self.dS_var_mk12 = WienerJumpProcess(r=0.05, sigma=0.25, lambd_=lambda S: 0.062 * (S / 50)**-0.5, eta=1)

    def set_payoff_components(self):
        # Composants de paiement pour l'obligation convertible
        self.A = Annuity(self.T, np.arange(0.5, self.T + 0.5, 0.5), C=4, N=100, R=0)
        self.P = Time(Put(self.T, 105, self.A), times=[3])
        self.C = Time(Call(self.T, 110, self.A), times=[(2, 5)])
        self.S = CallA(self.T, 0)
        self.B = Stack([self.A, self.P, self.C])  # Composante obligataire
        self.E = self.S  # Composante action
        self.payoff = Stack([self.A, self.P, self.C, self.S])  # Payoff total

# Instanciation de la configuration
config = ConvertibleBondConfig()
