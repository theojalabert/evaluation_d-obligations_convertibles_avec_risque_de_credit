###########################################
###      PROJET - RISQUE DE CRÉDIT      ###
### ALLERS - JALABERT - SIMONEAU-FRIGGI ###
###           file = payoff.py          ###
###########################################

"""
Diverses fonctions de payoff pour les produits financiers dérivés.
"""

import numpy as np
import contextlib

__all__ = [
    "Payoff",
    "CallA", "CallE", "CallVR", "Forward", "PutA", "PutE", "PutV",
    "Stack", "Time", "UpAndOut",
    "Annuity", "AnnuityI",
    "VariableStrike",
]

class Payoff(object):
    """
    Classe de Base pour les derivative payoffs, handling terminal, transient, et default payoffs.
    """
    def __init__(self, T):
        self.T = T

    def __contains__(self, t):
        """Vérifier si l'instant t est compris dans le cycle de vie du produit dérivé."""
        return 0 <= t <= self.T

    def default(self, t, S):
        """Définir le paiement en cas de défaut au temps t."""
        assert t != self.T, "Default calculation is not valid at terminal time."
        return np.zeros(S.shape)

    def terminal(self, S):
        """Définir le payoff terminal."""
        return np.zeros(S.shape)

    def transient(self, t, V, S):
        """Définir le payoff transitoire pendant la durée de vie du produit dérivé."""
        assert t != self.T, "Transient calculation is not valid at terminal time."
        return V

    def coupon(self, t):
        """Définir les éventuels paiements de coupons au temps t."""
        return 0

class Forward(Payoff):
    """
    Un contrat à terme dont le paiement est linéaire en fonction du prix de l'action moins le prix d'exercice à l'échéance.
    """
    def __init__(self, T, K):
        super(Forward, self).__init__(T)
        self.K = np.double(K)

    def terminal(self, S):
        """Calculer le payoff terminal à S - K."""
        return S - self.K

class CallE(Payoff):
    """
    Le payoff d'une option d'achat Européenne, qui ne dépend que de la condition finale.
    """
    def __init__(self, T, K):
        super(CallE, self).__init__(T)
        self.K = np.double(K)

    def terminal(self, S):
        """Calculer le payoff terminal à max(S - K, 0)."""
        return np.maximum(S - self.K, 0)

class CallA(CallE):
    """
    Un CALL Américain, qui permet un exercice anticipé.
    """
    def __init__(self, T, K):
        super(CallA, self).__init__(T, K)

    def default(self, t, S):
        """Calculer la valeur résiduelle de l'option en cas de défaut."""
        assert t != self.T, "Default calculation is not valid at terminal time."
        return np.maximum(S - self.K, 0)

    def transient(self, t, V, S):
        """Calculer le payoff transitoire comme max(S - K, V), ce qui permet un exercice anticipé."""
        assert t != self.T, "Transient calculation is not valid at terminal time."
        return np.maximum(V, S - self.K)
class PutE(Payoff):
    """
    Le payoff d'un PUT Européen, avec un prix d'exercice K.
    """
    def __init__(self, T, K):
        super(PutE, self).__init__(T)
        self.K = np.double(K)

    def terminal(self, S):
        """Calculer le payoff terminal à max(K - S, 0)."""
        return np.maximum(self.K - S, 0)

class PutA(PutE):
    """
    Un PUT Américain, permettant l'exercice avant et à l'échéance.
    """
    def __init__(self, T, K):
        super(PutA, self).__init__(T, K)

    def default(self, t, S):
        """Calculer la valeur résiduelle en cas de défaut."""
        return np.maximum(self.K - S, 0)

    def transient(self, t, V, S):
        """Calculer le payoff transitoire comme max(K - S, V)."""
        return np.maximum(V, self.K - S)

class PutV(Payoff):
    """
    Un PUT Américain basé sur la valeur du portefeuille, avec un prix d'exercice K.
    """
    def __init__(self, T, K):
        super(PutV, self).__init__(T)
        self.K = np.double(K)

    def transient(self, t, V, _S):
        """Calculer le payoff transitoire comme V + max(K - V, 0)."""
        return np.maximum(V, self.K)

class CallVR(Payoff):
    """
    Une option reverse American Call option basée sur la valeur du portefeuille, avec un strike K.
    """
    def __init__(self, T, K):
        super(CallVR, self).__init__(T)
        self.K = np.double(K)

    def transient(self, t, V, _S):
        """Calculer le payoff transitoire comme V - max(V - K, 0)."""
        return np.minimum(V, self.K)

class Stack(Payoff):
    """
    Un payoff composite composé de plusieurs payoffs empilés.
    """
    def __init__(self, stack):
        super(Stack, self).__init__(max(payoff.T for payoff in stack))
        self.stack = tuple(stack)

    def default(self, t, S):
        """Calculer la valeur maximale par défaut parmi les payoffs empilés."""
        return np.max([payoff.default(t, S) for payoff in self.stack])

    def transient(self, t, V, S):
        """Agrégation des payoffs transitoires des composants empilés."""
        for payoff in self.stack:
            V = payoff.transient(t, V, S)
        return V

    def terminal(self, S):
        """Calculer la valeur terminale maximale parmi les payoffs empilés."""
        return np.max([payoff.terminal(S) for payoff in self.stack])

    def coupon(self, t):
        """Somme des coupons de tous les composants."""
        return sum(payoff.coupon(t) for payoff in self.stack)

class Time(Payoff):
    """
    Un dérivé avec activation et désactivation en fonction du temps.
    """
    def __init__(self, payoff, times):
        super(Time, self).__init__(payoff.T)
        self.payoff = payoff
        self.set_times(times)

    def set_times(self, times):
        """Définir des périodes actives pour le payoff, en distinguant les intervalles discrets et continus."""
        self._time_discrete = set()
        self._time_continuous = []
        for time in times:
            if isinstance(time, (tuple, list)) and len(time) == 2:
                self._time_continuous.append((max(np.double(time[0]), 0), min(np.double(time[1]), self.T)))
            else:
                self._time_discrete.add(np.double(time))

    def __contains__(self, t):
        return t in self._time_discrete or any(l <= t <= u for l, u in self._time_continuous)

    def default(self, t, S):
        if t in self:
            return self.payoff.default(t, S)
        return np.zeros_like(S)

    def transient(self, t, V, S):
        if t in self:
            return self.payoff.transient(t, V, S)
        return V

    def terminal(self, S):
        if self.T in self:
            return self.payoff.terminal(S)
        return 0

    def coupon(self, t):
        if t in self:
            return self.payoff.coupon(t)
        return 0

class UpAndOut(Payoff):
    """
    Une option à barrière avec une caractéristique "up-and-out" où l'option devient sans valeur si le prix de l'actif
    sous-jacent dépasse un certain niveau, L, à n'importe quel moment avant l'expiration.
    """
    def __init__(self, payoff, L):
        super(UpAndOut, self).__init__(payoff.T)
        self.payoff = payoff
        self.L = np.double(L)

    def default(self, t, S):
        """Calculer le payoff si l'actif n'a pas atteint ou dépassé la barrière."""
        assert t != self.T, "Default value should not be calculated at terminal time."
        V = np.zeros_like(S)
        idx = S < self.L
        V[idx] = self.payoff.default(t, S[idx])
        return V

    def transient(self, t, V, S):
        """Ils ne sont rémunérés que si le prix de l'actif est inférieur à la barrière."""
        Vp = np.zeros_like(S)
        idx = S < self.L
        Vp[idx] = self.payoff.transient(t, V[idx], S[idx])
        return Vp

    def terminal(self, S):
        """Déterminer le payoff final en tenant compte de la barrière."""
        V = np.zeros_like(S)
        idx = S < self.L
        V[idx] = self.payoff.terminal(S[idx])
        return V

    def coupon(self, t):
        """Les coupons sont payés si l'option n'est pas éliminée."""
        if t in self:
            return self.payoff.coupon(t)
        return 0

class Annuity(Payoff):
    """
    Une rente qui verse un coupon fixe à des moments déterminés et restitue la valeur nominale à l'échéance.
    """
    def __init__(self, T, times=(), C=0, N=0, R=0):
        super(Annuity, self).__init__(T)
        self.times = set(times)
        self.C = np.double(C)
        self.N = np.double(N)
        self.R = np.double(R)

    def default(self, t, S):
        """Définir la valeur résiduelle de la rente en cas de défaut."""
        assert t != self.T, "Default calculation should not occur at terminal time."
        return np.full_like(S, self.N * self.R)

    def terminal(self, S):
        """Calculer la valeur nominale payée à l'échéance."""
        return np.full_like(S, self.N)

    def coupon(self, t):
        """Calculer les paiements de coupons à des moments précis."""
        return self.C if t in self.times else 0.0

    def __contains__(self, t):
        """Vérifier si le temps actuel est une temps de paiement de coupon valide."""
        return t in self.times

class AnnuityI(Payoff):
    """
    Implémente un processus de payoff par annuités intrinsèques qui verse un coupon à des moments précis et un montant
    nominal à la fin, où les paiements de coupon sont intrinsèques à la valeur du portefeuille et ne sont pas garantis
    en raison d'éventuels écrasements par d'autres payoffs.
    """
    def __init__(self, T, times=(), C=0, N=0, R=0):
        super(AnnuityI, self).__init__(T)
        self.times = set(times)
        self.C = np.double(C)
        self.N = np.double(N)
        self.R = np.double(R)

    def default(self, t, S):
        """Calcule la valeur résiduelle en cas de défaut."""
        assert t != self.T, "Default calculation should not occur at terminal time."
        return np.full_like(S, self.N * self.R)

    def transient(self, t, V, S):
        """Ajoute le paiement du coupon à la valeur de payoff, le cas échéant."""
        if t in self.times:
            return V + self.C
        return V

    def terminal(self, S):
        """Calcule la valeur nominale de l'obligation à l'échéance."""
        V = np.full_like(S, self.N)
        if self.T in self.times:
            V += self.C
        return V

class VariableStrike(Payoff):
    """
    Classe Mixin permettant aux gains d'avoir un prix d'exercice variable qui peut changer dans le temps.
    """
    def __init__(self, T, K):
        super(VariableStrike, self).__init__(T)
        self.K = np.double(K)

    @contextlib.contextmanager
    def _strike(self, K):
        """
        Gestionnaire de contexte pour modifier temporairement le prix d'exercice pour la durée d'un bloc.
        """
        original_K = self.K
        try:
            self.K = K
            yield
        finally:
            self.K = original_K

    def strike(self, t):
        """Calculer le prix d'exercice dépendant du temps."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def transient(self, t, V, S):
        """Calculer le payoff transitoire en utilisant le prix d'exercice dépendant du temps."""
        with self._strike(self.strike(t)):
            return super(VariableStrike, self).transient(t, V, S)
