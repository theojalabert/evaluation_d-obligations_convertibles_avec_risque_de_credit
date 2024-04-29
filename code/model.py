###########################################
###      PROJET - RISQUE DE CRÉDIT      ###
### ALLERS - JALABERT - SIMONEAU-FRIGGI ###
###           file = model.py           ###
###########################################

"""
Framework pour la modélisation des processus de payoff à l'aide d'un modèle binomial ou d'un modèle à différences finies.
"""
import abc
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import warnings
np.seterr(divide="ignore")

__all__ = [
    "WienerJumpProcess",
    "BinomialModel", "FDEModel",
    "CrankNicolsonScheme", "ExplicitScheme", "ImplicitScheme",
    "RannacherScheme", "PenaltyScheme", "PenaltyRannacherScheme",
]


class WienerJumpProcess(object):
    """
    Un modèle de mouvement stochastique des actions utilisant un processus de dérive de Wiener
    pour modéliser la volatilité et un processus de saut de Poisson pour modéliser le défaut.

    Le prix de l'action suit :
        dS = (r + lambda * eta) * S_t * dt + sigma * S_t * dW_t - eta * S_t * dq_t
    où :
        dS      est le mouvement instantané du sous-jacent
        dt      est le changement instantané dans le temps
        dW_t    est le processus Wiener drift
        dq_t    est le processus de saut de Poisson avec une intensite lambda

        r       est le taux sans-risque
        lambda  est le taux d'aléa
        eta     est la chute du prix de l'action lors d'un événement de défaillance
        sigma   est la log-volatilité du prix de l'action
    """

    def __init__(self, r, sigma, lambd_=0, eta=1, cap_lambda=False):
        if sigma <= 0:
            raise ValueError("Volatility must be positive")
        self.r = np.double(r)
        self.sigma = np.double(sigma)
        self.lambd_ = lambd_ if callable(lambd_) else np.double(lambd_)
        self.eta = np.double(eta)
        self.cap_lambda = cap_lambda

    def binomial(self, dt, S=None):
        """Paramètres pour le modèle binomial."""
        if dt <= 0:
            raise ValueError("Time step must be positive")
        if self.sigma ** 2 < self.r ** 2 * dt:
            raise ValueError("Time step too big for given volatility")
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        l = 1 - self.eta

        if not callable(self.lambd_):
            lambd_ = self.lambd_
        elif S is not None:
            lambd_ = self.lambd_(S)
            if (lambd_ < 0).any():
                raise ValueError("Hazard rate must be non-negative")
        else:
            return (u, d, l, False)

        # Probability of up/down/loss
        lambda_limit = (np.log(u - l) - np.log(np.exp(self.r * dt) - l))
        if self.cap_lambda:
            lambd_ = np.minimum(lambd_, lambda_limit / dt)
        elif (lambd_ * dt > lambda_limit).any():
            raise ValueError("Time step too big for given hazard rate")
        po = 1 - np.exp(-lambd_ * dt)
        pu = (np.exp(self.r * dt) - d * (1 - po) - l * po) / (u - d)
        pd = 1 - pu - po
        if S is not None:
            return (pu, pd, po)
        else:
            return (u, d, l, (pu, pd, po))

    def fde_l(self):
        """
        Paramètre pour le saut de stock en cas de défaut.
        Renvoie la fraction de perte en cas de défaillance.
        """
        return 1 - self.eta

    def fde(self, dt, ds, S, scheme, boundary="diffequal", expfit=False):
        """
        Paramètres du schéma de différences finies pour résoudre l'EDP.

        Args :
        dt (float) : Taille du pas de temps.
        ds (float) : Taille du pas de stock.
        S (tableau) : Prix des actions.
        scheme (callable) : Schéma numérique utilisé pour la différence finie.
        boundary (str) : Type de condition limite appliquée.
        expfit (bool) : Utilise l'ajustement exponentiel pour mieux gérer les termes d'advection.

        Renvoie :
        tuple : Tableaux représentant les coefficients de la matrice tridiagonale et la probabilité par défaut.
        """
        if dt <= 0:
            raise ValueError("Time step must be positive")
        if ds <= 0:
            raise ValueError("Stock step must be positive")
        if (S < 0).any():
            raise ValueError("Stock must be non-negative")

        if callable(self.lambd_):
            lambd_ = self.lambd_(S)
        else:
            lambd_ = self.lambd_

        rdt = (self.r + lambd_) * dt
        rS = dt * (self.r + lambd_ * self.eta) * S / ds / 2

        if expfit:
            x = (self.r + lambd_ * self.eta) * ds / self.sigma ** 2 / S
            coth = 1. / np.tanh(x)
            sS = dt * coth * (self.r + lambd_ * self.eta) * S / ds / 2
        else:
            sS = dt * self.sigma ** 2 * S ** 2 / ds ** 2 / 2

        a = sS[1:] - rS[1:]
        b = -rdt - sS * 2
        c = sS[:-1] + rS[:-1]
        d = lambd_ * dt

        if boundary == "equal":
            b[0] += sS[0] - rS[0]
            b[-1] += sS[-1] + rS[-1]
        elif boundary == "diffequal":
            b[0] += 2 * (sS[0] - rS[0])
            c[0] -= sS[0] - rS[0]
            b[-1] += 2 * (sS[-1] + rS[-1])
            a[-1] -= sS[-1] + rS[-1]
        elif boundary != "ignore":
            raise ValueError("unknown boundary type: %s" % boundary)

        return (np.append(a, sS[0] - rS[0]), b, np.append(sS[-1] + rS[-1], c), d)

class Value(object):
    """
    Valorisation du portfolio.
    """
    def __init__(self, T, N):
        self.N = N
        self.t = np.linspace(0, T, N + 1)
        self.S = [None] * (N + 1)  # Stock prices at each node
        self.C = np.zeros(N + 1)   # Coupons at each node
        self.X = [None] * N        # Default values at each node
        self.V = [None] * (N + 1)  # Portfolio values at each node
        self.I = [None] * (N + 1)  # Implicit values of the portfolio

class BinomialModel(object):
    """
    Un modèle de treillis binomial pour l'évaluation des produits dérivés à l'aide d'un processus
    boursier et des valeurs intrinsèques de l'action.
    """
    class Value(Value):
        """
        Valorisation du portefeuille pour le modèle binomial.
        """
        def __float__(self):
            return self.V[0][0]

    def __init__(self, N, dS, V):
        self.N = int(N)
        self.dt = V.T / N  # Time increments for the binomial lattice
        self.dS = dS
        self.V = V  # Intrinsic value of the derivative

    def price(self, S0):
        """
        Déterminer le payoff pour un prix de l'action S0 au temps 0 à l'aide d'un modèle binomial.
        """
        So = np.double(S0)
        u, d, l, prob = self.dS.binomial(self.dt)
        erdt = np.exp(-self.dS.r * self.dt)
        P = BinomialModel.Value(self.V.T, self.N)
        if prob:
            pu, pd, po = prob

        # Calculer les cours terminaux des actions et les valeurs des produits dérivés
        P.S[-1] = S = np.array([S0 * u**(self.N - i) * d**i for i in range(self.N + 1)])
        if not prob:
            pu, pd, po = self.dS.binomial(self.dt, P.S[-1])
        P.C[-1] = C = self.V.coupon(self.V.T)
        P.V[-1] = V = self.V.terminal(S) + C
        P.I[-1] = I = V

        # Prix d'actualisation à rebours dans l'arbre binomial
        t = P.t
        for i in range(self.N - 1, -1, -1):
            P.S[i] = S = S[:-1] / u
            P.X[i] = X = self.V.default(t[i], S * l)
            P.C[i] = C = self.V.coupon(t[i])
            if not prob:
                pu, pd, po = self.dS.binomial(self.dt, S)
            V = erdt * (V[:-1] * pu + V[1:] * pd + X * po)
            P.I[i] = I = V
            P.V[i] = V = self.V.transient(t[i], V, S) + C

        return P

class Scheme(object):
    """
    Classe de base abstraite pour les schémas de différences utilisés dans les méthodes de différences finies.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, S):
        self.S = S

    def __call__(self, t, V, X, C, payoff):
        I = self.scheme(V, X)
        return payoff(t, I, self.S) + C, I

    @abc.abstractmethod
    def scheme(self, V, X):
        """Mise en œuvre de l'actualisation de la valeur du portefeuille sur une période."""
        pass

class ExplicitScheme(Scheme):
    """
    Explicit finite difference scheme.
    """
    def __init__(self, dS, dt, ds, S, **kwargs):
        super(ExplicitScheme, self).__init__(S)
        a, b, c, d = dS.fde(dt, ds, S, "explicit", **kwargs)
        self.L = sparse.dia_matrix(([a, 1 + b, c], [-1, 0, 1]), shape=(S.size, S.size))
        self.d = d

    def scheme(self, V, X):
        return self.L.dot(V) + self.d * X

class ImplicitScheme(Scheme):
    """
    Implicit finite difference scheme.
    """
    def __init__(self, dS, dt, ds, S, **kwargs):
        super(ImplicitScheme, self).__init__(S)
        a, b, c, d = dS.fde(dt, ds, S, "implicit", **kwargs)
        self.L = sparse.dia_matrix(([-a, 1 - b, -c], [-1, 0, 1]), shape=(S.size, S.size)).tocsr()
        self.d = d

    def scheme(self, V, X):
        return linalg.spsolve(self.L, V + self.d * X)

class CrankNicolsonScheme(Scheme):
    """
    Crank-Nicolson finite difference scheme.
    """
    def __init__(self, dS, dt, ds, S, **kwargs):
        super(CrankNicolsonScheme, self).__init__(S)
        a, b, c, d = dS.fde(dt, ds, S, "explicit", **kwargs)
        self.Le = sparse.dia_matrix(([a, 2 + b, c], [-1, 0, 1]), shape=(S.size, S.size))
        a, b, c, d = dS.fde(dt, ds, S, "implicit", **kwargs)
        self.Li = sparse.dia_matrix(([-a, 2 - b, -c], [-1, 0, 1]), shape=(S.size, S.size)).tocsr()
        self.d = 2 * d

    def scheme(self, V, X):
        return linalg.spsolve(self.Li, self.Le.dot(V) + self.d * X)

class RannacherScheme(CrankNicolsonScheme):
    """
    Rannacher time stepping: Crank-Nicolson scheme with fully implicit initial steps.
    """
    def __init__(self, dS, dt, ds, S, **kwargs):
        super(RannacherScheme, self).__init__(dS, dt, ds, S, **kwargs)
        a, b, c, d = dS.fde(dt / 4, ds, S, "implicit", **kwargs)
        self.Lq = sparse.dia_matrix(([-a, 1 - b, -c], [-1, 0, 1]), shape=(S.size, S.size)).tocsr()
        self.dq = d
        self.imp = 1  # Number of initial implicit steps

    def __call__(self, t, V, X, C, payoff):
        if self.imp > 0:
            I = self.scheme_implicit(V, X)
            self.imp -= 1
        else:
            I = self.scheme(V, X)
        if C != 0:
            self.imp += 1  # Re-trigger implicit steps after payoff events
        return payoff(t, I, self.S) + C, I

    def scheme_implicit(self, V, X):
        for i in range(4):  # Four quarter steps
            V = linalg.spsolve(self.Lq, V + self.dq * X)
        return V

class PenaltyScheme(CrankNicolsonScheme):
    """
    Étend le schéma de Crank-Nicolson en utilisant des itérations de pénalité pour renforcer
    les contraintes américaines.
    """
    def __init__(self, dS, dt, ds, S, tol=None, **kwargs):
        super(PenaltyScheme, self).__init__(dS, dt, ds, S, **kwargs)
        self.tol = np.double(tol if tol is not None else min(dt, ds)**2)
        assert self.tol > 0, "Tolerance must be positive"

    def __call__(self, t, V, X, C, payoff):
        Vx = self.Le.dot(V) + self.d * X
        I = Vk = linalg.spsolve(self.Li, Vx)
        Vs = payoff(t, Vk, self.S)
        Pk = (Vs != Vk) / self.tol

        if not np.any(Pk):
            return Vk + C, I

        for _ in range(32):
            Vk1 = linalg.spsolve(self.Li + sparse.diags([Pk], [0], shape=Vk.shape), Vx + Pk * Vs)
            if np.max(np.abs(Vk1 - Vk) / np.maximum(1, np.abs(Vk1))) < self.tol:
                break
            Vk, Pk = Vk1, (Vs != Vk1) / self.tol

        return Vk1 + C, I

class PenaltyRannacherScheme(PenaltyScheme, RannacherScheme):
    """
    Combine le schéma de pénalité et le schéma de Rannacher pour améliorer la stabilité et
    la précision dans l'évaluation des options américaines avec des payoffs discontinus.
    """
    def __call__(self, t, V, X, C, payoff):
        if self.imp > 0:
            I = self.scheme_implicit(V, X)
            V = payoff(t, I, self.S) + C
            self.imp -= 1
        else:
            V, I = super(PenaltyScheme, self).__call__(t, V, X, C, payoff)

        if C != 0:
            self.imp = 1  # Reinitialize implicit steps after payouts
        return V, I

class FDEModel(object):
    """
    Schéma d'équation aux différences finies pour l'évaluation des produits dérivés à l'aide
    d'un processus d'action et des valeurs intrinsèques de l'action.
    """
    class Value(Value):
        """
        Valorisation d'un portefeuille spécifique pour FDEModel.
        """
        def __init__(self, T, N, Sl, Su, K):
            super(FDEModel.Value, self).__init__(T, N)
            assert Su > Sl >= 0, "Invalid stock range"
            self.K = K
            self.S = np.linspace(Sl, Su, K + 1)

    def __init__(self, N, dS, V):
        self.N = int(N)
        self.dt = V.T / N
        self.dS = dS
        self.V = V

    def price(self, Sl, Su, K, scheme=PenaltyRannacherScheme, **kwargs):
        """
        Prix le gain pour les prix dans l'intervalle [Sl, Su] avec K incréments.
        """
        Sl, Su = map(np.double, (Sl, Su))
        K = int(K)
        P = FDEModel.Value(self.V.T, self.N, Sl, Su, K)
        S = P.S
        ds = S[1] - S[0]
        Sl = S * self.dS.fde_l()

        P.C[-1] = C = self.V.coupon(self.V.T)
        P.V[-1] = V = self.V.terminal(S) + C
        P.I[-1] = I = V

        scheme = scheme(self.dS, self.dt, ds, S, **kwargs)
        for i in range(self.N - 1, -1, -1):
            P.C[i] = C = self.V.coupon(P.t[i])
            P.X[i] = X = self.V.default(P.t[i], Sl)
            V, I = scheme(P.t[i], V, X, C, self.V.transient)
            P.V[i] = V
            P.I[i] = I

        return P


class FDEBVModel(FDEModel):
    """
    Étend le modèle FDEModel pour traiter les dérivés de prix en divisant la valeur intrinsèque
    en composantes d'obligations et d'actions, en utilisant un schéma d'équation de différence finie.
    """

    def __init__(self, N, dS, B, V):
        super(FDEBVModel, self).__init__(N, dS, V)
        self.B = B  # Bond component of the derivative

    def price(self, Sl, Su, K, scheme=CrankNicolsonScheme, **kwargs):
        """
        Fixe le prix du dérivé avec les composantes obligations et actions dans la fourchette de prix de l'action donnée.

        Args :
        Sl (float) : Limite inférieure du cours de l'action.
        Su (float) : Limite supérieure du cours de l'action.
        K (int) : Nombre d'incréments de prix.
        scheme (Schéma) : Schéma de différence finie à utiliser.

        Renvoie :
        FDEModel.Value : Valorisation du dérivé à chaque pas de temps.
        """
        Sl, Su = np.double(Sl), np.double(Su)
        K = int(K)
        P = FDEModel.Value(self.V.T, self.N, Sl, Su, K)
        S = P.S
        ds = S[1] - S[0]
        Sl = S * self.dS.fde_l()

        # Calculer le prix final de l'action et les valeurs des produits dérivés
        P.C[-1] = C = self.V.coupon(self.V.T)
        P.V[-1] = V = self.V.terminal(S) + C
        B = self.B.terminal(S) + C
        E = V - B  # Composant Equity

        # Initialisation de l'instance de schéma
        scheme_instance = scheme(self.dS, self.dt, ds, S, **kwargs)

        # Prix d'actualisation à l'envers en utilisant le système spécifié
        for i in range(self.N - 1, -1, -1):
            P.C[i] = C = self.V.coupon(P.t[i])
            P.X[i] = X = self.V.default(P.t[i], Sl)
            XB = self.B.default(P.t[i], Sl)
            XE = X - XB
            B = scheme_instance.scheme(B, XB)  # Application du schéma au composant bond
            E = scheme_instance.scheme(E, XE)  # Application du schéma au composant Equity
            # Traiter les étapes transitoires et de correction pour les obligations et les actions
            B_ = self.B.transient(P.t[i], B, S)
            B = np.minimum(B_, np.maximum(B_ - E, B))
            E = self.V.transient(P.t[i], B + E, S) - B
            B += C
            P.V[i] = V = B + E  # Combiner les obligations et les actions pour obtenir la valeur totale

        return P
