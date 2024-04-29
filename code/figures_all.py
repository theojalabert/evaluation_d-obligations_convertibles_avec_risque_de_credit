###########################################
###      PROJET - RISQUE DE CRÉDIT      ###
### ALLERS - JALABERT - SIMONEAU-FRIGGI ###
###        file = figures_all.py        ###
###########################################

### Voici les codes concernant toutes les figures utilisées dans ce projet, les codes sont à exécuter 1 à 1 ###

"""
Graphique comparatif entre AFV03 figure 2 et ce modèle.
"""
import matplotlib.pyplot as plt
import numpy as np

from model import FDEModel
from plot import plotmain
from convertible_bond import config  # Importation de la configuration

def main():
    S = np.arange(80, 121)
    Sl = 0
    Su = 200
    N = 128 * config.T  # Utilisation de la variable de temps de maturité de la configuration
    K = 8 * (Su - Sl)
    Sk = K * (S - Sl) / (Su - Sl)
    model1 = FDEModel(N, config.dS, config.payoff)
    model2 = FDEModel(N, config.dS_partial, config.payoff)
    model3 = FDEModel(N, config.dS_total, config.payoff)
    plt.plot(S, model1.price(Sl, Su, K).V[0][Sk])
    plt.plot(S, model2.price(Sl, Su, K).V[0][Sk])
    plt.plot(S, model3.price(Sl, Su, K).V[0][Sk])
    plt.ylim(100, 150)
    plt.xlabel("Stock Price")
    plt.ylabel("Convertible Bond Price")
    plt.legend(["No default", "Partial default", "Total Default"], loc=2)

if __name__ == "__main__":
    plotmain(main)

###########################################

"""
Graphique comparatif entre AFV03 figure 4 et ce modèle.
"""
import copy
import matplotlib.pyplot as plt
import numpy as np

from model import FDEModel
from plot import plotmain
from convertible_bond import config  # Importation de la configuration fusionnée

def main():
    S = np.arange(80, 121)
    Sl = 0
    Su = 200
    N = 128 * config.T  # Utilisation de la variable de temps de maturité de la configuration
    K = 8 * (Su - Sl)
    Sk = K * (S - Sl) / (Su - Sl)

    # Création de copies pour manipuler le taux de récupération sans affecter l'objet original
    annuity_default = copy.deepcopy(config.A)
    annuity_no_default = copy.deepcopy(config.A)

    # Modèle avec défaut total
    annuity_default.R = 1.0  # Mise à jour du taux de récupération pour le défaut total
    model_default_100 = FDEModel(N, config.dS_total, Stack([annuity_default, config.P, config.C, config.S]))

    # Modèle sans défaut
    model_no_default = FDEModel(N, config.dS, config.payoff)

    # Affichage des prix
    plt.plot(S, model_default_100.price(Sl, Su, K).V[0][Sk], label="Total default (R=100%)")
    plt.plot(S, model_no_default.price(Sl, Su, K).V[0][Sk], label="No default")

    # Modifications successives de R pour les autres cas de défaut total
    for R in [0.5, 0.0]:
        annuity_default.R = R
        model_default = FDEModel(N, config.dS_total, Stack([annuity_default, config.P, config.C, config.S]))
        plt.plot(S, model_default.price(Sl, Su, K).V[0][Sk], label=f"Total default (R={int(R*100)}%)")

    plt.ylim(100, 150)
    plt.xlabel("Stock Price")
    plt.ylabel("Convertible Bond Price")
    plt.legend(loc=2)
    plt.show()

if __name__ == "__main__":
    plotmain(main)

###########################################

"""
Graphique comparatif entre la figure 5 d'AFV03 et ce modèle.
"""
import copy
import matplotlib.pyplot as plt
import numpy as np

from model import FDEModel
from plot import plotmain
from convertible_bond import config  # Importation de la configuration fusionnée

def main():
    S = np.arange(24, 121)
    Sl = 0
    Su = 200
    N = 128 * config.T  # Utilisation de la variable de temps de maturité de la configuration
    K = 8 * (Su - Sl)
    Sk = K * (S - Sl) / (Su - Sl)

    # Copie des configurations de processus pour des taux de défaillance variables
    dS1 = copy.deepcopy(config.dS_total)  # Utilise dS_total car il implique un défaut total
    dS1.lambd_ = lambda S: 0.02 * (S / 100)**-1.2
    dS1.cap_lambda = True

    dS2 = copy.deepcopy(config.dS_total)
    dS2.lambd_ = lambda S: 0.02 * (S / 100)**-2.0
    dS2.cap_lambda = True

    # Modèles FDE avec différents taux de défaillance
    model1 = FDEModel(N, config.dS_total, config.payoff)
    model2 = FDEModel(N, dS1, config.payoff)
    model3 = FDEModel(N, dS2, config.payoff)

    plt.plot(S, model1.price(Sl, Su, K).V[0][Sk])
    plt.plot(S, model2.price(Sl + 1, Su, K - 8).V[0][Sk - 8])
    plt.plot(S, model3.price(Sl + 1, Su, K - 8).V[0][Sk - 8])

    plt.xlim(S[0], S[-1])
    plt.ylim(50, 150)
    plt.xlabel("Stock Price")
    plt.ylabel("Convertible Bond Price")
    plt.legend(["Constant hazard rate", "$\\alpha = -1.2$", "$\\alpha = -2.0$"], loc=2)
    plt.show()

if __name__ == "__main__":
    plotmain(main)

###########################################

"""
Graphique comparatif entre la figure 6 d'AFV03 et ce modèle.
"""
import matplotlib.pyplot as plt
import numpy as np

from model import FDEModel
from plot import plotmain
from convertible_bond import config  # Importation de la configuration fusionnée

def delta(S, model):
    Sl = 1
    Su = 200
    K = 8
    S = S - 1
    V = model.price(Sl, Su, 199 * K).V[0]
    return K * (V[S * K + 1] - V[S * K - 1]) / 2

def main():
    N = 128 * config.T  # Utilisation de la variable de temps de maturité de la configuration
    S = np.arange(24, 121)

    # Création de modèles FDE avec différents taux de défaillance
    model1 = FDEModel(N, config.dS_total, config.payoff)  # Utilisation de dS_total pour le taux de défaillance constant
    model2 = FDEModel(N, config.dS_var12, config.payoff)  # Utilisation de dS_var12 pour le taux de défaillance variable -1.2
    model3 = FDEModel(N, config.dS_var20, config.payoff)  # Utilisation de dS_var20 pour le taux de défaillance variable -2.0

    plt.plot(S, delta(S, model3))
    plt.plot(S, delta(S, model2))
    plt.plot(S, delta(S, model1))
    plt.xlim(S[0], S[-1])
    plt.ylim(-1, 1)
    plt.xlabel("Stock Price")
    plt.ylabel("Delta of Convertible Bond")
    plt.legend(["$\\alpha = -2.0$", "$\\alpha = -1.2$", "Constant hazard rate"], loc=4)
    plt.show()

if __name__ == "__main__":
    plotmain(main)

###########################################

"""
Graphique comparatif entre AFV03 figure 7 et ce modèle.
"""
import matplotlib.pyplot as plt
import numpy as np

from model import FDEModel
from plot import plotmain
from convertible_bond import config  # Importation de la configuration fusionnée

def gamma(S, model):
    Sl = 1
    Su = 200
    K = 8
    S = S - 1
    V = model.price(Sl, Su, 199 * K).V[0]
    return K * K * (V[S * K + 1] + V[S * K - 1] - 2 * V[S * K])

def main():
    N = 128 * config.T  # Utilisation de la variable de temps de maturité de la configuration
    S = np.arange(24, 121)

    # Création de modèles FDE avec différents taux de défaillance
    model1 = FDEModel(N, config.dS_total, config.payoff)  # Utilisation de dS_total pour le taux de défaillance constant
    model2 = FDEModel(N, config.dS_var12, config.payoff)  # Utilisation de dS_var12 pour le taux de défaillance variable -1.2
    model3 = FDEModel(N, config.dS_var20, config.payoff)  # Utilisation de dS_var20 pour le taux de défaillance variable -2.0

    plt.plot(S, gamma(S, model1))
    plt.plot(S, gamma(S, model2))
    plt.plot(S, gamma(S, model3))
    plt.xlim(S[0], S[-1])
    plt.ylim(-0.1, 0.1)
    plt.xlabel("Stock Price")
    plt.ylabel("Gamma of Convertible Bond")
    plt.legend(["Constant hazard rate", "$\\alpha = -1.2$", "$\\alpha = -2.0$"], loc=2)
    plt.show()

if __name__ == "__main__":
    plotmain(main)

###########################################

"""
Plot the traded surface using the binomial surface,
"""
import matplotlib.pyplot as plt
import numpy as np

from model import BinomialModel
from plot import plot_model, payoff as payoff_, HOLD, plotmain
from convertible_bond import config  # Utilisation de la configuration fusionnée

def plot_H(ax, Sl, Su, model, P):
    po = model.dS.binomial(model.dt)[3][2]

    y = slice(0, 1)
    for x, t in zip(range(len(P.t) - 1), P.t[:-1]):
        S = P.S[x][y]
        V = P.V[x][y]
        y1 = slice(y.start, y.stop + 1)
        S1 = P.S[x + 1][y1]
        V1 = P.V[x + 1][y1]
        I1 = P.I[x + 1][y1]
        t1 = P.t[x + 1]

        delta = np.diff(V1) / np.diff(S1)
        pi = (V - delta * S - (config.A.C if t in config.A.payment_times else 0)) * np.exp(model.dt * model.dS.r)

        Vu = pi + delta * S1[:-1]
        Vd = pi + delta * S1[1:]
        Xo  = pi + (1 - model.dS.eta) * delta * S

        H = -(Xo - P.X[x][y]) / (1 - po)
        np.testing.assert_array_almost_equal(H, (Vu - V1[:-1]) / po)
        np.testing.assert_array_almost_equal(H, (Vd - V1[1:]) / po)

        if x:
            flt_ = (Sl <= S_) & (S_ <= Su)
            flt = (Sl <= S) & (S <= Su)
            ax.plot_trisurf(np.append(t_[flt_], (t * np.ones(S.shape))[flt]),
                            np.append(S_[flt_], S[flt]),
                            np.append(H_[flt_], H[flt]), linewidth=0)

        start = 0
        stop = y1.stop - y1.start - 1
        while start <= stop and payoff_(t1, S1[start], V1[start], I1[start]) != HOLD:
            start += 1
        while start <= stop and payoff_(t1, S1[stop], V1[stop], I1[stop]) != HOLD:
            stop -= 1
        y = slice(y1.start + start, y1.start + stop + 1)
        t_ , H_, S_ = t * np.ones(S.shape), H, S

def main():
    N = 80
    S0 = 100
    Sl = 0
    Su = 200
    model = BinomialModel(N, config.dS_typical, config.payoff)  # Utilisation de dS_typical depuis config
    P = model.price(S0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_H(ax, Sl, Su, model, P)
    ax.set_ylim(Sl, Su)
    ax.set_xlabel("Time")
    ax.set_ylabel("Stock Price")
    ax.set_zlabel("Default Cost after Hedging ($H^{c}_t$)")

if __name__ == "__main__":
    plotmain(main)

###########################################

"""
Graphique montrant la surface d'évaluation de l'obligation convertible de l'exemple.
"""
import matplotlib.pyplot as plt

from model import plot_model  # Assurez-vous que cette importation est correcte selon votre structure de module
from plot import plotmain
from convertible_bond import config  # Utilisation de la configuration fusionnée

def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_model(ax, config.dS_total, config.payoff)  # Utilisation des paramètres depuis config

if __name__ == "__main__":
    plotmain(main)

###########################################

"""
Représenter graphiquement la convergence des différents modèles d'évaluation des obligations convertibles.
"""
import itertools
import matplotlib.pyplot as plt
import numpy as np
import time

from model import BinomialModel, FDEModel, ImplicitScheme, CrankNicolsonScheme, PenaltyScheme
from plot import plotmain
from convertible_bond import config  # Utilisation de la configuration fusionnée

def label(plt, xlabel):
    rng = np.arange(122.7, 123.5, 0.1)
    plt.set_ylim(rng[0], rng[-1])
    plt.set_yticks(rng, ["%.1f" for i in rng])
    plt.set_xlabel(xlabel)
    plt.set_ylabel("Convertible Bond Price")
    plt.legend(["Binomial", "FDE - Implicit", "FDE - Crank-Nicolson", "FDE - Penalty"],
               prop={"size": "small"})

def main():
    S0 = 100
    Sl = 0
    Su = 200
    X = range(11)
    T = config.T  # Utilisation de T depuis config
    N = lambda x: 2**(x + 1) * T
    K = lambda x: 2**((x + 1) // 2) * (Su - Sl)

    fig = plt.figure()
    fig.set_figwidth(1.8 * fig.get_figwidth())

    plt_x = fig.add_subplot(1, 2, 1)
    plt_t = fig.add_subplot(1, 2, 2)

    p = []
    t = []
    for x in X:
        start = time.time()
        p.append(float(BinomialModel(N(x), config.dS_total, config.payoff).price(S0)))
        t.append(time.time() - start)
    plt_x.plot(X, p)
    plt_t.plot(t, p)

    for scheme in (ImplicitScheme, CrankNicolsonScheme, PenaltyScheme):
        p = []
        t = []
        for x in X:
            k = K(x)
            Sk = k * (S0 - Sl) / (Su - Sl)
            start = time.time()
            p.append(FDEModel(N(x), config.dS_total, config.payoff).price(Sl, Su, k, scheme=scheme).V[0][Sk])
            t.append(time.time() - start)
        plt_x.plot(X, p[:len(X)])
        plt_t.plot(t, p)

    label(plt_x, "$\\delta_t = 2^{-(x + 1)}$; $\\delta_S = 2^{-\\frac{x + 1}{2}}$")
    label(plt_t, "Computational Time (seconds)")
    plt_t.set_xscale('log')

if __name__ == "__main__":
    plotmain(main)

###########################################

"""
Graphique montrant la tarification pour différentes valeurs lambda et eta.
"""
import matplotlib.pyplot as plt
import numpy as np
import copy

from model import FDEModel
from plot import plotmain
from convertible_bond import config  # Utilisation de la configuration fusionnée

def main():
    Sl = 0
    Su = 200
    T = config.T  # Utilisation de T depuis config
    N = 128 * T
    K = 8 * (Su - Sl)
    S = 100
    Sk = K * (S - Sl) / (Su - Sl)
    ETA = np.linspace(0, 1, 65)
    names = ["No default"]

    # Utilisation d'une copie pour éviter la modification de l'instance originale
    dS_copy = copy.deepcopy(config.dS)
    fde = FDEModel(N, dS_copy, config.payoff)

    # Affichage des prix lorsque eta varie, mais lambda est constant
    plt.plot(ETA, [fde.price(Sl, Su, K).V[0][Sk]] * len(ETA))

    for lambd_ in (0.01, 0.02, 0.03):
        dS_copy.lambd_ = lambd_
        V = []
        for eta in ETA:
            dS_copy.eta = eta
            V.append(FDEModel(N, dS_copy, config.payoff).price(Sl, Su, K).V[0][Sk])
        plt.plot(ETA, V)
        names.append(f"$\\lambda = {lambd_ * 100:.0f}\\%$")

    plt.xlabel("$\\eta$")
    plt.ylabel("Price at $S=100$")
    plt.legend(names, loc=3)
    plt.show()

if __name__ == "__main__":
    plotmain(main)

###########################################

"""
Graphique comparatif entre MK12 et ce modèle.
"""
import matplotlib.pyplot as plt
import numpy as np
np.seterr(divide="ignore")

from model import FDEModel
from plot import plotmain
from convertible_bond import config  # Utilisation de la configuration fusionnée

def main():
    S = np.linspace(0, 120, 120 * 8 + 1)
    Sl = 0
    Su = 200
    T = config.T  # Utilisation de T depuis config
    N = 128 * T
    K = 8 * (Su - Sl)
    Sk = (K * (S - Sl) / (Su - Sl)).astype(int)

    # Modèle avec un taux de défaillance constant
    model1 = FDEModel(N, config.dS_mk12, config.payoff)  # Config pour MK12 avec taux de défaillance constant
    # Modèle avec un taux de défaillance variable
    model2 = FDEModel(N, config.dS_var_mk12, config.payoff)  # Config pour MK12 avec taux de défaillance variable

    plt.plot(S, model1.price(Sl, Su, K).V[0][Sk], label="Constant $\\lambda$")
    plt.plot(S, np.append(config.A.R * config.A.N, model2.price(Sl + 1, Su, K - 8).V[0][Sk[:-1]]), label="Synthesis $\\lambda$")
    plt.ylim([40, 160])
    plt.xlabel("Stock Price")
    plt.ylabel("Convertible Bond Price")
    plt.legend(loc=2)
    plt.show()

if __name__ == "__main__":
    plotmain(main)

###########################################

"""
Graphique comparant différents temps call et surface de prix pour C.times = [2].
"""
import matplotlib.pyplot as plt
import numpy as np

from model import FDEModel
from plot import plot_model, plotmain
from convertible_bond import config  # Use centralized configuration

def main():
    S = np.linspace(0, 200, 200 * 8 + 1)
    Sl = 0
    Su = 250
    T = config.T  # Use maturity time from config
    N = 128 * T
    K = 8 * (Su - Sl)
    Sk = (K * (S - Sl) / (Su - Sl)).astype(int)

    # Configure call times dynamically
    original_times = config.C.times  # Save original call times
    config.C.times = [2, 5]  # Set new call times for comparison

    model = FDEModel(N, config.dS_total, config.payoff)
    fig = plt.figure()
    fig.set_figwidth(1.8 * fig.get_figwidth())

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(S, model.price(Sl, Su, K).V[0][Sk], label="$\\Omega^c = [2, 5)$")

    config.C.times = [2]  # Change call times
    ax.plot(S, model.price(Sl, Su, K).V[0][Sk], label="$\\Omega^c = \\{2\\}$")
    plt.xlabel("Stock Price")
    plt.ylabel("Convertible Bond Price")
    plt.legend()

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    plot_model(ax, config.dS_total, config.payoff)  # Use the centralized dS and payoff

    config.C.times = original_times  # Restore original call times after plotting

if __name__ == "__main__":
    plotmain(main)

###########################################

"""
Graph comparing different R and price surface for zero-redemption.
"""
import matplotlib.pyplot as plt
import numpy as np

from model import FDEModel
from plot import plot_model, plotmain
from convertible_bond import config  # Using the centralized configuration

def main():
    S = np.linspace(0, 200, 200 * 8 + 1)
    Sl = 0
    Su = 250
    T = config.T  # Use maturity time from config
    N = 128 * T
    K = 8 * (Su - Sl)
    Sk = (K * (S - Sl) / (Su - Sl)).astype(int)

    # Create a model using the centralized parameters
    model = FDEModel(N, config.dS_total, config.payoff)
    fig = plt.figure()
    fig.set_figwidth(1.8 * fig.get_figwidth())

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(S, model.price(Sl, Su, K).V[0][Sk], label="$R = 104$")

    # Temporary change R for zero redemption and re-plot
    original_N = config.A.N
    config.A.N = -config.A.C  # Adjust N for zero redemption scenario
    ax.plot(S, model.price(Sl, Su, K).V[0][Sk], label="$R = 0$")
    config.A.N = original_N  # Restore original value after plotting

    plt.xlabel("Stock Price")
    plt.ylabel("Convertible Bond Price")
    plt.legend()

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    plot_model(ax, config.dS_total, config.payoff)  # Plot using the centralized dS and payoff

if __name__ == "__main__":
    plotmain(main)

###########################################

"""
Graphique comparant les différents temps call.
"""
import matplotlib.pyplot as plt
import numpy as np

from model import FDEModel
from plot import plotmain
from convertible_bond import config  # Using the centralized configuration

def main():
    S = np.linspace(0, 200, 200 * 8 + 1)
    Sl = 0
    Su = 250
    T = config.T  # Use maturity time from config
    N = 128 * T
    K = 8 * (Su - Sl)
    Sk = (K * (S - Sl) / (Su - Sl)).astype(int)
    legend = []
    label = "$\\Omega^c = \\{%i\\}$"

    model = FDEModel(N, config.dS_total, config.payoff)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    original_times = config.C.times  # Save original call times
    for i in range(1, 5):
        config.C.times = [i]  # Set new call times for comparison
        ax.plot(S, model.price(Sl, Su, K).V[0][Sk], label=label % i)

    config.C.times = original_times  # Restore original call times after plotting

    plt.xlabel("Stock Price")
    plt.ylabel("Convertible Bond Price")
    plt.legend()

if __name__ == "__main__":
    plotmain(main)

###########################################

"""
Graphique comparant les différents temps put.
"""
import matplotlib.pyplot as plt
import numpy as np

from model import FDEModel
from plot import plotmain
from convertible_bond import config  # Using the centralized configuration

def main():
    S = np.linspace(0, 100, 100 * 8 + 1)
    Sl = 0
    Su = 150
    T = config.T  # Use maturity time from config
    N = 128 * T
    K = 8 * (Su - Sl)
    Sk = (K * (S - Sl) / (Su - Sl)).astype(int)
    legend = []
    label = "$\\Omega^p = \\{%i\\}$"

    model = FDEModel(N, config.dS_total, config.payoff)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    original_times = config.P.times  # Save original put times
    for i in range(1, 5):
        config.P.times = [i]  # Set new put times for comparison
        ax.plot(S, model.price(Sl, Su, K).V[0][Sk], label=label % i)

    config.P.times = original_times  # Restore original put times after plotting

    plt.xlabel("Stock Price")
    plt.ylabel("Convertible Bond Price")
    plt.legend()

if __name__ == "__main__":
    plotmain(main)

###########################################

"""
Graphique comparant différents temps de conversion et surfaces de prix pour \Omega^v = {3, 5}.
"""
import matplotlib.pyplot as plt
import numpy as np

from model import FDEModel
from plot import plot_model, plotmain
from convertible_bond import config  # Using the centralized configuration

def main():
    S = np.linspace(0, 200, 200 * 8 + 1)
    Sl = 0
    Su = 250
    T = config.T  # Use maturity time from config
    N = 128 * T
    K = 8 * (Su - Sl)
    Sk = (K * (S - Sl) / (Su - Sl)).astype(int)
    legend = []
    label = "$\\Omega^v = [%i, 5]$"

    model = FDEModel(N, config.dS_total, config.payoff)
    fig = plt.figure()
    fig.set_figwidth(1.8 * fig.get_figwidth())

    ax = fig.add_subplot(1, 2, 1)
    original_times = config.S.times  # Save original conversion times
    for i in range(1, 5):
        config.S.times = [(i, 5)]  # Set new conversion times for comparison
        ax.plot(S, model.price(Sl, Su, K).V[0][Sk], label=label % i)

    config.S.times = original_times  # Restore original conversion times after plotting

    plt.xlabel("Stock Price")
    plt.ylabel("Convertible Bond Price")
    plt.legend()

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    plot_model(ax, config.dS_total, config.payoff)  # Use the centralized dS and payoff

if __name__ == "__main__":
    plotmain(main)
