###########################################
###      PROJET - RISQUE DE CRÉDIT      ###
### ALLERS - JALABERT - SIMONEAU-FRIGGI ###
###           file = plot.py            ###
###########################################

"""
Fonctions d'utilité pour le tracé des modèles financiers et des gains.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

from convertible_bond import A, P, C, T, S as Conv
from model import FDEModel, CrankNicolsonScheme

colourise = cm.gist_ncar

# Définir le codage des couleurs des payoffs
PAYOFF_COLORS = {
    "put": 0. / 6,
    "call": 1. / 6,
    "conversion": 2. / 6,
    "forced_conversion": 3. / 6,
    "redemption": 4. / 6,
    "hold": 5. / 6
}

def payoff(t, S, V, I):
    """Déterminez la couleur en fonction du type de payoff à l'instant t."""
    if t in A and t != T:
        V -= A.C

    if V == S and t in Conv:
        if t in C and t != T and I > S:
            return PAYOFF_COLORS["forced_conversion"]
        return PAYOFF_COLORS["conversion"]
    elif t == T:
        return PAYOFF_COLORS["redemption"]
    elif t in P and V == P.payoff.strike(t):
        return PAYOFF_COLORS["put"]
    elif t in C and V == C.payoff.strike(t):
        return PAYOFF_COLORS["call"]
    else:
        return PAYOFF_COLORS["hold"]

def choices(P):
    """Générer une matrice de couleurs basée sur les conditions de payoff."""
    colours = np.zeros((len(P.S), len(P.t)))
    for y, t in enumerate(P.t):
        for x, S in enumerate(P.S):
            colours[x, y] = payoff(t, S, P.V[y][x], P.I[y][x])
    return colourise(colours)

def legend(ax):
    """Créer une légende pour le plot."""
    items = ["Put", "Call", "Conversion", "Forced conversion", "Redemption", "Hold"]
    proxy = [patches.Rectangle((0, 0), 1, 1, fc=colourise(PAYOFF_COLORS[item.lower()])) for item in items]
    ax.legend(proxy, items, loc='upper left', prop={'size': 10})

def plot_strips(ax, X, Y, Z, padding=0, facecolors=None):
    """Tracez le graphique sous la forme d'une série le long de l'axe Y pour visualiser les données financières."""
    for x in range(len(X)):
        width = (X[min(x+1, len(X)-1)] - X[max(x-1, 0)]) / 2 - padding
        Xs, Ys = np.meshgrid([X[x] - width, X[x] + width], Y)
        Zs = np.vstack([Z[:, x], Z[:, x]]).T
        ax.plot_surface(Xs, Ys, Zs, linewidth=0, facecolors=facecolors[:, x:x+1], alpha=0.75)

def plot_model(ax, dS, payoff):
    """Visualiser le model payoff en 3D."""
    N = 40
    model = FDEModel(N, dS, payoff)
    P = model.price(0, 250, 125, scheme=CrankNicolsonScheme)
    colours = choices(P)
    plot_strips(ax, P.t, P.S[:100], np.array(P.V)[:, :100].T, facecolors=colours[:100])
    ax.set_xlabel("Time")
    ax.set_ylabel("Stock Price")
    ax.set_zlabel("Portfolio Value")
    legend(ax)

def plotmain(main):
    """Fonction principale pour gérer les arguments de la ligne de commande et le backend de traçage."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--format', type=str, default='pdf', help='Image format')
    parser.add_argument('--show', action='store_true', help='Show image')
    parser.add_argument('--backend', type=str, help='Rendering backend')
    args = parser.parse_args()

    if args.backend:
        plt.switch_backend(args.backend)
    elif args.show:
        plt.switch_backend('Qt4Agg')
    else:
        plt.switch_backend('Agg')

    main()

    if args.show:
        plt.show()
    else:
        output_file = os.path.join('..', 'common', f'{os.path.basename(sys.argv[0])[:-3]}.{args.format}')
        plt.savefig(output_file)
