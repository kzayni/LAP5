"""
MEC6616 Aérodynamique Numérique


@author: Audrey Collard-Daigneault
Matricule: 1920374
    
    
@author: Mohamad Karim ZAYNI
Matricule: 2167132 
 

"""


#----------------------------------------------------------------------------#
#                                 MEC6616                                    #
#                        LAP4 Équations du momentum                          #
#               Collard-Daigneault Audrey, ZAYNI Mohamad Karim               #
#----------------------------------------------------------------------------#

#%% IMPORTATION DES LIBRAIRIES
import numpy as np
from case import Case
import matplotlib.pyplot as plt
import sympy as sp
from processing import Processing

#%% Données du problème
# Propriétés physiques
rho = 1 # masse volumique [kg/m³]
mu = 1  # viscosité dynamique [Pa*s]
U = 1   # Vitesse de la paroi mobile [m/s]

# Dimensions du domaine
b = 1  # Distance entre 2 plaques [m]
L = 1  # Longueur des plaques [m]

def execute(processing, simulations_parameters, postprocessing_parameters, sim_name):
    print("   • En execution")
    processing.set_simulations_and_postprocessing_parameters(simulations_parameters, postprocessing_parameters)
    processing.execute_simulations(sim_name)
    print("   • Simulation terminée")

def ask_P():
    P = input("   Choix du paramètre P (entre -3 et 3): ")
    while float(P) < -3. or float(P) > 3.:
        print("   Erreur")
        P = input("   Choix du paramètre P (entre -3 et 3): ")

    return float(P)


#%% Terme source de pression, champ de vitesse & solution analytique
x, y, P = sp.symbols('x y P')

# Pression
f_dpdx = sp.lambdify([x, y, P], -2*P, "numpy")
def dpdx(x, y, P):
    return f_dpdx(x, y, P)
def dpdy(x, y, P):
    return 0

# Vitesse et solution analytique
couette_flow = U*(y/b) + 1/(2*mu)*dpdx(x, y, P)*y*(y-b)
f_u = sp.lambdify([x, y, P], couette_flow, "numpy")
def u(x, y, P):
    return f_u(x, y, P)
def v(x, y, P):
    return 0

def dudn(x, y, P):
    return 0
def dvdn(x, y, P):
    return 0

def null(x, y, P): return 0

#%% Conditions frontières et domaine
# Conditions frontières (Neumann en entrée et en sortie & Dirichlet aux parois)
bcdata = (['NEUMANN', (dudn, dvdn)], ['DIRICHLET', (u, v)],
          ['NEUMANN', (dudn, dvdn)], ['DIRICHLET', (u, v)])

# Domaine
domain = [[0, 0], [L, 0], [L, b], [0, b]]

#%% Initialisation du cas et du processing
case_classic = Case(rho, mu, source_terms=(dpdx, dpdy), domain=domain)
processing = Processing(case_classic, bcdata)
processing.set_analytical_function((u, null))

#%% Simulations
print("Simulation avec P = 0,3 et un maillage QUAD/TRI")
simulations_parameters = [{'mesh_type': 'QUAD', 'Nx': 8, 'Ny': 8, 'method': 'CENTRE', 'P': 0, 'alpha': 0.75},
                          {'mesh_type': 'TRI', 'Nx': 8, 'Ny': 8, 'method': 'CENTRE', 'P': 0, 'alpha': 0.75},
                          {'mesh_type': 'QUAD', 'Nx': 8, 'Ny': 8, 'method': 'CENTRE', 'P': 3, 'alpha': 0.75},
                          {'mesh_type': 'TRI', 'Nx': 8, 'Ny': 8, 'method': 'CENTRE', 'P': 3, 'alpha': 0.75}]
postprocessing_parameters = {'pyvista': {'simulation': [0,1,2,3]}}
execute(processing, simulations_parameters, postprocessing_parameters, sim_name="couetteclassic_paramP")





