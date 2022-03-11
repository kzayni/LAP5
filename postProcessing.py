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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib import cm
import pyvista as pv
import pyvistaqt as pvQt
from meshGenerator import MeshGenerator
from meshPlotter import MeshPlotter


#%% Fonction interne
def Coupe_X(Coordonnees, X, Solution, Analytique, Plan):
    Elements_ds_coupe = []
    Solution_coupe = []
    Analytique_coupe = []
    eps = 1e-6  # Précision
    for i in range(len(Coordonnees)):
        if np.abs(Coordonnees[i, Plan] - X) < eps:
            Elements_ds_coupe.append(Coordonnees[i, :])
            Solution_coupe.append(Solution[i])
            Analytique_coupe.append(Analytique[i])
    Elements_ds_coupe = np.array(Elements_ds_coupe)
    Solution_coupe = np.array(Solution_coupe)
    return Elements_ds_coupe, Solution_coupe, Analytique_coupe

#%% Classe
class PostProcessing:
    """
    Effectuer le post_traitement après la résolution du problème selon une ou plusieurs simulations
    Parmeters
    ---------
    sim_name: str
    Nom d'un groupe de simulation

    Attributes
    ----------
    sim_name: str
    Nom d'un groupe de simulation

    data: Dict
    Dictionnaire qui comporte toutes les données de nécessaire pour effectuer le post-traitement.

    """
    def __init__(self,  sim_name):
        self.sim_name = sim_name
        self.data = []  # Initialisation du dictionnaire de données

    def get_sim_name(self):
        return self.sim_name

    def set_data(self, mesh, solutions, preprocessing_data, simulation_paramaters):
        """
        Ajouter des données après une simulation à un ensemble de simulation à étudier

        Parameters
        ----------
        mesh: mesh.Mesh
        Maillage de la simulation

        solutions: Tuple[Tuple[numpy.ndarray, numpy.ndarray], Tuple[numpy.ndarray, numpy.ndarray]]
        Solutions numérique (u, v) et solutions analytiques (u_exact, v_exact)

        preprocessing_data: Tuple[numpy.ndarray, numpy.ndarray],
        Arrays contenant les données de preprocessing (volumes et position du centroide des éléments)

        simulation_paramaters: Dict[str, str]
        Paramètres de simulation pour permettre leur affichage dans les titres ou les noms de sauvegarde

        Returns
        -------
        None
        """

        # Calcule les normes de vitesse
        phi_num, phi_ex = np.zeros(len(solutions[0][0])), np.zeros(len(solutions[0][0]))
        for i in range(len(solutions[0][0])):
            phi_num[i] = np.sqrt(solutions[0][0][i]**2 + solutions[0][1][i]**2)
            phi_ex[i] = np.sqrt(solutions[1][0][i]**2 + solutions[1][1][i]**2)

        # Stockage de données de la simulation
        # n: int                        -> Nombre d'éléments dans le maillage
        # mesh: Mesh                    -> maillage (utilise pour PyVista)
        # u_num, v_num: np.ndarray      -> Solutions numériques pour les 2 directions
        # u_exact, v_exact: np.ndarray  -> Solutions analytiques pour les 2 directions
        # phi_num: np.ndarray           -> Norme de la solution numérique
        # phi_exact: np.ndarray         -> Norme de la solution analytique
        # area: np.ndarray              -> Volume des éléments
        # position: np.ndarray          -> Position du centroides des éléments
        # P: float                      -> Valeur du paramètre P
        # method: str                   -> Méthode utilisée (correction pour la convection) CENTRE ou UPWIND
        self.data.append({'n': mesh.get_number_of_elements(),
                          'mesh': mesh,
                          'u_num': solutions[0][0], 'u_exact': solutions[1][0],
                          'v_num': solutions[0][1], 'v_exact': solutions[1][1],
                          'phi_num': phi_num,
                          'phi_exact': phi_ex,
                          'area': preprocessing_data[0],
                          'position': preprocessing_data[1],
                          'P': simulation_paramaters['P'],
                          'method': simulation_paramaters['method']})


    def show_solutions(self, i_sim):
        """
        Affichage des graphiques qui montrent la différence entre la solution numérique et la solution analytique
        à l'aide de fonctions tricontour dans matplotlib

        Parameters
        ----------
        i_sim: int
        Numéro de la simulation sélectionnée pour le post-processing de solution

        Returns
        -------
        None
        """

        figure, (num, ex) = plt.subplots(1, 2, figsize=(20, 6))
        cmap = cm.get_cmap('jet')

        # Titre de la figure
        sim_name = self.get_sim_name()
        title = f'Solution avec contours de la simulation "{sim_name}" de la vitesse u avec {self.data[i_sim]["n"]} ' \
                f'éléments pour P = {self.data[i_sim]["P"]} utilisant une méthode {self.data[i_sim]["method"]}'
        figure.suptitle(title)

        # Mise à l'échelle de la colorbar pour les 2 graphiques
        levels = np.linspace(np.min([self.data[i_sim]['u_num'], self.data[i_sim]['u_exact']]),
                             np.max([self.data[i_sim]['u_num'], self.data[i_sim]['u_exact']]), num=20)

        # Solution numérique
        c = num.tricontourf(self.data[i_sim]['position'][:, 0],
                            self.data[i_sim]['position'][:, 1],
                            self.data[i_sim]['u_num'], levels=levels, cmap=cmap)
        plt.colorbar(c, ax=num)
        num.tricontour(self.data[i_sim]['position'][:, 0],
                       self.data[i_sim]['position'][:, 1],
                       self.data[i_sim]['u_num'], '--', levels=levels, colors='k')
        num.set_xlabel("L (m)")
        num.set_ylabel("b (m)")
        num.set_title("Solution numérique")

        # Solution analytique
        c = ex.tricontourf(self.data[i_sim]['position'][:, 0],
                           self.data[i_sim]['position'][:, 1],
                           self.data[i_sim]['u_exact'], levels=levels, cmap=cmap)
        plt.colorbar(c, ax=ex)
        ex.tricontour(self.data[i_sim]['position'][:, 0],
                      self.data[i_sim]['position'][:, 1],
                      self.data[i_sim]['u_exact'], '--', levels=levels, colors='k')
        ex.set_xlabel("L (m)")
        ex.set_ylabel("b (m)")
        ex.set_title("Solution analytique analytique")

        save_path = f"images/{sim_name}_sim{i_sim}_contour.png"
        plt.savefig(save_path, dpi=200)
        plt.clf()

    def show_plan_solutions(self, i_sim, x_coupe, y_coupe):
        """
        Affiche les résultats selon un plan et x et un plan en y

        Parameters
        ----------
        i_sim: int
        Numéro de la simulation sélectionnée pour le post-processing de solution

        x_couple: float
        Position de la coupe en x

        y_coupe: float
        Position de la coupe en y

        Returns
        -------
        None
        """

        figure, (COUPEX, COUPEY) = plt.subplots(1, 2, figsize=(20, 6))

        # Titre de la figure
        sim_name = self.get_sim_name()
        title = f'Solution de plans de la simulation "{sim_name}" de la vitesse u avec {self.data[i_sim]["n"]} ' \
                f'éléments pour P = {self.data[i_sim]["P"]} utilisant une méthode {self.data[i_sim]["method"]}'
        figure.suptitle(title)

        # Chercher l'indice des éléments à un X ou Y donné
        centres = self.data[i_sim]['position']
        elem_ds_coupeX, solution_coupeX, solutionEX_coupeX = \
            Coupe_X(centres, x_coupe, self.data[i_sim]['u_num'], self.data[i_sim]['u_exact'], 0)
        elem_ds_coupeY, solution_coupeY, solutionEX_coupeY = \
            Coupe_X(centres, y_coupe, self.data[i_sim]['u_num'], self.data[i_sim]['u_exact'], 1)

        COUPEX.plot(solution_coupeX, elem_ds_coupeX[:, 1], label="Solution numérique")
        COUPEX.plot(solutionEX_coupeX, elem_ds_coupeX[:, 1], '--', label="Solution analytique")
        COUPEX.set_xlabel("Vitesse")
        COUPEX.set_ylabel("Y (m)")
        COUPEX.set_title(f"Solution dans une coupe à X = {x_coupe}")
        COUPEX.legend()

        COUPEY.plot(elem_ds_coupeY[:, 0], solution_coupeY, label="Solution numérique")
        COUPEY.plot(elem_ds_coupeY[:, 0], solutionEX_coupeY, '--', label="Solution analytique")
        COUPEY.set_xlabel("X (m)")
        COUPEY.set_ylabel("Vitesse")
        COUPEY.set_title(f"Solution dans une coupe à Y = {y_coupe}")
        COUPEY.legend()

        # Enregistrer
        save_path = f"images/{sim_name}_sim{i_sim}_plans.png"
        plt.savefig(save_path, dpi=200)
        plt.clf()

    def show_pyvista(self, i_sim, norm=True):
        pv.set_plot_theme("document")

        # Préparation du maillage
        plotter = MeshPlotter()
        nodes, elements = plotter.prepare_data_for_pyvista(self.data[i_sim]['mesh'])
        pv_mesh_num = pv.PolyData(nodes, elements)
        pv_mesh_ex = pv.PolyData(nodes, elements)

        # Affiche la norme de la vitesse ou la vitesse en u
        if norm is True:
            # Solutions numériques et analytiques
            pv_mesh_num['Vitesse numérique'] = self.data[i_sim]['phi_num']
            pv_mesh_ex['Vitesse analytique'] = self.data[i_sim]['phi_exact']
            levels = [np.min(np.append(self.data[i_sim]['phi_num'], self.data[i_sim]['phi_exact'])),
                      np.max(np.append(self.data[i_sim]['phi_num'], self.data[i_sim]['phi_exact']))]
        else:
            pv_mesh_num['Vitesse numérique'] = self.data[i_sim]['u_num']
            pv_mesh_ex['Vitesse analytique'] = self.data[i_sim]['u_exact']
            levels = [np.min(np.append(self.data[i_sim]['u_num'], self.data[i_sim]['u_exact'])),
                      np.max(np.append(self.data[i_sim]['u_num'], self.data[i_sim]['u_exact']))]

        # Création des graphiques
        pl = pv.Plotter(shape=(1, 2))  # Avant pvQt.BackgroundPlotter()

        # Solution numérique
        pl.add_text(f"Solution numérique u \n {self.data[i_sim]['n']} éléments\n "
                    f"P = {self.data[i_sim]['P']}\n Méthode = {self.data[i_sim]['method']}", font_size=15)
        pl.add_mesh(pv_mesh_num, show_edges=True, scalars='Vitesse numérique', cmap="jet", clim=levels)
        pl.camera_position = 'xy'
        pl.show_bounds()

        # Solution analytique
        pl.subplot(0, 1)
        pl.add_text(f"Solution analytique u \n {self.data[i_sim]['n']} éléments\n "
                    f"P = {self.data[i_sim]['P']}\n Méthode = {self.data[i_sim]['method']}", font_size=15)
        pl.add_mesh(pv_mesh_ex, show_edges=True, scalars='Vitesse analytique', cmap="jet", clim=levels)
        pl.camera_position = 'xy'
        pl.show_bounds()

        pl.link_views()
        sim_name = self.get_sim_name()
        save_path = f"images/{sim_name}_sim{i_sim}_pyvista.png"
        pl.show(screenshot=save_path)
        pl.clear()

    def show_error(self):
        """
        Affichage des graphiques d'ordre de convergence et calcul de l'erreur par rapport au solution exacte.
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Calcul l'erreur l'ajoute aux données et détermine l'ordre de convergence
        for i in range(len(self.data)):
            total_area = np.sum(self.data[i]['area'])
            area, n = self.data[i]['area'], self.data[i]['n']
            phi_num, phi_exact = self.data[i]['phi_num'], self.data[i]['phi_exact']
            E_L2 = np.sqrt(np.sum(area*(phi_num - phi_exact)**2)/total_area)
            self.data[i]['err_L2'] = E_L2
            self.data[i]['h'] = np.sqrt(total_area/n)

        p = np.polyfit(np.log([self.data[i]['h'] for i in range(len(self.data))]),
                  np.log([self.data[i]['err_L2'] for i in range(len(self.data))]), 1)

        # Graphique de l'erreur
        fig_E, ax_E = plt.subplots(figsize=(15, 10))
        fig_E.suptitle("Normes de l'erreur L² des solutions numériques sur une échelle de logarithmique", y=0.925)
        text = AnchoredText('Ordre de convergence: ' + str(round(p[0], 2)), loc='upper left')

        ax_E.loglog([self.data[i]['h'] for i in range(len(self.data))],
                  [self.data[i]['err_L2'] for i in range(len(self.data))], '.-')
        ax_E.minorticks_on()
        ax_E.grid(True, which="both", axis="both", ls="-")
        ax_E.set_xlabel('Grandeur (h)')
        ax_E.set_ylabel('Erreur (E)')
        ax_E.add_artist(text)

        # Enregistrer
        save_path = f"images/{self.sim_name}_error.png"
        plt.savefig(save_path, dpi=200)
        plt.clf()

