"""
MEC6616 Aérodynamique Numérique


@author: Audrey Collard-Daigneault
Matricule: 1920374


@author: Mohamad Karim ZAYNI
Matricule: 2167132


"""

# ----------------------------------------------------------------------------#
#                                 MEC6616                                    #
#                        LAP4 Équations du momentum                          #
#               Collard-Daigneault Audrey, ZAYNI Mohamad Karim               #
# ----------------------------------------------------------------------------#

# %% IMPORT
import numpy as np
from meshConnectivity import MeshConnectivity
from meshGenerator import MeshGenerator
from solver import FVMMomentum
from postProcessing import PostProcessing

import gradientLS as GLS

# %% CLASSE
class Processing:
    """
    Exécute les simulations demandées selon le cas et le post-traitement

    Parameters
    ----------
    case: case
    Cas traité qui a les informations sur la physique du problème

    bcdata: tuple
    Ensemble de donnée sur les conditions limites aux parois

    Attributes
    -------
    case: Case
    Cas traité qui a les informations sur la physique du problème

    bcdata: Tuple
    Ensemble de donnée sur les conditions limites aux parois

    simulations_parameters: List[Dict[str, str]]
    Liste de dictionnaires ayant le nom des paramètres de simulation et leurs valeurs
    
    postprocessing_parameters: Dict[str, Dict[str, float]]]
    Dictionnaire ayant le nom des paramètres de post-traitement et les arguments nécessaires

    """
    def __init__(self, case, bcdata):
        self.case = case
        self.bcdata = bcdata
        self.simulations_parameters = None
        self.postprocessing_parameters = None

        self.test = True

    def set_analytical_function(self, analytical_function) -> None:
        """
        Ajoute une solution analytique au problème simulé lorsque disponible et/ou nécessaire

        Parameters
        ----------
        analytical_function: Tuple[function, function]
        Fonction analytique du problème (u(x,y) et v(x,y))
        """
        self.analytical_function = analytical_function

    def set_simulations_and_postprocessing_parameters(self, simulations_parameters, postprocessing_parameters) -> None:
        """
        Paramétrise les simulations à exécuter et les post-traitement à réaliser

        Parameters
        ----------
        simulations_parameters: List[Dict[str, str]
        Liste de dictionnaires ayant le nom des paramètres de simulation et leurs valeurs

        postprocessing_parameters: Dict[str, Dict[str, float]]]
        Dictionnaire ayant le nom des paramètres de post-traitement et les arguments nécessaires

        Returns
        -------
        None
        """
        self.simulations_parameters = simulations_parameters
        self.postprocessing_parameters = postprocessing_parameters

    def execute_simulations(self, sim_name):
        """
        Exécute les simulations demandées

        Parameters
        ----------
        sim_name: str
        Titre de la simulation effectuée

        Returns
        -------
        None
        """
        postprocessing = PostProcessing(sim_name)

        # Excécute plusieurs simulations selon l'ensemble de parametres de simulation
        for sim_param in self.simulations_parameters:
            # Crée le mesh et calcule les informations extraites à partir du maillage
            mesh_obj = self.compute_mesh_and_connectivity(sim_param)
            preprocessing_data = self.execute_preprocessing(mesh_obj)

            # Initialise le solver et paramétrise la solution analytique et execute la solution selon la méthode
            solver = FVMMomentum(self.case, mesh_obj, self.bcdata, preprocessing_data)
            solver.set_analytical_function(self.analytical_function)
            solver.set_P(sim_param['P'])
            solutions = solver.solve(sim_param['method'], sim_param['alpha'])

            # Store les résultats de la simulation et des infos pertinentes pour le post-traitement
            postprocessing.set_data(mesh_obj, solutions, preprocessing_data, sim_param)

        # S'ils y a du post-traitement, il appelle la fonction qui les exécutes
        if self.postprocessing_parameters is not None:
            self.execute_postprocessing(postprocessing)

    def execute_postprocessing(self, postprocessing) -> None:
        """
        Exécute le post-traitement demandé

        Parameters
        ----------
        postprocessing: postProcessing.PostProcessing
        Object de post-traitement qui a storé les données pendant la simulation

        Returns
        -------
        None
        """

        # Itére sur les keys du dictionnaire pour déterminer les post-traitement à faire
        pp_params = self.postprocessing_parameters
        for param_name in pp_params:

            # Affiche les solutions avec la fonction tricontourf() de matplotlib
            if param_name == 'solutions':
                for sim in pp_params[param_name]['simulation']:
                    postprocessing.show_solutions(i_sim=sim)

            # Affiche la solution selon un plan en x et un en y
            elif param_name == 'plans':
                for sim in pp_params[param_name]['simulation']:
                    postprocessing.show_plan_solutions(i_sim=sim,
                                                       x_coupe=pp_params[param_name]['x'],
                                                       y_coupe=pp_params[param_name]['y'])

            # Calcule l'ordre de convergence et montre l'erreur L2 selon la longueur caractéristique l2
            elif param_name == 'error':
                postprocessing.show_error()

            # Montre les résultats numériques et analytique par PyVista avec le maillage
            elif param_name == 'pyvista':
                for sim in pp_params[param_name]['simulation']:
                    postprocessing.show_pyvista(sim)
            else:
                print(f'Demande de post traitement {param_name} invalide.')

    def compute_mesh_and_connectivity(self, mesh_parameters):
        """
        Exécute la connectivité avec le maillage généré.

        Parameters
        ----------
        mesh_parameters: Dict[str, str]
        Dictionnaire des paramètres pour mesh à générer

        Returns
        -------
        mesh_obj: Mesh
        Maillage de la simulation

        """
        mesher = MeshGenerator()
        mesh_obj = mesher.rectangle_points(self.case.get_domain(), mesh_parameters)
        conec = MeshConnectivity(mesh_obj, verbose=False)
        conec.compute_connectivity()

        return mesh_obj

    def execute_preprocessing(self, mesh_obj):
        """
        Effectue les calculs relatifs au maillage préalablement à l'utilisation du solver

        Parameters
        ----------
        mesh_obj: Mesh
        Maillage de la simulation

        Returns
        -------
        (volumes, centroids): Tuple[numpy.ndarray, numpy.ndarray]
        Arrays storant les volumes des éléments et la position du centroide

        """
        n_elem = mesh_obj.get_number_of_elements()  # Nombre d'éléments dans notre maillage
        volumes = np.zeros(n_elem)  # surface des éléments
        centroids = np.zeros((n_elem, 2))  # coordonnees des centroides

        # Détermine les centroides et l'aire de l'élément par les déterminants
        for i_elem in range(n_elem):
            nodes = mesh_obj.get_element_to_nodes(i_elem)
            area_matrices = [np.zeros([2, 2]) for i in range(len(nodes))]
            for i in range(len(nodes)):
                x, y = mesh_obj.get_node_to_xycoord(nodes[i])[0], mesh_obj.get_node_to_xycoord(nodes[i])[1]
                area_matrices[i][:, 0] = [x, y]
                area_matrices[i - 1][:, 1] = [x, y]

            # Calcule l'aire de l'élément
            volumes[i_elem] = np.sum([np.linalg.det(area_matrices[i]) for i in range(len(nodes))]) / 2

            # Calcule du position des centroides
            cx = (np.sum(
                [np.sum(area_matrices[i][0, :]) * np.linalg.det(area_matrices[i]) for i in range(len(nodes))]) /
                  (6 * volumes[i_elem]))
            cy = (np.sum(
                [np.sum(area_matrices[i][1, :]) * np.linalg.det(area_matrices[i]) for i in range(len(nodes))]) /
                  (6 * volumes[i_elem]))

            centroids[i_elem] = [cx, cy]

        return volumes, centroids

