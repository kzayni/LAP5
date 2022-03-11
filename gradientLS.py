"""
MEC6616 Aérodynamique Numérique


@author: Audrey Collard-Daigneault
Matricule: 1920374
    
    
@author: Mohamad Karim ZAYNI
Matricule: 2167132 
 

"""

# ----------------------------------------------------------------------------#
#                                 MEC6616                                     #
#                        LAP4 Équations du momentu                            #
#               Collard-Daigneault Audrey, ZAYNI Mohamad Karim                #
# ----------------------------------------------------------------------------#

# %% IMPORTATION DES LIBRAIRIES
import numpy as np


# %% Classe Gradient LS
class GradientLeastSquares:
    """
    Solveur utilisant la méthode des moindres carrés pour reconstruire le gradient

    Parameters
    ---------
    mesh_obj: Mesh
    Maillage de la simulation

    bcdata: Tuple
    Ensemble de donnée sur les conditions limites aux parois

    centroids: numpy.ndarray
    Array storant les coordonnées des centroides des éléments

    Attributes
    ----------
    mesh_obj: Mesh
    Maillage de la simulation

    bcdata: Tuple
    Ensemble de donnée sur les conditions limites aux parois

    centroids: numpy.ndarray
    Array storant les coordonnées des centroides des éléments

    """

    def __init__(self, mesh_obj, bcdata, centroids):
        self.mesh_obj = mesh_obj  # Maillage du cas
        self.bcdata = bcdata  # Conditions frontières
        self.centroids = centroids

    # Accesseurs
    def get_mesh(self):
        return self.mesh_obj

    def get_bcdata(self):
        return self.bcdata

    def get_centroids(self):
        return self.centroids

    def get_P(self):
        return self.P

    # Modificateurs
    def set_P(self, new):
        self.P = new

    # Calcule le gradient du cas étudié
    def solve(self, phiu, phiv):

        """
        Calcule les gradients des phi selon leur valeur
        
        Parameters
        ----------
        phiu: numpy.ndarray
        Valeurs de u

        phiv: numpy.ndarray
        Valeurs de v
            
        
        Returns
        -------
        (GRADu, GRADv): Tuple[numpy.ndarray, numpy.ndarray]
        GRADu -> Gradients dudx et dudy
        GRADv -> Gradients dvdx et dvdy

        """

        # Initialisation des matrices et des données
        NTRI = self.mesh_obj.get_number_of_elements()  # Nombre d'éléments
        ATA = np.zeros((NTRI, 2, 2))
        ATAI = np.zeros((NTRI, 2, 2))
        Bu, Bv = np.zeros((NTRI, 2)), np.zeros((NTRI, 2))
        GRADu, GRADv = np.zeros((NTRI, 2)), np.zeros((NTRI, 2))

        # Variables locales
        mesh = self.get_mesh()  # Maillage utilisé
        centroids = self.centroids  # Chercher les centres des éléments
        bcdata = self.get_bcdata()  # Conditions limites
        P = self.get_P()  # Paramètre modifié

        # Remplissage des matrices pour le cas d'une condition frontière (Libre, Dirichlet ou Neumann)
        for i_face in range(mesh.get_number_of_boundary_faces()):
            # Détermine le numéro de la frontière et les conditions associées
            tag = mesh.get_boundary_face_to_tag(i_face)
            bc_type, (bc_x, bc_y) = bcdata[tag]

            # Listes des noeuds et des éléments reliés à la face
            nodes = mesh.get_face_to_nodes(i_face)
            element = mesh.get_face_to_elements(i_face)[0]

            # Détermination des positions des points et de la distance
            xA = (mesh.get_node_to_xycoord(nodes[0])[0] + mesh.get_node_to_xycoord(nodes[1])[0]) / 2.
            yA = (mesh.get_node_to_xycoord(nodes[0])[1] + mesh.get_node_to_xycoord(nodes[1])[1]) / 2.
            xP, yP = centroids[element][0], centroids[element][1]
            dx, dy = xA - xP, yA - yP

            if bc_type == 'DIRICHLET':
                # Calcul la différence des phi entre le point au centre de la face et au centre de l'élément
                dphiu = bc_x(xA, yA, P) - phiu[element]
                dphiv = bc_y(xA, yA, P) - phiv[element]

                ALS = np.array([[dx * dx, dx * dy], [dy * dx, dy * dy]])
            elif bc_type == 'NEUMANN':
                # Modification de la position du point sur la face si Neumann
                (xa, ya), (xb, yb) = mesh.get_node_to_xycoord(nodes[0]), mesh.get_node_to_xycoord(nodes[1])
                dA = np.sqrt((xb - xa) ** 2 + (yb - ya) ** 2)
                n = np.array([(yb - ya) / dA, -(xb - xa) / dA])
                dx, dy = np.dot([dx, dy], n) * n

                # Application de la condition frontière au point sur la face perpendiculaire au point central
                dphiu = np.dot([dx, dy], n) * bc_x(xa, ya, P)
                dphiv = np.dot([dx, dy], n) * bc_y(xa, ya, P)

                ALS = np.array([[dx * dx, dx * dy], [dy * dx, dy * dy]])
            else:  # Condition libre
                ALS = np.zeros(2, 2)
                dphiu, dphiv = 0, 0

            ATA[element] += ALS

            Bu[element] += (np.array([dx, dy]) * dphiu)
            Bv[element] += (np.array([dx, dy]) * dphiv)



        # Remplissage des matrices pour les faces internes
        for i_face in range(mesh.get_number_of_boundary_faces(), mesh.get_number_of_faces()):
            elements = mesh.get_face_to_elements(i_face)
            dx, dy = centroids[elements[1]] - centroids[elements[0]]

            # Remplissage de la matrice ATA pour l'arête interne
            ALS = np.array([[dx * dx, dx * dy], [dy * dx, dy * dy]])
            ATA[elements[0]] += ALS
            ATA[elements[1]] += ALS

            # Remplisage du membre de droite
            dphiu = phiu[elements[1]] - phiu[elements[0]]
            dphiv = phiv[elements[1]] - phiv[elements[0]]

            Bu[elements[0]] += (np.array([dx, dy]) * dphiu)
            Bu[elements[1]] += (np.array([dx, dy]) * dphiu)
            Bv[elements[0]] += (np.array([dx, dy]) * dphiv)
            Bv[elements[1]] += (np.array([dx, dy]) * dphiv)

        # Résolution des systèmes matriciels pour tous les éléments
        for i_tri in range(NTRI):
            ATAI[i_tri] = np.linalg.inv(ATA[i_tri])
            GRADu[i_tri] = np.dot(ATAI[i_tri], Bu[i_tri])
            GRADv[i_tri] = np.dot(ATAI[i_tri], Bv[i_tri])

        return GRADu, GRADv
