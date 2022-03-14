"""
MEC6616 Aérodynamique Numérique


@author: Audrey Collard-Daigneault
Matricule: 1920374
    
    
@author: Mohamad Karim ZAYNI
Matricule: 2167132 
 

"""


#----------------------------------------------------------------------------#
#                                 MEC6616                                    #
#                        TPP2 Convection-Diffusion                           #
#               Collard-Daigneault Audrey, ZAYNI Mohamad Karim               #
#----------------------------------------------------------------------------#

#%% NOTES D'UTILISATION
"""

Classe pour calculer le gradient d'un champ donné avec la méthode des moindres carrés. 

"""

#%% IMPORTATION DES LIBRAIRIES

import numpy as np
import mesh_preprocessing as MPREP


#%% Classe Gradient LS

class GradientMoindresCarres:
    """
    Solveur utilisant la méthode des moindres carrés pour reconstruire le gradient
.

    Parmeters
    ---------
    exempmle: case
        L'exemple en cours de traitement
    

    Attributes
    ----------
    mesh_obj: mesh
        Maillage du problème
    
    bcdata: liste de float et str.
        Type + valeur des conditions aux limites imposées. 
        
    phi: ndarray float.
        Champ sur notre domaine
        
    gradient: ndarray float
        Gradient du champ phi

    """
    def __init__(self, case):
        self.case = case                 # Cas à résoudre
        self.mesh_obj = case.get_mesh()  # Maillage du cas
        self.bcdata = case.get_bc()      # Types de conditions frontière
        self.phi=0                      # Champ sur le maillage
        self.gradient=0                  #Gradient du champ phi
     
    #Accesseurs
    def get_case(self):
        """
        Retourne le cas en cours

        Parameters
        ----------
        None
        
        Returns
        -------
        case: Case
            cas en cours d'étude
        """
        return self.case
    
    def get_mesh(self):
        """
        Retourne maillage utilisé

        Parameters
        ----------
        None
        
        Returns
        -------
        mesh_obj: objet maillage.
            maillage utilisé
        """
        return self.mesh_obj
    
    def get_bcdata(self):
        """
        Retourne Conditions limites

        Parameters
        ----------
        None
        
        Returns
        -------
        bcdata: dictionnaire str/float
            Conditions limites
        """
        return self.bcdata
    
    def get_phi(self):
        """
        Retourne Champ de variable

        Parameters
        ----------
        None
        
        Returns
        -------
        phi: ndarray float
            Champ de variable
        """
        return self.phi
    
    def get_gradient(self):
        """
        Retourne gradient d'une variable

        Parameters
        ----------
        None
        
        Returns
        -------
        gradient: ndarray float
            gradient d'une variable
        """
        return self.gradient
    
    #Modificateurs
    def set_case(self,new):
        """
        Modifier le cas en cours

        Parameters
        ----------
        new: Case
            cas étudié

        Returns
        -------
        None
        
        """
        self.case=new
    
    def set_mesh(self,new):
        """
        Modifier le maillage.

        Parameters
        ----------
        new: Objet maillage
            Nouveau maillage. 

        Returns
        -------
        None

        """
        self.mesh_obj=new
    
    def set_bcdata(self,new):
        """
        Modifier les conditions limites

        Parameters
        ----------
        new: Dictionnaire str/float
            Conditions limites

        Returns
        -------
        None
        
        """
        self.bcdata=new
    
    def set_phi(self,new):
        """
        Modifier la variable

        Parameters
        ----------
        new: ndarray float
            Champ de variable

        Returns
        -------
        None
        
        """
        self.phi=new
    
    def set_gradient(self,new):
        """
        Modifier le gradient 

        Parameters
        ----------
        new: ndarray float
            nouveau gradient de la variable

        Returns
        -------
        None
        
        """
        self.gradient=new

    # Calcule le gradient du cas étudié
    def solve(self):
        
        """
        Calcule le gradient du cas étudié
        
        Parameters
        ----------
        None
            
        
        Returns
        -------
        None
        
        """
        
        # Initialisation des matrices et des données
        Case=self.get_case()
        Mesh=self.get_mesh() #Maillage du problème
        BCdata=self.get_bcdata() #Conditions aux limites
        
        Areas,Centroids=Case.get_areas_and_centroids()
        Phi=self.get_phi() #Variable qu'on travaille avec
        
        NTRI = Mesh.get_number_of_elements() #Nombre d'éléments
        ATA = np.zeros((NTRI, 2, 2))
        B = np.zeros((NTRI, 2))

        # Remplissage des matrices pour le cas d'une condition frontière (Dirichlet ou Neumann)
        for i_face in range(Mesh.get_number_of_boundary_faces()):
            tag = Mesh.get_boundary_face_to_tag(i_face)  # Numéro de la frontière de la face
            bc_type, bc_value = BCdata[tag]  # Condition frontière (Dirichlet ou Neumann)
            element = Mesh.get_face_to_elements(i_face)[0]  # Élément de la face

            # Détermination des positions des points et de la distance
            nodes = Mesh.get_face_to_nodes(i_face)
            xa = (Mesh.get_node_to_xycoord(nodes[0])[0] + Mesh.get_node_to_xycoord(nodes[1])[0]) / 2.
            ya = (Mesh.get_node_to_xycoord(nodes[0])[1] + Mesh.get_node_to_xycoord(nodes[1])[1]) / 2.
            xb, yb = Centroids[element][0], Centroids[element][1]
            dx, dy = xb - xa, yb - ya

            if bc_type == 'DIRICHLET':
                # Calcul la différence des phi entre le point au centre de la face et au centre de l'élément
                dphi = bc_value(xa, ya) - Phi[element]

            if bc_type == 'NEUMANN':
                # Modification de la position du point sur la face si Neumann
                (xa, ya), (xb, yb) = Mesh.get_node_to_xycoord(nodes[0]), Mesh.get_node_to_xycoord(nodes[1])
                dA = np.sqrt((xb - xa) ** 2 + (yb - ya) ** 2)
                n = np.array([(yb - ya) / dA, -(xb - xa) / dA])
                dx, dy = np.dot([dx, dy], n) * n

                # Application de la condition frontière au point sur la face perpendiculaire au point central
                dphi = np.dot([dx, dy], n) * bc_value(xa, ya)

            # Remplissage de la matrice ATA
            ALS = np.array([[dx * dx, dx * dy], [dy * dx, dy * dy]])
            ATA[element] += ALS

            # Remplisage du membre de droite
            B[element] += (np.array([dx, dy]) * dphi)

        # Remplissage des matrices pour les faces internes
        for i_face in range(Mesh.get_number_of_boundary_faces(), Mesh.get_number_of_faces()):
            elements = Mesh.get_face_to_elements(i_face)
            dx, dy = Centroids[elements[1]] - Centroids[elements[0]]

            # Remplissage de la matrice ATA pour l'arête interne
            ALS = np.array([[dx * dx, dx * dy], [dy * dx, dy * dy]])
            ATA[elements[0]] += ALS
            ATA[elements[1]] += ALS

            # Remplisage du membre de droite
            dphi = Phi[elements[0]] - Phi[elements[1]]
            B[elements[0]] += (np.array([dx, dy]) * dphi)
            B[elements[1]] += (np.array([dx, dy]) * dphi)

        # Résolution des systèmes matriciels pour tous les éléments
        ATAI = np.array([np.linalg.inv(ATA[i_tri]) for i_tri in range(NTRI)])
        GRAD = np.array([np.dot(ATAI[i_tri], B[i_tri]) for i_tri in range(NTRI)])

        self.set_gradient(GRAD)