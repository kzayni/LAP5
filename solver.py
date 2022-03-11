"""
MEC6616 Aérodynamique Numérique


@author: Audrey Collard-Daigneault
Matricule: 1920374
    
    
@author: Mohamad Karim ZAYNI
Matricule: 2167132 
 

"""

# ----------------------------------------------------------------------------#
#                                 MEC6616                                     #
#                        LAP4 Équations du momentum                           #
#               Collard-Daigneault Audrey, ZAYNI Mohamad Karim                #
# ----------------------------------------------------------------------------#

# %% IMPORTATION DES LIBRAIRIES

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
import gradientLS as GLS
import sys

# %% Fonctions Internes
def compute_lengths_and_unit_vectors(pta, ptb, ptA, ptP):
    """
    Calcule les distances et vecteurs unitaires nécessaires selon les coordonnées fournies

    Parameters
    ----------
    pta: Tuple[numpy.float64, numpy.float64]
    Coordonnée du noeud 1 de la face

    ptb: Tuple[numpy.float64, numpy.float64]
    Coordonnée du noeud 2 de la face

    ptA: numpy.ndarray
    Coordonnée du centroide de l'élément de droite

    ptP: numpy.ndarray) -> Tuple[numpy.float64, numpy.float64, numpy.ndarray, numpy.ndarray, numpy.ndarray]
    Coordonnée du centroide de l'élément de gauche

    Returns
    -------
    (dA, dKSI, n, eKSI, eETA): Tuple[numpy.float64, numpy.float64, numpy.ndarray, numpy.ndarray, numpy.ndarray]
    dA   -> Aire de la face
    dKSI -> Distance entre les deux centroides
    n    -> Vecteur normal de la face
    eKSI -> Vecteur unitaire du centroide de gauche vers celui de droite
    eETA -> Vecteur unitaire du noeud 1 vers 2

    """

    (xa, ya), (xb, yb), (xA, yA), (xP, yP) = pta, ptb, ptA, ptP

    # Détermination des distances
    dx, dy = (xb - xa), (yb - ya)
    dA = np.sqrt(dx ** 2 + dy ** 2)
    dKSI = np.sqrt((xA - xP) ** 2 + (yA - yP) ** 2)

    # Détermination des vecteurs
    n = np.array([dy / dA, -dx / dA])
    eKSI = np.array([(xA - xP) / dKSI, (yA - yP) / dKSI])
    eETA = np.array([dx / dA, dy / dA])

    return dA, dKSI, n, eKSI, eETA

def compute_source(P, dpdx, dpdy, volumes, centroids):
    """
    Calcule le terme source dû au gradient de pression
def compute_source(P: int, dpdx: function, dpdy: function, volumes: numpy.ndarray, centroids: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:


    Parameters
    ----------
    P: int
    Paramètre propre au problème étudié

    dpdx: function
    Fonction évaluant le terme dp/dx avec x, y, et P

    dpdy: function
    Fonction évaluant le terme dp/dy avec x, y, et P

    volumes: numpy.ndarray
    Array storant les volumes des éléments

    centroids: numpy.ndarray
    Array storant les coordonnées des centroides des éléments

    Returns
    -------
    (SGPXp, SGPYp): Tuple[numpy.ndarray, numpy.ndarray]
    SGPXp -> Terme de source pour l'équation du momentum en x lié à la pression
    SGPYp -> Terme de source pour l'équation du momentum en y lié à la pression

    """
    SGPXp, SGPYp = np.zeros(len(volumes)), np.zeros(len(volumes))

    # Calcule les gradients de pression aux centroids des éléments * volume
    for i in range(len(volumes)):
        SGPXp[i] = -dpdx(x=centroids[i][0], y=centroids[i][1], P=P) * volumes[i]
        SGPYp[i] = -dpdy(x=centroids[i][0], y=centroids[i][1], P=P) * volumes[i]

    return SGPXp, SGPYp


def compare_RC(U_RC,U_AVG):
    """
    Réaliser la comparaison entre les vitesses débitantes à travers les faces internes
    calculées par Rhie_Chow et celles calculées par une moyenne

    Parameters
    ----------
    U_RC: ndarray [float]
        Vitesse calculée par Rhie_Chow

    U_AVG: ndarray [float]
        vitesse moyennée sur les faces internes

    Returns
    -------
    diff: ndarray [float]
        renvoit la différence entre les deux arrays
        
    test_result: bool
        renvoit le résultat du test 
    """
    
    nb_internal_faces=len(U_AVG) #On prend les faces internes uniquement
    diff=np.zeros(nb_internal_faces)
    nb_boundary_faces=len(U_RC)-nb_internal_faces #Nombre de faces frontières
    eps=10e-10
    test_result= False
    
    #Boucle sur les faces internes
    for i_face in range(nb_internal_faces):
        diff[i_face]=np.abs(U_AVG[i_face]-U_RC[nb_boundary_faces+i_face])
        
        if(diff[i_face]<eps): #Test réussi en dessous de la précision souhaitée
            test_result=True
        else:
            test_result=False
        
    return diff,test_result

    
        
# %% Classe
class FVMMomentum:
    """
    Méthode de volumes finis pour un problème de transfert de momentum en 2D

    Parameters
    ---------
    case: Case
    Cas traité qui a les informations sur la physique du problème

    mesh_obj: Mesh
    Maillage de la simulation

    bcdata: Tuple
    Ensemble de donnée sur les conditions limites aux parois

    preprocessing_data: Tuple[numpy.ndarray, numpy.ndarray]
    Arrays storant les volumes des éléments et la position du centroide

    Attributes
    ----------
    case: Case
    Cas traité qui a les informations sur la physique du problème

    mesh_obj: Mesh
    Maillage de la simulation

    bcdata: Tuple
    Ensemble de donnée sur les conditions limites aux parois

    volumes: numpy.ndarray
    Array storant les volumes des éléments

    centroids: numpy.ndarray
    Array storant les coordonnées des centroides des éléments

    """

    def __init__(self, case, mesh_obj, bcdata, preprocessing_data):
        self.case = case  # Cas à résoudre
        self.mesh_obj = mesh_obj  # Maillage du cas
        self.bcdata = bcdata  # Conditions frontières
        self.volumes = preprocessing_data[0]
        self.centroids = preprocessing_data[1]

    def set_analytical_function(self, analytical_function):
        """
        Ajoute une solution analytique au problème simulé lorsque disponible et/ou nécessaire

        Parameters
        ----------
        analytical_function: Tuple[function, function]
        Fonction analytique du problème (u(x,y) et v(x,y))
        """
        self.analytical_function = analytical_function
    

    # Accesseurs
    def get_case(self):
        return self.case

    def get_mesh(self):
        return self.mesh_obj

    def get_bcdata(self):
        return self.bcdata

    def get_volumes(self):
        return self.volumes

    def get_centroids(self):
        return self.centroids

    def get_cross_diffusion(self):
        return self.get_cross_diffusion

    def get_analytical_function(self):
        return self.analytical_function
    

    def get_P(self):
         return self.P

    # Modificateurs
    def set_P(self, new):
        self.P = new

    # Solveur VF
    def solve(self, method="CENTRE", alpha=0.75):
        """
        Effectue les calculs relatifs au maillage préalablement à l'utilisation du solver

        Parameters
        ----------
        method: str = "CENTRE"
        Méthode pour la simulation en convection (CENTRE ou UPWIND)

        alpha: float = 0.75)
        Facteur de relaxation

        Returns
        -------
        (u, v), (PHI_EXu, PHI_EXv): Tuple[Tuple[numpy.ndarray, numpy.ndarray], Tuple[numpy.ndarray, numpy.ndarray]]
        (u, v) -> Solution numérique en x et y
        (PHI_EXu, PHI_EXv) -> Solution analytique en x et y

        """
        # Chercher les différentes variables
        mesh = self.get_mesh()          # Maillage utilisé
        case = self.get_case()          # cas en cours
        centroids = self.centroids      # Chercher les centres des éléments
        volumes = self.get_volumes()    # surfaces des éléments
        bcdata = self.get_bcdata()      # Conditions limites
        P = self.get_P()                # Paramètre modifié

        # Initialisation des matrices et des vecteurs pour u et v
        NELEM = self.mesh_obj.get_number_of_elements()
        U_AVG=np.zeros((mesh.get_number_of_faces()-mesh.get_number_of_boundary_faces())) #Vitesse débitante moyennée sur les faces internes uniquement !
        # Matrice/vecteurs pour l'itération i+1
        Bu0, Bv0 = np.zeros(NELEM), np.zeros(NELEM)

        # Test de convergence respecté
        convergence = False

        # Variables locales
        rho, mu = case.get_physical_properties()
        dpdx, dpdy = case.get_sources()
        analytical_function = self.get_analytical_function()

        solver_GLS = GLS.GradientLeastSquares(mesh, bcdata, centroids)
        solver_GLS.set_P(P)

        # Calcule les termes sources reliés au gradient de pression
        SGPXp, SGPYp = compute_source(P, dpdx, dpdy, volumes, centroids)

        # Valeur des vitesses à it = 0 (valeur posée)
        u, v = np.zeros(NELEM), np.zeros(NELEM)

        # Boucle pour l'algorithme principal de résolution non-linéaire
        it = 0
        while convergence is False:
            # Matrices/vecteurs pour l'itération i
            Au = np.zeros((NELEM, NELEM))
            Bu, Bv = np.zeros(NELEM), np.zeros(NELEM)
            PHIu, PHIv = np.zeros(NELEM), np.zeros(NELEM)
            PHI_EXu, PHI_EXv = np.zeros(NELEM), np.zeros(NELEM)
            GRADu, GRADv = np.zeros((NELEM, 2)), np.zeros((NELEM, 2))

            # Boucle pour le cross-diffusion
            for i in range(2):

                # Parcours les faces internes et remplis la matrice A et le vecteur B
                for i_face in range(mesh.get_number_of_boundary_faces(), mesh.get_number_of_faces()):
                    # Listes des noeuds et des éléments reliés à la face
                    nodes = mesh.get_face_to_nodes(i_face)
                    left_elem, right_elem = mesh.get_face_to_elements(i_face)

                    # Calcule des grandeurs et vecteurs géométriques pertinents
                    dA, dKSI, n, eKSI, eETA = \
                        compute_lengths_and_unit_vectors(pta=mesh.get_node_to_xycoord(nodes[0]),
                                                         ptb=mesh.get_node_to_xycoord(nodes[1]),
                                                         ptA=centroids[right_elem],
                                                         ptP=centroids[left_elem])

                    # Calcule du flux au centre de la face (moyenne simple : (Fxp+FxA)/2 = rho(uxp+uxA)/2)
                    Fx, Fy = rho*0.5*(u[left_elem] + u[right_elem]), rho*0.5*(v[left_elem] + v[right_elem])
                    F = np.dot([Fx, Fy], n) * dA  # Débit massique qui traverse la face
                    
                    #LAP 5: RHIE-CHOW
                    U_AVG[i_face-mesh.get_number_of_boundary_faces()]=np.dot([Fx, Fy], n) #Vitesse débitante moyennée

                    # Calcule les projections de vecteurs unitaires
                    PNKSI = np.dot(n, eKSI)       # Projection de n sur ξ
                    PKSIETA = np.dot(eKSI, eETA)  # Projection de ξ sur η

                    D = (1/PNKSI) * mu * (dA / dKSI)  # Direct gradient term

                    # Calcule le terme correction de cross-diffusion
                    Sdc_x = -mu * (PKSIETA/PNKSI) * 0.5*np.dot((GRADu[right_elem] + GRADu[left_elem]), eETA) * dA
                    Sdc_y = -mu * (PKSIETA/PNKSI) * 0.5*np.dot((GRADv[right_elem] + GRADv[left_elem]), eETA) * dA

                    # Ajoute la contribution de la convection à la matrice Au
                    if method == "CENTRE":
                        # Remplissage de la matrice et du vecteur
                        Au[left_elem, left_elem]   +=  D + 0.5*F
                        Au[right_elem, right_elem] +=  D - 0.5*F
                        Au[left_elem, right_elem]  += -D + 0.5*F
                        Au[right_elem, left_elem]  += -D - 0.5*F
                    elif method == "UPWIND":
                        # Remplissage de la matrice et du vecteur
                        Au[left_elem, left_elem]   +=  (D + max(F, 0))
                        Au[right_elem, right_elem] +=  (D + max(0, -F))
                        Au[left_elem, right_elem]  += (-D - max(0, -F))
                        Au[right_elem, left_elem]  += (-D - max(F, 0))

                    else:
                        print("La méthode choisie n'est pas convenable, veuillez choisir Centre ou Upwind")
                        sys.exit()

                    Bu[left_elem]  += Sdc_x
                    Bu[right_elem] -= Sdc_x
                    Bv[left_elem]  += Sdc_y
                    Bv[right_elem] -= Sdc_y


                # Parcours les faces sur les conditions frontières et remplis la matrice A et le vecteur B
                for i_face in range(mesh.get_number_of_boundary_faces()):
                    # Détermine le numéro de la frontière et les conditions associées
                    tag = mesh.get_boundary_face_to_tag(i_face)
                    bc_type, (bc_x, bc_y) = bcdata[tag]

                    # Listes des noeuds et des éléments reliés à la face
                    nodes = mesh.get_face_to_nodes(i_face)
                    element = mesh.get_face_to_elements(i_face)[0]

                    # Détermine la position du centre de la face
                    pt0, pt1 = mesh.get_node_to_xycoord(nodes[0]), mesh.get_node_to_xycoord(nodes[1])
                    xa, ya = 0.5*(pt0[0] + pt1[0]), 0.5*(pt0[1] + pt1[1])

                    dA, dKSI, n, eKSI, eETA = \
                        compute_lengths_and_unit_vectors(pta=mesh.get_node_to_xycoord(nodes[0]),
                                                         ptb=mesh.get_node_to_xycoord(nodes[1]),
                                                         ptA=(xa, ya),
                                                         ptP=centroids[element])

                    # Calcule les projections de vecteurs unitaires
                    dETA = dA  # Équivalent, mais noté pour éviter la confusion
                    PNKSI = np.dot(n, eKSI)  # Projection de n sur ξ
                    PKSIETA = np.dot(eKSI, eETA)  # Projection de ξ sur η


                    if bc_type == "DIRICHLET":
                        # Détermine le terme du gradient direct et le flux massique au centre de la frontière
                        D = (1/PNKSI) * mu * (dA/dKSI)
                        F = np.dot(rho*[bc_x(xa, ya, P), bc_y(xa, ya, P)], n) * dA

                        # Calcule du terme de cross-diffusion selon les phi aux noeuds de l'arête en x et y
                        phi0, phi1 = bc_x(pt0[0], pt0[1], P), bc_x(pt1[0], pt1[1], P)
                        Sdc_x = -mu * (PKSIETA/PNKSI) * rho * (phi1 - phi0)/dETA * dA

                        phi0, phi1 = bc_y(pt0[0], pt0[1], P), bc_y(pt1[0], pt1[1], P)
                        Sdc_y = -mu * (PKSIETA/PNKSI) * rho * (phi1 - phi0)/dETA * dA

                        if method == "CENTRE":
                            Au[element, element] += D
                            Bu[element] += (D - F) * bc_x(xa, ya, P) + Sdc_x
                            Bv[element] += (D - F) * bc_y(xa, ya, P) + Sdc_y
                        elif method == "UPWIND":
                            Au[element, element] += D + max(F, 0)
                            Bu[element] += (D + max(0, -F)) * bc_x(xa, ya, P) + Sdc_x
                            Bv[element] += (D + max(0, -F)) * bc_y(xa, ya, P) + Sdc_y
                        else:
                            print("La méthode choisie n'est pas convenable, veuillez choisir CENTRE ou UPWIND")
                            sys.exit()

                    elif bc_type == "NEUMANN" or "NEUMANNP":
                        phix = u[element] - bc_x(xa, ya, P) * PNKSI * dKSI  # ui à la CF
                        phiy = v[element] - bc_y(xa, ya, P) * PNKSI * dKSI  # vi à la CF

                        Fx, Fy = rho*phix, rho*phiy
                        F = np.dot(np.array([Fx, Fy]), n) * dA  # Débit massique qui traverse la face
                        
                        Au[element, element] += F
                        Bu[element] += (mu * dA - F * PNKSI * dKSI) * bc_x(xa, ya, P)
                        Bv[element] += (mu * dA - F * PNKSI * dKSI) * bc_y(xa, ya, P)


                # Ajout de la contribution du terme source sur les éléments et calcul de la solution analytique
                for i_elem in range(mesh.get_number_of_elements()):
                    Bu[i_elem] += SGPXp[i_elem]
                    Bv[i_elem] += SGPYp[i_elem]
                    PHI_EXu[i_elem] = analytical_function[0](centroids[i_elem][0], centroids[i_elem][1], P)
                    PHI_EXv[i_elem] = analytical_function[1](centroids[i_elem][0], centroids[i_elem][1], P)

                # Av = Au puisque les conditions sont de même type
                Av = Au

                # Résolution pour l'itération
                PHIu = linsolve.spsolve(sps.csr_matrix(Au, dtype=np.float64), Bu)
                PHIv = linsolve.spsolve(sps.csr_matrix(Av, dtype=np.float64), Bv)

                # Calcule des gradients pour le cross-diffusion
                GRADu, GRADv = solver_GLS.solve(PHIu, PHIv)

            # Vérification des normes de résidu pour l'itération précédante
            Ru = np.linalg.norm(np.dot(Au, u) - Bu0)
            Rv = np.linalg.norm(np.dot(Av, v) - Bv0)

            #print(f"Itération {it} : |Ru|={Ru:.7f}.  |Rv|={Rv:.7f}")
            tol = 1e-7
            if it != 0 and Ru < tol and Rv < tol:
                # Solution de l'itération précédence est bonne
                convergence = True
            else:
                # Sous-relaxation itérative avec la solution de l'itération précédante (u et v)
                u = alpha * PHIu + (1 - alpha) * u
                v = alpha * PHIv + (1 - alpha) * v

                # Store les vecteurs B pour le calcule des résidus
                Bu0 = Bu
                Bv0 = Bv

            it += 1
            
        U_RC=self.Rhie_Chow(Au, SGPXp, SGPYp, u, v,U_AVG) #Vitesse débitante avec Rhie- chow + tests
        return (u, v), (PHI_EXu, PHI_EXv)
    
    def Rhie_Chow(self,A,SGPXp,SGPYp,u,v,U_AVG):
        """
        Calculer la vitesse débitante normale à travers les faces par la méthode 
        de rhie chow et vérifie la différence avec la vitesse débitante moyennée
    
        Parameters
        ----------
        A: ndarray [float]
            Matrice de la dernière itération
    
        SGXp: function
            gradient de pression selon x
        
        SGYp: function
            gradient de pression selon y
            
        u: ndarray [float]
            Solution vitesse selon x
        
        v: ndarray [float]
            Solution vitesse selon y
            
        U_AVG: ndarray [float]
            Vitesse débitante calculée avec la moyenne de deux élements voisins
    
        Returns
        -------
        uf: ndarray [float]
            Vitesse débitante calculée avec la méthode Rhie Chow
            
        """
        # Chercher les différentes variables
        mesh = self.get_mesh()          # Maillage utilisé
        centroids = self.centroids      # Chercher les centres des éléments
        volumes = self.get_volumes()    # surfaces des éléments
        bcdata = self.get_bcdata()      # Conditions limites
        P = self.get_P()                # Paramètre modifié
        DAU=np.diag(A) #Diagonale de la matrice, on suppose u et v ont les mêmes CL
       
        #Initialisation des valeurs
        uf=np.zeros(mesh.get_number_of_faces())

        # Parcours les faces internes et remplis la matrice A et le vecteur B
        for i_face in range(mesh.get_number_of_boundary_faces(), mesh.get_number_of_faces()):
            # Listes des noeuds et des éléments reliés à la face
            nodes = mesh.get_face_to_nodes(i_face)
            left_elem, right_elem = mesh.get_face_to_elements(i_face)
            
            # Calcule des grandeurs et vecteurs géométriques pertinents
            dA, dKSI, n, eKSI, eETA = \
                compute_lengths_and_unit_vectors(pta=mesh.get_node_to_xycoord(nodes[0]),
                                                 ptb=mesh.get_node_to_xycoord(nodes[1]),
                                                 ptA=centroids[right_elem],
                                                 ptP=centroids[left_elem])
                        
            #Calcul de la vitesse débitante
            DVP=volumes[left_elem]/DAU[left_elem] #DVP/AP
            DVA=volumes[right_elem]/DAU[right_elem] #DVA/AA
            
            #Pression P= -2 param_P * x 
            Pp=SGPXp[left_elem]*centroids[left_elem,0] # au centre de l elem P
            Pa=SGPXp[right_elem]*centroids[right_elem,0] # au centre de l elem P
            
            #Gradient
            gradPP=np.array((SGPXp[left_elem],SGPYp[left_elem]))#gradient de pression elem gauche
            gradPA=np.array((SGPXp[right_elem],SGPYp[right_elem]))#gradient de pression elem droite
            
            #Vitesse
            Up=np.array((u[left_elem],v[left_elem]))
            Ua=np.array((u[right_elem],v[right_elem]))
            
            #Composante de uf (momentaire pour la lisibilité)
            uf1=np.dot(((Up+Ua)/2.0),n) 
            uf2=0.5*(DVP+DVA)*((Pp-Pa)/dKSI)
            uf3=0.5*np.dot((gradPP*DVP+gradPA*DVA),eKSI)
            
            uf[i_face]=uf1+uf2+uf3 #Vitesse débitante par rhie chow

        # Parcours les faces sur les conditions frontières et remplis la matrice A et le vecteur B
        for i_face in range(mesh.get_number_of_boundary_faces()):
            # Détermine le numéro de la frontière et les conditions associées
            tag = mesh.get_boundary_face_to_tag(i_face)
            bc_type, (bc_x, bc_y) = bcdata[tag]

            # Listes des noeuds et des éléments reliés à la face
            nodes = mesh.get_face_to_nodes(i_face)
            element = mesh.get_face_to_elements(i_face)[0]

            # Détermine la position du centre de la face
            pt0, pt1 = mesh.get_node_to_xycoord(nodes[0]), mesh.get_node_to_xycoord(nodes[1])
            xa, ya = 0.5*(pt0[0] + pt1[0]), 0.5*(pt0[1] + pt1[1])

            dA, dKSI, n, eKSI, eETA = \
                compute_lengths_and_unit_vectors(pta=mesh.get_node_to_xycoord(nodes[0]),
                                                 ptb=mesh.get_node_to_xycoord(nodes[1]),
                                                 ptA=(xa, ya),
                                                 ptP=centroids[element])

            if bc_type == "DIRICHLET":

                uf[i_face]=bc_x(xa, ya, P) * n[0] + bc_y(xa, ya, P) * n[1] #Calcul avec les valeurs imposées évaluée au centre de la face frontière
                    

            elif bc_type == "NEUMANN":

                #Calcul de la vitesse débitante
                DVP=volumes[element]/DAU[element] #DVP/AP
                
                #Pression P= -2 param_P * x 
                Pp=SGPXp[left_elem]*centroids[left_elem,0]# au centre de l elem P
                Ps=-2*P*xa # au centre de la face frontière
                
                #Gradient
                gradPP=np.array((SGPXp[element],SGPYp[element]))#gradient de pression elem gauche
                
                #Composante de uf (momentaire pour la lisibilité)
                uf1=np.dot([u[element],v[element]],n) 
                uf2=DVP*((Pp-Ps)/dKSI)
                uf3=DVP*np.dot(gradPP,eKSI)
                
                uf[i_face]=uf1+uf2+uf3 #Vitesse débitante par rhie chow
                
        diff,test_result=compare_RC(uf, U_AVG)
        
        if test_result is True:
            print("Test réussi")
            print("La différence maximale sur les faces internes est:")
            print(max(diff))
        else:
            print("Test échoué, veuillez vérifier vos données")
            
        return uf
        
