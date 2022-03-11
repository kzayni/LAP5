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

#%% Classe Case
class Case:
    """
    Contient les données reliées au cas étudié

        rho: int, mu: int, flow_velocities: Tuple[function, function], source_terms: Tuple[function, function], domain: List[int]) -> None:

    Parameters
    ---------
    rho: float
    Masse volumique du fluide

    mu: float
    Viscosité dynamique du fluide

    source_terms: Tuple[function, function]
    Fonctions pour évaluer les termes sources

    domain: List[float]
    Domaine de la géométrie étudiée


    Attributes
    ---------
    rho: float
    Masse volumique du fluide

    mu: float
    Viscosité dynamique du fluide

    source_terms: Tuple[function, function]
    Fonctions pour évaluer les termes sources

    domain: List[float]
    Domaine de la géométrie étudiée

    """
    
    def __init__(self, rho, mu,  source_terms, domain):
        self.rho = rho
        self.mu = mu
        self.source_terms = source_terms
        self.domain = domain
    
    #Accesseurs
    def get_sources(self):
        return self.source_terms

    def get_physical_properties(self):
        return self.rho, self.mu

    def set_sources(self, source_terms):
        self.source_terms = source_terms

    def get_domain(self):
        return self.domain