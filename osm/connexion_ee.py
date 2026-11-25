"""
Connexion à Google Earth Engine, partagée par demo_gba_patch.py et demo_openbuildings_temporal_patch.py.

Earth Engine demande :
  - un compte Google autorisé (une seule fois : `earthengine authenticate` dans un terminal) ;
  - un projet Google Cloud où l'API Earth Engine est activée (gratuit pour la recherche).

Le projet est lu dans la variable d'environnement EE_PROJECT. Si elle n'existe pas,
Earth Engine prend le projet par défaut enregistré lors de l'authentification.

Dépendance : pip install earthengine-api
"""
import os

import ee

_deja_initialise = False


def initialiser_ee():
    """Ouvre la session Earth Engine (une seule fois par exécution)."""
    global _deja_initialise
    if _deja_initialise:
        return
    projet = os.environ.get("EE_PROJECT")      # None -> projet par défaut de l'authentification
    try:
        ee.Initialize(project=projet)
    except Exception:                          # identifiants absents ou expirés
        print("Connexion Earth Engine impossible : ouverture de l'authentification Google ...")
        ee.Authenticate()
        ee.Initialize(project=projet)
    _deja_initialise = True
