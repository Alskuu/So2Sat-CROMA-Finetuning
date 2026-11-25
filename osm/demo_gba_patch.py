"""
Bâtiments GlobalBuildingAtlas (GBA, TU Munich) sur un patch So2Sat de 320 m (monde entier).

  1. emprise()        : d'un point central à un carré de 320 m (repris de demo_osm_patch.py)
  2. nom_tuile()      : dans quelle tuile GBA de 5° x 5° tombe le patch
  3. batiments_gba()  : requête Earth Engine, on ne récupère que les bâtiments du carré
  +  resumer()        : un aperçu de ce qu'on obtient

Source : https://github.com/zhu-xlab/GlobalBuildingAtlas  (polygones : licence ODbL)
GBA fusionne plusieurs sources de polygones (OSM, Microsoft, Google, ses propres détections
sur images PlanetScope) et estime une hauteur pour chaque bâtiment.

Pourquoi Earth Engine : GBA est distribué en GeoJSON par tuile de 5° (environ 1 Go pour
Nairobi, 10 Go pour Paris). Le catalogue communautaire d'Earth Engine (sat-io) contient
les mêmes tuiles : on demande au serveur uniquement les bâtiments du patch.

Dépendances : pip install earthengine-api shapely
"""
import math
from collections import Counter

import ee
from shapely.geometry import box, shape

from connexion_ee import initialiser_ee
from demo_osm_patch import EXEMPLES, emprise

DOSSIER_EE = "projects/sat-io/open-datasets/GLOBAL_BUILDING_ATLAS"
TAILLE_TUILE_DEG = 5


def nom_tuile(lat, lon):
    """Étape 2 : nom de la tuile GBA qui contient le point.

    Règle GBA : {e/w}{lon_min}_{n/s}{lat_max}_{e/w}{lon_max}_{n/s}{lat_min}
    Exemple : Paris (lat 48,9 ; lon 2,3) -> e000_n50_e005_n45
    """
    lon_min = math.floor(lon / TAILLE_TUILE_DEG) * TAILLE_TUILE_DEG
    lat_min = math.floor(lat / TAILLE_TUILE_DEG) * TAILLE_TUILE_DEG

    def texte_lon(x):
        return f"{'e' if x >= 0 else 'w'}{abs(x):03d}"

    def texte_lat(y):
        return f"{'n' if y >= 0 else 's'}{abs(y):02d}"

    return (f"{texte_lon(lon_min)}_{texte_lat(lat_min + TAILLE_TUILE_DEG)}_"
            f"{texte_lon(lon_min + TAILLE_TUILE_DEG)}_{texte_lat(lat_min)}")


def tuiles_du_patch(bbox):
    """Les tuiles des 4 coins : en général une seule."""
    s, w, n, e = bbox
    return sorted({nom_tuile(lat, lon) for lat in (s, n) for lon in (w, e)})


def batiments_gba(bbox):
    """Étape 3 : bâtiments GBA du carré, au format commun aux sources.

    Chaque bâtiment est un dictionnaire :
      geometrie  : polygone shapely en lon/lat, découpé au carré du patch
      hauteur_m  : hauteur estimée par GBA, None si absente
      variance   : variance de l'estimation de hauteur (propriété "var" de GBA)
      origine    : d'où vient le polygone (osm, microsoft, google, ou détection GBA)
    """
    initialiser_ee()
    s, w, n, e = bbox
    rectangle = ee.Geometry.Rectangle([w, s, e, n])
    carre = box(w, s, e, n)

    batiments = []
    for tuile in tuiles_du_patch(bbox):
        collection = ee.FeatureCollection(f"{DOSSIER_EE}/{tuile}").filterBounds(rectangle)
        for objet in collection.getInfo()["features"]:          # getInfo() : rapatrie le résultat en Python
            polygone = shape(objet["geometry"])
            if not polygone.is_valid:
                polygone = polygone.buffer(0)
            if not polygone.intersects(carre):
                continue
            proprietes = objet["properties"]
            hauteur = proprietes.get("height")
            batiments.append({
                "geometrie": polygone.intersection(carre),
                "hauteur_m": hauteur if hauteur is not None and hauteur > 0 else None,
                "variance": proprietes.get("var"),
                "origine": proprietes.get("source"),
            })
    return batiments


def resumer(batiments):
    """Aperçu : nombre, origine des polygones, hauteurs."""
    hauteurs = sorted(b["hauteur_m"] for b in batiments if b["hauteur_m"] is not None)
    return {
        "batiments": len(batiments),
        "origine_des_polygones": Counter(b["origine"] for b in batiments).most_common(),
        "batiments_avec_hauteur": len(hauteurs),
        "hauteur_mediane_m": round(hauteurs[len(hauteurs) // 2], 1) if hauteurs else None,
        "hauteurs_arrondies": Counter(round(h) for h in hauteurs).most_common(5),
    }


if __name__ == "__main__":
    for nom, lat, lon in EXEMPLES:
        print(f"\n=== {nom} ===")
        bbox = emprise(lat, lon)
        print("Tuile(s) GBA :", tuiles_du_patch(bbox))
        for cle, valeur in resumer(batiments_gba(bbox)).items():
            print(f"  {cle:24s} {valeur}")
