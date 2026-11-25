"""
Dynamic World sur un patch So2Sat de 320 m : le patch a-t-il changé depuis l'image So2Sat ?

  1. emprise()             : d'un point central à un carré de 320 m (repris de demo_osm_patch.py)
  2. classes_annee()       : la classe d'occupation du sol de chaque pixel de 10 m, pour une année
  3. part_bati()           : le % du patch classé « bâti » une année donnée
     probabilite_bati()    : la probabilité moyenne de « bâti » (indicateur continu, voir plus bas)
  4. evolution()           : ce % année par année, de 2016 à 2025
  5. changement()          : ce qui a changé entre deux années, pixel par pixel
  +  carte_changement()    : la grille des changements du bâti, pour la figure

Source : https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1
(licence CC BY 4.0)

Dynamic World classe CHAQUE image Sentinel-2 (depuis mi-2015) en 9 classes, à 10 m :
c'est le même capteur et la même résolution que So2Sat. Un patch So2Sat fait donc
exactement 32 x 32 pixels Dynamic World.

Pourquoi c'est utile : les images So2Sat datent d'environ 2017, les sources de bâtiments
de 2019 à aujourd'hui. Si le patch a beaucoup changé entre les deux dates, le texte généré
à partir des données récentes ne décrit plus ce qu'on voit sur l'image So2Sat.

Attention : la classe « bâti » de Dynamic World couvre les bâtiments MAIS aussi les routes
et les surfaces imperméables. On s'en sert pour mesurer des évolutions, pas pour comparer
son niveau aux surfaces de bâtiments des autres sources.

Deux limites, constatées sur nos patches :
  - en ville dense, la classe dominante est « bâti » sur 100 % des pixels chaque année : elle ne
    peut plus augmenter, donc elle ne voit que les changements en bordure de ville ;
  - la probabilité de « bâti » est continue, mais elle dérive d'une année à l'autre sans changement
    réel (Paris : de 0,51 à 0,69 entre 2016 et 2025) : il faut un patch stable comme référence de bruit.

Dépendances : pip install earthengine-api numpy
"""
from functools import lru_cache

import ee
import numpy as np

from connexion_ee import initialiser_ee
from demo_osm_patch import EXEMPLES, emprise

COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
CLASSES = ["eau", "arbres", "herbe", "végétation inondée", "cultures",
           "arbustes", "bâti", "sol nu", "neige"]           # dans l'ordre des numéros 0 à 8
BATI = CLASSES.index("bâti")                                # = 6
SANS_DONNEE = 255
PIXELS = 32                                                 # 320 m / 10 m
ANNEES = list(range(2016, 2026))


@lru_cache(maxsize=None)            # chaque (patch, année) n'est demandé qu'une fois à Earth Engine
def classes_annee(bbox, annee):
    """Étape 2 : grille 32 x 32 des classes de l'année, et nombre d'images utilisées.

    Une année contient des dizaines d'images (et des nuages) : pour chaque pixel, on garde
    la classe la plus fréquente sur l'année (le « mode »). La grille est alignée sur le
    carré du patch, ligne 0 = bord nord. SANS_DONNEE = aucune image valide pour ce pixel.
    """
    initialiser_ee()
    s, w, n, e = bbox
    images = (ee.ImageCollection(COLLECTION)
              .filterBounds(ee.Geometry.Rectangle([w, s, e, n]))
              .filterDate(f"{annee}-01-01", f"{annee + 1}-01-01"))
    classe_dominante = images.select("label").reduce(ee.Reducer.mode()).unmask(SANS_DONNEE).toByte()

    grille = ee.data.computePixels({
        "expression": classe_dominante.rename("classe"),
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": PIXELS, "height": PIXELS},
            "affineTransform": {"scaleX": (e - w) / PIXELS, "shearX": 0, "translateX": w,
                                "shearY": 0, "scaleY": -(n - s) / PIXELS, "translateY": n},
            "crsCode": "EPSG:4326",
        },
    })
    return np.array(grille["classe"]), images.size().getInfo()


def part_bati(bbox, annee):
    """Étape 3 : % des pixels (avec donnée) dont la classe dominante de l'année est « bâti »."""
    grille, _ = classes_annee(bbox, annee)
    valides = grille != SANS_DONNEE
    return round(100 * float(np.mean(grille[valides] == BATI)), 1) if valides.any() else None


@lru_cache(maxsize=None)
def probabilite_bati(bbox, annee):
    """Étape 3 bis : probabilité de « bâti » donnée par le modèle, en moyenne sur le patch.

    Pour chaque pixel on prend la médiane de l'année (robuste aux nuages), puis la moyenne du patch.
    """
    initialiser_ee()
    s, w, n, e = bbox
    rectangle = ee.Geometry.Rectangle([w, s, e, n])
    images = (ee.ImageCollection(COLLECTION).filterBounds(rectangle)
              .filterDate(f"{annee}-01-01", f"{annee + 1}-01-01"))
    valeur = images.select("built").median().reduceRegion(ee.Reducer.mean(), rectangle, 10).get("built")
    return round(valeur.getInfo(), 2)


def evolution(bbox, annees=ANNEES):
    """Étape 4 : % bâti, probabilité de bâti et nombre d'images, année par année."""
    return [{"annee": a, "bati_pct": part_bati(bbox, a), "proba_bati": probabilite_bati(bbox, a),
             "nb_images": classes_annee(bbox, a)[1]}
            for a in annees]


def changement(bbox, annee_a, annee_b):
    """Étape 5 : ce qui a changé entre deux années, en % des pixels du patch."""
    a, _ = classes_annee(bbox, annee_a)
    b, _ = classes_annee(bbox, annee_b)
    valides = (a != SANS_DONNEE) & (b != SANS_DONNEE)
    a, b = a[valides], b[valides]
    def pct(masque):
        return round(100 * float(np.mean(masque)), 1)

    return {
        "ecart_bati_pts": round(pct(b == BATI) - pct(a == BATI), 1),
        "devenus_batis_pct": pct((a != BATI) & (b == BATI)),
        "plus_batis_pct": pct((a == BATI) & (b != BATI)),
        "pixels_changes_pct": pct(a != b),                            # toutes classes confondues
    }


def carte_changement(bbox, annee_a, annee_b):
    """Grille 32 x 32 : 0 = jamais bâti, 1 = bâti les deux années, 2 = devenu bâti, 3 = n'est plus bâti."""
    a, _ = classes_annee(bbox, annee_a)
    b, _ = classes_annee(bbox, annee_b)
    carte = np.zeros_like(a)
    carte[(a == BATI) & (b == BATI)] = 1
    carte[(a != BATI) & (b == BATI)] = 2
    carte[(a == BATI) & (b != BATI)] = 3
    return carte


if __name__ == "__main__":
    for nom, lat, lon in EXEMPLES:
        print(f"\n=== {nom} ===")
        bbox = emprise(lat, lon)
        for ligne in evolution(bbox):
            print(f"  {ligne['annee']}  bâti {ligne['bati_pct']:5} %   probabilité {ligne['proba_bati']}"
                  f"   ({ligne['nb_images']} images)")
        print("  changement 2017 -> 2025 :", changement(bbox, 2017, 2025))
