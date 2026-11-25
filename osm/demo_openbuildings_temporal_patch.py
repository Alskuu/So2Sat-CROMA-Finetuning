"""
Google Open Buildings 2.5D Temporal sur un patch So2Sat de 320 m (Afrique, Asie du Sud et du
Sud-Est, Amérique latine ; pas l'Europe).

  1. emprise()                    : d'un point central à un carré de 320 m (repris de demo_osm_patch.py)
  2. image_annee()                : l'image Earth Engine d'une année (2016 à 2023)
  3. indicateurs_openbuildings()  : surface bâtie, nombre de bâtiments, classes de hauteur du carré
  4. carte_hauteurs()             : la grille des classes de hauteur, pour la figure
  5. evolution()                  : les indicateurs année par année (le « Temporal » du jeu de données)

Source : https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_Research_open-buildings-temporal_v1
(licence CC BY 4.0 ou ODbL, au choix)

Ce n'est PAS un jeu de polygones : ce sont des rasters annuels prédits à partir de Sentinel-2,
en pixels de 0,5 m (résolution effective d'environ 4 m), avec trois bandes :
  - building_presence          : confiance (0 à 1) qu'un pixel soit un bâtiment, non calibrée
  - building_height            : hauteur en mètres (plafonnée à 100 m)
  - building_fractional_count  : fraction de bâtiment portée par le pixel ; la somme sur une
                                 zone donne le nombre de bâtiments
Les indicateurs sont donc calculés sur les pixels, directement sur les serveurs d'Earth Engine.

Dépendances : pip install earthengine-api numpy
"""
import ee
import numpy as np

from connexion_ee import initialiser_ee
from demo_osm_patch import COTE_M, EXEMPLES, emprise

COLLECTION = "GOOGLE/Research/open-buildings-temporal/v1"
ANNEES = list(range(2016, 2024))

# Un pixel est compté comme bâti si la confiance dépasse ce seuil.
# La confiance n'est pas calibrée : 0,5 est un choix raisonnable mais arbitraire, à discuter.
SEUIL_PRESENCE = 0.5

SEUILS_HAUTEUR_M = [10, 25]   # bas / moyen / haut, comme pour les autres sources (classes LCZ)
ECHELLE_M = 1                 # résolution des calculs : 1 m suffit pour une résolution effective de 4 m
PIXELS_FIGURE = 160           # la carte de la figure fait 160 x 160 pixels, soit 2 m par pixel


def rectangle(bbox):
    s, w, n, e = bbox
    return ee.Geometry.Rectangle([w, s, e, n])


def couvre(bbox):
    """Vrai si le jeu de données a des images sur ce patch (faux en Europe, par exemple)."""
    initialiser_ee()
    return ee.ImageCollection(COLLECTION).filterBounds(rectangle(bbox)).size().getInfo() > 0


def image_annee(annee):
    """Étape 2 : l'image d'une année. Le jeu est découpé en tuiles : on les assemble (mosaic)."""
    return (ee.ImageCollection(COLLECTION)
            .filter(ee.Filter.calendarRange(annee, annee, "year"))
            .mosaic())


def indicateurs_openbuildings(bbox, annee=2023):
    """Étape 3 : les mêmes indicateurs que pour les sources en polygones.

    Toutes les surfaces sont des parts de pixels : la moyenne d'un masque 0/1 sur le carré
    donne directement la part du carré couverte par ce masque.
    """
    initialiser_ee()
    image = image_annee(annee)
    bati = image.select("building_presence").gt(SEUIL_PRESENCE)
    hauteur = image.select("building_height")

    # Masques 0/1, calculés seulement sur les pixels bâtis
    couches = ee.Image.cat([
        bati.rename("bati"),
        bati.And(hauteur.lt(SEUILS_HAUTEUR_M[0])).rename("bas"),
        bati.And(hauteur.gte(SEUILS_HAUTEUR_M[0])).And(hauteur.lt(SEUILS_HAUTEUR_M[1])).rename("moyen"),
        bati.And(hauteur.gte(SEUILS_HAUTEUR_M[1])).rename("haut"),
    ])
    parts = couches.reduceRegion(ee.Reducer.mean(), rectangle(bbox), ECHELLE_M, maxPixels=1e8).getInfo()

    # Hauteur médiane des pixels bâtis
    mediane = (hauteur.updateMask(bati)
               .reduceRegion(ee.Reducer.median(), rectangle(bbox), ECHELLE_M, maxPixels=1e8)
               .getInfo()["building_height"])

    # Nombre de bâtiments : somme des fractions, à la résolution native de 0,5 m
    nombre = (image.select("building_fractional_count")
              .reduceRegion(ee.Reducer.sum(), rectangle(bbox), 0.5, maxPixels=1e8)
              .getInfo()["building_fractional_count"])

    surface_batie_m2 = parts["bati"] * COTE_M ** 2
    return {
        "nb_batiments": round(nombre),
        "surface_batie_pct": round(100 * parts["bati"], 1),
        "surface_moyenne_m2": round(surface_batie_m2 / nombre) if nombre else None,
        "hauteur_connue_pct_surface": 100.0 if parts["bati"] else None,   # chaque pixel a une hauteur
        "hauteur_mediane_m": round(mediane, 1) if mediane is not None else None,
        **{f"{classe}_pct": round(100 * parts[classe] / parts["bati"], 1) if parts["bati"] else None
           for classe in ("bas", "moyen", "haut")},
    }


def carte_hauteurs(bbox, annee=2023):
    """Étape 4 : grille PIXELS_FIGURE x PIXELS_FIGURE des classes de hauteur.

    0 = non bâti, 1 = bas, 2 = moyen, 3 = haut. La grille est alignée exactement sur le carré
    du patch (ligne 0 = bord nord), pour la superposer aux autres sources dans la figure.
    """
    initialiser_ee()
    s, w, n, e = bbox
    image = image_annee(annee)
    bati = image.select("building_presence").gt(SEUIL_PRESENCE)
    hauteur = image.select("building_height")
    classes = (bati
               .add(hauteur.gte(SEUILS_HAUTEUR_M[0]))
               .add(hauteur.gte(SEUILS_HAUTEUR_M[1]))
               .multiply(bati)                 # 0 si non bâti
               .toByte()
               .rename("classe"))
    tableau = ee.data.computePixels({
        "expression": classes,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": PIXELS_FIGURE, "height": PIXELS_FIGURE},
            "affineTransform": {"scaleX": (e - w) / PIXELS_FIGURE, "shearX": 0, "translateX": w,
                                "shearY": 0, "scaleY": -(n - s) / PIXELS_FIGURE, "translateY": n},
            "crsCode": "EPSG:4326",
        },
    })
    return np.array(tableau["classe"])


def evolution(bbox):
    """Étape 5 : surface bâtie et nombre de bâtiments pour chaque année de 2016 à 2023."""
    lignes = []
    for annee in ANNEES:
        ind = indicateurs_openbuildings(bbox, annee)
        lignes.append({"annee": annee, "surface_batie_pct": ind["surface_batie_pct"],
                       "nb_batiments": ind["nb_batiments"], "hauteur_mediane_m": ind["hauteur_mediane_m"]})
    return lignes


if __name__ == "__main__":
    for nom, lat, lon in EXEMPLES:
        print(f"\n=== {nom} ===")
        bbox = emprise(lat, lon)
        if not couvre(bbox):
            print("  hors couverture Open Buildings (pas de données en Europe)")
            continue
        for cle, valeur in indicateurs_openbuildings(bbox).items():
            print(f"  {cle:28s} {valeur}")
        print("  évolution 2016-2023 (% bâti) :",
              [(l["annee"], l["surface_batie_pct"]) for l in evolution(bbox)])
