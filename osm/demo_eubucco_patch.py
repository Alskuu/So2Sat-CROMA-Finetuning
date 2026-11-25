"""
Bâtiments EUBUCCO v0.2 sur un patch So2Sat de 320 m (Europe seulement).

  1. emprise()            : d'un point central à un carré de 320 m (repris de demo_osm_patch.py)
  2. region_nuts()        : quel fichier EUBUCCO ouvrir (un fichier Parquet par région NUTS2)
  3. batiments_eubucco()  : requête DuckDB sur le fichier distant, découpée au carré du patch
  +  resumer()            : un aperçu de ce qu'on obtient

Source : https://docs.eubucco.com  (licence ODbL)
Les fichiers sont lus directement sur le serveur S3 d'EUBUCCO, sans compte ni téléchargement complet :
DuckDB ne lit que les morceaux du fichier dont la boîte englobante touche le patch.

Dépendances : pip install duckdb pandas shapely
"""
from collections import Counter

import duckdb
import pandas as pd
import shapely

from demo_osm_patch import EXEMPLES, emprise

# Un fichier par région NUTS2 : s3://eubucco/v0.2/buildings/parquet/nuts_id=<ID>/<ID>.parquet
URL_EUBUCCO = "s3://eubucco/v0.2/buildings/parquet/nuts_id={nuts}/{nuts}.parquet"

# Région NUTS2 de chaque lieu de test.
# À l'échelle du jeu de données, on ferait une jointure spatiale entre les patches
# et les polygones NUTS2 d'Eurostat (GISCO) au lieu de cette table écrite à la main.
REGIONS_NUTS = {
    "Paris - Quartier latin": "FR10",  # Île-de-France
}

HAUTEUR_PAR_ETAGE_M = 3  # pour convertir un nombre d'étages en hauteur quand la hauteur manque


def region_nuts(nom_lieu):
    """Étape 2 : code NUTS2 du lieu, ou None s'il est hors d'Europe (EUBUCCO ne couvre que l'Europe)."""
    return REGIONS_NUTS.get(nom_lieu)


def connexion_eubucco():
    """Ouvre DuckDB avec ce qu'il faut pour lire le serveur S3 public d'EUBUCCO."""
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")     # lecture de fichiers distants
    con.execute("INSTALL spatial; LOAD spatial;")   # fonctions géographiques ST_...
    con.execute("SET s3_endpoint = 's3.eubucco.com';")
    con.execute("SET s3_url_style = 'path';")
    con.execute("SET s3_region = 'eu';")
    con.execute("SET s3_access_key_id = ''; SET s3_secret_access_key = '';")  # accès anonyme
    return con


# La requête, en trois temps :
#   a) on passe le carré du patch de lon/lat (EPSG:4326) dans la projection d'EUBUCCO (EPSG:3035, en mètres) ;
#   b) on pré-filtre avec la colonne "bbox" de chaque bâtiment : c'est ce qui permet à DuckDB
#      de sauter les morceaux du fichier qui sont loin du patch, au lieu de télécharger les 545 Mo ;
#   c) on découpe chaque bâtiment au carré, on calcule sa surface en m², et on renvoie la géométrie
#      en lon/lat pour pouvoir la comparer aux autres sources.
REQUETE = """
WITH carre AS (
    SELECT ST_Transform(ST_MakeEnvelope(?, ?, ?, ?), 'EPSG:4326', 'EPSG:3035', always_xy := true) AS g
)
SELECT
    b.height, b.floors, b.height_source,
    b.type, b.subtype, b.construction_year,
    ST_Area(ST_Intersection(b.geometry::GEOMETRY, c.g)) AS surface_m2,
    ST_AsWKB(ST_Transform(ST_Intersection(b.geometry::GEOMETRY, c.g),
                          'EPSG:3035', 'EPSG:4326', always_xy := true)) AS geometrie_wkb
FROM read_parquet(?) AS b, carre AS c
WHERE b.bbox.xmax >= ST_XMin(c.g) AND b.bbox.xmin <= ST_XMax(c.g)   -- b) pré-filtre rapide
  AND b.bbox.ymax >= ST_YMin(c.g) AND b.bbox.ymin <= ST_YMax(c.g)
  AND ST_Intersects(b.geometry::GEOMETRY, c.g)                      --    puis test exact
"""


def batiments_eubucco(bbox, nuts_id):
    """Étape 3 : bâtiments EUBUCCO du carré, au format commun aux trois sources.

    Chaque bâtiment est un dictionnaire :
      geometrie  : polygone shapely en lon/lat, découpé au carré du patch
      hauteur_m  : hauteur en mètres (ou étages x 3 m si la hauteur manque), None si rien
      + les attributs propres à EUBUCCO (type, sous-type, année, origine de la hauteur)
    """
    s, w, n, e = bbox
    con = connexion_eubucco()
    table = con.execute(REQUETE, [w, s, e, n, URL_EUBUCCO.format(nuts=nuts_id)]).df()

    batiments = []
    for ligne in table.itertuples():
        if pd.notna(ligne.height):                     # pd.notna : vrai si la valeur n'est pas manquante
            hauteur = float(ligne.height)
        elif pd.notna(ligne.floors):
            hauteur = float(ligne.floors) * HAUTEUR_PAR_ETAGE_M
        else:
            hauteur = None
        batiments.append({
            "geometrie": shapely.from_wkb(bytes(ligne.geometrie_wkb)),
            "hauteur_m": hauteur,
            "surface_m2": ligne.surface_m2,
            "origine_hauteur": ligne.height_source,
            "type": ligne.type,
            "sous_type": ligne.subtype,
            "annee": ligne.construction_year,
        })
    return batiments


def resumer(batiments):
    """Aperçu : nombre, surface bâtie, et d'où viennent les hauteurs."""
    avec_hauteur = [b for b in batiments if b["hauteur_m"] is not None]
    hauteurs = sorted(b["hauteur_m"] for b in avec_hauteur)
    return {
        "batiments": len(batiments),
        "surface_batie_m2": round(sum(b["surface_m2"] for b in batiments)),
        "batiments_avec_hauteur": len(avec_hauteur),
        "hauteur_mediane_m": hauteurs[len(hauteurs) // 2] if hauteurs else None,
        # "estimated" = hauteur prédite par un modèle EUBUCCO, les autres valeurs = source mesurée
        "origine_des_hauteurs": Counter(b["origine_hauteur"] for b in batiments).most_common(),
        "types": Counter(b["type"] for b in batiments).most_common(),
        "sous_types": Counter(b["sous_type"] for b in batiments).most_common(5),
        "annees_connues": sum(pd.notna(b["annee"]) for b in batiments),
    }


if __name__ == "__main__":
    for nom, lat, lon in EXEMPLES:
        print(f"\n=== {nom} ===")
        nuts = region_nuts(nom)
        if nuts is None:
            print("  hors couverture EUBUCCO (Europe seulement)")
            continue
        bbox = emprise(lat, lon)
        batiments = batiments_eubucco(bbox, nuts)
        print(f"Région NUTS2 : {nuts}")
        for cle, valeur in resumer(batiments).items():
            print(f"  {cle:24s} {valeur}")
