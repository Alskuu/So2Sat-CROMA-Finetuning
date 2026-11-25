"""
Bâtiments Microsoft Global ML Building Footprints sur un patch So2Sat de 320 m (monde entier).

  1. emprise()               : d'un point central à un carré de 320 m (repris de demo_osm_patch.py)
  2. quadkey()               : dans quelle tuile Microsoft tombe le patch
  3. urls_des_tuiles()       : où télécharger cette tuile (index officiel de Microsoft)
  4. batiments_microsoft()   : lecture de la tuile, on garde les bâtiments du carré
  +  resumer()               : un aperçu de ce qu'on obtient

Source : https://github.com/microsoft/GlobalMLBuildingFootprints  (licence ODbL)
Les bâtiments sont découpés en tuiles carrées (système "quadkey" de Bing Maps, niveau 9,
environ 70 km de côté à l'équateur). Chaque tuile est un fichier .csv.gz qui contient
un bâtiment GeoJSON par ligne, avec deux propriétés : "height" (-1 si inconnue) et "confidence".

Les tuiles pèsent environ 100 Mo : on les télécharge une seule fois, dans osm/cache/microsoft/.

Dépendances : pip install pandas shapely
"""
import gzip
import json
import math
import re
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd
import shapely
from shapely.geometry import box, shape

from demo_osm_patch import EXEMPLES, emprise

URL_INDEX = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
NIVEAU_TUILES = 9
DOSSIER_CACHE = Path(__file__).parent / "cache" / "microsoft"


def quadkey(lat, lon, niveau=NIVEAU_TUILES):
    """Étape 2 : identifiant de la tuile Bing Maps (web Mercator) qui contient le point.

    À chaque niveau, une tuile est coupée en 4 (0 = haut gauche, 1 = haut droite,
    2 = bas gauche, 3 = bas droite) : le quadkey est la suite de ces choix, un chiffre par niveau.
    """
    nb_tuiles = 2 ** niveau
    x = int((lon + 180) / 360 * nb_tuiles)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * nb_tuiles)
    chiffres = []
    for i in range(niveau - 1, -1, -1):          # du niveau le plus grossier au plus fin
        bit_x = (x >> i) & 1
        bit_y = (y >> i) & 1
        chiffres.append(str(bit_x + 2 * bit_y))
    return "".join(chiffres)


def tuiles_du_patch(bbox):
    """Les quadkeys des 4 coins : en général un seul, deux si le patch est à cheval sur une frontière de tuile."""
    s, w, n, e = bbox
    return sorted({quadkey(lat, lon) for lat in (s, n) for lon in (w, e)})


def urls_des_tuiles(quadkeys):
    """Étape 3 : adresses des fichiers à télécharger, lues dans l'index officiel de Microsoft.

    On cherche par quadkey seulement : pas besoin de connaître le pays du patch.
    """
    DOSSIER_CACHE.mkdir(parents=True, exist_ok=True)
    chemin_index = DOSSIER_CACHE / "dataset-links.csv"
    if not chemin_index.exists():
        urllib.request.urlretrieve(URL_INDEX, chemin_index)
    index = pd.read_csv(chemin_index, dtype=str)       # dtype=str : garder les 0 en tête des quadkeys
    return index[index["QuadKey"].isin(quadkeys)]["Url"].tolist()


def fichier_local(url):
    """Télécharge la tuile si elle n'est pas déjà dans le cache, et renvoie son chemin local."""
    # Le quadkey et le nom du morceau ("part-...") suffisent à rendre le nom de fichier unique
    quadkey_url = re.search(r"quadkey=(\d+)", url).group(1)
    chemin = DOSSIER_CACHE / f"{quadkey_url}_{url.rsplit('/', 1)[-1]}"
    if not chemin.exists():
        print(f"  téléchargement de la tuile {quadkey_url} (une seule fois) ...")
        urllib.request.urlretrieve(url, chemin)
    return chemin


# Repère le premier couple de coordonnées d'une ligne, "[lon, lat]", sans décoder tout le JSON
PREMIER_POINT = re.compile(r"\[\[\[(-?[\d.]+), (-?[\d.]+)\]")
MARGE_DEG = 0.01  # environ 1 km : un bâtiment dont le premier sommet est plus loin ne peut pas toucher le patch


def batiments_microsoft(bbox):
    """Étape 4 : bâtiments Microsoft du carré, au format commun aux trois sources.

    Chaque bâtiment est un dictionnaire :
      geometrie  : polygone shapely en lon/lat, découpé au carré du patch
      hauteur_m  : hauteur en mètres, None si Microsoft ne la donne pas
      confiance  : confiance du modèle de détection (entre 0 et 1)
    """
    s, w, n, e = bbox
    carre = box(w, s, e, n)
    batiments = []
    for url in urls_des_tuiles(tuiles_du_patch(bbox)):
        with gzip.open(fichier_local(url), "rt") as fichier:
            for ligne in fichier:
                # Tri rapide : une tuile contient des centaines de milliers de bâtiments,
                # on ne décode en entier que ceux qui sont près du patch.
                point = PREMIER_POINT.search(ligne)
                if point is None:
                    continue
                lon, lat = float(point.group(1)), float(point.group(2))
                if not (w - MARGE_DEG < lon < e + MARGE_DEG and s - MARGE_DEG < lat < n + MARGE_DEG):
                    continue

                objet = json.loads(ligne)
                polygone = shape(objet["geometry"])
                if not polygone.intersects(carre):
                    continue
                hauteur = objet["properties"].get("height", -1)       # -1 = inconnue
                confiance = objet["properties"].get("confidence", -1)  # -1 = non fournie
                batiments.append({
                    "geometrie": polygone.intersection(carre),
                    "hauteur_m": hauteur if hauteur is not None and hauteur > 0 else None,
                    "confiance": confiance if confiance is not None and confiance >= 0 else None,
                })
    return batiments


def resumer(batiments):
    """Aperçu : nombre, part des hauteurs connues, et confiance du modèle."""
    hauteurs = sorted(b["hauteur_m"] for b in batiments if b["hauteur_m"] is not None)
    confiances = [b["confiance"] for b in batiments if b["confiance"] is not None]
    return {
        "batiments": len(batiments),
        "batiments_avec_hauteur": len(hauteurs),
        "hauteur_mediane_m": round(hauteurs[len(hauteurs) // 2], 1) if hauteurs else None,
        "hauteurs_arrondies": Counter(round(h) for h in hauteurs).most_common(5),
        "confiance_moyenne": round(sum(confiances) / len(confiances), 2) if confiances else None,
        "types_de_geometrie": Counter(b["geometrie"].geom_type for b in batiments).most_common(),
    }


if __name__ == "__main__":
    for nom, lat, lon in EXEMPLES:
        print(f"\n=== {nom} ===")
        bbox = emprise(lat, lon)
        print("Tuile(s) Microsoft :", tuiles_du_patch(bbox))
        batiments = batiments_microsoft(bbox)
        for cle, valeur in resumer(batiments).items():
            print(f"  {cle:24s} {valeur}")
