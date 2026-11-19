"""
  1. emprise()     : d'un point central à un carré de 320 m (patch So2Sat)
  2. objets_osm()  : tous les objets OpenStreetMap présents dans ce carré (API principale d'OSM)
  3. trier()       : on ne garde que les tags utiles pour les LCZ
  +  resumer()     : un aperçu de ce qu'on obtient

"""
import json
import math
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter

EXEMPLES = [
    ("Paris - Quartier latin", 48.8535, 2.3462),
    ("Nairobi - centre", -1.2850, 36.8230),
    ("Nairobi - Kibera", -1.3125, 36.7865),
]

COTE_M = 320  # 32 pixels x 10 m

# Serveurs Overpass publics
SERVEURS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# Les clés OSM qu'on garde
TAGS_GARDES = ["building", "building:levels", "height", "landuse",
               "natural", "highway", "waterway", "leisure"]


def emprise(lat, lon, cote_m=COTE_M):
    """Étape 1 : carré de cote_m mètres centré sur (lat, lon) -> (sud, ouest, nord, est)."""
    dlat = cote_m / 2 / 111_320                                  # 1 degré de latitude = 111,32 km
    dlon = cote_m / 2 / (111_320 * math.cos(math.radians(lat)))  # un degré de longitude rétrécit vers les pôles
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def lire_xml_osm(texte_xml):
    """Transforme la réponse XML d'OSM en liste d'objets {"type": ..., "tags": {...}}."""
    elements = []
    for el in ET.fromstring(texte_xml):
        if el.tag in ("node", "way", "relation"):
            tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
            if tags:
                elements.append({"type": el.tag, "id": el.get("id"), "tags": tags})
    return elements


def objets_osm(bbox):
    """Étape 2 : demande à OSM tous les objets du carré.
    On utilise d'abord l'API principale d'OpenStreetMap (fiable, mais réservée aux petites zones),
    puis, si elle échoue, les serveurs Overpass publics (souvent saturés).
    """

    s, w, n, e = bbox
    entete = {"User-Agent": "demo-pfe-lcz"}

    # 1) API principale : ordre des coordonnées = ouest, sud, est, nord
    url = f"https://api.openstreetmap.org/api/0.6/map?bbox={w},{s},{e},{n}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=entete), timeout=60) as reponse:
            return lire_xml_osm(reponse.read())
    except (urllib.error.URLError, TimeoutError, ET.ParseError) as erreur:
        print(f"  (échec sur api.openstreetmap.org : {erreur} -> on essaie Overpass)")

    # 2) Secours : Overpass
    requete = (
        f"[out:json][timeout:25][bbox:{s},{w},{n},{e}];"
        "(nwr[building];nwr[landuse];nwr[natural];way[highway];way[waterway];nwr[leisure];);"
        "out tags;"
    )
    donnees = urllib.parse.urlencode({"data": requete}).encode()
    for serveur in SERVEURS:
        try:
            with urllib.request.urlopen(urllib.request.Request(serveur, data=donnees, headers=entete), timeout=40) as reponse:
                return json.load(reponse)["elements"]
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as erreur:
            print(f"  (échec sur {serveur.split('/')[2]} : {erreur})")
    raise RuntimeError("Aucun serveur n'a répondu. Vérifie ta connexion ou réessaie plus tard.")


def trier(elements):
    """Étape 3 : pour chaque objet, on ne garde que les tags de TAGS_GARDES."""
    objets = []
    for el in elements:
        tags = {k: v for k, v in el.get("tags", {}).items() if k in TAGS_GARDES}
        if tags:
            objets.append(tags)
    return objets


def resumer(objets):
    """Aperçu : combien d'objets de chaque sorte, et à quel point les étages sont renseignés."""
    batiments = [o for o in objets if "building" in o]
    avec_etages = [o for o in batiments if "building:levels" in o]
    return {
        "batiments": len(batiments),
        "batiments_avec_etages": len(avec_etages),
        "etages": Counter(o["building:levels"] for o in avec_etages).most_common(5),
        "types_de_batiment": Counter(o["building"] for o in batiments).most_common(5),
        "usage_du_sol": Counter(o["landuse"] for o in objets if "landuse" in o).most_common(),
        "routes": Counter(o["highway"] for o in objets if "highway" in o).most_common(5),
        "naturel": Counter(o["natural"] for o in objets if "natural" in o).most_common(),
    }


if __name__ == "__main__":
    for nom, lat, lon in EXEMPLES:
        bbox = emprise(lat, lon)
        elements = objets_osm(bbox)
        objets = trier(elements)
        print(f"\n=== {nom} ===")   
        print("Emprise (sud, ouest, nord, est) :", tuple(round(x, 5) for x in bbox))
        print(f"Objets renvoyés par OSM : {len(elements)}  |  gardés après tri : {len(objets)}")
        for cle, valeur in resumer(objets).items():
            print(f"  {cle:24s} {valeur}")
        print("  Trois objets bruts après tri :", objets[:3])
