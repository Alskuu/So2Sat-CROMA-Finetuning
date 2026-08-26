"""
Pour une ville donnée, récupère son code EPSG dans le dictionnaire
`CITY_EPSG` (repris de save_geotiff.py, seule source vérifiée du projet
associant ville -> EPSG) puis affiche la répartition exacte des 17 classes
LCZ pour cette ville dans les splits globaux So2Sat (training/validation/
testing.h5 + leur `_geo.h5` associé, champs "epsg" et "coord").

Si la ville n'est pas dans `CITY_EPSG`, lève une erreur explicite plutôt que
de deviner.

Plusieurs villes peuvent partager le même code EPSG (même fuseau UTM) --
ex. Rio de Janeiro et Sao Paulo sont toutes les deux en EPSG:32723. Dans ce
cas, chaque patch de ce groupe EPSG est désambiguïsé géographiquement :
conversion lon/lat via `coord`+`epsg` (utils/geo_utils.py::to_lonlat, même
logique que segmentation_set.py, puis assignation au centre-ville connu le plus proche (`CITY_CENTERS`,
distance haversine). Les villes candidates étant toujours séparées de
centaines de km (les patchs font 320 m), cette désambiguïsation est fiable.
Si le centre d'une ville candidate n'est pas dans `CITY_CENTERS`, le script
retombe sur une répartition ambiguë et un avertissement
explicite plutôt que de deviner.

A executer en premier lieu pour Rio de Janeiro (présente dans le training
set, EPSG 32723) -- c'est la valeur par défaut de --city.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

# Ce script vit dans data_viz/, un sous-dossier de cromacode/ : il faut
# ajouter cromacode/ a sys.path AVANT d'importer `utils` pour ne pas laisser
# un formateur/isort trie cet import au-dessus de ce insert, sinon
# `ModuleNotFoundError: No module named 'utils'`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.geo_utils import epsg327_false_northing_fix, to_lonlat  # noqa: E402  isort: skip

LCZ_CLASSES_17 = [
    "1. Compact high-rise", "2. Compact mid-rise", "3. Compact low-rise",
    "4. Open high-rise", "5. Open mid-rise", "6. Open low-rise",
    "7. Lightweight low-rise", "8. Large low-rise", "9. Sparsely built",
    "10. Heavy industry", "A. Dense trees", "B. Scattered trees",
    "C. Bush, scrub", "D. Low plants", "E. Bare rock/paved",
    "F. Bare soil/sand", "G. Water",
]

# Splits globaux (pas les fichiers par continent produits par
# segmentation_set.py).
SPLITS = ["training", "validation", "testing"]

# Repris tel quel de save_geotiff.py -- training set + val/test set (ce
# second dict y est commente, mais c'est la meme source d'information).
CITY_EPSG = {
    "amsterdam": 32631, "beijing": 32650, "berlin": 32633, "bogota": 32618,
    "buenosaires": 32721, "cairo": 32636, "capetown": 32734, "caracas": 32619,
    "changsha": 32649, "chicago": 32616, "cologne": 32632, "dhaka": 32646,
    "dongying": 32650, "hongkong": 32650, "islamabad": 32643, "istanbul": 32635,
    "kyoto": 32653, "lima": 32718, "lisbon": 32629, "london": 32630,
    "losangeles": 32611, "madrid": 32630, "melbourne": 32755, "milan": 32632,
    "nanjing": 32650, "newyork": 32618, "orangitown": 32618, "paris": 32631,
    "philadelphia": 32618, "qingdao": 32651, "quezon": 32651, "riodejaneiro": 32723,
    "rome": 32633, "salvador": 32724, "saopaulo": 32723, "shanghai": 32651,
    "shenzhen": 32650, "tokyo": 32654, "vancouver": 32610, "washingtondc": 32618,
    "wuhan": 32650, "zurich": 32632,
    "guangzhou": 32649, "jakarta": 32748, "moscow": 32637, "mumbai": 32643,
    "munich": 32632, "nairobi": 32737, "sanfrancisco": 32610, "santiago": 32719,
    "sydney": 32756, "tehran": 32639,
}

# Coordonnees approximatives (lat, lon en degres, WGS84) des centres-villes,
# uniquement pour les villes qui collisionnent reellement dans CITY_EPSG --
# necessaires pour desambiguiser les patchs d'un meme fuseau UTM (cf.
# cities_sharing_epsg). Sourcees manuellement (connaissance geographique
# generale), pas besoin de precision au metre : les villes candidates sont
# toujours separees de centaines de km.
CITY_CENTERS = {
    "amsterdam": (52.3676, 4.9041), "paris": (48.8566, 2.3522),
    "beijing": (39.9042, 116.4074), "dongying": (37.4341, 118.6749),
    "hongkong": (22.3193, 114.1694), "nanjing": (32.0603, 118.7969),
    "shenzhen": (22.5431, 114.0579), "wuhan": (30.5928, 114.3055),
    "berlin": (52.5200, 13.4050), "rome": (41.9028, 12.4964),
    "bogota": (4.7110, -74.0721), "newyork": (40.7128, -74.0060),
    "orangitown": (41.0562, -73.9382), "philadelphia": (39.9526, -75.1652),
    "washingtondc": (38.9072, -77.0369),
    "changsha": (28.2282, 112.9388), "guangzhou": (23.1291, 113.2644),
    "cologne": (50.9375, 6.9603), "milan": (45.4642, 9.1900),
    "munich": (48.1351, 11.5820), "zurich": (47.3769, 8.5417),
    "islamabad": (33.6844, 73.0479), "mumbai": (19.0760, 72.8777),
    "london": (51.5074, -0.1278), "madrid": (40.4168, -3.7038),
    "qingdao": (36.0671, 120.3826), "quezon": (14.6760, 121.0437),
    "shanghai": (31.2304, 121.4737),
    "riodejaneiro": (-22.9068, -43.1729), "saopaulo": (-23.5505, -46.6333),
    "vancouver": (49.2827, -123.1207), "sanfrancisco": (37.7749, -122.4194),
}


def normalize_city(name: str) -> str:
    """'Rio de Janeiro' -> 'riodejaneiro', meme convention que CITY_EPSG."""
    return "".join(ch for ch in name.strip().lower() if ch.isalnum())


def get_city_epsg(city: str) -> int:
    """Renvoie le code EPSG de `city`, ou leve ValueError si elle est absente
    de CITY_EPSG (aucune supposition faite dans ce cas)."""
    key = normalize_city(city)
    if key not in CITY_EPSG:
        raise ValueError(
            f"Ville '{city}' (normalisee : '{key}') absente de CITY_EPSG. "
            f"Villes connues : {', '.join(sorted(CITY_EPSG))}."
        )
    return CITY_EPSG[key]


def cities_sharing_epsg(epsg: int, exclude: str) -> list:
    """Autres villes de CITY_EPSG partageant le meme code EPSG que `exclude`."""
    return sorted(c for c, e in CITY_EPSG.items() if e == epsg and c != exclude)


def haversine_km(lon1, lat1, lon2, lat2):
    """Distance haversine (km) entre deux points lon/lat, vectorisable numpy."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * \
        np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def classify_nearest_city(lon: np.ndarray, lat: np.ndarray, candidates: list) -> np.ndarray:
    """Pour chaque point lon/lat, renvoie la ville de `candidates` (cles
    normalisees, presentes dans CITY_CENTERS) dont le centre est le plus proche."""
    dists = np.stack([
        haversine_km(lon, lat, CITY_CENTERS[c][1], CITY_CENTERS[c][0])
        for c in candidates
    ], axis=1)  # (n_points, n_candidates)
    nearest_idx = np.argmin(dists, axis=1)
    candidates_arr = np.array(candidates)
    return candidates_arr[nearest_idx]


def load_coord_xy(geo_h5_path: Path, idx: np.ndarray):
    """Lit les colonnes x/y de `coord` pour les lignes `idx`. Leve une
    KeyError explicite (cles disponibles listees) si `coord` est absent,
    plutot que de supposer silencieusement un champ qui n'existe pas."""
    with h5py.File(geo_h5_path, "r") as f:
        if "coord" not in f:
            raise KeyError(
                f"Champ 'coord' absent de {geo_h5_path}. Cles disponibles : "
                f"{list(f.keys())}. Ajustez load_coord_xy si le champ porte "
                f"un autre nom dans vos fichiers."
            )
        coord = f["coord"][idx]  # (len(idx), 6), colonnes 0-1 = x, y
    return coord[:, 0].astype(np.float64), coord[:, 1].astype(np.float64)


def epsg_distribution_for_split(data_dir: Path, split: str, target_epsg: int,
                                target_city: str = None, candidates: list = None):
    """Renvoie un np.ndarray de comptes par classe LCZ (len 17) pour ce split,
    ou None si le split est absent ou si l'EPSG (ou la ville, apres
    desambiguisation) n'y figure pas.

    Si `candidates` est fourni (plusieurs villes partagent `target_epsg`),
    ne garde que les patchs geographiquement assignes a `target_city`
    (cf. classify_nearest_city) avant de compter les classes LCZ.
    """
    h5_path = data_dir / f"{split}.h5"
    geo_h5_path = data_dir / f"{split}_geo.h5"

    if not h5_path.exists() or not geo_h5_path.exists():
        print(
            f"[{split}] fichier(s) introuvable(s) ({h5_path} / {geo_h5_path}) -- ignore.")
        return None

    with h5py.File(geo_h5_path, "r") as f:
        epsg_raw = f["epsg"][:]
    epsg = np.asarray(epsg_raw).reshape(
        len(epsg_raw), -1)[:, 0].astype(np.int64)

    mask = epsg == target_epsg
    n_matches = int(mask.sum())
    if n_matches == 0:
        print(f"[{split}] aucun patch avec EPSG:{target_epsg} dans ce split.")
        return None

    idx = np.where(mask)[0]

    if candidates:
        x, y = load_coord_xy(geo_h5_path, idx)
        epsg_for_idx = np.full(len(idx), target_epsg, dtype=np.int64)
        y_fixed = epsg327_false_northing_fix(y, epsg_for_idx)
        lon, lat = to_lonlat(x, y_fixed, epsg_for_idx)
        assigned = classify_nearest_city(lon, lat, candidates)
        keep = assigned == target_city
        n_kept = int(keep.sum())
        print(f"[{split}] {n_kept}/{n_matches} patch(s) EPSG:{target_epsg} "
              f"assignes a '{target_city}' apres desambiguisation geographique.")
        idx = idx[keep]
        if n_kept == 0:
            return None

    with h5py.File(h5_path, "r") as f:
        labels_onehot = f["label"][idx]  # (len(idx), 17)

    lcz = np.argmax(labels_onehot, axis=1)
    counts = np.bincount(lcz, minlength=len(LCZ_CLASSES_17))
    if not candidates:
        print(f"[{split}] {n_matches} patch(s) trouve(s) pour EPSG:{target_epsg}.")
    return counts


def print_distribution(title: str, counts: np.ndarray):
    total = int(counts.sum())
    print(f"\n=== {title} (total : {total} patchs) ===")
    for i, cls_name in enumerate(LCZ_CLASSES_17):
        if counts[i] == 0:
            continue
        pct = 100.0 * counts[i] / total
        print(f"  {cls_name:25s} : {counts[i]:5d} ({pct:5.2f} %)")


def main():
    p = argparse.ArgumentParser(
        description="Repartition des classes LCZ pour une ville donnee (via son code EPSG), "
                    "dans les splits So2Sat globaux.")
    p.add_argument("--city", type=str, default="Rio de Janeiro")
    p.add_argument("--data-dir", type=str, default="D:/LCZ42")
    p.add_argument("--splits", type=str, nargs="+", default=SPLITS)
    args = p.parse_args()

    target_key = normalize_city(args.city)
    target_epsg = get_city_epsg(args.city)
    print(f"'{args.city}' -> EPSG:{target_epsg}")

    others = cities_sharing_epsg(target_epsg, target_key)
    candidates = None
    if others:
        all_candidates = [target_key] + others
        missing_centers = [c for c in all_candidates if c not in CITY_CENTERS]
        if missing_centers:
            print(
                f"[ATTENTION] EPSG:{target_epsg} est aussi utilise par : {', '.join(others)}. "
                f"Centre-ville inconnu pour {', '.join(missing_centers)} -- desambiguisation "
                f"geographique impossible : la repartition ci-dessous n'est donc PAS specifique "
                f"a '{args.city}', elle couvre TOUTES les villes listees ci-dessus a la fois."
            )
            group_label = (
                f"EPSG:{target_epsg} -- ville(s) indiscernables dans ce dataset : "
                f"{args.city} + {', '.join(others)}"
            )
        else:
            candidates = all_candidates
            print(
                f"[INFO] EPSG:{target_epsg} est aussi utilise par : {', '.join(others)} -- "
                f"desambiguisation geographique appliquee (distance au centre-ville le plus "
                f"proche, cf. CITY_CENTERS)."
            )
            group_label = f"{args.city} (EPSG:{target_epsg}, desambiguise parmi {', '.join(all_candidates)})"
    else:
        group_label = f"{args.city} (EPSG:{target_epsg})"

    data_dir = Path(args.data_dir)
    total_counts = np.zeros(len(LCZ_CLASSES_17), dtype=np.int64)
    found_in_any = False

    for split in args.splits:
        counts = epsg_distribution_for_split(
            data_dir, split, target_epsg, target_city=target_key, candidates=candidates)
        if counts is None:
            continue
        found_in_any = True
        print_distribution(f"{group_label} -- split '{split}'", counts)
        total_counts += counts

    if not found_in_any:
        print(f"\nAucun patch pour '{args.city}' trouve dans les splits "
              f"{args.splits} sous {data_dir}.")
        return

    print_distribution(f"{group_label} -- TOUS SPLITS CONFONDUS", total_counts)


if __name__ == "__main__":
    main()
