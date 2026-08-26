"""
02_split_by_continent.py
--------------------------
Pipeline complet :
Choisissez le jeu de donnée que vous voulez tout en bas de ce script, celui-ci va ensuite :
  1. Lire par exexmple testing_geo.h5 (coord UTM + epsg) -> convertit en lon/lat
  2. Jointure spatiale avec les polygones pays Natural Earth -> region
     (continent, sauf l'Asie qui est subdivisee en "Asie de l'Est" et
     "Asie de l'Ouest" sur la base du pays)
  3. Ecrit testing_<region>.h5 (sen1/sen2/label) ET
     testing_geo_<region>.h5 (coord/epsg/tfw), alignes ligne a ligne,
     pour chaque region (Antarctique exclue).

Tous les patchs qui sont considérés comme n'appartenant à aucune de nos
regions correspondent sûrement à de l'eau, ce qui explique que ce n'est pas
inclu dans une region ; nous effectuons alors un fallback qui permet de
considérer que ces patchs appartiennent a la region la plus proche en
distance (via la CRS distance metric EPSG 6933 assez précise sur tout le
globe terrestre).

L'idx d'un point correspond directement a la ligne dans testing.h5 ET
testing_geo.h5 (meme ordre, meme longueur) : c'est ce qui permet de
recouper les deux fichiers sans info supplementaire.

Corrections appliquees (bug UTM systemique) :
  - EPSG 327xx : fix du "false northing" quand y < 0

Corrections NON appliquees ici (specifiques a des villes du train/val,
a verifier si elles concernent aussi testing avant de les repliquer) :
  - Londres, Nanjing, Hong Kong, Shenzhen (offsets EPSG +/-1)
  - Cluster Cuba (erreur connue, non corrigee dans le pipeline d'origine)

Decoupage Asie de l'Est / Asie de l'Ouest :
  - Base sur le champ pays ("ADMIN") du shapefile Natural Earth, PAS sur le
    champ SUBREGION officiel de l'ONU (qui classerait Pakistan/Inde en
    "Asie du Sud" et non "Asie de l'Ouest"). Ici c'est un decoupage
    "maison" adapte au jeu de villes du dataset : la liste WEST_ASIA_COUNTRIES
    ci-dessous fait foi. Ajoutez-y d'autres pays si besoin (ex: Iran, Turquie,
    Arabie Saoudite...), tout pays d'Asie absent de cette liste est classe
    "Asie de l'Est" par defaut.
"""

import h5py
import numpy as np
import pandas as pd
import geopandas as gpd

from utils.geo_utils import (
    COUNTRIES_SHP,
    epsg327_false_northing_fix,
    to_lonlat,
    points_gdf_from_lonlat,
)

# ============================================================
# Chemins - a adapter
# ============================================================
OUT_DIR = "D:/LCZ42/per_continent/"  # doit exister
# CRS metrique global utilise uniquement pour le calcul de distance
# (World Cylindrical Equal Area -> distances comparables partout,
# contrairement a EPSG:4326 en degres).
METRIC_CRS = "EPSG:6933"

# Pays d'Asie a classer dans "Asie de l'Ouest" (decoupage maison, voir
# note en tete de fichier). Complete cette liste si d'autres villes du
# dataset tombent dans des pays qui devraient etre "Asie de l'Ouest".
WEST_ASIA_COUNTRIES = [
    "Pakistan",
    "India",
    "Iran",
    "Turkey",
    "Bangladesh"
]

CONTINENT_TO_SUFFIX = {
    "Africa": "afrique",
    "East Asia": "asie_est",
    "West Asia": "asie_ouest",
    "Europe": "europe",
    "North America": "ameriquenord",
    "South America": "ameriquesud",
    "Oceania": "oceanie",
}
# Regions "valides" = celles qu'on garde comme cible possible pour le
# fallback "plus proche en distance". Antarctique / open ocean ne sont
# jamais candidats.
VALID_REGIONS = [c for c, s in CONTINENT_TO_SUFFIX.items() if s is not None]

# Toute region absente du dict, ou point tombant en mer (jointure -> NaN),
# part dans "autre" (utile pour reperer les patchs cotiers mal geolocalises).


# ============================================================
# Etape 1 : UTM -> lon/lat
# ============================================================
def build_points(path):
    path_geo = path.replace(".h5", "_geo.h5")
    with h5py.File(path_geo, "r") as f:
        coord = f["coord"][:]  # (n, 6)
        epsg = f["epsg"][:, 0].astype(np.int64)

    x, y = coord[:, 0].astype(np.float64), coord[:, 1].astype(np.float64)
    y = epsg327_false_northing_fix(y, epsg)
    lon, lat = to_lonlat(x, y, epsg)

    return points_gdf_from_lonlat(lon, lat)


# ============================================================
# Etape 2 : jointure spatiale -> region (continent, Asie subdivisee)
# ============================================================
def load_countries_with_region():
    """Charge le shapefile pays et ajoute une colonne REGION qui vaut
    CONTINENT partout, sauf pour l'Asie qui est subdivisee en
    'East Asia' / 'West Asia' selon WEST_ASIA_COUNTRIES."""
    countries = gpd.read_file(COUNTRIES_SHP).to_crs("EPSG:4326")[
        ["ADMIN", "CONTINENT", "geometry"]]

    countries["REGION"] = countries["CONTINENT"]
    is_asia = countries["CONTINENT"] == "Asia"
    is_west = is_asia & countries["ADMIN"].isin(WEST_ASIA_COUNTRIES)
    countries.loc[is_west, "REGION"] = "West Asia"
    countries.loc[is_asia & ~is_west, "REGION"] = "East Asia"

    return countries


def assign_continents(gdf):
    countries = load_countries_with_region()
    j = gpd.sjoin(gdf, countries, predicate="within", how="left")

    df = pd.DataFrame(
        {"idx": j["idx"], "country": j["ADMIN"], "continent": j["REGION"]})
    n_missing = df["continent"].isna().sum()

    print("=== Repartition par region ===")
    print(df["continent"].value_counts(dropna=False))
    print(f"\n{n_missing} points sans region exacte -> fallback 'plus proche'.")

    if n_missing == 0:
        return df

    # --- Passe 2 : fallback "region la plus proche" pour les points manques ---
    # On dissout les pays valides (hors Antarctique/open ocean) par region,
    # pour avoir une seule geometrie par region cible.
    valid_countries = countries[countries["REGION"].isin(VALID_REGIONS)]
    regions_dissolved = (
        valid_countries.dissolve(by="REGION")[["geometry"]].reset_index()
    )

    # Reprojection dans un CRS metrique pour un calcul de distance correct.
    missing_mask = df["continent"].isna().to_numpy()
    gdf_missing = gdf.loc[missing_mask].to_crs(METRIC_CRS)
    regions_metric = regions_dissolved.to_crs(METRIC_CRS)

    nearest = gpd.sjoin_nearest(
        gdf_missing, regions_metric,
        how="left", distance_col="dist_m",
    )
    # sjoin_nearest peut dupliquer une ligne en cas d'egalite parfaite de distance
    # entre 2 regions (tres rare) -> on garde la premiere occurrence.
    nearest = nearest.drop_duplicates(subset="idx", keep="first")

    print("\n=== Fallback : region la plus proche (points sans jointure exacte) ===")
    print(nearest["REGION"].value_counts(dropna=False))
    print(
        f"Distance a la region la plus proche - "
        f"min: {nearest['dist_m'].min()/1000:.1f} km, "
        f"median: {nearest['dist_m'].median()/1000:.1f} km, "
        f"max: {nearest['dist_m'].max()/1000:.1f} km"
    )

    # On reinjecte les regions trouvees par fallback dans df, a la place des NaN.
    nearest_lookup = nearest.set_index("idx")["REGION"]
    df = df.set_index("idx")
    df.loc[df["continent"].isna(), "continent"] = df.index.map(
        nearest_lookup).to_numpy()[df["continent"].isna().to_numpy()]
    df = df.reset_index()

    still_missing = df["continent"].isna().sum()
    if still_missing > 0:
        # Ne devrait jamais arriver (sjoin_nearest est total sur des polygones
        # valides), mais on le signale si jamais un idx a echappe au fallback.
        print(
            f"\n[ATTENTION] {still_missing} points restent sans region "
            f"apres fallback -> a investiguer (idx: "
            f"{df.loc[df['continent'].isna(), 'idx'].tolist()[:20]}...)"
        )

    return df


# ============================================================
# Etape 3 : ecriture des fichiers par region
# ============================================================
def split_h5(df, path):
    groups = {}
    for _, row in df.iterrows():
        continent = row["continent"]
        if pd.isna(continent):
            suffix = "autre"
        else:
            suffix = CONTINENT_TO_SUFFIX.get(continent, "autre")
        if suffix is None:
            continue  # Antarctique / open ocean : on ignore
        groups.setdefault(suffix, []).append(int(row["idx"]))

    path_geo = path.replace(".h5", "_geo.h5")
    with h5py.File(path, "r") as f_img, h5py.File(path_geo, "r") as f_geo:
        img_keys = list(f_img.keys())   # label, sen1, sen2
        geo_keys = list(f_geo.keys())   # coord, epsg, tfw

        for suffix, idx_list in groups.items():
            # h5py exige un ordre croissant en fancy indexing
            idx_sorted = np.array(sorted(idx_list))

            set_type = path.replace("D:/LCZ42/", "")
            set_type = set_type.replace(".h5", "")
            out_img_path = f"{OUT_DIR}{set_type}_{suffix}.h5"
            with h5py.File(out_img_path, "w") as fout:
                for key in img_keys:
                    fout.create_dataset(key, data=f_img[key][idx_sorted])
                fout.create_dataset("orig_idx", data=idx_sorted)

            out_geo_path = f"{OUT_DIR}{set_type}_geo_{suffix}.h5"
            with h5py.File(out_geo_path, "w") as fout:
                for key in geo_keys:
                    fout.create_dataset(key, data=f_geo[key][idx_sorted])
                fout.create_dataset("orig_idx", data=idx_sorted)

            print(
                f"{suffix:12s} : {len(idx_sorted):6d} patchs -> {out_img_path} + {out_geo_path}")

    n_total = sum(len(v) for v in groups.values())
    n_input = len(df)
    print(
        f"\nTotal ecrit : {n_total} / {n_input} points ({n_input - n_total} ignores).")


# ============================================================
def main(path):

    pts = build_points(path)
    df = assign_continents(pts)
    split_h5(df, path)


if __name__ == "__main__":
    path = "D:/LCZ42/validation.h5"
    main(path)
    training_path = "D:/LCZ42/training.h5"
    main(training_path)
