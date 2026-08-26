import logging
from pathlib import Path

import h5py
import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS

try:
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box
    _HAS_GPD = True
except ImportError:
    _HAS_GPD = False

from testing_model import (
    H5_DIR_CONTINENTS,
    LCZ_CLASSES_17,
    URBAN_ONLY_STEPS,
    parse_args,
    run_loop,
)
from dataset import So2SatDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Palette LCZ officielle WUDAPT/RUB (identique au layer Earth Engine
# RUB/RUBCLIM/LCZ/global_lcz_map). index i <-> LCZ_CLASSES_17[i].
LCZ_COLORS_HEX = [
    "8c0000", "d10000", "ff0000", "bf4d00", "ff6600",
    "ff9955", "faee05", "bcbcbc", "ffccaa", "555555",
    "006a00", "00aa00", "648525", "b9db79", "000000",
    "fbf7ae", "6a6aff",
]

# ============================================================
# Fichiers de test par continent (produits par 02_split_by_continent.py).
# Pas de fichier geo pour "global" -> exclu ici, contrairement a
# testing_model.ALL_CONTINENTS.
# ============================================================
GEO_CONTINENTS = [
    "europe", "afrique", "asie_ouest", "asie_est",
    "ameriquenord", "ameriquesud", "oceanie",
]
CONTINENT_TO_H5 = {c: H5_DIR_CONTINENTS +
                   f"testing_{c}.h5" for c in GEO_CONTINENTS}
CONTINENT_TO_GEO_H5 = {c: H5_DIR_CONTINENTS +
                       f"testing_geo_{c}.h5" for c in GEO_CONTINENTS}

# Bandes Sentinel-2 dans sen2, ordre documente dans le README officiel du
# dataset (github.com/zhu-xlab/So2Sat-LCZ42) :
#   1:B2 2:B3 3:B4 4:B5 5:B6 6:B7 7:B8 8:B8A 9:B11 10:B12
SEN2_BAND_NAMES = ["B2", "B3", "B4", "B5", "B6",
                   "B7", "B8", "B8A", "B11", "B12"]

PATCH_SIZE_PX = 32  # patches So2Sat toujours 32x32

# ============================================================
# Geolocalisation (geotransform + epsg) et export GeoTIFF + GPKG
# ============================================================
# continent -> (geo6, epsg) complets, ou None si absent
_geo_cache = {}
_dataset_indices_cache = {}  # continent -> indices (urban_only=True)


def get_geo_arrays(continent: str):
    if continent not in _geo_cache:
        path = CONTINENT_TO_GEO_H5[continent]
        if not Path(path).exists():
            logger.warning(
                f"[{continent}] {path} introuvable -- geotiffs/gpkg desactives pour ce continent.")
            _geo_cache[continent] = None
        else:
            with h5py.File(path, "r") as f:
                geo6 = f["tfw"][:]  # (N,6) -- champ nommé "tfw" mais confirme
                # empiriquement (test visuel QGIS) comme
                # etant un GeoTransform GDAL brut, pas un
                # vrai world file ESRI.
                epsg_raw = f["epsg"][:]
                # Les données sont d'anciennes données sûrement générées par Matlab avec donc deux dimensions, bien que la deuxième dimension soit inutile..
                epsg = epsg_raw[:, 0].astype(
                    np.int64)
            _geo_cache[continent] = (geo6, epsg)
    return _geo_cache[continent]


def get_urban_only_indices(continent: str):
    """Reconstruit dataset.indices pour urban_only=True, une seule fois par continent."""
    if continent not in _dataset_indices_cache:
        ds = So2SatDataset(
            h5_path=CONTINENT_TO_H5[continent], interpolation_mtd="nearest", urban_only=True)
        if not hasattr(ds, "indices"):
            raise AttributeError(
                "So2SatDataset n'expose pas d'attribut 'indices' : impossible de retrouver "
                "la vraie ligne du .h5 pour les etapes urban_only."
            )
        _dataset_indices_cache[continent] = np.asarray(ds.indices)
    return _dataset_indices_cache[continent]


def resolve_real_row(idx_dataset: int, continent: str, k: int) -> int:
    """Traduit l'idx_dataset (position dans le DataLoader) en vraie ligne du .h5 continent."""
    if k in URBAN_ONLY_STEPS:
        return int(get_urban_only_indices(continent)[idx_dataset])
    return idx_dataset


def geotransform_to_affine(geo6, patch_size: int = PATCH_SIZE_PX) -> Affine:
    """
    Convertit un GeoTransform GDAL brut (6 valeurs) en objet Affine rasterio.

    Correction empirique (comparaison visuelle sur points d'eau et lisieres
    de foret vs imagerie de reference, tests repetes) : le y_origin fourni
    par les donnees place systematiquement toute geometrie (rasters ET
    points GPKG) exactement une hauteur de patch trop au nord --
    32 pixels * 10m = 320m. On corrige en soustrayant cette hauteur avant
    construction de l'Affine.
    """
    geo6 = [float(v) for v in np.asarray(geo6).reshape(-1)]
    x_origin, pixel_width, x_skew, y_origin, y_skew, pixel_height = geo6

    y_origin_corrected = y_origin - patch_size * pixel_height

    geo6_corrected = [x_origin, pixel_width, x_skew,
                      y_origin_corrected, y_skew, pixel_height]
    return Affine.from_gdal(*geo6_corrected)


def check_transform_sane(transform: Affine, expected_px: float = 10.0,
                         tol_ratio: float = 0.05):
    a, b, d, e = transform.a, transform.b, transform.d, transform.e
    if not np.isfinite([a, b, d, e]).all():
        return False, f"non-finite: a={a},b={b},d={d},e={e}"
    lo, hi = expected_px * (1 - tol_ratio), expected_px * (1 + tol_ratio)
    if not (lo <= abs(a) <= hi):
        return False, f"|a| hors tolerance ({expected_px}m attendu): a={a}"
    if not (lo <= abs(e) <= hi):
        return False, f"|e| hors tolerance ({expected_px}m attendu): e={e}"
    if abs(b) > 1e-6 or abs(d) > 1e-6:
        return False, f"rotation non nulle inattendue: b={b}, d={d}"
    return True, "ok"


def write_geotiff_for_example(continent: str, k: int, idx_dataset: int, kind: str,
                              true_c: int, pred_c: int, LCZ_CLASSES: list, out_dir: Path):
    """
    Ecrit un GeoTIFF Sentinel-2 (10 bandes brutes, non redimensionnees) pour un
    exemple correct/faux, georeference via geotransform+epsg. Renvoie un dict
    {"path", "real_row", "epsg", "transform"} si succes, sinon None.
    """
    geo = get_geo_arrays(continent)
    if geo is None:
        return None
    geo6_arr, epsg_arr = geo

    try:
        real_row = resolve_real_row(idx_dataset, continent, k)
    except Exception as exc:
        logger.warning(
            f"[{continent}] resolution idx={idx_dataset} impossible ({exc}) -- geotiff ignore.")
        return None

    epsg_code = int(epsg_arr[real_row])
    try:
        crs = CRS.from_epsg(epsg_code)
    except Exception as exc:
        logger.warning(
            f"[{continent}] EPSG {epsg_code} invalide ({exc}) -- geotiff ignore.")
        return None

    transform = geotransform_to_affine(geo6_arr[real_row])
    # Pour effectuer une vérification de cohérence sur la transformation affine avant écriture, pour logger un warning
    ok, reason = check_transform_sane(transform)
    if not ok:
        logger.warning(
            f"[{continent}] row={real_row} transform suspect ({reason}) -- geotiff ignore.")
        return None

    with h5py.File(CONTINENT_TO_H5[continent], "r") as f:
        sen2_hwc = f["sen2"][real_row]  # (32, 32, 10) brut, 10m/pixel

    sen2_chw = np.transpose(sen2_hwc, (2, 0, 1)).astype(
        np.float32)  # (10, H, W)

    # Flip Nord-Sud : confirme empiriquement (test 4 coins, QGIS) que la ligne 0
    # du tableau correspond au SUD de la scene, pas au nord comme l'attend la
    # convention image standard combinee a notre geotransform (pixel_height > 0).
    # np.flip() renvoie une vue a stride negatif -> .copy() pour un tableau
    # contigu, necessaire pour rasterio/GDAL.
    sen2_chw = np.flip(sen2_chw, axis=1).copy()

    # Permutation B2 <-> B4 (indices 0 et 2) pour un rendu RGB correct par
    # defaut dans QGIS : bande1=B4(rouge), bande2=B3(vert), bande3=B2(bleu)
    band_order = [2, 1, 0, 3, 4, 5, 6, 7, 8, 9]
    sen2_chw = sen2_chw[band_order]
    band_names = [SEN2_BAND_NAMES[i] for i in band_order]

    H, W = sen2_chw.shape[1], sen2_chw.shape[2]

    true_code = LCZ_CLASSES[true_c].split(".")[0].strip()
    pred_code = LCZ_CLASSES[pred_c].split(".")[0].strip()
    fname = f"{continent}_etape{k}_{kind}_vrai{true_code}_pred{pred_code}_row{real_row}.tif"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname

    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=H, width=W,
        count=sen2_chw.shape[0],
        dtype=sen2_chw.dtype,
        crs=crs,
        transform=transform,
        compress="DEFLATE",
    ) as dst:
        dst.write(sen2_chw)
        dst.descriptions = band_names

    return {"path": str(out_path), "real_row": real_row,
            "epsg": epsg_code, "transform": transform}


def _build_examples_gdf(gpkg_rows: list):
    """
    Reprojette chaque emprise de patch (UTM, un EPSG par ligne) en EPSG:4326
    et renvoie UN GeoDataFrame commun -- utilise en amont de
    `build_examples_gpkg` ET `build_examples_kml`, pour ne calculer/
    reprojeter les polygones qu'une seule fois quel que soit le nombre de
    formats de sortie demandes.

    gpkg_rows: liste de dicts avec cles
        transform, epsg, true_lcz, pred_lcz, correct, continent, etape_k, row
    """
    if not _HAS_GPD:
        logger.warning(
            "geopandas/shapely non installes (pip install geopandas shapely) "
            "-- GPKG/KML non generes, seuls les GeoTIFF ont ete ecrits."
        )
        return None
    if not gpkg_rows:
        logger.warning("Aucun exemple valide -- GPKG/KML non generes.")
        return None

    records = []
    for r in gpkg_rows:
        transform = r["transform"]
        x0, y0 = transform.c, transform.f
        x1 = x0 + PATCH_SIZE_PX * transform.a
        y1 = y0 + PATCH_SIZE_PX * transform.e
        geom_utm = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

        gdf_row = gpd.GeoDataFrame(
            {"row": [r["row"]], "continent": [r["continent"]], "etape_k": [r["etape_k"]],
             "true_lcz": [r["true_lcz"]], "pred_lcz": [r["pred_lcz"]],
             "correct": [r["correct"]], "epsg_source": [r["epsg"]]},
            geometry=[geom_utm], crs=f"EPSG:{r['epsg']}",
        )
        # CRS unique de stockage GPKG/KML
        records.append(gdf_row.to_crs(epsg=4326))

    gdf_all = pd.concat(records, ignore_index=True)
    return gpd.GeoDataFrame(gdf_all, geometry="geometry", crs="EPSG:4326")


def build_examples_gpkg(gdf_all, out_path: Path, layer_name: str = "lcz_examples"):
    """
    Ecrit UNE couche vectorielle GPKG avec un polygone (l'emprise du patch)
    par exemple correct/wrong deja exporte en GeoTIFF -- meme ensemble de
    patches, juste une representation vectorielle legere en plus, pratique
    pour tout voir d'un coup dans QGIS sans charger des dizaines de rasters
    un par un. C'est un des deux recapitulatifs structures produits par ce
    script (l'autre etant le KML pour Google Earth Pro, cf.
    build_examples_kml) : pas de CSV a cote (cf. save_examples_geotiff), le
    GPKG porte deja toute l'information utile et ne peut pas mentir sur un
    fichier non ecrit puisqu'il n'est rempli qu'a partir des GeoTIFF
    reellement ecrits.
    """
    if gdf_all is None:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf_all.to_file(out_path, layer=layer_name, driver="GPKG")
    logger.info(f"GPKG ecrit : {out_path} ({len(gdf_all)} patches)")
    return str(out_path)


def hex_to_kml_color(hex_rgb: str, alpha_hex: str = "80") -> str:
    """RRGGBB (comme LCZ_COLORS_HEX/QGIS) -> AABBGGRR (ordre KML)."""
    r, g, b = hex_rgb[0:2], hex_rgb[2:4], hex_rgb[4:6]
    return f"{alpha_hex}{b}{g}{r}".lower()


def build_examples_kml(gdf_all, out_path: Path,
                       lcz_classes: list = LCZ_CLASSES_17,
                       colors_hex: list = LCZ_COLORS_HEX):
    """
    Ecrit un KML equivalent au GPKG (memes emprises de patch, en EPSG:4326
    comme l'exige KML), ouvrable directement dans Google Earth Pro sans
    passer par QGIS. Remplissage colore par classe LCZ vraie (meme palette
    que write_lcz_qml_style/le style QGIS categorise), contour vert si la
    prediction est correcte, rouge sinon -- un repere visuel immediat que le
    GPKG seul n'offre pas sans configurer un style QGIS a deux couches
    (cf. create_good_bad_gpkgs.py::write_qgis_style).
    """
    if gdf_all is None:
        return None

    fill_by_class = {
        cls_name: hex_to_kml_color(hexcode)
        for cls_name, hexcode in zip(lcz_classes, colors_hex)
    }
    OUTLINE_CORRECT = "ff00cc00"   # vert opaque (AABBGGRR)
    OUTLINE_WRONG = "ff0000ff"     # rouge opaque (AABBGGRR)

    styles_xml = []
    for cls_name, fill_color in fill_by_class.items():
        style_id = f"style_{lcz_classes.index(cls_name)}"
        for suffix, outline in (("_ok", OUTLINE_CORRECT), ("_wrong", OUTLINE_WRONG)):
            styles_xml.append(f'''
  <Style id="{style_id}{suffix}">
    <LineStyle><color>{outline}</color><width>2</width></LineStyle>
    <PolyStyle><color>{fill_color}</color><fill>1</fill><outline>1</outline></PolyStyle>
  </Style>''')

    placemarks_xml = []
    for _, r in gdf_all.iterrows():
        cls_idx = lcz_classes.index(r["true_lcz"])
        suffix = "_ok" if bool(r["correct"]) else "_wrong"
        style_id = f"style_{cls_idx}{suffix}"

        coords = " ".join(f"{lon},{lat},0" for lon,
                          lat in r.geometry.exterior.coords)
        name = f"{r['continent']} k={r['etape_k']} row={r['row']} vrai={r['true_lcz']} pred={r['pred_lcz']}"

        placemarks_xml.append(f'''
  <Placemark>
    <name>{name}</name>
    <styleUrl>#{style_id}</styleUrl>
    <ExtendedData>
      <Data name="continent"><value>{r['continent']}</value></Data>
      <Data name="etape_k"><value>{r['etape_k']}</value></Data>
      <Data name="true_lcz"><value>{r['true_lcz']}</value></Data>
      <Data name="pred_lcz"><value>{r['pred_lcz']}</value></Data>
      <Data name="correct"><value>{bool(r['correct'])}</value></Data>
      <Data name="row"><value>{r['row']}</value></Data>
    </ExtendedData>
    <Polygon>
      <outerBoundaryIs>
        <LinearRing>
          <coordinates>{coords}</coordinates>
        </LinearRing>
      </outerBoundaryIs>
    </Polygon>
  </Placemark>''')

    kml = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>{out_path.stem}</name>{"".join(styles_xml)}{"".join(placemarks_xml)}
</Document>
</kml>
'''
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(kml, encoding="utf-8")
    logger.info(f"KML ecrit : {out_path} ({len(gdf_all)} patches)")
    return str(out_path)


def write_lcz_qml_style(out_qml: Path, field_name: str = "true_lcz",
                        lcz_classes: list = LCZ_CLASSES_17,
                        colors_hex: list = LCZ_COLORS_HEX):
    """Style QGIS categorise sur `field_name`, palette LCZ officielle WUDAPT.
    A charger via : clic droit sur la couche > Proprietes > Symbologie >
    Style > Charger le style."""
    categories, symbols = "", ""
    for cls_name, hex_color in zip(lcz_classes, colors_hex):
        categories += (
            f'\n      <category value="{cls_name}" symbol="{cls_name}" '
            f'label="{cls_name}" render="true"/>'
        )
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        symbols += f'''
      <symbol type="fill" name="{cls_name}" force_rhr="0" alpha="1">
        <layer class="SimpleFill" enabled="1">
          <prop k="color" v="{r},{g},{b},255"/>
          <prop k="outline_color" v="35,35,35,255"/>
          <prop k="outline_width" v="0.2"/>
          <prop k="style" v="solid"/>
        </layer>
      </symbol>'''

    qml = f'''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28">
  <renderer-v2 type="categorizedSymbol" attr="{field_name}">
    <categories>{categories}
    </categories>
    <symbols>{symbols}
    </symbols>
  </renderer-v2>
</qgis>
'''
    out_qml.parent.mkdir(parents=True, exist_ok=True)
    out_qml.write_text(qml, encoding="utf-8")
    logger.info(f"Style QGIS ecrit : {out_qml}")
    return str(out_qml)


def save_examples_geotiff(examples, LCZ_CLASSES, k, continent, out_root="figures/exemples"):
    """
    Callback `on_examples` pour `testing_model.run_one` : ecrit un GeoTIFF
    Sentinel-2 par exemple correct/faux, puis GPKG (QGIS) + KML (Google Earth
    Pro) avec les footprints de ces memes exemples -- deux formats au choix
    pour la meme information, sans CSV intermediaire (cf. docstrings de
    build_examples_gpkg/build_examples_kml).
    """
    out_dir = Path(out_root) / f"etape_{k}_{continent}"
    out_dir.mkdir(parents=True, exist_ok=True)

    gpkg_rows = []

    for true_c in range(len(LCZ_CLASSES)):
        info = examples.get(true_c)
        if info is None:
            continue
        cls_name = LCZ_CLASSES[true_c]

        for kind in ("correct", "wrong"):
            entry = info.get(kind)
            if entry is None:
                continue

            if kind == "correct":
                idx_dataset = entry
                pred_c = true_c
            else:
                idx_dataset, pred_c = entry

            result = write_geotiff_for_example(
                continent, k, idx_dataset, kind, true_c, pred_c, LCZ_CLASSES, out_dir
            )
            if result is None:
                continue

            gpkg_rows.append({
                "transform": result["transform"], "epsg": result["epsg"],
                "row": result["real_row"], "continent": continent, "etape_k": k,
                "true_lcz": cls_name, "pred_lcz": LCZ_CLASSES[pred_c],
                "correct": bool(kind == "correct"),
            })

    logger.info(f"GeoTIFFs sauvegardes dans {out_dir}")

    gdf_all = _build_examples_gdf(gpkg_rows)

    gpkg_path = out_dir / f"exemples_{continent}_etape{k}.gpkg"
    build_examples_gpkg(gdf_all, gpkg_path)

    kml_path = out_dir / f"exemples_{continent}_etape{k}.kml"
    build_examples_kml(gdf_all, kml_path)

    qml_path = Path(out_root) / "lcz_style_true.qml"
    if not qml_path.exists():
        write_lcz_qml_style(qml_path, field_name="true_lcz")


def main(k):
    args = parse_args()
    run_loop(
        k, GEO_CONTINENTS, args.save_examples, args.examples_dir,
        on_examples=save_examples_geotiff,
        summary_path="figures/resume_test_continents_geotiff.csv",
    )


if __name__ == "__main__":
    for k in [3, 4, 6]:
        main(k)
