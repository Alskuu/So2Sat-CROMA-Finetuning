"""
geo_utils.py
------------
Fonctions de géolocalisation partagées entre segmentation_set.py et le
notebook geolocations.ipynb : conversion des coordonnées UTM (+ EPSG par
point) du dataset So2Sat vers lon/lat, et construction du GeoDataFrame de
points qui en résulte.

Note : save_geotiff.py ne réutilise pas ces fonctions car il travaille
directement sur le GeoTransform (Affine) de chaque patch plutôt que sur des
coordonnées ponctuelles x/y + EPSG.
"""

import numpy as np
import geopandas as gpd
from pyproj import Transformer

# Shapefile Natural Earth utilisé pour les jointures spatiales pays/continent
COUNTRIES_SHP = "D:/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"


def epsg327_false_northing_fix(y_m: np.ndarray, epsg: np.ndarray) -> np.ndarray:
    """EPSG 327xx (UTM hemisphere sud) : false northing = 10 000 000 m a l'equateur."""
    y_fixed = y_m.copy()
    mask = (epsg >= 32701) & (epsg <= 32760) & (y_fixed < 0)
    y_fixed[mask] += 10_000_000.0
    return y_fixed


def to_lonlat(x_m: np.ndarray, y_m: np.ndarray, epsg: np.ndarray):
    """Convertit des coordonnées projetées (x, y) en lon/lat, un EPSG par point."""
    lon = np.zeros(len(x_m), dtype=np.float64)
    lat = np.zeros(len(y_m), dtype=np.float64)
    for code in np.unique(epsg):
        mask = epsg == code
        tr = Transformer.from_crs(int(code), 4326, always_xy=True)
        lon[mask], lat[mask] = tr.transform(x_m[mask], y_m[mask])
    return lon, lat


def points_gdf_from_lonlat(lon: np.ndarray, lat: np.ndarray, idx: np.ndarray = None, extra_cols: dict = None) -> gpd.GeoDataFrame:
    """Construit un GeoDataFrame de points EPSG:4326 avec une colonne idx (et colonnes optionnelles)."""
    n = len(lon)
    data = {"idx": idx if idx is not None else np.arange(n, dtype=np.int64)}
    if extra_cols:
        data.update(extra_cols)
    return gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(lon, lat),
        crs="EPSG:4326",
    )
