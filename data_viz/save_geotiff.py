# %% -*- coding: utf-8 -*-
import os
import h5py
import numpy as np
import rasterio
from rasterio.crs import CRS
from affine import Affine

# ============================================================
# Config
# ============================================================
new_root = r'D:\LCZ42'
geo_fname = r'training.h5'
aux_fname = r'training_geo.h5'

max_export_per_city = 5

expected_pixel_size = 10.0
pixel_tol_ratio = 0.05
PATCH_SIZE_PX = 32  # patches So2Sat toujours 32x32

out_root = os.path.join(new_root, "exports_geotiffs/training")
os.makedirs(out_root, exist_ok=True)

# training set
city_epsg = {
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
    "wuhan": 32650, "zurich": 32632
}

# val and test set
# city_epsg = {
#     "guangzhou": 32649,
#     "jakarta": 32748,
#     "moscow": 32637,
#     "mumbai": 32643,
#     "munich": 32632,
#     "nairobi": 32737,
#     "sanfrancisco": 32610,
#     "santiago": 32719,
#     "sydney": 32756,
#     "tehran": 32639
# }

# ============================================================
# Helpers
# ============================================================


def norm_city_name(x) -> str:
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", errors="ignore").strip().lower()
    return str(x).strip().lower()


def safe_name(s: str) -> str:
    return "".join([c if c.isalnum() or c in ("_", "-") else "_" for c in s])

# The isalnum() method returns True if all the characters are alphanumeric, meaning alphabet letter (a-z) and numbers (0-9).


def geotransform_to_affine(geo6: np.ndarray, patch_size: int = PATCH_SIZE_PX) -> Affine:
    """
    Convertit un GeoTransform GDAL brut (6 valeurs) en objet Affine rasterio.

    Correction empirique (meme constat que dans
    cromacode/testing_model_contiboucle_geotiff.py, verifie visuellement sous
    QGIS) : le y_origin fourni par les donnees place systematiquement la
    geometrie une hauteur de patch trop au nord -- patch_size * pixel_height.
    On corrige en soustrayant cette hauteur avant construction de l'Affine.
    """
    geo6 = [float(v) for v in np.asarray(geo6).reshape(-1)]
    x_origin, pixel_width, x_skew, y_origin, y_skew, pixel_height = geo6
    y_origin_corrected = y_origin - patch_size * pixel_height
    geo6_corrected = [x_origin, pixel_width, x_skew,
                       y_origin_corrected, y_skew, pixel_height]
    return Affine.from_gdal(*geo6_corrected)


def affine_to_geotransform_params(transform: Affine) -> np.ndarray:
    A = float(transform.a)
    B = float(transform.b)
    C = float(transform.c)
    D = float(transform.d)
    E = float(transform.e)
    F = float(transform.f)
    return np.array([C, A, B, F, D, E])


def check_transform_sane(transform: Affine, expected_px: float, tol_ratio: float = 0.05):
    a = float(transform.a)
    b = float(transform.b)
    d = float(transform.d)
    e = float(transform.e)

    # On vérifie que la largeur(a) et la hauteur (e) du pixel soient finis, de même pour l'inclinaison et la rotation de l'image
    if not np.isfinite([a, b, d, e]).all():
        return False, f"non-finite: a={a},b={b},d={d},e={e}"

    # Marges d'erreur que l'on accepte ou non
    lo = expected_px * (1 - tol_ratio)
    hi = expected_px * (1 + tol_ratio)

    if not (lo <= abs(a) <= hi):
        return False, f"|a| not ~ {expected_px}: a={a}"
    if not (lo <= abs(e) <= hi):
        return False, f"|e| not ~ {expected_px}: e={e}"
    if abs(b) > 1e-6 or abs(d) > 1e-6:
        return False, f"rotation terms not near 0: b={b}, d={d}"
    return True, f"sane: a={a}, e={e}, b={b}, d={d}"


# ============================================================
# Main
# ============================================================
h5_geo_path = os.path.join(new_root, geo_fname)
h5_aux_path = os.path.join(new_root, aux_fname)

with h5py.File(h5_geo_path, "r") as f_geo, h5py.File(h5_aux_path, "r") as f_aux:
    d_sen1 = f_geo["sen1"]   # (N,H,W,C1)
    d_sen2 = f_geo["sen2"]   # (N,H,W,C2)

    d_gt = f_aux["tfw"]    # (N,6) GeoTransform order
    d_epsg = f_aux["epsg"]   # (N,1) or (N,)
    d_city = f_aux["city"] if "city" in f_aux else f_aux["cityList"]

    N = d_sen1.shape[0]
    print("Total N:", N)

    city_all = np.array(d_city[:])  # bytes
    city_all = np.array([norm_city_name(c) for c in city_all], dtype=object)
    epsg_all = np.squeeze(np.array(d_epsg[:])).astype(int)

    total_exported = 0
    total_skipped = 0

    for city in city_epsg.keys():
        city_key = city.strip().lower()
        idx = np.where(city_all == city_key)[0]
        print(f"\n=== City '{city_key}': found {len(idx)} samples ===")

        if len(idx) == 0:
            print("  [WARN] no samples for this city in aux city list.")
            continue

        idx = idx[:max_export_per_city]

        city_dir = os.path.join(out_root, safe_name(city_key))
        os.makedirs(city_dir, exist_ok=True)

        exported = 0
        skipped = 0

        for global_i in idx:
            epsg_code = int(epsg_all[global_i])

            # CRS
            try:
                crs_obj = CRS.from_epsg(epsg_code)
            except Exception as e:
                skipped += 1
                total_skipped += 1
                print(
                    f"  [SKIP] idx={global_i} invalid EPSG:{epsg_code}. err={e}")
                continue

            # Geotransform -> Affine
            gt6 = np.array(d_gt[global_i], dtype=np.float64).reshape(-1)
            if gt6.shape[0] != 6:
                skipped += 1
                total_skipped += 1
                print(
                    f"  [SKIP] idx={global_i} bad geotransform shape: {gt6.shape}")
                continue

            transform = geotransform_to_affine(gt6)

            # read patch
            sen1_hwc = d_sen1[global_i]
            sen2_hwc = d_sen2[global_i]
            if sen1_hwc.ndim != 3 or sen2_hwc.ndim != 3:
                skipped += 1
                total_skipped += 1
                print(
                    f"  [SKIP] idx={global_i} bad patch shape sen1={sen1_hwc.shape}, sen2={sen2_hwc.shape}")
                continue

            H, W, C1 = sen1_hwc.shape
            _, _, C2 = sen2_hwc.shape

            ok, reason = check_transform_sane(
                transform, expected_pixel_size, pixel_tol_ratio)
            if not ok:
                skipped += 1
                total_skipped += 1
                print(
                    f"  [SKIP] idx={global_i} transform insane: {reason} epsg={epsg_code}")
                continue

            # HWC -> CHW
            sen1_chw = np.transpose(sen1_hwc, (2, 0, 1))
            sen2_chw = np.transpose(sen2_hwc, (2, 0, 1))

            # Flip Nord-Sud : la ligne 0 du tableau correspond au SUD de la
            # scene (meme constat empirique que dans
            # testing_model_contiboucle_geotiff.py), pas au nord comme
            # l'attend la convention image standard combinee a notre
            # geotransform (pixel_height > 0). np.flip() renvoie une vue a
            # stride negatif -> .copy() pour un tableau contigu, necessaire
            # pour rasterio/GDAL.
            sen1_chw = np.flip(sen1_chw, axis=1).copy()
            sen2_chw = np.flip(sen2_chw, axis=1).copy()

            # output paths
            sen1_out = os.path.join(
                city_dir, f"{safe_name(city_key)}_sen1_{int(global_i)}_epsg{epsg_code}.tif")
            sen2_out = os.path.join(
                city_dir, f"{safe_name(city_key)}_sen2_{int(global_i)}_epsg{epsg_code}.tif")

            # write GeoTIFF
            with rasterio.open(
                sen1_out, "w",
                driver="GTiff",
                height=H, width=W,
                count=C1,
                dtype=sen1_chw.dtype,
                crs=crs_obj,
                transform=transform,
                tiled=True,
                compress="DEFLATE"
            ) as dst:
                dst.write(sen1_chw)

            with rasterio.open(
                sen2_out, "w",
                driver="GTiff",
                height=H, width=W,
                count=C2,
                dtype=sen2_chw.dtype,
                crs=crs_obj,
                transform=transform,
                tiled=True,
                compress="DEFLATE"
            ) as dst:
                dst.write(sen2_chw)

            exported += 1
            total_exported += 1
            print(f"  [OK] idx={global_i} epsg={epsg_code} saved")

        print(f"City done: exported={exported}, skipped={skipped}")

print("\n====================================================")
print("ALL DONE.")
print(f"total_exported={total_exported}, total_skipped={total_skipped}")
print(f"Outputs saved to:\n{out_root}")
