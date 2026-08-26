"""
Convertit les GeoTIFF scientifiques produits par
create_gpkg_kml.py::write_geotiff_for_example (10 bandes Sentinel-2,
float32, valeurs de reflectance brutes non bornees 0-255) en copies 8-bit
RGB "pretes a afficher", seul format que Google Earth Pro sait rendre
correctement -- contrairement a QGIS, GEP ne gere pas les rasters
scientifiques multi-bandes/haute profondeur.

Le georeferencement (CRS + transform affine) est un ensemble de metadonnees
GDAL independant du type de pixel : la copie garde le meme `crs` et le meme
`transform` que l'original, donc s'ouvre exactement a la meme position dans
GEP -- seul le format des pixels change (10 bandes float32 -> 3 bandes uint8).

Parcourt figures/exemples/etape_*_*/*.tif et ecrit chaque copie dans
figures/exemples/GEP/etape_{k}_{continent}/<meme nom de fichier> (structure
miroir, jamais reparcourue par le scan puisque "GEP" ne matche pas le motif
"etape_*_*").
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio


def stretch_to_uint8(band: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    """Etirement de contraste par percentiles (clip [low_pct, high_pct])
    puis mise a l'echelle vers uint8 -- methode standard pour les
    composites RGB satellite, evite qu'un pixel aberrant (nuage, reflet)
    n'ecrase tout le contraste comme le ferait un min-max naif."""
    lo, hi = np.percentile(band, [low_pct, high_pct])
    if hi <= lo:
        # Bande quasi constante (ex. patch entierement nuageux/uniforme) :
        # pas d'etirement possible, renvoie du gris moyen plutot que de
        # diviser par zero.
        return np.full(band.shape, 128, dtype=np.uint8)
    clipped = np.clip(band, lo, hi)
    scaled = (clipped - lo) / (hi - lo) * 255.0
    return scaled.astype(np.uint8)


def convert_one(src_path: Path, dst_path: Path, low_pct: float, high_pct: float):
    with rasterio.open(src_path) as src:
        # Bandes 1-3 = R,G,B (deja dans cet ordre grace a `band_order` dans
        # write_geotiff_for_example) -- on ignore les bandes 4-10.
        rgb = src.read([1, 2, 3]).astype(np.float64)
        profile = src.profile

    rgb_8bit = np.stack([
        stretch_to_uint8(rgb[i], low_pct, high_pct) for i in range(3)
    ])

    profile.update(count=3, dtype="uint8", compress="DEFLATE", photometric="RGB")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(rgb_8bit)


def main():
    p = argparse.ArgumentParser(
        description="Cree des copies 8-bit RGB des GeoTIFF d'exemples, pour visualisation "
                    "dans Google Earth Pro (georeferencement preserve).")
    p.add_argument("--examples-dir", type=str, default="figures/exemples")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Defaut : <examples-dir>/GEP")
    p.add_argument("--low-pct", type=float, default=2.0)
    p.add_argument("--high-pct", type=float, default=98.0)
    p.add_argument("--dry-run", action="store_true",
                   help="Affiche les conversions sans les executer.")
    args = p.parse_args()

    examples_dir = Path(args.examples_dir)
    out_dir = Path(args.out_dir) if args.out_dir else examples_dir / "GEP"

    if not examples_dir.exists():
        raise SystemExit(f"Dossier introuvable : {examples_dir}")

    tif_paths = sorted(examples_dir.glob("etape_*_*/*.tif"))
    if not tif_paths:
        print(f"Aucun GeoTIFF trouve sous {examples_dir}/etape_*_*/*.tif.")
        return

    n_done = 0
    for src_path in tif_paths:
        rel = src_path.relative_to(examples_dir)  # etape_k_continent/fichier.tif
        dst_path = out_dir / rel

        print(f"{'[DRY-RUN] ' if args.dry_run else ''}{src_path} -> {dst_path}")
        if not args.dry_run:
            convert_one(src_path, dst_path, args.low_pct, args.high_pct)
        n_done += 1

    verb = "aurait converti" if args.dry_run else "a converti"
    print(f"\nTermine : {verb} {n_done} GeoTIFF vers {out_dir}.")


if __name__ == "__main__":
    main()
