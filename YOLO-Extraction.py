# -*- coding: utf-8 -*-
"""
YOLO-Extraction.py
Extract building footprints from GoogleMap-Images.tif using a YOLOv8
instance-segmentation model pretrained on satellite building imagery
(keremberke/yolov8m-building-segmentation).

The image is processed in overlapping tiles; per-tile masks are merged
back into a full-resolution binary mask.

Outputs:
    YOLO-Extraction.tif      binary building mask (georeferenced, EPSG:3857)
    YOLO-Extraction.geojson  building polygons (EPSG:4326)

Usage:
    python YOLO-Extraction.py [--input GoogleMap-Images.tif]
                              [--conf 0.25] [--tile 640] [--overlap 128]
"""

import argparse
import os

# A system-wide PROJ_LIB/PROJ_DATA (e.g. from PostgreSQL/PostGIS) can point to an
# incompatible proj.db — clear it so rasterio uses its own bundled PROJ database.
os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)

import geopandas as gpd
import numpy as np
import rasterio
import torch
from huggingface_hub import hf_hub_download
from samgeo import raster_to_vector
from ultralytics import YOLO

MODEL_REPO = "keremberke/yolov8m-building-segmentation"
MODEL_FILE = "best.pt"


def main():
    parser = argparse.ArgumentParser(description="Extract buildings with YOLOv8 segmentation")
    parser.add_argument("--input", default="GoogleMap-Images.tif", help="Input georeferenced image")
    parser.add_argument("--mask", default="YOLO-Extraction.tif", help="Output mask GeoTIFF")
    parser.add_argument("--vector", default="YOLO-Extraction.geojson", help="Output GeoJSON")
    parser.add_argument("--conf", type=float, default=0.30, help="Detection confidence threshold")
    parser.add_argument("--tile", type=int, default=640, help="Tile size in pixels")
    parser.add_argument("--overlap", type=int, default=128, help="Tile overlap in pixels")
    parser.add_argument("--min-area", type=float, default=10.0,
                        help="Drop polygons smaller than this area in m2")
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device: {'cuda (' + torch.cuda.get_device_name(0) + ')' if device == 0 else 'cpu'}")

    print(f"Loading {MODEL_REPO} ...")
    model = YOLO(hf_hub_download(MODEL_REPO, MODEL_FILE))

    with rasterio.open(args.input) as src:
        img = src.read().transpose(1, 2, 0)  # HWC, RGB
        profile = src.profile
    height, width = img.shape[:2]

    full_mask = np.zeros((height, width), dtype=np.uint8)
    step = args.tile - args.overlap
    xs = list(range(0, max(width - args.overlap, 1), step))
    ys = list(range(0, max(height - args.overlap, 1), step))
    n_tiles = len(xs) * len(ys)
    print(f"Processing {n_tiles} tiles ({len(xs)} x {len(ys)}, "
          f"{args.tile}px, overlap {args.overlap}px) ...")

    done = 0
    for y0 in ys:
        for x0 in xs:
            x1, y1 = min(x0 + args.tile, width), min(y0 + args.tile, height)
            tile = img[y0:y1, x0:x1]
            # BGR expected by ultralytics for ndarray input
            results = model.predict(tile[:, :, ::-1], imgsz=args.tile, conf=args.conf,
                                    device=device, retina_masks=True, verbose=False)
            r = results[0]
            if r.masks is not None:
                tile_mask = r.masks.data.any(dim=0).cpu().numpy()
                full_mask[y0:y1, x0:x1] |= tile_mask[: y1 - y0, : x1 - x0].astype(np.uint8)
            done += 1
        print(f"  {done}/{n_tiles} tiles")

    print(f"Building pixels: {100 * full_mask.mean():.1f}%")

    profile.update(count=1, dtype="uint8", nodata=None)
    with rasterio.open(args.mask, "w", **profile) as dst:
        dst.write(full_mask * 255, 1)
    print(f"Saved mask {args.mask}")

    raster_to_vector(args.mask, args.vector, dst_crs="EPSG:4326")

    # Drop speck polygons below the minimum area (measured in EPSG:3857 meters)
    gdf = gpd.read_file(args.vector)
    n_before = len(gdf)
    gdf = gdf[gdf.geometry.to_crs("EPSG:3857").area >= args.min_area]
    gdf.to_file(args.vector, driver="GeoJSON")
    print(f"Saved vector {args.vector} ({len(gdf)} buildings, "
          f"dropped {n_before - len(gdf)} specks < {args.min_area} m2)")


if __name__ == "__main__":
    main()
