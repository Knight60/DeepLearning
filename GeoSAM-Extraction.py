# -*- coding: utf-8 -*-
"""
GeoSAM-Extraction.py
Extract building footprints from GoogleMap-Images.tif using LangSAM
(GroundingDINO text prompt + Segment Anything Model) via segment-geospatial.

Outputs:
    GeoSAM-Extraction.tif      binary building mask (georeferenced, EPSG:3857)
    GeoSAM-Extraction.geojson  building polygons (EPSG:4326)

Usage:
    python GeoSAM-Extraction.py [--input GoogleMap-Images.tif]
                                [--prompt building]
                                [--box-threshold 0.24] [--text-threshold 0.24]
"""

import argparse
import os

# A system-wide PROJ_LIB/PROJ_DATA (e.g. from PostgreSQL/PostGIS) can point to an
# incompatible proj.db — clear it so rasterio uses its own bundled PROJ database.
os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)

import geopandas as gpd
import rasterio
import torch
from samgeo.text_sam import LangSAM
from samgeo import raster_to_vector


def main():
    parser = argparse.ArgumentParser(description="Extract buildings with LangSAM (GroundingDINO + SAM)")
    parser.add_argument("--input", default="GoogleMap-Images.tif", help="Input georeferenced image")
    parser.add_argument("--mask", default="GeoSAM-Extraction.tif", help="Output mask GeoTIFF")
    parser.add_argument("--vector", default="GeoSAM-Extraction.geojson", help="Output GeoJSON")
    parser.add_argument("--prompt", default="building", help="Text prompt for detection")
    parser.add_argument("--box-threshold", type=float, default=0.20, help="GroundingDINO box threshold")
    parser.add_argument("--text-threshold", type=float, default=0.24, help="GroundingDINO text threshold")
    parser.add_argument("--max-box-frac", type=float, default=0.4,
                        help="Reject detection boxes covering more than this fraction of the image")
    parser.add_argument("--min-area", type=float, default=10.0,
                        help="Drop polygons smaller than this area in m2")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    sam = LangSAM()

    # GroundingDINO often returns one low-confidence box spanning the whole
    # scene; dropping oversized boxes keeps only individual buildings.
    with rasterio.open(args.input) as src:
        img_area = src.width * src.height

    def keep_detection(box, mask, logit, phrase, index):
        x0, y0, x1, y1 = [float(v) for v in box]
        frac = (x1 - x0) * (y1 - y0) / img_area
        if frac > args.max_box_frac:
            print(f"  dropping box {index} covering {frac:.0%} of the image")
            return False
        return True

    print(f"Detecting '{args.prompt}' in {args.input} ...")
    sam.predict(
        args.input,
        args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        output=args.mask,
        dtype="uint8",
        detection_filter=keep_detection,
    )
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
