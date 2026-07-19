# -*- coding: utf-8 -*-
"""
GoogleMap-Download.py
Download Google Maps satellite tiles covering a bounding box and merge
them into a georeferenced GeoTIFF (EPSG:3857).

Usage:
    python GoogleMap-Download.py [--zoom 18] [--output GoogleMap-Images.tif]
"""

import argparse
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

# A system-wide PROJ_LIB/PROJ_DATA (e.g. from PostgreSQL/PostGIS) can point to an
# incompatible proj.db — clear it so rasterio uses its own bundled PROJ database.
os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)

import numpy as np
import requests
from PIL import Image
import rasterio
from rasterio.transform import Affine

# Bounding box from ee.Geometry.Polygon (lon/lat, EPSG:4326)
WEST = 100.56242857150896
SOUTH = 14.349677166816866
EAST = 100.57015333347185
NORTH = 14.355331490931219

TILE_SIZE = 256
EARTH_RADIUS = 6378137.0
ORIGIN_SHIFT = math.pi * EARTH_RADIUS  # 20037508.342789244

TILE_URL = "https://mt{server}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def lonlat_to_global_pixel(lon, lat, zoom):
    """Convert lon/lat to global pixel coordinates at the given zoom."""
    n = 2 ** zoom * TILE_SIZE
    px = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    py = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return px, py


def global_pixel_to_mercator(px, py, zoom):
    """Convert global pixel coordinates to EPSG:3857 meters."""
    res = (2.0 * ORIGIN_SHIFT) / (2 ** zoom * TILE_SIZE)
    mx = px * res - ORIGIN_SHIFT
    my = ORIGIN_SHIFT - py * res
    return mx, my


def download_tile(x, y, zoom, session, retries=3):
    url = TILE_URL.format(server=(x + y) % 4, x=x, y=y, z=zoom)
    for attempt in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            return x, y, np.asarray(img, dtype=np.uint8)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser(description="Download Google satellite imagery to GeoTIFF")
    parser.add_argument("--zoom", type=int, default=18, help="Tile zoom level (default: 18)")
    parser.add_argument("--output", default="GoogleMap-Images.tif", help="Output GeoTIFF path")
    parser.add_argument("--workers", type=int, default=16, help="Parallel download threads")
    args = parser.parse_args()
    zoom = args.zoom

    # Pixel extent of the bbox in global pixel space
    px_min, py_min = lonlat_to_global_pixel(WEST, NORTH, zoom)   # top-left
    px_max, py_max = lonlat_to_global_pixel(EAST, SOUTH, zoom)   # bottom-right
    px_min, py_min = int(math.floor(px_min)), int(math.floor(py_min))
    px_max, py_max = int(math.ceil(px_max)), int(math.ceil(py_max))
    width, height = px_max - px_min, py_max - py_min

    # Tile range covering that pixel extent
    tx_min, ty_min = px_min // TILE_SIZE, py_min // TILE_SIZE
    tx_max, ty_max = (px_max - 1) // TILE_SIZE, (py_max - 1) // TILE_SIZE
    tiles = [(x, y) for y in range(ty_min, ty_max + 1) for x in range(tx_min, tx_max + 1)]
    print(f"Zoom {zoom}: {len(tiles)} tiles "
          f"({tx_max - tx_min + 1} x {ty_max - ty_min + 1}), output {width} x {height} px")

    mosaic = np.zeros((height, width, 3), dtype=np.uint8)
    session = requests.Session()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_tile, x, y, zoom, session) for x, y in tiles]
        for fut in as_completed(futures):
            x, y, data = fut.result()
            # Position of this tile inside the mosaic (may be partially outside)
            ox, oy = x * TILE_SIZE - px_min, y * TILE_SIZE - py_min
            sx0, sy0 = max(0, -ox), max(0, -oy)
            dx0, dy0 = max(0, ox), max(0, oy)
            sx1 = TILE_SIZE - max(0, ox + TILE_SIZE - width)
            sy1 = TILE_SIZE - max(0, oy + TILE_SIZE - height)
            mosaic[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = data[sy0:sy1, sx0:sx1]
            done += 1
            if done % 100 == 0 or done == len(tiles):
                print(f"  downloaded {done}/{len(tiles)} tiles")

    # Georeference: top-left corner and resolution in EPSG:3857
    mx_min, my_max = global_pixel_to_mercator(px_min, py_min, zoom)
    res = (2.0 * ORIGIN_SHIFT) / (2 ** zoom * TILE_SIZE)
    transform = Affine(res, 0.0, mx_min, 0.0, -res, my_max)

    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:3857",
        "transform": transform,
        "compress": "lzw",
        "tiled": True,
        "photometric": "RGB",
    }
    with rasterio.open(args.output, "w", **profile) as dst:
        dst.write(mosaic.transpose(2, 0, 1))

    # Pyramid levels: halve until the smallest overview is under ~256 px
    levels = []
    factor = 2
    while max(width, height) // factor >= 256:
        levels.append(factor)
        factor *= 2

    # External pyramid: TIFF_USE_OVR forces overviews into a sidecar .ovr file
    with rasterio.Env(TIFF_USE_OVR=True):
        with rasterio.open(args.output, "r+") as dst:
            dst.build_overviews(levels, rasterio.enums.Resampling.average)

    print(f"Saved {args.output} ({width} x {height}, ~{res:.2f} m/px, EPSG:3857)")
    print(f"External pyramid {args.output}.ovr levels {levels}")


if __name__ == "__main__":
    main()
