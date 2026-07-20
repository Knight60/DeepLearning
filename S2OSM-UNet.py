# -*- coding: utf-8 -*-
"""
S2OSM-UNet.py
Sentinel-2 LULC classification with a U-Net (encoder-decoder + skip connections),
compared head-to-head against the CNN-1D and CNN-2D models from S2OSM-CNN.py on
the same BigQuery training samples and the same 70/30 split.

The BigQuery export only has point labels (one class per point, no dense masks),
so U-Net is trained with point supervision: each training point becomes a 32x32
patch, and the per-pixel loss is masked to the single center pixel that has a
known label (sample_weight=1 there, 0 everywhere else). At inference the network
is fully convolutional, so the whole AOI is classified in large tiles in one pass
per tile instead of one patch per pixel.

Reuses artifacts from a previous S2OSM-CNN.py run when present (full_composite.tif,
cnn1d_s2osm.keras, cnn2d_s2osm.keras, scaler_1d.json) instead of recomputing them.

Outputs:
    unet_s2osm.keras                        trained U-Net
    unet_comparison_report.txt              CNN-1D vs CNN-2D vs U-Net metrics
    classified_unet.tif                     whole-AOI U-Net map (GeoTIFF)
    classification_comparison_3way.png      true color vs CNN-1D vs CNN-2D vs U-Net

Usage:
    python S2OSM-UNet.py [--project ee-dancingriver2] [--patch-size 32] [--epochs 60]
"""

import argparse
import json
import os

# A system-wide PROJ_LIB/PROJ_DATA (e.g. from PostgreSQL/PostGIS) can point to an
# outdated proj.db that GDAL/rasterio in this venv cannot use. Redirect PROJ to the
# data bundled with rasterio so CRS lookups work inside this environment only.
os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)

# tensorflow's bundled oneDNN/OpenMP runtime must initialize before GDAL-based
# libraries (rasterio, geemap/ee) load their own copy, or its DLL init fails on
# Windows (ERROR_DLL_INIT_FAILED).
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

import ee
import geemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

PROJECT_ID = "ee-dancingriver2"
BQ_TABLE = "`ee-dancingriver2.ML.s2_training4classes`"
AOI_BOUNDS = [100.45, 14.25, 100.65, 14.45]
AOI = None  # built in main(), after ee.Initialize()
BANDS = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "EVI", "NDWI", "NDBI"]
S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
CLASS_PALETTE = ["#0000FF", "#FF0000", "#FFFF00", "#00FF00"]  # water / urban / agri / tree, label order 1..4
CNN2D_PATCH = 9  # must match S2OSM-CNN.py's --patch-size default
RANDOM_SEED = 42


# ================== Earth Engine + BigQuery ==================

def init_ee(project_id):
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)


def load_samples():
    return ee.FeatureCollection.runBigQuery(query=f"SELECT * FROM {BQ_TABLE}", geometryColumn="geo")


def fc_to_dataframe(fc, batch_size=5000):
    total = fc.size().getInfo()
    fc_list = fc.toList(total)
    rows = []
    for start in range(0, total, batch_size):
        chunk = ee.FeatureCollection(fc_list.slice(start, min(start + batch_size, total)))
        for feat in chunk.getInfo()["features"]:
            props = dict(feat["properties"])
            lon, lat = feat["geometry"]["coordinates"]
            props["lon"], props["lat"] = lon, lat
            rows.append(props)
    return pd.DataFrame(rows)


# ================== Sentinel-2 composite (shared by patch sampling + full-AOI classification) ==================

def mask_s2(img):
    scl = img.select("SCL")
    clear = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return img.updateMask(clear).divide(10000).copyProperties(img, ["system:time_start"])


def build_composite():
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(AOI)
        .filterDate("2025-11-01", "2026-02-28")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .map(mask_s2)
        .median()
        .select(S2_BANDS)
        .clip(AOI)
    )
    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
    evi = s2.expression(
        "2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)",
        {"NIR": s2.select("B8"), "RED": s2.select("B4"), "BLUE": s2.select("B2")},
    ).rename("EVI")
    ndwi = s2.normalizedDifference(["B3", "B8"]).rename("NDWI")
    ndbi = s2.normalizedDifference(["B11", "B8"]).rename("NDBI")
    return s2.addBands([ndvi, evi, ndwi, ndbi]).select(BANDS)


# ================== spatial patch sampling (shared: 32x32 for U-Net, sliced to 9x9 for CNN-2D reuse) ==================

def sample_patches(df, composite, patch_size, scale=10, batch_size=150):
    radius = patch_size // 2
    full = 2 * radius + 1
    patch_image = composite.neighborhoodToArray(ee.Kernel.square(radius, "pixels"))

    n = len(df)
    patches = np.zeros((n, patch_size, patch_size, len(BANDS)), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)

    for start in range(0, n, batch_size):
        chunk = df.iloc[start:start + batch_size]
        feats = [
            ee.Feature(ee.Geometry.Point([row.lon, row.lat]), {"idx": int(i)})
            for i, row in chunk.iterrows()
        ]
        sampled = patch_image.sampleRegions(collection=ee.FeatureCollection(feats), scale=scale, geometries=False)
        for feat in sampled.getInfo()["features"]:
            idx = feat["properties"]["idx"]
            try:
                stacked = np.stack(
                    [np.array(feat["properties"][b], dtype=np.float32) for b in BANDS], axis=-1
                )
            except (KeyError, ValueError):
                continue
            if stacked.shape == (full, full, len(BANDS)):
                patches[idx] = stacked[:patch_size, :patch_size, :]  # crop (2r+1) down to patch_size
                valid[idx] = True
        print(f"  patches sampled: {min(start + batch_size, n)}/{n}")

    return patches, valid


def scale_patches(patches, scaler):
    mean = scaler.mean_.reshape(1, 1, 1, -1)
    scale = scaler.scale_.reshape(1, 1, 1, -1)
    return (patches - mean) / scale


# ================== U-Net ==================

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return x


def build_unet(patch_size, n_bands, n_classes):
    inputs = layers.Input(shape=(patch_size, patch_size, n_bands))
    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)
    bottleneck = conv_block(p2, 128)
    u2 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(bottleneck)
    u2 = layers.Concatenate()([u2, c2])
    d2 = conv_block(u2, 64)
    u1 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(d2)
    u1 = layers.Concatenate()([u1, c1])
    d1 = conv_block(u1, 32)
    outputs = layers.Conv2D(n_classes, 1, activation="softmax")(d1)
    model = models.Model(inputs, outputs, name="unet")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_point_targets(labels, patch_size, center_idx):
    n = len(labels)
    y = np.zeros((n, patch_size, patch_size), dtype=np.int32)
    w = np.zeros((n, patch_size, patch_size), dtype=np.float32)
    y[:, center_idx, center_idx] = labels
    w[:, center_idx, center_idx] = 1.0
    return y, w


def classify_unet_full(model, arr, scaler, block=256):
    n_bands, h, w = arr.shape
    hwc = np.nan_to_num(np.transpose(arr, (1, 2, 0)))
    mean = scaler.mean_.reshape(1, 1, -1)
    scale = scaler.scale_.reshape(1, 1, -1)
    scaled = (hwc - mean) / scale
    pad_h, pad_w = (-h) % block, (-w) % block
    padded = np.pad(scaled, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    ph, pw = padded.shape[:2]
    out = np.zeros((ph, pw), dtype=np.int32)
    for r0 in range(0, ph, block):
        for c0 in range(0, pw, block):
            tile = padded[r0:r0 + block, c0:c0 + block, :][np.newaxis]
            out[r0:r0 + block, c0:c0 + block] = np.argmax(model.predict(tile, verbose=0)[0], axis=-1)
        print(f"  U-Net classifying rows {min(r0 + block, ph)}/{ph}")
    return out[:h, :w]


# ================== shared evaluation ==================

def evaluate_model(name, y_true, y_pred, class_labels, report_lines):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_labels)))
    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        producers = np.nan_to_num(np.diag(cm) / cm.sum(axis=1))
        consumers = np.nan_to_num(np.diag(cm) / cm.sum(axis=0))

    report_lines.append(f"--- {name} ---")
    report_lines.append("Confusion matrix (rows=true, cols=pred):")
    report_lines.append(str(pd.DataFrame(cm, index=class_labels, columns=class_labels)))
    report_lines.append(f"Overall accuracy: {oa:.4f}")
    report_lines.append(f"Kappa: {kappa:.4f}")
    for i, c in enumerate(class_labels):
        report_lines.append(f"  class {c}: producer's={producers[i]:.3f}  consumer's={consumers[i]:.3f}")
    report_lines.append("")
    print("\n".join(report_lines[-(len(class_labels) + 5):]))
    return {"name": name, "oa": oa, "kappa": kappa, "producers": producers, "consumers": consumers, "cm": cm}


def save_classified_geotiff(class_map, ref_profile, out_path):
    profile = ref_profile.copy()
    profile.update(count=1, dtype="int16", nodata=-1)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(class_map.astype("int16"), 1)


def plot_comparison_3way(arr, map_1d, map_2d, map_unet, class_labels, out_png):
    rgb_idx = [BANDS.index(b) for b in ("B4", "B3", "B2")]
    rgb = np.clip(np.transpose(arr[rgb_idx], (1, 2, 0)) / 0.3, 0, 1)
    cmap = ListedColormap(CLASS_PALETTE[: len(class_labels)])

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 true color")
    for ax, m, title in zip(axes[1:], [map_1d, map_2d, map_unet], ["CNN-1D", "CNN-2D", "U-Net"]):
        ax.imshow(np.ma.masked_less(m, 0), cmap=cmap, vmin=0, vmax=len(class_labels) - 1)
        ax.set_title(title)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved comparison figure -> {out_png}")


# ================== main ==================

def main():
    parser = argparse.ArgumentParser(description="Train U-Net and compare against saved CNN-1D/CNN-2D models")
    parser.add_argument("--project", default=PROJECT_ID)
    parser.add_argument("--patch-size", type=int, default=32, help="U-Net input patch size (must be divisible by 4)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--scale", type=int, default=10)
    parser.add_argument("--skip-full-classification", action="store_true")
    args = parser.parse_args()
    assert args.patch_size % 4 == 0, "--patch-size must be divisible by 4 (two 2x2 poolings)"

    init_ee(args.project)
    global AOI
    AOI = ee.Geometry.Rectangle(AOI_BOUNDS)

    print("Loading training samples from BigQuery...")
    df = fc_to_dataframe(load_samples())
    print(f"Loaded {len(df)} points")

    with open("scaler_1d.json") as f:
        meta = json.load(f)
    scaler = StandardScaler()
    scaler.mean_ = np.array(meta["mean"])
    scaler.scale_ = np.array(meta["scale"])
    scaler.n_features_in_ = len(BANDS)
    class_labels = meta["classes"]
    n_classes = len(class_labels)
    label_to_idx = {int(c): i for i, c in enumerate(class_labels)}
    df["label_enc"] = df["label"].map(label_to_idx)

    train_df_all = df[df["random"] < 0.7].reset_index(drop=True)
    test_df_all = df[df["random"] >= 0.7].reset_index(drop=True)
    print(f"train points: {len(train_df_all)}   test points: {len(test_df_all)}")

    report_lines = []
    tf.random.set_seed(RANDOM_SEED)

    # ---- CNN-1D: reload saved model, evaluate fresh on the tabular test split ----
    cnn1d = models.load_model("cnn1d_s2osm.keras")
    X_test_1d = scaler.transform(test_df_all[BANDS].values)[..., np.newaxis]
    pred_1d = np.argmax(cnn1d.predict(X_test_1d, verbose=0), axis=1)
    result_1d = evaluate_model("CNN-1D", test_df_all["label_enc"].values, pred_1d, class_labels, report_lines)

    # ---- sample big patches once; CNN-2D's 9x9 window is a crop of the same patch ----
    print("\n=== Sampling spatial patches (shared by CNN-2D re-eval and U-Net) ===")
    composite = build_composite()
    patches, valid = sample_patches(df, composite, args.patch_size)
    df_valid = df[valid].reset_index(drop=True)
    patches_valid = patches[valid]
    print(f"Valid patches: {len(df_valid)}/{len(df)} (points near AOI edge are dropped)")

    center = args.patch_size // 2
    r2 = CNN2D_PATCH // 2
    sub9 = patches_valid[:, center - r2:center + r2 + 1, center - r2:center + r2 + 1, :]

    train_mask = df_valid["random"] < 0.7
    test_mask = df_valid["random"] >= 0.7

    # ---- CNN-2D: reload saved model, evaluate on the cropped 9x9 windows ----
    cnn2d = models.load_model("cnn2d_s2osm.keras")
    X_test_2d = scale_patches(sub9[test_mask.values], scaler)
    y_test_2d = df_valid.loc[test_mask, "label_enc"].values
    pred_2d = np.argmax(cnn2d.predict(X_test_2d, verbose=0), axis=1)
    result_2d = evaluate_model("CNN-2D", y_test_2d, pred_2d, class_labels, report_lines)

    # ---- U-Net: train with point supervision on the big patches ----
    print("\n=== Training U-Net ===")
    X_train_u = scale_patches(patches_valid[train_mask.values], scaler)
    X_test_u = scale_patches(patches_valid[test_mask.values], scaler)
    y_train_labels = df_valid.loc[train_mask, "label_enc"].values
    y_test_labels = df_valid.loc[test_mask, "label_enc"].values

    y_train_u, w_train_u = build_point_targets(y_train_labels, args.patch_size, center)

    unet = build_unet(args.patch_size, len(BANDS), n_classes)
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    unet.fit(
        X_train_u, y_train_u, sample_weight=w_train_u, validation_split=0.15,
        epochs=args.epochs, batch_size=args.batch_size, callbacks=[early_stop], verbose=2,
    )
    unet.save("unet_s2osm.keras")

    pred_u_full = unet.predict(X_test_u, verbose=0)
    pred_u = np.argmax(pred_u_full[:, center, center, :], axis=1)
    result_u = evaluate_model("U-Net", y_test_labels, pred_u, class_labels, report_lines)

    # ================== comparison report ==================
    summary = pd.DataFrame([
        {"model": r["name"], "overall_accuracy": r["oa"], "kappa": r["kappa"]}
        for r in (result_1d, result_2d, result_u)
    ])
    report_lines.append("=== Summary (CNN-1D vs CNN-2D vs U-Net) ===")
    report_lines.append(str(summary))
    with open("unet_comparison_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print("\n" + str(summary))
    print("Saved unet_comparison_report.txt, unet_s2osm.keras")

    if args.skip_full_classification:
        return

    # ================== classify the whole AOI ==================
    if os.path.exists("full_composite.tif"):
        print("Reusing existing full_composite.tif")
        with rasterio.open("full_composite.tif") as src:
            arr = src.read().astype(np.float32)
            profile = src.profile
    else:
        print(f"Downloading Sentinel-2 composite -> full_composite.tif")
        geemap.download_ee_image(composite, filename="full_composite.tif", region=AOI, scale=args.scale, dtype="float32")
        with rasterio.open("full_composite.tif") as src:
            arr = src.read().astype(np.float32)
            profile = src.profile

    def load_or_none(path):
        if os.path.exists(path):
            with rasterio.open(path) as src:
                return src.read(1)
        return None

    map_1d = load_or_none("classified_cnn1d.tif")
    map_2d = load_or_none("classified_cnn2d.tif")
    if map_1d is None or map_2d is None:
        print("classified_cnn1d.tif / classified_cnn2d.tif not found; run S2OSM-CNN.py first for those two maps.")
        return

    print("Classifying whole AOI with U-Net...")
    map_unet = classify_unet_full(unet, arr, scaler)
    save_classified_geotiff(map_unet, profile, "classified_unet.tif")

    plot_comparison_3way(arr, map_1d, map_2d, map_unet, class_labels, "classification_comparison_3way.png")


if __name__ == "__main__":
    main()
