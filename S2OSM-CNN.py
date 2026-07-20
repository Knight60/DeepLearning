# -*- coding: utf-8 -*-
"""
S2OSM-CNN.py
Sentinel-2 LULC classification: trains CNN models instead of Random Forest,
using the same training samples (Earth Engine FeatureCollection.runBigQuery
against the same GCP project) as the original RF workflow.

Two CNN architectures are trained and compared, since a single tabular BigQuery
export gives per-pixel band values with no spatial context:
    - CNN-1D : Conv1D over the 10-band/index feature vector of each point
               (uses the BigQuery export directly, no extra EE queries)
    - CNN-2D : Conv2D over a small spatial patch (PATCH_SIZE x PATCH_SIZE) sampled
               from Earth Engine around each training point

Outputs:
    cnn1d_s2osm.keras / cnn2d_s2osm.keras   trained models
    scaler_1d.json                          band mean/std used to standardize inputs
    comparison_report.txt                   OA / kappa / producer's / consumer's accuracy
    classified_cnn1d.tif / classified_cnn2d.tif   whole-AOI classified maps (GeoTIFF)
    classification_comparison.png           true color vs CNN-1D vs CNN-2D map

Usage:
    python S2OSM-CNN.py [--project ee-dancingriver2] [--epochs 40]
                        [--patch-size 9] [--skip-full-classification]
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
# Windows (ERROR_DLL_INIT_FAILED) — see run_log.txt from earlier attempts.
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
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ID = "ee-dancingriver2"
BQ_TABLE = "`ee-dancingriver2.ML.s2_training4classes`"
AOI_BOUNDS = [100.45, 14.25, 100.65, 14.45]
AOI = None  # built in main(), after ee.Initialize() — ee.Geometry needs an initialized client
BANDS = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "EVI", "NDWI", "NDBI"]
S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
CLASS_PALETTE = ["#0000FF", "#FF0000", "#FFFF00", "#00FF00"]  # water / urban / agri / tree, label order 1..4
RANDOM_SEED = 42


# ================== 1. Earth Engine init + BigQuery samples ==================

def init_ee(project_id):
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)


def load_samples():
    return ee.FeatureCollection.runBigQuery(
        query=f"SELECT * FROM {BQ_TABLE}", geometryColumn="geo"
    )


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


# ================== 2. Sentinel-2 composite (shared by patch sampling + full-area classification) ==================

def mask_s2(img):
    scl = img.select("SCL")
    clear = (
        scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    )
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


# ================== 3. Spatial patch sampling for CNN-2D ==================

def sample_patches(df, composite, patch_size, scale=10, batch_size=200):
    radius = patch_size // 2
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
        sampled = patch_image.sampleRegions(
            collection=ee.FeatureCollection(feats), scale=scale, geometries=False
        )
        for feat in sampled.getInfo()["features"]:
            idx = feat["properties"]["idx"]
            try:
                stacked = np.stack(
                    [np.array(feat["properties"][b], dtype=np.float32) for b in BANDS], axis=-1
                )
            except (KeyError, ValueError):
                continue
            if stacked.shape == (patch_size, patch_size, len(BANDS)):
                patches[idx] = stacked
                valid[idx] = True
        print(f"  patches sampled: {min(start + batch_size, n)}/{n}")

    return patches, valid


# ================== 4. Model definitions ==================

def build_1dcnn(n_features, n_classes):
    model = models.Sequential([
        layers.Input(shape=(n_features, 1)),
        layers.Conv1D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv1D(64, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ], name="cnn1d")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_2dcnn(patch_size, n_bands, n_classes):
    model = models.Sequential([
        layers.Input(shape=(patch_size, patch_size, n_bands)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ], name="cnn2d")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ================== 5. Evaluation ==================

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
    return {"name": name, "oa": oa, "kappa": kappa, "producers": producers, "consumers": consumers}


def permutation_importance_1d(model, X_test, y_test, feature_names, n_repeats=5, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    baseline = model.evaluate(X_test, y_test, verbose=0)[1]
    flat = X_test[..., 0]
    importances = {}
    for i, fname in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            perm = flat.copy()
            rng.shuffle(perm[:, i])
            acc = model.evaluate(perm[..., np.newaxis], y_test, verbose=0)[1]
            drops.append(max(baseline - acc, 0.0))
        importances[fname] = float(np.mean(drops))
    total = sum(importances.values()) or 1.0
    return {k: v / total * 100 for k, v in importances.items()}


# ================== 6. Whole-AOI classification ==================

def export_full_composite(composite, out_tif, scale):
    print(f"Downloading Sentinel-2 composite for full-AOI classification -> {out_tif}")
    # ee_export_image issues a single getDownloadURL request (50MB cap); this AOI at
    # 10m/10 bands is ~270MB, so use download_ee_image which tiles automatically.
    geemap.download_ee_image(composite, filename=out_tif, region=AOI, scale=scale, dtype="float32")
    with rasterio.open(out_tif) as src:
        arr = src.read().astype(np.float32)  # (bands, H, W)
        profile = src.profile
    return arr, profile


def classify_1dcnn_full(model, arr, scaler, n_classes):
    n_bands, h, w = arr.shape
    flat = arr.reshape(n_bands, -1).T  # (H*W, bands)
    valid = ~np.any(np.isnan(flat), axis=1)
    out = np.full(flat.shape[0], -1, dtype=np.int32)
    scaled = scaler.transform(flat[valid])[..., np.newaxis]
    preds = np.argmax(model.predict(scaled, batch_size=4096, verbose=0), axis=1)
    out[valid] = preds
    return out.reshape(h, w)


def classify_2dcnn_full(model, arr, scaler, patch_size, batch_rows=32):
    n_bands, h, w = arr.shape
    radius = patch_size // 2
    hwc = np.nan_to_num(np.transpose(arr, (1, 2, 0)))  # (H, W, bands)
    mean = scaler.mean_.reshape(1, 1, -1)
    scale = scaler.scale_.reshape(1, 1, -1)
    scaled = (hwc - mean) / scale
    padded = np.pad(scaled, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")

    # strided view, no copy until a chunk is sliced out below
    windows = np.lib.stride_tricks.sliding_window_view(padded, (patch_size, patch_size, n_bands))
    windows = windows[:, :, 0]  # (H, W, patch, patch, bands)

    out = np.full((h, w), -1, dtype=np.int32)
    for r0 in range(0, h, batch_rows):
        r1 = min(r0 + batch_rows, h)
        batch = windows[r0:r1].reshape(-1, patch_size, patch_size, n_bands)
        preds = np.argmax(model.predict(batch, batch_size=1024, verbose=0), axis=1)
        out[r0:r1, :] = preds.reshape(r1 - r0, w)
        print(f"  CNN-2D classifying rows {r1}/{h}")
    return out


def save_classified_geotiff(class_map, ref_profile, out_path):
    profile = ref_profile.copy()
    profile.update(count=1, dtype="int16", nodata=-1)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(class_map.astype("int16"), 1)


def plot_comparison(arr, map_1d, map_2d, class_labels, out_png):
    rgb_idx = [BANDS.index(b) for b in ("B4", "B3", "B2")]
    rgb = np.clip(np.transpose(arr[rgb_idx], (1, 2, 0)) / 0.3, 0, 1)
    cmap = ListedColormap(CLASS_PALETTE[: len(class_labels)])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 true color")
    axes[1].imshow(np.ma.masked_less(map_1d, 0), cmap=cmap, vmin=0, vmax=len(class_labels) - 1)
    axes[1].set_title("CNN-1D classification")
    axes[2].imshow(np.ma.masked_less(map_2d, 0), cmap=cmap, vmin=0, vmax=len(class_labels) - 1)
    axes[2].set_title("CNN-2D classification")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved comparison figure -> {out_png}")


# ================== main ==================

def main():
    parser = argparse.ArgumentParser(description="Train CNN-1D and CNN-2D LULC classifiers from BigQuery samples")
    parser.add_argument("--project", default=PROJECT_ID, help="GCP/Earth Engine project ID")
    parser.add_argument("--patch-size", type=int, default=9, help="Spatial patch size (odd) for CNN-2D")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--scale", type=int, default=10, help="Pixel scale (m) for full-AOI export")
    parser.add_argument("--skip-full-classification", action="store_true",
                        help="Skip downloading + classifying the whole AOI (train/evaluate only)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Load previously saved cnn1d_s2osm.keras/cnn2d_s2osm.keras + scaler_1d.json "
                             "instead of retraining, and go straight to full-AOI classification")
    args = parser.parse_args()

    init_ee(args.project)
    global AOI
    AOI = ee.Geometry.Rectangle(AOI_BOUNDS)

    if args.skip_training:
        cnn1d = models.load_model("cnn1d_s2osm.keras")
        cnn2d = models.load_model("cnn2d_s2osm.keras")
        with open("scaler_1d.json") as f:
            meta = json.load(f)
        scaler = StandardScaler()
        scaler.mean_ = np.array(meta["mean"])
        scaler.scale_ = np.array(meta["scale"])
        scaler.n_features_in_ = len(BANDS)
        class_labels = meta["classes"]
        n_classes = len(class_labels)
        composite = build_composite()
        print(f"Loaded saved models ({n_classes} classes: {class_labels})")

        if args.skip_full_classification:
            return
        arr, profile = export_full_composite(composite, "full_composite.tif", args.scale)
        print("Classifying whole AOI with CNN-1D...")
        map_1d = classify_1dcnn_full(cnn1d, arr, scaler, n_classes)
        save_classified_geotiff(map_1d, profile, "classified_cnn1d.tif")
        print("Classifying whole AOI with CNN-2D...")
        map_2d = classify_2dcnn_full(cnn2d, arr, scaler, args.patch_size)
        save_classified_geotiff(map_2d, profile, "classified_cnn2d.tif")
        plot_comparison(arr, map_1d, map_2d, class_labels, "classification_comparison.png")
        return

    # ---- 1. load training samples from BigQuery ----
    print("Loading training samples from BigQuery...")
    samples_fc = load_samples()
    df = fc_to_dataframe(samples_fc)
    print(f"Loaded {len(df)} points")

    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])
    class_labels = list(le.classes_)
    n_classes = len(class_labels)

    # ---- 2. train/test split 70/30 (same 'random' column as the RF version) ----
    train_df = df[df["random"] < 0.7].reset_index(drop=True)
    test_df = df[df["random"] >= 0.7].reset_index(drop=True)
    print(f"train points: {len(train_df)}   test points: {len(test_df)}")

    report_lines = []
    tf.random.set_seed(RANDOM_SEED)
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)

    # ================== CNN-1D (tabular bands/indices) ==================
    print("\n=== Training CNN-1D ===")
    scaler = StandardScaler().fit(train_df[BANDS].values)
    X_train_1d = scaler.transform(train_df[BANDS].values)[..., np.newaxis]
    X_test_1d = scaler.transform(test_df[BANDS].values)[..., np.newaxis]
    y_train = train_df["label_enc"].values
    y_test = test_df["label_enc"].values

    cnn1d = build_1dcnn(len(BANDS), n_classes)
    cnn1d.fit(
        X_train_1d, y_train, validation_split=0.15,
        epochs=args.epochs, batch_size=args.batch_size, callbacks=[early_stop], verbose=2,
    )
    cnn1d.save("cnn1d_s2osm.keras")
    with open("scaler_1d.json", "w") as f:
        json.dump({"bands": BANDS, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                   "classes": [str(c) for c in class_labels]}, f, indent=2)

    pred_1d = np.argmax(cnn1d.predict(X_test_1d, verbose=0), axis=1)
    result_1d = evaluate_model("CNN-1D", y_test, pred_1d, class_labels, report_lines)

    importance_1d = permutation_importance_1d(cnn1d, X_test_1d, y_test, BANDS)
    report_lines.append("Permutation importance, CNN-1D (%):")
    for k, v in sorted(importance_1d.items(), key=lambda kv: -kv[1]):
        report_lines.append(f"  {k}: {v:.2f}")
    report_lines.append("")
    print("Permutation importance (%):", {k: round(v, 2) for k, v in importance_1d.items()})

    # ================== CNN-2D (spatial patches) ==================
    print("\n=== Sampling spatial patches for CNN-2D ===")
    composite = build_composite()
    patches, valid = sample_patches(df, composite, args.patch_size)
    df_valid = df[valid].reset_index(drop=True)
    patches_valid = patches[valid]
    print(f"Valid patches: {len(df_valid)}/{len(df)} (points near AOI edge are dropped)")

    train_mask_2d = df_valid["random"] < 0.7
    test_mask_2d = df_valid["random"] >= 0.7

    X_train_2d = scale_patches(patches_valid[train_mask_2d.values], scaler)
    X_test_2d = scale_patches(patches_valid[test_mask_2d.values], scaler)
    y_train_2d = df_valid.loc[train_mask_2d, "label_enc"].values
    y_test_2d = df_valid.loc[test_mask_2d, "label_enc"].values

    print("\n=== Training CNN-2D ===")
    cnn2d = build_2dcnn(args.patch_size, len(BANDS), n_classes)
    cnn2d.fit(
        X_train_2d, y_train_2d, validation_split=0.15,
        epochs=args.epochs, batch_size=32, callbacks=[early_stop], verbose=2,
    )
    cnn2d.save("cnn2d_s2osm.keras")

    pred_2d = np.argmax(cnn2d.predict(X_test_2d, verbose=0), axis=1)
    result_2d = evaluate_model("CNN-2D", y_test_2d, pred_2d, class_labels, report_lines)

    # ================== comparison report ==================
    summary = pd.DataFrame([
        {"model": result_1d["name"], "overall_accuracy": result_1d["oa"], "kappa": result_1d["kappa"]},
        {"model": result_2d["name"], "overall_accuracy": result_2d["oa"], "kappa": result_2d["kappa"]},
    ])
    report_lines.append("=== Summary ===")
    report_lines.append(str(summary))
    with open("comparison_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print("\n" + str(summary))
    print("Saved comparison_report.txt, cnn1d_s2osm.keras, cnn2d_s2osm.keras")

    if args.skip_full_classification:
        return

    # ================== 7. classify the whole AOI with both models ==================
    arr, profile = export_full_composite(composite, "full_composite.tif", args.scale)

    print("Classifying whole AOI with CNN-1D...")
    map_1d = classify_1dcnn_full(cnn1d, arr, scaler, n_classes)
    save_classified_geotiff(map_1d, profile, "classified_cnn1d.tif")

    print("Classifying whole AOI with CNN-2D...")
    map_2d = classify_2dcnn_full(cnn2d, arr, scaler, args.patch_size)
    save_classified_geotiff(map_2d, profile, "classified_cnn2d.tif")

    plot_comparison(arr, map_1d, map_2d, class_labels, "classification_comparison.png")


def scale_patches(patches, scaler):
    mean = scaler.mean_.reshape(1, 1, 1, -1)
    scale = scaler.scale_.reshape(1, 1, 1, -1)
    return (patches - mean) / scale


if __name__ == "__main__":
    main()
