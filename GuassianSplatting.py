#!/usr/bin/env python
"""GuassianSplatting.py — DJI_0348 -> Gaussian Splatting (.ply) -> GuassianSplatting.html

Pipeline (uses the nerfstudio install at D:\\Developing\\NerfStudio):
  1. ns-process-data  : COLMAP camera poses from the DJI_0348 images
  2. ns-train         : splatfacto (3D Gaussian Splatting) training
  3. ns-export        : gaussian-splat -> splat.ply  (copied to GuassianSplatting.ply)
  4. compress         : SuperSplat's splat-transform -> GuassianSplatting.compressed.ply
                        (~10-20x smaller, loads much faster)
  5. build HTML       : the REAL SuperSplat editor app (the same one launched by
                        D:\\Developing\\NerfStudio\\supersplat.bat) is copied into
                        ./supersplat/ next to this script.  GuassianSplatting.html is a
                        tiny redirect page that opens supersplat/index.html?load=...
                        so it auto-loads the compressed splat - full editor, full zoom,
                        nothing embedded/duplicated.  Must be viewed via --serve (the
                        editor is an ES-module app; it needs http://, not file://).

Usage:
  python GuassianSplatting.py                 # full pipeline
  python GuassianSplatting.py --iterations 15000
  python GuassianSplatting.py --html-only     # rebuild HTML from existing .ply
  python GuassianSplatting.py --serve         # open the editor in a browser
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.parse
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "DJI_0348"
WORK_DIR = SCRIPT_DIR / "GaussianSplatting"          # workspace (colmap/outputs/exports)
COLMAP_DIR = WORK_DIR / "colmap"
OUTPUTS_DIR = WORK_DIR / "outputs"
EXPORTS_DIR = WORK_DIR / "exports"
PLY_OUT = SCRIPT_DIR / "GuassianSplatting.ply"
COMPRESSED_OUT = SCRIPT_DIR / "GuassianSplatting.compressed.ply"
HTML_OUT = SCRIPT_DIR / "GuassianSplatting.html"
EDITOR_DIR = SCRIPT_DIR / "supersplat"               # copy of the real SuperSplat editor app

NS_DIR = Path(r"D:\Developing\NerfStudio")
NS_SCRIPTS = NS_DIR / ".venv" / "Scripts"
SETUP_ENV_BAT = NS_DIR / "_setup_env.bat"            # MSVC + CUDA 11.8 env for gsplat JIT
EDITOR_DIST = NS_DIR / "tools" / "supersplat" / "dist"   # built SuperSplat editor (same as supersplat.bat)
# SuperSplat's own converter: .ply -> .compressed.ply
SPLAT_TRANSFORM = (NS_DIR / "tools" / "supersplat" / "node_modules"
                   / "@playcanvas" / "splat-transform" / "bin" / "cli.mjs")

if sys.platform == "win32":  # nerfstudio's rich output needs UTF-8 console
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def log(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def count_images() -> int:
    return sum(1 for p in DATA_DIR.iterdir()
               if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


# --------------------------------------------------------------------------- #
# Run an ns-* tool inside the MSVC/CUDA env (_setup_env.bat), same as the
# train_with_env.bat workflow: needed because gsplat JIT-compiles CUDA kernels.
# --------------------------------------------------------------------------- #
def run_ns(args: list[str]) -> None:
    inner = subprocess.list2cmdline(args)
    print(f"> {inner}", flush=True)
    # cmd.exe quote-mangles nested `call "..." && "..."` strings, so stage the
    # step in a throwaway .bat instead
    bat = WORK_DIR / "_run_step.bat"
    bat.write_text(
        "@echo off\r\n"
        f'call "{SETUP_ENV_BAT}" || exit /b 1\r\n'
        f"{inner}\r\n",
        encoding="ascii",
    )
    rc = subprocess.call(["cmd", "/d", "/c", str(bat)], cwd=str(NS_DIR))
    if rc != 0:
        sys.exit(f"[ERROR] command failed (exit {rc}): {inner}")


# --------------------------------------------------------------------------- #
# 1) COLMAP via ns-process-data
# --------------------------------------------------------------------------- #
def step_colmap(matching: str, force: bool) -> None:
    transforms = COLMAP_DIR / "transforms.json"
    if transforms.exists() and not force:
        log(f"COLMAP already done ({transforms}) - skipping (use --force to redo)")
        return
    if not DATA_DIR.is_dir():
        sys.exit(f"[ERROR] image folder not found: {DATA_DIR}")
    n = count_images()
    log(f"COLMAP: ns-process-data on {n} images ({matching} matching)")
    if force and COLMAP_DIR.exists():
        shutil.rmtree(COLMAP_DIR)
    run_ns([
        str(NS_SCRIPTS / "ns-process-data.exe"), "images",
        "--data", str(DATA_DIR),
        "--output-dir", str(COLMAP_DIR),
        "--matching-method", matching,
    ])


# --------------------------------------------------------------------------- #
# 2) Train splatfacto
# --------------------------------------------------------------------------- #
def step_train(iterations: int, downscale: int) -> None:
    log(f"Training splatfacto: {iterations} iterations, images /{downscale} "
        f"({3840 // downscale}x{2160 // downscale})")
    run_ns([
        str(NS_SCRIPTS / "ns-train.exe"), "splatfacto",
        "--data", str(COLMAP_DIR),
        "--output-dir", str(OUTPUTS_DIR),
        "--max-num-iterations", str(iterations),
        "--viewer.quit-on-train-completion", "True",
        "nerfstudio-data",
        "--downscale-factor", str(downscale),
    ])


def latest_config() -> Path:
    configs = sorted(OUTPUTS_DIR.rglob("config.yml"), key=lambda p: p.stat().st_mtime)
    if not configs:
        sys.exit(f"[ERROR] no training run (config.yml) found under {OUTPUTS_DIR}")
    return configs[-1]


# --------------------------------------------------------------------------- #
# 3) Export gaussian splat .ply
# --------------------------------------------------------------------------- #
def step_export() -> None:
    config = latest_config()
    log(f"Exporting gaussian splat from {config.parent.name}")
    run_ns([
        str(NS_SCRIPTS / "ns-export.exe"), "gaussian-splat",
        "--load-config", str(config),
        "--output-dir", str(EXPORTS_DIR),
    ])
    ply = EXPORTS_DIR / "splat.ply"
    if not ply.exists():
        plys = sorted(EXPORTS_DIR.glob("*.ply"), key=lambda p: p.stat().st_mtime)
        if not plys:
            sys.exit(f"[ERROR] no .ply produced in {EXPORTS_DIR}")
        ply = plys[-1]
    shutil.copy2(ply, PLY_OUT)
    log(f"PLY ready: {PLY_OUT} ({PLY_OUT.stat().st_size / 1024**2:.1f} MB)")


# --------------------------------------------------------------------------- #
# 4) Build GuassianSplatting.html: a redirect into the REAL SuperSplat editor
#    (copied from D:\Developing\NerfStudio\tools\supersplat\dist, the same app
#    supersplat.bat serves), auto-loading the splat via its `?load=` param.
#    Full editor UI, full zoom range, full edit/crop/export - not a re-implementation.
# --------------------------------------------------------------------------- #
REDIRECT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DJI_0348 — Gaussian Splatting</title>
<meta http-equiv="refresh" content="0; url=__EDITOR_URL__">
<style>
  html, body { margin: 0; height: 100%; display: flex; align-items: center; justify-content: center;
               background: #0d1117; font-family: 'Segoe UI', system-ui, sans-serif; color: #e6edf3; }
  a { color: #58a6ff; }
</style>
</head>
<body>
<script>location.replace('__EDITOR_URL__');</script>
<p>Opening the SuperSplat editor with __META__ …<br>
If nothing happens, <a href="__EDITOR_URL__">click here</a>.</p>
</body>
</html>
"""


def step_html() -> None:
    if not PLY_OUT.exists():
        sys.exit(f"[ERROR] {PLY_OUT} not found - run the export step first")

    # --- compress the raw .ply with SuperSplat's splat-transform (10-20x smaller,
    #     loads much faster in the editor); falls back to the raw .ply if unavailable ---
    splat_file = PLY_OUT
    node = shutil.which("node")
    if node and SPLAT_TRANSFORM.exists():
        if (not COMPRESSED_OUT.exists()
                or COMPRESSED_OUT.stat().st_mtime < PLY_OUT.stat().st_mtime):
            log("Compressing PLY for the editor (splat-transform)")
            rc = subprocess.call([node, str(SPLAT_TRANSFORM), "-w",
                                  str(PLY_OUT), str(COMPRESSED_OUT)])
            if rc != 0:
                print(f"[WARN] splat-transform failed (exit {rc}) - "
                      f"the editor will load the raw .ply instead", flush=True)
        if (COMPRESSED_OUT.exists()
                and COMPRESSED_OUT.stat().st_mtime >= PLY_OUT.stat().st_mtime):
            splat_file = COMPRESSED_OUT
    else:
        print("[WARN] node / splat-transform not found - "
              "the editor will load the raw .ply", flush=True)

    # --- copy the real SuperSplat editor app in (only when missing/stale) ---
    if not EDITOR_DIST.is_dir():
        sys.exit(f"[ERROR] SuperSplat editor build not found: {EDITOR_DIST}")
    if (not EDITOR_DIR.is_dir()
            or EDITOR_DIST.stat().st_mtime > EDITOR_DIR.stat().st_mtime):
        log(f"Copying SuperSplat editor app -> {EDITOR_DIR}")
        if EDITOR_DIR.is_dir():
            shutil.rmtree(EDITOR_DIR)
        shutil.copytree(EDITOR_DIST, EDITOR_DIR)

    # --- redirect page: opens the editor pointed at the splat (absolute path from
    #     the server root, so it works regardless of query-string encoding depth) ---
    log("Building GuassianSplatting.html (redirect into the SuperSplat editor)")
    load_url = "/" + splat_file.name
    query = urllib.parse.urlencode({"load": load_url, "filename": splat_file.name})
    editor_url = f"/{EDITOR_DIR.name}/index.html?{query}"
    meta = f"{count_images()} DJI images · {splat_file.name} ({splat_file.stat().st_size / 1024**2:.0f} MB)"

    html = (REDIRECT_TEMPLATE
            .replace("__EDITOR_URL__", editor_url)
            .replace("__META__", meta))
    HTML_OUT.write_text(html, encoding="utf-8")
    log(f"HTML ready: {HTML_OUT} - view with: python GuassianSplatting.py --serve")


def serve(port: int) -> None:
    """Serve the folder over HTTP and open the viewer (fetch() cannot read file://)."""
    import http.server
    import socketserver
    import webbrowser
    from functools import partial

    if not HTML_OUT.exists():
        sys.exit(f"[ERROR] {HTML_OUT} not found - run the pipeline first")
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(SCRIPT_DIR))
    socketserver.TCPServer.allow_reuse_address = True
    httpd = None
    for p in range(port, port + 10):
        try:
            httpd = socketserver.TCPServer(("127.0.0.1", p), handler)
            break
        except OSError:
            continue
    if httpd is None:
        sys.exit(f"[ERROR] no free port in {port}-{port + 9}")
    url = f"http://127.0.0.1:{p}/{HTML_OUT.name}"
    log(f"Serving {SCRIPT_DIR} at {url}  (Ctrl+C to stop)")
    webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="DJI_0348 -> Gaussian Splatting PLY + HTML viewer")
    ap.add_argument("--iterations", type=int, default=30000, help="splatfacto iterations")
    ap.add_argument("--downscale", type=int, default=2, choices=[1, 2, 4, 8],
                    help="training image downscale (2 => 1920x1080, fits 12GB VRAM)")
    ap.add_argument("--matching-method", default="exhaustive",
                    choices=["exhaustive", "sequential", "vocab_tree"])
    ap.add_argument("--force", action="store_true", help="redo COLMAP even if it exists")
    ap.add_argument("--skip-colmap", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--html-only", action="store_true",
                    help="only rebuild the HTML from the existing GuassianSplatting.ply")
    ap.add_argument("--serve", action="store_true",
                    help="serve the folder over HTTP and open the viewer in a browser")
    ap.add_argument("--port", type=int, default=8090, help="port for --serve")
    args = ap.parse_args()

    if args.serve and args.html_only is False and HTML_OUT.exists():
        serve(args.port)         # view-only: don't re-run the pipeline
        return

    WORK_DIR.mkdir(exist_ok=True)

    if not args.html_only:
        if not args.skip_colmap:
            step_colmap(args.matching_method, args.force)
        if not args.skip_train:
            step_train(args.iterations, args.downscale)
        if not args.skip_export:
            step_export()
    step_html()

    if args.serve:
        serve(args.port)


if __name__ == "__main__":
    main()
