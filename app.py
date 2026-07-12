from pathlib import Path
from datetime import datetime
import shutil
import zipfile
import cv2
import gradio as gr
import pandas as pd
import numpy as np
from ultralytics import YOLO
import threading
import time
import traceback
import os
import torch
import sqlite3
import hashlib
import secrets
import uuid
import json
import html
import base64
import re
from urllib.parse import quote

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
DATASETS_DIR = ROOT / "datasets"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
BRAND_LOGOS_DIR = ROOT / "assets" / "brand_logos"

CURRENT_TRAINING = {"state": None}
CURRENT_ANALYSIS = {"state": None}

BLUE = "#46619c"
BLUE_DARK = "#344a78"
GREEN = "#3f8b5b"
RED = "#a9172b"
TEXT = "#1f2937"
MUTED = "#6b7280"
BORDER = "#e5e7eb"
BG = "#f8fafc"

for folder in [DATASETS_DIR, MODELS_DIR, RESULTS_DIR, BRAND_LOGOS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def best_device():
    if torch.cuda.is_available(): return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"

DEVICE = best_device()


def time_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_time(seconds):
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


def format_elapsed(seconds):
    seconds = int(seconds)
    minutes = seconds // 60
    sec = seconds % 60
    return f"{minutes} min {sec:02d} s"


def safe_round(x, n=2):
    try:
        if pd.isna(x):
            return 0
        return round(float(x), n)
    except Exception:
        return x


def find_data_yaml(folder):
    candidates = list(folder.rglob("data.yaml")) + list(folder.rglob("data.yml"))
    return candidates[0] if candidates else None


def count_dataset_images(data_yaml):
    root = data_yaml.parent
    counts = {"train": 0, "valid": 0, "test": 0}
    splits = {
        "train": ["train/images", "images/train", "train"],
        "valid": ["valid/images", "val/images", "images/valid", "images/val", "valid", "val"],
        "test": ["test/images", "images/test", "test"],
    }
    for split_name, possible_dirs in splits.items():
        for d in possible_dirs:
            p = root / d
            if p.exists():
                counts[split_name] = sum(
                    len(list(p.rglob(ext)))
                    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]
                )
                break
    return counts


def alert_html(message, tone="error"):
    color = RED if tone == "error" else GREEN if tone == "success" else BLUE
    bg = "#fff5f5" if tone == "error" else "#f0fdf4" if tone == "success" else "#f8fafc"
    return f"""
    <div class="alert" style="border-color:{color}22;background:{bg};color:{color};">
        {message}
    </div>
    """


def progress_html(state):
    percent = max(0, min(100, float(state.get("percent", 0))))
    label = state.get("label", "En attente")
    elapsed = format_elapsed(time.time() - state["start_time"]) if state.get("start_time") else "0 min 00 s"

    color = BLUE
    if state.get("done"):
        color = GREEN
    if state.get("stopped"):
        color = RED
    if state.get("error"):
        color = "#ef4444"

    error_block = ""
    if state.get("error"):
        error_block = f"<div class='progress-error'>Erreur : {state.get('error')}</div>"

    return f"""
    <div class="progress-card">
        <div class="progress-top">
            <strong>{label}</strong>
            <span>{percent:.1f}% — {elapsed}</span>
        </div>
        <div class="progress-track">
            <div class="progress-fill" style="width:{percent:.1f}%; background:{color};"></div>
        </div>
        {error_block}
    </div>
    """


def train_worker(zip_file, epochs, image_size, state):
    try:
        state["start_time"] = time.time()
        state["label"] = "Décompression du ZIP..."
        state["percent"] = 0

        run = "train_" + time_id()
        dataset_path = DATASETS_DIR / run
        dataset_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(dataset_path)

        data_yaml = find_data_yaml(dataset_path)
        if data_yaml is None:
            raise RuntimeError("Aucun fichier data.yaml trouvé. Exporte le dataset depuis Roboflow au format YOLOv8.")

        counts = count_dataset_images(data_yaml)
        state["label"] = (
            f"Dataset chargé : Train : {counts['train']} images · "
            f"Validation : {counts['valid']} images · Test : {counts['test']} images"
        )
        state["percent"] = 2

        total_epochs = int(epochs)
        batch_state = {"current": 0, "total": 1}
        model = YOLO("yolov8n.pt")

        def on_train_start(trainer):
            try:
                batch_state["total"] = max(1, len(trainer.train_loader))
            except Exception:
                batch_state["total"] = 1
            state["label"] = "Entraînement démarré"
            state["percent"] = 3

        def on_train_epoch_start(trainer):
            batch_state["current"] = 0

        def on_train_batch_end(trainer):
            batch_state["current"] += 1
            epoch_index = int(getattr(trainer, "epoch", 0))
            current_batch = batch_state["current"]
            total_batches = max(1, batch_state["total"])
            progress_value = (epoch_index + current_batch / total_batches) / max(1, total_epochs)
            state["percent"] = progress_value * 100
            if state.get("stop_requested"):
                state["stopped"] = True
                state["label"] = f"Arrêt demandé — sauvegarde du meilleur modèle disponible (epoch {epoch_index + 1}/{total_epochs})"
                trainer.stop = True
            else:
                state["label"] = f"Epoch {epoch_index + 1}/{total_epochs} — batch {current_batch}/{total_batches}"

        model.add_callback("on_train_start", on_train_start)
        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        model.add_callback("on_train_batch_end", on_train_batch_end)

        train_kwargs = dict(
            data=str(data_yaml),
            epochs=total_epochs,
            imgsz=int(image_size),
            project=str(RESULTS_DIR / "training"),
            name=run,
            exist_ok=True,
            device=DEVICE,
            workers=min(8, max(1, (os.cpu_count() or 2) - 1)),
            amp=(DEVICE != "cpu"),
            batch=-1 if DEVICE == 0 else 8,
            patience=8,
            cache=True,
            plots=False,
            verbose=False,
        )
        train_results = model.train(**train_kwargs)
        metrics = getattr(train_results, "results_dict", None) or getattr(train_results, "box", None) or {}
        metrics_dict = metrics if isinstance(metrics, dict) else {}
        map50 = metrics_dict.get("metrics/mAP50(B)", metrics_dict.get("metrics/mAP50", None))
        map5095 = metrics_dict.get("metrics/mAP50-95(B)", metrics_dict.get("metrics/mAP50-95", None))
        precision = metrics_dict.get("metrics/precision(B)", metrics_dict.get("metrics/precision", None))
        def pct_value(v):
            try:
                return f"{float(v) * 100:.1f} %"
            except Exception:
                return "Non disponible"
        def raw_pct(v):
            try:
                return round(float(v) * 100, 1)
            except Exception:
                return 0.0
        m1, m2, m3 = raw_pct(map50), raw_pct(map5095), raw_pct(precision)
        state["metrics_html"] = f"""
        <div class='metric-card animated-metrics'>
            <div class='metric-title'>Précision du modèle après entraînement</div>
            <div class='metric-grid'>
                <div class='metric-box metric-box-clean'><span>mAP50</span><strong class='count-up' data-target='{m1}'>0.0 %</strong></div>
                <div class='metric-box metric-box-clean'><span>mAP50-95</span><strong class='count-up' data-target='{m2}'>0.0 %</strong></div>
                <div class='metric-box metric-box-clean'><span>Précision</span><strong class='count-up' data-target='{m3}'>0.0 %</strong></div>
            </div>
            <div class='metric-note'>Ces indicateurs évaluent le modèle sur le jeu de validation. Ils sont différents de la confiance des détections dans le tableau d’analyse.</div>
        </div>
        """

        best = RESULTS_DIR / "training" / run / "weights" / "best.pt"
        if not best.exists():
            best = RESULTS_DIR / "training" / run / "weights" / "last.pt"
        if not best.exists():
            raise RuntimeError("L'entraînement n'a pas produit de modèle best.pt ou last.pt.")

        saved_model = MODELS_DIR / f"best_{run}.pt"
        shutil.copy(best, saved_model)

        if state.get("stopped"):
            state["label"] = f"Entraînement arrêté — modèle sauvegardé dans models/{saved_model.name}"
        else:
            state["percent"] = 100
            state["label"] = f"Entraînement terminé — modèle sauvegardé dans models/{saved_model.name}"

        state["model_path"] = str(saved_model)
        state["done"] = True

    except Exception as e:
        state["error"] = str(e)
        state["done"] = True
        state["model_path"] = None
        print(traceback.format_exc())


def train_model(zip_file, epochs, image_size):
    if zip_file is None:
        yield alert_html("Ajoute un ZIP Roboflow/YOLOv8."), None, "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
        return

    state = {
        "start_time": time.time(),
        "label": "Préparation de l'entraînement...",
        "percent": 0,
        "done": False,
        "error": None,
        "model_path": None,
        "stop_requested": False,
        "stopped": False,
    }
    CURRENT_TRAINING["state"] = state

    thread = threading.Thread(target=train_worker, args=(zip_file, epochs, image_size, state), daemon=True)
    thread.start()

    while not state["done"]:
        yield progress_html(state), None, state.get("metrics_html", ""), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        time.sleep(1)

    yield progress_html(state), state.get("model_path"), state.get("metrics_html", ""), gr.update(visible=bool(state.get("model_path"))), gr.update(visible=False), gr.update(visible=False)

def stop_training():
    state = CURRENT_TRAINING.get("state")
    if state is None or state.get("done"):
        return alert_html("Aucun entraînement en cours.", "info")
    state["stop_requested"] = True
    state["stopped"] = True
    state["label"] = "Arrêt demandé — l'entraînement va s'arrêter au prochain batch."
    return progress_html(state)


def make_sequences(df, fps, max_gap_seconds=1.0):
    if df.empty:
        return pd.DataFrame(columns=["logo", "debut_sec", "fin_sec", "duree_sec", "debut_timecode", "fin_timecode"])
    rows = []
    max_gap_frames = int(max_gap_seconds * fps)
    for logo, group in df.groupby("logo"):
        frames = sorted(group["frame"].unique())
        if not frames:
            continue
        start = previous = frames[0]
        for current in frames[1:]:
            if current - previous > max_gap_frames:
                rows.append({
                    "logo": logo,
                    "debut_sec": start / fps,
                    "fin_sec": previous / fps,
                    "duree_sec": max((previous - start + 1) / fps, 1 / fps),
                    "debut_timecode": format_time(start / fps),
                    "fin_timecode": format_time(previous / fps),
                })
                start = current
            previous = current
        rows.append({
            "logo": logo,
            "debut_sec": start / fps,
            "fin_sec": previous / fps,
            "duree_sec": max((previous - start + 1) / fps, 1 / fps),
            "debut_timecode": format_time(start / fps),
            "fin_timecode": format_time(previous / fps),
        })
    return pd.DataFrame(rows)


def draw_custom_detections(frame, result, names, selected_brands=None):
    annotated = frame.copy()
    for box in result.boxes or []:
        class_id = int(box.cls[0])
        logo = names.get(class_id, str(class_id))
        if selected_brands and logo not in selected_brands:
            continue
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [int(float(v)) for v in box.xyxy[0]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (70, 97, 156), 2)
        label = f"{logo} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        y_text = max(y1 - 6, th + 6)
        cv2.rectangle(annotated, (x1, y_text - th - 5), (x1 + tw + 8, y_text + baseline), (70, 97, 156), -1)
        cv2.putText(annotated, label, (x1 + 4, y_text - 3), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return annotated


def build_stats(df, fps, duration, width, height, sequences):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    frame_area = max(1, width * height)
    df = df.copy()
    df["surface_pct"] = df["surface_bbox"] / frame_area * 100
    df["centre_x"] = (df["x1"] + df["x2"]) / 2
    df["centre_y"] = (df["y1"] + df["y2"]) / 2
    df["central"] = (
        (df["centre_x"] >= width / 3) &
        (df["centre_x"] <= 2 * width / 3) &
        (df["centre_y"] >= height / 3) &
        (df["centre_y"] <= 2 * height / 3)
    )

    per_frame_logo = df.groupby(["logo", "frame"], as_index=False).agg(
        nb_logos_frame=("logo", "count"),
        surface_pct_frame=("surface_pct", "sum"),
        central_frame=("central", "max"),
        seconde=("seconde", "min"),
    )

    base = df.groupby("logo").agg(
        detections_techniques=("logo", "count"),
        frames_avec_logo=("frame", "nunique"),
        confiance_moyenne=("confiance", "mean"),
        premiere_apparition_sec=("seconde", "min"),
        derniere_apparition_sec=("seconde", "max"),
    ).reset_index()

    frame_stats = per_frame_logo.groupby("logo").agg(
        logos_simultanes_moy=("nb_logos_frame", "mean"),
        logos_simultanes_max=("nb_logos_frame", "max"),
        occupation_ecran_moy_pct=("surface_pct_frame", "mean"),
        occupation_ecran_max_pct=("surface_pct_frame", "max"),
        centralite_pct=("central_frame", "mean"),
    ).reset_index()
    frame_stats["centralite_pct"] = frame_stats["centralite_pct"] * 100

    base = base.merge(frame_stats, on="logo", how="left")
    base["temps_visible_secondes"] = base["frames_avec_logo"] * (1 / fps)
    base["pourcentage_video"] = base["temps_visible_secondes"] / duration * 100 if duration else 0
    base["premiere_apparition"] = base["premiere_apparition_sec"].apply(format_time)
    base["derniere_apparition"] = base["derniere_apparition_sec"].apply(format_time)

    total_surface = df["surface_pct"].sum()
    sov = df.groupby("logo")["surface_pct"].sum().reset_index(name="surface_totale_pct_points")
    sov["part_de_voix_visuelle_pct"] = (sov["surface_totale_pct_points"] / total_surface * 100) if total_surface else 0
    base = base.merge(sov[["logo", "part_de_voix_visuelle_pct"]], on="logo", how="left")

    if sequences is not None and not sequences.empty:
        seq_stats = sequences.groupby("logo").agg(
            apparitions_distinctes=("logo", "count"),
            duree_moyenne_apparition_sec=("duree_sec", "mean"),
            duree_max_apparition_sec=("duree_sec", "max"),
        ).reset_index()
        base = base.merge(seq_stats, on="logo", how="left")
    else:
        base["apparitions_distinctes"] = 0
        base["duree_moyenne_apparition_sec"] = 0
        base["duree_max_apparition_sec"] = 0

    base = base.fillna(0).sort_values("temps_visible_secondes", ascending=False)

    commercial = base[[
        "logo",
        "temps_visible_secondes",
        "pourcentage_video",
        "apparitions_distinctes",
        "duree_moyenne_apparition_sec",
        "duree_max_apparition_sec",
        "logos_simultanes_moy",
        "logos_simultanes_max",
        "occupation_ecran_moy_pct",
        "occupation_ecran_max_pct",
        "centralite_pct",
        "part_de_voix_visuelle_pct",
        "confiance_moyenne",
    ]].copy()

    commercial = commercial.rename(columns={
        "logo": "Logo",
        "temps_visible_secondes": "Temps visible (s)",
        "pourcentage_video": "% vidéo",
        "apparitions_distinctes": "Nb de séquences",
        "duree_moyenne_apparition_sec": "Durée moy. séquence (s)",
        "duree_max_apparition_sec": "Durée max. séquence (s)",
        "logos_simultanes_moy": "Logos simultanés moy.",
        "logos_simultanes_max": "Logos simultanés max",
        "occupation_ecran_moy_pct": "Occupation moy. (%)",
        "occupation_ecran_max_pct": "Occupation max (%)",
        "centralite_pct": "Centralité (%)",
        "part_de_voix_visuelle_pct": "Part de voix (%)",
        "confiance_moyenne": "Netteté moy.",
    })

    for col in commercial.columns:
        if col != "Logo":
            commercial[col] = commercial[col].apply(lambda x: safe_round(x, 2))

    return base, commercial, per_frame_logo




def brand_slug(name):
    value = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return value or "marque"


def brand_logo_path(name):
    slug = brand_slug(name)
    for extension in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = BRAND_LOGOS_DIR / f"{slug}{extension}"
        if candidate.exists():
            return candidate
    try:
        with db_connection() as conn:
            row = conn.execute(
                "SELECT stored_filename FROM brand_logos WHERE LOWER(brand_name) = LOWER(?) ORDER BY created_at DESC LIMIT 1",
                (str(name).strip(),),
            ).fetchone()
        if row:
            candidate = BRAND_LOGOS_DIR / Path(row["stored_filename"]).name
            if candidate.exists():
                return candidate
    except Exception:
        pass
    return None


def brand_logo_data_uri(name):
    path = brand_logo_path(name)
    if path is None:
        return None
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    if path.suffix.lower() == ".webp":
        mime = "image/webp"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def brand_label_html(name, image_only=False):
    clean = html.escape(str(name))
    uri = brand_logo_data_uri(name)
    if not uri:
        return clean
    name_part = "" if image_only else f"<span>{clean}</span>"
    return (
        "<span class='brand-label'>"
        f"<img class='brand-logo' src='{uri}' alt='{clean}'>"
        f"{name_part}</span>"
    )


def load_model_brand_choices(model_path):
    if not model_path:
        return gr.update(choices=[], value=[]), alert_html("Validez d’abord le modèle.", "info")
    try:
        model = YOLO(model_path)
        names = model.names or {}
        brands = [str(names[key]) for key in sorted(names)] if isinstance(names, dict) else [str(x) for x in names]
        return (
            gr.update(choices=brands, value=brands),
            alert_html(f"{len(brands)} marque(s) disponible(s). Toutes sont sélectionnées par défaut.", "success"),
        )
    except Exception as error:
        return gr.update(choices=[], value=[]), alert_html(f"Impossible de lire les classes du modèle : {html.escape(str(error))}", "error")

def make_table_html(commercial):
    if commercial is None or commercial.empty:
        return "<div class='table-empty'>Aucun résultat à afficher pour le moment.</div>"

    df = commercial.copy()
    if "Logo" in df.columns:
        df["Logo"] = df["Logo"].apply(brand_label_html)

    visibility_cols = [c for c in [
        "Logo", "Temps visible (s)", "% vidéo",
        "Nb de séquences", "Durée moy. séquence (s)", "Durée max. séquence (s)"
    ] if c in df.columns]

    quality_cols = [c for c in [
        "Logo", "Occupation moy. (%)", "Occupation max (%)",
        "Centralité (%)", "Part de voix (%)", "Netteté moy."
    ] if c in df.columns]

    # Petits pictos SVG sobres, non déformés, dans le bleu du site.
    icons = {
        "Logo": "tag",
        "Temps visible (s)": "clock",
        "% vidéo": "film",
        "Nb de séquences": "repeat",
        "Durée moy. séquence (s)": "wave",
        "Durée max. séquence (s)": "pin",
        "Occupation moy. (%)": "square",
        "Occupation max (%)": "grid",
        "Centralité (%)": "target",
        "Part de voix (%)": "bars",
        "Netteté moy.": "diamond",
    }

    svg = {
        "tag": "<svg viewBox='0 0 24 24'><path d='M4 5h8l8 8-7 7-8-8V5z'/><circle cx='9' cy='9' r='1.5'/></svg>",
        "clock": "<svg viewBox='0 0 24 24'><circle cx='12' cy='12' r='8'/><path d='M12 7v5l3 2'/></svg>",
        "film": "<svg viewBox='0 0 24 24'><rect x='5' y='4' width='14' height='16' rx='2'/><path d='M8 4v16M16 4v16M5 9h14M5 15h14'/></svg>",
        "repeat": "<svg viewBox='0 0 24 24'><path d='M7 7h9l2 2M17 17H8l-2-2M18 9V5M6 15v4'/></svg>",
        "wave": "<svg viewBox='0 0 24 24'><path d='M4 13c2.5-5 5.5 5 8 0s5.5 5 8 0'/></svg>",
        "pin": "<svg viewBox='0 0 24 24'><path d='M12 21s6-5.4 6-11a6 6 0 0 0-12 0c0 5.6 6 11 6 11z'/><circle cx='12' cy='10' r='2'/></svg>",
        "square": "<svg viewBox='0 0 24 24'><rect x='6' y='6' width='12' height='12' rx='2'/></svg>",
        "grid": "<svg viewBox='0 0 24 24'><path d='M5 5h6v6H5zM13 5h6v6h-6zM5 13h6v6H5zM13 13h6v6h-6z'/></svg>",
        "target": "<svg viewBox='0 0 24 24'><circle cx='12' cy='12' r='8'/><circle cx='12' cy='12' r='3'/><path d='M12 2v4M12 18v4M2 12h4M18 12h4'/></svg>",
        "bars": "<svg viewBox='0 0 24 24'><path d='M6 19V10M12 19V5M18 19v-7'/></svg>",
        "diamond": "<svg viewBox='0 0 24 24'><path d='M12 3l8 9-8 9-8-9 8-9z'/></svg>",
    }

    def label(col):
        return f"<span class='mini-icon'>{svg[icons[col]]}</span><span>{col}</span>"

    def render_table(cols):
        small = df[cols].copy()
        html = small.to_html(index=False, escape=False, classes="results-table")
        for c in cols:
            html = html.replace(f"<th>{c}</th>", f"<th>{label(c)}</th>")
        return html

    return f"""
    <div class='split-results'>
      <div class='mini-table-card'>
        <div class='mini-table-title'><span class='title-dot'></span>Visibilité temporelle</div>
        <div class='table-scroll'>{render_table(visibility_cols)}</div>
      </div>
      <div class='mini-table-card'>
        <div class='mini-table-title'><span class='title-dot'></span>Qualité d’exposition</div>
        <div class='table-scroll'>{render_table(quality_cols)}</div>
      </div>
    </div>
    <div class='table-note'><span class='mini-icon note-icon'>{svg["diamond"]}</span>Netteté moy. = confiance moyenne des détections YOLO ; ce n’est pas la précision globale du modèle.</div>
    """

def make_kpis_html(commercial):
    if commercial is None or commercial.empty:
        return ""
    top = commercial.iloc[0]
    best_sov = commercial.sort_values("Part de voix (%)", ascending=False).iloc[0]
    return f"""
    <div class="kpi-grid">
        <div class="kpi-card"><div class="kpi-label">Logo le plus visible</div><div class="kpi-value">{brand_label_html(top['Logo'])}</div></div>
        <div class="kpi-card"><div class="kpi-label">Temps d'exposition</div><div class="kpi-value">{top['Temps visible (s)']} s</div></div>
        <div class="kpi-card"><div class="kpi-label">Occupation max</div><div class="kpi-value">{top['Occupation max (%)']} %</div></div>
        <div class="kpi-card"><div class="kpi-label">Meilleure part de voix</div><div class="kpi-value">{brand_label_html(best_sov['Logo'])} · {best_sov['Part de voix (%)']} %</div></div>
    </div>
    """



BRAND_COLOR_PALETTE = [
    "#6F9CBE",  # bleu doux
    "#5DBCAC",  # bleu ciel
    "#63D549",  # vert sauge
    "#C7CB36",  # vert tendre
    "#BE7BC3",  # lavande
    "#8706A4",  # violet pastel
    "#EB885D",  # saumon doux
    "#E2BD73",  # beige doré
    "#CD7D85",  # turquoise grisé
    "#5B79B9",  # gris bleuté
    "#636363",  # taupe rosé
    "#003A8A",  # bleu-gris
]


def brand_style_from_commercial(commercial):
    """
    Classement décroissant selon le temps visible et attribution
    d'une couleur distincte et stable à chaque marque.
    """
    if commercial is None or commercial.empty:
        return [], {}

    ranked = (
        commercial.sort_values("Temps visible (s)", ascending=False)["Logo"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    colors = {
        logo: BRAND_COLOR_PALETTE[index % len(BRAND_COLOR_PALETTE)]
        for index, logo in enumerate(ranked)
    }
    return ranked, colors


def hex_to_bgr(hex_color):
    value = hex_color.lstrip("#")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return (b, g, r)


def draw_detection_records(frame, records, color_map):
    annotated = frame.copy()

    for record in records:
        logo = str(record["logo"])
        conf = float(record["confiance"])
        x1 = int(float(record["x1"]))
        y1 = int(float(record["y1"]))
        x2 = int(float(record["x2"]))
        y2 = int(float(record["y2"]))

        color = hex_to_bgr(color_map.get(logo, "#2563eb"))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{logo} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.72
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        y_text = max(y1 - 6, th + 6)

        cv2.rectangle(
            annotated,
            (x1, y_text - th - 5),
            (x1 + tw + 9, y_text + baseline),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 4, y_text - 3),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return annotated


def rebuild_annotated_video(
    source_video,
    detections_df,
    output_video,
    fps,
    width,
    height,
    color_map,
):
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        return

    temporary_video = Path(output_video).with_name("video_annotee_coloree_temp.mp4")
    writer = cv2.VideoWriter(
        str(temporary_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    records_by_frame = {}
    if detections_df is not None and not detections_df.empty:
        for frame_number, group in detections_df.groupby("frame"):
            records_by_frame[int(frame_number)] = group.to_dict("records")

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        annotated = draw_detection_records(
            frame,
            records_by_frame.get(frame_index, []),
            color_map,
        )
        writer.write(annotated)
        frame_index += 1

    cap.release()
    writer.release()

    output_path = Path(output_video)
    if temporary_video.exists() and temporary_video.stat().st_size > 0:
        temporary_video.replace(output_path)



def figure_base(figsize=(7, 4)):
    fig, ax = plt.subplots(figsize=figsize, dpi=170)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#e5e7eb")
    ax.spines["bottom"].set_color("#e5e7eb")
    ax.tick_params(colors="#475569", labelsize=8)
    ax.grid(axis="x", color="#eef2f7", linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def save_ranking_chart(commercial, out_dir, color_map=None):
    path = out_dir / "graph_classement_temps.png"
    if commercial.empty:
        return None

    ranked_order, default_colors = brand_style_from_commercial(commercial)
    colors = color_map or default_colors

    # Ascendant pour barh : la plus grande barre apparaît visuellement en haut.
    data = (
        commercial[commercial["Logo"].astype(str).isin(ranked_order[:10])]
        .copy()
        .sort_values("Temps visible (s)", ascending=True)
    )

    fig, ax = figure_base((8.8, 4.6))
    ax.barh(
        data["Logo"],
        data["Temps visible (s)"],
        color=[colors.get(str(logo), "#46619c") for logo in data["Logo"]],
        height=0.38,
    )
    ax.set_xlabel("Temps visible (s)", fontsize=9, color="#64748b")
    ax.set_ylabel("")
    ax.set_title("", pad=0)

    xmax = max(float(data["Temps visible (s)"].max()) * 1.15, 1)
    ax.set_xlim(0, xmax)

    for index, value in enumerate(data["Temps visible (s)"]):
        ax.text(
            value + xmax * 0.015,
            index,
            f"{value:.1f}s",
            va="center",
            fontsize=8,
            color="#0f172a",
        )

    fig.tight_layout(pad=1.6)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def save_sov_chart(commercial, out_dir, color_map=None):
    path = out_dir / "graph_part_de_voix.png"
    if commercial.empty:
        return None

    _, default_colors = brand_style_from_commercial(commercial)
    colors = color_map or default_colors

    data = commercial.sort_values("Part de voix (%)", ascending=False).head(6).copy()
    other = (
        commercial.sort_values("Part de voix (%)", ascending=False)
        .iloc[6:]["Part de voix (%)"]
        .sum()
    )
    if other > 0:
        data = pd.concat(
            [
                data,
                pd.DataFrame(
                    {"Logo": ["Autres"], "Part de voix (%)": [other]}
                ),
            ],
            ignore_index=True,
        )

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=170)
    fig.patch.set_facecolor("white")

    values = data["Part de voix (%)"].astype(float).values
    labels = data["Logo"].astype(str).values
    pie_colors = [
        colors.get(str(label), "#d7e0f3" if str(label) == "Autres" else "#46619c")
        for label in labels
    ]

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=pie_colors,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 4 else "",
        startangle=90,
        pctdistance=0.75,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 8, "color": "#172033"},
    )

    centre = plt.Circle((0, 0), 0.46, fc="white")
    ax.add_artist(centre)
    ax.text(
        0, 0.04, "Part de voix",
        ha="center", va="center",
        fontsize=10, fontweight="bold", color="#172033",
    )
    ax.text(
        0, -0.10, "visuelle",
        ha="center", va="center",
        fontsize=8, color="#69758b",
    )

    for label in autotexts:
        label.set_color("white")
        label.set_fontweight("bold")
        label.set_fontsize(8)

    ax.axis("equal")
    fig.tight_layout(pad=1.2)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)

def save_occupation_chart(
    per_frame_logo,
    out_dir,
    brand_order=None,
    color_map=None,
):
    path = out_dir / "graph_occupation_temps.png"
    if per_frame_logo.empty:
        return None

    available = set(per_frame_logo["logo"].astype(str).unique())
    ordered = [
        logo for logo in (brand_order or [])
        if str(logo) in available
    ]
    for logo in per_frame_logo["logo"].astype(str).drop_duplicates():
        if logo not in ordered:
            ordered.append(logo)

    logos = ordered
    colors = color_map or {
        logo: BRAND_COLOR_PALETTE[index % len(BRAND_COLOR_PALETTE)]
        for index, logo in enumerate(logos)
    }

    count = len(logos)
    columns = 2 if count > 1 else 1
    rows = int(np.ceil(count / columns))

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(8.8, max(3.8, rows * 2.25)),
        dpi=170,
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    all_seconds = per_frame_logo["seconde"].astype(float)
    xmin = float(all_seconds.min()) if not all_seconds.empty else 0
    xmax = float(all_seconds.max()) if not all_seconds.empty else 1

    for index, logo in enumerate(logos):
        row_index = index // columns
        column_index = index % columns
        ax = axes[row_index][column_index]

        series = (
            per_frame_logo[
                per_frame_logo["logo"].astype(str) == str(logo)
            ]
            .groupby("seconde", as_index=False)["surface_pct_frame"]
            .sum()
            .sort_values("seconde")
        )

        # Pas de ligne artificielle entre deux détections très éloignées :
        # on casse la courbe quand l'écart temporel est trop important.
        if len(series) > 1:
            gaps = series["seconde"].diff()
            typical_gap = gaps[gaps > 0].median()
            if pd.isna(typical_gap) or typical_gap <= 0:
                typical_gap = 1.0
            break_threshold = max(typical_gap * 3.0, 1.0)

            plot_x = []
            plot_y = []
            previous_second = None

            for _, point in series.iterrows():
                second = float(point["seconde"])
                value = float(point["surface_pct_frame"])

                if (
                    previous_second is not None
                    and second - previous_second > break_threshold
                ):
                    plot_x.append(np.nan)
                    plot_y.append(np.nan)

                plot_x.append(second)
                plot_y.append(value)
                previous_second = second
        else:
            plot_x = series["seconde"].tolist()
            plot_y = series["surface_pct_frame"].tolist()

        color = colors.get(str(logo), "#46619c")
        ax.plot(plot_x, plot_y, linewidth=0.75, color=color, alpha=0.88, solid_capstyle="round")
        ax.fill_between(
            plot_x,
            plot_y,
            0,
            color=color,
            alpha=0.035,
        )

        ax.set_title(
            str(logo),
            loc="left",
            fontsize=9,
            fontweight="bold",
            color="#172033",
            pad=7,
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(bottom=0)
        ax.grid(axis="both", color="#eef2f7", linewidth=0.7)
        ax.tick_params(colors="#64748b", labelsize=7)

        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#e5e7eb")
        ax.spines["bottom"].set_color("#e5e7eb")

        if row_index == rows - 1:
            ax.set_xlabel("Temps (s)", fontsize=7, color="#64748b")
        if column_index == 0:
            ax.set_ylabel("Occupation (%)", fontsize=7, color="#64748b")

    # Masquer les emplacements inutilisés.
    for index in range(count, rows * columns):
        axes[index // columns][index % columns].axis("off")

    fig.tight_layout(pad=1.6, h_pad=1.7, w_pad=1.4)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def save_timeline_chart(
    sequences,
    out_dir,
    brand_order=None,
    color_map=None,
):
    path = out_dir / "graph_timeline.png"
    if sequences is None or sequences.empty:
        return None

    available = set(sequences["logo"].astype(str).unique())
    ordered = [
        logo for logo in (brand_order or [])
        if str(logo) in available
    ]

    # Marques éventuelles non présentes dans commercial.
    for logo in sequences["logo"].astype(str).drop_duplicates():
        if logo not in ordered:
            ordered.append(logo)

    logos = ordered[:12]
    colors = color_map or {
        logo: BRAND_COLOR_PALETTE[index % len(BRAND_COLOR_PALETTE)]
        for index, logo in enumerate(logos)
    }

    fig, ax = figure_base((8.8, max(3.8, 0.48 * len(logos) + 1.2)))

    for y, logo in enumerate(logos):
        sub = sequences[sequences["logo"].astype(str) == str(logo)]
        for _, row in sub.iterrows():
            ax.broken_barh(
                [(
                    row["debut_sec"],
                    max(0.08, row["fin_sec"] - row["debut_sec"]),
                )],
                (y - 0.22, 0.44),
                facecolors=colors.get(str(logo), "#46619c"),
                alpha=0.86,
            )

    ax.set_yticks(range(len(logos)))
    ax.set_yticklabels(logos, fontsize=8, color="#0f172a")
    ax.invert_yaxis()  # la marque la plus visible est en haut, comme le classement
    ax.set_xlabel("Temps dans la vidéo (s)", fontsize=9, color="#64748b")
    ax.set_title("", pad=0)
    ax.grid(axis="x", color="#eef2f7")

    fig.tight_layout(pad=1.6)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def save_heatmap(
    df,
    out_dir,
    width,
    height,
    first_frame_path=None,
    brand_order=None,
    color_map=None,
):
    from matplotlib.colors import LinearSegmentedColormap

    path = out_dir / "carte_densite_visibilite.png"
    if df.empty:
        return None

    available = set(df["logo"].astype(str).unique())
    logos = [
        str(logo) for logo in (brand_order or [])
        if str(logo) in available
    ]
    for logo in df["logo"].astype(str).drop_duplicates():
        if logo not in logos:
            logos.append(logo)

    count = len(logos)
    columns = 2 if count > 1 else 1
    rows = int(np.ceil(count / columns))

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(10.5, max(4.2, rows * 3.25)),
        dpi=180,
        squeeze=False,
    )
    fig.patch.set_facecolor("#f7f9fc")

    for index, logo in enumerate(logos):
        ax = axes[index // columns][index % columns]
        sub = df[df["logo"].astype(str) == logo]

        x = (
            ((sub["x1"] + sub["x2"]) / 2 / max(1, width))
            .clip(0, 1)
            .to_numpy()
        )
        y = (
            ((sub["y1"] + sub["y2"]) / 2 / max(1, height))
            .clip(0, 1)
            .to_numpy()
        )

        heat, _, _ = np.histogram2d(
            y,
            x,
            bins=(30, 52),
            range=[[0, 1], [0, 1]],
        )
        heat = cv2.GaussianBlur(
            heat.astype(np.float32),
            (0, 0),
            sigmaX=2.0,
            sigmaY=2.0,
        )

        brand_color = (color_map or {}).get(logo, "#2563eb")
        cmap = LinearSegmentedColormap.from_list(
            f"heat_{index}",
            ["#05070d", brand_color, "#fff4b0"],
        )

        ax.imshow(
            heat,
            cmap=cmap,
            interpolation="bicubic",
            extent=[0, 100, 100, 0],
            aspect="auto",
        )

        for value in (33.33, 66.66):
            ax.axvline(value, color="white", alpha=.24, lw=.8)
            ax.axhline(value, color="white", alpha=.24, lw=.8)

        ax.set_title(
            logo,
            loc="left",
            fontsize=10,
            fontweight="bold",
            color="#172033",
            pad=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for index in range(count, rows * columns):
        axes[index // columns][index % columns].axis("off")

    fig.tight_layout(pad=1.2, h_pad=1.4, w_pad=1.2)
    fig.savefig(path, bbox_inches="tight", facecolor="#f7f9fc")
    plt.close(fig)
    return str(path)

def save_best_frames(df, frames_dir, out_dir, top_n=5):
    if df.empty or not frames_dir.exists():
        return []
    per_frame = df.groupby("frame", as_index=False).agg(
        seconde=("seconde", "min"),
        logo=("logo", lambda s: ", ".join(sorted(set(s))[:3])),
        nb_logos=("logo", "count"),
        surface_pct=("surface_bbox", "sum"),
    )
    # surface_pct is still pixel area here; only used to rank frames.
    per_frame = per_frame.sort_values(["surface_pct", "nb_logos"], ascending=False).head(top_n)
    gallery = []
    for rank, row in enumerate(per_frame.itertuples(), start=1):
        src = frames_dir / f"frame_{int(row.frame):06d}.jpg"
        if src.exists():
            caption = f"#{rank} · {format_time(row.seconde)} · {row.nb_logos} logo(s) · {row.logo}"
            gallery.append((str(src), caption))
    return gallery


def analyze_worker(model_file, video_file, confidence, frame_skip, image_size, selected_brands, state):
    try:
        state["start_time"] = time.time()
        state["label"] = "Préparation de l'analyse..."
        state["percent"] = 0

        run = "analysis_" + time_id()
        output_dir = RESULTS_DIR / run
        charts_dir = output_dir / "graphiques"
        frames_dir = output_dir / "meilleures_images"
        output_dir.mkdir(parents=True, exist_ok=True)
        charts_dir.mkdir(exist_ok=True)
        frames_dir.mkdir(exist_ok=True)

        model = YOLO(model_file)
        available_brands = set(str(v) for v in (model.names.values() if isinstance(model.names, dict) else model.names))
        selected_brands = [str(x) for x in (selected_brands or []) if str(x) in available_brands]
        if not selected_brands:
            selected_brands = sorted(available_brands)
        selected_brand_set = set(selected_brands)
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise RuntimeError("Impossible d'ouvrir la vidéo.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps else 0

        output_video = output_dir / "video_annotee.mp4"
        detections_csv = output_dir / "detections_completes.csv"
        stats_csv = output_dir / "statistiques_logos.csv"
        table_csv = output_dir / "tableau_commercial.csv"
        sequences_csv = output_dir / "sequences_apparition.csv"
        first_frame_path = output_dir / "image_reference.jpg"

        writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        detections = []
        frame_index = 0
        frame_skip = int(frame_skip)
        image_size = int(image_size)
        saved_reference = False
        candidate_frames = []

        while True:
            if state.get("stop_requested"):
                state["stopped"] = True
                state["label"] = "Analyse arrêtée — sauvegarde des résultats partiels..."
                break

            ok, frame = cap.read()
            if not ok:
                break

            if not saved_reference:
                cv2.imwrite(str(first_frame_path), frame)
                saved_reference = True

            annotated = frame.copy()
            frame_detections = []

            if frame_index % frame_skip == 0:
                result = model.predict(frame, conf=float(confidence), imgsz=image_size, verbose=False, device=DEVICE, half=(DEVICE == 0))[0]
                annotated = draw_custom_detections(frame, result, model.names, selected_brand_set)

                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        logo = model.names.get(class_id, str(class_id))
                        if logo not in selected_brand_set:
                            continue
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                        surface = max(0, x2 - x1) * max(0, y2 - y1)
                        record = {
                            "frame": frame_index,
                            "seconde": frame_index / fps,
                            "timecode": format_time(frame_index / fps),
                            "logo": logo,
                            "confiance": conf,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "surface_bbox": surface,
                        }
                        detections.append(record)
                        frame_detections.append(record)

                if frame_detections:
                    total_surface = sum(d["surface_bbox"] for d in frame_detections)
                    candidate_frames.append((total_surface, frame_index, annotated.copy()))
                    candidate_frames = sorted(candidate_frames, key=lambda x: x[0], reverse=True)[:12]

            writer.write(annotated)
            frame_index += 1

            if total_frames > 0 and frame_index % 10 == 0:
                state["percent"] = frame_index / total_frames * 100
                state["label"] = f"Analyse vidéo — frame {frame_index}/{total_frames}"

        cap.release()
        writer.release()

        for _, idx, img in candidate_frames[:8]:
            cv2.imwrite(str(frames_dir / f"frame_{idx:06d}.jpg"), img)

        df = pd.DataFrame(detections)
        df.to_csv(detections_csv, index=False)
        (output_dir / "analyse_metadata.json").write_text(json.dumps({
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "selected_brands": selected_brands,
            "confidence": float(confidence),
            "frame_skip": int(frame_skip),
            "image_size": int(image_size),
            "video_duration_seconds": float(duration),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        if df.empty:
            empty = pd.DataFrame()
            empty.to_csv(stats_csv, index=False)
            empty.to_csv(table_csv, index=False)
            empty.to_csv(sequences_csv, index=False)
            state.update({
                "label": "Analyse terminée — aucun logo détecté" if not state.get("stopped") else "Analyse arrêtée — aucun logo détecté avant l'arrêt",
                "video_path": str(output_video),
                "detections_path": str(detections_csv),
                "stats_path": str(stats_csv),
                "table_path": str(table_csv),
                "stats_table": empty,
                "table_html": make_table_html(empty),
                "kpis_html": "",
                "ranking_chart": None,
                "sov_chart": None,
                "timeline_chart": None,
                "occupation_chart": None,
                "heatmap_path": None,
                "gallery": [],
                "report_path": None,
                "done": True,
            })
            return

        seq = make_sequences(df, fps)
        seq.to_csv(sequences_csv, index=False)
        stats, commercial, per_frame_logo = build_stats(df, fps, duration, width, height, seq)
        stats.to_csv(stats_csv, index=False)
        commercial.to_csv(table_csv, index=False)

        brand_order, brand_colors = brand_style_from_commercial(commercial)

        ranking_chart = save_ranking_chart(
            commercial,
            charts_dir,
            color_map=brand_colors,
        )
        sov_chart = save_sov_chart(
            commercial,
            charts_dir,
            color_map=brand_colors,
        )
        timeline_chart = save_timeline_chart(
            seq,
            charts_dir,
            brand_order=brand_order,
            color_map=brand_colors,
        )
        occupation_chart = save_occupation_chart(
            per_frame_logo,
            charts_dir,
            brand_order=brand_order,
            color_map=brand_colors,
        )

        # Reconstruit la vidéo avec une couleur d'encadrement par marque.
        rebuild_annotated_video(
            video_file,
            df,
            output_video,
            fps,
            width,
            height,
            brand_colors,
        )
        heatmap_path = save_heatmap(
            df,
            charts_dir,
            width,
            height,
            first_frame_path,
            brand_order=brand_order,
            color_map=brand_colors,
        )
        gallery = save_best_frames(df, frames_dir, output_dir)
        try:
            report_path = make_report_pdf_file(
                commercial,
                [
                    ("Classement par temps visible", ranking_chart),
                    ("Part de voix visuelle", sov_chart),
                    ("Timeline des apparitions", timeline_chart),
                    ("Occupation écran dans le temps", occupation_chart),
                    ("Carte de densité des positions", heatmap_path),
                ],
                output_dir,
            )
        except Exception as report_error:
            print("Erreur génération rapport HTML :", report_error)
            report_path = None

        if state.get("stopped"):
            state["label"] = f"Analyse arrêtée — résultats partiels sauvegardés dans results/{run}"
        else:
            state["percent"] = 100
            state["label"] = f"Analyse terminée — résultats sauvegardés dans results/{run}"

        state.update({
            "video_path": str(output_video),
            "detections_path": str(detections_csv),
            "stats_path": str(stats_csv),
            "table_path": str(table_csv),
            "stats_table": commercial,
            "table_html": make_table_html(commercial),
            "kpis_html": make_kpis_html(commercial),
            "ranking_chart": ranking_chart,
            "sov_chart": sov_chart,
            "timeline_chart": timeline_chart,
            "occupation_chart": occupation_chart,
            "heatmap_path": heatmap_path,
            "gallery": gallery,
            "report_path": report_path,
            "done": True,
        })

    except Exception as e:
        state["error"] = str(e)
        state["done"] = True
        state.update({
            "video_path": None,
            "detections_path": None,
            "stats_path": None,
            "table_path": None,
            "stats_table": pd.DataFrame(),
            "table_html": make_table_html(pd.DataFrame()),
            "kpis_html": "",
            "ranking_chart": None,
            "sov_chart": None,
            "timeline_chart": None,
            "occupation_chart": None,
            "heatmap_path": None,
            "gallery": [],
            "report_path": None,
        })
        print(traceback.format_exc())



def make_report_pdf_file(commercial, chart_paths, output_dir):
    """Rapport premium, lisible et cohérent avec l'identité visuelle de l'application."""
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import FancyBboxPatch, Circle
    from PIL import Image as PILImage
    import textwrap

    path = output_dir / "rapport_visibilite_logos.pdf"
    NAVY, BLUE, BLUE2 = "#172033", "#46619c", "#7c8fbd"
    BG, CARD, LINE = "#eef3f8", "#ffffff", "#dfe7f2"
    TEXT, MUTED, PALE = "#172033", "#64748b", "#f6f9fd"

    def page():
        fig = plt.figure(figsize=(11.69, 8.27), dpi=240, facecolor=BG)
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        return fig, ax

    def box(ax, x, y, w, h, fc=CARD, ec=LINE, radius=.018, lw=.8):
        p = FancyBboxPatch((x,y),w,h, boxstyle=f"round,pad=0.008,rounding_size={radius}",
                           transform=ax.transAxes, facecolor=fc, edgecolor=ec, linewidth=lw)
        ax.add_patch(p); return p

    def txt(ax, x, y, s, size=10, color=TEXT, weight="normal", family="DejaVu Sans", ha="left", va="baseline"):
        ax.text(x,y,str(s), transform=ax.transAxes, fontsize=size, color=color,
                fontweight=weight, fontfamily=family, ha=ha, va=va)

    def head(ax, title, subtitle, no):
        txt(ax,.065,.935,"TSM  ·  SPONSORING INTELLIGENCE",8,BLUE,"bold")
        txt(ax,.065,.875,title,24,TEXT,"bold","DejaVu Serif")
        txt(ax,.065,.835,subtitle,9,MUTED)
        txt(ax,.935,.925,f"{no:02d}",15,"#b9c5d8","bold",ha="right")
        ax.plot([.065,.935],[.805,.805],transform=ax.transAxes,color=LINE,lw=.8)

    def chart_lookup(word):
        for title, p in chart_paths:
            if word.lower() in title.lower() and p and Path(p).exists(): return p
        return None

    def image_in(fig, path_img, x,y,w,h):
        if not path_img or not Path(path_img).exists(): return
        im = PILImage.open(path_img).convert("RGB")
        ia = fig.add_axes([x,y,w,h]); ia.imshow(im); ia.axis("off")

    def insight(ax, x,y,w,h,title,body):
        box(ax,x,y,w,h,PALE,LINE,.014)
        ax.add_patch(Circle((x+.025,y+h-.033),.007,transform=ax.transAxes,facecolor=BLUE,edgecolor="none"))
        txt(ax,x+.043,y+h-.036,title,9,BLUE,"bold",va="center")
        txt(ax,x+.025,y+h-.067,"\n".join(textwrap.wrap(body,105)),8.4,MUTED,va="top")

    data = commercial.copy() if commercial is not None else pd.DataFrame()
    if data.empty:
        with PdfPages(path) as pdf:
            fig,ax=page(); head(ax,"Rapport de visibilité","Aucune donnée exploitable n'a été détectée.",1); pdf.savefig(fig); plt.close(fig)
        return str(path)

    top = data.sort_values("Temps visible (s)",ascending=False).iloc[0]
    sov_top = data.sort_values("Part de voix (%)",ascending=False).iloc[0]
    logo=str(top["Logo"]); temps=float(top["Temps visible (s)"]); pct=float(top["% vidéo"])
    occ=float(top["Occupation max (%)"]); sov=float(sov_top["Part de voix (%)"])
    seq=int(round(float(top["Nb de séquences"]))); central=float(top["Centralité (%)"]); net=float(top["Netteté moy."])

    with PdfPages(path) as pdf:
        # 1 — couverture éditoriale
        fig,ax=page()
        box(ax,.045,.055,.91,.89,"#f9fbfe",LINE,.035)
        box(ax,.07,.52,.86,.37,NAVY,NAVY,.03,0)
        ax.add_patch(Circle((.82,.73),.18,transform=ax.transAxes,facecolor=BLUE,edgecolor="none",alpha=.62))
        ax.add_patch(Circle((.91,.84),.08,transform=ax.transAxes,facecolor="white",edgecolor="none",alpha=.12))
        txt(ax,.105,.82,"TSM  ·  SPONSORING INTELLIGENCE",9,"#dce7ff","bold")
        txt(ax,.105,.735,"Rapport de visibilité",31,"white","bold","DejaVu Serif")
        txt(ax,.105,.675,"des marques",31,"white","bold","DejaVu Serif")
        txt(ax,.105,.595,"Analyse de l’exposition visuelle détectée dans la vidéo",11,"#dce7ff")
        labels=[("MARQUE DOMINANTE",logo),("TEMPS VISIBLE",f"{temps:.2f} s"),("OCCUPATION MAX",f"{occ:.2f} %"),("PART DE VOIX",f"{sov:.1f} %")]
        cw=.197
        for i,(lab,val) in enumerate(labels):
            x=.07+i*(cw+.024); box(ax,x,.325,cw,.125,CARD,LINE,.018)
            txt(ax,x+.022,.415,lab,7.2,MUTED,"bold"); txt(ax,x+.022,.36,val,15.5,TEXT,"bold","DejaVu Serif")
        insight(ax,.07,.145,.86,.11,"SYNTHÈSE EXÉCUTIVE",f"{logo} est la marque la plus visible : {temps:.2f} s d’exposition cumulée ({pct:.1f} % de la vidéo), réparties sur {seq} séquences. Sa part de voix visuelle atteint {sov:.1f} %.")
        pdf.savefig(fig,facecolor=fig.get_facecolor()); plt.close(fig)

        # 2 — synthèse KPI + tableaux propres
        fig,ax=page(); head(ax,"Synthèse des performances","Une lecture compacte des indicateurs temporels et qualitatifs.",2)
        kpis=[("TEMPS VISIBLE",f"{temps:.2f} s"),("PART DE LA VIDÉO",f"{pct:.1f} %"),("SÉQUENCES",str(seq)),("CENTRALITÉ",f"{central:.1f} %"),("NETTETÉ MOY.",f"{net:.2f}")]
        for i,(lab,val) in enumerate(kpis):
            x=.065+i*.176; box(ax,x,.66,.155,.105,CARD,LINE,.014)
            txt(ax,x+.018,.728,lab,6.7,MUTED,"bold"); txt(ax,x+.018,.684,val,15,TEXT,"bold","DejaVu Serif")
        cols1=["Logo","Temps visible (s)","% vidéo","Nb de séquences","Durée max. séquence (s)"]
        cols2=["Logo","Occupation max (%)","Centralité (%)","Part de voix (%)","Netteté moy."]
        def draw_table(y,title,cols):
            txt(ax,.07,y+.19,title,12,TEXT,"bold","DejaVu Serif")
            vals=data[cols].head(6).copy()
            table=ax.table(cellText=vals.values, colLabels=cols, cellLoc="center", bbox=[.07,y,.86,.15])
            table.auto_set_font_size(False); table.set_fontsize(7.3)
            for (r,c),cell in table.get_celld().items():
                cell.set_edgecolor(LINE); cell.set_linewidth(.55)
                if r==0: cell.set_facecolor("#f1f5fa"); cell.set_text_props(weight="bold",color=TEXT)
                else: cell.set_facecolor("white"); cell.set_text_props(color=TEXT)
        draw_table(.42,"Visibilité temporelle",cols1); draw_table(.17,"Qualité d’exposition",cols2)
        insight(ax,.07,.045,.86,.075,"À RETENIR",f"La visibilité de {logo} combine durée, répétition et qualité de placement. La confiance moyenne ({net:.2f}) décrit la netteté des détections et non la précision globale du modèle.")
        pdf.savefig(fig,facecolor=fig.get_facecolor()); plt.close(fig)

        # 3 — dashboard graphique, grands visuels
        fig,ax=page(); head(ax,"Dashboard graphique","Les quatre vues essentielles, regroupées sur une page.",3)
        charts=[("Classement par temps visible",chart_lookup("Classement")),("Part de voix visuelle",chart_lookup("Part de voix")),("Timeline des apparitions",chart_lookup("Timeline")),("Occupation écran dans le temps",chart_lookup("Occupation"))]
        positions=[(.06,.47,.425,.29),(.515,.47,.425,.29),(.06,.12,.425,.29),(.515,.12,.425,.29)]
        for (title,p),(x,y,w,h) in zip(charts,positions):
            box(ax,x,y,w,h,CARD,LINE,.016); txt(ax,x+.02,y+h-.035,title,10,TEXT,"bold","DejaVu Serif",va="center")
            image_in(fig,p,x+.025,y+.025,w-.05,h-.075)
        pdf.savefig(fig,facecolor=fig.get_facecolor()); plt.close(fig)

        # 4 — analyse commerciale
        fig,ax=page(); head(ax,"Analyse de visibilité","Interprétation des résultats pour une lecture marque et sponsoring.",4)
        blocks=[
            ("01","Présence",f"{logo} cumule {temps:.2f} secondes de visibilité, soit {pct:.1f} % de la durée analysée. L’exposition est répartie sur {seq} séquences distinctes."),
            ("02","Impact visuel",f"Le pic d’occupation atteint {occ:.2f} % de l’écran. La centralité de {central:.1f} % indique la fréquence avec laquelle la marque apparaît dans la zone centrale de l’image."),
            ("03","Domination concurrentielle",f"La meilleure part de voix visuelle atteint {sov:.1f} %. Cet indicateur compare le poids visuel relatif des marques détectées dans la séquence analysée."),
        ]
        yy=.65
        for no,title,body in blocks:
            box(ax,.07,yy,.86,.14,CARD,LINE,.016); txt(ax,.095,yy+.092,no,18,BLUE,"bold","DejaVu Serif"); txt(ax,.16,yy+.096,title,12,TEXT,"bold","DejaVu Serif"); txt(ax,.16,yy+.064,"\n".join(textwrap.wrap(body,112)),8.5,MUTED,va="top"); yy-=.18
        insight(ax,.07,.09,.86,.105,"CONCLUSION",f"Sur cette vidéo, {logo} constitue le principal actif de visibilité mesuré. Le rapport met en évidence non seulement la durée d’exposition, mais aussi la répétition, la surface occupée, la centralité et la part de voix — des indicateurs complémentaires pour valoriser une présence sponsor.")
        pdf.savefig(fig,facecolor=fig.get_facecolor()); plt.close(fig)

        # 5 — carte de densité
        fig,ax=page(); head(ax,"Carte de densité des positions","Localisation des zones où les logos apparaissent le plus fréquemment.",5)
        p=chart_lookup("densité")
        box(ax,.065,.24,.87,.52,CARD,LINE,.018); image_in(fig,p,.085,.27,.83,.45)
        insight(ax,.065,.09,.87,.095,"LECTURE", "Les zones les plus chaudes correspondent aux emplacements les plus fréquemment occupés par les détections. Cette vue aide à comparer la valeur des emplacements à l’écran et la récurrence des zones d’exposition.")
        pdf.savefig(fig,facecolor=fig.get_facecolor()); plt.close(fig)
    return str(path)

def analyze_video(model_file, video_file, confidence, frame_skip, image_size, selected_brands):
    empty = pd.DataFrame()
    if model_file is None:
        yield alert_html("Ajoute un modèle best.pt."), None, None, None, None, make_table_html(empty), "", None, None, None, None, None, None, gr.update(visible=False), gr.update(visible=True)
        return
    if video_file is None:
        yield alert_html("Ajoute une vidéo à analyser."), None, None, None, None, make_table_html(empty), "", None, None, None, None, None, None, gr.update(visible=False), gr.update(visible=True)
        return

    state = {
        "start_time": time.time(),
        "label": "Préparation de l'analyse...",
        "percent": 0,
        "done": False,
        "error": None,
        "stop_requested": False,
        "stopped": False,
    }
    CURRENT_ANALYSIS["state"] = state

    thread = threading.Thread(target=analyze_worker, args=(model_file, video_file, confidence, frame_skip, image_size, selected_brands, state), daemon=True)
    thread.start()

    while not state["done"]:
        yield progress_html(state), None, None, None, None, make_table_html(empty), "", None, None, None, None, None, None, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
        time.sleep(1)

    yield (
        progress_html(state),
        state.get("video_path"),
        state.get("detections_path"),
        state.get("stats_path"),
        state.get("table_path"),
        state.get("table_html", make_table_html(empty)),
        state.get("kpis_html", ""),
        state.get("ranking_chart"),
        state.get("sov_chart"),
        state.get("timeline_chart"),
        state.get("occupation_chart"),
        state.get("heatmap_path"),
        state.get("report_path"),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
    )

def stop_analysis():
    state = CURRENT_ANALYSIS.get("state")
    if state is None or state.get("done"):
        return alert_html("Aucune analyse en cours.", "info")
    state["stop_requested"] = True
    state["stopped"] = True
    state["label"] = "Arrêt demandé — l'analyse va s'arrêter."
    return progress_html(state)


def validate_file_step(file, next_visible=True):
    if file is None:
        return alert_html("Fichier manquant.", "error"), gr.update(visible=False), gr.update(visible=True)
    return alert_html("Fichier validé. Étape suivante débloquée.", "success"), gr.update(visible=next_visible), gr.update(visible=False)





def select_library_visibility(value):
    labels = {
        "private": "Privé — seulement moi",
        "selected": "Partagé — utilisateurs choisis",
    }
    value = value if value in labels else "private"
    return (
        labels[value],
        gr.update(variant="primary" if value == "private" else "secondary"),
        gr.update(variant="primary" if value == "selected" else "secondary"),
    )


def library_progress_html(active_step=1):
    items = [
        ("Fichier", "Choisir la ressource"),
        ("Informations", "Nom et description"),
        ("Partage", "Définir les accès"),
    ]
    parts = []
    for index, (title, subtitle) in enumerate(items, start=1):
        state = "active" if index == active_step else ("done" if index < active_step else "")
        check = "✓" if index < active_step else str(index)
        parts.append(
            f"""<div class="library-progress-item {state}">
                    <span>{check}</span>
                    <div><strong>{title}</strong><small>{subtitle}</small></div>
                </div>"""
        )
        if index < len(items):
            line_state = "done" if index < active_step else ""
            parts.append(f'<div class="library-progress-line {line_state}"></div>')
    return '<div class="library-wizard-progress">' + "".join(parts) + "</div>"


def library_go_to_step_2(uploaded_file):
    if uploaded_file is None:
        return (
            alert_html("Ajoutez d’abord un dataset, un modèle ou une vidéo.", "error"),
            library_progress_html(1),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    return (
        "",
        library_progress_html(2),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
    )


def library_go_to_step_3(display_name):
    clean_name = (display_name or "").strip()
    if not clean_name:
        return (
            alert_html("Indiquez un nom pour cette ressource.", "error"),
            library_progress_html(2),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
        )

    with db_connection() as conn:
        duplicate_name = conn.execute(
            """
            SELECT display_name, owner
            FROM library_items
            WHERE LOWER(TRIM(display_name)) = LOWER(TRIM(?))
            LIMIT 1
            """,
            (clean_name,),
        ).fetchone()

    if duplicate_name is not None:
        return (
            alert_html(
                f"Le nom « {clean_name} » est déjà utilisé "
                f"(propriétaire : {duplicate_name['owner']}).",
                "error",
            ),
            library_progress_html(2),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
        )

    return (
        "",
        library_progress_html(3),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
    )


def library_back_to_step_1():
    return (
        "",
        library_progress_html(1),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def library_back_to_step_2():
    return (
        "",
        library_progress_html(2),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
    )


def save_library_item_wizard(
    uploaded_file,
    display_name,
    visibility_label,
    shared_users,
    description,
    request: gr.Request,
):
    status, table_html, users_html = save_library_item(
        uploaded_file,
        display_name,
        visibility_label,
        shared_users,
        description,
        request,
    )

    success = "background:#f0fdf4" in status
    if success:
        return (
            status,
            table_html,
            users_html,
            library_progress_html(1),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            None,
            "",
            "",
            "Privé — seulement moi",
            "",
            gr.update(variant="primary"),
            gr.update(variant="secondary"),
            gr.update(variant="secondary"),
        )

    return (
        status,
        table_html,
        users_html,
        library_progress_html(3),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
    )


def validate_training_dataset(source_mode, uploaded_file, library_item_id, request: gr.Request):
    if source_mode == "Choisir dans la bibliothèque":
        username = request_username(request)
        row = get_library_item(library_item_id, username)
        if row["item_type"] != "Dataset ZIP":
            return (
                alert_html("La ressource choisie n’est pas un dataset ZIP.", "error"),
                None,
                gr.update(visible=False),
                gr.update(visible=True),
            )
        return (
            alert_html(
                f"Dataset « {row['display_name']} » sélectionné depuis la bibliothèque.",
                "success",
            ),
            str(row["stored_path"]),
            gr.update(visible=True),
            gr.update(visible=False),
        )

    if uploaded_file is None:
        return (
            alert_html("Ajoutez un fichier ZIP.", "error"),
            None,
            gr.update(visible=False),
            gr.update(visible=True),
        )

    return (
        alert_html("Fichier ZIP validé. Étape suivante débloquée.", "success"),
        uploaded_file,
        gr.update(visible=True),
        gr.update(visible=False),
    )



def validate_model_step(file):
    return validate_file_step(file, True)


def validate_video_step(file):
    return validate_file_step(file, True)


# ============================================================
# AUTHENTIFICATION ET BIBLIOTHÈQUE PARTAGÉE
# ============================================================

APP_DB = ROOT / "app_data.sqlite3"
LIBRARY_DIR = ROOT / "library"
RESULT_ARCHIVES_DIR = ROOT / "saved_results"
RESULT_ARCHIVES_V2_DIR = ROOT / "saved_results_v2"
LIBRARY_DIR.mkdir(exist_ok=True)
RESULT_ARCHIVES_DIR.mkdir(exist_ok=True)
RESULT_ARCHIVES_V2_DIR.mkdir(exist_ok=True)

ADMIN_USERNAME = os.getenv("APP_ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("APP_ADMIN_PASSWORD", "TSM-Admin-2026!")


def db_connection():
    conn = sqlite3.connect(APP_DB, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password, salt=None):
    if not isinstance(password, str) or len(password) < 8:
        raise ValueError("Le mot de passe doit contenir au moins 8 caractères.")
    salt_bytes = bytes.fromhex(salt) if salt else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt_bytes,
        260_000,
    )
    return salt_bytes.hex(), digest.hex()


def init_app_database():
    with db_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS library_items (
                id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                display_name TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                item_type TEXT NOT NULL,
                visibility TEXT NOT NULL DEFAULT 'private',
                shared_with TEXT NOT NULL DEFAULT '[]',
                description TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(owner) REFERENCES users(username)
            );

            CREATE TABLE IF NOT EXISTS brand_logos (
                id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                brand_name TEXT NOT NULL,
                stored_filename TEXT NOT NULL,
                visibility TEXT NOT NULL DEFAULT 'private',
                shared_with TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(owner, brand_name),
                FOREIGN KEY(owner) REFERENCES users(username)
            );

            CREATE TABLE IF NOT EXISTS result_archives (
                id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                title TEXT NOT NULL,
                files_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(owner) REFERENCES users(username)
            );

            CREATE TABLE IF NOT EXISTS result_archives_v2 (
                id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                title TEXT NOT NULL,
                files_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(owner) REFERENCES users(username)
            );
            """
        )

        # Migration douce pour les anciennes bases déjà créées.
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(library_items)").fetchall()
        }
        if "content_hash" not in columns:
            conn.execute(
                "ALTER TABLE library_items ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''"
            )

        existing = conn.execute(
            "SELECT username FROM users WHERE username = ?",
            (ADMIN_USERNAME,),
        ).fetchone()

        if existing is None:
            salt, password_hash = hash_password(ADMIN_PASSWORD)
            conn.execute(
                """
                INSERT INTO users(username, password_hash, salt, is_admin, is_active)
                VALUES (?, ?, ?, 1, 1)
                """,
                (ADMIN_USERNAME, password_hash, salt),
            )

        # Compte client de démonstration, créé uniquement s'il n'existe pas.
        demo_username = os.getenv("APP_DEMO_USERNAME", "client")
        demo_password = os.getenv("APP_DEMO_PASSWORD", "Client-TSM-2026!")
        demo_existing = conn.execute(
            "SELECT username FROM users WHERE username = ?",
            (demo_username,),
        ).fetchone()
        if demo_existing is None:
            demo_salt, demo_hash = hash_password(demo_password)
            conn.execute(
                """
                INSERT INTO users(username, password_hash, salt, is_admin, is_active)
                VALUES (?, ?, ?, 0, 1)
                """,
                (demo_username, demo_hash, demo_salt),
            )


init_app_database()


def authenticate_user(username, password):
    username = (username or "").strip()
    with db_connection() as conn:
        row = conn.execute(
            """
            SELECT username, password_hash, salt, is_active
            FROM users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()

    if row is None or not row["is_active"]:
        return False

    try:
        _, candidate_hash = hash_password(password or "", row["salt"])
    except Exception:
        return False

    return secrets.compare_digest(candidate_hash, row["password_hash"])


def request_username(request: gr.Request):
    username = getattr(request, "username", None)
    if not username:
        raise gr.Error("Vous devez être connecté.")
    return username


def user_is_admin(username):
    with db_connection() as conn:
        row = conn.execute(
            "SELECT is_admin, is_active FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    return bool(row and row["is_admin"] and row["is_active"])


def create_or_update_user(username, password, make_admin, request: gr.Request):
    current_user = request_username(request)
    if not user_is_admin(current_user):
        return alert_html("Action réservée à l’administrateur.", "error"), render_users_html(current_user)

    username = (username or "").strip()
    if not username or any(c.isspace() for c in username):
        return alert_html("Choisissez un identifiant sans espace.", "error"), render_users_html(current_user)

    try:
        salt, password_hash = hash_password(password or "")
    except ValueError as error:
        return alert_html(str(error), "error"), render_users_html(current_user)

    with db_connection() as conn:
        exists = conn.execute(
            "SELECT username FROM users WHERE username = ?",
            (username,),
        ).fetchone()

        if exists:
            return (
                alert_html(
                    f"Le nom d’utilisateur « {username} » existe déjà. Choisissez-en un autre.",
                    "error",
                ),
                render_users_html(current_user),
            )

        conn.execute(
            """
            INSERT INTO users(username, password_hash, salt, is_admin, is_active)
            VALUES (?, ?, ?, ?, 1)
            """,
            (username, password_hash, salt, int(bool(make_admin))),
        )
        message = f"Le compte « {username} » a été créé."

    return alert_html(message, "success"), render_users_html(current_user)


def toggle_user_access(username, request: gr.Request):
    current_user = request_username(request)
    if not user_is_admin(current_user):
        return alert_html("Action réservée à l’administrateur.", "error"), render_users_html(current_user)

    username = (username or "").strip()
    if not username:
        return alert_html("Indiquez un identifiant.", "error"), render_users_html(current_user)
    if username == current_user:
        return alert_html("Vous ne pouvez pas désactiver votre propre compte.", "error"), render_users_html(current_user)

    with db_connection() as conn:
        row = conn.execute(
            "SELECT is_active FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if row is None:
            return alert_html("Compte introuvable.", "error"), render_users_html(current_user)

        new_status = 0 if row["is_active"] else 1
        conn.execute(
            "UPDATE users SET is_active = ? WHERE username = ?",
            (new_status, username),
        )

    action = "réactivé" if new_status else "désactivé"
    return alert_html(f"Le compte « {username} » a été {action}.", "success"), render_users_html(current_user)


def render_users_html(current_user):
    if not user_is_admin(current_user):
        return "<div class='library-empty'>La gestion des comptes est réservée à l’administrateur.</div>"

    with db_connection() as conn:
        rows = conn.execute(
            """
            SELECT username, is_admin, is_active, created_at
            FROM users
            ORDER BY is_admin DESC, username COLLATE NOCASE
            """
        ).fetchall()

    body = "".join(
        f"""
        <tr>
            <td>{row['username']}</td>
            <td>{'Administrateur' if row['is_admin'] else 'Utilisateur'}</td>
            <td>{'Actif' if row['is_active'] else 'Désactivé'}</td>
            <td>{row['created_at']}</td>
        </tr>
        """
        for row in rows
    )
    return f"""
    <div class="library-table-wrap">
        <table class="library-table">
            <thead><tr><th>Identifiant</th><th>Rôle</th><th>Accès</th><th>Création</th></tr></thead>
            <tbody>{body}</tbody>
        </table>
    </div>
    """



def render_profile_html(request: gr.Request):
    username = request_username(request)
    role = "Administrateur" if user_is_admin(username) else "Utilisateur"
    initials = "".join(part[0].upper() for part in username.replace("_", " ").split()[:2]) or "U"
    return f"""
    <section class="profile-page">
      <div class="profile-shell">
        <div class="profile-visual">
          <div class="profile-visual-orb orb-one"></div>
          <div class="profile-visual-orb orb-two"></div>
          <div class="profile-logo-large">
            <svg viewBox="0 0 64 64" aria-hidden="true">
              <path d="M13 32c5.5-8.5 11.8-13 19-13s13.5 4.5 19 13c-5.5 8.5-11.8 13-19 13S18.5 40.5 13 32Z" fill="none" stroke="white" stroke-width="4"/>
              <circle cx="32" cy="32" r="7" fill="white"/>
            </svg>
          </div>
          <div class="profile-visual-copy">
            <span>ESPACE PERSONNEL</span>
            <h2>Votre compte, en toute simplicité.</h2>
            <p>Consultez vos informations de connexion et quittez l’application en toute sécurité.</p>
          </div>
        </div>
        <div class="profile-card">
          <div class="profile-avatar">{initials}</div>
          <div class="profile-kicker">Compte connecté</div>
          <h1>{username}</h1>
          <div class="profile-role">{role}</div>
          <div class="profile-info-list">
            <div class="profile-info-row"><span>Identifiant</span><strong>{username}</strong></div>
            <div class="profile-info-row"><span>Rôle</span><strong>{role}</strong></div>
            <div class="profile-info-row"><span>État du compte</span><strong class="profile-active">Actif</strong></div>
          </div>
          <div class="profile-divider"></div>
          <div class="logout-copy">
            <strong>Terminer la session</strong>
            <span>Vous devrez saisir à nouveau vos identifiants pour revenir dans l’application.</span>
          </div>
          <button class="logout-button" type="button" onclick="window.safeLogout()">
            <span class="logout-icon">↗</span> Se déconnecter
          </button>
        </div>
      </div>
    </section>
    """


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as file_handle:
        while True:
            chunk = file_handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()



def normalize_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None
    if isinstance(uploaded_file, str):
        return Path(uploaded_file)
    if hasattr(uploaded_file, "name"):
        return Path(uploaded_file.name)
    return Path(str(uploaded_file))


def infer_library_type(path):
    suffix = path.suffix.lower()
    if suffix == ".zip":
        return "Dataset ZIP"
    if suffix == ".pt":
        return "Modèle YOLO"
    if suffix in {".mp4", ".mov", ".avi", ".mkv"}:
        return "Vidéo de match"
    raise ValueError(
        "Formats acceptés : dataset .zip, modèle .pt ou vidéo .mp4, .mov, .avi, .mkv."
    )


def save_library_item(uploaded_file, display_name, visibility_label, shared_users, description, request: gr.Request):
    username = request_username(request)
    source = normalize_uploaded_file(uploaded_file)
    if source is None or not source.exists():
        return alert_html("Ajoutez un dataset ZIP, un modèle PT ou une vidéo de match.", "error"), *refresh_library_outputs(request)

    try:
        item_type = infer_library_type(source)
    except ValueError as error:
        return alert_html(str(error), "error"), *refresh_library_outputs(request)

    visibility_map = {
        "Privé — seulement moi": "private",
        "Partagé — utilisateurs choisis": "selected",
    }
    visibility = visibility_map.get(visibility_label, "private")

    requested_users = [
        value.strip()
        for value in (shared_users or "").replace(";", ",").split(",")
        if value.strip()
    ]

    if visibility == "selected":
        with db_connection() as conn:
            valid_users = {
                row["username"]
                for row in conn.execute(
                    "SELECT username FROM users WHERE is_active = 1"
                ).fetchall()
            }
        invalid = sorted(set(requested_users) - valid_users)
        if invalid:
            return alert_html(
                "Utilisateurs inconnus ou désactivés : " + ", ".join(invalid),
                "error",
            ), *refresh_library_outputs(request)
        if not requested_users:
            return alert_html(
                "Indiquez au moins un utilisateur avec qui partager le fichier.",
                "error",
            ), *refresh_library_outputs(request)

    content_hash = file_sha256(source)

    with db_connection() as conn:
        duplicate = conn.execute(
            """
            SELECT display_name, owner
            FROM library_items
            WHERE content_hash = ?
            LIMIT 1
            """,
            (content_hash,),
        ).fetchone()

    if duplicate is not None:
        return (
            alert_html(
                f"Ce fichier existe déjà dans la bibliothèque sous le nom "
                f"« {duplicate['display_name']} » (propriétaire : {duplicate['owner']}).",
                "error",
            ),
            *refresh_library_outputs(request),
        )

    item_id = uuid.uuid4().hex
    owner_dir = LIBRARY_DIR / username
    owner_dir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{item_id}{source.suffix.lower()}"
    destination = owner_dir / stored_name
    shutil.copy2(source, destination)

    clean_name = (display_name or "").strip() or source.stem
    clean_description = (description or "").strip()

    with db_connection() as conn:
        duplicate_name = conn.execute(
            """
            SELECT display_name, owner
            FROM library_items
            WHERE LOWER(TRIM(display_name)) = LOWER(TRIM(?))
            LIMIT 1
            """,
            (clean_name,),
        ).fetchone()

    if duplicate_name is not None:
        return (
            alert_html(
                f"Le nom « {clean_name} » est déjà utilisé dans la bibliothèque "
                f"(propriétaire : {duplicate_name['owner']}). Choisissez un autre nom.",
                "error",
            ),
            *refresh_library_outputs(request),
        )

    with db_connection() as conn:
        conn.execute(
            """
            INSERT INTO library_items(
                id, owner, display_name, stored_path, item_type,
                visibility, shared_with, description, content_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                username,
                clean_name,
                str(destination),
                item_type,
                visibility,
                json.dumps(requested_users, ensure_ascii=False),
                clean_description,
                content_hash,
            ),
        )

    return alert_html(
        f"« {clean_name} » a été enregistré dans la bibliothèque.",
        "success",
    ), *refresh_library_outputs(request)


def accessible_library_rows(username):
    with db_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, owner, display_name, stored_path, item_type,
                   visibility, shared_with, description, created_at
            FROM library_items
            ORDER BY created_at DESC
            """
        ).fetchall()

    accessible = []
    for row in rows:
        shared_with = json.loads(row["shared_with"] or "[]")
        allowed = (
            row["owner"] == username
            or row["visibility"] == "all"
            or (row["visibility"] == "selected" and username in shared_with)
            or user_is_admin(username)
        )
        if allowed and Path(row["stored_path"]).exists():
            accessible.append(row)
    return accessible


def library_choice_label(row):
    ownership = "Moi" if row["owner"] else row["owner"]
    return f"{row['display_name']} · {row['item_type']} · propriétaire : {row['owner']}"


def render_library_html(username):
    rows = accessible_library_rows(username)
    if not rows:
        return "<div class='library-empty'>Aucune ressource accessible pour le moment.</div>"

    visibility_labels = {
        "private": "Privé",
        "all": "Tous les utilisateurs",
        "selected": "Utilisateurs choisis",
    }

    body = ""
    for row in rows:
        shared = ", ".join(json.loads(row["shared_with"] or "[]"))
        sharing = visibility_labels.get(row["visibility"], row["visibility"])
        if shared:
            sharing += f" : {shared}"

        safe_name = html.escape(str(row["display_name"]))
        safe_type = html.escape(str(row["item_type"]))
        safe_owner = html.escape(str(row["owner"]))
        safe_sharing = html.escape(sharing)
        safe_date = html.escape(str(row["created_at"]))
        safe_description = html.escape((row["description"] or "").strip())

        description_html = (
            f'<div class="library-description">{safe_description}</div>'
            if safe_description else ""
        )

        file_path = Path(row["stored_path"]).resolve()
        download_url = "/gradio_api/file=" + quote(str(file_path))
        delete_button = ""

        can_delete = row["owner"] == username or user_is_admin(username)

        if can_delete:
            delete_label = "Supprimer (admin)" if user_is_admin(username) and row["owner"] != username else "Supprimer"
            delete_title = (
                "Supprimer cette ressource en tant qu’administrateur"
                if user_is_admin(username) and row["owner"] != username
                else "Supprimer cette ressource"
            )
            delete_class = (
                "library-inline-delete library-inline-delete-admin"
                if user_is_admin(username) and row["owner"] != username
                else "library-inline-delete"
            )

            delete_button = f"""
            <button
                type="button"
                class="{delete_class}"
                data-library-id="{html.escape(str(row['id']))}"
                data-library-name="{safe_name}"
                title="{delete_title}"
            >
                {delete_label}
            </button>
            """

        body += f"""
        <tr>
            <td data-sort-value="{safe_name.lower()}">
                <strong>{safe_name}</strong>{description_html}
            </td>
            <td data-sort-value="{safe_type.lower()}">{safe_type}</td>
            <td data-sort-value="{safe_owner.lower()}">{safe_owner}</td>
            <td data-sort-value="{safe_sharing.lower()}">{safe_sharing}</td>
            <td data-sort-value="{safe_date}">{safe_date}</td>
            <td>
                <div class="library-row-actions">
                    <a class="library-download-link" href="{download_url}" download>
                        <span class="download-arrow">↓</span>Télécharger
                    </a>
                    {delete_button}
                </div>
            </td>
        </tr>
        """

    table_id = "library-sortable-table"
    return f"""
    <div class="library-sort-help">
        Cliquez sur <strong>Type</strong>, <strong>Propriétaire</strong>,
        <strong>Partage</strong> ou <strong>Ajout</strong> pour trier le tableau.
    </div>

    <div class="library-table-wrap">
        <table class="library-table library-sortable" id="{table_id}">
            <thead>
                <tr>
                    <th>
                        <button type="button" class="library-sort-button" data-column="0">
                            <span class="library-icon">◇</span>Nom <span class="sort-indicator">↕</span>
                        </button>
                    </th>
                    <th>
                        <button type="button" class="library-sort-button" data-column="1">
                            <span class="library-icon">▣</span>Type <span class="sort-indicator">↕</span>
                        </button>
                    </th>
                    <th>
                        <button type="button" class="library-sort-button" data-column="2">
                            <span class="library-icon">○</span>Propriétaire <span class="sort-indicator">↕</span>
                        </button>
                    </th>
                    <th>
                        <button type="button" class="library-sort-button" data-column="3">
                            <span class="library-icon">⌁</span>Partage <span class="sort-indicator">↕</span>
                        </button>
                    </th>
                    <th>
                        <button type="button" class="library-sort-button" data-column="4" data-type="date">
                            <span class="library-icon">◷</span>Ajout <span class="sort-indicator">↕</span>
                        </button>
                    </th>
                    <th>
                        <span class="table-head-label">
                            <span class="library-icon">↓</span>Actions
                        </span>
                    </th>
                </tr>
            </thead>
            <tbody>{body}</tbody>
        </table>
    </div>
    """



def refresh_library_tab(request: gr.Request):
    username = request_username(request)
    return render_library_html(username)


def owned_library_choices(request: gr.Request):
    username = request_username(request)
    rows = [
        row for row in accessible_library_rows(username)
        if row["owner"] == username
    ]
    choices = [
        (f"{row['display_name']} · {row['item_type']}", row["id"])
        for row in rows
    ]
    return gr.update(
        choices=choices,
        value=choices[0][1] if choices else None,
    )


def delete_owned_library_item(item_id, request: gr.Request):
    username = request_username(request)
    item_id = (item_id or "").strip()

    if not item_id:
        return (
            alert_html("Sélectionnez une ressource à supprimer.", "error"),
            render_library_html(username),
            "",
        )

    with db_connection() as conn:
        row = conn.execute(
            """
            SELECT id, owner, display_name, stored_path
            FROM library_items
            WHERE id = ?
            """,
            (item_id,),
        ).fetchone()

        if row is None:
            return (
                alert_html("Cette ressource n’existe plus.", "error"),
                render_library_html(username),
                "",
            )

        is_owner = row["owner"] == username
        is_admin = user_is_admin(username)

        if not is_owner and not is_admin:
            return (
                alert_html(
                    "Vous pouvez uniquement supprimer les fichiers que vous avez importés.",
                    "error",
                ),
                render_library_html(username),
                "",
            )

        stored_path = Path(row["stored_path"]).expanduser()

        conn.execute(
            "DELETE FROM library_items WHERE id = ?",
            (item_id,),
        )

        still_exists = conn.execute(
            "SELECT 1 FROM library_items WHERE id = ?",
            (item_id,),
        ).fetchone()

        if still_exists is not None:
            return (
                alert_html("La suppression en base de données a échoué.", "error"),
                render_library_html(username),
                "",
            )

    file_warning = ""
    try:
        if stored_path.exists():
            stored_path.unlink()
        if stored_path.exists():
            file_warning = " Le fichier physique n’a pas pu être supprimé du disque."
    except Exception as error:
        file_warning = f" Le fichier physique n’a pas pu être supprimé : {html.escape(str(error))}"

    deleted_as_admin = user_is_admin(username) and row["owner"] != username
    action_detail = (
        f" Le fichier appartenait à « {html.escape(row['owner'])} »."
        if deleted_as_admin
        else ""
    )

    return (
        alert_html(
            f"« {html.escape(row['display_name'])} » a bien été supprimé."
            f"{action_detail}{file_warning}",
            "success" if not file_warning else "info",
        ),
        render_library_html(username),
        "",
    )



def _normalize_result_path(value):
    """Récupère un vrai chemin local depuis une valeur Gradio."""
    if value is None:
        return None

    if isinstance(value, Path):
        candidate = value
    elif isinstance(value, str):
        candidate = Path(value)
    elif isinstance(value, dict):
        raw = value.get("path") or value.get("name") or value.get("orig_name")
        candidate = Path(raw) if raw else None
    elif hasattr(value, "name"):
        candidate = Path(value.name)
    else:
        candidate = Path(str(value))

    if candidate is None:
        return None

    try:
        candidate = candidate.expanduser().resolve()
    except Exception:
        return None

    return candidate if candidate.exists() and candidate.is_file() else None


def _copy_result_file(source_value, destination_dir, filename):
    source = _normalize_result_path(source_value)
    if source is None:
        return None

    destination_dir.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    destination = destination_dir / f"{filename}{suffix}"
    shutil.copy2(source, destination)

    if not destination.exists() or destination.stat().st_size <= 0:
        return None

    return str(destination.resolve())


def saved_results_dropdown_update(username, selected_id=None):
    with db_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, title, created_at
            FROM result_archives_v2
            WHERE owner = ?
            ORDER BY created_at DESC
            """,
            (username,),
        ).fetchall()

    choices = [
        (f"{row['title']} · {row['created_at']}", row["id"])
        for row in rows
    ]

    value = selected_id
    if value is None and choices:
        value = choices[0][1]

    return gr.update(choices=choices, value=value)


def load_saved_result(archive_id, request: gr.Request):
    username = request_username(request)

    if not archive_id:
        return (
            gr.update(visible=False),
            "",
            None,
            None,
            None,
            None,
            None,
        )

    with db_connection() as conn:
        row = conn.execute(
            """
            SELECT title, files_json, created_at
            FROM result_archives_v2
            WHERE id = ? AND owner = ?
            """,
            (archive_id, username),
        ).fetchone()

    if row is None:
        return (
            gr.update(visible=False),
            "",
            None,
            None,
            None,
            None,
            None,
        )

    files = json.loads(row["files_json"] or "{}")

    def existing(key):
        raw = files.get(key)
        if not raw:
            return None
        path = Path(raw)
        return str(path.resolve()) if path.exists() and path.is_file() else None

    summary = f"""
    <div class="saved-result-summary">
        <div class="saved-result-summary-kicker">Analyse enregistrée</div>
        <div class="saved-result-summary-title">{html.escape(row['title'])}</div>
        <div class="saved-result-summary-date">Ajoutée le {html.escape(row['created_at'])}</div>
    </div>
    """

    return (
        gr.update(visible=True),
        summary,
        existing("report"),
        existing("video"),
        existing("detections"),
        existing("stats"),
        existing("commercial"),
    )



def delete_saved_result(archive_id, request: gr.Request):
    username = request_username(request)
    archive_id = (archive_id or "").strip()

    if not archive_id:
        return (
            alert_html("Sélectionnez des résultats à supprimer.", "error"),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    with db_connection() as conn:
        row = conn.execute(
            """
            SELECT id, title
            FROM result_archives_v2
            WHERE id = ? AND owner = ?
            """,
            (archive_id, username),
        ).fetchone()

        if row is None:
            return (
                alert_html("Ces résultats n’existent plus.", "error"),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
            )

        conn.execute(
            "DELETE FROM result_archives_v2 WHERE id = ? AND owner = ?",
            (archive_id, username),
        )

    shutil.rmtree(
        RESULT_ARCHIVES_V2_DIR / username / archive_id,
        ignore_errors=True,
    )

    with db_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, title, created_at
            FROM result_archives_v2
            WHERE owner = ?
            ORDER BY created_at DESC
            """,
            (username,),
        ).fetchall()

    status = alert_html(
        f"Les résultats « {html.escape(row['title'])} » ont été supprimés.",
        "success",
    )

    if not rows:
        return (
            status,
            gr.update(choices=[], value=None, visible=False),
            gr.update(visible=False),
            "",
            None,
            None,
            None,
            None,
            None,
        )

    choices = [
        (f"{item['title']} · {item['created_at']}", item["id"])
        for item in rows
    ]
    next_id = rows[0]["id"]
    details = load_saved_result(next_id, request)

    return (
        status,
        gr.update(choices=choices, value=next_id, visible=True),
        *details,
    )



def refresh_saved_results(request: gr.Request):
    username = request_username(request)

    with db_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, title, created_at
            FROM result_archives_v2
            WHERE owner = ?
            ORDER BY created_at DESC
            """,
            (username,),
        ).fetchall()

    if not rows:
        return (
            gr.update(choices=[], value=None, visible=False),
            gr.update(visible=False),
            "",
            None,
            None,
            None,
            None,
            None,
        )

    choices = [
        (f"{row['title']} · {row['created_at']}", row["id"])
        for row in rows
    ]
    archive_id = rows[0]["id"]
    details = load_saved_result(archive_id, request)

    return (
        gr.update(choices=choices, value=archive_id, visible=True),
        *details,
    )



def save_current_results(
    title,
    report_path,
    video_path,
    detections_path,
    stats_path,
    commercial_path,
    request: gr.Request,
):
    username = request_username(request)
    clean_title = (title or "").strip()

    if not clean_title:
        return (
            alert_html("Donnez un nom à ces résultats.", "error"),
            saved_results_dropdown_update(username),
            gr.update(),
            gr.update(visible=True),
        )

    with db_connection() as conn:
        duplicate = conn.execute(
            """
            SELECT id
            FROM result_archives_v2
            WHERE owner = ?
              AND LOWER(TRIM(title)) = LOWER(TRIM(?))
            LIMIT 1
            """,
            (username, clean_title),
        ).fetchone()

    if duplicate is not None:
        return (
            alert_html(
                f"Vous avez déjà enregistré des résultats sous le nom « {clean_title} ».",
                "error",
            ),
            saved_results_dropdown_update(username),
            gr.update(),
            gr.update(visible=True),
        )

    sources = {
        "report": report_path,
        "video": video_path,
        "detections": detections_path,
        "stats": stats_path,
        "commercial": commercial_path,
    }

    normalized = {
        key: _normalize_result_path(value)
        for key, value in sources.items()
    }

    # On exige les fichiers essentiels issus d'une analyse terminée.
    missing = [
        key for key in ("report", "video", "detections", "stats", "commercial")
        if normalized.get(key) is None
    ]

    if missing:
        labels = {
            "report": "rapport HTML",
            "video": "vidéo annotée",
            "detections": "CSV des détections",
            "stats": "CSV des statistiques",
            "commercial": "tableau commercial",
        }
        readable = ", ".join(labels[key] for key in missing)
        return (
            alert_html(
                "Impossible d’enregistrer : fichiers manquants — " + readable + ".",
                "error",
            ),
            saved_results_dropdown_update(username),
            gr.update(),
            gr.update(visible=True),
        )

    archive_id = uuid.uuid4().hex
    archive_dir = RESULT_ARCHIVES_V2_DIR / username / archive_id

    try:
        archive_dir.mkdir(parents=True, exist_ok=False)

        files = {
            "report": _copy_result_file(normalized["report"], archive_dir, "rapport_visibilite"),
            "video": _copy_result_file(normalized["video"], archive_dir, "video_annotee"),
            "detections": _copy_result_file(normalized["detections"], archive_dir, "detections_completes"),
            "stats": _copy_result_file(normalized["stats"], archive_dir, "statistiques_logos"),
            "commercial": _copy_result_file(normalized["commercial"], archive_dir, "tableau_commercial"),
        }

        if any(value is None for value in files.values()):
            raise RuntimeError("La copie d’un ou plusieurs fichiers a échoué.")

        with db_connection() as conn:
            conn.execute(
                """
                INSERT INTO result_archives_v2(id, owner, title, files_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    archive_id,
                    username,
                    clean_title,
                    json.dumps(files, ensure_ascii=False),
                ),
            )

    except Exception as error:
        shutil.rmtree(archive_dir, ignore_errors=True)
        return (
            alert_html(
                f"Échec de l’enregistrement : {html.escape(str(error))}",
                "error",
            ),
            saved_results_dropdown_update(username),
            gr.update(),
            gr.update(visible=True),
        )

    return (
        alert_html(
            f"Les résultats « {clean_title} » ont bien été enregistrés.",
            "success",
        ),
        saved_results_dropdown_update(username, archive_id),
        "",
        gr.update(visible=False),
    )



def refresh_library_outputs(request: gr.Request):
    username = request_username(request)
    return (
        render_library_html(username),
        render_users_html(username),
    )


def refresh_library(request: gr.Request):
    return refresh_library_outputs(request)


def get_library_item(item_id, username):
    if not item_id:
        raise gr.Error("Sélectionnez un fichier dans la bibliothèque.")

    rows = accessible_library_rows(username)
    for row in rows:
        if row["id"] == item_id:
            return row
    raise gr.Error("Ce fichier n’est pas accessible avec votre compte.")



def refresh_training_library(request: gr.Request):
    username = request_username(request)
    rows = [
        row for row in accessible_library_rows(username)
        if row["item_type"] == "Dataset ZIP"
    ]
    choices = [(library_choice_label(row), row["id"]) for row in rows]
    return gr.update(
        choices=choices,
        value=choices[0][1] if choices else None,
    )


def choose_training_zip_from_library(item_id, request: gr.Request):
    username = request_username(request)
    row = get_library_item(item_id, username)
    if row["item_type"] != "Dataset ZIP":
        raise gr.Error("La ressource sélectionnée n’est pas un dataset ZIP.")
    return (
        str(row["stored_path"]),
        alert_html(
            f"Dataset « {row['display_name']} » sélectionné depuis la bibliothèque.",
            "success",
        ),
        gr.update(visible=True),
        gr.update(visible=False),
    )


def use_library_zip(item_id, request: gr.Request):
    username = request_username(request)
    row = get_library_item(item_id, username)
    if row["item_type"] != "Dataset ZIP":
        raise gr.Error("Le fichier sélectionné n’est pas un dataset ZIP.")
    return (
        str(row["stored_path"]),
        alert_html(f"Dataset « {row['display_name']} » chargé depuis la bibliothèque.", "success"),
        gr.update(visible=True),
        gr.update(visible=False),
    )


def use_library_model(item_id, request: gr.Request):
    username = request_username(request)
    row = get_library_item(item_id, username)
    if row["item_type"] != "Modèle YOLO":
        raise gr.Error("Le fichier sélectionné n’est pas un modèle YOLO.")
    return (
        str(row["stored_path"]),
        alert_html(f"Modèle « {row['display_name']} » chargé depuis la bibliothèque.", "success"),
        gr.update(visible=True),
        gr.update(visible=False),
    )



def refresh_analysis_libraries(request: gr.Request):
    username = request_username(request)
    rows = accessible_library_rows(username)

    model_rows = [row for row in rows if row["item_type"] == "Modèle YOLO"]
    video_rows = [row for row in rows if row["item_type"] == "Vidéo de match"]

    model_choices = [(library_choice_label(row), row["id"]) for row in model_rows]
    video_choices = [(library_choice_label(row), row["id"]) for row in video_rows]

    return (
        gr.update(
            choices=model_choices,
            value=model_choices[0][1] if model_choices else None,
        ),
        gr.update(
            choices=video_choices,
            value=video_choices[0][1] if video_choices else None,
        ),
    )


def validate_analysis_model(source_mode, uploaded_file, library_item_id, request: gr.Request):
    if source_mode == "Choisir dans la bibliothèque":
        username = request_username(request)
        row = get_library_item(library_item_id, username)
        if row["item_type"] != "Modèle YOLO":
            return (
                alert_html("La ressource sélectionnée n’est pas un modèle YOLO.", "error"),
                None,
                gr.update(visible=False),
                gr.update(visible=True),
            )
        return (
            alert_html(
                f"Modèle « {row['display_name']} » sélectionné depuis la bibliothèque.",
                "success",
            ),
            str(row["stored_path"]),
            gr.update(visible=True),
            gr.update(visible=False),
        )

    if uploaded_file is None:
        return (
            alert_html("Ajoutez un modèle YOLO au format .pt.", "error"),
            None,
            gr.update(visible=False),
            gr.update(visible=True),
        )

    return (
        alert_html("Modèle validé. Vous pouvez maintenant choisir la vidéo.", "success"),
        uploaded_file,
        gr.update(visible=True),
        gr.update(visible=False),
    )


def validate_analysis_video(source_mode, uploaded_file, library_item_id, request: gr.Request):
    if source_mode == "Choisir dans la bibliothèque":
        username = request_username(request)
        row = get_library_item(library_item_id, username)
        if row["item_type"] != "Vidéo de match":
            return (
                alert_html("La ressource sélectionnée n’est pas une vidéo de match.", "error"),
                None,
                gr.update(visible=False),
                gr.update(visible=True),
            )
        return (
            alert_html(
                f"Vidéo « {row['display_name']} » sélectionnée depuis la bibliothèque.",
                "success",
            ),
            str(row["stored_path"]),
            gr.update(visible=True),
            gr.update(visible=False),
        )

    if uploaded_file is None:
        return (
            alert_html("Ajoutez une vidéo MP4, MOV, AVI ou MKV.", "error"),
            None,
            gr.update(visible=False),
            gr.update(visible=True),
        )

    return (
        alert_html("Vidéo validée. Les paramètres d’analyse sont disponibles.", "success"),
        uploaded_file,
        gr.update(visible=True),
        gr.update(visible=False),
    )


def download_library_item(item_id, request: gr.Request):
    username = request_username(request)
    row = get_library_item(item_id, username)
    return str(row["stored_path"])


def delete_library_item(item_id, request: gr.Request):
    username = request_username(request)
    row = get_library_item(item_id, username)

    if row["owner"] != username and not user_is_admin(username):
        return alert_html("Seul le propriétaire ou l’administrateur peut supprimer ce fichier.", "error"), *refresh_library_outputs(request)

    stored_path = Path(row["stored_path"])
    with db_connection() as conn:
        conn.execute("DELETE FROM library_items WHERE id = ?", (item_id,))
    stored_path.unlink(missing_ok=True)

    return alert_html("Le fichier a été supprimé de la bibliothèque.", "success"), *refresh_library_outputs(request)



def _read_user_commercial_archives(username):
    frames = []
    with db_connection() as conn:
        rows = conn.execute(
            "SELECT id, title, files_json, created_at FROM result_archives_v2 WHERE owner = ? ORDER BY created_at",
            (username,),
        ).fetchall()
    for row in rows:
        try:
            files = json.loads(row["files_json"] or "{}")
            path = Path(files.get("commercial", ""))
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            if frame.empty or "Logo" not in frame.columns:
                continue
            frame["Analyse"] = row["title"]
            frame["Date analyse"] = pd.to_datetime(row["created_at"], errors="coerce")
            frame["Archive ID"] = row["id"]
            frames.append(frame)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _consolidation_match_choices(data):
    if data is None or data.empty:
        return []
    archives = (
        data[["Archive ID", "Analyse", "Date analyse"]]
        .drop_duplicates(subset=["Archive ID"])
        .sort_values("Date analyse", ascending=False)
    )
    choices = []
    for row in archives.itertuples(index=False):
        archive_id = str(getattr(row, "_0", row[0]))
        title = str(getattr(row, "Analyse", row[1]))
        date_value = getattr(row, "_2", row[2])
        date_label = pd.to_datetime(date_value, errors="coerce")
        date_text = date_label.strftime("%d/%m/%Y") if not pd.isna(date_label) else "date inconnue"
        choices.append((f"{title} · {date_text}", archive_id))
    return choices


def refresh_consolidation_filters(request: gr.Request):
    username = request_username(request)
    data = _read_user_commercial_archives(username)
    if data.empty:
        return (
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=[]),
            gr.update(choices=[], value=[]),
            alert_html("Aucune analyse enregistrée à consolider.", "info"),
        )
    years = sorted(data["Date analyse"].dropna().dt.year.astype(int).unique().tolist(), reverse=True)
    brands = sorted(data["Logo"].astype(str).dropna().unique().tolist())
    default_year = years[0] if years else None
    filtered = data[data["Date analyse"].dt.year == default_year] if default_year else data
    match_choices = _consolidation_match_choices(filtered)
    match_values = [value for _, value in match_choices]
    return (
        gr.update(choices=years, value=default_year),
        gr.update(choices=brands, value=brands),
        gr.update(choices=match_choices, value=match_values),
        "",
    )


def update_consolidation_matches(year, selected_brands, request: gr.Request):
    username = request_username(request)
    data = _read_user_commercial_archives(username)
    if data.empty:
        return gr.update(choices=[], value=[])
    if year:
        data = data[data["Date analyse"].dt.year == int(year)]
    if selected_brands:
        data = data[data["Logo"].astype(str).isin([str(x) for x in selected_brands])]
    choices = _consolidation_match_choices(data)
    return gr.update(choices=choices, value=[value for _, value in choices])


def _save_consolidated_charts(data, consolidated, run_dir):
    charts_dir = Path(run_dir) / "graphiques"
    charts_dir.mkdir(parents=True, exist_ok=True)
    order = consolidated.sort_values("Temps visible (s)", ascending=False)["Logo"].astype(str).tolist()
    colors = {brand: BRAND_COLOR_PALETTE[i % len(BRAND_COLOR_PALETTE)] for i, brand in enumerate(order)}
    paths = {}

    plot_data = consolidated.sort_values("Temps visible (s)", ascending=True).tail(14)
    fig, ax = figure_base((9.2, 5.1))
    ax.barh(plot_data["Logo"], plot_data["Temps visible (s)"], color=[colors.get(str(x), BLUE) for x in plot_data["Logo"]], height=.46)
    ax.set_xlabel("Temps visible cumulé (secondes)", fontsize=9, color=MUTED)
    for i, value in enumerate(plot_data["Temps visible (s)"].astype(float)):
        ax.text(value + max(float(plot_data["Temps visible (s)"].max()), 1)*.015, i, f"{value:.1f} s", va="center", fontsize=8)
    fig.tight_layout(); paths["Classement cumulé"] = charts_dir / "classement_cumule.png"
    fig.savefig(paths["Classement cumulé"], bbox_inches="tight", facecolor="white"); plt.close(fig)

    sov_data = consolidated.sort_values("Part de voix (%)", ascending=False).head(8)
    fig, ax = plt.subplots(figsize=(7.6, 5.1), dpi=170); fig.patch.set_facecolor("white")
    vals=sov_data["Part de voix (%)"].astype(float); labels=sov_data["Logo"].astype(str)
    _, _, autotexts=ax.pie(vals, labels=labels, autopct=lambda x:f"{x:.1f}%" if x>=4 else "", startangle=90, pctdistance=.76,
        colors=[colors.get(x, BLUE) for x in labels], wedgeprops={"width":.42,"edgecolor":"white","linewidth":2}, textprops={"fontsize":8,"color":TEXT})
    for t in autotexts: t.set_color("white"); t.set_fontweight("bold")
    ax.text(0,0.03,"Part de voix",ha="center",va="center",fontsize=11,fontweight="bold",color=TEXT)
    ax.text(0,-.12,"moyenne",ha="center",va="center",fontsize=8,color=MUTED)
    ax.axis("equal"); fig.tight_layout(); paths["Part de voix moyenne"] = charts_dir / "part_de_voix_moyenne.png"
    fig.savefig(paths["Part de voix moyenne"], bbox_inches="tight", facecolor="white"); plt.close(fig)

    # Évolution par fichier / match, et non par date.
    trend = data.groupby(["Analyse", "Logo"], as_index=False)["Temps visible (s)"].sum()
    match_order = (
        data[["Archive ID", "Analyse", "Date analyse"]]
        .drop_duplicates(subset=["Archive ID"])
        .sort_values("Date analyse", na_position="last")["Analyse"]
        .astype(str)
        .tolist()
    )
    match_order = list(dict.fromkeys(match_order))
    x_positions = np.arange(len(match_order))
    fig, ax = figure_base((9.6, 5.5))
    for brand in order[:10]:
        sub = trend[trend["Logo"].astype(str) == brand].set_index("Analyse")
        values = [float(sub.loc[name, "Temps visible (s)"]) if name in sub.index else 0.0 for name in match_order]
        ax.plot(x_positions, values, marker="o", linewidth=1.8, label=brand, color=colors.get(brand, BLUE))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(match_order, rotation=25, ha="right")
    ax.set_xlabel("Nom du fichier / match", fontsize=9, color=MUTED)
    ax.set_ylabel("Temps visible (s)", fontsize=9, color=MUTED)
    ax.grid(axis="both", color="#eef2f7", linewidth=.8)
    ax.legend(frameon=False, fontsize=8, ncol=min(4, max(1, len(order[:10]))), loc="lower center", bbox_to_anchor=(0.5, 1.02), borderaxespad=0)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    paths["Évolution par match"] = charts_dir / "evolution_par_match.png"
    fig.savefig(paths["Évolution par match"], bbox_inches="tight", facecolor="white"); plt.close(fig)

    match_data = data.groupby(["Analyse","Logo"], as_index=False)["Temps visible (s)"].sum()
    pivot = match_data.pivot(index="Analyse", columns="Logo", values="Temps visible (s)").fillna(0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index[:12]]
    fig, ax = plt.subplots(figsize=(10, max(4.8, .48*len(pivot)+1.8)), dpi=170); fig.patch.set_facecolor("white")
    left=np.zeros(len(pivot))
    for brand in [b for b in order if b in pivot.columns]:
        values=pivot[brand].to_numpy(float)
        ax.barh(pivot.index, values, left=left, label=brand, color=colors.get(brand, BLUE), height=.5); left += values
    ax.invert_yaxis(); ax.set_xlabel("Temps visible (s)", fontsize=9, color=MUTED)
    ax.spines[["top","right"]].set_visible(False); ax.grid(axis="x", color="#eef2f7", linewidth=.8)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(.5,-.25))
    fig.tight_layout(); paths["Comparaison des matchs"] = charts_dir / "comparaison_matchs.png"
    fig.savefig(paths["Comparaison des matchs"], bbox_inches="tight", facecolor="white"); plt.close(fig)

    return {k:str(v) for k,v in paths.items()}


def _make_consolidated_html_report(data, consolidated, charts, run_dir, year):
    def image_uri(path):
        p = Path(path)
        return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii") if p.exists() else ""

    def esc(value):
        return html.escape(str(value))

    top = consolidated.iloc[0]
    match_count = int(data["Archive ID"].astype(str).nunique())
    brand_count = int(data["Logo"].astype(str).nunique())
    total_time = float(consolidated["Temps visible (s)"].sum())
    selected_files = data[["Archive ID", "Analyse"]].drop_duplicates(subset=["Archive ID"])["Analyse"].astype(str).tolist()

    second = consolidated.iloc[1] if len(consolidated) > 1 else None
    top_name = str(top["Logo"])
    top_time = float(top.get("Temps visible (s)", 0))
    top_sov = float(top.get("Part de voix (%)", 0))
    top_sequences = float(top.get("Nb de séquences", 0))
    avg_per_match = top_time / max(match_count, 1)
    lead_text = ""
    if second is not None:
        gap = top_time - float(second.get("Temps visible (s)", 0))
        lead_text = f" Elle devance {esc(second['Logo'])} de {gap:.1f} secondes sur la sélection."

    analysis_html = f"""
    <p><strong>{esc(top_name)}</strong> domine la sélection avec <strong>{top_time:.1f} secondes</strong> de visibilité cumulée,
    soit une moyenne de <strong>{avg_per_match:.1f} secondes par match</strong>.{lead_text}</p>
    <p>La marque totalise <strong>{top_sequences:.0f} séquences</strong> et une part de voix moyenne de
    <strong>{top_sov:.1f} %</strong>. Une forte durée cumulée indique une présence répétée ; elle doit toutefois être lue avec
    l’occupation de l’écran, la centralité et la netteté pour juger la qualité réelle de l’exposition.</p>
    <p>Les écarts entre les fichiers permettent d’identifier les matchs les plus performants et les contextes où la visibilité
    est la plus régulière. Une exposition équilibrée entre plusieurs matchs est généralement plus robuste qu’un résultat reposant
    sur un seul pic exceptionnel.</p>
    """

    cols = [c for c in ["Logo", "Nombre de matchs", "Temps visible (s)", "% vidéo", "Nb de séquences", "Occupation moy. (%)", "Occupation max (%)", "Centralité (%)", "Part de voix (%)", "Netteté moy."] if c in consolidated.columns]
    head = "".join(f"<th>{esc(c)}</th>" for c in cols)
    body = ""
    for _, row in consolidated[cols].iterrows():
        body += "<tr>" + "".join(f"<td>{esc(row[c] if c == 'Logo' else round(float(row[c]), 2))}</td>" for c in cols) + "</tr>"

    chart_cards = "".join(
        '<article class="chart-card" onclick="openZoom(this)"><div class="chart-top"><h3>' + esc(title) + '</h3><span>Cliquer pour agrandir</span></div><img src="' + image_uri(path) + '" alt="' + esc(title) + '"></article>'
        for title, path in charts.items()
    )
    files_html = "".join(f"<li>{esc(name)}</li>" for name in selected_files)
    definitions = [
        ("Temps visible", "Durée cumulée pendant laquelle au moins une détection de la marque est présente."),
        ("% vidéo", "Part de la durée de la vidéo pendant laquelle la marque est visible."),
        ("Nombre de séquences", "Nombre d’apparitions distinctes, séparées par une interruption suffisante."),
        ("Occupation moyenne / maximale", "Part de l’écran occupée par les zones de détection, en moyenne ou au meilleur pic."),
        ("Centralité", "Proportion des apparitions situées dans la zone centrale de l’image."),
        ("Part de voix visuelle", "Poids visuel relatif de la marque par rapport à l’ensemble des marques détectées."),
        ("Netteté moyenne", "Confiance moyenne des détections YOLO ; ce n’est pas la précision globale du modèle."),
    ]
    definitions_html = "".join(f"<div class='definition'><strong>{esc(title)}</strong><p>{esc(text)}</p></div>" for title, text in definitions)
    report_path = Path(run_dir) / f"rapport_consolide_{year or 'toutes_annees'}.html"
    html_doc = f"""<!doctype html><html lang='fr'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Rapport consolidé TSM</title>
<style>:root{{--blue:#46619c;--dark:#172033;--muted:#64748b;--bg:#eef3f8;--line:#dfe7f2}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--dark);font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}}.report{{max-width:1220px;margin:auto;padding:30px}}.actions{{position:sticky;top:14px;z-index:10;display:flex;justify-content:flex-end}}button{{border:0;border-radius:999px;padding:12px 20px;background:var(--blue);color:white;font-weight:800;cursor:pointer;box-shadow:0 14px 34px #46619c40}}.page{{background:#f9fbfe;border:1px solid var(--line);border-radius:32px;padding:44px;margin:18px 0 28px;min-height:620px;box-shadow:0 20px 55px #46619c12}}.cover{{display:flex;min-height:700px;flex-direction:column;justify-content:space-between;background:linear-gradient(145deg,#172033,#46619c);color:white}}h1{{font-family:Georgia,serif;font-size:58px;line-height:1.02;max-width:780px;margin:0 0 22px}}h2{{font-family:Georgia,serif;font-size:34px;margin:6px 0}}h3{{margin:0;font-size:17px}}.subtitle{{font-size:18px;line-height:1.6;max-width:820px;color:#dce7ff}}.header{{display:flex;justify-content:space-between;align-items:flex-start;border-bottom:1px solid var(--line);padding-bottom:20px;margin-bottom:28px}}.header small{{color:var(--blue);font-weight:900;letter-spacing:.12em}}.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}}.kpi{{background:white;color:var(--dark);border:1px solid var(--line);border-radius:20px;padding:20px}}.cover .kpi{{background:#ffffff14;color:white;border-color:#ffffff2b}}.kpi small{{display:block;color:var(--muted);font-weight:800;text-transform:uppercase;font-size:11px;letter-spacing:.08em}}.cover .kpi small{{color:#dce7ff}}.kpi strong{{display:block;font-family:Georgia,serif;font-size:25px;margin-top:10px}}.highlight{{background:white;border-left:5px solid var(--blue);border-radius:20px;padding:24px;font-size:15px;line-height:1.7}}.charts{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}}.chart-card{{background:white;border:1px solid var(--line);border-radius:24px;padding:20px;min-height:360px;cursor:zoom-in;transition:.2s ease}}.chart-card:hover{{transform:translateY(-3px);box-shadow:0 18px 40px #46619c20}}.chart-top{{display:flex;justify-content:space-between;align-items:center;gap:10px}}.chart-top span{{font-size:11px;color:var(--muted)}}.chart-card img{{width:100%;height:300px;object-fit:contain;margin-top:12px}}.definitions{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}.definition{{background:white;border:1px solid var(--line);border-radius:18px;padding:18px}}.definition strong{{color:var(--blue)}}.definition p{{margin:8px 0 0;color:var(--muted);line-height:1.55;font-size:13px}}.file-list{{columns:2;margin:0;padding-left:20px;color:var(--muted)}}table{{width:100%;border-collapse:separate;border-spacing:0;background:white;border:1px solid var(--line);border-radius:18px;overflow:hidden}}th,td{{padding:12px;border-bottom:1px solid #edf2f8;text-align:left;font-size:12px}}th{{background:#f1f5fa;position:sticky;top:0}}td:first-child{{font-weight:800;color:var(--blue)}}.table-wrap{{overflow:auto;max-height:470px;border-radius:18px}}.zoom{{display:none;position:fixed;inset:0;background:#0b1020e8;z-index:100;align-items:center;justify-content:center;padding:32px;cursor:zoom-out}}.zoom.open{{display:flex}}.zoom img{{max-width:96vw;max-height:90vh;background:white;border-radius:18px;padding:12px;box-shadow:0 25px 80px #0008}}.zoom-close{{position:absolute;right:24px;top:18px;font-size:34px;color:white}}@media print{{body{{background:white}}.actions,.chart-top span{{display:none}}.report{{padding:0;max-width:none}}.page{{width:297mm;min-height:210mm;margin:0;border:0;border-radius:0;box-shadow:none;page-break-after:always}}.chart-card{{break-inside:avoid}}@page{{size:A4 landscape;margin:0}}}}@media(max-width:900px){{.kpis,.charts,.definitions{{grid-template-columns:1fr}}h1{{font-size:42px}}.page{{padding:24px}}.file-list{{columns:1}}}}</style></head><body><div class='report'><div class='actions'><button onclick='window.print()'>Exporter en PDF</button></div>
<section class='page cover'><div><small>TSM · SPONSORING INTELLIGENCE</small><h1>Rapport consolidé de visibilité</h1><p class='subtitle'>Bilan détaillé de la visibilité des marques sur une sélection précise de fichiers et de matchs.</p></div><div class='kpis'><div class='kpi'><small>Période</small><strong>{esc(year or 'Toutes années')}</strong></div><div class='kpi'><small>Fichiers inclus</small><strong>{match_count}</strong></div><div class='kpi'><small>Marques</small><strong>{brand_count}</strong></div><div class='kpi'><small>Temps cumulé</small><strong>{total_time:.1f} s</strong></div></div></section>
<section class='page'><div class='header'><div><small>01 · SYNTHÈSE</small><h2>Résultats clés</h2></div><strong>TSM</strong></div><div class='kpis'><div class='kpi'><small>Marque dominante</small><strong>{esc(top_name)}</strong></div><div class='kpi'><small>Temps visible</small><strong>{top_time:.1f} s</strong></div><div class='kpi'><small>Séquences</small><strong>{top_sequences:.0f}</strong></div><div class='kpi'><small>Part de voix moy.</small><strong>{top_sov:.1f}%</strong></div></div><div class='highlight' style='margin-top:28px'><h3>Analyse intelligente des résultats</h3>{analysis_html}</div></section>
<section class='page'><div class='header'><div><small>02 · DASHBOARD</small><h2>Graphiques interactifs</h2></div><strong>Cliquer pour zoomer</strong></div><div class='charts'>{chart_cards}</div></section>
<section class='page'><div class='header'><div><small>03 · DONNÉES</small><h2>Tableau consolidé</h2></div><strong>{brand_count} marques</strong></div><div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></section>
<section class='page'><div class='header'><div><small>04 · MÉTHODOLOGIE</small><h2>Définition des indicateurs</h2></div><strong>Guide de lecture</strong></div><div class='definitions'>{definitions_html}</div></section>
<section class='page'><div class='header'><div><small>05 · PÉRIMÈTRE</small><h2>Fichiers inclus dans l’analyse</h2></div><strong>{match_count}</strong></div><ul class='file-list'>{files_html}</ul></section>
</div><div class='zoom' id='zoom' onclick='closeZoom()'><span class='zoom-close'>×</span><img id='zoom-img' alt='Graphique agrandi'></div><script>function openZoom(card){{const img=card.querySelector('img');document.getElementById('zoom-img').src=img.src;document.getElementById('zoom').classList.add('open');}}function closeZoom(){{document.getElementById('zoom').classList.remove('open');}}document.addEventListener('keydown',e=>{{if(e.key==='Escape')closeZoom();}});</script></body></html>"""
    report_path.write_text(html_doc, encoding="utf-8")
    return str(report_path)


def generate_consolidated_report(year, selected_brands, selected_matches, request: gr.Request):
    username=request_username(request)
    data=_read_user_commercial_archives(username)
    hidden=gr.update(visible=False)
    if data.empty:
        return alert_html("Aucune analyse enregistrée à consolider.","error"),"",None,None,None,None,None,None,hidden
    if year: data=data[data["Date analyse"].dt.year==int(year)]
    if selected_brands: data=data[data["Logo"].astype(str).isin([str(x) for x in selected_brands])]
    if not selected_matches:
        return alert_html("Sélectionne au moins un match.","error"),"",None,None,None,None,None,None,hidden
    data=data[data["Archive ID"].astype(str).isin({str(x) for x in selected_matches})]
    if data.empty:
        return alert_html("Aucune donnée ne correspond à cette sélection.","error"),"",None,None,None,None,None,None,hidden
    numeric_cols=["Temps visible (s)","% vidéo","Nb de séquences","Durée moy. séquence (s)","Durée max. séquence (s)","Occupation moy. (%)","Occupation max (%)","Centralité (%)","Part de voix (%)","Netteté moy."]
    for col in numeric_cols:
        if col in data.columns: data[col]=pd.to_numeric(data[col],errors="coerce").fillna(0)
    agg={"Temps visible (s)":"sum","% vidéo":"mean","Nb de séquences":"sum","Durée moy. séquence (s)":"mean","Durée max. séquence (s)":"max","Occupation moy. (%)":"mean","Occupation max (%)":"max","Centralité (%)":"mean","Part de voix (%)":"mean","Netteté moy.":"mean","Analyse":"nunique"}
    agg={k:v for k,v in agg.items() if k in data.columns}
    consolidated=data.groupby("Logo",as_index=False).agg(agg).rename(columns={"Analyse":"Nombre de matchs"}).sort_values("Temps visible (s)",ascending=False)
    run_dir=RESULTS_DIR/("consolidation_"+time_id()); run_dir.mkdir(parents=True,exist_ok=True)
    csv_path=run_dir/f"statistiques_consolidees_{year or 'toutes_annees'}.csv"; consolidated.to_csv(csv_path,index=False)
    charts=_save_consolidated_charts(data,consolidated,run_dir)
    report_path=_make_consolidated_html_report(data,consolidated,charts,run_dir,year)
    table_html=make_table_html(consolidated.rename(columns={"Nombre de matchs":"Nb de matchs"}))
    return alert_html(f"Consolidation terminée : {data['Archive ID'].astype(str).nunique()} match(s).","success"),table_html,charts.get("Classement cumulé"),charts.get("Part de voix moyenne"),charts.get("Évolution par match"),charts.get("Comparaison des matchs"),str(csv_path),str(report_path),gr.update(visible=True)

def _active_usernames(exclude=None):
    with db_connection() as conn:
        rows = conn.execute(
            "SELECT username FROM users WHERE is_active = 1 ORDER BY username COLLATE NOCASE"
        ).fetchall()
    return [row["username"] for row in rows if row["username"] != exclude]


def _logo_rows_for_user(username):
    # Récupère automatiquement les anciens logos déjà présents dans le dossier.
    with db_connection() as conn:
        known = {row["stored_filename"] for row in conn.execute("SELECT stored_filename FROM brand_logos").fetchall()}
        for extension in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            for path in BRAND_LOGOS_DIR.glob(extension):
                if path.name in known:
                    continue
                conn.execute(
                    "INSERT OR IGNORE INTO brand_logos(id, owner, brand_name, stored_filename, visibility, shared_with) VALUES (?, ?, ?, ?, 'private', '[]')",
                    (uuid.uuid4().hex, username, path.stem.replace("_", " ").title(), path.name),
                )
        rows = conn.execute(
            """
            SELECT id, owner, brand_name, stored_filename, visibility, shared_with, created_at
            FROM brand_logos
            ORDER BY brand_name COLLATE NOCASE
            """
        ).fetchall()
    visible = []
    for row in rows:
        try:
            shared = json.loads(row["shared_with"] or "[]")
        except Exception:
            shared = []
        if row["owner"] == username or (row["visibility"] == "selected" and username in shared):
            visible.append(dict(row))
    return visible


def list_brand_logos_html(request: gr.Request):
    username = request_username(request)
    rows = _logo_rows_for_user(username)
    if not rows:
        return "<div class='library-empty'>Aucun logo accessible pour le moment.</div>"

    html_rows = []
    for row in rows:
        path = BRAND_LOGOS_DIR / Path(row["stored_filename"]).name
        if not path.exists():
            continue
        brand = str(row["brand_name"])
        mime = "image/webp" if path.suffix.lower() == ".webp" else ("image/png" if path.suffix.lower() == ".png" else "image/jpeg")
        uri = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
        try:
            shared = json.loads(row["shared_with"] or "[]")
        except Exception:
            shared = []
        sharing = "Privé" if row["visibility"] == "private" else "Utilisateurs choisis : " + (", ".join(shared) if shared else "—")
        created = row["created_at"] or ""
        can_delete = row["owner"] == username or user_is_admin(username)
        action = (
            f"<button type='button' class='library-inline-delete logo-inline-delete' "
            f"data-logo-id='{html.escape(str(row['id']))}' data-logo-name='{html.escape(brand)}'>Supprimer</button>"
            if can_delete else "<span class='library-readonly'>Lecture seule</span>"
        )
        html_rows.append(
            "<tr>"
            f"<td data-sort-value='{html.escape(brand.lower())}'><div class='logo-preview-cell'><img class='logo-table-img' src='{uri}' alt='{html.escape(brand)}'></div></td>"
            f"<td data-sort-value='{html.escape(brand.lower())}'><strong>{html.escape(brand)}</strong></td>"
            f"<td data-sort-value='{html.escape(str(row['owner']).lower())}'>{html.escape(str(row['owner']))}</td>"
            f"<td data-sort-value='{html.escape(sharing.lower())}'>{html.escape(sharing)}</td>"
            f"<td data-sort-value='{html.escape(created)}'>{html.escape(created)}</td>"
            f"<td>{action}</td>"
            "</tr>"
        )

    return f"""
    <div class="library-table-wrap">
        <table class="library-table library-sortable" id="brand-logo-sortable-table">
            <thead>
                <tr>
                    <th><button type="button" class="library-sort-button" data-column="0"><span class="library-icon">◈</span>Aperçu <span class="sort-indicator">↕</span></button></th>
                    <th><button type="button" class="library-sort-button" data-column="1"><span class="library-icon">◇</span>Marque <span class="sort-indicator">↕</span></button></th>
                    <th><button type="button" class="library-sort-button" data-column="2"><span class="library-icon">○</span>Propriétaire <span class="sort-indicator">↕</span></button></th>
                    <th><button type="button" class="library-sort-button" data-column="3"><span class="library-icon">⌁</span>Partage <span class="sort-indicator">↕</span></button></th>
                    <th><button type="button" class="library-sort-button" data-column="4" data-type="date"><span class="library-icon">◷</span>Ajout <span class="sort-indicator">↕</span></button></th>
                    <th><span class="table-head-label"><span class="library-icon">×</span>Actions</span></th>
                </tr>
            </thead>
            <tbody>{''.join(html_rows)}</tbody>
        </table>
    </div>
    """


def refresh_brand_logo_manager(request: gr.Request):
    return list_brand_logos_html(request)


def logo_progress_html(active_step=1):
    items = [("Fichier", "Choisir le logo"), ("Informations", "Nom de la marque"), ("Partage", "Définir les accès")]
    parts = []
    for index, (title, subtitle) in enumerate(items, start=1):
        state = "active" if index == active_step else ("done" if index < active_step else "")
        marker = "✓" if index < active_step else str(index)
        parts.append(f"""<div class="library-progress-item {state}"><span>{marker}</span><div><strong>{title}</strong><small>{subtitle}</small></div></div>""")
        if index < len(items):
            line_state = "done" if index < active_step else ""
            parts.append(f'<div class="library-progress-line {line_state}"></div>')
    return '<div class="library-wizard-progress logo-wizard-progress">' + "".join(parts) + "</div>"


def brand_logo_visibility_changed(visibility):
    shared = visibility == "Partagé — utilisateurs choisis"
    if shared:
        return gr.update(visible=True)
    return gr.update(visible=False, value="")


def logo_go_to_step_2(uploaded_file):
    if uploaded_file is None:
        return alert_html("Ajoutez d’abord une image PNG, JPG ou WEBP.", "error"), logo_progress_html(1), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    return "", logo_progress_html(2), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)


def logo_go_to_step_3(brand_name):
    if not (brand_name or "").strip():
        return alert_html("Indiquez le nom exact de la marque.", "error"), logo_progress_html(2), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    return "", logo_progress_html(3), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)


def logo_back_to_step_1():
    return "", logo_progress_html(1), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)


def logo_back_to_step_2():
    return "", logo_progress_html(2), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

def save_brand_logo(uploaded_file, brand_name, visibility, shared_users_text, request: gr.Request):
    username = request_username(request)
    if uploaded_file is None:
        return alert_html("Ajoute d’abord une image de logo.", "error"), list_brand_logos_html(request), gr.update(visible=False)
    clean = (brand_name or "").strip()
    if not clean:
        return alert_html("Indique le nom exact de la classe YOLO, par exemple Nike.", "error"), list_brand_logos_html(request), gr.update(visible=False)
    source = Path(uploaded_file)
    ext = source.suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
        return alert_html("Format non accepté.", "error"), list_brand_logos_html(request), gr.update(visible=False)

    visibility_code = "selected" if visibility == "Partagé — utilisateurs choisis" else "private"
    shared_users = [x.strip() for x in re.split(r"[,;\n]+", shared_users_text or "") if x.strip()]
    if visibility_code == "selected" and not shared_users:
        return alert_html("Écris au moins un nom d’utilisateur avec qui partager le logo.", "error"), list_brand_logos_html(request), gr.update(visible=True)

    valid_users = set(_active_usernames(exclude=username))
    invalid = sorted(set(shared_users) - valid_users)
    if invalid:
        return alert_html("Utilisateurs inconnus : " + ", ".join(invalid), "error"), list_brand_logos_html(request), gr.update(visible=True)

    with db_connection() as conn:
        existing = conn.execute(
            "SELECT id, stored_filename FROM brand_logos WHERE owner = ? AND LOWER(brand_name) = LOWER(?)",
            (username, clean),
        ).fetchone()
        logo_id = existing["id"] if existing else uuid.uuid4().hex
        stored_filename = f"{brand_slug(clean)}_{username}_{logo_id[:8]}{ext}"
        destination = BRAND_LOGOS_DIR / stored_filename
        if existing:
            old_path = BRAND_LOGOS_DIR / Path(existing["stored_filename"]).name
            if old_path.exists() and old_path != destination:
                old_path.unlink(missing_ok=True)
        shutil.copy2(source, destination)
        conn.execute(
            """
            INSERT INTO brand_logos(id, owner, brand_name, stored_filename, visibility, shared_with)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner, brand_name) DO UPDATE SET
                stored_filename = excluded.stored_filename,
                visibility = excluded.visibility,
                shared_with = excluded.shared_with,
                created_at = CURRENT_TIMESTAMP
            """,
            (logo_id, username, clean, stored_filename, visibility_code, json.dumps(shared_users, ensure_ascii=False)),
        )

    return alert_html(f"Logo de {html.escape(clean)} ajouté avec succès.", "success"), list_brand_logos_html(request), gr.update(visible=False)


def delete_brand_logo_by_id(logo_id, request: gr.Request):
    username = request_username(request)
    logo_id = str(logo_id or "").strip()
    if not logo_id:
        return alert_html("Aucun logo sélectionné.", "error"), list_brand_logos_html(request)
    with db_connection() as conn:
        if user_is_admin(username):
            row = conn.execute("SELECT stored_filename FROM brand_logos WHERE id = ?", (logo_id,)).fetchone()
        else:
            row = conn.execute("SELECT stored_filename FROM brand_logos WHERE id = ? AND owner = ?", (logo_id, username)).fetchone()
        if not row:
            return alert_html("Ce logo ne peut pas être supprimé avec ce compte.", "error"), list_brand_logos_html(request)
        target = BRAND_LOGOS_DIR / Path(row["stored_filename"]).name
        if target.exists() and target.parent.resolve() == BRAND_LOGOS_DIR.resolve():
            target.unlink(missing_ok=True)
        if user_is_admin(username):
            conn.execute("DELETE FROM brand_logos WHERE id = ?", (logo_id,))
        else:
            conn.execute("DELETE FROM brand_logos WHERE id = ? AND owner = ?", (logo_id, username))
    return alert_html("Logo supprimé.", "success"), list_brand_logos_html(request)

css = f"""
.brand-label {{ display:inline-flex; align-items:center; gap:9px; font-weight:650; white-space:nowrap; }}
.brand-logo {{ width:34px; height:24px; object-fit:contain; border-radius:5px; background:white; padding:2px; }}
.logo-library-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:14px; margin-top:12px; }}
.logo-library-card {{ display:flex; align-items:center; gap:14px; background:white; border:1px solid var(--line); border-radius:17px; padding:14px; min-height:82px; }}
.logo-library-preview {{ width:62px; height:50px; display:grid; place-items:center; background:#f8fafc; border-radius:12px; }}
.logo-library-preview img {{ max-width:50px; max-height:38px; object-fit:contain; }}
.logo-library-card strong {{ display:block; font-size:13px; }} .logo-library-card small {{ display:block; margin-top:5px; color:var(--muted); font-size:10px; }}
.consolidated-report-highlight{{background:linear-gradient(135deg,#172033,#46619c)!important;border-radius:24px!important;padding:22px!important;margin-bottom:20px!important;box-shadow:0 18px 45px rgba(70,97,156,.25)!important}}.consolidated-report-highlight label,.consolidated-report-highlight .label-wrap{{color:white!important}}.report-highlight-copy{{display:flex;flex-direction:column;gap:5px;color:white!important;margin-bottom:10px}}.report-highlight-copy strong{{font-family:Georgia,serif;font-size:22px;color:#ffffff!important;text-decoration:none!important}}.report-highlight-copy span{{color:#dce7ff!important;font-size:13px}}

.logo-empty-state {{display:flex;flex-direction:column;gap:5px;padding:24px;border:1px dashed #cbd5e1;border-radius:18px;background:#f8fafc;color:#475569}}
.logo-recap-table th {{background:#f3f6fb;color:#46619c;font-size:11px;letter-spacing:.06em;text-transform:uppercase;padding:14px 16px;text-align:left;border-bottom:1px solid #dfe7f2}}
.logo-recap-table td {{padding:15px 16px;border-bottom:1px solid #edf2f7;vertical-align:middle;color:#334155}}
.logo-recap-table tbody tr:hover {{background:#f8fbff}}
.logo-preview-cell {{width:72px;height:48px;display:grid;place-items:center;background:#f8fafc;border:1px solid #edf2f7;border-radius:12px}}
.logo-table-img {{max-width:58px;max-height:34px;object-fit:contain}}
.logo-brand-name {{font-weight:800;color:#172033;font-size:14px}}
.logo-recap-table small {{color:#8491a6}}
.logo-visibility-badge {{display:inline-flex;align-items:center;padding:6px 10px;border-radius:999px;font-size:11px;font-weight:750}}
.logo-visibility-badge.private {{background:#eef2ff;color:#46619c}}
.logo-visibility-badge.shared {{background:#ecfdf5;color:#287a50}}
.logo-delete-check {{display:flex;align-items:center;gap:8px;color:#a9172b;font-size:12px;font-weight:700}}
.logo-delete-check input {{width:17px;height:17px;accent-color:#a9172b}}
.library-inline-delete.logo-inline-delete {{min-width:92px!important;}}
.library-readonly {{color:#94a3b8;font-size:12px;font-weight:700;}}
#brand-logo-sortable-table td:first-child {{width:110px;}}
#brand-logo-sortable-table .logo-preview-cell {{margin:0 auto 0 0;}}
.logo-manager-grid {{display:grid;grid-template-columns:1.15fr .85fr;gap:18px}}
@media(max-width:900px){{.logo-manager-grid{{grid-template-columns:1fr}}}}

.logo-recap-title {{ font-family:'Fraunces',serif; font-size:22px; font-weight:700; margin:2px 0 16px; }}
.logo-table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:18px; background:white; }}
.logo-recap-table {{ width:100%; border-collapse:collapse; }}
.logo-recap-table th {{ background:#f3f6fb; color:var(--muted); text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.06em; padding:13px 16px; }}
.logo-recap-table td {{ padding:13px 16px; border-top:1px solid #edf2f8; font-size:13px; }}
.logo-table-img {{ width:58px; height:36px; object-fit:contain; background:white; border-radius:8px; }}
.logo-library-empty {{ padding:24px; border:1px dashed #cbd5e1; border-radius:18px; background:#f8fafc; display:flex; flex-direction:column; gap:6px; color:var(--muted); }}

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');
:root {{ --blue:#46619c; --blue-dark:#2f4270; --ink:#172033; --muted:#64748b; --bg:#eef3f8; --paper:#ffffff; --line:#e7edf5; --green:#3f8b5b; --green-bg:#edf8f1; --red:#a9172b; --red-dark:#821020; --sand:#f5efe6; }}
html, body, .gradio-container, .gradio-container > .main {{ background:var(--bg)!important; margin:0!important; }}
.gradio-container {{ max-width:none!important; width:100%!important; min-height:100vh!important; font-family:'Inter',system-ui,sans-serif!important; color:var(--ink)!important; }}
footer {{ display:none!important; }}
#app {{ width:calc(100% - 32px); max-width:none; margin:0 auto; padding:28px 0 80px; }}
.topbar {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:22px; position:sticky; top:12px; z-index:500; }}
.brand {{ display:flex; align-items:center; gap:12px; }}
.logo-mark {{ width:46px; height:46px; border-radius:16px; display:grid; place-items:center; color:white; font-weight:900; background:linear-gradient(135deg,var(--blue),#6f86bd); box-shadow:0 14px 30px rgba(70,97,156,.24); position:relative; overflow:hidden; }}
.logo-mark:before {{ content:''; position:absolute; width:26px; height:26px; border:3px solid rgba(255,255,255,.78); border-radius:999px; }}
.logo-mark:after {{ content:''; position:absolute; width:8px; height:8px; background:white; border-radius:999px; box-shadow:13px 0 0 rgba(255,255,255,.78), 6px 12px 0 rgba(255,255,255,.62); }}
.brand-title {{ font-family:'Fraunces',serif; font-size:28px; font-weight:700; letter-spacing:-.035em; }}
.brand-subtitle {{ font-size:12px; color:var(--muted); margin-top:2px; }}
.menu-wrap {{ position:relative; }}
.menu-btn {{ border:1px solid var(--line); background:rgba(255,255,255,.86); backdrop-filter:blur(12px); border-radius:16px; padding:12px 17px; cursor:pointer; font:800 13px 'Inter'; box-shadow:0 16px 42px rgba(25,35,58,.10); transition:all .18s ease; }}
.menu-btn:hover {{ background:var(--blue); color:white; transform:translateY(-1px); box-shadow:0 20px 46px rgba(70,97,156,.25); }}
.menu-content {{
    display:none;
    position:fixed;
    right:28px;
    top:86px;
    width:255px;
    max-height:calc(100vh - 105px);
    overflow-y:auto;
    overflow-x:hidden;
    background:rgba(255,255,255,.97);
    backdrop-filter:blur(18px);
    border:1px solid var(--line);
    border-radius:20px;
    box-shadow:0 26px 70px rgba(25,35,58,.20);
    padding:9px;
    z-index:2147483647;
}}
.menu-content.open {{ display:block; animation:menuIn .14s ease-out; }} @keyframes menuIn {{ from {{opacity:0; transform:translateY(-6px)}} to {{opacity:1; transform:translateY(0)}} }}
.menu-item {{ display:flex; align-items:center; gap:10px; width:100%; border:0; background:transparent; text-align:left; padding:13px 12px; border-radius:13px; font:800 13px 'Inter'; color:var(--ink); cursor:pointer; transition:all .16s; }}
.menu-item:hover {{ background:#edf2fb; color:var(--blue); transform:translateX(2px); }}
.menu-dot {{ width:8px; height:8px; border-radius:99px; background:var(--blue); opacity:.35; }}
#main-tabs [role='tablist'], #main-tabs > .tab-nav, #main-tabs .tab-nav {{ display:none!important; }}
.hero {{ position:relative; overflow:hidden; background:linear-gradient(135deg,#111a2d 0%,#26375e 48%,#5570aa 100%); border-radius:34px; padding:clamp(48px,7vh,86px) 46px; color:white; box-shadow:0 28px 80px rgba(33,48,82,.22); margin-bottom:0; min-height: calc(100vh - 155px); display:flex; align-items:center; box-sizing:border-box; }}
.hero:before {{ content:''; position:absolute; inset:0; background:radial-gradient(circle at 75% 15%, rgba(255,255,255,.22), transparent 32%), radial-gradient(circle at 8% 85%, rgba(245,239,230,.18), transparent 28%); }}
.hero:after {{ content:'◌'; position:absolute; right:50px; top:35px; font-size:170px; line-height:1; color:rgba(255,255,255,.10); font-family:serif; }}
.hero-content {{ position:relative; z-index:1; }}
.hero-kicker {{ font-size:11px; letter-spacing:.18em; text-transform:uppercase; font-weight:900; color:#dce7ff; margin-bottom:16px; }}
.hero h1 {{ font-family:'Fraunces',serif; font-size:52px; line-height:1.02; max-width:820px; margin:0; letter-spacing:-.05em; color:white; }}
.hero p {{ max-width:760px; color:#edf4ff; line-height:1.65; font-size:15px; margin:18px 0 0; }}
.workflow {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:34px; position:relative; z-index:1; }}
.workflow-item {{ background:rgba(255,255,255,.11); border:1px solid rgba(255,255,255,.17); backdrop-filter:blur(10px); border-radius:20px; padding:18px; transition:.2s; }}
.workflow-item:hover {{ transform:translateY(-4px); background:rgba(255,255,255,.16); }}
.workflow-number {{ font-size:11px; font-weight:900; color:#dce7ff; }} .workflow-title {{ font-size:16px; font-weight:900; margin-top:9px; color:white; }} .workflow-text {{ font-size:12px; color:#e7eefc; margin-top:5px; }}
.home-card {{ background:white; border-radius:24px; padding:26px; box-shadow:0 16px 55px rgba(30,43,70,.07); margin-top:18px; }}
.section-head {{ margin:28px 0 18px; }} .section-kicker {{ font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:var(--blue); font-weight:900; }} .section-title {{ font-family:'Fraunces',serif; font-size:31px; font-weight:700; letter-spacing:-.035em; margin-top:5px; }} .section-subtitle {{ font-size:13px; color:var(--muted); margin-top:6px; }}
.step-card {{ background:white!important; border:0!important; border-radius:24px!important; padding:18px!important; box-shadow:0 14px 44px rgba(30,43,70,.065)!important; margin-bottom:24px!important; }}
.gradio-group,.group,fieldset,.form,.panel,.block,.block.padded {{ border:0!important; box-shadow:none!important; }}
.block > .label, .block-title, .wrap > .label-wrap, .label-wrap {{ display:none!important; }}
.alert {{ border:0!important; border-radius:14px!important; padding:12px 15px!important; font-size:12px!important; font-weight:800!important; box-shadow:none!important; }}
button.primary,#train_button,#analyze_button,#validate_zip_button,#validate_model_button,#validate_video_button,#see_results_button {{ background:linear-gradient(135deg,var(--blue),#5d78b4)!important; border:0!important; color:#fff!important; border-radius:14px!important; min-height:46px!important; font-weight:900!important; transition:all .18s ease!important; box-shadow:0 12px 24px rgba(70,97,156,.16)!important; }}
button.primary:hover,#train_button:hover,#analyze_button:hover,#validate_zip_button:hover,#validate_model_button:hover,#validate_video_button:hover,#see_results_button:hover {{ background:linear-gradient(135deg,var(--blue-dark),#405b91)!important; transform:translateY(-2px)!important; box-shadow:0 18px 38px rgba(70,97,156,.30)!important; filter:saturate(1.08); }}
#stop_train_button,#stop_analyze_button {{ background:linear-gradient(135deg,var(--red),#c93d50)!important; color:#fff!important; border:0!important; border-radius:14px!important; min-height:46px!important; font-weight:900!important; transition:all .18s ease!important; box-shadow:0 12px 24px rgba(169,23,43,.18)!important; }}
#stop_train_button:hover,#stop_analyze_button:hover {{ background:linear-gradient(135deg,var(--red-dark),#a9172b)!important; transform:translateY(-2px)!important; box-shadow:0 18px 38px rgba(169,23,43,.30)!important; }}
input[type=range] {{ accent-color:var(--blue)!important; }}
.progress-card {{ background:white; border:0; border-radius:18px; padding:15px 17px; box-shadow:0 10px 36px rgba(30,43,70,.07); }} .progress-top {{ display:flex; justify-content:space-between; gap:12px; font-size:12px; }} .progress-track {{ height:8px; background:#e7edf5; border-radius:99px; overflow:hidden; margin-top:11px; }} .progress-fill {{ height:100%; border-radius:99px; transition:width .3s; }}
.kpi-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin:10px 0 22px; }} .kpi-card {{ background:white; border:0; border-radius:22px; padding:20px; box-shadow:0 14px 44px rgba(30,43,70,.07); }} .kpi-label {{ font-size:11px; color:var(--muted); font-weight:800; }} .kpi-value {{ font-family:'Fraunces',serif; font-size:25px; font-weight:700; margin-top:7px; color:var(--ink); }}
.chart-card {{ background:white!important; border:0!important; border-radius:24px!important; padding:18px!important; box-shadow:0 14px 44px rgba(30,43,70,.07)!important; overflow:visible!important; }} .chart-title {{ font-family:'Fraunces',serif; font-size:21px; font-weight:700; color:var(--ink); padding:2px 2px 14px; }} .chart-card img {{ border-radius:14px!important; }} .main-table {{ background:white!important; border:0!important; border-radius:22px!important; padding:10px!important; box-shadow:0 14px 44px rgba(30,43,70,.07)!important; }}

.results-section-title {{ font-family:'Fraunces',serif; font-size:24px; font-weight:700; color:var(--ink); margin:30px 2px 14px; letter-spacing:-.025em; }}
.main-table {{ overflow-x:auto!important; width:100%!important; }}
.main-table .table-wrap, .main-table .wrap {{ overflow-x:auto!important; width:100%!important; }}
.main-table table {{ min-width:1450px!important; width:max-content!important; }}
.main-table th, .main-table td {{ white-space:normal!important; min-width:105px!important; }}
.main-table th:first-child, .main-table td:first-child {{ position:sticky!important; left:0!important; z-index:3!important; background:white!important; min-width:160px!important; font-weight:800!important; }}
.chart-card {{ min-height:430px!important; }}
.chart-card img {{ width:100%!important; height:auto!important; object-fit:contain!important; }}
.brand-mark svg {{ display:block; filter:drop-shadow(0 8px 14px rgba(70,97,156,.22)); }}


.logo-mark {{ display:none!important; }}
.metric-card {{ background:white; border-radius:22px; padding:18px; margin:14px 0; box-shadow:0 14px 44px rgba(30,43,70,.07); }}
.metric-title {{ font-family:'Fraunces',serif; font-size:22px; font-weight:700; margin-bottom:12px; }}
.metric-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }}
.metric-grid div {{ background:#f6f8fc; border-radius:16px; padding:14px; }}
.metric-grid span {{ display:block; font-size:11px; color:var(--muted); font-weight:800; }}
.metric-grid strong {{ display:block; margin-top:5px; font-size:22px; color:var(--blue); }}
.metric-note,.table-note {{ font-size:12px; color:var(--muted); font-weight:700; margin-top:10px; }}


.brand {{ display:flex!important; align-items:center!important; gap:14px!important; }}
.brand-copy {{ display:flex!important; flex-direction:column!important; justify-content:center!important; }}
.brand-home {{ flex:0 0 auto!important; }}
.brand-title {{ margin:0!important; line-height:1.05!important; }}
.brand-subtitle {{ margin-top:5px!important; }}
.results-table {{ border-collapse:separate!important; border-spacing:0!important; overflow:hidden!important; border-radius:16px!important; }}
.results-table th {{ background:linear-gradient(135deg,#46619c,#5d78b4)!important; color:white!important; border-bottom:0!important; }}
.results-table th:first-child {{ background:linear-gradient(135deg,#46619c,#5d78b4)!important; }}
.results-table tbody tr:nth-child(even) td {{ background:#f5f8fd!important; }}
.results-table tbody tr:hover td {{ background:#eaf0fb!important; transition:background .16s ease!important; }}
.results-table td {{ border-bottom:1px solid #e6edf7!important; }}
.results-table td:first-child {{ color:#46619c!important; }}
.metric-box {{ position:relative!important; overflow:hidden!important; }}
.metric-track {{ height:7px!important; border-radius:999px!important; background:#dfe7f3!important; margin-top:12px!important; overflow:hidden!important; }}
.metric-track i {{ display:block!important; height:100%!important; width:0; border-radius:999px!important; background:#46619c!important; animation:metricGrow 1.4s cubic-bezier(.2,.8,.2,1) forwards!important; }}
@keyframes metricGrow {{ from {{ width:0; }} to {{ width:var(--target); }} }}


@property --num {{ syntax: "<number>"; initial-value: 0; inherits: false; }}
.animated-number {{ animation: numberUp 1.6s cubic-bezier(.2,.8,.2,1) forwards; counter-reset: n calc(var(--num) * 10); }}
.animated-number::after {{ content: counter(n) " %"; }}
@keyframes numberUp {{ from {{ --num: 0; }} to {{ --num: var(--value); }} }}
.metric-track i {{ display:block!important; height:100%!important; width:0; border-radius:999px!important; background:linear-gradient(90deg,#46619c,#7c8fbd)!important; animation:metricGrow 1.6s cubic-bezier(.2,.8,.2,1) forwards!important; }}
@keyframes metricGrow {{ from {{ width:0; }} to {{ width:calc(var(--value) * 1%); }} }}
.split-results {{ display:grid!important; grid-template-columns:1fr 1fr!important; gap:18px!important; align-items:start!important; }}
.mini-table-card {{ background:white!important; border-radius:24px!important; padding:16px!important; box-shadow:0 14px 44px rgba(30,43,70,.07)!important; }}
.mini-table-title {{ font-family:'Fraunces',serif!important; font-size:20px!important; font-weight:800!important; margin:0 0 12px!important; color:#172033!important; }}
.th-icon {{ display:inline-flex!important; align-items:center!important; justify-content:center!important; margin-right:7px!important; }}
.table-note {{ font-size:12px!important; color:#64748b!important; font-weight:800!important; margin:12px 2px 0!important; }}
.results-table {{ border-collapse:separate!important; border-spacing:0!important; min-width:620px!important; overflow:hidden!important; border-radius:16px!important; }}
.results-table th {{ background:linear-gradient(135deg,#46619c,#5d78b4)!important; color:white!important; font-size:12px!important; border:0!important; padding:13px 12px!important; }}
.results-table td {{ background:white!important; border-bottom:1px solid #e9eef7!important; color:#263044!important; font-size:13px!important; padding:12px!important; }}
.results-table tbody tr:nth-child(even) td {{ background:#f5f8fd!important; }}
.results-table tbody tr:hover td {{ background:#eaf0fb!important; }}
.results-table td:first-child {{ color:#46619c!important; font-weight:900!important; }}
@media(max-width:1050px) {{ .split-results {{ grid-template-columns:1fr!important; }} }}


/* Compteurs de précision corrigés */
.animated-number::after {{ content:none!important; }}
.count-up {{ font-variant-numeric:tabular-nums!important; }}
.metric-track i {{ display:block!important; height:100%!important; width:0%; border-radius:999px!important; background:linear-gradient(90deg,#46619c,#7c8fbd)!important; transition:width 2.8s cubic-bezier(.2,.8,.2,1)!important; }}


/* V14 - compteurs sans barres */
.metric-track {{ display:none!important; }}
.metric-box-clean {{ min-height:92px!important; display:flex!important; flex-direction:column!important; justify-content:center!important; }}
.count-up {{ font-variant-numeric:tabular-nums!important; }}
/* V14 - tableaux modernes sans dégradé */
.split-results {{ display:grid!important; grid-template-columns:1fr 1fr!important; gap:18px!important; align-items:start!important; }}
.mini-table-card {{ background:#ffffff!important; border:1px solid #dbe4f3!important; border-radius:24px!important; padding:16px!important; box-shadow:0 14px 44px rgba(30,43,70,.07)!important; }}
.mini-table-title {{ font-family:'Fraunces',serif!important; font-size:21px!important; font-weight:800!important; margin:0 0 14px!important; color:#172033!important; display:flex!important; align-items:center!important; gap:10px!important; }}
.table-scroll {{ overflow-x:auto!important; box-shadow:none!important; padding:0!important; border:1px solid #dbe4f3!important; border-radius:16px!important; }}
.results-table {{ border-collapse:collapse!important; width:100%!important; min-width:620px!important; border-radius:16px!important; overflow:hidden!important; }}
.results-table th {{ background:#f3f6fb!important; color:#172033!important; font-size:12px!important; border-right:1px solid #dbe4f3!important; border-bottom:1px solid #dbe4f3!important; padding:13px 12px!important; font-weight:900!important; }}
.results-table th:first-child {{ background:#f3f6fb!important; }}
.results-table td {{ background:#ffffff!important; border-right:1px solid #e2eaf5!important; border-bottom:1px solid #e2eaf5!important; color:#263044!important; font-size:13px!important; padding:12px!important; }}
.results-table tbody tr:nth-child(even) td {{ background:#f8fbff!important; }}
.results-table tbody tr:hover td {{ background:#edf3fb!important; transition:background .16s ease!important; }}
.results-table td:first-child {{ color:#46619c!important; font-weight:900!important; }}
.table-note {{ display:flex!important; align-items:center!important; gap:8px!important; font-size:12px!important; color:#64748b!important; font-weight:800!important; margin:12px 2px 0!important; }}
.line-icon,.line-title-icon {{ display:inline-block!important; position:relative!important; width:18px!important; height:18px!important; flex:0 0 18px!important; border:2px solid #46619c!important; border-radius:5px!important; box-sizing:border-box!important; opacity:.95!important; }}
.line-title-icon {{ width:22px!important; height:22px!important; border-radius:7px!important; }}
.i-clock {{ border-radius:50%!important; }}
.i-clock::before {{ content:''; position:absolute; left:8px; top:3px; width:2px; height:6px; background:#46619c; border-radius:2px; }}
.i-clock::after {{ content:''; position:absolute; left:8px; top:8px; width:5px; height:2px; background:#46619c; border-radius:2px; transform:rotate(25deg); transform-origin:left center; }}
.i-film::before,.i-film::after {{ content:''; position:absolute; top:2px; bottom:2px; width:2px; background:#46619c; }}
.i-film::before {{ left:4px; }} .i-film::after {{ right:4px; }}
.i-loop {{ border-radius:50%!important; }} .i-loop::after {{ content:''; position:absolute; right:-3px; top:2px; width:5px; height:5px; border-top:2px solid #46619c; border-right:2px solid #46619c; transform:rotate(45deg); }}
.i-wave {{ border:0!important; border-top:3px solid #46619c!important; border-radius:0!important; transform:translateY(8px); }}
.i-pin {{ border-radius:50% 50% 50% 0!important; transform:rotate(-45deg) scale(.85); }} .i-pin::after {{ content:''; position:absolute; width:5px; height:5px; border-radius:50%; background:#46619c; left:4px; top:4px; }}
.i-square {{ border-radius:4px!important; }}
.i-grid::before {{ content:''; position:absolute; left:7px; top:0; bottom:0; border-left:2px solid #46619c; }} .i-grid::after {{ content:''; position:absolute; top:7px; left:0; right:0; border-top:2px solid #46619c; }}
.i-target {{ border-radius:50%!important; }} .i-target::before {{ content:''; position:absolute; inset:4px; border:2px solid #46619c; border-radius:50%; }} .i-target::after {{ content:''; position:absolute; left:7px; top:-4px; height:24px; border-left:1px solid #46619c; }}
.i-bars {{ border:0!important; border-radius:0!important; }} .i-bars::before {{ content:''; position:absolute; left:2px; bottom:2px; width:3px; height:7px; background:#46619c; box-shadow:6px -4px 0 #46619c,12px -9px 0 #46619c; border-radius:2px; }}
.i-spark {{ transform:rotate(45deg) scale(.82); border-radius:2px!important; }}
.i-tag {{ border-radius:5px 5px 5px 1px!important; transform:rotate(-45deg) scale(.82); }} .i-tag::after {{ content:''; position:absolute; width:4px; height:4px; border-radius:50%; background:#46619c; right:2px; top:2px; }}
.i-dot {{ border-radius:50%!important; }}
@media(max-width:1050px) {{ .split-results {{ grid-template-columns:1fr!important; }} }}


/* Tableaux V15 : icônes propres et séparations nettes */
.mini-icon {{ display:inline-flex!important; width:14px!important; height:14px!important; min-width:14px!important; margin-right:7px!important; vertical-align:-2px!important; }}
.mini-icon svg {{ width:14px!important; height:14px!important; display:block!important; fill:none!important; stroke:#46619c!important; stroke-width:2!important; stroke-linecap:round!important; stroke-linejoin:round!important; }}
.note-icon {{ margin-right:6px!important; }}
.title-dot {{ display:inline-block!important; width:10px!important; height:10px!important; border:2px solid #46619c!important; border-radius:50%!important; margin-right:9px!important; vertical-align:1px!important; }}
.mini-table-card {{ border:1px solid #dbe4f3!important; box-shadow:0 16px 46px rgba(30,43,70,.065)!important; }}
.results-table {{ border-collapse:separate!important; border-spacing:0!important; min-width:620px!important; border:1px solid #dbe4f3!important; border-radius:16px!important; overflow:hidden!important; }}
.results-table th {{ background:#f5f8fd!important; color:#172033!important; border-right:1px solid #dbe4f3!important; border-bottom:1px solid #dbe4f3!important; font-size:12px!important; padding:13px 12px!important; }}
.results-table th:last-child, .results-table td:last-child {{ border-right:0!important; }}
.results-table td {{ background:#ffffff!important; border-right:1px solid #e7edf5!important; border-bottom:1px solid #e7edf5!important; color:#263044!important; padding:12px!important; }}
.results-table tbody tr:nth-child(even) td {{ background:#fbfdff!important; }}
.results-table tbody tr:hover td {{ background:#edf3fc!important; }}
.results-table td:first-child {{ color:#46619c!important; font-weight:900!important; }}
.table-note {{ display:flex!important; align-items:center!important; gap:2px!important; }}


/* V16 - Tableaux plus propres : petits pictos, séparations douces, aucun trait noir */
.mini-icon {{ width:10px!important; height:10px!important; min-width:10px!important; margin-right:5px!important; vertical-align:-1px!important; }}
.mini-icon svg {{ width:10px!important; height:10px!important; stroke:#46619c!important; stroke-width:1.75!important; }}
.note-icon {{ width:11px!important; height:11px!important; min-width:11px!important; }}
.title-dot {{ width:7px!important; height:7px!important; border:1.7px solid #46619c!important; margin-right:8px!important; }}
.mini-table-card {{ border:1px solid #e0e8f4!important; box-shadow:0 14px 38px rgba(30,43,70,.055)!important; }}
.table-scroll {{ box-shadow:none!important; border-radius:16px!important; background:transparent!important; padding:0!important; }}
.results-table {{ border-collapse:separate!important; border-spacing:0!important; min-width:610px!important; border:1px solid #e0e8f4!important; border-radius:15px!important; overflow:hidden!important; background:white!important; }}
.results-table th {{ background:#f7f9fd!important; color:#172033!important; border-right:1px solid #e0e8f4!important; border-bottom:1px solid #e0e8f4!important; font-size:11.5px!important; padding:12px 11px!important; font-weight:900!important; }}
.results-table td {{ background:#ffffff!important; border-right:1px solid #edf2f8!important; border-bottom:1px solid #edf2f8!important; color:#263044!important; font-size:12.5px!important; padding:12px 11px!important; }}
.results-table tr:last-child td {{ border-bottom:0!important; }}
.results-table th:last-child, .results-table td:last-child {{ border-right:0!important; }}
.results-table tbody tr:nth-child(even) td {{ background:#fbfdff!important; }}
.results-table tbody tr:hover td {{ background:#f0f5fc!important; }}
.results-table td:first-child {{ color:#46619c!important; font-weight:900!important; }}
.table-note {{ font-size:11.5px!important; color:#65748a!important; }}


.results-intro {{ margin-bottom:8px!important; }}
.results-downloads-title {{ font-size:13px; font-weight:900; color:var(--muted); margin:18px 2px 8px; text-transform:uppercase; letter-spacing:.08em; }}
.results-block {{ background:rgba(255,255,255,.42); border:1px solid rgba(221,230,241,.9); border-radius:28px; padding:4px 18px 22px; margin:22px 0; }}
.results-block .results-section-title {{ margin-top:20px; }}

@media(max-width:850px){{ #app{{width:calc(100% - 24px)}} .workflow,.kpi-grid{{grid-template-columns:1fr}} .hero{{padding:34px 26px}} .hero h1{{font-size:38px}} }}

/* === FINAL TABLE OVERRIDE: contours doux, aucun trait noir === */
.results-table {{ border:1px solid #e4eaf3!important; border-collapse:separate!important; border-spacing:0!important; }}
.results-table th {{ border-right:1px solid #e4eaf3!important; border-bottom:1px solid #dfe6f0!important; }}
.results-table td {{ border-right:1px solid #edf1f6!important; border-bottom:1px solid #edf1f6!important; }}
.results-table th:last-child,.results-table td:last-child {{ border-right:0!important; }}
.results-table tr:last-child td {{ border-bottom:0!important; }}
.mini-table-card,.table-scroll {{ border-color:#e4eaf3!important; box-shadow:none!important; }}


/* === V27 : accueil fixe/aéré + résultats espacés + tableaux sans traits noirs === */
.hero {{
  min-height: calc(100vh - 118px) !important;
  height: calc(100vh - 118px) !important;
  padding: clamp(64px, 11vh, 120px) 70px !important;
  align-items: center !important;
  margin-bottom: 0 !important;
}}
.hero-content {{
  width: 100% !important;
  max-width: none !important;
}}
.hero-kicker {{
  letter-spacing: .32em !important;
  margin-bottom: 26px !important;
}}
.hero h1 {{
  max-width: 880px !important;
  font-size: clamp(48px, 4.3vw, 70px) !important;
  line-height: .98 !important;
}}
.hero p {{
  max-width: 850px !important;
  margin-top: 26px !important;
  font-size: 16px !important;
  line-height: 1.75 !important;
}}
.workflow {{
  grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
  gap: 24px !important;
  margin-top: 58px !important;
  width: 100% !important;
}}
.workflow-item {{
  min-height: 94px !important;
  padding: 24px 28px !important;
  border-radius: 22px !important;
}}
.workflow-title {{
  font-size: 18px !important;
  margin-top: 12px !important;
}}
.workflow-text {{
  font-size: 13px !important;
  margin-top: 7px !important;
}}
.results-intro {{ margin-bottom: 28px !important; }}
.kpi-grid {{ gap: 22px !important; margin: 26px 0 42px !important; }}
.kpi-card {{ padding: 24px 22px !important; }}
.results-block {{
  margin: 36px 0 44px !important;
  padding: 10px 22px 30px !important;
  border-radius: 30px !important;
}}
.results-block .results-section-title {{ margin: 26px 2px 24px !important; }}
.results-downloads-title {{ margin: 28px 2px 14px !important; }}
.split-results {{ gap: 26px !important; }}
.mini-table-card {{ padding: 20px !important; border-color:#edf2f8!important; }}
.mini-table-title {{ margin-bottom:18px!important; }}
.table-scroll {{ border:1px solid #edf2f8!important; border-radius:16px!important; }}
.results-table,
.results-table th,
.results-table td {{
  border-color:#edf2f8!important;
  box-shadow:none!important;
  outline:0!important;
}}
.results-table {{ border:1px solid #edf2f8!important; border-collapse:separate!important; border-spacing:0!important; }}
.results-table th {{ border-right:1px solid #edf2f8!important; border-bottom:1px solid #e9eff7!important; background:#f7f9fd!important; }}
.results-table td {{ border-right:1px solid #f0f4fa!important; border-bottom:1px solid #f0f4fa!important; }}
.results-table th:last-child,.results-table td:last-child{{ border-right:0!important; }}
.results-table tr:last-child td{{ border-bottom:0!important; }}
@media(max-width:900px){{
  .hero{{height:auto!important; min-height:calc(100vh - 120px)!important; padding:48px 28px!important;}}
  .workflow{{grid-template-columns:1fr!important; gap:14px!important; margin-top:34px!important;}}
}}


/* === V31 : accueil sans défilement + respiration finale === */
body.home-active {{
  overflow: hidden !important;
}}
body.home-active #app {{
  padding-bottom: 0 !important;
}}
body.home-active .topbar {{
  margin-bottom: 18px !important;
}}
body.home-active .hero {{
  height: calc(100vh - 96px) !important;
  min-height: calc(100vh - 96px) !important;
  padding-top: clamp(72px, 12vh, 130px) !important;
  padding-bottom: clamp(72px, 12vh, 130px) !important;
}}


/* === V33 : accueil fixe sans scrollbar + rapport vraiment mis en avant === */
html:has(body.home-active) {{
  height:100vh!important;
  max-height:100vh!important;
  overflow:hidden!important;
}}
body.home-active {{
  height:100vh!important;
  max-height:100vh!important;
  overflow:hidden!important;
}}
body.home-active .gradio-container,
body.home-active .gradio-container > .main,
body.home-active main,
body.home-active .main {{
  height:100vh!important;
  max-height:100vh!important;
  overflow:hidden!important;
}}
body.home-active #app {{
  height:100vh!important;
  max-height:100vh!important;
  overflow:hidden!important;
  padding:16px 0 0!important;
  box-sizing:border-box!important;
}}
body.home-active .topbar {{
  margin-bottom:12px!important;
  flex:0 0 auto!important;
}}
body.home-active .hero {{
  height:calc(100vh - 96px)!important;
  min-height:0!important;
  max-height:calc(100vh - 96px)!important;
  padding:clamp(38px,6vh,72px) 70px!important;
  box-sizing:border-box!important;
  overflow:hidden!important;
}}
body.home-active .hero h1 {{
  font-size:clamp(44px,4vw,64px)!important;
}}
body.home-active .workflow {{
  margin-top:clamp(28px,4vh,46px)!important;
}}
body.home-active::-webkit-scrollbar {{
  display:none!important;
}}
.report-feature-title {{
  font-family:'Fraunces',serif;
  font-size:30px;
  font-weight:900;
  letter-spacing:-.04em;
  color:var(--ink);
  margin:30px 2px 8px;
}}
.report-feature-subtitle {{
  color:var(--muted);
  font-weight:800;
  font-size:13px;
  margin:0 2px 16px;
}}
.report-download-card {{
  background:linear-gradient(135deg,#172033 0%,#46619c 100%)!important;
  border-radius:28px!important;
  padding:28px!important;
  box-shadow:0 28px 80px rgba(70,97,156,.26)!important;
  border:1px solid rgba(255,255,255,.24)!important;
  margin:0 0 38px!important;
}}
.report-download-card .file-preview,
.report-download-card .wrap,
.report-download-card .block,
.report-download-card .file,
.report-download-card [data-testid="file"] {{
  background:rgba(255,255,255,.98)!important;
  border-radius:18px!important;
}}
.report-download-card * {{
  border-color:rgba(255,255,255,.30)!important;
}}

@media(max-width:900px) {{
  body.home-active .hero {{
    height:calc(100svh - 112px)!important;
    max-height:calc(100svh - 112px)!important;
    padding:28px 24px!important;
  }}
}}

/* === V34 : accueil vraiment fixe + bloc bleu plus raisonnable === */
body.home-active,
body.home-active .gradio-container,
body.home-active .gradio-container > .main,
body.home-active main,
body.home-active .main,
body.home-active #app {{
  overflow:hidden!important;
  overscroll-behavior:none!important;
}}
body.home-active #app {{
  height:100svh!important;
  max-height:100svh!important;
  padding:14px 0 0!important;
}}
body.home-active .topbar {{
  margin-bottom:10px!important;
}}
body.home-active .hero {{
  height:calc(100svh - 126px)!important;
  min-height:0!important;
  max-height:calc(100svh - 126px)!important;
  padding:clamp(34px,5vh,62px) 64px!important;
  align-items:center!important;
}}
body.home-active .hero h1 {{
  font-size:clamp(42px,3.8vw,60px)!important;
  max-width:880px!important;
}}
body.home-active .hero p {{
  max-width:760px!important;
  margin-top:16px!important;
}}
body.home-active .workflow {{
  grid-template-columns:repeat(3, minmax(0,1fr))!important;
  gap:18px!important;
  margin-top:clamp(26px,4vh,42px)!important;
}}
body.home-active .workflow-item {{
  padding:22px 24px!important;
  min-height:118px!important;
}}
body.home-active::-webkit-scrollbar,
body.home-active .gradio-container::-webkit-scrollbar,
body.home-active main::-webkit-scrollbar {{
  width:0!important;
  display:none!important;
}}

/* === V34 : rapport mis en avant plus élégant === */
.report-feature-title {{
  margin-top:38px!important;
}}
.report-download-card {{
  position:relative!important;
  overflow:hidden!important;
  display:block!important;
  background:linear-gradient(135deg,#f8fbff 0%,#eef4ff 100%)!important;
  border:1px solid #dbe6f6!important;
  border-left:12px solid #46619c!important;
  border-radius:28px!important;
  padding:24px 26px 26px 26px!important;
  box-shadow:0 22px 62px rgba(70,97,156,.18)!important;
  margin:0 0 42px!important;
}}

.report-download-card .file-preview,
.report-download-card .wrap,
.report-download-card .block,
.report-download-card .file,
.report-download-card [data-testid="file"] {{
  background:#ffffff!important;
  border:1px solid #dfe8f6!important;
  border-radius:18px!important;
  box-shadow:0 12px 28px rgba(30,43,70,.08)!important;
}}
.report-download-card * {{
  border-color:#dfe8f6!important;
}}


/* === PATCH FINAL : accueil fixe sans scrollbar, pages travail scrollables === */
/* Quand on n'est PAS sur l'accueil, on réactive explicitement le scroll pour Entraînement et Analyse. */
body:not(.home-active),
body:not(.home-active) .gradio-container,
body:not(.home-active) .gradio-container > .main,
body:not(.home-active) main,
body:not(.home-active) .main,
body:not(.home-active) #app {{
  height:auto!important;
  max-height:none!important;
  overflow-y:auto!important;
  overflow-x:hidden!important;
}}

/* Sur l'accueil uniquement : page fixe, sans barre de scroll. */
html:has(body.home-active),
body.home-active,
body.home-active .gradio-container,
body.home-active .gradio-container > .main,
body.home-active main,
body.home-active .main {{
  height:100svh!important;
  max-height:100svh!important;
  overflow:hidden!important;
}}
body.home-active #app {{
  height:100svh!important;
  max-height:100svh!important;
  overflow:hidden!important;
  padding:12px 0 0!important;
  box-sizing:border-box!important;
}}
body.home-active .topbar {{
  margin-bottom:10px!important;
}}
body.home-active .hero {{
  width:100%!important;
  height:min(650px, calc(100svh - 150px))!important;
  min-height:0!important;
  max-height:calc(100svh - 150px)!important;
  border-radius:30px!important;
  padding:clamp(28px,4.5vh,52px) 58px!important;
  align-items:center!important;
  overflow:hidden!important;
  box-sizing:border-box!important;
}}
body.home-active .hero h1 {{
  font-size:clamp(38px,3.4vw,54px)!important;
  line-height:1!important;
  max-width:760px!important;
}}
body.home-active .hero p {{
  max-width:700px!important;
  margin-top:14px!important;
  font-size:14px!important;
  line-height:1.55!important;
}}
body.home-active .workflow {{
  gap:14px!important;
  margin-top:clamp(20px,3vh,30px)!important;
}}
body.home-active .workflow-item {{
  min-height:82px!important;
  padding:16px 20px!important;
  border-radius:18px!important;
}}
body.home-active .workflow-title {{
  font-size:15px!important;
  margin-top:7px!important;
}}
body.home-active .workflow-text {{
  font-size:11.5px!important;
  margin-top:4px!important;
}}
body.home-active::-webkit-scrollbar,
body.home-active .gradio-container::-webkit-scrollbar,
body.home-active main::-webkit-scrollbar,
body.home-active .main::-webkit-scrollbar,
body.home-active #app::-webkit-scrollbar {{
  width:0!important;
  height:0!important;
  display:none!important;
}}

@media(max-width:900px){{
  body.home-active .hero{{
    height:calc(100svh - 132px)!important;
    max-height:calc(100svh - 132px)!important;
    padding:24px 22px!important;
  }}
  body.home-active .workflow{{
    grid-template-columns:1fr!important;
    gap:9px!important;
    margin-top:18px!important;
  }}
  body.home-active .workflow-item{{
    min-height:auto!important;
    padding:12px 16px!important;
  }}
}}


/* === CORRECTION FINALE : cadre bleu accueil plus petit === */
body.home-active .hero {{
    height: calc(100svh - 230px) !important;
    min-height: 0 !important;
    max-height: calc(100svh - 230px) !important;
    padding: 28px 58px !important;
    box-sizing: border-box !important;
    overflow: hidden !important;
}}

body.home-active #app {{
    height: 100svh !important;
    max-height: 100svh !important;
    overflow: hidden !important;
    padding-top: 8px !important;
    padding-bottom: 8px !important;
    box-sizing: border-box !important;
}}


.library-grid {{ display:grid; grid-template-columns:minmax(0,1.25fr) minmax(320px,.75fr); gap:22px; align-items:start; }}
.library-card {{ background:white!important; border:0!important; border-radius:24px!important; padding:22px!important; box-shadow:0 14px 44px rgba(30,43,70,.07)!important; }}
.library-table-wrap {{ width:100%; overflow:auto; border-radius:18px; border:1px solid var(--line); background:white; }}
.library-table {{ width:100%; min-width:760px; border:0!important; border-collapse:separate; border-spacing:0; }}
.library-table th {{ background:#f3f6fb; color:var(--ink); font-size:11px; text-transform:uppercase; letter-spacing:.05em; }}
.library-table td {{ vertical-align:top; }}
.library-description {{ margin-top:5px; color:var(--muted); font-size:11px; font-weight:500; }}
.library-empty {{ padding:24px; border-radius:18px; background:#f7f9fc; color:var(--muted); font-weight:700; }}
.library-help {{ padding:15px 17px; border-radius:16px; background:#edf2fb; color:#344a78; font-size:12px; line-height:1.55; margin-bottom:16px; }}
.user-admin-card {{ margin-top:24px; }}
@media(max-width:900px) {{ .library-grid {{ grid-template-columns:1fr; }} }}


.library-grid {{ grid-template-columns:1fr!important; }}
.library-download-link {{
    display:inline-flex;
    align-items:center;
    justify-content:center;
    min-height:34px;
    padding:0 14px;
    border-radius:10px;
    background:linear-gradient(135deg,var(--blue),#5d78b4);
    color:white!important;
    font-size:12px;
    font-weight:800;
    text-decoration:none!important;
    white-space:nowrap;
}}
.library-download-link:hover {{ filter:brightness(.95); transform:translateY(-1px); }}


/* Tableau premium de la bibliothèque */
.library-files-card {{
    padding-bottom:20px!important;
    margin-bottom:0!important;
}}
.library-files-title {{
    margin-bottom:2px!important;
}}
.library-table-wrap {{
    margin-top:18px!important;
    border:0!important;
    border-radius:18px!important;
    background:#fff!important;
    overflow:auto!important;
    box-shadow:inset 0 0 0 1px #edf1f7!important;
}}
.library-table {{
    width:100%!important;
    min-width:860px!important;
    border-collapse:separate!important;
    border-spacing:0!important;
    border:0!important;
}}
.library-table thead th {{
    padding:15px 16px!important;
    background:#f4f7fb!important;
    color:#506079!important;
    border:0!important;
    border-bottom:1px solid #e8eef6!important;
    font-size:10px!important;
    font-weight:800!important;
    letter-spacing:.07em!important;
    text-transform:uppercase!important;
    text-align:left!important;
    white-space:nowrap!important;
}}
.library-table thead th:first-child {{
    border-top-left-radius:18px!important;
}}
.library-table thead th:last-child {{
    border-top-right-radius:18px!important;
}}
.library-table tbody td {{
    padding:15px 16px!important;
    color:#27344a!important;
    border:0!important;
    border-bottom:1px solid #edf1f6!important;
    background:#fff!important;
    vertical-align:middle!important;
    font-size:12px!important;
}}
.library-table tbody tr:last-child td {{
    border-bottom:0!important;
}}
.library-table tbody tr:hover td {{
    background:#f8faff!important;
    transition:background .18s ease!important;
}}
.library-table tbody td:first-child strong {{
    color:#172033!important;
    font-weight:800!important;
}}
.table-head-label {{
    display:inline-flex!important;
    align-items:center!important;
    gap:7px!important;
}}
.library-icon {{
    display:inline-grid!important;
    place-items:center!important;
    width:20px!important;
    height:20px!important;
    border-radius:7px!important;
    background:#e8eef9!important;
    color:var(--blue)!important;
    font-size:12px!important;
    line-height:1!important;
    font-weight:900!important;
}}
.library-download-link {{
    display:inline-flex!important;
    align-items:center!important;
    justify-content:center!important;
    gap:7px!important;
    min-height:36px!important;
    padding:0 15px!important;
    border-radius:11px!important;
    background:linear-gradient(135deg,#46619c,#6380be)!important;
    color:white!important;
    font-size:12px!important;
    font-weight:800!important;
    text-decoration:none!important;
    white-space:nowrap!important;
    box-shadow:0 8px 18px rgba(70,97,156,.18)!important;
    transition:transform .18s ease, filter .18s ease, box-shadow .18s ease!important;
}}
.library-download-link:hover {{
    filter:brightness(1.12)!important;
    transform:translateY(-2px)!important;
    box-shadow:0 12px 26px rgba(70,97,156,.28)!important;
}}
.library-download-link:active {{
    filter:brightness(.90)!important;
    transform:translateY(0)!important;
}}
.download-arrow {{
    font-size:15px!important;
    line-height:1!important;
}}


/* =========================================================
   BIBLIOTHÈQUE — version plus légère, aérée et moins chargée
   ========================================================= */
.library-card {{
    padding:24px!important;
    border-radius:22px!important;
}}
.library-help {{
    padding:14px 16px!important;
    margin-bottom:18px!important;
    border-radius:14px!important;
    background:#f1f5fb!important;
    border:1px solid #e7edf6!important;
    color:#415272!important;
}}
.library-help strong {{
    color:#172033!important;
}}
.library-card .block {{
    margin-bottom:10px!important;
}}
.library-card textarea,
.library-card input {{
    border-radius:12px!important;
}}
.library-card [data-testid="file-upload"] {{
    min-height:170px!important;
    border:1px dashed #cbd7e8!important;
    border-radius:16px!important;
    background:#fbfcfe!important;
}}
.library-card [data-testid="file-upload"]:hover {{
    background:#f6f9fd!important;
    border-color:#8fa6d0!important;
}}
.library-files-card {{
    margin-top:18px!important;
    padding:24px!important;
}}
.library-table-wrap {{
    margin-top:16px!important;
    border-radius:16px!important;
    box-shadow:none!important;
    border:1px solid #edf1f6!important;
}}
.library-table thead th {{
    padding:13px 15px!important;
    background:#f7f9fc!important;
}}
.library-table tbody td {{
    padding:14px 15px!important;
}}
.library-table tbody tr:hover td {{
    background:#fafcff!important;
}}
.library-description {{
    margin-top:4px!important;
    color:#8a96a8!important;
    font-size:11px!important;
}}
.library-icon {{
    width:18px!important;
    height:18px!important;
    border-radius:6px!important;
    background:#edf2fa!important;
    font-size:10px!important;
}}
.library-download-link {{
    min-height:34px!important;
    padding:0 13px!important;
    border-radius:10px!important;
    box-shadow:0 6px 14px rgba(70,97,156,.14)!important;
}}
.library-download-link:hover {{
    filter:brightness(1.10)!important;
    transform:translateY(-1px)!important;
    box-shadow:0 9px 20px rgba(70,97,156,.22)!important;
}}

/* =========================================================
   PAGE DE CONNEXION GRADIO — plein écran, thème bleu du site
   ========================================================= */
.login-container {{
    min-height:100vh!important;
    width:100%!important;
    max-width:none!important;
    display:grid!important;
    place-items:center!important;
    padding:42px 24px!important;
    box-sizing:border-box!important;
    background:
        radial-gradient(circle at 78% 18%, rgba(111,134,189,.28), transparent 28%),
        radial-gradient(circle at 12% 85%, rgba(70,97,156,.16), transparent 30%),
        linear-gradient(135deg,#eef3f8 0%,#e8eef7 52%,#dce6f4 100%)!important;
}}
.login-form {{
    width:min(560px,94vw)!important;
    max-width:560px!important;
    padding:42px!important;
    border:1px solid rgba(255,255,255,.76)!important;
    border-radius:28px!important;
    background:rgba(255,255,255,.92)!important;
    backdrop-filter:blur(18px)!important;
    box-shadow:0 28px 80px rgba(32,48,82,.18)!important;
}}
.login-form > div:first-child {{
    width:100%!important;
}}
.login-form h1,
.login-form h2 {{
    color:#172033!important;
    font-family:'Fraunces',serif!important;
}}
.login-form input {{
    min-height:50px!important;
    border-radius:14px!important;
    border:1px solid #dfe7f2!important;
    background:#f9fbfe!important;
    color:#172033!important;
}}
.login-form input:focus {{
    border-color:#6d86bc!important;
    box-shadow:0 0 0 4px rgba(70,97,156,.12)!important;
}}
.login-form button,
.login-form button.primary {{
    min-height:50px!important;
    border:0!important;
    border-radius:14px!important;
    background:linear-gradient(135deg,#46619c,#6380be)!important;
    color:white!important;
    font-weight:800!important;
    box-shadow:0 12px 26px rgba(70,97,156,.22)!important;
    transition:transform .18s ease,filter .18s ease,box-shadow .18s ease!important;
}}
.login-form button:hover,
.login-form button.primary:hover {{
    background:linear-gradient(135deg,#3d568d,#5875b2)!important;
    filter:brightness(1.06)!important;
    transform:translateY(-2px)!important;
    box-shadow:0 17px 34px rgba(70,97,156,.30)!important;
}}
.auth-brand {{
    display:flex;
    align-items:center;
    gap:14px;
    margin-bottom:30px;
}}
.auth-logo {{
    width:52px;
    height:52px;
    flex:0 0 52px;
    border-radius:17px;
    position:relative;
    background:linear-gradient(135deg,#46619c,#7188bd);
    box-shadow:0 12px 26px rgba(70,97,156,.28);
}}
.auth-logo-ring {{
    position:absolute;
    inset:12px;
    border:3px solid rgba(255,255,255,.84);
    border-radius:50%;
}}
.auth-logo-dot {{
    position:absolute;
    width:7px;
    height:7px;
    border-radius:50%;
    background:white;
}}
.auth-dot-one {{ left:11px; top:22px; }}
.auth-dot-two {{ right:11px; top:15px; }}
.auth-dot-three {{ right:14px; bottom:11px; }}
.auth-title {{
    font:700 23px 'Fraunces',serif;
    color:#172033;
    letter-spacing:-.035em;
}}
.auth-subtitle {{
    margin-top:3px;
    color:#68778f;
    font:500 11px 'Inter',sans-serif;
}}
.auth-heading {{
    margin:0 0 8px;
    color:#172033;
    font:700 34px 'Fraunces',serif;
    letter-spacing:-.045em;
}}
.auth-copy {{
    margin-bottom:24px;
    color:#68778f;
    font:500 13px/1.6 'Inter',sans-serif;
}}
@media(max-width:650px) {{
    .login-form {{
        padding:30px 24px!important;
        border-radius:22px!important;
    }}
    .auth-heading {{ font-size:29px; }}
}}

/* Espace profil et déconnexion */
.profile-page {{ height:calc(100vh - 118px); min-height:0; padding:18px 0; overflow:hidden; box-sizing:border-box; }}
body.profile-view, html.profile-view {{ overflow:hidden!important; height:100%!important; }}
body.profile-view .gradio-container {{ height:100vh!important; overflow:hidden!important; }}
body.profile-view #main-tabs {{ height:calc(100vh - 88px)!important; overflow:hidden!important; }}
body.profile-view #main-tabs > div, body.profile-view #main-tabs [role='tabpanel'] {{ max-height:100%!important; overflow:hidden!important; }}
.profile-shell {{
    height:100%; min-height:0; display:grid; grid-template-columns:1.05fr .95fr;
    overflow:hidden; border-radius:30px; background:white;
    border:1px solid #e6ecf5; box-shadow:0 24px 70px rgba(31,48,82,.12);
}}
.profile-visual {{
    position:relative; overflow:hidden; padding:58px; display:flex; flex-direction:column;
    justify-content:space-between; color:white;
    background:linear-gradient(145deg,#263d70 0%,#46619c 55%,#6d85b9 100%);
}}
.profile-visual-orb {{ position:absolute; border-radius:999px; background:rgba(255,255,255,.10); }}
.orb-one {{ width:330px; height:330px; right:-120px; top:-90px; }}
.orb-two {{ width:190px; height:190px; left:-70px; bottom:-55px; }}
.profile-logo-large {{ width:78px; height:78px; padding:13px; border-radius:24px; background:rgba(255,255,255,.13); backdrop-filter:blur(10px); }}
.profile-logo-large svg {{ width:100%; height:100%; }}
.profile-visual-copy {{ position:relative; z-index:1; max-width:470px; }}
.profile-visual-copy span {{ font-size:11px; font-weight:800; letter-spacing:.18em; color:#dce7ff; }}
.profile-visual-copy h2 {{ margin:14px 0 16px; font:700 42px/1.08 'Fraunces',serif; letter-spacing:-.045em; color:white!important; }}
.profile-visual-copy p {{ margin:0; max-width:420px; font-size:14px; line-height:1.75; color:#e6edfb; }}
.profile-card {{ padding:62px 64px; display:flex; flex-direction:column; justify-content:center; background:#fff; }}
.profile-avatar {{ width:78px; height:78px; display:grid; place-items:center; border-radius:24px; color:white; font:800 25px 'Inter',sans-serif; background:linear-gradient(135deg,#46619c,#7d91bf); box-shadow:0 14px 30px rgba(70,97,156,.25); }}
.profile-kicker {{ margin-top:25px; color:#7b89a0; font-size:11px; font-weight:800; letter-spacing:.15em; text-transform:uppercase; }}
.profile-card h1 {{ margin:8px 0 8px; color:#172033!important; font:700 36px 'Fraunces',serif; letter-spacing:-.04em; }}
.profile-role {{ display:inline-flex; align-self:flex-start; padding:7px 12px; border-radius:999px; color:#46619c; background:#eef3fb; font-size:12px; font-weight:800; }}
.profile-info-list {{ margin-top:31px; border:1px solid #e8edf5; border-radius:18px; overflow:hidden; }}
.profile-info-row {{ display:flex; justify-content:space-between; gap:22px; padding:16px 18px; border-bottom:1px solid #edf1f6; font-size:13px; }}
.profile-info-row:last-child {{ border-bottom:0; }}
.profile-info-row span {{ color:#778398; }}
.profile-info-row strong {{ color:#26334b; }}
.profile-active {{ color:#3f8b5b!important; }}
.profile-divider {{ height:1px; margin:30px 0 23px; background:#edf1f6; }}
.logout-copy {{ display:flex; flex-direction:column; gap:5px; }}
.logout-copy strong {{ color:#172033; font-size:14px; }}
.logout-copy span {{ color:#7a879b; font-size:12px; line-height:1.55; }}
.logout-button {{
    margin-top:18px; min-height:50px; width:100%; border:0; border-radius:14px;
    color:white; background:linear-gradient(135deg,#46619c,#607bb5); font-weight:800;
    box-shadow:0 12px 25px rgba(70,97,156,.20); cursor:pointer; transition:.2s ease;
}}
.logout-button:hover {{ transform:translateY(-2px); filter:brightness(1.06); box-shadow:0 17px 34px rgba(70,97,156,.29); }}
.logout-icon {{ display:inline-block; margin-right:7px; transform:rotate(90deg); }}
@media(max-width:900px) {{ body.profile-view, html.profile-view {{ overflow:auto!important; height:auto!important; }} .profile-page {{ height:auto; min-height:0; overflow:visible; }} .profile-shell {{ height:auto; min-height:610px; grid-template-columns:1fr; }} .profile-visual {{ min-height:300px; padding:38px; }} .profile-card {{ padding:42px 34px; }} }}

/* Verrouillage total de la page Profil sur ordinateur */
@media(min-width:901px) {{
    html:has(.profile-page), body:has(.profile-page) {{
        height:100vh!important;
        min-height:100vh!important;
        overflow:hidden!important;
    }}
    body:has(.profile-page) .gradio-container {{
        height:100vh!important;
        min-height:100vh!important;
        overflow:hidden!important;
    }}
    body:has(.profile-page) [role='tabpanel']:has(.profile-page) {{
        overflow:hidden!important;
        height:calc(100vh - 94px)!important;
        max-height:calc(100vh - 94px)!important;
    }}
    .profile-page {{
        position:fixed!important;
        top:122px!important;
        left:24px!important;
        right:24px!important;
        bottom:18px!important;
        width:auto!important;
        height:auto!important;
        max-height:none!important;
        margin:0!important;
        padding:0!important;
        overflow:hidden!important;
        z-index:20!important;
    }}
    .profile-shell {{
        width:100%!important;
        height:100%!important;
        max-height:100%!important;
        overflow:hidden!important;
    }}
    .profile-visual, .profile-card {{
        min-height:0!important;
        max-height:100%!important;
        overflow:hidden!important;
    }}
}}


/* BIBLIOTHÈQUE — PARCOURS EN 3 ÉTAPES */
.library-heading{{max-width:980px!important;margin-bottom:22px!important}}
.library-wizard-card{{
    max-width:1180px!important;margin:0 auto 28px!important;padding:28px!important;
    border:1px solid #e3eaf4!important;border-radius:28px!important;background:#fff!important;
    box-shadow:0 18px 55px rgba(30,43,70,.07)!important
}}
.library-wizard-progress{{
    display:grid;grid-template-columns:auto 1fr auto 1fr auto;align-items:center;gap:16px;
    margin-bottom:28px;padding:18px 20px;border-radius:20px;background:#f4f7fb
}}
.library-progress-item{{display:flex;align-items:center;gap:11px;color:#7a8598}}
.library-progress-item span{{
    display:grid;place-items:center;width:34px;height:34px;flex:0 0 34px;
    border:1px solid #d9e2ef;border-radius:50%;background:#fff;color:#64748b;font-weight:900
}}
.library-progress-item.active span{{
    border-color:#46619c;background:#46619c;color:#fff;box-shadow:0 8px 20px rgba(70,97,156,.22)
}}
.library-progress-item strong{{display:block;color:#22304a;font-size:12px}}
.library-progress-item small{{display:block;margin-top:2px;color:#8893a5;font-size:10px}}
.library-progress-line{{height:1px;background:#dce4ef}}
.library-step-panel{{padding:6px 4px 2px!important;border:0!important;background:transparent!important}}
.library-step-intro{{display:flex;align-items:flex-start;gap:18px;margin:2px 0 24px}}
.library-step-number{{color:#9aaccc;font:700 34px/1 Georgia,serif;letter-spacing:-.04em}}
.library-step-title{{color:#172033;font:700 25px/1.15 Georgia,serif;letter-spacing:-.035em}}
.library-step-text{{margin-top:7px;color:#758196;font-size:13px;line-height:1.55}}
.library-step-panel .file-preview,
.library-step-panel [data-testid="file-upload"]{{
    min-height:255px!important;border:1px dashed #cfd9e8!important;border-radius:20px!important;background:#fbfcfe!important
}}
.library-step-panel textarea,.library-step-panel input{{
    border-radius:14px!important;border-color:#dde5f0!important;background:#fbfcfe!important;box-shadow:none!important
}}
.library-step-panel textarea:focus,.library-step-panel input:focus{{
    border-color:#7890c0!important;box-shadow:0 0 0 4px rgba(70,97,156,.10)!important
}}
.library-step-actions{{margin-top:18px!important;gap:12px!important}}
.library-step-actions button{{min-height:50px!important;border-radius:14px!important}}
.library-files-card{{max-width:1180px!important;margin:0 auto 40px!important;padding:26px!important}}
.library-list-heading{{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:18px}}
.library-files-title{{margin-top:3px!important;font-size:27px!important}}
@media(max-width:760px){{
    .library-wizard-card,.library-files-card{{padding:18px!important;border-radius:22px!important}}
    .library-wizard-progress{{grid-template-columns:1fr;gap:10px}}
    .library-progress-line{{display:none}}
    .library-progress-item small{{display:none}}
    .library-step-actions{{flex-direction:column!important}}
}}


/* Correctifs Bibliothèque */
.library-progress-item.done span{{
    border-color:#8aa0cb!important;
    background:#eef3fb!important;
    color:#46619c!important;
}}
.library-progress-line.done{{
    background:#8fa4cc!important;
}}
#library_visibility_choice,
#library_visibility_choice *{{
    pointer-events:auto!important;
}}
#library_visibility_choice{{
    position:relative!important;
    z-index:8!important;
}}
#library_visibility_choice label{{
    cursor:pointer!important;
}}
#library_visibility_choice input[type="radio"]{{
    appearance:auto!important;
    -webkit-appearance:radio!important;
    width:17px!important;
    height:17px!important;
    min-height:17px!important;
    opacity:1!important;
    visibility:visible!important;
    pointer-events:auto!important;
    accent-color:#46619c!important;
}}
.library-sort-help{{
    margin:0 0 12px;
    color:#7a8699;
    font-size:12px;
}}
.library-sort-button{{
    display:inline-flex!important;
    align-items:center!important;
    gap:7px!important;
    width:auto!important;
    min-width:0!important;
    min-height:0!important;
    margin:0!important;
    padding:0!important;
    border:0!important;
    border-radius:0!important;
    background:transparent!important;
    color:inherit!important;
    font:inherit!important;
    box-shadow:none!important;
    cursor:pointer!important;
}}
.library-sort-button:hover{{
    color:#46619c!important;
    transform:none!important;
    filter:none!important;
}}
.sort-indicator{{
    color:#8da0c4;
    font-size:12px;
}}


/* Choix de visibilité réellement cliquables */
.visibility-field-label{{
    margin:2px 0 9px;
    color:#536078;
    font-size:13px;
    font-weight:700;
}}
.visibility-button-row{{
    gap:10px!important;
    align-items:stretch!important;
}}
.visibility-choice-button{{
    min-height:48px!important;
    border-radius:13px!important;
    pointer-events:auto!important;
    cursor:pointer!important;
}}
.visibility-choice-button.secondary{{
    background:#fff!important;
    color:#344159!important;
    border:1px solid #dce4ef!important;
    box-shadow:none!important;
}}
.visibility-choice-button.primary{{
    background:#46619c!important;
    color:#fff!important;
    border:1px solid #46619c!important;
}}
.visibility-selection-note{{
    margin:10px 0 8px;
    padding:10px 13px;
    border-radius:11px;
    background:#f3f6fb;
    color:#647086;
    font-size:12px;
}}
.library-sort-button{{
    position:relative!important;
    z-index:20!important;
    pointer-events:auto!important;
    cursor:pointer!important;
    user-select:none!important;
}}
.library-sort-button *{{
    pointer-events:none!important;
}}


/* Ajustements des choix de visibilité */
.visibility-button-row{{
    gap:8px!important;
}}
.visibility-choice-button{{
    min-height:40px!important;
    height:40px!important;
    padding:0 14px!important;
    border-radius:11px!important;
    font-size:12px!important;
    font-weight:700!important;
    box-shadow:none!important;
}}
.visibility-choice-button.primary{{
    box-shadow:0 7px 18px rgba(70,97,156,.16)!important;
}}
.visibility-selection-note{{
    display:none!important;
}}


/* Derniers ajustements — bibliothèque pleine largeur et boutons plus légers */
.library-heading,
.library-wizard-card,
.library-files-card{{
    width:calc(100% - 32px)!important;
    max-width:none!important;
    margin-left:16px!important;
    margin-right:16px!important;
}}
.library-wizard-card,
.library-files-card{{
    box-sizing:border-box!important;
}}
.visibility-button-row{{
    align-items:center!important;
}}
.visibility-choice-button{{
    flex:1 1 0!important;
    min-width:0!important;
    min-height:38px!important;
    height:38px!important;
    padding:0 12px!important;
    border-radius:10px!important;
    font-size:11.5px!important;
    line-height:1!important;
    transform:none!important;
    transition:background-color .16s ease,border-color .16s ease,color .16s ease,box-shadow .16s ease!important;
}}
.visibility-choice-button.secondary{{
    background:#ffffff!important;
    color:#3e4b62!important;
    border:1px solid #d8e1ed!important;
    box-shadow:none!important;
}}
.visibility-choice-button.secondary:hover{{
    background:#f1f5fb!important;
    border-color:#aebfda!important;
    color:#304979!important;
    transform:none!important;
    filter:none!important;
}}
.visibility-choice-button.primary{{
    background:#6f86b8!important;
    color:#ffffff!important;
    border:1px solid #6f86b8!important;
    box-shadow:0 4px 12px rgba(70,97,156,.12)!important;
    transform:none!important;
}}
.visibility-choice-button.primary:hover{{
    background:#647cad!important;
    border-color:#647cad!important;
    transform:none!important;
    filter:none!important;
}}
@media(max-width:760px){{
    .library-heading,
    .library-wizard-card,
    .library-files-card{{
        width:calc(100% - 20px)!important;
        margin-left:10px!important;
        margin-right:10px!important;
    }}
}}


/* Sources bibliothèque dans l'onglet Analyse */
#validate_model_button,
#validate_video_button{{
    margin-top:14px!important;
}}


/* Résultats enregistrés */
.saved-results-heading,
.saved-results-grid{{
    width:calc(100% - 32px)!important;
    max-width:none!important;
    margin-left:16px!important;
    margin-right:16px!important;
}}
.saved-results-grid{{
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
    gap:18px;
}}
.saved-result-card{{
    display:flex;
    flex-direction:column;
    justify-content:space-between;
    gap:22px;
    padding:24px;
    border:1px solid #e0e7f1;
    border-radius:22px;
    background:#fff;
    box-shadow:0 14px 38px rgba(35,49,78,.07);
}}
.saved-result-date{{
    margin-bottom:7px;
    color:#8490a4;
    font-size:11px;
    font-weight:700;
}}
.saved-result-card h3{{
    margin:0;
    color:#172033;
    font:700 23px/1.2 Georgia,serif;
}}
.saved-result-links{{
    display:flex;
    flex-wrap:wrap;
    gap:8px;
}}
.saved-result-link{{
    padding:9px 12px;
    border:1px solid #dbe4f0;
    border-radius:10px;
    background:#f7f9fc;
    color:#46619c!important;
    font-size:11px;
    font-weight:800;
    text-decoration:none!important;
}}
.saved-result-link:hover{{
    background:#edf2fa;
    border-color:#b8c7df;
}}
.saved-results-empty{{
    width:calc(100% - 32px);
    margin:0 16px;
    padding:35px;
    border:1px dashed #cfd9e8;
    border-radius:22px;
    background:#fff;
    color:#758196;
    text-align:center;
}}

/* Sauvegarde des résultats actuels */
.save-results-card{{
    margin:18px 0 26px!important;
    padding:20px!important;
    border:1px solid #dfe7f2!important;
    border-radius:18px!important;
    background:#f7f9fc!important;
}}
.save-results-title{{
    color:#172033;
    font:700 20px/1.2 Georgia,serif;
}}
.save-results-text{{
    margin:5px 0 14px;
    color:#758196;
    font-size:12px;
}}

/* Suppression bibliothèque */
.library-delete-title{{
    margin-top:28px;
    color:#172033;
    font:700 20px/1.2 Georgia,serif;
}}
.library-delete-help{{
    margin:5px 0 13px;
    color:#7b879a;
    font-size:12px;
}}
.library-delete-row{{
    align-items:flex-end!important;
    gap:12px!important;
}}
.library-delete-button{{
    max-width:150px!important;
    min-height:44px!important;
    border:1px solid #e6b8bf!important;
    background:#fff7f8!important;
    color:#a9172b!important;
    border-radius:12px!important;
    box-shadow:none!important;
}}
.library-delete-button:hover{{
    background:#a9172b!important;
    color:#fff!important;
    transform:none!important;
}}


/* Actions directement dans chaque ligne de bibliothèque */
.library-row-actions{{
    display:flex;
    align-items:center;
    gap:8px;
    flex-wrap:nowrap;
}}
.library-inline-delete{{
    min-height:36px!important;
    height:36px!important;
    width:auto!important;
    min-width:0!important;
    margin:0!important;
    padding:0 12px!important;
    border:1px solid #e7bcc3!important;
    border-radius:10px!important;
    background:#fff8f9!important;
    color:#a9172b!important;
    font-size:11px!important;
    font-weight:800!important;
    box-shadow:none!important;
    cursor:pointer!important;
}}
.library-inline-delete:hover{{
    background:#a9172b!important;
    border-color:#a9172b!important;
    color:#fff!important;
    transform:none!important;
    filter:none!important;
}}
.saved-result-report{{
    background:#46619c!important;
    border-color:#46619c!important;
    color:#fff!important;
}}
.saved-result-missing{{
    color:#a9172b;
    font-size:12px;
}}


/* Résultats enregistrés — version fiable avec composants Gradio */
.saved-results-heading,
.saved-result-files-card,
#component-0{{
    max-width:none!important;
}}
.saved-result-files-card{{
    width:calc(100% - 32px)!important;
    margin:18px 16px 40px!important;
    padding:24px!important;
    border:1px solid #dfe7f2!important;
    border-radius:22px!important;
    background:#fff!important;
    box-shadow:0 14px 38px rgba(35,49,78,.06)!important;
}}
.saved-result-summary{{
    width:calc(100% - 32px);
    margin:14px 16px 0;
    padding:18px 20px;
    border-radius:16px;
    background:#eef3fb;
    border:1px solid #dbe5f3;
}}
.saved-result-summary-kicker{{
    color:#46619c;
    font-size:10px;
    font-weight:800;
    letter-spacing:.16em;
    text-transform:uppercase;
}}
.saved-result-summary-title{{
    margin-top:6px;
    color:#172033;
    font:700 24px/1.2 Georgia,serif;
}}
.saved-result-summary-date{{
    margin-top:5px;
    color:#778399;
    font-size:12px;
}}
.saved-results-empty{{
    width:calc(100% - 32px);
    margin:14px 16px;
    padding:30px;
    border:1px dashed #cad6e7;
    border-radius:18px;
    background:#fff;
    color:#78859a;
    text-align:center;
}}


/* Boutons de partage : aucune variation de taille à la sélection */
.visibility-button-row{{
    display:grid!important;
    grid-template-columns:repeat(3,minmax(0,1fr))!important;
    gap:10px!important;
    align-items:stretch!important;
}}
.visibility-button-row > div{{
    width:100%!important;
    min-width:0!important;
    max-width:none!important;
    flex:none!important;
}}
.visibility-choice-button,
.visibility-choice-button.primary,
.visibility-choice-button.secondary,
.visibility-choice-button:hover,
.visibility-choice-button.primary:hover,
.visibility-choice-button.secondary:hover{{
    display:flex!important;
    align-items:center!important;
    justify-content:center!important;
    width:100%!important;
    min-width:0!important;
    max-width:none!important;
    height:40px!important;
    min-height:40px!important;
    max-height:40px!important;
    margin:0!important;
    padding:0 12px!important;
    box-sizing:border-box!important;
    transform:none!important;
    scale:1!important;
    flex:none!important;
    line-height:1!important;
}}
.visibility-choice-button.primary{{
    background:#7b91bf!important;
    border-color:#7b91bf!important;
    box-shadow:0 3px 9px rgba(70,97,156,.10)!important;
}}
.visibility-choice-button.primary:hover{{
    background:#7087b7!important;
    border-color:#7087b7!important;
}}


/* Rapport principal fortement mis en avant */
.saved-report-hero-copy{{
    margin:4px 0 12px;
    padding:22px 24px 18px;
    border-left:7px solid #46619c;
    border-radius:18px 18px 8px 8px;
    background:linear-gradient(135deg,#eef3fb,#f8faff);
}}
.saved-report-hero-kicker{{
    color:#46619c;
    font-size:10px;
    font-weight:900;
    letter-spacing:.18em;
    text-transform:uppercase;
}}
.saved-report-hero-title{{
    margin-top:7px;
    color:#172033;
    font:700 27px/1.15 Georgia,serif;
}}
.saved-report-hero-text{{
    max-width:850px;
    margin-top:7px;
    color:#69768b;
    font-size:13px;
    line-height:1.55;
}}
#saved_report_main_file{{
    margin-top:0!important;
    padding:16px!important;
    border:2px solid #9eb0d1!important;
    border-radius:0 0 18px 18px!important;
    background:#fff!important;
    box-shadow:0 12px 30px rgba(70,97,156,.10)!important;
}}
@media(max-width:760px){{
    .visibility-button-row{{
        grid-template-columns:1fr!important;
    }}
}}


/* Pont technique de suppression : présent dans le DOM mais invisible */
.library-delete-bridge{{
    position:fixed!important;
    left:-10000px!important;
    top:-10000px!important;
    width:1px!important;
    height:1px!important;
    overflow:hidden!important;
    opacity:0!important;
    pointer-events:auto!important;
}}


/* Barre d'actions des résultats enregistrés */
.saved-results-toolbar{{
    width:calc(100% - 32px)!important;
    margin:0 16px!important;
    align-items:flex-end!important;
    gap:12px!important;
}}
.delete-saved-result-button{{
    min-width:190px!important;
    max-width:220px!important;
    min-height:44px!important;
    height:44px!important;
    margin-bottom:1px!important;
    border:1px solid #e3b4bc!important;
    border-radius:12px!important;
    background:#fff7f8!important;
    color:#a9172b!important;
    box-shadow:none!important;
}}
.delete-saved-result-button:hover{{
    background:#a9172b!important;
    border-color:#a9172b!important;
    color:#fff!important;
    transform:none!important;
}}
@media(max-width:760px){{
    .saved-results-toolbar{{
        flex-direction:column!important;
    }}
    .delete-saved-result-button{{
        width:100%!important;
        max-width:none!important;
    }}
}}


/* Suppression des résultats placée tout en bas */
.delete-saved-result-bottom-card{{
    width:calc(100% - 32px)!important;
    margin:0 16px 44px!important;
    padding:20px!important;
    border:1px solid #efd7db!important;
    border-radius:18px!important;
    background:#fffafb!important;
}}
.delete-saved-result-copy{{
    display:flex;
    flex-direction:column;
    gap:5px;
    margin-bottom:14px;
}}
.delete-saved-result-copy strong{{
    color:#7f1725;
    font:700 18px/1.2 Georgia,serif;
}}
.delete-saved-result-copy span{{
    color:#8b6570;
    font-size:12px;
}}
.delete-saved-result-bottom-card .delete-saved-result-button{{
    width:220px!important;
    max-width:220px!important;
    min-width:220px!important;
    height:44px!important;
    min-height:44px!important;
}}


.library-inline-delete-admin{{
    border-color:#d79aa5!important;
    background:#fff1f3!important;
    color:#8f1022!important;
}}
.library-inline-delete-admin:hover{{
    background:#821020!important;
    border-color:#821020!important;
    color:#fff!important;
}}


.topbar,
.menu-wrap,
#app,
.gradio-container,
.gradio-container > .main,
#main-tabs {{
    overflow:visible!important;
}}
.menu-content.open {{
    display:block!important;
}}
@media(max-width:760px) {{
    .menu-content {{
        right:14px;
        top:76px;
        width:min(255px,calc(100vw - 28px));
        max-height:calc(100vh - 92px);
    }}
}}

/* LOGOS - parcours identique a la bibliotheque */
.logo-wizard-card textarea,
.logo-wizard-card input {{
    overflow-x:hidden!important;
    resize:none!important;
    scrollbar-width:none!important;
}}
.logo-wizard-card textarea::-webkit-scrollbar,
.logo-wizard-card input::-webkit-scrollbar {{ display:none!important; }}
.logo-wizard-card .logo-step-panel {{ border:0!important; box-shadow:none!important; background:transparent!important; }}
.logo-wizard-card .logo-step-panel > div {{ border-bottom:0!important; box-shadow:none!important; }}
.logo-wizard-progress {{ margin-bottom:30px!important; }}
.logo-wizard-card .library-step-number {{
    font-family:Georgia, 'Times New Roman', serif!important;
    font-size:36px!important;
    font-weight:700!important;
    line-height:1!important;
    color:#9aaccc!important;
    letter-spacing:-.05em!important;
}}
.logo-wizard-card .library-step-title {{
    font-family:Georgia, 'Times New Roman', serif!important;
    font-size:25px!important;
    font-weight:700!important;
    color:#172033!important;
    letter-spacing:-.035em!important;
}}
.logo-wizard-card .library-step-text {{ margin:7px 0 0!important; color:#758196!important; font-size:13px!important; }}


"""
js = r"""
() => {
  function cleanText(el){ return (el?.innerText || el?.textContent || '').trim().toLowerCase(); }

  window.toggleAppMenu = (e) => {
    if (e) e.stopPropagation();
    const menu = document.getElementById('app-menu-content');
    if (menu) menu.classList.toggle('open');
  };

  window.safeLogout = () => {
    const ok = window.confirm('Voulez-vous vraiment vous déconnecter ?');
    if (ok) window.location.assign('/logout');
  };

  window.goAppTab = (target) => {
    const map = {accueil:'accueil', archives:'résultats enregistrés', bibliotheque:'bibliothèque', entrainement:'entraînement', analyse:'2 · analyse', consolidation:'3 · analyse consolidée', logos:'logos des marques', resultats:'résultats actuels', profil:'profil'};
    const needle = (map[target] || target).toLowerCase();
    const tabs = [...document.querySelectorAll('#main-tabs [role="tab"], #main-tabs button')];
    const btn = tabs.find(b => cleanText(b).includes(needle));
    if (btn) {
      btn.click();
      setTimeout(() => window.scrollTo({top:0, behavior:'smooth'}), 80);
    }
    document.getElementById('app-menu-content')?.classList.remove('open');
  };

  function syncProfileView() {
    const selected = [...document.querySelectorAll('#main-tabs [role="tab"]')]
      .find(el => el.getAttribute('aria-selected') === 'true');
    const onProfile = cleanText(selected).includes('profil');
    document.body.classList.toggle('profile-view', onProfile);
    document.documentElement.classList.toggle('profile-view', onProfile);
  }

  document.addEventListener('click', (event) => {
    if (event.target.closest('#main-tabs [role="tab"], #main-tabs button')) {
      setTimeout(syncProfileView, 80);
    }
  });
  setTimeout(syncProfileView, 250);

  document.addEventListener('click', () => {
    document.getElementById('app-menu-content')?.classList.remove('open');
  });

  document.addEventListener('click', (e) => {
    const menuBtn = e.target.closest('.menu-btn');
    if (menuBtn) { window.toggleAppMenu(e); return; }
    const item = e.target.closest('.menu-item');
    if (item) {
      const label = cleanText(item);
      if (label.includes('accueil')) window.goAppTab('accueil');
      else if (label.includes('bibliothèque') || label.includes('bibliotheque')) window.goAppTab('bibliotheque');
      else if (label.includes('entraîner') || label.includes('entrain')) window.goAppTab('entrainement');
      else if (label.includes('consolid')) window.goAppTab('consolidation');
      else if (label.includes('logo')) window.goAppTab('logos');
      else if (label.includes('analyser') || label.includes('analyse')) window.goAppTab('analyse');
      else if (label.includes('résultats') || label.includes('resultats')) window.goAppTab('archives');
    }
  }, true);

  function animateMetrics(){
    document.querySelectorAll('.animated-metrics').forEach(root=>{
      if(root.dataset.animated === '1') return;
      root.dataset.animated = '1';
      const duration = 2800;
      root.querySelectorAll('.count-up').forEach(el=>{
        const target = parseFloat(el.dataset.target || '0');
        const start = performance.now();
        function tick(now){
          const p = Math.min(1, (now-start)/duration);
          const eased = 1 - Math.pow(1-p, 3);
          el.textContent = (target*eased).toFixed(1) + ' %';
          if(p < 1) requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
      });
      root.querySelectorAll('.metric-track i').forEach(bar=>{
        const target = Math.max(0, Math.min(100, parseFloat(bar.dataset.target || '0')));
        bar.style.width = '0%';
        setTimeout(()=>{ bar.style.width = target + '%'; }, 80);
      });
    });
  }
  setInterval(animateMetrics, 300);

  function syncHomeScrollState(){
    const hero = document.querySelector('.hero');
    if (!hero) { document.body.classList.remove('home-active'); return; }
    const rect = hero.getBoundingClientRect();
    const style = window.getComputedStyle(hero);
    const visible = style.display !== 'none' && rect.width > 0 && rect.height > 0 && rect.bottom > 0 && rect.top < window.innerHeight;
    document.body.classList.toggle('home-active', visible);
  }
  syncHomeScrollState();
  setInterval(syncHomeScrollState, 350);

  function prepareNativeLoginButton(){
    if (!document.querySelector('.auth-screen')) return;

    const form = document.querySelector('form');
    if (!form) return;

    const buttons = Array.from(form.querySelectorAll('button'));
    const loginButton = buttons.find((button) => {
      if (button.closest('.auth-screen')) return false;
      const type = (button.getAttribute('type') || '').toLowerCase();
      return type === 'submit' || button === buttons[buttons.length - 1];
    });

    if (!loginButton) return;

    loginButton.textContent = 'Se connecter';
    loginButton.setAttribute('aria-label', 'Se connecter');
  }

  prepareNativeLoginButton();
  setInterval(prepareNativeLoginButton, 300);

  // Suppression inline d'une ressource appartenant à l'utilisateur.
  document.addEventListener('click', (event) => {
    const deleteButton = event.target.closest('.library-inline-delete');
    if (!deleteButton) return;

    event.preventDefault();
    event.stopPropagation();

    const itemId = deleteButton.dataset.libraryId || '';
    const itemName = deleteButton.dataset.libraryName || 'cette ressource';

    if (!itemId) return;
    if (!window.confirm(`Supprimer définitivement « ${itemName} » ?`)) return;

    const bridgeRoot = document.querySelector('#library_delete_id_bridge');
    const bridge =
      bridgeRoot?.querySelector('textarea') ||
      bridgeRoot?.querySelector('input') ||
      document.querySelector('#library_delete_id_bridge textarea') ||
      document.querySelector('#library_delete_id_bridge input');

    const trigger =
      document.querySelector('#library_delete_trigger_bridge button') ||
      document.querySelector('#library_delete_trigger_bridge');

    if (!bridge || !trigger) {
      console.error('Pont de suppression introuvable.');
      return;
    }

    const prototype =
      bridge.tagName === 'TEXTAREA'
        ? window.HTMLTextAreaElement.prototype
        : window.HTMLInputElement.prototype;

    const setter = Object.getOwnPropertyDescriptor(prototype, 'value')?.set;
    if (setter) setter.call(bridge, itemId);
    else bridge.value = itemId;

    bridge.dispatchEvent(new Event('input', { bubbles: true }));
    bridge.dispatchEvent(new Event('change', { bubbles: true }));

    setTimeout(() => {
      trigger.click();
    }, 180);
  }, true);


  // Suppression directe d'un logo depuis le tableau triable.
  document.addEventListener('click', (event) => {
    const deleteButton = event.target.closest('.logo-inline-delete');
    if (!deleteButton) return;
    event.preventDefault();
    event.stopPropagation();
    const logoId = deleteButton.dataset.logoId || '';
    const logoName = deleteButton.dataset.logoName || 'ce logo';
    if (!logoId || !window.confirm(`Supprimer définitivement « ${logoName} » ?`)) return;
    const bridgeRoot = document.querySelector('#brand_logo_delete_id_bridge');
    const bridge = bridgeRoot?.querySelector('textarea') || bridgeRoot?.querySelector('input');
    const trigger = document.querySelector('#brand_logo_delete_trigger_bridge button') || document.querySelector('#brand_logo_delete_trigger_bridge');
    if (!bridge || !trigger) return;
    const proto = bridge.tagName === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
    if (setter) setter.call(bridge, logoId); else bridge.value = logoId;
    bridge.dispatchEvent(new Event('input', { bubbles: true }));
    bridge.dispatchEvent(new Event('change', { bubbles: true }));
    setTimeout(() => trigger.click(), 160);
  }, true);

  // Tri délégué : fonctionne aussi quand Gradio régénère le tableau.
  document.addEventListener('click', (event) => {
    const button = event.target.closest('.library-sort-button');
    if (!button) return;

    event.preventDefault();
    event.stopPropagation();

    const table = button.closest('table.library-sortable');
    if (!table || !table.tBodies.length) return;

    const column = Number(button.dataset.column || 0);
    const isDate = button.dataset.type === 'date';
    const previousColumn = Number(table.dataset.sortColumn ?? -1);
    const previousDirection = table.dataset.sortDirection || 'desc';
    const direction =
      previousColumn === column && previousDirection === 'asc' ? 'desc' : 'asc';

    const tbody = table.tBodies[0];
    const rows = Array.from(tbody.rows);

    rows.sort((a, b) => {
      let av = a.cells[column]?.dataset.sortValue || a.cells[column]?.innerText || '';
      let bv = b.cells[column]?.dataset.sortValue || b.cells[column]?.innerText || '';

      if (isDate) {
        av = Date.parse(av.replace(' ', 'T')) || 0;
        bv = Date.parse(bv.replace(' ', 'T')) || 0;
      } else {
        av = av.trim().toLocaleLowerCase('fr');
        bv = bv.trim().toLocaleLowerCase('fr');
      }

      if (av < bv) return direction === 'asc' ? -1 : 1;
      if (av > bv) return direction === 'asc' ? 1 : -1;
      return 0;
    });

    rows.forEach((row) => tbody.appendChild(row));
    table.dataset.sortColumn = String(column);
    table.dataset.sortDirection = direction;

    table.querySelectorAll('.sort-indicator').forEach((indicator) => {
      indicator.textContent = '↕';
    });
    const activeIndicator = button.querySelector('.sort-indicator');
    if (activeIndicator) activeIndicator.textContent = direction === 'asc' ? '↑' : '↓';
  }, true);

}
"""
with gr.Blocks(title="Logo Analyzer Pro", css=css, js=js) as demo:
    with gr.Column(elem_id="app"):
        gr.HTML("""<div class="topbar"><div class="brand"><button class="brand-home" onclick="window.goAppTab && window.goAppTab(\'accueil\')" aria-label="Retour à l’accueil"><div class="brand-mark" aria-hidden="true">
<svg viewBox="0 0 64 64" width="38" height="38" role="img">
  <rect x="4" y="4" width="56" height="56" rx="16" fill="#46619c"/>
  <path d="M15 32c5-8 10.8-12 17-12s12 4 17 12c-5 8-10.8 12-17 12S20 40 15 32Z" fill="none" stroke="white" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="32" cy="32" r="6.5" fill="white"/>
  <path d="M12 20v-6h6M46 14h6v6M52 44v6h-6M18 50h-6v-6" fill="none" stroke="#dce7ff" stroke-width="3" stroke-linecap="round"/>
</svg>
</div></button><div class="brand-copy"><div class="brand-title">Analyseur de logos</div><div class="brand-subtitle">Intelligence visuelle pour le sponsoring sportif</div></div></div><div class="menu-wrap"><button class="menu-btn" onclick="toggleAppMenu(event)">☰ &nbsp;Menu</button><div class="menu-content" id="app-menu-content" onclick="event.stopPropagation()"><button class="menu-item" onclick="goAppTab('accueil')"><span class="menu-dot"></span>Accueil</button><button class="menu-item" onclick="goAppTab('entrainement')"><span class="menu-dot"></span>Entraîner un modèle</button><button class="menu-item" onclick="goAppTab('analyse')"><span class="menu-dot"></span>Analyser une vidéo</button><button class="menu-item" onclick="goAppTab('consolidation')"><span class="menu-dot"></span>Analyse consolidée</button><button class="menu-item" onclick="goAppTab('logos')"><span class="menu-dot"></span>Logos des marques</button><button class="menu-item" onclick="goAppTab('archives')"><span class="menu-dot"></span>Résultats</button><button class="menu-item" onclick="goAppTab('bibliotheque')"><span class="menu-dot"></span>Bibliothèque</button><button class="menu-item" onclick="goAppTab('profil')"><span class="menu-dot"></span>Profil</button></div></div></div>""")
        with gr.Tabs(elem_id="main-tabs"):
            with gr.Tab("Accueil"):
                gr.HTML("""<section class="hero"><div class="hero-content"><div class="hero-kicker">Sponsoring intelligence</div><h1>Transformer chaque image en mesure de visibilité.</h1><p>Importez un modèle, analysez une vidéo sportive et obtenez une lecture claire de l’exposition des marques : durée, occupation écran, centralité, part de voix et meilleurs moments.</p><div class="workflow"><div class="workflow-item"><div class="workflow-number">01</div><div class="workflow-title">Préparer</div><div class="workflow-text">Dataset ou modèle prêt à l’emploi.</div></div><div class="workflow-item"><div class="workflow-number">02</div><div class="workflow-title">Analyser</div><div class="workflow-text">Détection vidéo accélérée et automatisée.</div></div><div class="workflow-item"><div class="workflow-number">03</div><div class="workflow-title">Présenter</div><div class="workflow-text">Stats, graphiques et meilleures apparitions.</div></div></div></div></section>""")
            with gr.Tab("Résultats enregistrés") as saved_results_tab:
                gr.HTML("""
                <div class="section-head saved-results-heading">
                  <div class="section-kicker">Historique</div>
                  <div class="section-title">Résultats des analyses enregistrées</div>
                  <div class="section-subtitle">
                    Les fichiers d’une analyse apparaissent uniquement après son enregistrement.
                  </div>
                </div>
                """)

                saved_results_select = gr.Dropdown(
                    label="Analyse enregistrée",
                    choices=[],
                    interactive=True,
                    visible=False,
                )

                with gr.Column(
                    visible=False,
                    elem_classes=["saved-results-content"],
                ) as saved_results_content:
                    saved_result_summary = gr.HTML()

                    with gr.Column(elem_classes=["saved-result-files-card"]):
                        gr.HTML("""
                        <div class="saved-report-hero-copy">
                          <div class="saved-report-hero-kicker">Document principal</div>
                          <div class="saved-report-hero-title">Rapport complet de visibilité</div>
                          <div class="saved-report-hero-text">
                            Synthèse, indicateurs, tableaux et graphiques de l’analyse.
                          </div>
                        </div>
                        """)
                        saved_report_file = gr.File(
                            label="Rapport complet de visibilité",
                            interactive=False,
                            elem_id="saved_report_main_file",
                        )
                        saved_video_file = gr.Video(
                            label="Vidéo annotée",
                            interactive=False,
                            show_download_button=True,
                        )
                        with gr.Row():
                            saved_detections_file = gr.File(
                                label="Détections CSV",
                                interactive=False,
                            )
                            saved_stats_file = gr.File(
                                label="Statistiques CSV",
                                interactive=False,
                            )
                            saved_commercial_file = gr.File(
                                label="Tableau commercial CSV",
                                interactive=False,
                            )

                    with gr.Column(elem_classes=["delete-saved-result-bottom-card"]):
                        gr.HTML("""
                        <div class="delete-saved-result-copy">
                          <strong>Supprimer cette analyse enregistrée</strong>
                          <span>Le rapport, la vidéo et les CSV seront définitivement supprimés.</span>
                        </div>
                        """)
                        delete_saved_result_button = gr.Button(
                            "Supprimer ces résultats",
                            elem_classes=["delete-saved-result-button"],
                        )
                        delete_saved_result_status = gr.HTML()

                saved_results_tab.select(
                    refresh_saved_results,
                    outputs=[
                        saved_results_select,
                        saved_results_content,
                        saved_result_summary,
                        saved_report_file,
                        saved_video_file,
                        saved_detections_file,
                        saved_stats_file,
                        saved_commercial_file,
                    ],
                )

                saved_results_select.change(
                    load_saved_result,
                    inputs=[saved_results_select],
                    outputs=[
                        saved_results_content,
                        saved_result_summary,
                        saved_report_file,
                        saved_video_file,
                        saved_detections_file,
                        saved_stats_file,
                        saved_commercial_file,
                    ],
                    queue=False,
                )

                delete_saved_result_button.click(
                    delete_saved_result,
                    inputs=[saved_results_select],
                    outputs=[
                        delete_saved_result_status,
                        saved_results_select,
                        saved_results_content,
                        saved_result_summary,
                        saved_report_file,
                        saved_video_file,
                        saved_detections_file,
                        saved_stats_file,
                        saved_commercial_file,
                    ],
                    queue=False,
                )

            with gr.Tab("Bibliothèque") as library_tab:
                gr.HTML("""
                <div class="section-head library-heading">
                  <div class="section-kicker">Ressources partagées</div>
                  <div class="section-title">Bibliothèque de datasets et de modèles</div>
                  <div class="section-subtitle">Ajoutez une ressource en trois étapes simples, puis retrouvez tous les fichiers accessibles juste en dessous.</div>
                </div>
                """)

                with gr.Column(elem_classes=["library-wizard-card"]):
                    library_progress = gr.HTML(library_progress_html(1))
                    library_step_status = gr.HTML()

                    with gr.Column(visible=True, elem_classes=["library-step-panel"]) as library_step_1:
                        gr.HTML("""
                        <div class="library-step-intro">
                          <div class="library-step-number">01</div>
                          <div><div class="library-step-title">Choisir le fichier</div><div class="library-step-text">Importez un dataset ZIP, un modèle YOLO ou une vidéo de match.</div></div>
                        </div>
                        """)
                        library_upload = gr.File(
                            label="Dataset ZIP, modèle PT ou vidéo de match",
                            file_types=[".zip", ".pt", ".mp4", ".mov", ".avi", ".mkv"],
                        )
                        library_next_1 = gr.Button("Continuer vers les informations →", variant="primary")

                    with gr.Column(visible=False, elem_classes=["library-step-panel"]) as library_step_2:
                        gr.HTML("""
                        <div class="library-step-intro">
                          <div class="library-step-number">02</div>
                          <div><div class="library-step-title">Présenter la ressource</div><div class="library-step-text">Donnez-lui un nom clair afin de la retrouver facilement plus tard.</div></div>
                        </div>
                        """)
                        library_name = gr.Textbox(label="Nom affiché", placeholder="Ex. Dataset Volkswagen ou Modèle sponsors 2026")
                        library_description = gr.Textbox(label="Description", placeholder="Décrivez brièvement les logos ou le contenu du fichier.", lines=4)
                        with gr.Row(elem_classes=["library-step-actions"]):
                            library_back_2 = gr.Button("← Retour")
                            library_next_2 = gr.Button("Continuer vers le partage →", variant="primary")

                    with gr.Column(visible=False, elem_classes=["library-step-panel"]) as library_step_3:
                        gr.HTML("""
                        <div class="library-step-intro">
                          <div class="library-step-number">03</div>
                          <div><div class="library-step-title">Choisir la visibilité</div><div class="library-step-text">Gardez le fichier privé ou partagez-le avec les autres comptes.</div></div>
                        </div>
                        """)
                        gr.HTML("<div class='visibility-field-label'>Visibilité</div>")
                        library_visibility = gr.Textbox(
                            value="Privé — seulement moi",
                            visible=False,
                        )
                        with gr.Row(elem_classes=["visibility-button-row"]):
                            visibility_private_button = gr.Button(
                                "Privé — seulement moi",
                                variant="primary",
                                elem_classes=["visibility-choice-button"],
                            )
                            visibility_selected_button = gr.Button(
                                "Partagé — utilisateurs choisis",
                                variant="secondary",
                                elem_classes=["visibility-choice-button"],
                            )
                        library_shared_users = gr.Textbox(label="Utilisateurs autorisés", placeholder="Ex. client1, client2 — seulement pour « utilisateurs choisis »")
                        with gr.Row(elem_classes=["library-step-actions"]):
                            library_back_3 = gr.Button("← Retour")
                            library_save_button = gr.Button("Enregistrer dans la bibliothèque", variant="primary")
                        library_save_status = gr.HTML()

                with gr.Column(elem_classes=["library-card", "library-files-card"]):
                    gr.HTML("""
                    <div class="library-list-heading">
                      <div>
                        <div class="section-kicker">Vos ressources</div>
                        <div class="section-title library-files-title">Fichiers accessibles</div>
                        <div class="section-subtitle">La liste se met à jour automatiquement après chaque ajout.</div>
                      </div>
                    </div>
                    """)
                    library_table = gr.HTML()
                    library_delete_id = gr.Textbox(
                        value="",
                        visible=True,
                        elem_id="library_delete_id_bridge",
                        elem_classes=["library-delete-bridge"],
                    )
                    library_delete_trigger = gr.Button(
                        "Supprimer la ressource sélectionnée",
                        visible=True,
                        elem_id="library_delete_trigger_bridge",
                        elem_classes=["library-delete-bridge"],
                    )
                    library_delete_status = gr.HTML()




                # Composants d'administration conservés mais masqués afin de ne pas
                # créer une bande blanche inutile sous le tableau.
                with gr.Column(visible=False):
                    admin_username_input = gr.Textbox(label="Identifiant du compte")
                    admin_password_input = gr.Textbox(label="Nouveau mot de passe", type="password")
                    admin_make_admin = gr.Checkbox(label="Compte administrateur", value=False)
                    admin_save_user_button = gr.Button("Créer le compte")
                    admin_toggle_user_button = gr.Button("Activer / désactiver")
                    admin_user_status = gr.HTML()
                    admin_users_table = gr.HTML()

            with gr.Tab("Profil") as profile_tab:
                profile_html = gr.HTML()

            with gr.Tab("1 · Entraînement") as training_tab:
                gr.HTML('<div class="section-head"><div class="section-kicker">Étape 1</div><div class="section-title">Importer le dataset</div><div class="section-subtitle">Ajoutez un ZIP Roboflow au format YOLOv8.</div></div>')
                with gr.Column(elem_classes=["step-card"]):
                    gr.HTML('<div class="file-help">Vous pouvez importer un nouveau ZIP ou réutiliser un dataset déjà enregistré dans la bibliothèque.</div>')
                    training_zip_source = gr.Radio(
                        choices=[
                            "Importer un nouveau ZIP",
                            "Choisir dans la bibliothèque",
                        ],
                        value="Importer un nouveau ZIP",
                        label="Source du dataset",
                    )
                    with gr.Column(visible=True) as training_upload_group:
                        zip_input=gr.File(label="Nouveau fichier ZIP", file_types=[".zip"])
                    with gr.Column(visible=False) as training_library_group:
                        train_library_zip_select = gr.Dropdown(
                            label="Dataset disponible dans la bibliothèque",
                            choices=[],
                        )
                    validate_zip_button=gr.Button("Valider le dataset →", elem_id="validate_zip_button")
                    zip_status=gr.HTML()
                with gr.Column(visible=False, elem_classes=["step-card"]) as train_settings_group:
                    gr.HTML('<div class="section-kicker">Étape 2</div><div class="section-title">Paramètres d’entraînement</div><div class="section-subtitle">Le mode rapide est sélectionné par défaut.</div>')
                    epochs_input=gr.Slider(1,100,value=10,step=1,label="Nombre d'epochs")
                    size_input=gr.Dropdown(choices=[("416 · rapide",416),("640 · équilibré",640),("800 · précision",800)],value=640,label="Taille image",allow_custom_value=False,filterable=False)
                    train_button=gr.Button("Lancer l'entraînement",variant="primary",elem_id="train_button")
                    stop_train_button=gr.Button("Arrêter l'entraînement",elem_id="stop_train_button",visible=False)
                train_status=gr.HTML()
                train_metrics=gr.HTML()
                with gr.Column(visible=False) as trained_model_group:
                    gr.HTML('<div class="section-head"><div class="section-title">Modèle entraîné</div><div class="section-subtitle">Le fichier .pt généré apparaîtra ici.</div></div>')
                    trained_model=gr.File(label="")
                training_zip_source.change(
                    lambda source: (
                        gr.update(visible=source == "Importer un nouveau ZIP"),
                        gr.update(visible=source == "Choisir dans la bibliothèque"),
                    ),
                    inputs=[training_zip_source],
                    outputs=[training_upload_group, training_library_group],
                    queue=False,
                )
                training_tab.select(
                    refresh_training_library,
                    outputs=[train_library_zip_select],
                )
                validate_zip_button.click(
                    validate_training_dataset,
                    inputs=[
                        training_zip_source,
                        zip_input,
                        train_library_zip_select,
                    ],
                    outputs=[
                        zip_status,
                        zip_input,
                        train_settings_group,
                        validate_zip_button,
                    ],
                    queue=False,
                )
                train_button.click(train_model,inputs=[zip_input,epochs_input,size_input],outputs=[train_status,trained_model,train_metrics,trained_model_group,stop_train_button,train_button],show_progress="hidden")
                stop_train_button.click(stop_training,outputs=train_status,queue=False)
            with gr.Tab("2 · Analyse") as analysis_tab:
                gr.HTML('<div class="section-head"><div class="section-kicker">Étape 1</div><div class="section-title">Importer le modèle</div><div class="section-subtitle">Importez un modèle .pt ou choisissez-en un dans la bibliothèque.</div></div>')

                with gr.Column(elem_classes=["step-card"]):
                    gr.HTML('<div class="file-help">Vous pouvez importer un nouveau modèle ou réutiliser un modèle déjà enregistré dans la bibliothèque.</div>')

                    model_source = gr.Radio(
                        choices=[
                            "Importer un nouveau modèle",
                            "Choisir dans la bibliothèque",
                        ],
                        value="Importer un nouveau modèle",
                        label="Source du modèle",
                    )

                    with gr.Column(visible=True) as model_upload_group:
                        model_input = gr.File(
                            label="Nouveau modèle YOLO",
                            file_types=[".pt"],
                        )

                    with gr.Column(visible=False) as model_library_group:
                        model_library_select = gr.Dropdown(
                            label="Modèle disponible dans la bibliothèque",
                            choices=[],
                        )

                    validate_model_button = gr.Button(
                        "Valider le modèle →",
                        elem_id="validate_model_button",
                    )
                    model_status = gr.HTML()

                with gr.Column(visible=False, elem_classes=["step-card"]) as video_step_group:
                    gr.HTML('<div class="section-kicker">Étape 2</div><div class="section-title">Importer la vidéo</div><div class="section-subtitle">Importez une vidéo ou choisissez un match déjà enregistré dans la bibliothèque.</div>')
                    gr.HTML('<div class="file-help">Formats acceptés : MP4, MOV, AVI, MKV · Taille maximale : 10 Go</div>')

                    video_source = gr.Radio(
                        choices=[
                            "Importer une nouvelle vidéo",
                            "Choisir dans la bibliothèque",
                        ],
                        value="Importer une nouvelle vidéo",
                        label="Source de la vidéo",
                    )

                    with gr.Column(visible=True) as video_upload_group:
                        video_input = gr.File(
                            label="Nouvelle vidéo de match",
                            file_types=[".mp4", ".mov", ".avi", ".mkv"],
                        )

                    with gr.Column(visible=False) as video_library_group:
                        video_library_select = gr.Dropdown(
                            label="Vidéo disponible dans la bibliothèque",
                            choices=[],
                        )

                    validate_video_button = gr.Button(
                        "Valider la vidéo →",
                        elem_id="validate_video_button",
                    )
                    video_status = gr.HTML()

                with gr.Column(visible=False, elem_classes=["step-card"]) as analyze_settings_group:
                    gr.HTML('<div class="section-kicker">Étape 3</div><div class="section-title">Lancer l’analyse</div><div class="section-subtitle">Réglages rapides par défaut ; augmentez la précision seulement si nécessaire.</div>')
                    confidence_input = gr.Slider(.05, .95, value=.35, step=.05, label="Seuil de confiance")
                    frame_skip_input = gr.Slider(1, 10, value=1, step=1, label="Analyser 1 frame sur...")
                    analysis_size_input = gr.Dropdown(
                        choices=[
                            ("416 · rapide", 416),
                            ("640 · équilibré", 640),
                            ("800 · précision", 800),
                        ],
                        value=640,
                        label="Taille image",
                        allow_custom_value=False,
                        filterable=False,
                    )
                    brand_selector = gr.CheckboxGroup(
                        choices=[],
                        value=[],
                        label="Marques à analyser",
                        info="Choisissez uniquement les marques à conserver dans la vidéo, les statistiques et le rapport.",
                    )
                    brand_selector_status = gr.HTML()
                    analyze_button = gr.Button(
                        "Lancer l'analyse",
                        variant="primary",
                        elem_id="analyze_button",
                    )
                    stop_analyze_button = gr.Button(
                        "Arrêter l'analyse",
                        elem_id="stop_analyze_button",
                        visible=False,
                    )
                    see_results_button = gr.Button(
                        "Voir les résultats →",
                        elem_id="see_results_button",
                        visible=False,
                    )

                analyze_status = gr.HTML()

                model_source.change(
                    lambda source: (
                        gr.update(visible=source == "Importer un nouveau modèle"),
                        gr.update(visible=source == "Choisir dans la bibliothèque"),
                    ),
                    inputs=[model_source],
                    outputs=[model_upload_group, model_library_group],
                    queue=False,
                )

                video_source.change(
                    lambda source: (
                        gr.update(visible=source == "Importer une nouvelle vidéo"),
                        gr.update(visible=source == "Choisir dans la bibliothèque"),
                    ),
                    inputs=[video_source],
                    outputs=[video_upload_group, video_library_group],
                    queue=False,
                )

                analysis_tab.select(
                    refresh_analysis_libraries,
                    outputs=[model_library_select, video_library_select],
                )

                validate_model_button.click(
                    validate_analysis_model,
                    inputs=[
                        model_source,
                        model_input,
                        model_library_select,
                    ],
                    outputs=[
                        model_status,
                        model_input,
                        video_step_group,
                        validate_model_button,
                    ],
                    queue=False,
                )
                validate_model_button.click(
                    load_model_brand_choices,
                    inputs=[model_input],
                    outputs=[brand_selector, brand_selector_status],
                    queue=False,
                )

                validate_video_button.click(
                    validate_analysis_video,
                    inputs=[
                        video_source,
                        video_input,
                        video_library_select,
                    ],
                    outputs=[
                        video_status,
                        video_input,
                        analyze_settings_group,
                        validate_video_button,
                    ],
                    queue=False,
                )


            with gr.Tab("3 · Analyse consolidée") as consolidation_tab:
                gr.HTML('<div class="section-head"><div class="section-kicker">Pilotage annuel</div><div class="section-title">Compiler plusieurs analyses</div><div class="section-subtitle">Choisissez uniquement les matchs et les marques que vous souhaitez comparer.</div></div>')
                with gr.Row():
                    consolidation_year = gr.Dropdown(label="Année", choices=[])
                    consolidation_brands = gr.CheckboxGroup(label="Marques", choices=[], value=[])
                consolidation_matches = gr.CheckboxGroup(label="Matchs à inclure", choices=[], value=[], info="Sélectionnez seulement les 2, 3 ou autres matchs à comparer.")
                consolidation_generate = gr.Button("Générer les statistiques consolidées", variant="primary")
                consolidation_status = gr.HTML()
                with gr.Column(visible=False) as consolidation_results_group:
                    with gr.Column(elem_classes=["consolidated-report-highlight"]):
                        gr.HTML("<div class='report-highlight-copy'><strong>Rapport consolidé complet</strong><span>Ouvrez le rapport HTML pour consulter l’analyse, zoomer sur les graphiques et l’exporter en PDF.</span></div>")
                        consolidation_report = gr.File(label="Rapport HTML complet — à ouvrir en priorité", interactive=False)
                    consolidation_table = gr.HTML()
                    with gr.Row():
                        consolidation_chart = gr.Image(label="Classement cumulé", type="filepath")
                        consolidation_sov_chart = gr.Image(label="Part de voix moyenne", type="filepath")
                    with gr.Row():
                        consolidation_trend_chart = gr.Image(label="Évolution par fichier / match", type="filepath")
                        consolidation_matches_chart = gr.Image(label="Comparaison des matchs", type="filepath")
                    consolidation_csv = gr.File(label="CSV consolidé", interactive=False)
                consolidation_tab.select(refresh_consolidation_filters, outputs=[consolidation_year, consolidation_brands, consolidation_matches, consolidation_status])
                consolidation_year.change(update_consolidation_matches, inputs=[consolidation_year, consolidation_brands], outputs=[consolidation_matches])
                consolidation_brands.change(update_consolidation_matches, inputs=[consolidation_year, consolidation_brands], outputs=[consolidation_matches])
                consolidation_generate.click(generate_consolidated_report, inputs=[consolidation_year, consolidation_brands, consolidation_matches], outputs=[consolidation_status, consolidation_table, consolidation_chart, consolidation_sov_chart, consolidation_trend_chart, consolidation_matches_chart, consolidation_csv, consolidation_report, consolidation_results_group])
            with gr.Tab("Logos des marques") as brand_logos_tab:
                gr.HTML("<div class='section-head'><div class='section-kicker'>Identité des marques</div><div class='section-title'>Bibliothèque de logos</div><div class='section-subtitle'>Ajoutez un logo en trois étapes, puis gérez et triez les logos accessibles comme dans la bibliothèque.</div></div>")
                with gr.Column(elem_classes=["library-wizard-card", "logo-wizard-card"]):
                    brand_logo_status = gr.HTML()
                    logo_progress = gr.HTML(logo_progress_html(1))
                    with gr.Column(visible=True, elem_classes=["library-step-panel", "logo-step-panel"]) as logo_step_1:
                        gr.HTML('<div class="library-step-intro"><span class="library-step-number">01</span><div><strong class="library-step-title">Choisir le fichier</strong><p class="library-step-text">Importez une image PNG, JPG ou WEBP du logo.</p></div></div>')
                        brand_logo_upload = gr.File(label="Image du logo", file_types=[".png", ".jpg", ".jpeg", ".webp"])
                        logo_next_1 = gr.Button("Continuer", variant="primary")
                    with gr.Column(visible=False, elem_classes=["library-step-panel", "logo-step-panel"]) as logo_step_2:
                        gr.HTML('<div class="library-step-intro"><span class="library-step-number">02</span><div><strong class="library-step-title">Informations</strong><p class="library-step-text">Écrivez exactement le nom utilisé par la classe YOLO.</p></div></div>')
                        brand_logo_name = gr.Textbox(label="Nom exact de la marque", placeholder="Ex. Nike")
                        with gr.Row():
                            logo_back_1 = gr.Button("← Retour")
                            logo_next_2 = gr.Button("Continuer", variant="primary")
                    with gr.Column(visible=False, elem_classes=["library-step-panel", "logo-step-panel"]) as logo_step_3:
                        gr.HTML('<div class="library-step-intro"><span class="library-step-number">03</span><div><strong class="library-step-title">Choisir la visibilité</strong><p class="library-step-text">Gardez le logo privé ou partagez-le avec certains comptes.</p></div></div>')
                        brand_logo_visibility = gr.Radio(
                            choices=["Privé — seulement moi", "Partagé — utilisateurs choisis"],
                            value="Privé — seulement moi",
                            label="Visibilité du logo",
                        )
                        brand_logo_shared_users = gr.Textbox(
                            label="Utilisateurs autorisés",
                            placeholder="Ex. client, partenaire2",
                            info="Séparez plusieurs identifiants par une virgule.",
                            visible=False,
                        )
                        with gr.Row():
                            logo_back_2 = gr.Button("← Retour")
                            brand_logo_save = gr.Button("Enregistrer le logo", variant="primary")

                with gr.Column(elem_classes=["library-card", "library-files-card"]):
                    gr.HTML('<div class="section-title library-files-title" style="font-size:24px">Logos accessibles</div><div class="section-subtitle">Cliquez sur les en-têtes pour trier. Le bouton Supprimer apparaît uniquement pour vos logos.</div>')
                    brand_logo_gallery = gr.HTML()

                brand_logo_delete_id_bridge = gr.Textbox(visible=False, elem_id="brand_logo_delete_id_bridge")
                brand_logo_delete_trigger_bridge = gr.Button("Supprimer", visible=False, elem_id="brand_logo_delete_trigger_bridge")

                brand_logos_tab.select(refresh_brand_logo_manager, outputs=[brand_logo_gallery])
                logo_next_1.click(logo_go_to_step_2, inputs=[brand_logo_upload], outputs=[brand_logo_status, logo_progress, logo_step_1, logo_step_2, logo_step_3])
                logo_next_2.click(logo_go_to_step_3, inputs=[brand_logo_name], outputs=[brand_logo_status, logo_progress, logo_step_1, logo_step_2, logo_step_3])
                logo_back_1.click(logo_back_to_step_1, outputs=[brand_logo_status, logo_progress, logo_step_1, logo_step_2, logo_step_3])
                logo_back_2.click(logo_back_to_step_2, outputs=[brand_logo_status, logo_progress, logo_step_1, logo_step_2, logo_step_3])
                brand_logo_visibility.change(brand_logo_visibility_changed, inputs=[brand_logo_visibility], outputs=[brand_logo_shared_users])
                brand_logo_save.click(
                    save_brand_logo,
                    inputs=[brand_logo_upload, brand_logo_name, brand_logo_visibility, brand_logo_shared_users],
                    outputs=[brand_logo_status, brand_logo_gallery, logo_step_3],
                )
                brand_logo_delete_trigger_bridge.click(
                    delete_brand_logo_by_id,
                    inputs=[brand_logo_delete_id_bridge],
                    outputs=[brand_logo_status, brand_logo_gallery],
                )
            with gr.Tab("4 · Résultats actuels"):
                gr.HTML('<div class="section-head results-intro"><div class="section-kicker">Résultats</div><div class="section-title">Synthèse de la visibilité des marques</div><div class="section-subtitle">Commencez par le rapport complet, puis consultez les indicateurs et les graphiques de détail.</div></div>')
                kpis_html=gr.HTML()

                gr.HTML('<div class="report-feature-title">Rapport principal</div><div class="report-feature-subtitle">Le document le plus important : il regroupe la synthèse, les tableaux et les graphiques exportables en PDF.</div>')
                with gr.Column(elem_classes=["report-download-card"]):
                    report_file=gr.File(label="Rapport HTML complet", interactive=False)

                with gr.Column(elem_classes=["save-results-card"]):
                    gr.HTML("""
                    <div class="save-results-title">Conserver cette analyse</div>
                    <div class="save-results-text">
                      Donnez un nom aux résultats pour les retrouver dans la page « Résultats » du menu.
                    </div>
                    """)
                    saved_result_title = gr.Textbox(
                        label="Nom des résultats",
                        placeholder="Ex. Match PSG – Marseille · 11 juillet 2026",
                    )
                    save_results_button = gr.Button(
                        "Enregistrer les résultats",
                        variant="primary",
                    )
                    save_results_status = gr.HTML()

                gr.HTML('<div class="results-section-title">1 · Indicateurs détaillés</div>')
                stats_table=gr.HTML()
                gr.HTML('<div class="results-downloads-title">Fichiers CSV complémentaires</div>')
                with gr.Row():
                    detections_file=gr.File(label="CSV détections complètes")
                    stats_file=gr.File(label="CSV statistiques techniques")
                    commercial_file=gr.File(label="CSV tableau commercial")

                gr.HTML('<div class="results-section-title">2 · Comparaison des performances</div>')
                with gr.Row():
                    with gr.Column(elem_classes=["chart-card"]):
                        gr.HTML('<div class="chart-title">Classement par temps visible</div>')
                        ranking_chart=gr.Image(show_label=False,type="filepath")
                    with gr.Column(elem_classes=["chart-card"]):
                        gr.HTML('<div class="chart-title">Part de voix visuelle</div>')
                        sov_chart=gr.Image(show_label=False,type="filepath")

                with gr.Row():
                    with gr.Column(elem_classes=["chart-card"]):
                        gr.HTML('<div class="chart-title">Timeline des apparitions</div>')
                        timeline_chart=gr.Image(show_label=False,type="filepath")
                    with gr.Column(elem_classes=["chart-card"]):
                        gr.HTML('<div class="chart-title">Occupation écran dans le temps</div>')
                        occupation_chart=gr.Image(show_label=False,type="filepath")

                gr.HTML('<div class="results-section-title">3 · Positionnement dans l’image</div>')
                with gr.Column(elem_classes=["chart-card"]):
                    heatmap_chart=gr.Image(show_label=False,type="filepath")

                gr.HTML('<div class="results-section-title">4 · Vérification vidéo</div>')
                annotated_video=gr.Video(
                    label="Vidéo annotée",
                    interactive=False,
                    show_download_button=True,
                )
                analyze_button.click(analyze_video,inputs=[model_input,video_input,confidence_input,frame_skip_input,analysis_size_input,brand_selector],outputs=[analyze_status,annotated_video,detections_file,stats_file,commercial_file,stats_table,kpis_html,ranking_chart,sov_chart,timeline_chart,occupation_chart,heatmap_chart,report_file,stop_analyze_button,analyze_button,see_results_button],show_progress="hidden")
                analyze_button.click(
                    lambda: (
                        gr.update(visible=True),
                        "",
                        "",
                    ),
                    outputs=[
                        save_results_button,
                        saved_result_title,
                        save_results_status,
                    ],
                    queue=False,
                )
                save_results_button.click(
                    save_current_results,
                    inputs=[
                        saved_result_title,
                        report_file,
                        annotated_video,
                        detections_file,
                        stats_file,
                        commercial_file,
                    ],
                    outputs=[
                        save_results_status,
                        saved_results_select,
                        saved_result_title,
                        save_results_button,
                    ],
                    queue=False,
                )
                stop_analyze_button.click(stop_analysis,outputs=analyze_status,queue=False)
                see_results_button.click(None, None, None, js="() => { window.goAppTab && window.goAppTab('resultats'); }")


                library_tab.select(
                    refresh_library_tab,
                    outputs=[library_table],
                )

                library_delete_trigger.click(
                    delete_owned_library_item,
                    inputs=[library_delete_id],
                    outputs=[
                        library_delete_status,
                        library_table,
                        library_delete_id,
                    ],
                    queue=False,
                )

                visibility_private_button.click(
                    lambda: select_library_visibility("private"),
                    inputs=None,
                    outputs=[
                        library_visibility,
                        visibility_private_button,
                        visibility_selected_button,
                    ],
                    queue=False,
                )
                visibility_selected_button.click(
                    lambda: select_library_visibility("selected"),
                    inputs=None,
                    outputs=[
                        library_visibility,
                        visibility_private_button,
                        visibility_selected_button,
                    ],
                    queue=False,
                )

                library_next_1.click(
                    library_go_to_step_2,
                    inputs=[library_upload],
                    outputs=[library_step_status, library_progress, library_step_1, library_step_2, library_step_3],
                )
                library_next_2.click(
                    library_go_to_step_3,
                    inputs=[library_name],
                    outputs=[library_step_status, library_progress, library_step_1, library_step_2, library_step_3],
                )
                library_back_2.click(
                    library_back_to_step_1,
                    outputs=[library_step_status, library_progress, library_step_1, library_step_2, library_step_3],
                )
                library_back_3.click(
                    library_back_to_step_2,
                    outputs=[library_step_status, library_progress, library_step_1, library_step_2, library_step_3],
                )

                library_save_button.click(
                    save_library_item_wizard,
                    inputs=[
                        library_upload,
                        library_name,
                        library_visibility,
                        library_shared_users,
                        library_description,
                    ],
                    outputs=[
                        library_save_status,
                        library_table,
                        admin_users_table,
                        library_progress,
                        library_step_1,
                        library_step_2,
                        library_step_3,
                        library_upload,
                        library_name,
                        library_description,
                        library_visibility,
                        library_shared_users,
                        visibility_private_button,
                        visibility_selected_button,
                    ],
                )
                library_tab.select(
                    refresh_library,
                    outputs=[
                        library_table,
                        admin_users_table,
                    ],
                )
                admin_save_user_button.click(
                    create_or_update_user,
                    inputs=[admin_username_input, admin_password_input, admin_make_admin],
                    outputs=[admin_user_status, admin_users_table],
                )
                admin_toggle_user_button.click(
                    toggle_user_access,
                    inputs=[admin_username_input],
                    outputs=[admin_user_status, admin_users_table],
                )

    # Le callback de chargement doit être déclaré pendant la construction du Blocks.
    # Placé ici, il peut remplir la page Profil avec l’utilisateur authentifié.
    demo.load(render_profile_html, inputs=None, outputs=profile_html)


# ============================================================
# RAPPORT PREMIUM HTML EXPORTABLE EN PDF
# Cette fonction garde le même nom que l'ancienne génération PDF
# pour ne pas casser le reste de l'application.
# Elle génère maintenant un fichier HTML autonome, beaucoup plus esthétique,
# que l'on peut ouvrir puis exporter en PDF depuis le navigateur.
# ============================================================
def make_report_pdf_file(commercial, chart_paths, output_dir):
    import base64
    import html
    from pathlib import Path

    output_dir = Path(output_dir)
    path = output_dir / "rapport_visibilite_marques.html"

    def esc(value):
        return html.escape(str(value))

    def fmt(value, n=1, suffix=""):
        try:
            return f"{float(value):.{n}f}{suffix}"
        except Exception:
            return f"{value}{suffix}"

    def image_to_data_uri(img_path):
        if not img_path or not Path(img_path).exists():
            return ""
        p = Path(img_path)
        mime = "image/png"
        if p.suffix.lower() in [".jpg", ".jpeg"]:
            mime = "image/jpeg"
        data = base64.b64encode(p.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"

    def chart_uri(title_contains):
        for title, pth in chart_paths:
            if title_contains.lower() in title.lower():
                return image_to_data_uri(pth)
        return ""

    has_data = commercial is not None and not commercial.empty
    if has_data:
        commercial = commercial.copy()
        top = commercial.sort_values("Temps visible (s)", ascending=False).iloc[0]
        by_sov = commercial.sort_values("Part de voix (%)", ascending=False).iloc[0]
        logo = str(top.get("Logo", "-"))
        temps = float(top.get("Temps visible (s)", 0))
        pct_video = float(top.get("% vidéo", 0))
        nb_seq = float(top.get("Nb de séquences", 0))
        duree_max = float(top.get("Durée max. séquence (s)", 0))
        occ_max = float(top.get("Occupation max (%)", 0))
        centralite = float(top.get("Centralité (%)", 0))
        nettete = float(top.get("Netteté moy.", 0))
        sov_logo = str(by_sov.get("Logo", logo))
        sov = float(by_sov.get("Part de voix (%)", 0))
    else:
        commercial = pd.DataFrame()
        logo = sov_logo = "-"
        temps = pct_video = nb_seq = duree_max = occ_max = centralite = nettete = sov = 0

    visibility_cols = [c for c in ["Logo", "Temps visible (s)", "% vidéo", "Nb de séquences", "Durée moy. séquence (s)", "Durée max. séquence (s)"] if has_data and c in commercial.columns]
    quality_cols = [c for c in ["Logo", "Occupation moy. (%)", "Occupation max (%)", "Centralité (%)", "Part de voix (%)", "Netteté moy."] if has_data and c in commercial.columns]

    def table_html(cols):
        if not has_data or not cols:
            return "<p class='empty'>Aucune donnée à afficher.</p>"
        rows = []
        rows.append("<thead><tr>" + "".join(f"<th>{esc(c)}</th>" for c in cols) + "</tr></thead>")
        body = []
        for _, row in commercial[cols].head(12).iterrows():
            body.append("<tr>" + "".join(f"<td>{esc(row[c])}</td>" for c in cols) + "</tr>")
        rows.append("<tbody>" + "".join(body) + "</tbody>")
        return "<table>" + "".join(rows) + "</table>"

    def brand_list(names, max_items=3):
        names = [str(x) for x in names if str(x).strip()]
        if not names:
            return ""
        if len(names) == 1:
            return names[0]
        if len(names) <= max_items:
            return ", ".join(names[:-1]) + " et " + names[-1]
        return ", ".join(names[:max_items]) + f" et {len(names) - max_items} autres marques"

    def smart_summary():
        if not has_data:
            return "Aucune donnée exploitable n’a été détectée sur cette vidéo."
        ordered_time = commercial.sort_values("Temps visible (s)", ascending=False)
        ordered_sov = commercial.sort_values("Part de voix (%)", ascending=False)
        n = len(commercial)
        leader = ordered_time.iloc[0]
        leader_logo = str(leader["Logo"])
        leader_time = float(leader["Temps visible (s)"])
        leader_pct = float(leader["% vidéo"])
        leader_seq = float(leader["Nb de séquences"])
        leader_sov = float(leader["Part de voix (%)"])
        leader_cent = float(leader["Centralité (%)"])
        if n == 1:
            return (f"{leader_logo} concentre toute la visibilité détectée sur cette vidéo. La marque cumule "
                    f"{leader_time:.2f} s d’exposition, soit {leader_pct:.1f} % de la durée analysée, avec "
                    f"{leader_seq:.0f} séquences. Le résultat montre surtout la qualité du placement : "
                    f"occupation maximale de {float(leader['Occupation max (%)']):.2f} %, centralité de {leader_cent:.1f} % "
                    f"et netteté moyenne de {float(leader['Netteté moy.']):.2f}.")
        second = ordered_time.iloc[1]
        gap = leader_time - float(second["Temps visible (s)"])
        sov_names = brand_list(ordered_sov["Logo"].head(3).tolist())
        strong = ordered_time[ordered_time["% vidéo"].astype(float) >= 20]
        if len(strong) >= 2:
            dynamic = f"Plusieurs marques ressortent nettement, notamment {brand_list(strong['Logo'].head(3).tolist())}."
        else:
            dynamic = f"{leader_logo} domine l’exposition temporelle, avec {gap:.2f} s d’avance sur {second['Logo']}."
        return (f"{dynamic} La marque la plus visible est {leader_logo} avec {leader_time:.2f} s d’exposition "
                f"({leader_pct:.1f} % de la vidéo) et {leader_seq:.0f} séquences. La part de voix est portée principalement "
                f"par {sov_names}, ce qui permet de distinguer la durée d’apparition du poids visuel réel à l’écran.")

    def detailed_analysis_rows():
        if not has_data:
            return [("01", "Lecture", "Aucune marque n’a été détectée avec les paramètres actuels.")]
        ordered_time = commercial.sort_values("Temps visible (s)", ascending=False)
        ordered_sov = commercial.sort_values("Part de voix (%)", ascending=False)
        ordered_occ = commercial.sort_values("Occupation max (%)", ascending=False)
        n = len(commercial)
        leader = ordered_time.iloc[0]
        sov_leader = ordered_sov.iloc[0]
        occ_leader = ordered_occ.iloc[0]
        rows = []
        if n == 1:
            rows.append(("01", "Présence", f"{leader['Logo']} est la seule marque détectée. Elle apparaît {float(leader['Nb de séquences']):.0f} fois, pour {float(leader['Temps visible (s)']):.2f} s cumulées, soit {float(leader['% vidéo']):.1f} % de la vidéo."))
            rows.append(("02", "Qualité du placement", f"Le pic d’occupation atteint {float(leader['Occupation max (%)']):.2f} % de l’écran. La centralité de {float(leader['Centralité (%)']):.1f} % indique que la marque n’est pas seulement présente : elle est aussi régulièrement placée dans une zone lisible de l’image."))
            rows.append(("03", "Valeur sponsor", f"La part de voix visuelle atteint {float(leader['Part de voix (%)']):.1f} %. Sur cette vidéo, l’analyse sert donc surtout à valoriser la durée, la répétition et la qualité de l’exposition plutôt qu’à comparer plusieurs marques."))
            return rows
        second = ordered_time.iloc[1]
        time_gap = float(leader["Temps visible (s)"]) - float(second["Temps visible (s)"])
        rows.append(("01", "Hiérarchie des marques", f"{leader['Logo']} arrive en tête du temps visible avec {float(leader['Temps visible (s)']):.2f} s. L’écart avec {second['Logo']} est de {time_gap:.2f} s, ce qui permet d’identifier la marque la plus exposée sur la durée."))
        rows.append(("02", "Poids visuel réel", f"En part de voix visuelle, {sov_leader['Logo']} domine avec {float(sov_leader['Part de voix (%)']):.1f} %. Cet indicateur complète le temps visible : une marque peut apparaître moins longtemps mais occuper davantage d’espace à l’écran."))
        rows.append(("03", "Qualité d’exposition", f"Le meilleur pic d’occupation est obtenu par {occ_leader['Logo']} avec {float(occ_leader['Occupation max (%)']):.2f} % de l’écran. La centralité et la netteté permettent de juger si l’exposition est réellement exploitable pour un sponsor."))
        return rows

    ranking = chart_uri("Classement")
    sov_img = chart_uri("Part de voix")
    timeline = chart_uri("Timeline")
    occupation = chart_uri("Occupation")
    heatmap = chart_uri("Carte")
    summary = esc(smart_summary())
    analysis_html = "".join(
        f'<div class="analysis-row"><div class="num">{no}</div><div><h3>{esc(title)}</h3><p>{esc(body)}</p></div></div>'
        for no, title, body in detailed_analysis_rows()
    )

    html_doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rapport de visibilité des marques</title>
<style>
:root {{
  --blue:#46619c; --blue-dark:#172033; --blue-soft:#eef3f8; --ink:#172033;
  --muted:#6b7588; --line:#dbe4f3; --paper:#ffffff; --bg:#eef3f8; --sand:#f5efe6;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Arial,sans-serif; }}
.report {{ max-width:1180px; margin:0 auto; padding:34px 28px 70px; }}
.top-actions {{ position:sticky; top:14px; z-index:10; display:flex; justify-content:flex-end; gap:10px; margin-bottom:14px; }}
.print-btn {{ border:0; border-radius:999px; padding:12px 18px; background:var(--blue); color:white; font-weight:900; cursor:pointer; box-shadow:0 14px 34px rgba(70,97,156,.25); transition:background .18s ease, transform .18s ease, box-shadow .18s ease; }}
.print-btn:hover {{ background:#344a78; transform:translateY(-2px); box-shadow:0 18px 42px rgba(70,97,156,.34); }}
.print-btn:active {{ transform:translateY(0); background:#2f4270; }}
.page {{ min-height:760px; background:#f8fbff; border:1px solid var(--line); border-radius:34px; padding:44px; margin:0 0 26px; box-shadow:0 24px 80px rgba(30,43,70,.08); page-break-after:always; overflow:hidden; }}
.cover {{ background:linear-gradient(135deg,#111a2d 0%,#26375e 50%,#5570aa 100%); color:white; position:relative; min-height:820px; display:flex; flex-direction:column; justify-content:center; gap:70px; }}
.cover:before {{ content:""; position:absolute; right:-90px; top:55px; width:420px; height:420px; border-radius:50%; background:rgba(255,255,255,.12); }}
.cover:after {{ content:""; position:absolute; right:180px; bottom:120px; width:190px; height:190px; border-radius:50%; background:rgba(255,255,255,.08); }}
.kicker {{ font-size:12px; letter-spacing:.16em; text-transform:uppercase; color:var(--blue); font-weight:950; }}
.cover .kicker {{ color:#dce7ff; }}
h1,h2,h3 {{ margin:0; letter-spacing:-.045em; font-family:Georgia,"Times New Roman",serif; }}
h1 {{ font-size:68px; line-height:.98; max-width:760px; position:relative; z-index:1; }}
h2 {{ font-size:42px; line-height:1.05; margin-top:12px; }}
h3 {{ font-size:22px; }}
.subtitle {{ margin-top:14px; color:#e8eefb; font-size:18px; max-width:680px; line-height:1.55; position:relative; z-index:1; }}
.header {{ display:flex; justify-content:space-between; align-items:flex-start; border-bottom:1px solid var(--line); padding-bottom:24px; margin-bottom:30px; }}
.page-no {{ font-size:26px; font-weight:950; color:#bcc8dc; }}
.grid-kpi {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; position:relative; z-index:1; }}
.kpi {{ background:rgba(255,255,255,.95); color:var(--ink); border:1px solid rgba(255,255,255,.45); border-radius:24px; padding:22px; min-height:128px; box-shadow:0 20px 55px rgba(0,0,0,.10); }}
.kpi-label {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; font-weight:950; }}
.kpi-value {{ font-size:28px; margin-top:12px; font-weight:950; font-family:Georgia,"Times New Roman",serif; letter-spacing:-.03em; }}
.cover-summary {{ position:relative; z-index:1; background:white; border:1px solid var(--line); border-radius:26px; padding:24px; color:var(--muted); line-height:1.6; box-shadow:0 16px 45px rgba(30,43,70,.05); }}
.cover-summary strong {{ color:var(--ink); }}
.cards {{ display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin:22px 0 26px; }}
.small-card {{ background:white; border:1px solid var(--line); border-radius:18px; padding:18px; min-height:102px; }}
.small-card strong {{ display:block; font-size:25px; margin-top:10px; font-family:Georgia,"Times New Roman",serif; }}
.tables {{ display:grid; grid-template-columns:1fr; gap:26px; margin-top:26px; }}
.table-card {{ background:white; border:1px solid var(--line); border-radius:24px; padding:24px; box-shadow:0 16px 45px rgba(30,43,70,.05); }}
.table-card h3 {{ margin-bottom:18px; }}
table {{ width:100%; border-collapse:separate; border-spacing:0; overflow:hidden; border:1px solid var(--line); border-radius:16px; font-size:13px; }}
th {{ background:#f3f6fb; text-align:left; padding:14px 15px; font-size:12px; color:#263044; border-bottom:1px solid var(--line); }}
td {{ padding:14px 15px; border-bottom:1px solid #edf2f8; color:#334155; }}
tr:last-child td {{ border-bottom:0; }}
td:first-child {{ font-weight:900; color:var(--blue); }}
.note {{ margin-top:22px; background:#f7faff; border:1px solid var(--line); border-radius:20px; padding:18px; color:var(--muted); line-height:1.55; }}
.chart-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
.chart-card {{ background:white; border:1px solid var(--line); border-radius:26px; padding:22px; min-height:315px; box-shadow:0 16px 45px rgba(30,43,70,.05); display:flex; flex-direction:column; }}
.chart-card img {{ width:100%; height:245px; object-fit:contain; margin-top:12px; cursor:zoom-in; transition:transform .18s ease, filter .18s ease; }}
.chart-card img:hover {{ transform:scale(1.018); filter:contrast(1.03); }}
.chart-wide {{ min-height:520px; }}
.chart-wide img {{ height:390px; }}
.lightbox {{ position:fixed; inset:0; background:rgba(15,23,42,.82); display:none; align-items:center; justify-content:center; z-index:999; padding:34px; }}
.lightbox.open {{ display:flex; }}
.lightbox img {{ max-width:94vw; max-height:88vh; background:white; border-radius:18px; padding:14px; box-shadow:0 30px 80px rgba(0,0,0,.35); }}
.lightbox-close {{ position:fixed; top:22px; right:26px; border:0; border-radius:999px; background:white; color:var(--ink); font-weight:900; padding:10px 14px; cursor:pointer; }}
.indicator-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:20px; }}
.indicator-card {{ background:white; border:1px solid var(--line); border-radius:22px; padding:22px; min-height:130px; box-shadow:0 16px 45px rgba(30,43,70,.05); }}
.indicator-card h3 {{ color:var(--ink); margin-bottom:10px; }}
.indicator-card p {{ margin:0; color:var(--muted); line-height:1.55; }}
.analysis-grid {{ display:grid; grid-template-columns:1fr; gap:18px; margin-top:18px; }}
.analysis-row {{ display:grid; grid-template-columns:88px 1fr; gap:20px; align-items:start; background:white; border:1px solid var(--line); border-radius:24px; padding:26px; }}
.num {{ font-family:Georgia,"Times New Roman",serif; font-size:34px; font-weight:950; color:var(--blue); }}
.analysis-row p {{ margin:9px 0 0; color:var(--muted); line-height:1.55; }}
.empty {{ color:var(--muted); }}
@media print {{
  body {{ background:white; }} .top-actions, .lightbox {{ display:none!important; }} .report {{ max-width:none; padding:0; }}
  .page {{ width:297mm; min-height:210mm; margin:0; border:0; border-radius:0; box-shadow:none; page-break-after:always; }}
  @page {{ size:A4 landscape; margin:0; }}
}}
@media(max-width:900px) {{ .grid-kpi,.cards,.chart-grid {{ grid-template-columns:1fr; }} h1 {{ font-size:44px; }} .page {{ padding:24px; }} }}
</style>
</head>
<body>
<div class="report">
  <div class="top-actions"><button class="print-btn" onclick="window.print()">Exporter en PDF</button></div>

  <section class="page cover">
    <div>
      <h1>Rapport de visibilité des marques</h1>
      <p class="subtitle">Analyse commerciale de l’exposition visuelle détectée dans la vidéo : durée, répétition, occupation écran, centralité et part de voix.</p>
    </div>
    <div class="grid-kpi">
      <div class="kpi"><div class="kpi-label">Marque dominante</div><div class="kpi-value">{esc(logo)}</div></div>
      <div class="kpi"><div class="kpi-label">Temps visible</div><div class="kpi-value">{fmt(temps,2)} s</div></div>
      <div class="kpi"><div class="kpi-label">Occupation max</div><div class="kpi-value">{fmt(occ_max,2)} %</div></div>
      <div class="kpi"><div class="kpi-label">Part de voix</div><div class="kpi-value">{fmt(sov,1)} %</div></div>
    </div>
  </section>

  <section class="page">
    <div class="header"><div><div class="kicker">Analyse de visibilité</div><h2>Synthèse des performances</h2></div><div class="page-no">02</div></div>
    <div class="cards">
      <div class="small-card"><div class="kpi-label">Temps visible</div><strong>{fmt(temps,2)} s</strong></div>
      <div class="small-card"><div class="kpi-label">Part vidéo</div><strong>{fmt(pct_video,1)} %</strong></div>
      <div class="small-card"><div class="kpi-label">Séquences</div><strong>{fmt(nb_seq,0)}</strong></div>
      <div class="small-card"><div class="kpi-label">Centralité</div><strong>{fmt(centralite,1)} %</strong></div>
      <div class="small-card"><div class="kpi-label">Netteté moy.</div><strong>{fmt(nettete,2)}</strong></div>
    </div>
    <div class="cover-summary"><strong>Synthèse.</strong> {summary}</div>
    <div class="tables">
      <div class="table-card"><h3>Visibilité temporelle</h3>{table_html(visibility_cols)}</div>
      <div class="table-card"><h3>Qualité d’exposition</h3>{table_html(quality_cols)}</div>
    </div>
  </section>

  <section class="page">
    <div class="header"><div><div class="kicker">Analyse de visibilité</div><h2>Dashboard graphique</h2></div><div class="page-no">03</div></div>
    <div class="chart-grid">
      <div class="chart-card"><h3>Classement par temps visible</h3><img onclick="openChart(this.src)" src="{ranking}" alt="Classement par temps visible"></div>
      <div class="chart-card"><h3>Part de voix visuelle</h3><img onclick="openChart(this.src)" src="{sov_img}" alt="Part de voix visuelle"></div>
      <div class="chart-card"><h3>Timeline des apparitions</h3><img onclick="openChart(this.src)" src="{timeline}" alt="Timeline"></div>
      <div class="chart-card"><h3>Occupation écran dans le temps</h3><img onclick="openChart(this.src)" src="{occupation}" alt="Occupation écran"></div>
    </div>
  </section>

  <section class="page">
    <div class="header"><div><div class="kicker">Analyse de visibilité</div><h2>Analyse commerciale</h2></div><div class="page-no">04</div></div>
    <div class="analysis-grid">
      {analysis_html}
    </div>
  </section>

  <section class="page">
    <div class="header"><div><div class="kicker">Analyse de visibilité</div><h2>Définition des indicateurs</h2></div><div class="page-no">05</div></div>
    <div class="indicator-grid">
      <div class="indicator-card"><h3>Temps visible</h3><p>Durée cumulée pendant laquelle une marque est détectée dans la vidéo. C’est l’indicateur principal pour mesurer l’exposition brute.</p></div>
      <div class="indicator-card"><h3>Part vidéo</h3><p>Pourcentage de la vidéo durant lequel la marque apparaît. Il permet de comparer des vidéos de durées différentes.</p></div>
      <div class="indicator-card"><h3>Nombre de séquences</h3><p>Nombre d’apparitions distinctes. Une marque peut être visible longtemps en une seule fois, ou revenir plusieurs fois dans la vidéo.</p></div>
      <div class="indicator-card"><h3>Occupation écran</h3><p>Surface occupée par le logo dans l’image. Plus elle est élevée, plus la marque est visuellement présente.</p></div>
      <div class="indicator-card"><h3>Centralité</h3><p>Part des détections situées dans la zone centrale de l’image. Elle aide à évaluer si l’exposition est réellement lisible.</p></div>
      <div class="indicator-card"><h3>Part de voix visuelle</h3><p>Poids visuel relatif d’une marque par rapport aux autres marques détectées, en tenant compte de la surface occupée.</p></div>
      <div class="indicator-card"><h3>Netteté moyenne</h3><p>Confiance moyenne des détections YOLO. Ce score aide à juger la qualité des détections, mais ne remplace pas la précision globale du modèle.</p></div>
      <div class="indicator-card"><h3>Carte de densité</h3><p>Visualisation des zones où les logos apparaissent le plus souvent. Elle sert à comprendre la valeur des emplacements à l’écran.</p></div>
    </div>
  </section>

  <section class="page">
    <div class="header"><div><div class="kicker">Analyse de visibilité</div><h2>Carte de densité des positions</h2></div><div class="page-no">06</div></div>
    <div class="chart-card chart-wide"><h3>Zones d’exposition à l’écran</h3><img onclick="openChart(this.src)" src="{heatmap}" alt="Carte de densité"></div>
  </section>
</div>
<div class="lightbox" id="chartLightbox" onclick="closeChart()"><button class="lightbox-close" type="button">Fermer</button><img id="lightboxImg" alt="Graphique agrandi"></div>
<script>
function openChart(src) {{ document.getElementById('lightboxImg').src = src; document.getElementById('chartLightbox').classList.add('open'); }}
function closeChart() {{ document.getElementById('chartLightbox').classList.remove('open'); }}
document.addEventListener('keydown', function(e) {{ if (e.key === 'Escape') closeChart(); }});
</script>
</body>
</html>"""

    path.write_text(html_doc, encoding="utf-8")
    return str(path)

if __name__ == "__main__":
    print(f"Compte administrateur initial : {ADMIN_USERNAME}")
    print("Compte client de test : client / Client-TSM-2026!")
    print("Définissez APP_ADMIN_USERNAME et APP_ADMIN_PASSWORD avant le lancement pour personnaliser les accès.")
    demo.launch(
        inbrowser=True,
        max_file_size="10gb",
        auth=authenticate_user,
        auth_message="""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

:root{
  --auth-blue:#46619c;
  --auth-ink:#172033;
  --auth-muted:#6b778c;
  --auth-border:#dfe6f0;
}

/* Le décor est fixé derrière le vrai formulaire Gradio. */
.auth-screen{
  position:fixed;
  inset:0;
  z-index:5;
  display:grid;
  grid-template-columns:52% 48%;
  width:100vw;
  height:100vh;
  overflow:hidden;
  background:#fff;
  font-family:'DM Sans',Arial,sans-serif;
  pointer-events:none;
}

.auth-blue-panel{
  position:relative;
  padding:clamp(42px,5vw,76px);
  overflow:hidden;
  color:#fff;
  background:
    radial-gradient(circle at 108% 8%,rgba(255,255,255,.14) 0 21%,transparent 21.3%),
    radial-gradient(circle at 4% 104%,rgba(255,255,255,.11) 0 17%,transparent 17.3%),
    linear-gradient(145deg,#2e477b 0%,#46619c 57%,#7088ba 100%);
  display:flex;
  flex-direction:column;
  justify-content:space-between;
}

.auth-blue-panel:after{
  content:"";
  position:absolute;
  width:430px;
  height:430px;
  top:-185px;
  right:-175px;
  border:1px solid rgba(255,255,255,.15);
  border-radius:50%;
}

.auth-brand{
  position:relative;
  z-index:2;
  display:flex;
  align-items:center;
  gap:16px;
}

.auth-logo{
  position:relative;
  width:55px;
  height:55px;
  border:1px solid rgba(255,255,255,.28);
  border-radius:17px;
  background:rgba(255,255,255,.12);
}

.auth-logo:before,
.auth-logo:after{
  display:none!important;
  content:none!important;
}
.auth-logo-svg{
  display:flex!important;
  align-items:center!important;
  justify-content:center!important;
  background:transparent!important;
  border:0!important;
}

.auth-brand-name{
  color:#fff;
  font:700 clamp(21px,1.8vw,29px)/1.05 'Fraunces',Georgia,serif;
  letter-spacing:-.035em;
}

.auth-brand-sub{
  margin-top:5px;
  color:#dfe8fa;
  font-size:12px;
}

.auth-copy{
  position:relative;
  z-index:2;
  max-width:570px;
}

.auth-kicker{
  margin-bottom:18px;
  color:#dce6f8;
  font-size:10px;
  font-weight:800;
  letter-spacing:.24em;
  text-transform:uppercase;
}

.auth-copy h2{
  max-width:570px;
  margin:0;
  color:#fff;
  font:700 clamp(40px,4vw,66px)/1.04 'Fraunces',Georgia,serif;
  letter-spacing:-.052em;
}

.auth-copy p{
  max-width:510px;
  margin:22px 0 0;
  color:#e9effb;
  font-size:clamp(13px,1vw,16px);
  line-height:1.72;
}

.auth-secure{
  position:relative;
  z-index:2;
  display:flex;
  align-items:center;
  gap:10px;
  color:#dce6f8;
  font-size:11px;
  font-weight:700;
}

.auth-dot{
  width:8px;
  height:8px;
  border-radius:50%;
  background:#fff;
  box-shadow:0 0 0 5px rgba(255,255,255,.10);
}

.auth-white-panel{
  position:relative;
  background:#fff;
}

.auth-form-heading{
  position:absolute;
  top:calc(50% - 235px);
  left:14.5%;
  width:71%;
}

.auth-form-heading .small{
  margin-bottom:15px;
  color:var(--auth-blue);
  font-size:10px;
  font-weight:800;
  letter-spacing:.22em;
  text-transform:uppercase;
}

.auth-form-heading h1{
  margin:0;
  color:var(--auth-ink);
  font:700 clamp(42px,3.8vw,60px)/1 'Fraunces',Georgia,serif;
  letter-spacing:-.052em;
}

.auth-form-heading p{
  margin:15px 0 0;
  color:var(--auth-muted);
  font-size:14px;
  line-height:1.6;
}

.auth-user-label,
.auth-pass-label{
  position:absolute;
  left:14.5%;
  color:#44526a;
  font-size:12px;
  font-weight:800;
}

.auth-user-label{top:calc(50% - 92px)}
.auth-pass-label{top:calc(50% + 8px)}

/* Réinitialisation du conteneur de connexion Gradio. */
html:has(.auth-screen),
body:has(.auth-screen){
  width:100%!important;
  height:100%!important;
  min-height:100%!important;
  margin:0!important;
  overflow:hidden!important;
  background:#fff!important;
}

body:has(.auth-screen) gradio-app,
body:has(.auth-screen) .gradio-container,
body:has(.auth-screen) main,
body:has(.auth-screen) .main,
body:has(.auth-screen) .wrap,
body:has(.auth-screen) .contain{
  width:100%!important;
  max-width:none!important;
  min-height:100vh!important;
  margin:0!important;
  padding:0!important;
  border:0!important;
  background:transparent!important;
  box-shadow:none!important;
}

/* On masque uniquement le titre "Login" natif, pas les champs. */
body:has(.auth-screen) h1{
  display:none!important;
}

/* Le bloc auth_message ne doit jamais bloquer les clics. */
body:has(.auth-screen) .auth-screen,
body:has(.auth-screen) .auth-screen *{
  pointer-events:none!important;
}

/* Les vrais champs Gradio sont placés sur la partie droite. */
body:has(.auth-screen) input[type="text"],
body:has(.auth-screen) input[type="password"]{
  position:fixed!important;
  left:59vw!important;
  z-index:100!important;
  width:380px!important;
  max-width:380px!important;
  min-width:320px!important;
  height:52px!important;
  margin:0!important;
  padding:0 17px!important;
  border:1px solid var(--auth-border)!important;
  border-radius:14px!important;
  background:#fbfcfe!important;
  color:var(--auth-ink)!important;
  font-family:'DM Sans',Arial,sans-serif!important;
  font-size:14px!important;
  box-shadow:none!important;
  outline:none!important;
  pointer-events:auto!important;
  opacity:1!important;
  visibility:visible!important;
}

body:has(.auth-screen) input[type="text"]{
  top:calc(50vh - 70px)!important;
}

body:has(.auth-screen) input[type="password"]{
  top:calc(50vh + 30px)!important;
}

body:has(.auth-screen) input[type="text"]:focus,
body:has(.auth-screen) input[type="password"]:focus{
  border-color:#7187b8!important;
  background:#fff!important;
  box-shadow:0 0 0 4px rgba(70,97,156,.11)!important;
}

/* Les labels natifs restent invisibles car les libellés sont déjà dessinés. */
body:has(.auth-screen) label:has(input[type="text"]),
body:has(.auth-screen) label:has(input[type="password"]){
  position:static!important;
  margin:0!important;
  padding:0!important;
  border:0!important;
  background:transparent!important;
}

/* Bouton réel de soumission : il reste connecté à authenticate_user. */
body:has(.auth-screen) button[type="submit"],
body:has(.auth-screen) form button:last-of-type{
  position:fixed!important;
  top:calc(50vh + 122px)!important;
  left:59vw!important;
  z-index:101!important;
  width:220px!important;
  max-width:220px!important;
  min-width:220px!important;
  height:48px!important;
  margin:0!important;
  padding:0 22px!important;
  border:0!important;
  border-radius:14px!important;
  background:linear-gradient(135deg,#46619c,#607bb5)!important;
  color:#fff!important;
  font-family:'DM Sans',Arial,sans-serif!important;
  font-size:14px!important;
  font-weight:800!important;
  box-shadow:0 14px 30px rgba(70,97,156,.24)!important;
  cursor:pointer!important;
  pointer-events:auto!important;
  opacity:1!important;
  visibility:visible!important;
  transition:.18s ease!important;
}

body:has(.auth-screen) button[type="submit"]:hover,
body:has(.auth-screen) form button:last-of-type:hover{
  transform:translateY(-2px)!important;
  filter:brightness(1.04)!important;
}

/* Messages d'erreur de connexion. */
body:has(.auth-screen) form > div:not(:has(.auth-screen)){
  z-index:90;
}

@media(max-width:850px){
  html:has(.auth-screen),
  body:has(.auth-screen){
    overflow:auto!important;
  }

  .auth-screen{
    position:absolute;
    grid-template-columns:1fr;
    grid-template-rows:38vh 62vh;
    min-height:100vh;
    height:100vh;
  }

  .auth-blue-panel{
    padding:28px 25px;
  }

  .auth-copy h2{font-size:34px}
  .auth-copy p,.auth-secure{display:none}

  .auth-form-heading{
    top:38px;
    left:24px;
    width:calc(100% - 48px);
  }

  .auth-form-heading h1{font-size:39px}
  .auth-user-label{top:180px;left:24px}
  .auth-pass-label{top:280px;left:24px}

  body:has(.auth-screen) input[type="text"],
  body:has(.auth-screen) input[type="password"]{
    left:24px!important;
    width:calc(100vw - 48px)!important;
    max-width:none!important;
    min-width:0!important;
  }

  body:has(.auth-screen) button[type="submit"],
  body:has(.auth-screen) form button:last-of-type{
    left:24px!important;
    width:190px!important;
    max-width:190px!important;
    min-width:0!important;
  }

  body:has(.auth-screen) input[type="text"]{top:calc(38vh + 205px)!important}
  body:has(.auth-screen) input[type="password"]{top:calc(38vh + 305px)!important}
  body:has(.auth-screen) button[type="submit"],
  body:has(.auth-screen) form button:last-of-type{top:calc(38vh + 397px)!important}
}
</style>

<div class="auth-screen">
  <section class="auth-blue-panel">
    <div class="auth-brand">
      <div class="auth-logo auth-logo-svg">
        <svg viewBox="0 0 64 64" width="55" height="55" aria-hidden="true">
          <rect x="4" y="4" width="56" height="56" rx="16" fill="rgba(255,255,255,.12)" stroke="rgba(255,255,255,.28)"/>
          <path d="M15 32c5-8 10.8-12 17-12s12 4 17 12c-5 8-10.8 12-17 12S20 40 15 32Z"
                fill="none" stroke="white" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
          <circle cx="32" cy="32" r="6.5" fill="white"/>
          <path d="M12 20v-6h6M46 14h6v6M52 44v6h-6M18 50h-6v-6"
                fill="none" stroke="#dce7ff" stroke-width="3" stroke-linecap="round"/>
        </svg>
      </div>
      <div>
        <div class="auth-brand-name">Analyseur de logos</div>
        <div class="auth-brand-sub">Intelligence visuelle pour le sponsoring sportif</div>
      </div>
    </div>

    <div class="auth-copy">
      <div class="auth-kicker">Espace personnel</div>
      <h2>Votre espace d’analyse, en toute simplicité.</h2>
      <p>Connectez-vous pour retrouver vos modèles, vos vidéos analysées, votre bibliothèque et l’ensemble de vos résultats.</p>
    </div>

    <div class="auth-secure">
      <span class="auth-dot"></span>
      Connexion privée et sécurisée
    </div>
  </section>

  <section class="auth-white-panel">
    <div class="auth-form-heading">
      <div class="small">Bienvenue</div>
      <h1>Connexion</h1>
      <p>Saisissez votre identifiant et votre mot de passe pour accéder à l’application.</p>
    </div>
    <div class="auth-user-label">Identifiant</div>
    <div class="auth-pass-label">Mot de passe</div>
  </section>
</div>
""",
        allowed_paths=[
            str(LIBRARY_DIR.resolve()),
            str(RESULTS_DIR.resolve()),
            str(RESULT_ARCHIVES_DIR.resolve()),
            str(RESULT_ARCHIVES_V2_DIR.resolve()),
        ],
    )
