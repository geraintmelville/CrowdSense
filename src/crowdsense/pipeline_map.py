"""Interactive pipeline map for the CrowdSense dashboard.

Everything you are likely to edit lives in the three blocks near the top:
LANES, NODES and EDGES. The code below them turns those into a clickable SVG
with a side panel that shows details for whichever step is selected.

Use it in a Streamlit view:

    from crowdsense.pipeline_map import render_pipeline_map
    render_pipeline_map()

Preview it without Streamlit (writes a standalone HTML file):

    python pipeline_map.py --out pipeline_map.html

How to edit
-----------
* Add or change a step: edit its dict in NODES. `col` is 0-2 (left to right)
  and `row` is 0-10 (top to bottom). Two steps can't share the same cell.
* Add or change an arrow: edit EDGES. Each entry is (from_id, to_id) or
  (from_id, to_id, options). Arrows in the same row are drawn straight across,
  arrows that line up are drawn straight down, and anything else gets an
  elbow. Options are pixel offsets from the left edge of a box:
      out   where the arrow leaves the source box (default: the box's `port`)
      into  where it enters the target box (default: the target's `port`)
      mid   y position of the elbow's horizontal run
* Lane labels: LANES maps a row number to the label drawn just above that row.
* `port` on a node is where arrows attach along its top/bottom edge. Left-hand
  boxes use 150 so vertical arrows clear the lane labels; everything else
  defaults to the box centre (88).
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

# =============================================================================
# EDIT HERE
# =============================================================================

LANES = {
    0: "Ingest",
    3: "Features and labels",
    6: "Modelling",
    7: "Evaluate and package",
    8: "Demo",
}

NODES = [
    # ------------------------------------------------------------------ Ingest
    dict(
        id="raw", kind="data", col=1, row=0,
        title="Raw footage", sub="Full mp4s + clip ZIPs",
        summary="Full-length futsal match recordings (roughly 1.5 to 2 hours each) plus, for each "
                "match, a ZIP of highlight clips that were cut by hand.",
        where=["data/raw/video/", "data/raw/clips/"],
        notes=["Only clips whose filename description is 'goal' are kept.",
               "A highlight clip is the editor's cut, not the acoustic event, so it includes lead-in "
               "and sometimes the restart whistle."],
    ),
    dict(
        id="extract_audio", kind="script", col=0, row=1, port=150,
        title="extract_audio", sub="Video to mono WAV",
        summary="Runs ffmpeg over every match mp4 and writes a mono 22.05 kHz PCM WAV.",
        where=["preprocessing/extract_audio.py"],
        notes=["Skips any match whose WAV already exists.",
               "ffmpeg comes from imageio-ffmpeg, so nothing needs installing system-wide."],
    ),
    dict(
        id="build_clip_database", kind="script", col=2, row=1,
        title="build_clip_database", sub="Pairs matches and clips",
        summary="Pairs each raw match with its clip ZIP, parses clip number, timestamp and description "
                "from the filenames, probes each clip's duration, and writes the matches and clips tables.",
        where=["preprocessing/build_clip_database.py", "preprocessing/functions.py"],
        notes=["Matching uses normalised opposition name + date, and flags unmatched or ambiguous ZIPs.",
               "audio_length_sec is read from the match's WAV, so it is only filled in if extract_audio "
               "has already produced that file. The two branches run in parallel in the diagram, but "
               "this one value ties them together."],
    ),
    dict(
        id="wav", kind="data", col=0, row=2, port=150,
        title="WAV audio", sub="22.05 kHz per match",
        summary="One mono WAV per match. It is resampled to 16 kHz in memory when YAMNet reads it.",
        where=["data/raw/audio/<match>.wav"],
    ),
    dict(
        id="db", kind="data", col=2, row=2,
        title="Clip database", sub="clips_data.db + CSVs",
        summary="SQLite database with a matches table (match_id, raw_filename, zip_filename, match_date, "
                "audio_length_sec) and a clips table (match_id, clip_number, timestamp_formatted, "
                "description, filename, length_sec). Both are also exported as CSV.",
        where=["data/metadata/clips_data.db", "data/metadata/matches.csv", "data/metadata/clips.csv"],
        notes=["matches.csv supplies the raw footage length used for the budget calculation."],
    ),
    # ------------------------------------------------------ Features and labels
    dict(
        id="extract_features", kind="script", col=0, row=3, port=150,
        title="extract_features", sub="YAMNet scores + 16 PCA",
        summary="Runs YAMNet over each match's audio at its native cadence (0.96 s window every 0.48 s), "
                "keeps 11 AudioSet score columns, and compresses the 1024-dim embedding to 16 PCA components.",
        where=["preprocessing/extract_features.py", "modelling/inference.py"],
        notes=["PCA is fit on training matches only. The 8 test matches are transformed with that basis, "
               "never used to fit it.",
               "The fitted PCA is saved to pca_transform.npz so the demo can reuse the exact same projection.",
               "Audio is processed in 28.8 s chunks with a 0.96 s lookahead to keep memory bounded."],
    ),
    dict(
        id="parquet", kind="data", col=0, row=4, port=150,
        title="Feature parquet", sub="11 scores + 16 PCA",
        summary="One parquet file per match. Each row is one YAMNet frame: match_id, raw_filename, "
                "start_sec, 11 yamnet_score columns and 16 yamnet_embedding_pca columns.",
        where=["data/processed/features/096_048/<match>.parquet", "data/processed/features/096_048/_meta.csv"],
        notes=["Score classes: Shout, Yell, Screaming, Whistling, Cheering, Applause, Crowd, Chatter, "
               "Hubbub, Clapping, Children shouting."],
    ),
    dict(
        id="extract_labels", kind="script", col=1, row=4,
        title="extract_labels", sub="3s peak-centred goals",
        summary="For each goal clip, converts its timestamp and length into a span of match time, widens it "
                "by 5 s each side, sums the Cheering + Crowd + Applause scores per frame, and takes the "
                "loudest frame. The positive label is a 3 s window centred on that peak.",
        where=["preprocessing/extract_labels.py"],
        notes=["Training on the whole editor's cut dilutes the signal, which is why labels are narrowed "
               "to the acoustic peak.",
               "Whistling is deliberately left out of the peak search so it stays available as a separate "
               "later feature."],
    ),
    dict(
        id="labels", kind="data", col=2, row=4,
        title="labels.csv", sub="Refined + clip bounds",
        summary="One row per goal with both the refined 3 s label (label_start_sec, label_end_sec) and the "
                "original editor clip bounds (clip_start_sec, clip_end_sec), plus peak_time_sec and peak_score "
                "for auditing.",
        where=["data/processed/labels/labels.csv"],
        notes=["Refined bounds define the training targets. Original clip bounds are what recall is "
               "scored against, so a goal only counts as found if a candidate window fully contains the clip."],
    ),
    dict(
        id="training", kind="data", col=1, row=5,
        title="Training data", sub="Features + labels",
        summary="Not a file. Feature rows are joined to labels in memory: a 0.96 s window is positive if it "
                "overlaps a refined label. Everything outside the held-out test matches is used for training.",
        where=["modelling/functions.py (load_features, build_targets)"],
        notes=["The test set is the 8 most recent matches (IDs 6, 22, 19, 13, 30, 26, 32, 25), since real "
               "use will be on recent footage."],
    ),
    # --------------------------------------------------------------- Modelling
    dict(
        id="tune_model", kind="script", col=2, row=6,
        title="tune_model", sub="Grouped CV, AUC search",
        summary="Random search over XGBoost settings. Each sampled model is fitted once per GroupKFold split "
                "(grouped by match) and its out-of-fold probabilities are pooled. Models are ranked by the "
                "area under the recall-vs-budget curve.",
        where=["modelling/tune_model.py", "constants/constants.py (MODEL_PARAM_DISTRIBUTIONS, CANDIDATE_PARAM_GRID)"],
        notes=["For every fitted model the lookback / postroll / merge-gap grid is swept against the same "
               "pooled probabilities with no refitting, so window settings are tuned for free.",
               "Ranking by curve AUC means the result doesn't hinge on one threshold."],
    ),
    dict(
        id="tuned", kind="data", col=1, row=6,
        title="Tuned settings", sub="Params + window combo",
        summary="Search results sorted best first: model hyperparameters plus the winning lookback, postroll "
                "and merge gap. Downstream scripts read the first row.",
        where=["data/modelling/parameters/model_search_results.csv"],
        notes=["Last run's best: pooled AUC 0.822 with lookback 20 s, postroll 5 s and merge gap 15 s.",
               "The top candidates are close (AUC 0.80 to 0.82, fold std about 0.015), so small differences "
               "between them aren't strong evidence."],
    ),
    dict(
        id="train_predict", kind="script", col=0, row=6, port=150,
        title="train_predict", sub="Fit train, score test",
        summary="Fits one XGBoost model with the tuned parameters on every non-test match, then writes a "
                "probability for each window of the 8 test matches.",
        where=["modelling/train_predict.py"],
        notes=["No threshold is chosen here. That decision is made later by looking at the recall-vs-budget curve."],
    ),
    # ------------------------------------------------------ Evaluate and package
    dict(
        id="test_probs", kind="data", col=0, row=7, port=150,
        title="Test probabilities", sub="Per-window scores",
        summary="match_id, raw_filename, start_sec and probability for every window of the held-out matches.",
        where=["data/modelling/predictions/yamnet_audio_test_probabilities.csv"],
    ),
    dict(
        id="eval_model", kind="script", col=1, row=7,
        title="eval_model", sub="Recall vs budget curve",
        summary="Sweeps thresholds over the test probabilities. At each one it merges the padded windows and "
                "reports recall (goals fully contained) against budget (candidate seconds / raw seconds).",
        where=["modelling/eval_model.py"],
        notes=["Reports recall at 10%, 20%, 30%, 40% and 50% budget plus the full curve AUC, and saves a plot.",
               "Uses the tuned lookback, postroll and merge gap unless overridden on the command line."],
    ),
    dict(
        id="save_final_model", kind="script", col=2, row=7,
        title="save_final_model", sub="OOF threshold + refit",
        summary="Picks a decision threshold from out-of-fold predictions so the candidate budget lands "
                "closest to 30%, then refits the model on all training matches and saves it with its settings.",
        where=["modelling/save_final_model.py", "modelling/model_artifact.py"],
        notes=["The test matches are excluded from both the threshold selection and the final fit.",
               "It also reads the training data and the tuned settings. Only the tuned-settings arrow is drawn "
               "to keep the diagram readable."],
    ),
    # -------------------------------------------------------------------- Demo
    dict(
        id="demo_audio", kind="data", col=0, row=8, port=150,
        title="Demo audio", sub="Match WAV (full demo)",
        summary="The audio for the demo match, extracted from the full video by the demo prep script. "
                "The Full demo runs YAMNet over this live.",
        where=["demo/raw/audio/", "python -m demo.prepare_demo"],
        notes=["Raw media is git-ignored."],
    ),
    dict(
        id="demo_features", kind="data", col=1, row=8,
        title="Demo features", sub="Cached parquet (quick)",
        summary="Model-compatible feature files for the demo match, generated ahead of time. "
                "The Quick demo scores these directly and skips YAMNet.",
        where=["demo/features/"],
        notes=["These are tracked in git so the hosted app can start quickly."],
    ),
    dict(
        id="bundle", kind="data", col=2, row=8,
        title="Model bundle", sub="model.ubj + model.json",
        summary="The trained XGBoost model plus a JSON file holding the threshold, feature columns, "
                "window and stride, lookback / postroll / merge gap, and the PCA components and mean.",
        where=["data/modelling/final_model/model.ubj", "data/modelling/final_model/model.json"],
    ),
    dict(
        id="streamlit_demo", kind="script", col=1, row=9,
        title="Streamlit demo", sub="YAMNet + XGBoost",
        summary="Scores every audio frame with the model bundle, then turns the scores into candidate "
                "windows. Full demo extracts features from the audio first; Quick demo uses the cached features.",
        where=["src/crowdsense/views/dashboard.py", "modelling/inference.py"],
        notes=["The threshold is searched per match so the candidate footage lands near the 30% budget. "
               "The threshold stored in the bundle is only a fallback."],
    ),
    dict(
        id="candidates", kind="data", col=2, row=9,
        title="Candidate windows", sub="Padded and merged",
        summary="Each frame above the threshold is padded (lookback before, postroll after) and overlapping "
                "windows within the merge gap are joined. Times are shown as HH:MM:SS.",
    ),
    dict(
        id="reviewer", kind="data", col=2, row=10,
        title="Reviewer", sub="YouTube timestamp links",
        summary="A person checks each candidate by jumping straight to its timestamp in the match video, "
                "so they only watch a fraction of the footage instead of the whole match.",
        notes=["The demo match link can be changed with CROWDSENSE_DEMO_YOUTUBE_URL."],
    ),
]

EDGES = [
    ("raw", "extract_audio"),
    ("raw", "build_clip_database"),
    ("extract_audio", "wav"),
    ("build_clip_database", "db"),
    ("wav", "extract_features"),
    ("db", "extract_labels", dict(into=128)),
    ("extract_features", "parquet"),
    ("parquet", "extract_labels"),
    ("extract_labels", "labels"),
    ("parquet", "training", dict(into=48)),
    ("labels", "training", dict(into=128)),
    ("training", "tune_model"),
    ("training", "train_predict"),
    ("tune_model", "tuned"),
    ("tuned", "train_predict"),
    ("train_predict", "test_probs"),
    ("tuned", "save_final_model", dict(out=128)),
    ("test_probs", "eval_model"),
    ("save_final_model", "bundle"),
    ("demo_audio", "streamlit_demo", dict(into=48)),
    ("demo_features", "streamlit_demo"),
    ("bundle", "streamlit_demo", dict(into=128)),
    ("streamlit_demo", "candidates"),
    ("candidates", "reviewer"),
]

# =============================================================================
# Rendering (you shouldn't need to touch anything below this line)
# =============================================================================

BOX_W, BOX_H = 176, 52
COL_X = (40, 252, 464)
TOP, ROW_PITCH, SVG_W = 44, 96, 680
PORT_DEFAULT = BOX_W // 2

LIGHT_VARS = (
    "--fg:#1f1f1e;--muted:#5f5e5a;--line:#8a8982;--accent:#534ab7;"
    "--s-fill:#e1f5ee;--s-stroke:#0f6e56;--s-title:#04342c;--s-sub:#0f6e56;"
    "--d-fill:#f1efe8;--d-stroke:#5f5e5a;--d-title:#2c2c2a;--d-sub:#5f5e5a;"
    "--panel:#ffffff;--panel-border:#d3d1c7;--chip:#f1efe8;--chip-hover:#e4e2d9;"
)
DARK_VARS = (
    "--fg:#ecebe6;--muted:#b4b2a9;--line:#8a8982;--accent:#afa9ec;"
    "--s-fill:#085041;--s-stroke:#5dcaa5;--s-title:#9fe1cb;--s-sub:#5dcaa5;"
    "--d-fill:#444441;--d-stroke:#b4b2a9;--d-title:#d3d1c7;--d-sub:#b4b2a9;"
    "--panel:#2a2a28;--panel-border:#4a4a47;--chip:#3a3a37;--chip-hover:#4a4a47;"
)

PAGE_TEMPLATE = """<!doctype html>
<html lang="en" data-theme="__THEME__">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CrowdSense pipeline map</title>
<style>
:root{__LIGHT__}
:root[data-theme="dark"]{__DARK__}
@media (prefers-color-scheme: dark){:root[data-theme="auto"]{__DARK__}}
*{box-sizing:border-box}
html,body{margin:0;background:transparent;color:var(--fg);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.hint{font-size:13px;color:var(--muted);margin:0 0 8px 4px}
.wrap{display:flex;gap:24px;align-items:flex-start;position:relative}
.map{flex:0 0 680px;max-width:100%}
svg{display:block;width:100%;height:auto}
.th{font-size:14px;font-weight:500}
.ts{font-size:12px}
.lane{font-size:12px;fill:var(--muted)}
.edge{fill:none;stroke:var(--line);stroke-width:1.5}
.edge.active{stroke:var(--accent);stroke-width:2.4}
.node{cursor:pointer;outline:none}
.node rect{stroke-width:1}
.node.script rect{fill:var(--s-fill);stroke:var(--s-stroke)}
.node.script .th{fill:var(--s-title)} .node.script .ts{fill:var(--s-sub)}
.node.data rect{fill:var(--d-fill);stroke:var(--d-stroke)}
.node.data .th{fill:var(--d-title)} .node.data .ts{fill:var(--d-sub)}
.node:hover rect{stroke-width:2}
.node:focus-visible rect{stroke:var(--accent);stroke-width:3}
.node.selected rect{stroke:var(--accent);stroke-width:3}
.legend text{font-size:12px;fill:var(--muted)}
.panel{display:none;flex:1 1 280px;max-width:440px;background:var(--panel);
  border:1px solid var(--panel-border);border-radius:10px;padding:16px 18px 14px}
.panel.open{display:block}
.panel h3{margin:2px 0 6px;font-size:17px;font-weight:600}
.panel h4{margin:14px 0 6px;font-size:13px;font-weight:600;color:var(--muted)}
.panel p{margin:0;font-size:14px;line-height:1.5}
.panel ul{margin:0;padding-left:18px;font-size:13px;line-height:1.5}
.panel li+li{margin-top:4px}
.panel ul.mono{list-style:none;padding:0}
.panel ul.mono li{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;
  background:var(--chip);border-radius:6px;padding:3px 7px;word-break:break-all}
.kind{font-size:12px;color:var(--muted)}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{font:inherit;font-size:12px;color:var(--fg);background:var(--chip);border:0;border-radius:999px;
  padding:4px 10px;cursor:pointer}
.chip:hover{background:var(--chip-hover)}
.chip:focus-visible{outline:2px solid var(--accent)}
.close{position:absolute;top:8px;right:10px;border:0;background:none;color:var(--muted);
  font-size:20px;line-height:1;cursor:pointer;padding:4px}
.panel{position:relative}
@media (max-width:900px){
  .wrap{display:block}
  .panel{position:absolute;left:8px;right:8px;z-index:5;max-width:none;box-shadow:0 4px 18px rgba(0,0,0,.25)}
}
@media (prefers-reduced-motion:no-preference){.edge{transition:stroke .15s}}
</style>
</head>
<body>
<p class="hint">Click a step to see what it does, what it reads and writes, and where the code lives.</p>
<div class="wrap" id="wrap">
  <div class="map" id="map">__SVG__</div>
  <aside class="panel" id="panel" aria-live="polite">
    <button class="close" id="close" type="button" aria-label="Close details">&times;</button>
    <div class="kind" id="d-kind"></div>
    <h3 id="d-title"></h3>
    <p id="d-summary"></p>
    <section id="s-where"><h4>Where it lives</h4><ul class="mono" id="d-where"></ul></section>
    <section id="s-in"><h4>Receives from</h4><div class="chips" id="d-in"></div></section>
    <section id="s-out"><h4>Feeds into</h4><div class="chips" id="d-out"></div></section>
    <section id="s-notes"><h4>Worth knowing</h4><ul id="d-notes"></ul></section>
  </aside>
</div>
<script>
(function () {
  var DATA = __DATA__;
  var svg = document.getElementById('map-svg');
  var wrap = document.getElementById('wrap');
  var mapEl = document.getElementById('map');
  var panel = document.getElementById('panel');
  var nodes = {};
  svg.querySelectorAll('.node').forEach(function (g) { nodes[g.dataset.id] = g; });
  var edgeEls = Array.prototype.slice.call(svg.querySelectorAll('.edge'));

  function el(id) { return document.getElementById(id); }

  function fillList(listId, sectionId, items) {
    var list = el(listId);
    list.replaceChildren();
    items.forEach(function (text) {
      var li = document.createElement('li');
      li.textContent = text;
      list.appendChild(li);
    });
    el(sectionId).hidden = items.length === 0;
  }

  function fillChips(boxId, sectionId, ids) {
    var box = el(boxId);
    box.replaceChildren();
    ids.forEach(function (id) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'chip';
      b.textContent = DATA.nodes[id].title;
      b.addEventListener('click', function () { select(id); });
      box.appendChild(b);
    });
    el(sectionId).hidden = ids.length === 0;
  }

  function place(id) {
    var narrow = window.matchMedia('(max-width: 900px)').matches;
    var g = nodes[id].getBoundingClientRect();
    var maxTop = Math.max(0, mapEl.offsetHeight - panel.offsetHeight);
    if (narrow) {
      var top = g.bottom - wrap.getBoundingClientRect().top + 8;
      panel.style.marginTop = '0';
      panel.style.top = Math.max(0, Math.min(top, maxTop)) + 'px';
    } else {
      var offset = g.top - mapEl.getBoundingClientRect().top - 8;
      panel.style.top = 'auto';
      panel.style.marginTop = Math.max(0, Math.min(offset, maxTop)) + 'px';
    }
  }

  function select(id) {
    var n = DATA.nodes[id];
    if (!n) return;
    Object.keys(nodes).forEach(function (k) { nodes[k].classList.toggle('selected', k === id); });
    edgeEls.forEach(function (e) {
      e.classList.toggle('active', e.dataset.src === id || e.dataset.dst === id);
    });
    el('d-kind').textContent = n.kind;
    el('d-title').textContent = n.title;
    el('d-summary').textContent = n.summary;
    fillList('d-where', 's-where', n.where);
    fillList('d-notes', 's-notes', n.notes);
    fillChips('d-in', 's-in', n.receives);
    fillChips('d-out', 's-out', n.feeds);
    panel.classList.add('open');
    place(id);
  }

  function clear() {
    panel.classList.remove('open');
    Object.keys(nodes).forEach(function (k) { nodes[k].classList.remove('selected'); });
    edgeEls.forEach(function (e) { e.classList.remove('active'); });
  }

  Object.keys(nodes).forEach(function (id) {
    var g = nodes[id];
    g.addEventListener('click', function () { select(id); });
    g.addEventListener('keydown', function (ev) {
      if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); select(id); }
    });
  });
  el('close').addEventListener('click', clear);
  document.addEventListener('keydown', function (ev) { if (ev.key === 'Escape') clear(); });
})();
</script>
</body>
</html>
"""


def _validate() -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    cells: dict[tuple[int, int], str] = {}
    for node in NODES:
        node_id = node["id"]
        if node_id in by_id:
            raise ValueError(f"Duplicate node id: {node_id!r}")
        if node["kind"] not in ("script", "data"):
            raise ValueError(f"{node_id}: kind must be 'script' or 'data'")
        if not 0 <= node["col"] < len(COL_X):
            raise ValueError(f"{node_id}: col must be 0-{len(COL_X) - 1}")
        cell = (node["col"], node["row"])
        if cell in cells:
            raise ValueError(f"{node_id} and {cells[cell]} share col={cell[0]}, row={cell[1]}")
        cells[cell] = node_id
        by_id[node_id] = node
    for edge in EDGES:
        for end in edge[:2]:
            if end not in by_id:
                raise ValueError(f"Edge {edge[:2]} refers to unknown node {end!r}")
    return by_id


def _origin(node: dict) -> tuple[int, int]:
    return COL_X[node["col"]], TOP + node["row"] * ROW_PITCH


def _edge_path(src: dict, dst: dict, opts: dict) -> str:
    sx, sy = _origin(src)
    dx, dy = _origin(dst)
    if src["row"] == dst["row"]:
        y = sy + BOX_H // 2
        if dx > sx:
            x1, x2 = sx + BOX_W + 2, dx - 2
        else:
            x1, x2 = sx - 2, dx + BOX_W + 2
        return f"M{x1} {y} H{x2}"
    if dy <= sy:
        raise ValueError(f"Edge {src['id']} -> {dst['id']} must point downwards or sideways")
    x1 = sx + opts.get("out", src.get("port", PORT_DEFAULT))
    x2 = dx + opts.get("into", dst.get("port", PORT_DEFAULT))
    y1, y2 = sy + BOX_H, dy - 2
    if x1 == x2:
        return f"M{x1} {y1} V{y2}"
    mid = opts.get("mid", dy - 22)
    return f"M{x1} {y1} V{mid} H{x2} V{y2}"


def _svg_height() -> int:
    last_row = max(node["row"] for node in NODES)
    return TOP + last_row * ROW_PITCH + BOX_H + 64


def _build_svg(by_id: dict[str, dict]) -> str:
    height = _svg_height()
    parts = [
        f'<svg id="map-svg" viewBox="0 0 {SVG_W} {height}" role="group" aria-label="CrowdSense pipeline">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" '
        'orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" '
        'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>',
    ]
    for row, label in LANES.items():
        parts.append(f'<text class="lane" x="40" y="{TOP + row * ROW_PITCH - 12}">{html.escape(label)}</text>')
    for edge in EDGES:
        src, dst = by_id[edge[0]], by_id[edge[1]]
        opts = edge[2] if len(edge) > 2 else {}
        parts.append(
            f'<path class="edge" data-src="{src["id"]}" data-dst="{dst["id"]}" '
            f'd="{_edge_path(src, dst, opts)}" marker-end="url(#arrow)"/>'
        )
    for node in NODES:
        x, y = _origin(node)
        cx = x + BOX_W // 2
        kind_label = "script" if node["kind"] == "script" else "data"
        parts.append(
            f'<g class="node {node["kind"]}" data-id="{node["id"]}" tabindex="0" role="button" '
            f'aria-label="{html.escape(node["title"])} ({kind_label}). Show details">'
            f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8"/>'
            f'<text class="th" x="{cx}" y="{y + 18}" text-anchor="middle" dominant-baseline="central">'
            f'{html.escape(node["title"])}</text>'
            f'<text class="ts" x="{cx}" y="{y + 36}" text-anchor="middle" dominant-baseline="central">'
            f'{html.escape(node["sub"])}</text></g>'
        )
    legend_y = _svg_height() - 37
    parts.append(
        f'<g class="legend">'
        f'<rect x="40" y="{legend_y}" width="14" height="14" rx="3" '
        f'style="fill:var(--s-fill);stroke:var(--s-stroke)"/>'
        f'<text x="62" y="{legend_y + 8}" dominant-baseline="central">Script or process</text>'
        f'<rect x="190" y="{legend_y}" width="14" height="14" rx="3" '
        f'style="fill:var(--d-fill);stroke:var(--d-stroke)"/>'
        f'<text x="212" y="{legend_y + 8}" dominant-baseline="central">Data or artefact</text></g>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _payload(by_id: dict[str, dict]) -> dict:
    payload = {}
    for node_id, node in by_id.items():
        payload[node_id] = {
            "title": node["title"],
            "kind": "Script or process" if node["kind"] == "script" else "Data or artefact",
            "summary": node.get("summary", ""),
            "where": node.get("where", []),
            "notes": node.get("notes", []),
            "receives": [e[0] for e in EDGES if e[1] == node_id],
            "feeds": [e[1] for e in EDGES if e[0] == node_id],
        }
    return {"nodes": payload}


def build_html(theme: str = "auto") -> str:
    """Return the complete, self-contained HTML page. theme: auto, light or dark."""
    if theme not in ("auto", "light", "dark"):
        raise ValueError("theme must be 'auto', 'light' or 'dark'")
    by_id = _validate()
    data = json.dumps(_payload(by_id)).replace("</", "<\\/")
    return (
        PAGE_TEMPLATE
        .replace("__LIGHT__", LIGHT_VARS)
        .replace("__DARK__", DARK_VARS)
        .replace("__THEME__", theme)
        .replace("__SVG__", _build_svg(by_id))
        .replace("__DATA__", data)
    )


def _detect_theme() -> str:
    """Match Streamlit's active theme where the installed version exposes it."""
    try:
        import streamlit as st
    except ImportError:
        return "auto"
    for getter in (lambda: st.context.theme.type, lambda: st.get_option("theme.base")):
        try:
            value = getter()
        except Exception:
            continue
        if value in ("light", "dark"):
            return value
    return "auto"


def render_pipeline_map(height: int | None = None, theme: str | None = None) -> None:
    """Draw the clickable pipeline map in a Streamlit page.

    height: iframe height in px. The default fits the diagram at full width.
    theme: 'light', 'dark' or 'auto'. Default follows the Streamlit theme when it can be read.
    """
    import streamlit.components.v1 as components

    components.html(
        build_html(theme or _detect_theme()),
        height=height or _svg_height() + 48,
        scrolling=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write the pipeline map to a standalone HTML file.")
    parser.add_argument("--out", type=Path, default=Path("pipeline_map.html"))
    parser.add_argument("--theme", choices=("auto", "light", "dark"), default="auto")
    args = parser.parse_args()
    args.out.write_text(build_html(args.theme), encoding="utf-8")
    print(f"Wrote {args.out}")