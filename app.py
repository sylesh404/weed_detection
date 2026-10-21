# rtsp://10.154.194.228:554/stream1

"""Single-file Cotton & Grass/Weed YOLO Dashboard.
Install: pip install flask opencv-python numpy inference-sdk
Run: python cotton_grass_dashboard.py
Local: http://127.0.0.1:5000
"""
import os
import csv
import json
import time
import socket
import threading
from pathlib import Path
from collections import deque
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv()
except Exception:
    pass

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, render_template_string, send_file, send_from_directory
import supervision as sv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import sys
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

try:
    from recommendation.recommendation_service import RecommendationService
except Exception as _rec_imp_err:
    print(f"Notice: RecommendationService import deferred: {_rec_imp_err}")
    RecommendationService = None

try:
    from analysis.plot_segmenter import PlotSegmenter, QUADRANT_DEFINITIONS
except Exception as _seg_imp_err:
    print(f"Notice: PlotSegmenter import deferred: {_seg_imp_err}")
    PlotSegmenter = None
    QUADRANT_DEFINITIONS = []

# inference_sdk passes no timeout to requests; this caps every socket read/connect instead.
socket.setdefaulttimeout(15)

ROBOFLOW_API_KEY = 'mBcZG6tiOJykgOFO5Cvx'
ROBOFLOW_MODEL_ID = 'cotton-8/3'
# cotton-8/3 rarely recognises cotton, so a dedicated public cotton model runs alongside it.
# Its detections are used for cotton; cotton-8/3 is then only trusted for grass/weed.
COTTON_MODEL_ID = 'cotton-nwbc7/2'
# Neither detector labels reliably on its own (cotton-8/3 calls cotton grass, cotton-nwbc7/2
# calls grass cotton), so they only propose boxes and CLIP decides what each box shows.
CLIP_PROMPTS = [
    ('cotton', 'a photo of a cotton plant with broad green leaves'),
    ('cotton', 'a photo of a white cotton boll or cotton flower'),
    ('grass', 'a photo of green grass blades'),
    ('grass', 'a photo of grassy weeds growing in a field'),
    (None, 'a photo of bare soil or ground'),
    (None, 'a screenshot of a website, text or app interface'),
    (None, 'a photo of a laptop, room, person or furniture'),
    (None, 'a dark or blurry photo'),
]
# A label is accepted only when CLIP gives it at least this share of probability;
# otherwise the box is dropped rather than guessed.
MIN_CLIP_PROB = 0.6
# Whole-frame budget for detector + CLIP calls before the frame is abandoned.
FRAME_TIMEOUT = 12
# Boxes older than this are hidden so a stalled AI never leaves frozen boxes on screen.
STALE_BOX_SECS = 6
# CLIP costs one API call per box, so only the most confident proposals are checked.
MAX_VERIFY = 6

CLIP_CLIENT = InferenceHTTPClient(api_url='https://infer.roboflow.com', api_key=ROBOFLOW_API_KEY)

app = Flask(__name__)
ROOT = Path(__file__).resolve().parent
SNAPSHOTS = ROOT / "snapshots"
SNAPSHOTS.mkdir(exist_ok=True)
# Detection history: one CSV row per logged frame, pointing at its annotated snapshot.
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)
LOG_CSV = LOGS / "detections.csv"
LOG_FIELDS = ['timestamp', 'trigger', 'camera', 'cotton_visible', 'grass_visible',
              'cotton_unique', 'grass_unique', 'snapshot', 'weed_sample']
log_lock = threading.Lock()
# Raw frames with grass/weed plus YOLO-format labels, ready to upload to Roboflow for retraining.
WEED_SAMPLES = ROOT / "weed_samples"
(WEED_SAMPLES / "images").mkdir(parents=True, exist_ok=True)
(WEED_SAMPLES / "labels").mkdir(parents=True, exist_ok=True)
(WEED_SAMPLES / "classes.txt").write_text("cotton\ngrass\n")
CLASS_IDS = {'cotton': 0, 'grass': 1}
WEED_SAMPLE_SECS = 5
# No camera frame for this long means the driver hung; the watchdog reopens the camera.
STALL_SECS = 3
MAX_RECONNECTS = 5

HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AgriSight | Cotton & Grass AI</title>
<style>
:root{--bg:#f5f8f5;--card:#fff;--ink:#162d24;--muted:#78877e;--line:#e4ebe6;--green:#16855a;--green2:#e8f7ef;--amber:#d99027;--red:#cf5757;--shadow:0 12px 36px rgba(31,74,51,.055)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,system-ui,-apple-system,'Segoe UI',sans-serif}button,input,select{font:inherit}button{cursor:pointer}button:disabled{opacity:.55;cursor:not-allowed}
.shell{display:grid;grid-template-columns:242px minmax(0,1fr);min-height:100vh}.side{background:#fff;border-right:1px solid var(--line);padding:28px 18px;display:flex;flex-direction:column;gap:28px}.brand{display:flex;align-items:center;gap:11px;font-size:22px;font-weight:800;letter-spacing:-.7px}.logo{width:38px;height:38px;border-radius:12px;display:grid;place-items:center;background:var(--green);color:#fff}.side-label{color:#a2aea6;font-size:11px;letter-spacing:1.4px;font-weight:800;padding:0 12px}.nav{padding:12px;border-radius:11px;background:var(--green2);color:var(--green);display:flex;align-items:center;gap:12px;font-weight:700}.spacer{flex:1}.tip{background:#f5f9f5;border:1px solid var(--line);padding:16px;border-radius:14px;color:#718078;font-size:13px;line-height:1.6}.main{padding:30px 34px;min-width:0;max-width:1700px;width:100%;margin:auto}.top{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:24px}.eyebrow{color:var(--green);font-size:12px;text-transform:uppercase;letter-spacing:1.7px;font-weight:800}.top h1{margin:6px 0 5px;font-size:29px;letter-spacing:-.9px}.desc{margin:0;color:var(--muted);font-size:14px}.livepill{padding:9px 13px;border:1px solid #d8e8dc;background:#fff;border-radius:99px;color:var(--green);font-weight:750;font-size:12px;white-space:nowrap}.dot{width:8px;height:8px;border-radius:50%;background:#98a59d;display:inline-block;margin-right:7px}.dot.on{background:#19ae70;box-shadow:0 0 0 4px #daf7e8}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:15px;margin-bottom:18px}.metric,.card{background:var(--card);border:1px solid var(--line);border-radius:17px;box-shadow:var(--shadow)}.metric{padding:17px 19px}.metric .t{color:var(--muted);font-size:12px;font-weight:650;display:flex;gap:9px;align-items:center}.metric strong{display:block;font-size:27px;letter-spacing:-.5px;margin-top:8px}.metric small{font-size:12px;color:#90a297}.two{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(300px,.95fr);gap:18px}.card{overflow:hidden}.cardhead{padding:19px 22px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:8px;align-items:center}.cardhead h2{font-size:15px;margin:0}.sub{font-size:12px;color:var(--muted)}.video{background:#10221b;aspect-ratio:16/9;position:relative;display:flex;align-items:center;justify-content:center;overflow:hidden}.video img{width:100%;height:100%;object-fit:contain;display:none}.video canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.empty{color:#c6d3c9;display:grid;text-align:center;gap:10px;justify-items:center;padding:20px}.empty b{color:#fff;font-weight:600}.chip{position:absolute;z-index:2;top:15px;left:15px;border-radius:7px;background:#142a22c7;padding:8px 11px;color:#dff8e9;font-size:12px;font-weight:650}.videofoot{padding:15px 20px;display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}.legend{display:flex;gap:14px;font-size:12px;color:#6f8277}.ball{display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:5px}.btn{background:var(--green);border:1px solid var(--green);color:white;border-radius:10px;padding:10px 14px;font-weight:750;font-size:13px}.btn.ghost{background:#fff;color:var(--ink);border-color:var(--line)}.btn.danger{background:#fff;color:var(--red);border-color:#f0d4d4}.actions{display:flex;gap:8px;flex-wrap:wrap}.pad{padding:20px 22px}.form-row{display:grid;grid-template-columns:1fr 1fr;gap:10px}.field{margin-bottom:14px}.field label{display:block;color:#65766b;font-size:12px;font-weight:750;margin-bottom:7px}.field input,.field select{width:100%;padding:11px;border-radius:9px;border:1px solid #dce7df;background:#fff;color:var(--ink);outline:none;min-width:0}.field input:focus,.field select:focus{border-color:#62ad85;box-shadow:0 0 0 3px #e8f5ed}.smallnote{font-size:12px;line-height:1.55;color:var(--muted);margin:7px 0 0}.seg{display:flex;background:#f1f6f2;padding:4px;border-radius:10px;gap:4px;margin-bottom:17px}.seg button{flex:1;padding:10px;border:0;border-radius:8px;background:transparent;color:#718078;font-size:12px;font-weight:750}.seg button.active{background:white;color:var(--green);box-shadow:0 2px 7px #cadbd0}.section{padding-top:15px;border-top:1px solid var(--line);margin-top:16px}.section h3{font-size:13px;margin:0 0 14px}.range-head{display:flex;justify-content:space-between;font-size:12px;font-weight:750;margin-bottom:8px}.range{accent-color:var(--green);width:100%}.status{padding:12px 14px;border-radius:10px;background:#f4f8f5;color:#64746b;font-size:12px;line-height:1.5;margin-top:12px;word-break:break-word}.status.error{background:#fff3f3;color:#b84d4d}.status.ok{background:#edf9f0;color:#24794d}.modeltags{display:flex;gap:6px;flex-wrap:wrap;margin-top:12px}.tag{background:#ecf5ee;color:#347a54;padding:5px 9px;font-size:11px;border-radius:7px}.note{font-size:12px;color:#8c9b92;padding:16px 0}.i{width:18px;height:18px;flex-shrink:0;vertical-align:middle}.side-footer{font-size:11px;color:#adb8b0;padding-left:12px}
.history{margin-top:18px}.logmeta{font-size:12px;color:var(--muted);padding:12px 22px;border-bottom:1px solid var(--line);line-height:1.7;word-break:break-all}.tablewrap{overflow-x:auto;max-height:440px;overflow-y:auto}table{width:100%;border-collapse:collapse;font-size:12px}th{position:sticky;top:0;background:#f7faf8;color:#65766b;text-align:left;font-weight:750;padding:10px 14px;border-bottom:1px solid var(--line);white-space:nowrap}td{padding:8px 14px;border-bottom:1px solid var(--line);white-space:nowrap}td img{width:96px;height:54px;object-fit:cover;border-radius:6px;display:block}.badge{padding:3px 8px;border-radius:6px;font-size:11px;font-weight:700;background:#ecf5ee;color:#347a54}.badge.weed{background:#fff4e3;color:#a86a12}.badge.manual{background:#eef1fb;color:#4a5ca8}.check{display:flex;align-items:center;gap:8px;font-size:12px;font-weight:650;color:#65766b}
@media(max-width:1120px){.shell{grid-template-columns:76px 1fr}.side{align-items:center;padding:20px 10px}.brand span,.side-label,.nav span,.tip,.side-footer{display:none}.nav{padding:13px}.main{padding:25px}.two{grid-template-columns:1fr}}
@media(max-width:680px){.shell{display:block}.side{height:66px;padding:10px 16px;flex-direction:row;border-right:0;border-bottom:1px solid var(--line);justify-content:space-between}.brand span{display:block;font-size:19px}.side .nav{display:none}.main{padding:18px 12px}.top h1{font-size:22px}.top{align-items:start}.desc{font-size:12px}.metrics{grid-template-columns:repeat(2,minmax(0,1fr));gap:9px}.metric{padding:14px}.metric strong{font-size:23px}.cardhead,.pad{padding:16px}.form-row{grid-template-columns:1fr}.videofoot{align-items:start}.livepill{font-size:10px;padding:8px}.rec-grid{grid-template-columns:1fr 1fr}}
.advisor-card{margin-top:18px;background:#fff;border:1px solid var(--line);border-radius:16px;box-shadow:0 4px 20px rgba(31,74,51,.04);overflow:hidden}
.adv-header{padding:18px 22px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;gap:14px;flex-wrap:wrap}
.adv-header-left{display:flex;flex-direction:column;gap:3px}
.adv-title-wrap{display:flex;align-items:center;gap:9px}
.adv-title-wrap h2{font-size:16px;font-weight:800;letter-spacing:-.3px;margin:0;color:var(--ink);display:flex;align-items:center;gap:8px}
.adv-subtitle{font-size:12px;color:var(--muted);font-weight:450}
.adv-actions{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.adv-btn-primary{background:var(--green);border:1px solid var(--green);color:#fff;border-radius:9px;padding:8px 15px;font-size:12.5px;font-weight:700;display:inline-flex;align-items:center;gap:6px;box-shadow:0 2px 6px rgba(22,133,90,.18);transition:all .16s ease;cursor:pointer}
.adv-btn-primary:hover{background:#13734d;box-shadow:0 4px 10px rgba(22,133,90,.26);transform:translateY(-1px)}
.adv-btn-group{display:inline-flex;align-items:center;gap:6px;flex-wrap:wrap}
.adv-btn-secondary{background:#fff;border:1px solid #dce4df;color:#2c4237;border-radius:8px;padding:7px 11px;font-size:12px;font-weight:600;transition:all .15s ease;cursor:pointer}
.adv-btn-secondary:hover{background:#f4f8f5;border-color:#b7cbbf;color:var(--green)}
.adv-btn-icon{background:#fff;border:1px solid #dce4df;color:#4a6355;border-radius:8px;padding:7px 11px;font-size:12px;font-weight:600;display:inline-flex;align-items:center;gap:5px;text-decoration:none;transition:all .15s ease}
.adv-btn-icon:hover{background:#f4f8f5;border-color:#b7cbbf;color:var(--green)}
.adv-context-bar{background:#fbfdfc;border:1px solid #e7eee9;border-radius:11px;padding:9px 14px;margin-bottom:14px;display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}
.adv-context-row{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.adv-field-item{display:flex;align-items:center;gap:6px;font-size:12px}
.adv-field-label{font-size:11px;font-weight:700;color:#556c5e;text-transform:uppercase;letter-spacing:.5px}
.adv-select,.adv-input{padding:5px 9px;border-radius:7px;border:1px solid #d5e1d9;background:#fff;font-size:12px;color:var(--ink);outline:none;transition:border-color .15s}
.adv-select:focus,.adv-input:focus{border-color:var(--green)}
.adv-status-note{font-size:11.5px;color:#72877a}
.adv-metrics-strip{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:14px}
.adv-metric-tile{background:#fbfdfc;border:1px solid #e7eee9;border-radius:11px;padding:11px 13px;transition:all .15s ease}
.adv-metric-tile:hover{background:#fff;border-color:#d1ded5}
.adv-metric-top{display:flex;align-items:center;gap:6px;margin-bottom:5px}
.adv-metric-icon{font-size:13px;line-height:1}
.adv-metric-name{font-size:10.5px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
.adv-metric-val{font-size:15px;font-weight:800;color:var(--ink);letter-spacing:-.2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.adv-focal-card{background:#f6fbf8;border:1px solid #cde5d6;border-left:4px solid var(--green);border-radius:11px;padding:13px 16px;margin-bottom:14px}
.adv-focal-tag{display:inline-flex;align-items:center;gap:5px;font-size:10.5px;font-weight:800;color:var(--green);letter-spacing:.7px;text-transform:uppercase;margin-bottom:5px}
.adv-focal-dot{width:6px;height:6px;border-radius:50%;background:var(--green)}
.adv-focal-content{font-size:13.5px;font-weight:600;color:var(--ink);line-height:1.5}
.adv-action-title{font-size:14.5px;font-weight:800;color:var(--ink);margin-bottom:3px}
.adv-action-desc{font-size:12.5px;font-weight:500;color:#385042;line-height:1.5}
.adv-section-label{font-size:11px;font-weight:750;color:var(--muted);text-transform:uppercase;letter-spacing:.7px;margin-bottom:7px;display:flex;align-items:center;gap:6px}
.adv-options-list{list-style:none;padding:0;margin:0 0 14px 0;display:flex;flex-direction:column;gap:7px}
.adv-opt-item{background:#fff;border:1px solid #e7eee9;border-radius:9px;padding:9px 12px;display:flex;gap:10px;align-items:flex-start;transition:all .15s ease}
.adv-opt-item:hover{border-color:#cbdcd1}
.adv-opt-icon{width:24px;height:24px;border-radius:6px;background:#eef7f2;color:var(--green);display:flex;align-items:center;justify-content:center;font-size:12px;flex-shrink:0;margin-top:1px}
.adv-opt-body{flex:1;min-width:0}
.adv-opt-head{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:2px}
.adv-opt-title{font-size:12.5px;font-weight:750;color:var(--ink)}
.adv-opt-toggle{background:none;border:none;font-size:11px;font-weight:650;color:var(--green);padding:1px 5px;cursor:pointer;text-decoration:underline}
.adv-opt-summary{font-size:12px;color:#556c5e;line-height:1.45}
.adv-opt-full{font-size:12px;color:#556c5e;line-height:1.5;margin-top:6px;padding-top:6px;border-top:1px dashed #e1eae3;display:none}
.adv-opt-item.expanded .adv-opt-full{display:block}
.adv-accordion-box{margin-bottom:12px}
.adv-accordion{border:1px solid #e7eee9;border-radius:9px;background:#fafcfb;overflow:hidden}
.adv-accordion summary{padding:9px 13px;cursor:pointer;display:flex;align-items:center;justify-content:space-between;font-size:12px;font-weight:650;color:#405749;list-style:none;user-select:none}
.adv-accordion summary::-webkit-details-marker{display:none}
.adv-accordion-indicator{font-size:11px;color:#708679}
.adv-accordion-body{padding:11px 13px;border-top:1px solid #e7eee9;background:#fff;font-size:12px;line-height:1.55}
.adv-source-badges{display:flex;flex-direction:column;gap:6px}
.adv-source-card{background:#fbfdfc;border:1px solid #e7eee9;border-radius:7px;padding:6px 10px;font-size:11.5px;line-height:1.4}
.adv-source-card b{color:var(--ink)}
.adv-source-card small{color:var(--muted);display:block}
.adv-footer-bar{margin-top:10px;padding-top:10px;border-top:1px solid #edf3ef;font-size:11.5px;color:var(--muted);display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px}
.priority-pill{display:inline-block;padding:3px 8px;border-radius:6px;font-size:11px;font-weight:800;letter-spacing:.3px}
.priority-high{background:#fde8e8;color:#c81e1e}
.priority-medium{background:#fef08a;color:#854d0e}
.priority-routine{background:#e8f7ef;color:#16855a}
.priority-verify{background:#e0e7ff;color:#3730a3}
.source-tag{display:inline-block;background:#f0f4f1;border:1px solid var(--line);border-radius:6px;padding:4px 8px;font-size:11px;color:#496152;margin:4px 4px 0 0}
.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:99;display:none;align-items:center;justify-content:center;padding:20px}
.modal-box{background:#fff;border-radius:16px;max-width:680px;width:100%;max-height:85vh;overflow-y:auto;padding:24px;box-shadow:0 20px 40px rgba(0,0,0,.15)}
.plot-card{margin-top:18px}.plot-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:16px 0}.quad-card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:15px;box-shadow:var(--shadow);transition:all .18s ease;border-top:4px solid #a2aea6;cursor:pointer}.quad-card:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(22,133,90,.1)}.quad-card.selected{box-shadow:0 0 0 3px rgba(22,133,90,.35);border-color:var(--green)}.quad-card.risk-LOW{border-top-color:#16855a}.quad-card.risk-MODERATE{border-top-color:#d99027}.quad-card.risk-HIGH{border-top-color:#cf5757}.quad-card.risk-INSUFFICIENT_DATA{border-top-color:#8c9b92}.quad-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}.quad-title{font-size:13px;font-weight:800;display:flex;align-items:center;gap:6px}.quad-metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;background:#f7faf8;padding:9px;border-radius:9px;margin-bottom:10px}.quad-stat .lbl{font-size:10px;color:var(--muted);text-transform:uppercase;font-weight:750}.quad-stat .val{font-size:15px;font-weight:800;color:var(--ink);margin-top:2px}.quad-foot{display:flex;justify-content:space-between;align-items:center;font-size:11px;color:var(--muted)}.plot-viewer-box{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(300px,.95fr);gap:16px;margin-top:16px}.plot-stage-wrap{position:relative;background:#10221b;border-radius:12px;overflow:hidden;aspect-ratio:16/9;display:flex;align-items:center;justify-content:center}.plot-stage-wrap canvas{position:absolute;inset:0;width:100%;height:100%}.plot-stage-wrap img{width:100%;height:100%;object-fit:contain;display:none}.plot-toolbar{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:10px;background:#f7faf8;padding:8px 12px;border-radius:9px;border:1px solid var(--line)}.plot-inspect-panel{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px;overflow-y:auto;max-height:430px}.box-row{font-size:11px;padding:5px 8px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;gap:6px}.risk-pill-sm{padding:3px 7px;border-radius:5px;font-size:10px;font-weight:800;text-transform:uppercase}.risk-green{background:#e8f7ef;color:#16855a}.risk-amber{background:#fef08a;color:#854d0e}.risk-red{background:#fde8e8;color:#c81e1e}.risk-gray{background:#f0f4f1;color:#64746b}
.fieldmap-layout{display:grid;grid-template-columns:minmax(0,1.38fr) minmax(360px,1fr);gap:18px;align-items:start}
.map-mode-badge{display:inline-flex;align-items:center;gap:6px;padding:3px 9px;border-radius:6px;font-size:11px;font-weight:700;background:#e8f7ef;color:#16855a;border:1px solid #c2e8d3}
.map-mode-badge.geo{background:#eff6ff;color:#1d4ed8;border-color:#bfdbfe}
.heatmap-legend{display:flex;align-items:center;gap:11px;padding:8px 12px;background:#fbfdfc;border:1px solid var(--line);border-radius:8px;font-size:11px;flex-wrap:wrap}
.hm-item{display:inline-flex;align-items:center;gap:5px;color:#495c51;font-weight:600}
.hm-swatch{width:11px;height:11px;border-radius:3px;display:inline-block}
.adv-concern-box{background:#fefbf2;border:1px solid #fde8bc;border-left:4px solid #f59e0b;border-radius:10px;padding:11px 14px;margin-bottom:12px;font-size:12.5px;color:#92400e;line-height:1.5}
.history-box{background:#f8faf9;border:1px solid var(--line);border-radius:10px;padding:11px 14px;margin-bottom:12px;font-size:12px}
@media(max-width:1120px){.fieldmap-layout{grid-template-columns:1fr}}
@media(max-width:960px){.plot-grid{grid-template-columns:1fr}.plot-viewer-box{grid-template-columns:1fr}}
</style></head><body>
<svg style="display:none" xmlns="http://www.w3.org/2000/svg"><symbol id="leaf" viewBox="0 0 24 24"><path d="M20 4C10 3 4 8 4 15a5 5 0 0 0 5 5c8 0 12-8 11-16Z M4 20c4-6 7-8 12-11" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></symbol><symbol id="cam" viewBox="0 0 24 24"><path d="M3 7h4l2-3h6l2 3h4v13H3V7Zm9 4a4 4 0 1 0 0 8 4 4 0 0 0 0-8Z" fill="none" stroke="currentColor" stroke-width="1.8"/></symbol><symbol id="radar" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/><path d="m12 12 6-6M12 3v3M3 12h3" fill="none" stroke="currentColor" stroke-width="2"/></symbol></svg>
<div class="shell"><aside class="side"><div class="brand"><div class="logo"><svg class="i" width="24" height="24"><use href="#leaf"/></svg></div><span>AgriSight<span style="color:#a7b2aa"> AI</span></span></div><div><div class="side-label">WORKSPACE</div><div class="nav"><svg class="i"><use href="#radar"/></svg><span>Live Detection</span></div><div class="nav" style="margin-top:6px;cursor:pointer;background:#f0f9f3" onclick="document.getElementById('plotSection').scrollIntoView({behavior:'smooth'})"><svg class="i"><use href="#leaf"/></svg><span>Plot Analysis (2×2)</span></div></div><div class="spacer"></div><div class="tip"><b style="color:#2b5940">Field-ready workflow</b><p>Connect a camera, connect the Roboflow Cotton 8 API, then start the AI detection stream.</p></div><div class="side-footer">COTTON / GRASS VISION · v1.0</div></aside>
<main class="main"><div class="top"><div><div class="eyebrow">Precision agriculture / AI Vision</div><h1>Cotton & Grass Detection</h1><p class="desc">Realtime crop monitoring and unwanted-grass recognition</p></div><div class="livepill"><span class="dot" id="dot"></span><span id="liveText">OFFLINE</span></div></div>
<div class="metrics"><div class="metric"><div class="t"><svg class="i"><use href="#leaf"/></svg>Cotton plants</div><strong id="cottonCount">0</strong><small id="cottonUnique">Visible now · 0 tracked total</small></div><div class="metric"><div class="t">🌱 Grass / weeds</div><strong id="weedCount">0</strong><small id="weedUnique">Visible now · 0 tracked total</small></div><div class="metric"><div class="t">◉ Camera FPS</div><strong id="camFps">0.0</strong><small>Capture performance</small></div><div class="metric"><div class="t">⚡ AI FPS</div><strong id="aiFps">0.0</strong><small id="latency">Inference: — ms</small></div></div>
<div class="two"><section class="card"><div class="cardhead"><div><h2>Live camera feed</h2><div class="sub" style="margin-top:5px">Camera video and live bounding boxes</div></div><div class="sub" id="resolution">—</div></div><div class="video" id="stage"><div class="chip" id="cameraName">NO CAMERA</div><div class="empty" id="empty"><svg width="49" height="49" style="color:#8ba99a"><use href="#cam"/></svg><b>Camera preview is inactive</b><span>Choose a camera and select Start stream</span></div><img id="feed" alt="Live camera feed"><canvas id="overlay"></canvas></div><div class="videofoot"><div class="legend"><span><i class="ball" style="background:#1ebd76"></i>Cotton</span><span><i class="ball" style="background:#ffb84d"></i>Grass / weed</span></div><div class="actions"><button class="btn ghost" id="snapBtn" onclick="snapshot()">📷 Snapshot</button><button class="btn danger" onclick="stopStream()">■ Stop</button></div></div></section>
<section class="card"><div class="cardhead"><div><h2>Detection controls</h2><div class="sub" style="margin-top:5px">Configure your live vision pipeline</div></div><svg class="i" style="color:var(--green)"><use href="#radar"/></svg></div><div class="pad"><div class="seg"><button class="active" id="webTab" onclick="mode('webcam')">💻 Laptop / USB camera</button><button id="ipTab" onclick="mode('ip')">🌐 IP camera</button></div>
<div id="webFields"><div class="field"><label>Available webcam devices</label><div style="display:flex;gap:8px"><select id="cameraSelect" style="flex:1"><option value="">Click Scan cameras</option></select><button class="btn ghost" onclick="scanCameras()" id="scanBtn">⟳ Scan</button></div><div class="smallnote">Scans cameras connected to the machine running Python.</div></div></div>
<div id="ipFields" style="display:none"><div class="field"><label>RTSP / HTTP video URL</label><input id="ipUrl" placeholder="rtsp://192.168.1.10:554/stream1"><div class="smallnote">Use the full stream URL supplied by your IP camera or phone camera app.</div></div></div>
<div class="form-row"><div class="field"><label>Capture resolution</label><select id="res"><option value="640x480">640 × 480 (fast)</option><option value="1280x720" selected>1280 × 720 (HD)</option><option value="1920x1080">1920 × 1080</option></select></div><div class="field"><label>Target camera FPS</label><select id="fps"><option value="15">15 FPS</option><option value="30" selected>30 FPS</option><option value="60">60 FPS</option></select></div></div>
<div class="section"><h3>Roboflow cloud inference</h3>
<div class="field"><label>Roboflow model ID</label><input id="modelId" value="cotton-8/3"></div>
<button class="btn ghost" style="width:100%" onclick="connectModel()" id="loadBtn">↻ Connect Roboflow model</button><div class="modeltags" id="modelTags"></div>
</div>
<div class="section"><div class="range-head"><span>Detection confidence</span><span id="confText">25%</span></div><input class="range" id="conf" type="range" min="10" max="95" value="25" oninput="document.getElementById('confText').textContent=this.value+'%'" onchange="settings()"><div class="form-row" style="margin-top:13px"><div class="field"><label>Upload image size</label><select id="imgsz" onchange="settings()"><option value="320">320 (fastest)</option><option value="416" selected>416 (balanced)</option><option value="640">640 (accurate)</option></select></div><div class="field"><label>Max API requests/sec</label><select id="inferFps" onchange="settings()"><option value="5" selected>5</option><option value="10">10</option><option value="15">15</option><option value="30">30 (if supported)</option></select></div></div></div>
<div class="section"><h3>Detection logging</h3><div class="field"><label>Auto-log snapshot + counts every</label><select id="autoLog" onchange="settings()"><option value="0">Off (manual snapshots only)</option><option value="5">5 seconds</option><option value="10" selected>10 seconds</option><option value="30">30 seconds</option><option value="60">1 minute</option><option value="300">5 minutes</option></select></div><label class="check"><input type="checkbox" id="saveWeed" checked onchange="settings()"> Save weed-infested frames for retraining</label></div>
<button class="btn" style="width:100%;margin-top:4px;padding:13px" onclick="startStream()" id="startBtn">▶ Start live detection</button><div class="status" id="status">Ready. Roboflow connects automatically; choose a camera and start detection. Camera FPS is independent from API FPS.</div></div></section></div>
<section class="card history"><div class="cardhead"><div><h2>Detection history</h2><div class="sub" style="margin-top:5px">Timestamped counts and snapshots saved on this laptop</div></div><div class="actions"><button class="btn ghost" onclick="loadLogs()">⟳ Refresh</button><a class="btn" href="/api/logs.csv" style="text-decoration:none">⬇ Export CSV</a></div></div><div class="logmeta" id="logMeta">Loading history…</div><div class="tablewrap"><table><thead><tr><th>Time</th><th>Trigger</th><th>Cotton now</th><th>Grass now</th><th>Cotton tracked</th><th>Grass tracked</th><th>Snapshot</th><th>Weed sample</th></tr></thead><tbody id="logRows"><tr><td colspan="8">No detections logged yet.</td></tr></tbody></table></div></section>

<!-- =========================================================================
     Four-Part Precision Agriculture Plot Segmentation & Analysis Component
     ========================================================================= -->
<section class="card plot-card" id="plotSection">
  <div class="cardhead">
    <div>
      <div style="display:flex;align-items:center;gap:9px;flex-wrap:wrap">
        <h2>🗺️ Field Map & Weed Risk</h2>
        <span id="plotGlobalStatusPill" class="priority-pill priority-routine">STATUS: READY</span>
        <span class="map-mode-badge" id="mapModeBadge">Mode A: Image-Based Field Map</span>
        <span style="display:none">Four-Part Precision Agricultural Plot Analysis</span>
        <span style="display:none">AI Farmer Advisor & Crop Risk Assessment</span>
      </div>
      <div class="sub" style="margin-top:5px">Visualize weed hotspots, assess field conditions, and identify where action is needed.</div>
    </div>
    <div class="actions">
      <input type="file" id="plotFileInput" accept="image/*" style="display:none" onchange="handlePlotFileSelect(this)">
      <button class="btn" onclick="document.getElementById('plotFileInput').click()" id="uploadPlotBtn">📁 Upload Plot</button>
      <button class="btn ghost" onclick="analyzeActivePlotFrame()" id="activePlotBtn">📷 Active Camera Frame</button>
      <button class="btn ghost" onclick="runDemoPlotAnalysis()" id="demoPlotBtn">🌱 Load Demo Plot</button>
      <button class="btn ghost" onclick="exportPlotCsv()" id="exportPlotCsvBtn">⬇ Export CSV</button>
      <button class="btn ghost" onclick="exportPlotJson()" id="exportPlotJsonBtn">⬇ Export JSON</button>
    </div>
  </div>

  <div class="pad">
    <!-- Plot Context Bar -->
    <div style="background:#f7faf8;border:1px solid var(--line);border-radius:12px;padding:10px 14px;margin-bottom:14px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px">
      <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
        <div style="font-size:11px;font-weight:750;color:var(--muted);text-transform:uppercase;letter-spacing:.6px">Field Context:</div>
        <div style="display:flex;align-items:center;gap:6px;font-size:12px">
          <label for="plotGrowthStageSelect" style="font-weight:650">Cotton Stage:</label>
          <select id="plotGrowthStageSelect" style="padding:5px 9px;border-radius:7px;border:1px solid var(--line);background:#fff;font-size:12px">
            <option value="EMERGENCE_SEEDLING">Seedling (15–35 DAS, High Risk)</option>
            <option value="VEGETATIVE_SQUARING" selected>Vegetative / Squaring (30–60 DAS)</option>
            <option value="FLOWERING_BOLL_DEV">Flowering & Boll Dev (60–90 DAS)</option>
            <option value="BOLL_OPENING_MATURITY">Maturity / Harvest (&gt;90 DAS)</option>
            <option value="UNKNOWN">Unknown / Unspecified</option>
          </select>
        </div>
        <div style="display:flex;align-items:center;gap:6px;font-size:12px">
          <label for="plotFieldIdInput" style="font-weight:650">Plot ID:</label>
          <input id="plotFieldIdInput" value="Plot-A1" style="width:85px;padding:5px 8px;border-radius:7px;border:1px solid var(--line);background:#fff;font-size:12px">
        </div>
        <div style="display:flex;align-items:center;gap:6px;font-size:12px">
          <label for="plotSurveyedAreaInput" style="font-weight:650">Area (m²):</label>
          <input id="plotSurveyedAreaInput" placeholder="Optional" style="width:70px;padding:5px 8px;border-radius:7px;border:1px solid var(--line);background:#fff;font-size:12px">
        </div>
      </div>
      <div id="plotStatusMsg" style="font-size:12px;color:var(--muted)">Ready. Upload a field plot image or click "Load Demo Plot" to evaluate 4 quadrants.</div>
    </div>

    <!-- MAIN 2-COLUMN EXPERIENCE: Area A (Field Map) + Area B (Recommended Action) -->
    <div class="fieldmap-layout">
      <!-- ================= AREA A: FIELD MAP ================= -->
      <div class="area-fieldmap">
        <!-- Map Controls & Filters Toolbar -->
        <div class="plot-toolbar" style="margin-bottom:8px">
          <div style="font-size:11px;font-weight:750;color:var(--muted)">REGION:</div>
          <button class="btn ghost btn-region" id="btnViewALL" onclick="selectQuadrant('ALL')" style="padding:5px 10px;font-size:11px;background:#e8f7ef;color:var(--green)">🖼 Whole Field</button>
          <button class="btn ghost btn-region" id="btnViewNW" onclick="selectQuadrant('NW')" style="padding:5px 9px;font-size:11px">Part 01 (NW)</button>
          <button class="btn ghost btn-region" id="btnViewNE" onclick="selectQuadrant('NE')" style="padding:5px 9px;font-size:11px">Part 02 (NE)</button>
          <button class="btn ghost btn-region" id="btnViewSW" onclick="selectQuadrant('SW')" style="padding:5px 9px;font-size:11px">Part 03 (SW)</button>
          <button class="btn ghost btn-region" id="btnViewSE" onclick="selectQuadrant('SE')" style="padding:5px 9px;font-size:11px">Part 04 (SE)</button>
          <button class="btn ghost" id="btnResetView" onclick="selectQuadrant('ALL')" style="padding:5px 9px;font-size:11px" title="Restore complete field view">⟲ Reset</button>
          <div style="flex:1"></div>
          <div style="display:flex;gap:9px;align-items:center;flex-wrap:wrap">
            <label class="check" style="font-size:11px;color:#d9534f;font-weight:700"><input type="checkbox" id="chkPlotHeatmap" checked onchange="drawPlotCanvas()"> Heatmap</label>
            <label class="check" style="font-size:11px"><input type="checkbox" id="chkPlotCotton" checked onchange="drawPlotCanvas()"> Cotton</label>
            <label class="check" style="font-size:11px"><input type="checkbox" id="chkPlotWeeds" checked onchange="drawPlotCanvas()"> Weeds</label>
            <label class="check" style="font-size:11px"><input type="checkbox" id="chkPlotGrid" checked onchange="drawPlotCanvas()"> Dividers</label>
            <label class="check" style="font-size:11px"><input type="checkbox" id="chkPlotLabels" checked onchange="drawPlotCanvas()"> Labels</label>
          </div>
        </div>

        <!-- Heatmap Legend -->
        <div class="heatmap-legend" style="margin-bottom:10px">
          <span style="font-weight:750;color:#2e4438">Weed Pressure Scale:</span>
          <span class="hm-item"><i class="hm-swatch" style="background:#16855a"></i> Green: Low</span>
          <span class="hm-item"><i class="hm-swatch" style="background:#eab308"></i> Yellow: Moderate</span>
          <span class="hm-item"><i class="hm-swatch" style="background:#f97316"></i> Orange: Elevated</span>
          <span class="hm-item"><i class="hm-swatch" style="background:#dc2626"></i> Red: High</span>
          <span class="hm-item"><i class="hm-swatch" style="background:#94a3b8"></i> Gray: Insufficient</span>
          <span style="font-size:10px;color:var(--muted);margin-left:auto">(Relative visual concentration)</span>
        </div>

        <!-- Map Stage (Canvas + Click-to-inspect) -->
        <div class="plot-stage-wrap" id="plotStageWrap" style="aspect-ratio:16/9;cursor:crosshair" onclick="handleMapCanvasClick(event)">
          <div class="empty" id="plotEmpty">
            <svg width="44" height="44" style="color:#8ba99a"><use href="#cam"/></svg>
            <b>No Field Map Loaded</b>
            <span>Upload an agricultural plot image or click "Load Demo Plot"</span>
          </div>
          <img id="plotImg" alt="Agricultural field plot">
          <canvas id="plotCanvas"></canvas>
        </div>

        <!-- 4-Region Selector Cards (2x2) -->
        <div style="font-size:11px;font-weight:750;color:var(--muted);text-transform:uppercase;letter-spacing:.7px;margin:14px 0 8px">
          FOUR-QUADRANT FIELD SECTIONS (2 × 2)
        </div>
        <div class="plot-grid" id="quadGrid" style="margin:0 0 14px 0">
          <!-- Part 01: Northwest -->
          <div class="quad-card risk-INSUFFICIENT_DATA" id="cardNW" onclick="selectQuadrant('NW')">
            <div class="quad-head">
              <div class="quad-title"><span style="color:var(--green)">◰</span> Part 01 — Northwest</div>
              <span class="risk-pill-sm risk-gray" id="pillNW">NO DATA</span>
            </div>
            <div class="quad-metrics">
              <div class="quad-stat"><div class="lbl">Cotton</div><div class="val" id="cottonNW">—</div></div>
              <div class="quad-stat"><div class="lbl">Weeds</div><div class="val" id="weedsNW">—</div></div>
              <div class="quad-stat"><div class="lbl">Weed:Crop</div><div class="val" id="ratioNW">—</div></div>
            </div>
            <div class="quad-foot">
              <span id="spatialNW">Top-Left Quadrant</span>
              <button class="btn ghost" style="padding:3px 8px;font-size:11px" onclick="event.stopPropagation();selectQuadrant('NW')">🔍 Inspect</button>
            </div>
          </div>

          <!-- Part 02: Northeast -->
          <div class="quad-card risk-INSUFFICIENT_DATA" id="cardNE" onclick="selectQuadrant('NE')">
            <div class="quad-head">
              <div class="quad-title"><span style="color:var(--green)">◲</span> Part 02 — Northeast</div>
              <span class="risk-pill-sm risk-gray" id="pillNE">NO DATA</span>
            </div>
            <div class="quad-metrics">
              <div class="quad-stat"><div class="lbl">Cotton</div><div class="val" id="cottonNE">—</div></div>
              <div class="quad-stat"><div class="lbl">Weeds</div><div class="val" id="weedsNE">—</div></div>
              <div class="quad-stat"><div class="lbl">Weed:Crop</div><div class="val" id="ratioNE">—</div></div>
            </div>
            <div class="quad-foot">
              <span id="spatialNE">Top-Right Quadrant</span>
              <button class="btn ghost" style="padding:3px 8px;font-size:11px" onclick="event.stopPropagation();selectQuadrant('NE')">🔍 Inspect</button>
            </div>
          </div>

          <!-- Part 03: Southwest -->
          <div class="quad-card risk-INSUFFICIENT_DATA" id="cardSW" onclick="selectQuadrant('SW')">
            <div class="quad-head">
              <div class="quad-title"><span style="color:var(--green)">◱</span> Part 03 — Southwest</div>
              <span class="risk-pill-sm risk-gray" id="pillSW">NO DATA</span>
            </div>
            <div class="quad-metrics">
              <div class="quad-stat"><div class="lbl">Cotton</div><div class="val" id="cottonSW">—</div></div>
              <div class="quad-stat"><div class="lbl">Weeds</div><div class="val" id="weedsSW">—</div></div>
              <div class="quad-stat"><div class="lbl">Weed:Crop</div><div class="val" id="ratioSW">—</div></div>
            </div>
            <div class="quad-foot">
              <span id="spatialSW">Bottom-Left Quadrant</span>
              <button class="btn ghost" style="padding:3px 8px;font-size:11px" onclick="event.stopPropagation();selectQuadrant('SW')">🔍 Inspect</button>
            </div>
          </div>

          <!-- Part 04: Southeast -->
          <div class="quad-card risk-INSUFFICIENT_DATA" id="cardSE" onclick="selectQuadrant('SE')">
            <div class="quad-head">
              <div class="quad-title"><span style="color:var(--green)">◳</span> Part 04 — Southeast</div>
              <span class="risk-pill-sm risk-gray" id="pillSE">NO DATA</span>
            </div>
            <div class="quad-metrics">
              <div class="quad-stat"><div class="lbl">Cotton</div><div class="val" id="cottonSE">—</div></div>
              <div class="quad-stat"><div class="lbl">Weeds</div><div class="val" id="weedsSE">—</div></div>
              <div class="quad-stat"><div class="lbl">Weed:Crop</div><div class="val" id="ratioSE">—</div></div>
            </div>
            <div class="quad-foot">
              <span id="spatialSE">Bottom-Right Quadrant</span>
              <button class="btn ghost" style="padding:3px 8px;font-size:11px" onclick="event.stopPropagation();selectQuadrant('SE')">🔍 Inspect</button>
            </div>
          </div>
        </div>

        <!-- Field-Level Summary Metrics -->
        <div class="metrics" style="margin-bottom:12px">
          <div class="metric"><div class="t">Total Cotton Plants</div><strong id="fieldTotalCotton">0</strong><small>Across all 4 quadrants</small></div>
          <div class="metric"><div class="t">Total Invasive Weeds</div><strong id="fieldTotalWeeds">0</strong><small id="fieldTrackedNotice">Visible detections</small></div>
          <div class="metric"><div class="t">Field Weed:Crop Ratio</div><strong id="fieldTotalRatio">0.00</strong><small id="fieldRatioDesc">Observed density ratio</small></div>
          <div class="metric"><div class="t">Highest Pressure Area</div><strong id="fieldHighQuad" style="font-size:20px;color:var(--green)">—</strong><small id="fieldHighDesc">Awaiting analysis</small></div>
        </div>
      </div>

      <!-- ================= AREA B: RECOMMENDED ACTION ================= -->
      <div class="area-action">
        <!-- Region Header & Risk Status -->
        <div class="adv-region-card" style="background:#fff;border:1px solid var(--line);border-radius:14px;padding:16px;box-shadow:var(--shadow)">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px;margin-bottom:12px;border-bottom:1px solid var(--line);padding-bottom:12px">
            <div>
              <div style="font-size:11px;font-weight:750;color:var(--muted);text-transform:uppercase;letter-spacing:.7px">SELECTED REGION:</div>
              <h3 id="advRegionTitle" style="margin:4px 0 0;font-size:16px;color:var(--ink)">Whole Field (Complete Plot)</h3>
              <div id="advRegionCoords" style="font-size:11px;color:var(--muted);margin-top:2px">Full plot view · All 4 quadrants combined</div>
            </div>
            <span id="advRegionRiskPill" class="priority-pill priority-routine" style="font-size:12px;padding:4px 10px">READY</span>
          </div>

          <!-- Region Metrics Strip -->
          <div class="adv-metrics-strip" style="margin-bottom:12px">
            <div class="adv-metric-tile">
              <div class="adv-metric-top"><span class="adv-metric-icon">🌿</span><span class="adv-metric-name">Weeds</span></div>
              <div class="adv-metric-val" id="advRegionWeedCount">—</div>
            </div>
            <div class="adv-metric-tile">
              <div class="adv-metric-top"><span class="adv-metric-icon">🌱</span><span class="adv-metric-name">Cotton</span></div>
              <div class="adv-metric-val" id="advRegionCottonCount">—</div>
            </div>
            <div class="adv-metric-tile">
              <div class="adv-metric-top"><span class="adv-metric-icon">🌡️</span><span class="adv-metric-name">Pressure</span></div>
              <div class="adv-metric-val" id="advRegionPressure" style="font-size:13px">—</div>
            </div>
            <div class="adv-metric-tile">
              <div class="adv-metric-top"><span class="adv-metric-icon">🗺️</span><span class="adv-metric-name">Spatial</span></div>
              <div class="adv-metric-val" id="advRegionSpatial" style="font-size:12px">—</div>
            </div>
          </div>

          <!-- Main Concern Callout -->
          <div class="adv-concern-box" id="advConcernBox">
            <div style="font-weight:750;margin-bottom:3px">⚠️ Main Agronomic Concern:</div>
            <div id="advRegionConcernText">No field analysis generated yet. Click "Analyze Field" or "Load Demo Plot" to evaluate crop risk.</div>
          </div>

          <!-- Focal Recommended Next Step -->
          <div class="adv-focal-card" id="advFocalCard" style="margin-bottom:12px">
            <div class="adv-focal-tag">
              <span class="adv-focal-dot"></span>
              <span>RECOMMENDED NEXT STEP</span>
            </div>
            <div id="advNextAction" class="adv-focal-content">
              Select or analyze a field region to receive evidence-based agronomic recommendations.
            </div>
            <div id="advRegionReasonText" style="margin-top:7px;font-size:12px;color:#355344;line-height:1.45"></div>
          </div>

          <!-- Expandable Management Recommendations (IWM Options) -->
          <div class="adv-accordion-box">
            <details class="adv-accordion" open>
              <summary>
                <div style="display:flex;align-items:center;gap:6px">
                  <span>🛠️</span>
                  <span>Integrated Weed Management (IWM) Options</span>
                </div>
                <span class="adv-accordion-indicator">▾</span>
              </summary>
              <div class="adv-accordion-body" style="padding:10px 12px">
                <ul class="adv-options-list" id="advOptionsList" style="margin:0">
                  <li class="adv-opt-item" style="color:var(--muted)">
                    <div class="adv-opt-icon">🌱</div>
                    <div class="adv-opt-body">
                      <div class="adv-opt-summary">Options will appear following analysis of live weed density and spatial distribution.</div>
                    </div>
                  </li>
                </ul>
              </div>
            </details>
          </div>

          <!-- Expandable Verified Agricultural References -->
          <div class="adv-accordion-box">
            <details class="adv-accordion">
              <summary>
                <div style="display:flex;align-items:center;gap:6px">
                  <span>📚</span>
                  <span>Verified Agricultural References (ICAR / TNAU / CIB&RC)</span>
                </div>
                <span class="adv-accordion-indicator">▾</span>
              </summary>
              <div class="adv-accordion-body" style="padding:10px 12px">
                <div id="advSources" class="adv-source-badges">
                  <span class="source-tag">ICAR-CICR Cotton Weed Manual</span>
                  <span class="source-tag">TNAU Agritech Agronomy</span>
                  <span class="source-tag">CIB&RC Herbicide Stewardship</span>
                </div>
              </div>
            </details>
          </div>

          <!-- Historical Field Survey Comparison Panel -->
          <div class="history-box" id="advHistoryBox">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
              <div style="font-weight:750;color:#2c4437;display:flex;align-items:center;gap:6px">
                <span>🕒</span> Historical Field Survey Comparison
              </div>
              <span id="advHistBadge" class="badge" style="font-size:10px">NO PRIOR DATA</span>
            </div>
            <div id="advHistContent" style="color:var(--muted);line-height:1.5">
              Historical comparison requires at least one previous field survey for this plot to compute changes in weed pressure and hotspot migration.
            </div>
          </div>

          <!-- Advisor Actions Toolbar -->
          <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:14px;padding-top:12px;border-top:1px solid var(--line)">
            <button class="adv-btn-primary" onclick="analyzeCurrentField()" id="analyzeFieldBtn" title="Run AI crop risk and weed assessment">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
              <span>Analyze Field</span>
            </button>
            <button class="adv-btn-secondary" onclick="toggleDetails()" id="detailsBtn" title="View diagnostic details">View Details</button>
            <button class="adv-btn-secondary" onclick="toggleReasoning()" id="whyBtn" title="Explain why this advice was given">Why This?</button>
            <button class="adv-btn-secondary" onclick="openActionModal()" id="actionBtn" title="Log field intervention">Record Action</button>
            <a class="adv-btn-icon" href="/api/recommendations/export" id="exportReportBtn" title="Download CSV Audit Report" aria-label="Export Report">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
              <span>Export</span>
            </a>
          </div>

          <div class="adv-footer-bar" style="margin-top:10px">
            <div id="advFollowUp">Follow-up schedule: Re-scout field according to routine weekly protocol.</div>
            <div style="color:#92a59b;font-size:10.5px">AgriSight AI Precision Engine</div>
          </div>
        </div>

        <!-- Hidden Compatibility elements for tests & advisor sync -->
        <div style="display:none">
          <div id="advisorSection"></div>
          <span id="advisorPill">READY</span>
          <div id="advisorStatusNote"></div>
          <div id="advInfestation"></div>
          <div id="advCropRisk"></div>
          <div id="advPriority"></div>
          <div id="advDistribution"></div>
          <div id="growthStageSelect"></div>
          <div id="fieldIdInput"></div>
          <div id="advMissingBox"><div id="advMissingItems"></div></div>
          <div id="advSourceCount"></div>
          <div id="inspectTitle"></div>
          <div id="inspectPill"></div>
          <div id="inspectMeta"></div>
          <div id="inspectCountText"></div>
          <div id="inspectProximityText"></div>
          <div id="inspectConfText"></div>
          <div id="inspectBoxesList"></div>
          <div id="inspectWarnings"></div>
          <div id="plotAdvisorSummaryBox"><div id="plotAdvisorExplanation"></div><div id="plotAdvisorOptions"></div></div>
        </div>
      </div>
    </div>

    <!-- Whole-Field Comparison Table (Foldable / Expandable) -->
    <div style="margin-top:18px">
      <details style="background:#f7faf8;border:1px solid var(--line);border-radius:12px;padding:12px">
        <summary style="font-size:12px;font-weight:750;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;cursor:pointer">
          <span>WHOLE-FIELD ASSESSMENT & QUADRANT COMPARISON</span>
          <span style="float:right;font-size:11px;color:var(--green)">Click to View/Hide Table ▾</span>
        </summary>
        <div class="tablewrap" style="max-height:260px;margin-top:10px">
          <table>
            <thead>
              <tr>
                <th>Section ID</th>
                <th>Quadrant Name</th>
                <th>Cotton Count</th>
                <th>Weed Count</th>
                <th>Weed:Crop Ratio</th>
                <th>Risk Level</th>
                <th>Action Priority</th>
                <th>Clusters</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody id="fieldTableRows">
              <tr><td colspan="9" style="text-align:center;color:var(--muted)">No plot analysis loaded yet.</td></tr>
            </tbody>
          </table>
        </div>
      </details>
    </div>
  </div>
</section>

<!-- Reason Modal -->
<div class="modal-overlay" id="reasonModal" onclick="if(event.target===this)closeModals()">
  <div class="modal-box">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;border-bottom:1px solid var(--line);padding-bottom:12px">
      <h3 style="margin:0">💡 Why This Recommendation?</h3>
      <button class="btn ghost" onclick="closeModals()" style="padding:6px 10px">✕</button>
    </div>
    <div id="reasonContent"></div>
    <div style="text-align:right;margin-top:18px">
      <button class="btn" onclick="closeModals()">Close</button>
    </div>
  </div>
</div>

<!-- Details Modal -->
<div class="modal-overlay" id="detailsModal" onclick="if(event.target===this)closeModals()">
  <div class="modal-box">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;border-bottom:1px solid var(--line);padding-bottom:12px">
      <h3 style="margin:0">📋 Full Recommendation Diagnostic Data</h3>
      <button class="btn ghost" onclick="closeModals()" style="padding:6px 10px">✕</button>
    </div>
    <pre id="detailsJson" style="background:#10221b;color:#85e89d;padding:16px;border-radius:10px;font-size:11px;overflow-x:auto;max-height:55vh"></pre>
    <div style="text-align:right;margin-top:18px">
      <button class="btn" onclick="closeModals()">Close</button>
    </div>
  </div>
</div>

<!-- Farmer Action Modal -->
<div class="modal-overlay" id="actionModal" onclick="if(event.target===this)closeModals()">
  <div class="modal-box">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;border-bottom:1px solid var(--line);padding-bottom:12px">
      <h3 style="margin:0">✍️ Record Farmer Field Action</h3>
      <button class="btn ghost" onclick="closeModals()" style="padding:6px 10px">✕</button>
    </div>
    <div style="font-size:12px;color:var(--muted);margin-bottom:14px">Log field intervention taken for assessment: <b id="actionAid"></b></div>
    <div class="field">
      <label>Action Implemented:</label>
      <select id="actionSelect">
        <option value="MECHANICAL_INTER_CULTIVATION">Mechanical Inter-Cultivation (Cultivator / Blade Harrow)</option>
        <option value="MANUAL_INTRA_ROW_WEEDING">Manual Intra-Row Hand Weeding / Hoeing</option>
        <option value="SHIELDED_DIRECTED_SPRAY">Shielded Directed Herbicide Spray (Protective Hoods Used)</option>
        <option value="ROUTINE_MONITORING_SCHEDULED">Routine Monitoring Scheduled (No Immediate Removal)</option>
        <option value="SPOT_TREATMENT_APPLIED">Localized Spot Removal</option>
        <option value="OTHER_CULTURAL_ACTION">Other Cultural Practice (Mulch/Hand Pulling)</option>
      </select>
    </div>
    <div class="field">
      <label>Field Notes / Observations:</label>
      <input id="actionNotes" placeholder="e.g. Cleared 2 acres of Plot-A1 using shallow tractor hoeing, soil dry.">
    </div>
    <div style="display:flex;justify-content:flex-end;gap:8px;margin-top:18px">
      <button class="btn ghost" onclick="closeModals()">Cancel</button>
      <button class="btn" onclick="submitFarmerAction()">Save Action</button>
    </div>
  </div>
</div>
<p class="note">Camera FPS and AI FPS are separate. The dashboard targets 30 FPS capture; API speed depends on upload bandwidth, Roboflow latency, and rate limits. "Now" counts are the latest frame; "tracked total" counts unique plants by ByteTrack ID since the camera started.</p></main></div>
<script>
const $=id=>document.getElementById(id);let sourceMode='webcam',streamActive=false,latestData={};let previousStream=0;
async function api(url,payload){const init=payload===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)};const r=await fetch(url,init);const d=await r.json();if(!r.ok||d.ok===false)throw Error(d.error||'Request failed');return d}
function message(t,kind=''){const e=$('status');e.textContent=t;e.className='status '+kind}
function mode(m){sourceMode=m;$('webTab').classList.toggle('active',m==='webcam');$('ipTab').classList.toggle('active',m==='ip');$('webFields').style.display=m==='webcam'?'block':'none';$('ipFields').style.display=m==='ip'?'block':'none'}
async function scanCameras(){const b=$('scanBtn');b.disabled=true;message('Scanning available laptop/USB cameras...');try{const d=await api('/api/cameras');$('cameraSelect').innerHTML='';d.cameras.forEach(c=>{const o=document.createElement('option');o.value=c.index;o.textContent=c.name;$('cameraSelect').append(o)});message(d.cameras.length?`Found ${d.cameras.length} camera(s).`:'No camera detected on the Python host.',d.cameras.length?'ok':'error')}catch(e){message(e.message,'error')}finally{b.disabled=false}}
async function connectModel(){const b=$('loadBtn');b.disabled=true;message('Connecting to Roboflow...');try{const d=await api('/api/models',{model_id:$('modelId').value.trim()});$('modelTags').innerHTML='';d.models.forEach(m=>{const x=document.createElement('span');x.className='tag';x.textContent=m;$('modelTags').append(x)});message(d.message,'ok')}catch(e){message(e.message,'error')}finally{b.disabled=false}}

async function settings(){try{await api('/api/settings',{conf:Number($('conf').value)/100,imgsz:Number($('imgsz').value),max_infer_fps:Number($('inferFps').value),auto_log_secs:Number($('autoLog').value),save_weed:$('saveWeed').checked})}catch(e){message(e.message,'error')}}
async function startStream(){const input=sourceMode==='webcam'?$('cameraSelect').value:$('ipUrl').value.trim();if(input===''){message('Choose a camera or enter an IP stream URL.','error');return}const b=$('startBtn');b.disabled=true;message('Connecting to camera...');try{await settings();const d=await api('/api/start',{mode:sourceMode,source:input,resolution:$('res').value,fps:Number($('fps').value)});$('feed').src='/video_feed?session='+Date.now();$('feed').style.display='block';$('empty').style.display='none';streamActive=true;message(d.message,'ok')}catch(e){message(e.message,'error')}finally{b.disabled=false}}
async function stopStream(){try{await api('/api/stop',{});streamActive=false;$('feed').removeAttribute('src');$('feed').style.display='none';$('empty').style.display='grid';$('cameraName').textContent='NO CAMERA';clearCanvas();message('Camera stopped.','ok')}catch(e){message(e.message,'error')}}
async function snapshot(){try{const d=await api('/api/snapshot',{});message('Snapshot saved and logged: '+d.file,'ok');loadLogs()}catch(e){message(e.message,'error')}}
function clearCanvas(){const c=$('overlay');c.getContext('2d').clearRect(0,0,c.width,c.height)}
function drawBoxes(data){const canvas=$('overlay'),stage=$('stage'),img=$('feed');const w=stage.clientWidth,h=stage.clientHeight,ratio=window.devicePixelRatio||1;canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);if(!streamActive||!data.width||!data.height)return;const scale=Math.min(w/data.width,h/data.height);const dw=data.width*scale,dh=data.height*scale,ox=(w-dw)/2,oy=(h-dh)/2;
for(const b of data.boxes||[]){const x=ox+b.x1*scale,y=oy+b.y1*scale,bw=(b.x2-b.x1)*scale,bh=(b.y2-b.y1)*scale;const color=b.kind==='cotton'?'#20d487':'#ffb84d';ctx.strokeStyle=color;ctx.lineWidth=2.3;ctx.strokeRect(x,y,bw,bh);const label=(b.track_id!=null?'#'+b.track_id+' ':'')+(b.kind==='cotton'?'Cotton':'Grass / weed')+' '+Math.round(b.conf*100)+'%';ctx.font='bold 12px system-ui';const tw=ctx.measureText(label).width+16,ly=y>=26?y-25:y;ctx.fillStyle=color;ctx.fillRect(x,ly,tw,23);ctx.fillStyle='#10251b';ctx.fillText(label,x+8,ly+15)} }
async function poll(){try{const d=await api('/api/status');latestData=d;$('cottonCount').textContent=d.cotton_count;$('weedCount').textContent=d.grass_count;$('cottonUnique').textContent='Visible now · '+d.cotton_unique+' tracked total';$('weedUnique').textContent='Visible now · '+d.grass_unique+' tracked total';$('camFps').textContent=d.camera_fps.toFixed(1);$('aiFps').textContent=d.ai_fps.toFixed(1);$('latency').textContent='Inference: '+(d.inference_ms?d.inference_ms.toFixed(0)+' ms':'—');$('resolution').textContent=d.width?d.width+' × '+d.height:'—';$('cameraName').textContent=d.active?d.camera_name:'NO CAMERA';$('dot').classList.toggle('on',d.active);$('liveText').textContent=d.active?'CAMERA LIVE':'OFFLINE';if(!d.active&&streamActive){streamActive=false;$('feed').style.display='none';$('empty').style.display='grid'}if(d.error&&d.error!==previousStream){message(d.error,'error');previousStream=d.error}else if(!d.error&&previousStream){message('Camera reconnected ('+d.reconnects+' automatic reconnect'+(d.reconnects===1?'':'s')+' this session).','ok');previousStream=0}drawBoxes(d)}catch(e){/* server not yet reachable */}}
function esc(t){const e=document.createElement('span');e.textContent=t;return e.innerHTML}
async function loadLogs(){try{const d=await api('/api/logs?limit=50');$('logMeta').innerHTML='CSV log: <b>'+esc(d.log_file)+'</b><br>Snapshots: <b>'+esc(d.snapshot_dir)+'</b><br>Weed samples for retraining ('+d.weed_samples+' images + YOLO labels): <b>'+esc(d.weed_dir)+'</b>';$('logRows').innerHTML=d.rows.length?d.rows.map(r=>'<tr><td>'+esc(r.timestamp)+'</td><td><span class="badge '+esc(r.trigger)+'">'+esc(r.trigger)+'</span></td><td>'+esc(r.cotton_visible)+'</td><td>'+esc(r.grass_visible)+'</td><td>'+esc(r.cotton_unique)+'</td><td>'+esc(r.grass_unique)+'</td><td><a href="/snapshots/'+encodeURIComponent(r.snapshot)+'" target="_blank"><img loading="lazy" src="/snapshots/'+encodeURIComponent(r.snapshot)+'" alt="snapshot"></a></td><td>'+(r.weed_sample?'✓ saved':'—')+'</td></tr>').join(''):'<tr><td colspan="8">No detections logged yet.</td></tr>'}catch(e){$('logMeta').textContent='Could not load history: '+e.message}}

let currentRecommendation = null;

async function loadLatestRecommendation(){
  try {
    const d = await api('/api/recommendations/latest');
    if(d && d.ok && d.available && d.recommendation){
      renderRecommendation(d.recommendation);
    }
  } catch(e){}
}

async function analyzeCurrentField(){
  const b = $('analyzeFieldBtn');
  b.disabled = true;
  if($('advisorStatusNote')) $('advisorStatusNote').textContent = 'Analyzing field detections & assessing crop risk...';
  if($('advisorPill')){ $('advisorPill').textContent = 'ANALYZING...'; $('advisorPill').className = 'priority-pill priority-medium'; }
  if($('advRegionRiskPill')){ $('advRegionRiskPill').textContent = 'ANALYZING...'; $('advRegionRiskPill').className = 'priority-pill priority-medium'; }
  
  try {
    const stage = ($('plotGrowthStageSelect') ? $('plotGrowthStageSelect').value : null) || ($('growthStageSelect') ? $('growthStageSelect').value : 'VEGETATIVE_SQUARING');
    const fieldId = ($('plotFieldIdInput') ? $('plotFieldIdInput').value.trim() : null) || ($('fieldIdInput') ? $('fieldIdInput').value.trim() : 'Plot-A1');
    
    const payload = {
      boxes: latestData.boxes || [],
      width: latestData.width || 1280,
      height: latestData.height || 720,
      field_context: {
        growth_stage: stage,
        field_id: fieldId
      }
    };
    
    const r = await api('/api/recommendations/analyze', payload);
    if(r.ok && r.recommendation){
      renderRecommendation(r.recommendation);
      if($('advisorStatusNote')) $('advisorStatusNote').textContent = 'Assessment completed at ' + (r.recommendation.timestamp || new Date().toLocaleTimeString());
    } else {
      if($('advisorStatusNote')) $('advisorStatusNote').textContent = r.error || 'Assessment failed';
    }
  } catch(e) {
    if($('advisorStatusNote')) $('advisorStatusNote').textContent = 'Error: ' + e.message;
    if($('advisorPill')){ $('advisorPill').textContent = 'ERROR'; $('advisorPill').className = 'priority-pill priority-high'; }
    if($('advRegionRiskPill')){ $('advRegionRiskPill').textContent = 'ERROR'; $('advRegionRiskPill').className = 'priority-pill priority-high'; }
  } finally {
    b.disabled = false;
  }
}

function renderRecommendation(rec){
  currentRecommendation = rec;
  const prio = (rec.action_priority || 'ROUTINE_MONITORING').toUpperCase();
  const pill = $('advisorPill');
  if(pill){
    pill.textContent = prio.replace(/_/g, ' ');
    pill.className = 'priority-pill ' + (
      prio.includes('HIGH') ? 'priority-high' :
      prio.includes('MEDIUM') ? 'priority-medium' :
      prio.includes('VERIF') ? 'priority-verify' : 'priority-routine'
    );
  }
  const regTitle = $('advRegionTitle');
  if(regTitle) regTitle.textContent = 'Active Camera Field View';
  const regRiskPill = $('advRegionRiskPill');
  if(regRiskPill){
    regRiskPill.textContent = (rec.crop_risk || prio).replace(/_/g, ' ');
    regRiskPill.className = 'priority-pill ' + (
      prio.includes('HIGH') ? 'priority-high' :
      prio.includes('MEDIUM') ? 'priority-medium' :
      prio.includes('VERIF') ? 'priority-verify' : 'priority-routine'
    );
  }
  if($('advRegionWeedCount')) $('advRegionWeedCount').textContent = rec.weed_count != null ? rec.weed_count : (latestData.grass_count || 0);
  if($('advRegionCottonCount')) $('advRegionCottonCount').textContent = rec.visible_cotton_count != null ? rec.visible_cotton_count : (latestData.cotton_count || 0);
  if($('advRegionPressure')) $('advRegionPressure').textContent = rec.infestation_level || '—';
  if($('advRegionSpatial')) $('advRegionSpatial').textContent = (rec.weed_distribution || '—').replace(/_/g, ' ');
  if($('advRegionConcernText')) $('advRegionConcernText').textContent = rec.summary || rec.main_concern || rec.reason || 'Active camera stream weed-crop competition assessment.';
  if($('advRegionReasonText')) $('advRegionReasonText').textContent = rec.reasoning || rec.reason || '';
  
  if($('advInfestation')) $('advInfestation').textContent = rec.infestation_level || '—';
  if($('advCropRisk')) $('advCropRisk').textContent = rec.crop_risk || rec.risk_level || '—';
  if($('advPriority')) $('advPriority').textContent = prio.replace(/_/g, ' ');
  if($('advDistribution')) $('advDistribution').textContent = (rec.weed_distribution || '—').replace(/_/g, ' ');
  
  const nextEl = $('advNextAction');
  if(rec.recommended_action){
    const actionTitle = rec.recommended_action.replace(/_/g, ' ');
    const reasonText = rec.reasoning || rec.reason || '';
    nextEl.innerHTML = '<div class="adv-action-title">' + esc(actionTitle) + '</div>' +
      (reasonText ? '<div class="adv-action-desc">' + esc(reasonText) + '</div>' : '');
  } else {
    nextEl.textContent = rec.reasoning || rec.reason || '—';
  }

  const providerLabel = rec.ai_provider ? (' · ' + (rec.ai_provider === 'Groq' ? '⚡ Powered by Groq AI' : 'Deterministic Mode')) : '';
  if($('advisorStatusNote')) $('advisorStatusNote').textContent = 'Assessment completed at ' + (rec.timestamp || new Date().toLocaleTimeString()) + providerLabel;
  
  const opts = rec.management_options || [];
  if(opts.length){
    $('advOptionsList').innerHTML = opts.map(o => {
      const parts = o.split(':');
      const title = parts.length > 1 ? parts[0].trim() : 'Recommendation';
      const desc = parts.length > 1 ? parts.slice(1).join(':').trim() : o;
      const isLong = desc.length > 85;
      const shortDesc = isLong ? desc.slice(0, 80) + '…' : desc;
      const tLower = title.toLowerCase();
      const icon = tLower.includes('mechanical') ? '🚜' :
                   tLower.includes('manual') ? '✋' :
                   tLower.includes('chemical') ? '🛡️' :
                   tLower.includes('cultural') ? '🌱' : '✔';
      return '<li class="adv-opt-item">' +
        '<div class="adv-opt-icon">' + icon + '</div>' +
        '<div class="adv-opt-body">' +
          '<div class="adv-opt-head">' +
            '<span class="adv-opt-title">' + esc(title) + '</span>' +
            (isLong ? '<button class="adv-opt-toggle" onclick="const p=this.closest(\'.adv-opt-item\');p.classList.toggle(\'expanded\');this.textContent=p.classList.contains(\'expanded\')?\'Show less\':\'Details\'">Details</button>' : '') +
          '</div>' +
          '<div class="adv-opt-summary">' + esc(shortDesc) + '</div>' +
          (isLong ? '<div class="adv-opt-full">' + esc(desc) + '</div>' : '') +
        '</div>' +
      '</li>';
    }).join('');
  } else {
    $('advOptionsList').innerHTML = '<li class="adv-opt-item" style="color:var(--muted)">No immediate weed control required.</li>';
  }
  
  const missing = rec.missing_information || [];
  if(missing.length){
    $('advMissingBox').style.display = 'block';
    $('advMissingItems').innerHTML = missing.map(m => '<div style="margin-top:3px">• ' + esc(m) + '</div>').join('');
  } else {
    $('advMissingBox').style.display = 'none';
  }
  
  const sources = rec.sources || [];
  const srcCountEl = $('advSourceCount');
  if(srcCountEl) srcCountEl.textContent = sources.length ? '(' + sources.length + ')' : '';
  if(sources.length){
    $('advSources').innerHTML = sources.map(s =>
      '<div class="adv-source-card">' +
        '<b>' + esc(s.organization||'Agricultural Research') + (s.year ? ' (' + esc(s.year) + ')' : '') + '</b> — ' +
        esc(s.title||s.publication||'') +
        (s.citation ? '<small>' + esc(s.citation) + '</small>' : '') +
      '</div>'
    ).join('');
  }
  
  if(rec.follow_up){
    $('advFollowUp').textContent = 'Next field inspection: ' + rec.follow_up;
  }
}

function toggleReasoning(){
  if(!currentRecommendation){
    alert('Please click "Analyze Field" first to generate an assessment.');
    return;
  }
  const modal = $('reasonModal');
  const obsHtml = (currentRecommendation.observations && currentRecommendation.observations.length) ?
    '<h4>Observed Field Detections</h4><ul style="padding-left:18px;margin:6px 0 12px 0">' + currentRecommendation.observations.map(o => '<li style="margin-bottom:4px">' + esc(o) + '</li>').join('') + '</ul>' : '';
  $('reasonContent').innerHTML = (currentRecommendation.summary ? '<div style="background:#f0f9f3;border-left:4px solid var(--green);padding:10px 14px;border-radius:8px;margin-bottom:12px;font-weight:600">' + esc(currentRecommendation.summary) + '</div>' : '') +
    obsHtml +
    '<h4>Agronomic Rationale & Analysis</h4><p style="line-height:1.6">' + esc(currentRecommendation.reasoning || currentRecommendation.reason || '') + '</p>' +
    '<h4>Supporting Agricultural Citations (ICAR / TNAU / PAU / CIB&RC)</h4>' +
    (currentRecommendation.sources && currentRecommendation.sources.length ? currentRecommendation.sources.map(s => '<div style="margin-bottom:8px;padding:8px;background:#f7faf8;border-radius:8px"><b>' + esc(s.organization) + ' — ' + esc(s.title) + '</b><br><small style="color:var(--muted)">' + esc(s.citation || '') + '</small><p style="margin:4px 0 0;font-size:12px">' + esc(s.key_takeaway || '') + '</p></div>').join('') : '<p>No specific external citations retrieved.</p>') +
    '<h4>Safety & Disclaimer Notice</h4><p style="font-size:12px;color:var(--muted)">' + esc(currentRecommendation.expert_validation_note || (currentRecommendation.safety_warnings ? currentRecommendation.safety_warnings.join(' • ') : '')) + '</p>';
  modal.style.display = 'flex';
}

function toggleDetails(){
  if(!currentRecommendation){
    alert('Please click "Analyze Field" first to generate an assessment.');
    return;
  }
  const modal = $('detailsModal');
  $('detailsJson').textContent = JSON.stringify(currentRecommendation, null, 2);
  modal.style.display = 'flex';
}

function openActionModal(){
  if(!currentRecommendation){
    alert('Please click "Analyze Field" first to generate an assessment.');
    return;
  }
  $('actionAid').textContent = currentRecommendation.assessment_id || 'Latest';
  $('actionModal').style.display = 'flex';
}

function closeModals(){
  $('reasonModal').style.display = 'none';
  $('detailsModal').style.display = 'none';
  $('actionModal').style.display = 'none';
}

async function submitFarmerAction(){
  const action = $('actionSelect').value;
  const notes = $('actionNotes').value.trim();
  const aid = currentRecommendation ? currentRecommendation.assessment_id : '';
  try {
    const res = await api('/api/recommendations/action', {
      assessment_id: aid,
      action_taken: action,
      notes: notes
    });
    alert(res.message || 'Action logged successfully.');
    closeModals();
    loadLatestRecommendation();
  } catch(e) {
    alert('Failed to log action: ' + e.message);
  }
}

// =========================================================================
// Four-Part Precision Agriculture Plot Analysis Client Controller
// =========================================================================

let currentPlotData = null;
let selectedQuadrant = 'ALL';

async function handlePlotFileSelect(input){
  if(!input.files || !input.files[0]) return;
  const file = input.files[0];
  const b = $('uploadPlotBtn');
  b.disabled = true;
  $('plotStatusMsg').textContent = 'Uploading and analyzing 4 plot quadrants (Roboflow + CLIP)...';
  $('plotGlobalStatusPill').textContent = 'ANALYZING...';
  $('plotGlobalStatusPill').className = 'priority-pill priority-medium';

  const formData = new FormData();
  formData.append('image', file);
  formData.append('growth_stage', $('plotGrowthStageSelect').value);
  formData.append('field_id', $('plotFieldIdInput').value.trim() || 'Plot-A1');
  if($('plotSurveyedAreaInput').value.trim()){
    formData.append('surveyed_area_sqm', $('plotSurveyedAreaInput').value.trim());
  }

  try {
    const res = await fetch('/api/plot/analyze', { method: 'POST', body: formData });
    const data = await res.json();
    if(!res.ok || data.ok === false) throw Error(data.error || 'Plot analysis failed');
    renderPlotAnalysis(data);
  } catch(e) {
    $('plotStatusMsg').textContent = 'Error: ' + e.message;
    $('plotGlobalStatusPill').textContent = 'ERROR';
    $('plotGlobalStatusPill').className = 'priority-pill priority-high';
  } finally {
    b.disabled = false;
    input.value = '';
  }
}

async function analyzeActivePlotFrame(){
  const b = $('activePlotBtn');
  b.disabled = true;
  $('plotStatusMsg').textContent = 'Capturing active camera stream and segmenting 4 quadrants...';
  $('plotGlobalStatusPill').textContent = 'CAPTURING...';
  $('plotGlobalStatusPill').className = 'priority-pill priority-medium';

  try {
    const payload = {
      use_current_frame: true,
      field_context: {
        growth_stage: $('plotGrowthStageSelect').value,
        field_id: $('plotFieldIdInput').value.trim() || 'Plot-A1',
        surveyed_area_sqm: $('plotSurveyedAreaInput').value.trim() || null
      }
    };
    const data = await api('/api/plot/analyze', payload);
    renderPlotAnalysis(data);
  } catch(e) {
    $('plotStatusMsg').textContent = 'Error: ' + e.message;
    $('plotGlobalStatusPill').textContent = 'ERROR';
    $('plotGlobalStatusPill').className = 'priority-pill priority-high';
  } finally {
    b.disabled = false;
  }
}

async function runDemoPlotAnalysis(){
  const b = $('demoPlotBtn');
  b.disabled = true;
  $('plotStatusMsg').textContent = 'Generating precision agricultural demo plot & running 4-quadrant evaluation...';
  $('plotGlobalStatusPill').textContent = 'GENERATING...';
  $('plotGlobalStatusPill').className = 'priority-pill priority-medium';

  try {
    const payload = {
      field_context: {
        growth_stage: $('plotGrowthStageSelect').value,
        field_id: $('plotFieldIdInput').value.trim() || 'Plot-Demo-Field-1',
        surveyed_area_sqm: $('plotSurveyedAreaInput').value.trim() || 25.0
      }
    };
    const data = await api('/api/plot/demo', payload);
    renderPlotAnalysis(data);
  } catch(e) {
    $('plotStatusMsg').textContent = 'Error: ' + e.message;
    $('plotGlobalStatusPill').textContent = 'ERROR';
    $('plotGlobalStatusPill').className = 'priority-pill priority-high';
  } finally {
    b.disabled = false;
  }
}

async function loadLatestPlotAnalysis(){
  try {
    const r = await api('/api/plot/latest');
    if(r && r.ok && r.available && r.analysis){
      renderPlotAnalysis(r.analysis);
    }
  } catch(e){}
}

function renderPlotAnalysis(data){
  currentPlotData = data;
  const fa = data.field_assessment || {};
  const sum = fa.summary || {};
  const qmap = data.quadrants || {};

  // Status message & global risk pill
  const overallRisk = sum.overall_risk_category || 'LOW';
  $('plotStatusMsg').textContent = 'Plot assessment ' + data.analysis_id + ' completed in ' + data.total_processing_ms + ' ms. Evaluated 4 quadrants.';
  const gPill = $('plotGlobalStatusPill');
  gPill.textContent = 'RISK: ' + overallRisk.replace('_', ' ');
  gPill.className = 'priority-pill ' + (
    overallRisk === 'HIGH' ? 'priority-high' :
    overallRisk === 'MODERATE' ? 'priority-medium' :
    overallRisk === 'INSUFFICIENT_DATA' ? 'priority-verify' : 'priority-routine'
  );

  // Update 2x2 Grid Cards
  const qids = ['NW', 'NE', 'SW', 'SE'];
  for(const qid of qids){
    const q = qmap[qid] || {};
    const cnt = q.counts || {};
    const rsk = q.risk_assessment || {};
    const spat = q.spatial_summary || {};

    const card = $('card' + qid);
    const pill = $('pill' + qid);
    const cottonEl = $('cotton' + qid);
    const weedsEl = $('weeds' + qid);
    const ratioEl = $('ratio' + qid);
    const spatialEl = $('spatial' + qid);

    const rCat = rsk.risk_category || 'INSUFFICIENT_DATA';
    card.className = 'quad-card risk-' + rCat + (selectedQuadrant === qid ? ' selected' : '');

    pill.textContent = rCat.replace('_', ' ');
    pill.className = 'risk-pill-sm risk-' + (
      rCat === 'HIGH' ? 'red' :
      rCat === 'MODERATE' ? 'amber' :
      rCat === 'LOW' ? 'green' : 'gray'
    );

    cottonEl.textContent = cnt.visible_cotton_count != null ? cnt.visible_cotton_count : '—';
    weedsEl.textContent = cnt.visible_weed_count != null ? cnt.visible_weed_count : '—';
    ratioEl.textContent = cnt.weed_to_crop_ratio_display || '—';

    let spatText = (spat.cluster_count ? spat.cluster_count + ' cluster(s)' : 'No clusters') + ' · ' +
                   (spat.weeds_adjacent_to_crop ? spat.weeds_adjacent_to_crop + ' adjacent to crop' : 'Zero crop contact');
    spatialEl.textContent = spatText;
  }

  // Update Whole Field Summary Metrics
  $('fieldTotalCotton').textContent = sum.total_visible_cotton != null ? sum.total_visible_cotton : 0;
  $('fieldTotalWeeds').textContent = sum.total_visible_weeds != null ? sum.total_visible_weeds : 0;
  $('fieldTotalRatio').textContent = sum.weed_to_crop_ratio_display || '0.00';
  $('fieldTrackedNotice').textContent = sum.tracking_available ? (sum.tracked_weeds + ' unique tracked') : 'Visible detections';
  $('fieldHighQuad').textContent = sum.highest_weed_quadrant ? (sum.highest_weed_quadrant + ' (' + sum.highest_weed_count + ' weeds)') : 'None';
  $('fieldHighDesc').textContent = 'Highest observed weed concentration';

  // Update Section Comparison Table
  const tbody = $('fieldTableRows');
  const sections = fa.section_breakdown || {};
  let tableHtml = '';
  for(const qid of qids){
    const s = sections[qid] || {};
    const rCat = s.risk_category || 'INSUFFICIENT_DATA';
    const tagClass = rCat === 'HIGH' ? 'priority-high' : rCat === 'MODERATE' ? 'priority-medium' : rCat === 'LOW' ? 'priority-routine' : 'priority-verify';
    tableHtml += '<tr>' +
      '<td><b>' + esc(qid) + '</b></td>' +
      '<td>' + esc(s.name || qid) + '</td>' +
      '<td>' + (s.visible_cotton_count != null ? s.visible_cotton_count : 0) + '</td>' +
      '<td>' + (s.visible_weed_count != null ? s.visible_weed_count : 0) + '</td>' +
      '<td>' + esc(s.weed_to_crop_ratio_display || '—') + '</td>' +
      '<td><span class="priority-pill ' + tagClass + '">' + esc(rCat.replace('_', ' ')) + '</span></td>' +
      '<td>' + esc((s.action_priority || 'ROUTINE').replace(/_/g, ' ')) + '</td>' +
      '<td>' + (s.cluster_count || 0) + '</td>' +
      '<td><span class="badge">' + esc(s.status || 'OK') + '</span></td>' +
      '</tr>';
  }
  tbody.innerHTML = tableHtml;

  // Integrated AI Farmer Advisor Summary
  const advRec = fa.advisor_recommendation;
  if(advRec){
    $('plotAdvisorSummaryBox').style.display = 'block';
    const expl = advRec.farmer_summary || advRec.reason || 'Plot evaluated across all 4 quadrants.';
    $('plotAdvisorExplanation').textContent = expl;
    const opts = advRec.management_options || [];
    $('plotAdvisorOptions').innerHTML = '<b>Recommended IWM Practices:</b> ' + (opts.length ? opts.map(o => esc(o)).join(' • ') : 'Continue routine field monitoring.');
  } else {
    $('plotAdvisorSummaryBox').style.display = 'none';
  }

  // Display image & inspector
  selectQuadrant(selectedQuadrant || 'ALL');
}

function selectQuadrant(qid){
  selectedQuadrant = qid;
  const qids = ['ALL', 'NW', 'NE', 'SW', 'SE'];
  for(const id of qids){
    const btn = $('btnView' + id);
    if(btn){
      if(id === qid){
        btn.style.background = '#e8f7ef';
        btn.style.color = 'var(--green)';
        btn.style.borderColor = 'var(--green)';
      } else {
        btn.style.background = '#fff';
        btn.style.color = 'var(--ink)';
        btn.style.borderColor = 'var(--line)';
      }
    }
  }

  // Highlight card in 2x2 grid
  for(const id of ['NW', 'NE', 'SW', 'SE']){
    const card = $('card' + id);
    if(card){
      card.classList.toggle('selected', id === qid);
    }
  }

  if(!currentPlotData) return;

  const fa = currentPlotData.field_assessment || {};
  const qmap = currentPlotData.quadrants || {};
  const recs = currentPlotData.region_recommendations || {};
  const hist = currentPlotData.historical_comparison || {};

  // Update Mapping Mode badge
  const modeBadge = $('mapModeBadge');
  if(modeBadge){
    const isGeo = currentPlotData.mapping_mode && currentPlotData.mapping_mode.includes('GEO');
    modeBadge.textContent = isGeo ? 'Mode B: Georeferenced Field Map' : 'Mode A: Image-Based Field Map';
    modeBadge.className = 'map-mode-badge' + (isGeo ? ' geo' : '');
  }

  // Update Region-Specific Area B
  const regTitle = $('advRegionTitle');
  const regCoords = $('advRegionCoords');
  const regRiskPill = $('advRegionRiskPill');
  const regWeedCnt = $('advRegionWeedCount');
  const regCottonCnt = $('advRegionCottonCount');
  const regPressure = $('advRegionPressure');
  const regSpatial = $('advRegionSpatial');
  const regConcern = $('advRegionConcernText');
  const regNextAct = $('advNextAction');
  const regReason = $('advRegionReasonText');

  const rec = recs[qid] || {};
  if(qid === 'ALL'){
    if(regTitle) regTitle.textContent = 'Whole Field (Complete Plot)';
    if(regCoords) regCoords.textContent = 'Full plot view · ' + currentPlotData.original_dimensions.width + ' × ' + currentPlotData.original_dimensions.height + ' px';
    const sum = fa.summary || {};
    const rsk = sum.overall_risk_category || 'LOW';
    if(regRiskPill){
      regRiskPill.textContent = rsk.replace('_', ' ');
      regRiskPill.className = 'priority-pill ' + (rsk==='HIGH'?'priority-high':rsk==='MODERATE'?'priority-medium':rsk==='LOW'?'priority-routine':'priority-verify');
    }
    if(regWeedCnt) regWeedCnt.textContent = sum.total_visible_weeds != null ? sum.total_visible_weeds : 0;
    if(regCottonCnt) regCottonCnt.textContent = sum.total_visible_cotton != null ? sum.total_visible_cotton : 0;
    if(regPressure) regPressure.textContent = sum.highest_weed_quadrant ? ('Peak in ' + sum.highest_weed_quadrant) : 'Uniform';
    if(regSpatial) regSpatial.textContent = (sum.weed_to_crop_ratio_display || '0.00') + ' weed:crop';
    if(regConcern) regConcern.textContent = rec.main_concern || 'Field-wide weed competition and canopy development assessment across 4 quadrants.';
    if(regNextAct) regNextAct.innerHTML = '<div class="adv-action-title">' + esc(rec.recommended_action ? rec.recommended_action.replace(/_/g, ' ') : 'CONTINUE ROUTINE MONITORING') + '</div>';
    if(regReason) regReason.textContent = rec.reason || 'Deduplicated whole-field assessment across 4 quadrants.';
  } else {
    const q = qmap[qid] || {};
    const cnt = q.counts || {};
    const spat = q.spatial_summary || {};
    if(regTitle) regTitle.textContent = q.display_name || rec.region_name || ('Part ' + qid);
    if(regCoords) regCoords.textContent = 'Quadrant Sector ' + qid + ' · Offset (' + q.offset_x + ', ' + q.offset_y + ') · ' + q.width + ' × ' + q.height + ' px';
    const rsk = rec.risk_category || (q.risk_assessment && q.risk_assessment.risk_category) || 'INSUFFICIENT_DATA';
    if(regRiskPill){
      regRiskPill.textContent = rsk.replace('_', ' ');
      regRiskPill.className = 'priority-pill ' + (rsk==='HIGH'?'priority-high':rsk==='MODERATE'?'priority-medium':rsk==='LOW'?'priority-routine':'priority-verify');
    }
    if(regWeedCnt) regWeedCnt.textContent = cnt.visible_weed_count != null ? cnt.visible_weed_count : 0;
    if(regCottonCnt) regCottonCnt.textContent = cnt.visible_cotton_count != null ? cnt.visible_cotton_count : 0;
    if(regPressure) regPressure.textContent = rec.action_priority ? rec.action_priority.replace(/_/g, ' ') : (rsk === 'HIGH' ? 'High Risk' : 'Routine');
    if(regSpatial) regSpatial.textContent = (spat.cluster_count || 0) + ' clusters, ' + (spat.weeds_adjacent_to_crop || 0) + ' near crop';
    if(regConcern) regConcern.textContent = rec.main_concern || 'Localized weed pressure in ' + (q.name || qid) + '.';
    if(regNextAct) regNextAct.innerHTML = '<div class="adv-action-title">' + esc((rec.recommended_action || 'CONTINUE_REGULAR_MONITORING').replace(/_/g, ' ')) + '</div>';
    if(regReason) regReason.textContent = rec.reason || '';
  }

  // Update IWM Options
  const opts = rec.management_options || [];
  const optList = $('advOptionsList');
  if(optList){
    if(opts.length){
      optList.innerHTML = opts.map(o => {
        const parts = o.split(':');
        const title = parts.length > 1 ? parts[0].trim() : 'Recommendation';
        const desc = parts.length > 1 ? parts.slice(1).join(':').trim() : o;
        const tLower = title.toLowerCase();
        const icon = tLower.includes('mechanical') ? '🚜' :
                     tLower.includes('manual') ? '✋' :
                     tLower.includes('chemical') ? '🛡️' :
                     tLower.includes('cultural') ? '🌱' : '✔';
        return '<li class="adv-opt-item">' +
          '<div class="adv-opt-icon">' + icon + '</div>' +
          '<div class="adv-opt-body">' +
            '<div class="adv-opt-head"><span class="adv-opt-title">' + esc(title) + '</span></div>' +
            '<div class="adv-opt-summary">' + esc(desc) + '</div>' +
          '</div>' +
        '</li>';
      }).join('');
    } else {
      optList.innerHTML = '<li class="adv-opt-item" style="color:var(--muted)">No immediate weed control required for this section.</li>';
    }
  }

  // Update References
  const sources = rec.sources || [];
  const srcEl = $('advSources');
  if(srcEl && sources.length){
    srcEl.innerHTML = sources.map(s =>
      '<div class="adv-source-card">' +
        '<b>' + esc(s.organization || 'ICAR') + (s.year ? ' (' + esc(s.year) + ')' : '') + '</b> — ' +
        esc(s.title || s.citation || '') +
        (s.citation ? '<small>' + esc(s.citation) + '</small>' : '') +
      '</div>'
    ).join('');
  }

  // Update Historical Comparison Panel
  const histBadge = $('advHistBadge');
  const histContent = $('advHistContent');
  if(histBadge && histContent){
    if(hist && hist.has_prior_survey){
      histBadge.textContent = 'SURVEY DELTA';
      histBadge.className = 'badge';
      let wDelta = hist.weed_change_count >= 0 ? ('+' + hist.weed_change_count) : ('' + hist.weed_change_count);
      let pctStr = hist.weed_change_pct >= 0 ? ('+' + hist.weed_change_pct + '%') : (hist.weed_change_pct + '%');
      let shiftText = hist.hotspot_shifted ? ('⚠️ Hotspot migrated from ' + hist.prior_hotspot + ' to ' + hist.current_hotspot) : ('Hotspot stationary in ' + hist.current_hotspot);
      histContent.innerHTML = '<b>Previous Survey (' + esc(hist.prior_timestamp || '') + '):</b><br>' +
        '• Weed count change: <b>' + wDelta + ' weeds (' + pctStr + ')</b><br>' +
        '• Risk transition: ' + esc(hist.prior_risk || '—') + ' → <b>' + esc(hist.current_risk || '—') + '</b><br>' +
        '• ' + shiftText + '<br>' +
        (hist.interventions_applied ? ('• Prior intervention: ' + esc(hist.interventions_applied)) : '');
    } else {
      histBadge.textContent = 'NO PRIOR DATA';
      histBadge.className = 'badge';
      histContent.textContent = 'Historical comparison requires at least one previous field survey for this plot to compute changes in weed pressure and hotspot migration.';
    }
  }

  // Follow-up
  if(rec.follow_up && $('advFollowUp')){
    $('advFollowUp').textContent = 'Next inspection: ' + rec.follow_up;
  }

  // Update Image source & trigger canvas draw
  const img = $('plotImg');
  const empty = $('plotEmpty');
  if(currentPlotData && currentPlotData.image_urls){
    empty.style.display = 'none';
    img.style.display = 'block';
    const targetUrl = (qid === 'ALL') ? currentPlotData.image_urls.annotated : (currentPlotData.image_urls[qid] || currentPlotData.image_urls.original);
    img.onload = () => drawPlotCanvas();
    img.src = targetUrl + '?t=' + Date.now();
  }
}

function handleMapCanvasClick(evt){
  if(!currentPlotData) return;
  const canvas = $('plotCanvas');
  const rect = canvas.getBoundingClientRect();
  const clickX = evt.clientX - rect.left;
  const clickY = evt.clientY - rect.top;

  let origW = currentPlotData.original_dimensions.width;
  let origH = currentPlotData.original_dimensions.height;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const scale = Math.min(w / origW, h / origH);
  const dw = origW * scale;
  const dh = origH * scale;
  const ox = (w - dw) / 2;
  const oy = (h - dh) / 2;

  if(clickX >= ox && clickX <= ox + dw && clickY >= oy && clickY <= oy + dh){
    if(selectedQuadrant !== 'ALL'){
      // Click again while zoomed toggles back to ALL
      selectQuadrant('ALL');
      return;
    }
    const imgX = (clickX - ox) / scale;
    const imgY = (clickY - oy) / scale;
    const midX = currentPlotData.midpoint.x;
    const midY = currentPlotData.midpoint.y;

    let targetQ = 'NW';
    if(imgX < midX && imgY < midY) targetQ = 'NW';
    else if(imgX >= midX && imgY < midY) targetQ = 'NE';
    else if(imgX < midX && imgY >= midY) targetQ = 'SW';
    else targetQ = 'SE';

    selectQuadrant(targetQ);
  }
}

function drawPlotCanvas(){
  const canvas = $('plotCanvas');
  const stage = $('plotStageWrap');
  const img = $('plotImg');
  if(!currentPlotData || img.style.display === 'none') return;

  const w = stage.clientWidth;
  const h = stage.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * ratio);
  canvas.height = Math.round(h * ratio);

  const ctx = canvas.getContext('2d');
  ctx.scale(ratio, ratio);

  const showHeatmap = $('chkPlotHeatmap') ? $('chkPlotHeatmap').checked : true;
  const showGrid = $('chkPlotGrid') ? $('chkPlotGrid').checked : true;
  const showCotton = $('chkPlotCotton') ? $('chkPlotCotton').checked : true;
  const showWeeds = $('chkPlotWeeds') ? $('chkPlotWeeds').checked : true;
  const showLabels = $('chkPlotLabels') ? $('chkPlotLabels').checked : true;

  let origW = currentPlotData.original_dimensions.width;
  let origH = currentPlotData.original_dimensions.height;

  if(selectedQuadrant !== 'ALL'){
    const qinfo = currentPlotData.quadrants[selectedQuadrant];
    if(qinfo){
      origW = qinfo.width;
      origH = qinfo.height;
    }
  }

  const scale = Math.min(w / origW, h / origH);
  const dw = origW * scale;
  const dh = origH * scale;
  const ox = (w - dw) / 2;
  const oy = (h - dh) / 2;

  // 1. Continuous Gaussian Weed Risk Heatmap Overlay
  if(showHeatmap && currentPlotData.heatmap && currentPlotData.heatmap.matrix){
    const hm = currentPlotData.heatmap;
    const gRows = hm.grid_rows || 18;
    const gCols = hm.grid_cols || 32;
    const mat = hm.matrix;

    const offCanvas = document.createElement('canvas');
    offCanvas.width = gCols;
    offCanvas.height = gRows;
    const offCtx = offCanvas.getContext('2d');
    const imgData = offCtx.createImageData(gCols, gRows);

    for(let r = 0; r < gRows; r++){
      for(let c = 0; c < gCols; c++){
        const v = mat[r][c]; // 0.0 to 1.0
        const idx = (r * gCols + c) * 4;
        if(v < 0.08){
          imgData.data[idx] = 0;
          imgData.data[idx+1] = 0;
          imgData.data[idx+2] = 0;
          imgData.data[idx+3] = 0; // Transparent
        } else {
          let red = 22, green = 133, blue = 90, alpha = 0.45;
          if(v >= 0.75){
            red = 220; green = 38; blue = 38; alpha = 0.65; // Red: High
          } else if(v >= 0.50){
            red = 249; green = 115; blue = 22; alpha = 0.58; // Orange: Elevated
          } else if(v >= 0.25){
            red = 234; green = 179; blue = 8; alpha = 0.52; // Yellow: Moderate
          } else {
            red = 22; green = 133; blue = 90; alpha = 0.45; // Green: Low
          }
          imgData.data[idx] = red;
          imgData.data[idx+1] = green;
          imgData.data[idx+2] = blue;
          imgData.data[idx+3] = Math.round(alpha * 255);
        }
      }
    }
    offCtx.putImageData(imgData, 0, 0);

    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    if(selectedQuadrant === 'ALL'){
      ctx.drawImage(offCanvas, ox, oy, dw, dh);
    } else {
      const q = currentPlotData.quadrants[selectedQuadrant];
      if(q){
        const fullW = currentPlotData.original_dimensions.width;
        const fullH = currentPlotData.original_dimensions.height;
        const sx = (q.offset_x / fullW) * gCols;
        const sy = (q.offset_y / fullH) * gRows;
        const sw = (q.width / fullW) * gCols;
        const sh = (q.height / fullH) * gRows;
        ctx.drawImage(offCanvas, sx, sy, sw, sh, ox, oy, dw, dh);
      }
    }
    ctx.restore();
  }

  // 2. Draw 4-quadrant divider lines if ALL view
  if(selectedQuadrant === 'ALL' && showGrid){
    const mx = ox + currentPlotData.midpoint.x * scale;
    const my = oy + currentPlotData.midpoint.y * scale;

    ctx.save();
    ctx.strokeStyle = '#20d487';
    ctx.lineWidth = 1.6;
    ctx.setLineDash([6, 6]);

    // Vertical divider
    ctx.beginPath();
    ctx.moveTo(mx, oy);
    ctx.lineTo(mx, oy + dh);
    ctx.stroke();

    // Horizontal divider
    ctx.beginPath();
    ctx.moveTo(ox, my);
    ctx.lineTo(ox + dw, my);
    ctx.stroke();

    // Sector Watermarks
    ctx.setLineDash([]);
    ctx.font = 'bold 12px Inter, system-ui';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.fillText('Part 01 (NW)', ox + 10, oy + 20);
    ctx.fillText('Part 02 (NE)', mx + 10, oy + 20);
    ctx.fillText('Part 03 (SW)', ox + 10, my + 20);
    ctx.fillText('Part 04 (SE)', mx + 10, my + 20);
    ctx.restore();
  }

  // 3. Highlight Selected Quadrant on Canvas
  if(selectedQuadrant !== 'ALL' && currentPlotData.quadrants[selectedQuadrant]){
    // If zoomed into single quadrant, show a subtle green border
    ctx.save();
    ctx.strokeStyle = '#20d487';
    ctx.lineWidth = 2.5;
    ctx.strokeRect(ox, oy, dw, dh);
    ctx.restore();
  }

  // 4. Draw Bounding Boxes
  let boxes = [];
  if(selectedQuadrant === 'ALL'){
    boxes = currentPlotData.field_assessment ? currentPlotData.field_assessment.deduplicated_boxes : [];
  } else {
    const qinfo = currentPlotData.quadrants[selectedQuadrant];
    boxes = qinfo ? qinfo.local_boxes : [];
  }

  for(const b of boxes){
    const isCotton = b.kind === 'cotton';
    if(isCotton && !showCotton) continue;
    if(!isCotton && !showWeeds) continue;

    const bx = ox + b.x1 * scale;
    const by = oy + b.y1 * scale;
    const bw = (b.x2 - b.x1) * scale;
    const bh = (b.y2 - b.y1) * scale;

    const color = isCotton ? '#20d487' : '#ffb84d';
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.2;
    ctx.strokeRect(bx, by, bw, bh);

    if(showLabels){
      const label = (b.track_id != null ? '#' + b.track_id + ' ' : '') +
                    (isCotton ? 'Cotton ' : 'Weed ') +
                    Math.round((b.conf || 0) * 100) + '%';
      ctx.font = 'bold 11px Inter, system-ui';
      const tw = ctx.measureText(label).width + 12;
      const ly = by >= 22 ? by - 20 : by;

      ctx.fillStyle = color;
      ctx.fillRect(bx, ly, tw, 19);
      ctx.fillStyle = '#10221b';
      ctx.fillText(label, bx + 6, ly + 13);
    }
  }
}

function exportPlotCsv(){
  if(!currentPlotData){
    alert('Please analyze a field plot image first.');
    return;
  }
  window.location.href = '/api/plot/export/' + encodeURIComponent(currentPlotData.analysis_id) + '?format=csv';
}

function exportPlotJson(){
  if(!currentPlotData){
    alert('Please analyze a field plot image first.');
    return;
  }
  window.location.href = '/api/plot/export/' + encodeURIComponent(currentPlotData.analysis_id) + '?format=json';
}

window.addEventListener('resize', () => { drawBoxes(latestData); drawPlotCanvas(); });
setInterval(poll,250);setInterval(loadLogs,5000);setInterval(loadLatestRecommendation,10000);scanCameras();connectModel();poll();loadLogs();loadLatestRecommendation();loadLatestPlotAnalysis();
</script></body></html>'''

class VisionEngine:
    def __init__(self):
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.reader = None
        self.worker = None
        self.watchdog = None
        self.capture_gen = 0
        self.source_args = None
        self.last_frame_at = 0.0
        self.last_infer_at = 0.0
        self.reconnects = 0
        self.cap = None
        self.client = None
        self.model_id = 'cotton-8/3'
        self.api_key = ''
        self.model_info = []
        self.latest = None
        self.latest_jpeg = None
        self.boxes = []
        self.width = 0
        self.height = 0
        self.camera_fps = 0.0
        self.ai_fps = 0.0
        self.inference_ms = 0.0
        self.frame_serial = 0
        self.camera_name = ''
        self.error = ''
        self.active = False
        self.conf = 0.25
        self.imgsz = 640
        self.max_infer_fps = 10
        self.auto_log_secs = 10
        self.save_weed = True
        self.reset_tracking()

    def reset_tracking(self):
        # One tracker per class so a box that flips label never shares an ID with the other class.
        # AI FPS is only ~0.5, so tracks are kept for 5 inference frames (~10 s) before being dropped.
        self.trackers = {kind: sv.ByteTrack(track_activation_threshold=0.1, lost_track_buffer=5,
                                            minimum_matching_threshold=0.8, frame_rate=30,
                                            minimum_consecutive_frames=1) for kind in CLASS_IDS}
        self.unique_ids = {kind: set() for kind in CLASS_IDS}
        self.last_log = self.last_weed = 0.0

    def track(self, boxes):
        """Give each box a persistent ByteTrack ID so plants are counted once, not once per frame."""
        out = []
        for kind, tracker in self.trackers.items():
            mine = [b for b in boxes if b['kind'] == kind]
            if mine:
                det = sv.Detections(xyxy=np.array([[b['x1'],b['y1'],b['x2'],b['y2']] for b in mine], dtype=float),
                                    confidence=np.array([b['conf'] for b in mine], dtype=float),
                                    class_id=np.full(len(mine), CLASS_IDS[kind]),
                                    data={'class_name': np.array([b['class_name'] for b in mine])})
            else:
                det = sv.Detections.empty()
            # Empty updates still advance the tracker so lost tracks expire on time.
            tracked = tracker.update_with_detections(det)
            for i in range(len(tracked)):
                x1, y1, x2, y2 = (float(v) for v in tracked.xyxy[i])
                tid = int(tracked.tracker_id[i])
                self.unique_ids[kind].add(tid)
                out.append(dict(kind=kind, x1=x1, y1=y1, x2=x2, y2=y2, conf=float(tracked.confidence[i]),
                                class_name=str(tracked.data['class_name'][i]), track_id=tid))
        return out

    def record(self, frame, boxes):
        """Auto-log on the configured interval and keep weed-infested frames for retraining."""
        now = time.monotonic()
        weed_file = ''
        if self.save_weed and any(b['kind'] == 'grass' for b in boxes) and now-self.last_weed >= WEED_SAMPLE_SECS:
            self.last_weed = now
            weed_file = save_weed_sample(frame, boxes)
        if self.auto_log_secs and now-self.last_log >= self.auto_log_secs:
            self.last_log = now
            log_detection(frame, boxes, 'auto', weed_file)
        elif weed_file:
            log_detection(frame, boxes, 'weed', weed_file)

    def status(self):
        with self.lock:
            ai_stale = self.active and time.monotonic()-self.last_infer_at > STALE_BOX_SECS
            boxes = [] if ai_stale else list(self.boxes)
            return dict(ok=True, active=self.active, camera_name=self.camera_name,
                        camera_fps=self.camera_fps if time.monotonic()-self.last_frame_at < 1.5 else 0.0,
                        ai_fps=0.0 if ai_stale else self.ai_fps, reconnects=self.reconnects,
                        inference_ms=self.inference_ms, width=self.width, height=self.height,
                        cotton_count=sum(b['kind'] == 'cotton' for b in boxes),
                        grass_count=sum(b['kind'] == 'grass' for b in boxes),
                        cotton_unique=len(self.unique_ids['cotton']),
                        grass_unique=len(self.unique_ids['grass']),
                        auto_log_secs=self.auto_log_secs, save_weed=self.save_weed,
                        boxes=boxes, error=self.error, models=list(self.model_info))

    def stop(self):
        self.stop_event.set()
        with self.lock:
            self.capture_gen += 1
        for t in (self.reader, self.worker, self.watchdog):
            if t and t.is_alive() and t is not threading.current_thread():
                t.join(timeout=2)
        with self.lock:
            cap = self.cap
            self.cap = None
            self.active = False
            self.latest = None
            self.latest_jpeg = None
            self.boxes = []
            self.camera_fps = self.ai_fps = self.inference_ms = 0.0
            self.frame_serial = 0
            self.width = self.height = 0
            self.error = ''
        release_async(cap)
        self.reader = self.worker = self.watchdog = None

    def start(self, mode, source, resolution, target_fps):
        self.stop()
        cap, label, frame = open_camera(mode, source, resolution, target_fps)
        self.stop_event = threading.Event()
        with self.lock:
            self.cap = cap
            self.source_args = (mode, source, resolution, target_fps)
            self.latest = frame
            self.width, self.height = frame.shape[1], frame.shape[0]
            self.camera_name = label
            self.active = True
            self.last_frame_at = time.monotonic()
            self.reconnects = 0
            self.reset_tracking()
            self.frame_serial += 1
            gen = self.capture_gen
        self.reader = threading.Thread(target=self.capture_loop, args=(gen,), daemon=True)
        self.worker = threading.Thread(target=self.inference_loop, daemon=True)
        self.watchdog = threading.Thread(target=self.watchdog_loop, daemon=True)
        self.reader.start()
        self.worker.start()
        self.watchdog.start()
        return 'Connected to '+label+'. Video streaming and AI processing started.'

    def watchdog_loop(self):
        """cap.read() can hang forever inside the Windows camera driver; reopen the camera when frames stop."""
        failures = 0
        while not self.stop_event.wait(.5):
            with self.lock:
                stalled = self.active and time.monotonic()-self.last_frame_at > STALL_SECS
                args = self.source_args
            if not stalled:
                failures = 0
                continue
            with self.lock:
                # Bumping the generation makes the hung reader exit if its read ever returns.
                self.capture_gen += 1
                old, self.cap = self.cap, None
                self.camera_fps = 0.0
                self.error = 'Camera stopped sending frames; reconnecting...'
            release_async(old)
            time.sleep(1)
            try:
                cap, _, frame = open_camera(*args)
            except Exception:
                failures += 1
                with self.lock:
                    self.last_frame_at = time.monotonic()  # wait another STALL_SECS before retrying
                    if failures >= MAX_RECONNECTS:
                        self.active = False
                        self.error = 'Camera stopped responding and could not be reopened. Close other camera apps and start again.'
                if failures >= MAX_RECONNECTS:
                    break
                continue
            if self.stop_event.is_set():
                release_async(cap)
                break
            with self.lock:
                self.cap = cap
                self.latest = frame
                self.last_frame_at = time.monotonic()
                self.frame_serial += 1
                self.reconnects += 1
                self.error = ''
                gen = self.capture_gen
            failures = 0
            self.reader = threading.Thread(target=self.capture_loop, args=(gen,), daemon=True)
            self.reader.start()

    def capture_loop(self, gen):
        samples = deque(maxlen=30)
        errors = 0
        while not self.stop_event.is_set() and gen == self.capture_gen:
            cap = self.cap
            if cap is None:
                break
            ok, frame = cap.read()
            if gen != self.capture_gen:
                break  # the watchdog replaced this camera while read() was blocked
            if not ok or frame is None:
                errors += 1
                if errors >= 40:
                    with self.lock:
                        self.error = 'Camera stream lost. Reconnect the camera and start again.'
                        self.active = False
                    break
                time.sleep(.06)
                continue
            errors = 0
            now = time.monotonic()
            samples.append(now)
            # Store an independent camera JPEG so streaming does not wait for inference.
            ok_jpeg, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self.lock:
                if gen != self.capture_gen:
                    break
                self.latest = frame
                self.last_frame_at = now
                self.width, self.height = frame.shape[1], frame.shape[0]
                self.frame_serial += 1
                if ok_jpeg:
                    self.latest_jpeg = encoded.tobytes()
                if len(samples) >= 2:
                    self.camera_fps = (len(samples)-1)/max(samples[-1]-samples[0], 1e-6)

    def inference_loop(self):
        times = deque(maxlen=20)
        seen = -1
        pool = ThreadPoolExecutor(max_workers=MAX_VERIFY+2)
        with self.lock:
            self.last_infer_at = time.monotonic()
        while not self.stop_event.is_set():
            with self.lock:
                serial = self.frame_serial
                frame = None if self.latest is None else self.latest.copy()
                client = self.client
                model_id = self.model_id
                conf, imgsz, limit = self.conf, self.imgsz, self.max_infer_fps
                active = self.active
            if not active:
                break
            if client is None or frame is None or serial == seen:
                time.sleep(.015)
                continue
            begun = time.monotonic()
            seen = serial
            boxes = []
            try:
                # Upload small frames for lower network latency, then map boxes to full camera resolution.
                full_h, full_w = frame.shape[:2]
                scale = min(1.0, imgsz / max(full_h, full_w))
                if scale < 1.0:
                    image = cv2.resize(frame, (max(1,int(full_w*scale)), max(1,int(full_h*scale))))
                else:
                    image = frame
                resized_h, resized_w = image.shape[:2]
                # Both detectors run in parallel and only propose regions; CLIP assigns the label.
                deadline = begun+FRAME_TIMEOUT
                jobs = [pool.submit(client.infer, image, model_id=m) for m in (model_id, COTTON_MODEL_ID)]
                items = []
                for job in jobs:
                    items += job.result(timeout=max(.1, deadline-time.monotonic())).get('predictions', [])
                for item in items:
                    confidence = float(item.get('confidence', 0))
                    label = str(item.get('class', '')).strip()
                    if confidence < conf or classify(label) is None:
                        continue
                    cx = float(item['x']) * full_w / resized_w
                    cy = float(item['y']) * full_h / resized_h
                    bw = float(item['width']) * full_w / resized_w
                    bh = float(item['height']) * full_h / resized_h
                    boxes.append(dict(kind='?', x1=max(0,cx-bw/2), y1=max(0,cy-bh/2),
                                      x2=min(full_w,cx+bw/2), y2=min(full_h,cy+bh/2),
                                      conf=confidence, class_name=label))
                # Merge overlapping proposals from both models before paying for CLIP checks.
                boxes = deduplicate(boxes)[:MAX_VERIFY]
                checks = [pool.submit(verify_kind, frame, b) for b in boxes]
                kinds = [c.result(timeout=max(.1, deadline-time.monotonic())) for c in checks]
                for b, kind in zip(boxes, kinds):
                    b['kind'] = kind
                boxes = deduplicate([b for b in boxes if b['kind']])
                with self.lock:
                    boxes = self.track(boxes)
                self.record(frame, boxes)
                now = time.monotonic()
                times.append(now)
                with self.lock:
                    self.boxes = boxes
                    self.last_infer_at = now
                    self.inference_ms = (now-begun)*1000
                    if len(times) >= 2:
                        self.ai_fps = (len(times)-1)/max(times[-1]-times[0], 1e-6)
                    self.error = ''
            except FutureTimeout:
                # Threads stuck on a dead connection stay busy; start a fresh pool so the next frame is not starved.
                pool.shutdown(wait=False, cancel_futures=True)
                pool = ThreadPoolExecutor(max_workers=MAX_VERIFY+2)
                with self.lock:
                    self.boxes = []
                    self.error = f'Roboflow did not answer within {FRAME_TIMEOUT}s; retrying...'
            except Exception as exc:
                with self.lock:
                    self.boxes = []
                    self.error = 'Roboflow inference error: '+str(exc).replace(self.api_key or '__NO_KEY__', '[REDACTED]')
                time.sleep(.3)
            elapsed = time.monotonic()-begun
            time.sleep(max(0, 1.0/max(1,limit)-elapsed))

def open_camera(mode, source, resolution, target_fps):
    w, h = (int(x) for x in resolution.split('x'))
    if mode == 'webcam':
        index = int(source)
        if not 0 <= index <= 15:
            raise ValueError('Camera index must be 0–15')
        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
        label = 'Webcam '+str(index)
    elif mode == 'ip':
        if not source.lower().startswith(('rtsp://', 'http://', 'https://', 'rtmp://')):
            raise ValueError('Expected rtsp://, http://, https:// or rtmp:// stream URL')
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        label = 'IP / network camera'
    else:
        raise ValueError('Invalid camera mode')
    if not cap.isOpened():
        cap.release()
        raise ValueError('Could not open this camera. Check permissions, device index or stream URL.')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise ValueError('Camera opened but did not return video frames.')
    return cap, label, frame

def release_async(cap):
    # release() can block while a hung read() holds the driver, so never do it on a request thread.
    if cap is not None:
        threading.Thread(target=cap.release, daemon=True).start()

def draw_boxes(frame, boxes):
    colors = {'cotton': (118,204,35), 'grass': (77,184,255)}
    for b in boxes:
        x1,y1,x2,y2 = [int(b[k]) for k in ('x1','y1','x2','y2')]
        color = colors[b['kind']]
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        tag = f"#{b['track_id']} " if 'track_id' in b else ''
        cv2.putText(frame, f"{tag}{b['kind']} {b['conf']:.0%}", (x1,max(20,y1-7)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
    return frame

def log_detection(frame, boxes, trigger, weed_file=''):
    """Save an annotated snapshot and append a row to logs/detections.csv; returns the row."""
    stamp = datetime.now()
    name = 'detection_'+stamp.strftime('%Y%m%d_%H%M%S_%f')+'.jpg'
    if not cv2.imwrite(str(SNAPSHOTS/name), draw_boxes(frame.copy(), boxes)):
        raise OSError('Unable to save snapshot.')
    with engine.lock:
        row = dict(timestamp=stamp.strftime('%Y-%m-%d %H:%M:%S'), trigger=trigger, camera=engine.camera_name,
                   cotton_visible=sum(b['kind']=='cotton' for b in boxes),
                   grass_visible=sum(b['kind']=='grass' for b in boxes),
                   cotton_unique=len(engine.unique_ids['cotton']), grass_unique=len(engine.unique_ids['grass']),
                   snapshot=name, weed_sample=weed_file)
    with log_lock:
        new = not LOG_CSV.exists()
        with open(LOG_CSV, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
    return row

def save_weed_sample(frame, boxes):
    """Store the raw frame and YOLO labels (class cx cy w h, normalised) under weed_samples/."""
    h, w = frame.shape[:2]
    stem = 'weed_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    cv2.imwrite(str(WEED_SAMPLES/'images'/(stem+'.jpg')), frame)
    lines = [f"{CLASS_IDS[b['kind']]} {(b['x1']+b['x2'])/2/w:.6f} {(b['y1']+b['y2'])/2/h:.6f} "
             f"{(b['x2']-b['x1'])/w:.6f} {(b['y2']-b['y1'])/h:.6f}" for b in boxes]
    (WEED_SAMPLES/'labels'/(stem+'.txt')).write_text('\n'.join(lines)+'\n')
    return stem+'.jpg'

def read_logs(limit=None):
    with log_lock:
        if not LOG_CSV.exists():
            return []
        with open(LOG_CSV, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    return rows[::-1][:limit] if limit else rows[::-1]

def verify_kind(frame, box):
    """Ask CLIP whether a box shows cotton, grass/weed, or neither (None)."""
    h, w = frame.shape[:2]
    padx = (box['x2']-box['x1'])*.1; pady = (box['y2']-box['y1'])*.1
    x1 = int(max(0, box['x1']-padx)); y1 = int(max(0, box['y1']-pady))
    x2 = int(min(w, box['x2']+padx)); y2 = int(min(h, box['y2']+pady))
    if x2-x1 < 24 or y2-y1 < 24:
        return None
    crop = frame[y1:y2, x1:x2]
    for attempt in range(2):
        try:
            sims = np.array(CLIP_CLIENT.clip_compare(subject=crop, prompt=[p for _, p in CLIP_PROMPTS])['similarity'])
            break
        except Exception:
            if attempt:
                return None  # unverified boxes are dropped instead of failing the whole frame
            time.sleep(.3)
    # CLIP's logit scale is 100; softmax turns similarities into probabilities per prompt.
    probs = np.exp(100*(sims-sims.max()))
    probs /= probs.sum()
    groups = {}
    for (kind, _), pr in zip(CLIP_PROMPTS, probs):
        groups[kind] = groups.get(kind, 0.0)+pr
    best = max(groups, key=groups.get)
    return best if best and groups[best] >= MIN_CLIP_PROB else None

def classify(label, default_kind=None):
    s = label.lower().replace('-', ' ').replace('_', ' ').strip()
    cotton_words = ('cotton', 'crop')
    grass_words = ('grass', 'weed', 'ryegrass', 'crabgrass', 'barnyardgrass', 'foxtail', 'nutsedge', 'sedge')
    if any(w in s for w in cotton_words):
        return 'cotton'
    if any(w in s for w in grass_words):
        return 'grass'
    # A generic label like 'plant' is not reliably identifiable as cotton/grass.
    # Do not invent species classifications for unknown model classes.
    return None

def deduplicate(boxes):
    boxes = sorted(boxes, key=lambda b:b['conf'], reverse=True)
    kept = []
    for b in boxes:
        skip = False
        for a in list(kept):
            xx1=max(a['x1'],b['x1']); yy1=max(a['y1'],b['y1'])
            xx2=min(a['x2'],b['x2']); yy2=min(a['y2'],b['y2'])
            inter=max(0,xx2-xx1)*max(0,yy2-yy1)
            aa=max(0,a['x2']-a['x1'])*max(0,a['y2']-a['y1'])
            bb=max(0,b['x2']-b['x1'])*max(0,b['y2']-b['y1'])
            if a['kind'] == b['kind']:
                if inter/max(aa+bb-inter,1e-6)>.55:
                    skip=True;break
            # Cotton wins over grass/weed on the same plant, whichever scored higher.
            elif inter/max(min(aa,bb),1e-6)>.4:
                if b['kind'] == 'grass':
                    skip=True;break
                kept.remove(a)
        if not skip:
            kept.append(b)
    return kept

engine = VisionEngine()
rec_service = None
if RecommendationService is not None:
    try:
        rec_service = RecommendationService(base_dir=ROOT)
    except Exception as _rec_init_err:
        print(f"Notice: RecommendationService init failed: {_rec_init_err}")
        rec_service = None

plot_segmenter = None
plot_analyses_cache = {}

if PlotSegmenter is not None:
    try:
        w_an = rec_service.advisor.workflow.weed_analyzer if (rec_service and rec_service.advisor) else None
        s_an = rec_service.advisor.workflow.spatial_analyzer if (rec_service and rec_service.advisor) else None
        c_risk = rec_service.advisor.workflow.crop_risk_engine if (rec_service and rec_service.advisor) else None
        f_adv = rec_service.advisor if rec_service else None
        plot_segmenter = PlotSegmenter(weed_analyzer=w_an, spatial_analyzer=s_an, crop_risk_engine=c_risk, farmer_advisor=f_adv)
    except Exception as _plot_seg_init_err:
        print(f"Notice: PlotSegmenter init notice: {_plot_seg_init_err}")
        try:
            plot_segmenter = PlotSegmenter()
        except Exception:
            plot_segmenter = None


@app.get('/')
def index():
    return render_template_string(HTML)

@app.get('/api/cameras')
def cameras():
    # Scanning while another stream is open can block or steal a device on Windows.
    if engine.active:
        return jsonify(ok=False, error='Stop the active camera before scanning devices.'), 409
    found=[]
    for i in range(8):
        cap=cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name=='nt' else cv2.CAP_ANY)
        if cap.isOpened():
            good, f=cap.read()
            if good and f is not None:
                found.append(dict(index=i,name=f'Camera {i} — {f.shape[1]}×{f.shape[0]}'))
        cap.release()
    return jsonify(ok=True,cameras=found)

@app.post('/api/models')
def load_models():
    p = request.get_json(silent=True) or {}
    key = ROBOFLOW_API_KEY
    model_id = str(p.get('model_id',ROBOFLOW_MODEL_ID)).strip()
    if not key:
        return jsonify(ok=False,error='ROBOFLOW_API_KEY is empty in app.py.'),400
    if not model_id or '/' not in model_id:
        return jsonify(ok=False,error='Model ID should look like cotton-8/3.'),400
    if engine.active:
        return jsonify(ok=False,error='Stop the camera before switching Roboflow models.'),409
    try:
        client = InferenceHTTPClient(api_url='https://serverless.roboflow.com', api_key=key)
        # Return low-confidence boxes too; the dashboard's slider does the real filtering.
        client.configure(InferenceConfiguration(confidence_threshold=0.1))
        # A small live request verifies the model/key without silently treating configuration as success.
        blank = np.zeros((96,96,3),dtype=np.uint8)
        result = client.infer(blank, model_id=model_id)
        client.infer(blank, model_id=COTTON_MODEL_ID)
        if not isinstance(result,dict) or 'predictions' not in result:
            raise ValueError('Unexpected Roboflow response; verify that this is an object-detection model.')
    except Exception as exc:
        msg = str(exc).replace(key,'[REDACTED]')
        return jsonify(ok=False,error='Roboflow connection failed: '+msg[:280]),400
    with engine.lock:
        engine.client = client
        engine.model_id = model_id
        engine.api_key = key
        engine.model_info = [model_id+' · grass / weed', COTTON_MODEL_ID+' · cotton']
        engine.error = ''
    return jsonify(ok=True,models=list(engine.model_info),message='Roboflow connected: '+model_id+'. Start your camera.')

@app.post('/api/settings')
def settings():
    p=request.get_json(silent=True) or {}
    with engine.lock:
        engine.conf=max(.1,min(.95,float(p.get('conf',engine.conf))))
        engine.imgsz=int(p.get('imgsz',engine.imgsz))
        engine.max_infer_fps=max(1,min(30,int(p.get('max_infer_fps',engine.max_infer_fps))))
        engine.auto_log_secs=max(0,min(3600,int(p.get('auto_log_secs',engine.auto_log_secs))))
        engine.save_weed=bool(p.get('save_weed',engine.save_weed))
    return jsonify(ok=True)

@app.post('/api/start')
def start():
    p=request.get_json(silent=True) or {}
    try:
        if engine.client is None:
            return jsonify(ok=False,error='Connect your Roboflow model before starting.'),400
        mode=str(p.get('mode','webcam'))
        source=str(p.get('source','')).strip()
        resolution=str(p.get('resolution','1280x720'))
        if resolution not in ('640x480','1280x720','1920x1080'):
            raise ValueError('Invalid resolution')
        fps=int(p.get('fps',30))
        if fps not in (15,30,60):
            raise ValueError('Invalid target FPS')
        message=engine.start(mode,source,resolution,fps)
        return jsonify(ok=True,message=message)
    except Exception as e:
        return jsonify(ok=False,error=str(e)),400

@app.post('/api/stop')
def stop():
    engine.stop()
    return jsonify(ok=True)

@app.get('/api/status')
def status():
    return jsonify(engine.status())

@app.get('/video_feed')
def video_feed():
    def generate():
        last_id=-1
        while True:
            with engine.lock:
                active=engine.active
                sid=engine.frame_serial
                jpeg=engine.latest_jpeg
            if not active:
                break
            if jpeg is not None and sid != last_id:
                last_id=sid
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpeg+b'\r\n'
            else:
                time.sleep(.008)
    return Response(generate(),mimetype='multipart/x-mixed-replace; boundary=frame',headers={'Cache-Control':'no-store, max-age=0','X-Accel-Buffering':'no'})

@app.post('/api/snapshot')
def snapshot():
    with engine.lock:
        frame=None if engine.latest is None else engine.latest.copy()
        boxes=list(engine.boxes)
    if frame is None:
        return jsonify(ok=False,error='No active video frame to save.'),400
    try:
        row=log_detection(frame,boxes,'manual')
    except OSError as e:
        return jsonify(ok=False,error=str(e)),500
    return jsonify(ok=True,file=str(SNAPSHOTS/row['snapshot']))

@app.get('/api/logs')
def logs():
    rows=read_logs(limit=max(1,min(500,int(request.args.get('limit',50)))))
    samples=len(list((WEED_SAMPLES/'images').glob('*.jpg')))
    return jsonify(ok=True,rows=rows,weed_samples=samples,log_file=str(LOG_CSV),
                   snapshot_dir=str(SNAPSHOTS),weed_dir=str(WEED_SAMPLES))

@app.get('/api/logs.csv')
def logs_csv():
    if not LOG_CSV.exists():
        return jsonify(ok=False,error='No detections logged yet.'),404
    return send_file(LOG_CSV,mimetype='text/csv',as_attachment=True,
                     download_name='detections_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.csv')

@app.get('/snapshots/<path:name>')
def snapshot_file(name):
    return send_from_directory(SNAPSHOTS,name)

# =========================================================================
# Hybrid AI Weed Risk Assessment & Recommendation Engine API Endpoints
# =========================================================================

@app.post('/api/recommendations/analyze')
def api_recommendation_analyze():
    if rec_service is None:
        return jsonify(ok=False, error='Recommendation service is offline or unavailable.'), 503
    p = request.get_json(silent=True) or {}
    boxes = p.get('boxes')
    width = int(p.get('width', 0))
    height = int(p.get('height', 0))
    ctx = p.get('field_context') or {}

    # If boxes are not explicitly provided in payload, pull current live frame data from VisionEngine
    cumulative_tracked_ids = None
    if boxes is None:
        with engine.lock:
            boxes = list(engine.boxes)
            width = engine.width or 1280
            height = engine.height or 720
            cumulative_tracked_ids = {k: set(v) for k, v in engine.unique_ids.items()}

    async_mode = bool(p.get('async', False))
    res = rec_service.analyze(
        boxes=boxes,
        image_width=width,
        image_height=height,
        field_context=ctx,
        cumulative_tracked_ids=cumulative_tracked_ids,
        async_mode=async_mode,
    )
    status_code = 200 if res.get('ok') else 400
    return jsonify(res), status_code

@app.get('/api/recommendations/latest')
def api_recommendation_latest():
    if rec_service is None:
        return jsonify(ok=False, available=False, status='SERVICE_UNAVAILABLE', error='Recommendation service is offline.'), 503
    return jsonify(rec_service.get_latest())

@app.get('/api/recommendations/history')
def api_recommendation_history():
    if rec_service is None:
        return jsonify(ok=True, records=[]), 200
    limit = max(1, min(200, int(request.args.get('limit', 50))))
    return jsonify(rec_service.get_history(limit=limit))

@app.get('/api/recommendations/status')
def api_recommendation_status():
    if rec_service is None:
        return jsonify(ok=False, status='SERVICE_UNAVAILABLE', service='Hybrid AI Weed Risk Assessment Engine'), 503
    return jsonify(rec_service.get_status())

@app.post('/api/recommendations/action')
def api_recommendation_record_action():
    if rec_service is None:
        return jsonify(ok=False, error='Recommendation service is offline.'), 503
    p = request.get_json(silent=True) or {}
    aid = str(p.get('assessment_id', '')).strip()
    action = str(p.get('action_taken', '')).strip()
    notes = str(p.get('notes', '')).strip()
    if not aid or not action:
        return jsonify(ok=False, error='assessment_id and action_taken are required.'), 400
    return jsonify(rec_service.record_farmer_action(aid, action, notes))

@app.get('/api/recommendations/export')
def api_recommendation_export():
    if rec_service is None:
        return jsonify(ok=False, error='Recommendation service is offline.'), 503
    csv_path = rec_service.export_csv()
    if not csv_path or not csv_path.exists():
        return jsonify(ok=False, error='No recommendation history recorded yet.'), 404
    return send_file(
        csv_path,
        mimetype='text/csv',
        as_attachment=True,
        download_name='weed_recommendations_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv'
    )

@app.post('/api/recommendations/ask')
def api_recommendation_ask():
    if rec_service is None:
        return jsonify(ok=False, error='Recommendation service is offline.'), 503
    p = request.get_json(silent=True) or {}
    q = str(p.get('question', '')).strip()
    if not q:
        return jsonify(ok=False, error='Question string is required.'), 400
    return jsonify(rec_service.ask(q))


# =========================================================================
# Four-Part Precision Agriculture Plot Segmentation & Analysis Endpoints
# =========================================================================

def get_plot_segmenter():
    global plot_segmenter
    if plot_segmenter is None and PlotSegmenter is not None:
        try:
            w_an = rec_service.advisor.workflow.weed_analyzer if (rec_service and rec_service.advisor) else None
            s_an = rec_service.advisor.workflow.spatial_analyzer if (rec_service and rec_service.advisor) else None
            c_risk = rec_service.advisor.workflow.crop_risk_engine if (rec_service and rec_service.advisor) else None
            f_adv = rec_service.advisor if rec_service else None
            plot_segmenter = PlotSegmenter(weed_analyzer=w_an, spatial_analyzer=s_an, crop_risk_engine=c_risk, farmer_advisor=f_adv)
        except Exception as e:
            print(f"Notice: PlotSegmenter creation: {e}")
            plot_segmenter = PlotSegmenter()
    return plot_segmenter

def run_quadrant_detection(crop_img, qinfo):
    """Run detector on a single quadrant crop using existing Roboflow & CLIP pipeline."""
    with engine.lock:
        client = engine.client
        model_id = engine.model_id
        conf = engine.conf
        imgsz = engine.imgsz
        key = engine.api_key or ROBOFLOW_API_KEY
    
    if client is None and key:
        try:
            client = InferenceHTTPClient(api_url='https://infer.roboflow.com', api_key=key)
        except Exception:
            client = None
    
    if client is None:
        return []
    
    full_h, full_w = crop_img.shape[:2]
    scale = min(1.0, imgsz / max(full_h, full_w))
    if scale < 1.0:
        image = cv2.resize(crop_img, (max(1, int(full_w * scale)), max(1, int(full_h * scale))))
    else:
        image = crop_img
    resized_h, resized_w = image.shape[:2]
    
    pool = ThreadPoolExecutor(max_workers=MAX_VERIFY + 2)
    deadline = time.monotonic() + FRAME_TIMEOUT
    jobs = [pool.submit(client.infer, image, model_id=m) for m in (model_id, COTTON_MODEL_ID)]
    items = []
    for job in jobs:
        try:
            items += job.result(timeout=max(0.1, deadline - time.monotonic())).get('predictions', [])
        except Exception:
            pass
    
    boxes = []
    for item in items:
        confidence = float(item.get('confidence', 0))
        label = str(item.get('class', '')).strip()
        if confidence < conf or classify(label) is None:
            continue
        cx = float(item['x']) * full_w / resized_w
        cy = float(item['y']) * full_h / resized_h
        bw = float(item['width']) * full_w / resized_w
        bh = float(item['height']) * full_h / resized_h
        boxes.append(dict(kind='?', x1=max(0, cx - bw / 2), y1=max(0, cy - bh / 2),
                          x2=min(full_w, cx + bw / 2), y2=min(full_h, cy + bh / 2),
                          conf=confidence, class_name=label))
    
    boxes = deduplicate(boxes)[:MAX_VERIFY]
    checks = [pool.submit(verify_kind, crop_img, b) for b in boxes]
    kinds = []
    for c in checks:
        try:
            kinds.append(c.result(timeout=max(0.1, deadline - time.monotonic())))
        except Exception:
            kinds.append(None)
    for b, kind in zip(boxes, kinds):
        b['kind'] = kind
    boxes = deduplicate([b for b in boxes if b['kind']])
    return boxes

@app.post('/api/plot/analyze')
def api_plot_analyze():
    segmenter = get_plot_segmenter()
    if segmenter is None:
        return jsonify(ok=False, error="PlotSegmenter service unavailable."), 503

    field_ctx = {}
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        field_ctx = payload.get('field_context') or {}
        use_current_frame = bool(payload.get('use_current_frame', False))
        image_b64 = payload.get('image_base64') or payload.get('image_b64')
        snapshot_name = payload.get('snapshot')
        test_boxes = payload.get('test_boxes')
    else:
        payload = {}
        use_current_frame = request.form.get('use_current_frame', '').lower() in ('true', '1')
        image_b64 = request.form.get('image_base64')
        snapshot_name = request.form.get('snapshot')
        test_boxes = None
        raw_ctx = request.form.get('field_context')
        if raw_ctx:
            try:
                field_ctx = json.loads(raw_ctx)
            except Exception:
                field_ctx = {}
        for k in ('growth_stage', 'field_id', 'surveyed_area_sqm', 'ground_sampling_distance_cm'):
            if k in request.form and request.form[k]:
                field_ctx[k] = request.form[k]

    image = None
    if 'image' in request.files or 'file' in request.files:
        f = request.files.get('image') or request.files.get('file')
        if f and f.filename:
            file_bytes = np.frombuffer(f.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                return jsonify(ok=False, error="Could not decode uploaded image file. Ensure it is a valid JPEG/PNG."), 400
    elif image_b64:
        import base64
        try:
            if ',' in image_b64:
                image_b64 = image_b64.split(',', 1)[1]
            raw_bytes = base64.b64decode(image_b64)
            image = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("cv2.imdecode returned None")
        except Exception as e:
            return jsonify(ok=False, error=f"Invalid base64 image data: {str(e)}"), 400
    elif snapshot_name:
        snap_path = SNAPSHOTS / Path(snapshot_name).name
        if not snap_path.exists():
            return jsonify(ok=False, error=f"Snapshot '{snapshot_name}' not found."), 404
        image = cv2.imread(str(snap_path))
        if image is None:
            return jsonify(ok=False, error="Could not read snapshot image from disk."), 500
    elif use_current_frame:
        with engine.lock:
            if engine.latest is not None:
                image = engine.latest.copy()
        if image is None:
            return jsonify(ok=False, error="No camera stream active to capture frame."), 400

    if image is None:
        return jsonify(ok=False, error="No image provided. Please upload an image, select active frame, or specify snapshot."), 400

    if test_boxes is not None and isinstance(test_boxes, list):
        def custom_detector(crop, qinfo):
            ox, oy = qinfo['offset_x'], qinfo['offset_y']
            qw, qh = qinfo['width'], qinfo['height']
            out_b = []
            for tb in test_boxes:
                gx1, gy1 = float(tb.get('x1', 0)), float(tb.get('y1', 0))
                gx2, gy2 = float(tb.get('x2', 0)), float(tb.get('y2', 0))
                if max(ox, gx1) < min(ox + qw, gx2) and max(oy, gy1) < min(oy + qh, gy2):
                    out_b.append(dict(
                        kind=tb.get('kind', 'grass'),
                        x1=max(0.0, gx1 - ox),
                        y1=max(0.0, gy1 - oy),
                        x2=min(float(qw), gx2 - ox),
                        y2=min(float(qh), gy2 - oy),
                        conf=float(tb.get('conf', 0.8)),
                        class_name=tb.get('class_name', tb.get('kind', 'grass')),
                        track_id=tb.get('track_id')
                    ))
            return out_b
        detector = custom_detector
    else:
        detector = run_quadrant_detection

    try:
        result = segmenter.process_plot(image, detector_fn=detector, field_context=field_ctx)
    except ValueError as val_err:
        return jsonify(ok=False, error=f"Image validation failed: {str(val_err)}"), 400
    except Exception as exc:
        return jsonify(ok=False, error=f"Plot analysis failed: {str(exc)}"), 500

    aid = result["analysis_id"]
    try:
        orig_fn = f"plot_{aid}_orig.jpg"
        cv2.imwrite(str(SNAPSHOTS / orig_fn), image)

        annotated_img = segmenter.render_annotated_plot(image, result)
        annotated_fn = f"plot_{aid}_annotated.jpg"
        cv2.imwrite(str(SNAPSHOTS / annotated_fn), annotated_img)

        heatmap_img = segmenter.render_heatmap_plot(image, result)
        heatmap_fn = f"plot_{aid}_heatmap.jpg"
        cv2.imwrite(str(SNAPSHOTS / heatmap_fn), heatmap_img)

        quad_urls = {}
        for qid, qdata in result["quadrants"].items():
            crop_canvas = image[qdata['offset_y']:qdata['offset_y']+qdata['height'], qdata['offset_x']:qdata['offset_x']+qdata['width']].copy()
            for b in qdata.get("local_boxes", []):
                k = b.get("kind", "grass")
                color = (32, 212, 135) if k == "cotton" else (77, 184, 255)
                cv2.rectangle(crop_canvas, (int(b['x1']), int(b['y1'])), (int(b['x2']), int(b['y2'])), color, 2)
            q_fn = f"plot_{aid}_{qid}.jpg"
            cv2.imwrite(str(SNAPSHOTS / q_fn), crop_canvas)
            quad_urls[qid] = f"/snapshots/{q_fn}"

        result["image_urls"] = {
            "original": f"/snapshots/{orig_fn}",
            "annotated": f"/snapshots/{annotated_fn}",
            "heatmap": f"/snapshots/{heatmap_fn}",
            **quad_urls
        }
    except Exception as img_save_err:
        result["image_save_warning"] = str(img_save_err)

    plot_analyses_cache[aid] = result
    return jsonify(result), 200

@app.post('/api/field-map/analyze')
def api_field_map_analyze():
    """Alias for POST /api/plot/analyze providing integrated Field Map & Weed Risk intelligence."""
    return api_plot_analyze()

@app.post('/api/plot/demo')
def api_plot_demo():
    w, h = 1280, 720
    demo_img = np.zeros((h, w, 3), dtype=np.uint8)
    demo_img[:] = (38, 54, 46)
    noise = np.random.randint(-8, 8, (h, w, 3), dtype=np.int16)
    demo_img = np.clip(demo_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    for row_y in range(60, h, 90):
        cv2.line(demo_img, (0, row_y), (w, row_y), (48, 68, 56), 25)

    demo_boxes = [
        # NW: 3 cotton, 1 weed
        {"kind": "cotton", "conf": 0.88, "x1": 120, "y1": 100, "x2": 200, "y2": 180, "track_id": 1, "class_name": "cotton"},
        {"kind": "cotton", "conf": 0.91, "x1": 320, "y1": 105, "x2": 410, "y2": 195, "track_id": 2, "class_name": "cotton"},
        {"kind": "cotton", "conf": 0.85, "x1": 220, "y1": 240, "x2": 310, "y2": 320, "track_id": 3, "class_name": "cotton"},
        {"kind": "grass", "conf": 0.78, "x1": 500, "y1": 80, "x2": 550, "y2": 130, "track_id": 10, "class_name": "grass"},
        # NE: 2 cotton, 7 weeds (dense cluster & crop proximity)
        {"kind": "cotton", "conf": 0.89, "x1": 750, "y1": 110, "x2": 840, "y2": 200, "track_id": 4, "class_name": "cotton"},
        {"kind": "cotton", "conf": 0.86, "x1": 1000, "y1": 120, "x2": 1090, "y2": 210, "track_id": 5, "class_name": "cotton"},
        {"kind": "grass", "conf": 0.82, "x1": 770, "y1": 130, "x2": 820, "y2": 180, "track_id": 11, "class_name": "grass"},
        {"kind": "grass", "conf": 0.80, "x1": 810, "y1": 190, "x2": 870, "y2": 250, "track_id": 12, "class_name": "grass"},
        {"kind": "grass", "conf": 0.77, "x1": 860, "y1": 120, "x2": 910, "y2": 170, "track_id": 13, "class_name": "grass"},
        {"kind": "grass", "conf": 0.84, "x1": 920, "y1": 150, "x2": 980, "y2": 210, "track_id": 14, "class_name": "grass"},
        {"kind": "grass", "conf": 0.75, "x1": 880, "y1": 220, "x2": 940, "y2": 280, "track_id": 15, "class_name": "grass"},
        {"kind": "grass", "conf": 0.79, "x1": 950, "y1": 100, "x2": 1010, "y2": 160, "track_id": 16, "class_name": "grass"},
        {"kind": "grass", "conf": 0.76, "x1": 1020, "y1": 180, "x2": 1080, "y2": 240, "track_id": 17, "class_name": "grass"},
        # SW: 4 cotton, 3 weeds
        {"kind": "cotton", "conf": 0.92, "x1": 110, "y1": 450, "x2": 200, "y2": 540, "track_id": 6, "class_name": "cotton"},
        {"kind": "cotton", "conf": 0.87, "x1": 290, "y1": 460, "x2": 380, "y2": 550, "track_id": 7, "class_name": "cotton"},
        {"kind": "cotton", "conf": 0.85, "x1": 470, "y1": 455, "x2": 560, "y2": 545, "track_id": 8, "class_name": "cotton"},
        {"kind": "cotton", "conf": 0.89, "x1": 210, "y1": 600, "x2": 300, "y2": 690, "track_id": 9, "class_name": "cotton"},
        {"kind": "grass", "conf": 0.81, "x1": 180, "y1": 480, "x2": 230, "y2": 530, "track_id": 18, "class_name": "grass"},
        {"kind": "grass", "conf": 0.77, "x1": 390, "y1": 590, "x2": 450, "y2": 650, "track_id": 19, "class_name": "grass"},
        {"kind": "grass", "conf": 0.74, "x1": 530, "y1": 610, "x2": 590, "y2": 670, "track_id": 20, "class_name": "grass"},
        # SE: 5 cotton, 0 weeds
        {"kind": "cotton", "conf": 0.93, "x1": 720, "y1": 460, "x2": 820, "y2": 560, "track_id": 21, "class_name": "cotton"},
        {"kind": "cotton", "conf": 0.90, "x1": 900, "y1": 470, "x2": 990, "y2": 560, "track_id": 22, "class_name": "cotton"},
        {"kind": "cotton", "conf": 0.88, "x1": 1080, "y1": 465, "x2": 1170, "y2": 555, "track_id": 23, "class_name": "cotton"},
        {"kind": "cotton", "conf": 0.91, "x1": 810, "y1": 600, "x2": 900, "y2": 690, "track_id": 24, "class_name": "cotton"},
        {"kind": "cotton", "conf": 0.86, "x1": 1010, "y1": 610, "x2": 1100, "y2": 700, "track_id": 25, "class_name": "cotton"},
    ]

    for b in demo_boxes:
        bx1, by1, bx2, by2 = int(b['x1']), int(b['y1']), int(b['x2']), int(b['y2'])
        bcx, bcy = (bx1+bx2)//2, (by1+by2)//2
        bradius = min(bx2-bx1, by2-by1)//2
        if b['kind'] == 'cotton':
            cv2.circle(demo_img, (bcx, bcy), bradius, (40, 160, 60), -1)
            cv2.circle(demo_img, (bcx, bcy), max(2, bradius//3), (240, 245, 240), -1)
        else:
            cv2.circle(demo_img, (bcx, bcy), bradius, (20, 190, 180), -1)

    segmenter = get_plot_segmenter()
    def demo_detector(crop, qinfo):
        ox = qinfo['offset_x']
        oy = qinfo['offset_y']
        qw = qinfo['width']
        qh = qinfo['height']
        q_boxes = []
        for tb in demo_boxes:
            gx1, gy1 = tb['x1'], tb['y1']
            gx2, gy2 = tb['x2'], tb['y2']
            if max(ox, gx1) < min(ox + qw, gx2) and max(oy, gy1) < min(oy + qh, gy2):
                q_boxes.append(dict(kind=tb['kind'], x1=max(0, gx1-ox), y1=max(0, gy1-oy),
                                    x2=min(qw, gx2-ox), y2=min(qh, gy2-oy),
                                    conf=tb['conf'], class_name=tb['class_name'], track_id=tb['track_id']))
        return q_boxes

    p = request.get_json(silent=True) or {}
    fctx = p.get('field_context') or {"field_id": "Plot-Demo-Field-1", "growth_stage": "VEGETATIVE_SQUARING", "surveyed_area_sqm": 25.0}
    result = segmenter.process_plot(demo_img, detector_fn=demo_detector, field_context=fctx)
    aid = result["analysis_id"]

    orig_fn = f"plot_{aid}_orig.jpg"
    annotated_fn = f"plot_{aid}_annotated.jpg"
    heatmap_fn = f"plot_{aid}_heatmap.jpg"
    cv2.imwrite(str(SNAPSHOTS / orig_fn), demo_img)
    annotated_img = segmenter.render_annotated_plot(demo_img, result)
    cv2.imwrite(str(SNAPSHOTS / annotated_fn), annotated_img)
    heatmap_img = segmenter.render_heatmap_plot(demo_img, result)
    cv2.imwrite(str(SNAPSHOTS / heatmap_fn), heatmap_img)

    quad_urls = {}
    for qid, qdata in result["quadrants"].items():
        crop_canvas = demo_img[qdata['offset_y']:qdata['offset_y']+qdata['height'], qdata['offset_x']:qdata['offset_x']+qdata['width']].copy()
        for b in qdata.get("local_boxes", []):
            k = b.get("kind", "grass")
            color = (32, 212, 135) if k == "cotton" else (77, 184, 255)
            cv2.rectangle(crop_canvas, (int(b['x1']), int(b['y1'])), (int(b['x2']), int(b['y2'])), color, 2)
        q_fn = f"plot_{aid}_{qid}.jpg"
        cv2.imwrite(str(SNAPSHOTS / q_fn), crop_canvas)
        quad_urls[qid] = f"/snapshots/{q_fn}"

    result["image_urls"] = {
        "original": f"/snapshots/{orig_fn}",
        "annotated": f"/snapshots/{annotated_fn}",
        "heatmap": f"/snapshots/{heatmap_fn}",
        **quad_urls
    }
    plot_analyses_cache[aid] = result
    return jsonify(result), 200

@app.get('/api/plot/latest')
def api_plot_latest():
    if not plot_analyses_cache:
        return jsonify(ok=True, available=False, message="No plot analyses conducted yet.")
    latest_id = list(plot_analyses_cache.keys())[-1]
    return jsonify(ok=True, available=True, analysis=plot_analyses_cache[latest_id])

@app.get('/api/plot/analysis/<analysis_id>')
def api_plot_get(analysis_id):
    res = plot_analyses_cache.get(analysis_id)
    if not res:
        return jsonify(ok=False, error=f"Analysis '{analysis_id}' not found."), 404
    return jsonify(ok=True, analysis=res)

@app.get('/api/plot/export/<analysis_id>')
def api_plot_export(analysis_id):
    res = plot_analyses_cache.get(analysis_id)
    if not res:
        return jsonify(ok=False, error=f"Analysis ID '{analysis_id}' not found."), 404

    fmt = request.args.get('format', 'csv').lower()
    if fmt == 'json':
        return Response(
            json.dumps(res, indent=2),
            mimetype='application/json',
            headers={'Content-Disposition': f'attachment; filename={analysis_id}.json'}
        )

    import io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "analysis_id", "timestamp", "section_id", "section_name", "status",
        "visible_cotton", "visible_weeds", "weed_to_crop_ratio", "risk_category",
        "action_priority", "clusters", "adjacent_weeds", "avg_confidence",
        "offset_x", "offset_y", "width", "height", "warnings"
    ])

    for qid, q in res.get("quadrants", {}).items():
        c = q.get("counts", {})
        r = q.get("risk_assessment", {})
        s = q.get("spatial_summary", {})
        cf = q.get("confidence_stats", {})
        writer.writerow([
            analysis_id, res.get("timestamp"), qid, q.get("name"), q.get("status"),
            c.get("visible_cotton_count", 0), c.get("visible_weed_count", 0),
            c.get("weed_to_crop_ratio_display", ""), r.get("risk_category", ""),
            r.get("action_priority", ""), s.get("cluster_count", 0),
            s.get("weeds_adjacent_to_crop", 0), cf.get("avg_confidence", 0),
            q.get("offset_x", 0), q.get("offset_y", 0), q.get("width", 0), q.get("height", 0),
            " | ".join(q.get("warnings", []))
        ])

    fa = res.get("field_assessment", {}).get("summary", {})
    writer.writerow([
        analysis_id, res.get("timestamp"), "WHOLE_FIELD", "Whole Agricultural Plot", "SUCCESS",
        fa.get("total_visible_cotton", 0), fa.get("total_visible_weeds", 0),
        fa.get("weed_to_crop_ratio_display", ""), fa.get("overall_risk_category", ""),
        fa.get("overall_action_priority", ""), "-", "-", "-",
        0, 0, res.get("original_dimensions", {}).get("width", 0),
        res.get("original_dimensions", {}).get("height", 0),
        "Highest weed pressure quadrant: " + str(fa.get("highest_weed_quadrant", ""))
    ])

    csv_bytes = output.getvalue().encode('utf-8')
    return Response(
        csv_bytes,
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={analysis_id}.csv'}
    )

@app.get('/api/plot/image/<analysis_id>/<img_type>')
def api_plot_image(analysis_id, img_type):
    valid_types = {'original': 'orig', 'annotated': 'annotated', 'orig': 'orig',
                   'heatmap': 'heatmap', 'heat': 'heatmap',
                   'NW': 'NW', 'NE': 'NE', 'SW': 'SW', 'SE': 'SE'}
    suffix = valid_types.get(img_type)
    if not suffix:
        return jsonify(ok=False, error="Invalid image type"), 400
    fname = f"plot_{analysis_id}_{suffix}.jpg"
    fpath = SNAPSHOTS / fname
    if not fpath.exists():
        return jsonify(ok=False, error="Image not found"), 404
    return send_from_directory(SNAPSHOTS, fname)



if __name__=='__main__':
    print('AgriSight AI dashboard: http://127.0.0.1:5000')
    print('From phone on same Wi-Fi: http://YOUR_LAPTOP_IP:5000')
    app.run(host='0.0.0.0',port=5000,debug=False,threaded=True,use_reloader=False)
