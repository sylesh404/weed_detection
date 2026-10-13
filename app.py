# rtsp://10.154.194.228:554/stream1

"""Single-file Cotton & Grass/Weed YOLO Dashboard.
Install: pip install flask opencv-python numpy inference-sdk
Run: python cotton_grass_dashboard.py
Local: http://127.0.0.1:5000
"""
import os
import csv
import time
import socket
import threading
from pathlib import Path
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, render_template_string, send_file, send_from_directory
import supervision as sv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

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
@media(max-width:680px){.shell{display:block}.side{height:66px;padding:10px 16px;flex-direction:row;border-right:0;border-bottom:1px solid var(--line);justify-content:space-between}.brand span{display:block;font-size:19px}.side .nav{display:none}.main{padding:18px 12px}.top h1{font-size:22px}.top{align-items:start}.desc{font-size:12px}.metrics{grid-template-columns:repeat(2,minmax(0,1fr));gap:9px}.metric{padding:14px}.metric strong{font-size:23px}.cardhead,.pad{padding:16px}.form-row{grid-template-columns:1fr}.videofoot{align-items:start}.livepill{font-size:10px;padding:8px}}
</style></head><body>
<svg style="display:none" xmlns="http://www.w3.org/2000/svg"><symbol id="leaf" viewBox="0 0 24 24"><path d="M20 4C10 3 4 8 4 15a5 5 0 0 0 5 5c8 0 12-8 11-16Z M4 20c4-6 7-8 12-11" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></symbol><symbol id="cam" viewBox="0 0 24 24"><path d="M3 7h4l2-3h6l2 3h4v13H3V7Zm9 4a4 4 0 1 0 0 8 4 4 0 0 0 0-8Z" fill="none" stroke="currentColor" stroke-width="1.8"/></symbol><symbol id="radar" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/><path d="m12 12 6-6M12 3v3M3 12h3" fill="none" stroke="currentColor" stroke-width="2"/></symbol></svg>
<div class="shell"><aside class="side"><div class="brand"><div class="logo"><svg class="i" width="24" height="24"><use href="#leaf"/></svg></div><span>AgriSight<span style="color:#a7b2aa"> AI</span></span></div><div><div class="side-label">WORKSPACE</div><div class="nav"><svg class="i"><use href="#radar"/></svg><span>Live Detection</span></div></div><div class="spacer"></div><div class="tip"><b style="color:#2b5940">Field-ready workflow</b><p>Connect a camera, connect the Roboflow Cotton 8 API, then start the AI detection stream.</p></div><div class="side-footer">COTTON / GRASS VISION · v1.0</div></aside>
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
setInterval(poll,250);setInterval(loadLogs,5000);window.addEventListener('resize',()=>drawBoxes(latestData));scanCameras();connectModel();poll();loadLogs();
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

if __name__=='__main__':
    print('AgriSight AI dashboard: http://127.0.0.1:5000')
    print('From phone on same Wi-Fi: http://YOUR_LAPTOP_IP:5000')
    app.run(host='0.0.0.0',port=5000,debug=False,threaded=True,use_reloader=False)
