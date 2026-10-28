"""Private, Codespaces-friendly video clipper.

The uploaded source stays in this Codespace only while the job is available.
Delete it from the page when finished; abandoned jobs expire automatically.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file
from werkzeug.utils import secure_filename

BASE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("CLIP_WORK_DIR", BASE / "work")).resolve()
OUT = Path(os.environ.get("CLIPS_DIR", BASE / "clips")).resolve()
WORK.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_HOURS", "6")) * 3600
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_GB", "20")) * 1024**3
MAX_CLIP_SECONDS = 180

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
jobs: dict[str, dict] = {}
_model = None
_model_lock = threading.Lock()

# These features are intentionally explainable heuristics, not a claim that a
# local speech model understands the video's story or visual context.
HOOKS = {
    "why": 1.2, "how": 1.0, "never": 1.2, "truth": 1.4, "secret": 1.4,
    "actually": 1.0, "mistake": 1.3, "biggest": 1.0, "worst": 1.0,
    "best": 0.8, "nobody": 1.1, "everyone": 0.6, "problem": 0.8,
    "wrong": 0.9, "real": 0.5, "miért": 1.2, "hogyan": 1.0,
    "soha": 1.2, "igazság": 1.4, "titok": 1.4, "valójában": 1.0,
    "hiba": 1.3, "legnagyobb": 1.0, "senki": 1.1, "mindenki": 0.6,
    "probléma": 0.8, "rossz": 0.9, "valódi": 0.5,
}


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from faster_whisper import WhisperModel
                _model = WhisperModel(
                    MODEL_SIZE, device="cpu", compute_type="int8",
                    cpu_threads=max(1, min(os.cpu_count() or 2, 8)),
                )
    return _model


def score(text: str, duration: float) -> float:
    clean = text.casefold()
    tokens = re.findall(r"\w+", clean, flags=re.UNICODE)
    if not tokens:
        return -100.0
    score_value = sum(weight for word, weight in HOOKS.items()
                      if re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", clean))
    score_value += min(clean.count("?"), 3) * 1.25
    score_value += min(len(re.findall(r"\d+", clean)), 2) * 0.7
    # Prefer a complete thought and an opening that sounds self-contained.
    if clean.rstrip().endswith((".", "?", "!")):
        score_value += 0.8
    if re.match(r"\s*(so|but|and|because|then|de|és|mert|akkor)\b", clean):
        score_value -= 0.8
    score_value -= abs(duration - 40) / 28
    # Penalize very sparse or unusually dense transcript windows.
    words_per_second = len(tokens) / max(duration, 1)
    if words_per_second < 0.7:
        score_value -= 0.5
    elif words_per_second > 5.0:
        score_value -= 0.5
    return round(score_value, 2)


def find_candidates(segs, lo=20, hi=60, top=12):
    """Build overlapping sentence-aligned windows, then rank transparently."""
    candidates = []
    # Candidate starts advance in roughly 12-second steps so a great thought
    # is not lost just because it did not begin at an arbitrary transcript edge.
    starts = list(range(0, len(segs), 1))
    next_allowed = -1.0
    for i in starts:
        start = float(segs[i]["start"])
        if start < next_allowed:
            continue
        best_j = None
        for j in range(i, len(segs)):
            end = float(segs[j]["end"])
            duration = end - start
            if duration > hi:
                break
            if duration >= lo and segs[j]["text"].rstrip().endswith((".", "?", "!")):
                best_j = j
                # Prefer a complete sentence near 40 sec; don't keep extending.
                if duration >= 32:
                    break
        if best_j is None:
            continue
        end = float(segs[best_j]["end"])
        duration = end - start
        text = " ".join(s["text"] for s in segs[i:best_j + 1]).strip()
        candidates.append({"start": round(start, 2), "end": round(end, 2),
                           "text": text, "score": score(text, duration)})
        next_allowed = start + 12
    candidates.sort(key=lambda item: item["score"], reverse=True)
    # Don't show near-duplicate windows as separate top picks.
    picked = []
    for candidate in candidates:
        if any(abs(candidate["start"] - prev["start"]) < 12 for prev in picked):
            continue
        candidate["id"] = len(picked)
        picked.append(candidate)
        if len(picked) == top:
            break
    return picked


def transcribe(job_id: str, language: str | None):
    job = jobs.get(job_id)
    if not job:
        return
    try:
        job["status"] = "transcribing"
        segments, info = get_model().transcribe(
            str(job["src"]), language=language or None, word_timestamps=True,
            vad_filter=True, beam_size=1,
        )
        output = []
        for segment in segments:
            words = [{"w": word.word.strip(), "s": word.start, "e": word.end}
                     for word in (segment.words or [])]
            output.append({"text": segment.text.strip(), "start": segment.start,
                           "end": segment.end, "words": words})
            job["progress"] = min(99, int(100 * segment.end / max(info.duration, 1)))
        job["duration"] = float(info.duration or 0)
        job["segments"] = output
        job["candidates"] = find_candidates(output)
        job["status"] = "ready" if output else "error"
        if not output:
            job["error"] = "No speech was detected. Try a different language or check the video audio."
    except Exception as exc:  # report the actual backend error to the UI
        job["status"] = "error"
        job["error"] = str(exc)
    finally:
        job["updated"] = time.time()


def ass_timestamp(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, rem = divmod(centiseconds, 360000)
    minutes, rem = divmod(rem, 6000)
    whole, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{whole:02d}.{cs:02d}"


def make_ass(segments, start, end, path):
    words = [word for segment in segments for word in segment["words"]
             if word["s"] >= start - 0.05 and word["e"] <= end + 0.05 and word["w"]]
    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, "
        "Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,78,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,"
        "1,1,3,2,60,60,300,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = []
    for i in range(0, len(words), 3):
        group = words[i:i + 3]
        a = max(0, group[0]["s"] - start)
        b = min(end - start, group[-1]["e"] - start)
        if i + 3 < len(words):
            b = max(b, min(end - start, words[i + 3]["s"] - start))
        text = " ".join(word["w"] for word in group).replace("{", "").replace("}", "").upper()
        lines.append(f"Dialogue: 0,{ass_timestamp(a)},{ass_timestamp(b)},Default,,0,0,0,,{text}")
    Path(path).write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


def cleanup_expired():
    now = time.time()
    for job_id, job in list(jobs.items()):
        if now - job["updated"] > JOB_TTL_SECONDS and job["status"] not in ("queued", "transcribing"):
            shutil.rmtree(job["dir"], ignore_errors=True)
            jobs.pop(job_id, None)


def clean_stale_workdirs():
    """A server restart must not leave old uploads occupying Codespace disk."""
    cutoff = time.time() - JOB_TTL_SECONDS
    for path in WORK.iterdir():
        try:
            if path.is_dir() and path.stat().st_mtime < cutoff:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


def housekeeping():
    while True:
        time.sleep(900)
        cleanup_expired()
        active_dirs = {job["dir"] for job in jobs.values()}
        cutoff = time.time() - JOB_TTL_SECONDS
        for path in WORK.iterdir():
            try:
                if path.is_dir() and path not in active_dirs and path.stat().st_mtime < cutoff:
                    shutil.rmtree(path, ignore_errors=True)
            except OSError:
                pass


clean_stale_workdirs()
threading.Thread(target=housekeeping, daemon=True).start()


PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>My Clipper</title>
<style>
  :root { color-scheme: dark; --bg:#111; --card:#1c1c1e; --fg:#f2f2f2; --mut:#aaa; --acc:#e10600; }
  * { box-sizing:border-box }
  body { margin:0; font-family:system-ui,-apple-system,sans-serif; background:var(--bg); color:var(--fg) }
  main { max-width:900px; margin:0 auto; padding:24px 16px 80px }
  h1 { font-size:24px } h2 { font-size:18px; margin:0 0 12px }
  .card { background:var(--card); border-radius:12px; padding:16px; margin:16px 0 }
  button { background:var(--acc); color:white; border:0; border-radius:8px; padding:10px 16px; font-size:15px; cursor:pointer }
  button:disabled { opacity:.55; cursor:wait } button.sec { background:#333 }
  a { color:#ff8b85 }
  input,select { background:#2a2a2c; color:var(--fg); border:1px solid #555; border-radius:6px; padding:7px 8px }
  input[type=number] { width:100px } input[type=range] { vertical-align:middle }
  video { width:100%; max-height:340px; background:#000; border-radius:8px }
  .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-top:10px }
  .mut { color:var(--mut); font-size:13px; line-height:1.5 }
  .bar { height:8px; background:#333; border-radius:4px; overflow:hidden; margin-top:9px }
  .bar div { height:100%; background:var(--acc); width:0; transition:width .3s }
  #message { white-space:pre-wrap; overflow-wrap:anywhere }
  .candidate-text { line-height:1.5 }
  .warning { border-left:3px solid #e0a400; padding-left:10px }
</style>
</head>
<body>
<main>
  <h1>My Clipper</h1>
  <section class="card" id="upload-card">
    <h2>Choose a video</h2>
    <p class="mut warning">This app runs in your GitHub Codespace. Your video is uploaded to that Codespace while you work on it. Delete the source when you finish. Unfinished jobs are removed after 6 hours; exported clips stay in the Codespace until you download them or delete them.</p>
    <div class="row">
      <input type="file" id="file" accept="video/mp4,video/quicktime,video/x-matroska,video/webm,video/x-m4v,video/x-msvideo">
      <select id="lang" aria-label="Video language">
        <option value="">Auto-detect language</option><option value="hu">Hungarian</option><option value="en">English</option>
      </select>
      <button id="go">Upload &amp; analyse</button>
    </div>
    <p class="mut">30-minute videos are supported. Processing time depends on the Codespace CPU. The first run downloads the speech model and can take longer. The app checks available storage before saving an upload.</p>
  </section>

  <section class="card" id="progress" hidden>
    <div id="message" role="status" aria-live="polite">Waiting…</div>
    <div class="bar"><div id="fill"></div></div>
  </section>

  <section id="result-actions" class="row" hidden>
    <button class="sec" id="delete">Delete source video and transcript</button>
    <span class="mut" id="saved"></span>
  </section>
  <section id="list" aria-live="polite"></section>
</main>
<script>
let jobId = null;
let pollTimer = null;
const byId = id => document.getElementById(id);
const statusBox = byId('progress');

function showError(message) {
  statusBox.hidden = false;
  byId('message').textContent = message;
  byId('fill').style.width = '0%';
}

byId('go').onclick = () => {
  const file = byId('file').files[0];
  if (!file) return showError('Choose a video first.');
  const form = new FormData();
  form.append('file', file);
  form.append('language', byId('lang').value);
  byId('go').disabled = true;
  byId('list').replaceChildren();
  byId('result-actions').hidden = true;
  statusBox.hidden = false;
  byId('message').textContent = `Uploading ${file.name} (${(file.size / 1024 ** 3).toFixed(2)} GB)…`;
  byId('fill').style.width = '0%';
  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/upload');
  xhr.upload.onprogress = event => {
    if (event.lengthComputable) {
      const percent = Math.floor(100 * event.loaded / event.total);
      byId('fill').style.width = percent + '%';
      byId('message').textContent = `Uploading… ${percent}%`;
    }
  };
  xhr.onerror = () => { byId('go').disabled = false; showError('Upload failed. Check your connection to the Codespace and try again.'); };
  xhr.onload = () => {
    let data;
    try { data = JSON.parse(xhr.responseText); } catch { data = {}; }
    if (xhr.status < 200 || xhr.status >= 300) {
      byId('go').disabled = false;
      return showError(data.error || `Upload failed (HTTP ${xhr.status}).`);
    }
    jobId = data.id;
    poll();
  };
  xhr.send(form);
};

async function poll() {
  clearTimeout(pollTimer);
  try {
    const response = await fetch('/api/job/' + encodeURIComponent(jobId));
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || 'This job expired or the Codespace restarted.');
    if (job.status === 'error') {
      byId('message').textContent = 'Analysis failed: ' + (job.error || 'Unknown error');
      byId('result-actions').hidden = false;
      byId('go').disabled = false;
      return;
    }
    if (job.status !== 'ready') {
      byId('message').textContent = job.status === 'transcribing'
        ? `Transcribing the full video… ${job.progress}% (first run may download the model)`
        : 'Starting transcription… The first run may take a few minutes to load the model.';
      byId('fill').style.width = job.progress + '%';
      pollTimer = setTimeout(poll, 1800);
      return;
    }
    byId('progress').hidden = true;
    byId('result-actions').hidden = false;
    byId('go').disabled = false;
    render(job.candidates);
    if (!job.candidates.length) showError('The transcript was created, but no 20–60 second candidate windows were found. Try another language or a longer recording with speech.');
  } catch (error) {
    showError(error.message || 'Could not check analysis status.');
    byId('go').disabled = false;
    if (jobId) byId('result-actions').hidden = false;
  }
}

const formatTime = seconds => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;

function render(candidates) {
  const list = byId('list');
  list.replaceChildren();
  if (!candidates.length) return;
  const heading = document.createElement('h2');
  heading.textContent = 'Suggested moments';
  list.append(heading);
  const note = document.createElement('p');
  note.className = 'mut';
  note.textContent = 'Ranked using transcript wording, sentence endings, and clip length. These are suggestions, not an AI judgment of the video’s meaning or visuals. Preview each moment before exporting.';
  list.append(note);
  candidates.forEach(candidate => {
    const card = document.createElement('article');
    card.className = 'card';
    const video = document.createElement('video');
    video.controls = true;
    video.preload = 'metadata';
    const start = document.createElement('input'); start.type = 'number'; start.step = '0.1'; start.className = 'start'; start.value = candidate.start;
    const end = document.createElement('input'); end.type = 'number'; end.step = '0.1'; end.className = 'end'; end.value = candidate.end;
    const setPreview = () => { video.src = `/api/video/${encodeURIComponent(jobId)}#t=${start.value},${end.value}`; };
    setPreview();
    const meta = document.createElement('p'); meta.className = 'mut';
    meta.textContent = `Suggestion ${candidate.id + 1} · score ${candidate.score} · ${formatTime(candidate.start)}–${formatTime(candidate.end)}`;
    const transcript = document.createElement('p'); transcript.className = 'candidate-text';
    transcript.textContent = candidate.text;
    const controls = document.createElement('div'); controls.className = 'row';
    const startLabel = document.createElement('label'); startLabel.textContent = 'Start '; startLabel.append(start);
    const endLabel = document.createElement('label'); endLabel.textContent = 'End '; endLabel.append(end);
    const captionLabel = document.createElement('label');
    const captions = document.createElement('input'); captions.type = 'checkbox'; captions.checked = true;
    captionLabel.append(captions, document.createTextNode(' Captions'));
    const cropLabel = document.createElement('label'); cropLabel.textContent = 'Crop position ';
    const crop = document.createElement('input'); crop.type = 'range'; crop.min = '0'; crop.max = '1'; crop.step = '0.05'; crop.value = '0.5'; cropLabel.append(crop);
    start.onchange = end.onchange = setPreview;
    controls.append(startLabel, endLabel, captionLabel, cropLabel);
    const actions = document.createElement('div'); actions.className = 'row';
    const exportButton = document.createElement('button'); exportButton.textContent = 'Export clip';
    const output = document.createElement('span'); output.className = 'mut';
    exportButton.onclick = async () => {
      exportButton.disabled = true; output.textContent = 'Exporting…';
      try {
        const response = await fetch('/api/export/' + encodeURIComponent(jobId), {
          method: 'POST', headers: {'Content-Type':'application/json'},
          body: JSON.stringify({start:start.value, end:end.value, captions:captions.checked, crop:crop.value})
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || `Export failed (HTTP ${response.status}).`);
        output.replaceChildren();
        const download = document.createElement('a');
        download.href = '/api/clips/' + encodeURIComponent(data.file);
        download.textContent = `Download ${data.file}`;
        const remove = document.createElement('button');
        remove.type = 'button'; remove.className = 'sec'; remove.textContent = 'Delete clip';
        remove.onclick = async () => {
          remove.disabled = true;
          try {
            const deleted = await fetch('/api/clips/' + encodeURIComponent(data.file), {method:'DELETE'});
            if (!deleted.ok) throw new Error('Could not delete the exported clip.');
            output.textContent = 'Exported clip deleted from the Codespace.';
          } catch (error) { output.textContent = error.message; remove.disabled = false; }
        };
        output.append(download, document.createTextNode(' · '), remove);
        byId('saved').textContent = `Exports: ${data.folder}`;
      } catch (error) { output.textContent = 'Failed: ' + error.message; }
      finally { exportButton.disabled = false; }
    };
    actions.append(exportButton, output);
    card.append(video, meta, transcript, controls, actions);
    list.append(card);
  });
}

byId('delete').onclick = async () => {
  if (!jobId || !confirm('Delete the uploaded source and transcript from the Codespace? Exported clips will stay in its clips folder.')) return;
  try {
    const response = await fetch('/api/delete/' + encodeURIComponent(jobId), {method:'POST'});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Could not delete this source.');
    clearTimeout(pollTimer);
    jobId = null;
    byId('list').replaceChildren();
    byId('result-actions').hidden = true;
    byId('progress').hidden = true;
    byId('saved').textContent = '';
  } catch (error) { showError(error.message); }
};
</script>
</body>
</html>
'''


@app.get("/")
def index():
    return app.response_class(PAGE, mimetype="text/html")


@app.get("/api/health")
def health():
    usage = shutil.disk_usage(WORK)
    return jsonify(ok=True, free_bytes=usage.free, model=MODEL_SIZE)


@app.post("/api/upload")
def upload():
    cleanup_expired()
    uploaded = request.files.get("file")
    if not uploaded or not uploaded.filename:
        return jsonify(error="Choose a video file first."), 400
    original = secure_filename(uploaded.filename) or "video.mp4"
    suffix = Path(original).suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}:
        return jsonify(error="Use an MP4, MOV, MKV, WebM, M4V, or AVI video."), 400
    content_length = request.content_length or 0
    if content_length > MAX_UPLOAD_BYTES:
        return jsonify(error=f"This upload exceeds the configured {MAX_UPLOAD_BYTES // 1024**3} GB limit."), 413
    free = shutil.disk_usage(WORK).free
    if content_length and content_length > free - 512 * 1024**2:
        return jsonify(error="There is not enough free Codespace storage for this upload. Delete old clips or use a smaller video."), 507
    job_id = uuid.uuid4().hex[:12]
    directory = WORK / job_id
    directory.mkdir()
    source = directory / ("source" + suffix)
    try:
        uploaded.save(source)
        if source.stat().st_size > free - 512 * 1024**2:
            shutil.rmtree(directory, ignore_errors=True)
            return jsonify(error="There is not enough free Codespace storage for this upload."), 507
    except OSError as exc:
        shutil.rmtree(directory, ignore_errors=True)
        return jsonify(error=f"Could not save the upload: {exc}"), 507
    job = {"id": job_id, "dir": directory, "src": source,
           "name": Path(original).stem[:100] or "video", "status": "queued",
           "progress": 0, "candidates": [], "segments": [], "n": 0,
           "updated": time.time(), "duration": 0}
    jobs[job_id] = job
    language = request.form.get("language") or None
    threading.Thread(target=transcribe, args=(job_id, language), daemon=True).start()
    return jsonify(id=job_id)


@app.get("/api/job/<job_id>")
def job_status(job_id):
    cleanup_expired()
    job = jobs.get(job_id) or abort(404)
    job["updated"] = time.time()
    return jsonify(status=job["status"], progress=job["progress"], error=job.get("error"),
                   candidates=job["candidates"], name=job["name"],
                   duration=job.get("duration", 0))


@app.get("/api/video/<job_id>")
def video(job_id):
    job = jobs.get(job_id) or abort(404)
    if not job["src"].exists():
        abort(410)
    return send_file(job["src"], conditional=True)


@app.post("/api/export/<job_id>")
def export(job_id):
    job = jobs.get(job_id) or abort(404)
    if job["status"] != "ready" or not job["src"].exists():
        return jsonify(error="This video is not ready to export."), 409
    data = request.get_json(silent=True) or {}
    try:
        start, end = float(data["start"]), float(data["end"])
        crop = min(1.0, max(0.0, float(data.get("crop", 0.5))))
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Start and end must be valid numbers."), 400
    duration = job["duration"]
    if not all(math.isfinite(value) for value in (start, end, crop)):
        return jsonify(error="Start and end must be finite numbers."), 400
    if start < 0 or end <= start or end > duration + 0.1:
        return jsonify(error=f"Choose a range within the {duration:.1f}-second source video."), 400
    if end - start > MAX_CLIP_SECONDS:
        return jsonify(error=f"Clips can be at most {MAX_CLIP_SECONDS} seconds."), 400
    job["n"] += 1
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", job["name"]).strip("_") or "clip"
    name = f"{safe_stem}_clip{job['n']}.mp4"
    output = OUT / name
    ass_file = job["dir"] / "clip.ass"
    filters = (f"scale=1080:1920:force_original_aspect_ratio=increase:force_divisible_by=2,"
               f"crop=1080:1920:x=(in_w-1080)*{crop}:y=(in_h-1920)/2,setsar=1")
    if data.get("captions", True):
        make_ass(job["segments"], start, end, ass_file)
        # Absolute escaped path is required by FFmpeg's subtitles filter.
        escaped = str(ass_file).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        filters += f",subtitles='{escaped}'"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start),
               "-i", str(job["src"]), "-t", str(end - start), "-vf", filters,
               "-c:v", "libx264", "-preset", "fast", "-crf", "21", "-pix_fmt", "yuv420p",
               "-c:a", "aac",
               "-movflags", "+faststart", str(output)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        output.unlink(missing_ok=True)
        return jsonify(error="Export took too long and was stopped. Try a shorter clip."), 504
    if result.returncode != 0:
        output.unlink(missing_ok=True)
        return jsonify(error=(result.stderr or "FFmpeg could not export this video.")[-1200:]), 500
    job["updated"] = time.time()
    return jsonify(file=name, folder=str(OUT), expires_in_hours=JOB_TTL_SECONDS // 3600)


@app.post("/api/delete/<job_id>")
def delete(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify(ok=True)
    if job["status"] in ("queued", "transcribing"):
        return jsonify(error="Wait for analysis to finish before deleting the source."), 409
    shutil.rmtree(job["dir"], ignore_errors=True)
    jobs.pop(job_id, None)
    return jsonify(ok=True)


@app.get("/api/clips/<path:name>")
def download_clip(name):
    if Path(name).name != name:
        abort(404)
    target = OUT / name
    if not target.is_file():
        abort(404)
    return send_file(target, as_attachment=True, download_name=name)


@app.delete("/api/clips/<path:name>")
def delete_clip(name):
    if Path(name).name != name:
        abort(404)
    target = OUT / name
    if not target.is_file():
        abort(404)
    target.unlink()
    return jsonify(ok=True)


@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify(error=f"Upload is larger than the configured {MAX_UPLOAD_BYTES // 1024**3} GB limit."), 413


if __name__ == "__main__":
    print(f"Clipper ready at http://127.0.0.1:5050 (model: {MODEL_SIZE})")
    app.run(host="0.0.0.0", port=5050, threaded=True)
