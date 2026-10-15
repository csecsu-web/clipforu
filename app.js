"""Personal clipper: long video -> candidate 9:16 shorts with captions.
Runs locally. Needs ffmpeg + faster-whisper. Source video is deleted when you confirm."""
import os
import re
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory, abort

BASE = Path(__file__).parent
WORK = BASE / "work"
OUT = Path(os.environ.get("CLIPS_DIR", Path.home() / "Movies" / "Clips"))  # finished clips land here
WORK.mkdir(exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")  # "base" = faster, "medium" = more accurate

app = Flask(__name__)
jobs = {}
_model = None

HOOKS = [
    "why", "how", "never", "truth", "secret", "actually", "mistake", "biggest", "worst", "best",
    "nobody", "everyone", "problem", "but", "wrong", "real", "villain", "hero",
    "miért", "hogyan", "soha", "igazság", "titok", "valójában", "hiba", "legnagyobb",
    "senki", "mindenki", "probléma", "de", "rossz", "valódi",
]


def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def score(text, dur):
    t = text.lower()
    s = sum(1 for h in HOOKS if re.search(r"\b" + h + r"\b", t))
    s += 2 * t.count("?")
    s += len(re.findall(r"\d+", t))
    s -= abs(dur - 40) / 20  # prefer ~40s clips
    first = t[:80]
    if "?" in first or any(h in first for h in HOOKS[:12]):
        s += 2  # strong opening
    return round(s, 2)


def find_candidates(segs, lo=25, hi=60, top=12):
    out, i = [], 0
    while i < len(segs):
        start = segs[i]["start"]
        j = i
        while j < len(segs):
            dur = segs[j]["end"] - start
            if dur >= lo and segs[j]["text"].rstrip().endswith((".", "?", "!")):
                break
            if dur >= hi:
                break
            j += 1
        j = min(j, len(segs) - 1)
        end = segs[j]["end"]
        dur = end - start
        text = " ".join(s["text"] for s in segs[i:j + 1])
        if dur >= 10:
            out.append({"start": round(start, 2), "end": round(end, 2),
                        "text": text, "score": score(text, dur)})
        i = j + 1
    out.sort(key=lambda c: -c["score"])
    for n, c in enumerate(out[:top]):
        c["id"] = n
    return out[:top]


def transcribe(job_id, language):
    job = jobs[job_id]
    try:
        job["status"] = "transcribing"
        segs, info = get_model().transcribe(
            str(job["src"]), language=language or None, word_timestamps=True, vad_filter=True)
        segments = []
        for s in segs:
            words = [{"w": w.word.strip(), "s": w.start, "e": w.end} for w in (s.words or [])]
            segments.append({"text": s.text.strip(), "start": s.start, "end": s.end, "words": words})
            job["progress"] = min(99, int(100 * s.end / max(info.duration, 1)))
        job["segments"] = segments
        job["candidates"] = find_candidates(segments)
        job["status"] = "ready"
    except Exception as e:  # noqa: BLE001
        job["status"] = "error"
        job["error"] = str(e)


def ts(t):
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def make_ass(segments, start, end, path):
    words = [w for s in segments for w in s["words"]
             if w["s"] >= start - 0.05 and w["e"] <= end + 0.05 and w["w"]]
    dur = end - start
    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, "
        "Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,84,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,"
        "1,6,2,2,60,60,520,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    chunks = [words[i:i + 3] for i in range(0, len(words), 3)]
    lines = []
    for k, ch in enumerate(chunks):
        a = max(ch[0]["s"] - start, 0)
        b = min(ch[-1]["e"] - start, dur)
        if k + 1 < len(chunks):
            b = max(b, min(chunks[k + 1][0]["s"] - start, dur))  # no flicker between chunks
        text = " ".join(w["w"] for w in ch).replace("{", "").replace("}", "").upper()
        lines.append(f"Dialogue: 0,{ts(a)},{ts(b)},Default,,0,0,0,,{text}")
    Path(path).write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


@app.get("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if not f:
        abort(400)
    job_id = uuid.uuid4().hex[:8]
    d = WORK / job_id
    d.mkdir()
    ext = Path(f.filename).suffix or ".mp4"
    src = d / f"source{ext}"
    f.save(src)
    jobs[job_id] = {"id": job_id, "dir": d, "src": src, "name": Path(f.filename).stem,
                    "status": "queued", "progress": 0, "candidates": [], "segments": [], "n": 0}
    threading.Thread(target=transcribe, args=(job_id, request.form.get("language") or None),
                     daemon=True).start()
    return jsonify(id=job_id)


@app.get("/api/job/<job_id>")
def job_status(job_id):
    j = jobs.get(job_id) or abort(404)
    return jsonify(status=j["status"], progress=j["progress"], error=j.get("error"),
                   candidates=j["candidates"], name=j["name"])


@app.get("/api/video/<job_id>")
def video(job_id):
    j = jobs.get(job_id) or abort(404)
    return send_file(j["src"], conditional=True)


@app.post("/api/export/<job_id>")
def export(job_id):
    j = jobs.get(job_id) or abort(404)
    data = request.get_json()
    start, end = float(data["start"]), float(data["end"])
    if end <= start:
        return jsonify(error="End must be after start"), 400
    j["n"] += 1
    name = f"{j['name']}_clip{j['n']}.mp4"
    vf = (f"crop=w=ih*9/16:h=ih:x=(iw-ih*9/16)*{float(data.get('crop', 0.5))}:y=0,"
          "scale=1080:1920")
    if data.get("captions", True):
        make_ass(j["segments"], start, end, j["dir"] / "clip.ass")
        vf += ",subtitles=clip.ass"
    cmd = ["ffmpeg", "-y", "-ss", str(start), "-i", str(j["src"].name), "-t", str(end - start),
           "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-c:a", "aac",
           "-movflags", "+faststart", str(OUT / name)]
    r = subprocess.run(cmd, cwd=j["dir"], capture_output=True, text=True)
    if r.returncode != 0:
        return jsonify(error=r.stderr[-400:]), 500
    return jsonify(file=name, folder=str(OUT))


@app.post("/api/delete/<job_id>")
def delete(job_id):
    j = jobs.pop(job_id, None)
    if j:
        shutil.rmtree(j["dir"], ignore_errors=True)
    return jsonify(ok=True)


if __name__ == "__main__":
    print(f"Open http://localhost:5050  |  clips saved to {OUT}")
    app.run(port=5050, threaded=True)
