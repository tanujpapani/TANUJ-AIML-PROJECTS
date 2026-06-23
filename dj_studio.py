"""
AI DJ Studio — single-file edition.

Everything (analysis, mixing, effects, server, and the web page) lives in this
one file so there's nothing to misplace.

Run it:
    python dj_studio.py

It installs nothing, picks a free port automatically, and opens your browser.
You need FFmpeg installed (winget install Gyan.FFmpeg) and the libraries from
requirements.txt (fastapi uvicorn python-multipart numpy scipy librosa
soundfile pydub).
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import tempfile
import threading
import uuid
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt
import librosa
from pydub import AudioSegment
import pydub.utils as pydub_utils
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

SR = 44100
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ===================================================================== AUDIO IO
def configure_ffmpeg() -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        winget = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
        if winget.exists():
            m = list(winget.rglob("ffmpeg.exe"))
            p = list(winget.rglob("ffprobe.exe"))
            if m:
                ffmpeg = str(m[0]); os.environ["PATH"] = str(m[0].parent) + os.pathsep + os.environ.get("PATH", "")
            if p:
                ffprobe = str(p[0]); os.environ["PATH"] = str(p[0].parent) + os.pathsep + os.environ.get("PATH", "")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("FFmpeg/FFprobe not found. Install: winget install Gyan.FFmpeg")
    AudioSegment.converter = ffmpeg
    AudioSegment.ffmpeg = ffmpeg
    AudioSegment.ffprobe = ffprobe
    pydub_utils.get_encoder_name = lambda: ffmpeg
    pydub_utils.get_prober_name = lambda: ffprobe


def load_stereo(path: str) -> np.ndarray:
    seg = AudioSegment.from_file(path).set_frame_rate(SR).set_channels(2)
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32).reshape(-1, 2) / 32768.0
    return samples


def save_mp3(stereo: np.ndarray, path: str, bitrate: str = "320k") -> None:
    data = (np.clip(stereo, -1.0, 1.0) * 32767.0).astype(np.int16)
    seg = AudioSegment(data.tobytes(), frame_rate=SR, sample_width=2, channels=2)
    seg.export(path, format="mp3", bitrate=bitrate)


# ===================================================================== ANALYSIS
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_CAMELOT_MAJOR = {"C": "8B", "G": "9B", "D": "10B", "A": "11B", "E": "12B", "B": "1B",
                  "F#": "2B", "C#": "3B", "G#": "4B", "D#": "5B", "A#": "6B", "F": "7B"}
_CAMELOT_MINOR = {"A": "8A", "E": "9A", "B": "10A", "F#": "11A", "C#": "12A", "G#": "1A",
                  "D#": "2A", "A#": "3A", "F": "4A", "C": "5A", "G": "6A", "D": "7A"}


@dataclass
class Track:
    name: str
    audio: np.ndarray
    bpm: float
    beats: np.ndarray
    key: str
    camelot: str
    energy: float
    extra: dict = field(default_factory=dict)


def _normalise_bpm(bpm: float) -> float:
    if bpm <= 0 or np.isnan(bpm):
        return 120.0
    while bpm < 85:
        bpm *= 2
    while bpm > 175:
        bpm /= 2
    return round(float(bpm), 2)


def _detect_key(mono: np.ndarray):
    chroma = librosa.feature.chroma_cqt(y=mono, sr=SR).mean(axis=1)
    if chroma.sum() <= 0:
        return "C major", "8B"
    best, choice = -1e9, ("C", "major")
    for i in range(12):
        rot = np.roll(chroma, -i)
        maj = float(np.corrcoef(rot, _MAJOR)[0, 1])
        mino = float(np.corrcoef(rot, _MINOR)[0, 1])
        if maj > best:
            best, choice = maj, (_NOTES[i], "major")
        if mino > best:
            best, choice = mino, (_NOTES[i], "minor")
    note, mode = choice
    cam = _CAMELOT_MAJOR[note] if mode == "major" else _CAMELOT_MINOR[note]
    return f"{note} {mode}", cam


def analyze(name: str, audio: np.ndarray) -> Track:
    mono = audio.mean(axis=1).astype(np.float32)
    if mono.size == 0:
        return Track(name, audio, 120.0, np.array([0]), "C major", "8B", 0.0)
    tempo, frames = librosa.beat.beat_track(y=mono, sr=SR, trim=False)
    bpm = _normalise_bpm(float(np.asarray(tempo).flatten()[0]))
    beats = librosa.frames_to_samples(frames)
    beats = beats[beats < len(mono)]
    if beats.size < 8:
        beats = np.arange(0, len(mono), int(SR * 60.0 / bpm))
    key, cam = _detect_key(mono)
    energy = float(np.sqrt(np.mean(mono ** 2)))
    return Track(name, audio, bpm, beats.astype(np.int64), key, cam, energy)


def _camelot_parts(code: str):
    return int(code[:-1]), code[-1]


def harmonic_distance(a: str, b: str) -> int:
    an, al = _camelot_parts(a)
    bn, bl = _camelot_parts(b)
    if a == b:
        return 0
    if al == bl and min((an - bn) % 12, (bn - an) % 12) == 1:
        return 1
    if an == bn and al != bl:
        return 1
    return 2 + min((an - bn) % 12, (bn - an) % 12)


def auto_order(tracks: List[Track]) -> List[int]:
    if len(tracks) <= 2:
        return list(range(len(tracks)))
    remaining = list(range(len(tracks)))
    start = min(remaining, key=lambda i: tracks[i].energy)
    order = [start]; remaining.remove(start)
    while remaining:
        cur = tracks[order[-1]]
        nxt = min(remaining, key=lambda i: (harmonic_distance(cur.camelot, tracks[i].camelot),
                                            abs(tracks[i].energy - cur.energy)))
        order.append(nxt); remaining.remove(nxt)
    return order


# ====================================================================== EFFECTS
def _resample_by_curve(seg: np.ndarray, read_pos: np.ndarray) -> np.ndarray:
    idx = np.arange(seg.shape[0], dtype=np.float64)
    out = np.empty((read_pos.shape[0], seg.shape[1]), dtype=np.float32)
    for c in range(seg.shape[1]):
        out[:, c] = np.interp(read_pos, idx, seg[:, c]).astype(np.float32)
    return out


def tape_stop(seg: np.ndarray) -> np.ndarray:
    n = seg.shape[0]
    if n < 8:
        return seg
    rate = 1.0 - np.arange(n) / n
    pos = np.cumsum(rate); pos *= (n - 1) / max(pos[-1], 1e-9)
    return _resample_by_curve(seg, pos)


def backspin(seg: np.ndarray) -> np.ndarray:
    n = seg.shape[0]
    if n < 8:
        return seg[::-1]
    rev = seg[::-1]
    out_n = max(8, int(n * 0.6))
    rate = np.linspace(0.4, 3.0, out_n)
    pos = np.cumsum(rate); pos *= (n - 1) / max(pos[-1], 1e-9)
    return _resample_by_curve(rev, pos)


def echo_out(seg: np.ndarray, delay_samples: int, feedback: float = 0.55, repeats: int = 4) -> np.ndarray:
    delay_samples = max(1, int(delay_samples))
    extra = delay_samples * repeats
    out = np.zeros((seg.shape[0] + extra, 2), dtype=np.float32)
    out[:seg.shape[0]] += seg
    for r in range(1, repeats + 1):
        start = r * delay_samples
        out[start:start + seg.shape[0]] += (seg * (feedback ** r)).astype(np.float32)
    out[seg.shape[0]:] *= np.linspace(1, 0, extra, dtype=np.float32).reshape(-1, 1)
    return out


def beat_loop_roll(audio: np.ndarray, start: int, loop_samples: int, repeats: int, rolling: bool = False) -> np.ndarray:
    start = max(0, start)
    pieces = []; cur = max(1, int(loop_samples))
    for _ in range(max(1, repeats)):
        seg = audio[start:start + cur]
        if seg.shape[0] < cur:
            seg = np.concatenate([seg, np.zeros((cur - seg.shape[0], 2), dtype=np.float32)], axis=0)
        pieces.append(seg)
        if rolling and cur > SR // 16:
            cur //= 2
    return np.concatenate(pieces, axis=0) if pieces else audio[:0]


# ======================================================================= MIXING
def bars_to_samples(bars: int, bpm: float) -> int:
    return max(1, int((bars * 4) * 60.0 / max(bpm, 1.0) * SR))


def _time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    if abs(rate - 1.0) < 1e-3:
        return audio
    chans = [librosa.effects.time_stretch(audio[:, c].astype(np.float32), rate=rate)
             for c in range(audio.shape[1])]
    n = min(len(c) for c in chans)
    return np.stack([c[:n] for c in chans], axis=1)


def _split_low_high(x: np.ndarray, cutoff: float = 200.0):
    sos = butter(4, cutoff / (SR / 2.0), btype="low", output="sos")
    low = sosfiltfilt(sos, x, axis=0)
    return low, x - low


def _slice(audio: np.ndarray, start: int, length: int) -> np.ndarray:
    start = max(0, start)
    piece = audio[start:start + length]
    if piece.shape[0] < length:
        piece = np.concatenate([piece, np.zeros((length - piece.shape[0], 2), dtype=np.float32)], axis=0)
    return piece


def crossfade(out_tail: np.ndarray, in_head: np.ndarray, bass_swap: bool) -> np.ndarray:
    L = min(out_tail.shape[0], in_head.shape[0])
    out_tail, in_head = out_tail[:L], in_head[:L]
    t = np.linspace(0.0, 1.0, L, dtype=np.float32).reshape(-1, 1)
    out_g, in_g = np.cos(t * np.pi / 2), np.sin(t * np.pi / 2)
    if not bass_swap:
        return out_tail * out_g + in_head * in_g
    lo, hi = _split_low_high(out_tail)
    lo2, hi2 = _split_low_high(in_head)
    bass_in = t * t * (3 - 2 * t)
    return (hi * out_g + hi2 * in_g + lo * (1 - bass_in) + lo2 * bass_in).astype(np.float32)


def _nearest_beat(beats: np.ndarray, pos: int) -> int:
    if beats.size == 0:
        return pos
    return int(beats[int(np.argmin(np.abs(beats - pos)))])


def _first_downbeat(beats: np.ndarray, min_sample: int) -> int:
    down = beats[::4]
    after = down[down >= min_sample]
    return int(after[0]) if after.size else _nearest_beat(beats, min_sample)


def _prepare(track: Track, target_bpm):
    if target_bpm and track.bpm:
        rate = target_bpm / track.bpm
        if 0.5 <= rate <= 2.0 and abs(rate - 1.0) > 1e-3:
            return _time_stretch(track.audio, rate), (track.beats / rate).astype(np.int64), target_bpm
    return track.audio, track.beats, track.bpm


def build_mix(tracks, order, *, crossfade_bars=8, tempo_match=True, bass_swap=True,
              segment_seconds=None, intro_skip_seconds=0.5, master_bpm=None,
              transition_fx="crossfade"):
    ordered = [tracks[i] for i in order]
    segment_seconds = segment_seconds or [None] * (len(ordered) - 1)

    active, active_beats, active_bpm = _prepare(ordered[0], master_bpm)
    active_pos = _first_downbeat(active_beats, int(intro_skip_seconds * SR))
    parts: List[np.ndarray] = []
    recipe = {"order": [t.name for t in ordered], "crossfade_bars": crossfade_bars,
              "tempo_match": tempo_match, "bass_swap": bass_swap, "master_bpm": master_bpm,
              "transition_fx": transition_fx, "transitions": []}

    for i in range(1, len(ordered)):
        incoming = ordered[i]
        xfade = bars_to_samples(crossfade_bars, active_bpm)
        bar = bars_to_samples(1, active_bpm)
        secs = segment_seconds[i - 1]
        play = int(secs * SR) if secs else int(45 * SR)

        if master_bpm:
            in_audio, in_beats, in_bpm = _prepare(incoming, master_bpm); matched = True
        else:
            rate = active_bpm / incoming.bpm if incoming.bpm else 1.0
            if tempo_match and 0.90 <= rate <= 1.11:
                in_audio = _time_stretch(incoming.audio, rate)
                in_beats = (incoming.beats / rate).astype(np.int64)
                in_bpm, matched = active_bpm, True
            else:
                in_audio, in_beats, in_bpm, matched = incoming.audio, incoming.beats, incoming.bpm, False
        cue = _first_downbeat(in_beats, int(intro_skip_seconds * SR))

        if transition_fx == "crossfade":
            solo_end = min(_nearest_beat(active_beats, active_pos + max(play - xfade, 0)), active.shape[0])
            parts.append(active[active_pos:solo_end])
            parts.append(crossfade(_slice(active, solo_end, xfade), _slice(in_audio, cue, xfade), bass_swap))
            next_pos = cue + xfade
        else:
            solo_end = min(_nearest_beat(active_beats, active_pos + play), active.shape[0])
            parts.append(active[active_pos:solo_end])
            if transition_fx == "spinback":
                parts.append(backspin(_slice(active, solo_end, int(0.6 * SR))))
            elif transition_fx == "tapestop":
                parts.append(tape_stop(_slice(active, solo_end, int(0.9 * SR))))
            elif transition_fx == "echo":
                parts.append(echo_out(_slice(active, solo_end, bar), bar // 4))
            elif transition_fx == "loop_roll":
                parts.append(beat_loop_roll(active, max(0, solo_end - bar), bar, 4, rolling=True))
            next_pos = cue

        recipe["transitions"].append({"from": ordered[i - 1].name, "to": incoming.name,
                                      "from_key": ordered[i - 1].camelot, "to_key": incoming.camelot,
                                      "tempo_matched": matched, "effect": transition_fx})
        active, active_beats, active_bpm, active_pos = in_audio, in_beats, in_bpm, next_pos

    parts.append(active[active_pos:])
    final = np.concatenate(parts, axis=0) if parts else np.zeros((SR, 2), np.float32)
    fade = min(int(2.5 * SR), final.shape[0] // 2)
    if fade > 0:
        final[:fade] *= np.linspace(0, 1, fade, dtype=np.float32).reshape(-1, 1)
        final[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32).reshape(-1, 1)
    recipe["duration_seconds"] = round(final.shape[0] / SR, 2)
    return final.astype(np.float32), recipe


# ========================================================================== APP
configure_ffmpeg()
app = FastAPI(title="AI DJ Studio")


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.post("/api/analyze")
async def analyze_endpoint(files: list[UploadFile] = File(...)):
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tracks = []
            for f in files:
                safe = "".join(c for c in (f.filename or "song") if c.isalnum() or c in " ._-").strip() or "song"
                p = Path(tmp) / f"{uuid.uuid4().hex[:8]}_{safe}"
                p.write_bytes(f.file.read())
                tracks.append(analyze(safe, load_stereo(str(p))))
        max_e = max((t.energy for t in tracks), default=1.0) or 1.0
        out = [{"name": t.name, "bpm": t.bpm, "key": t.key, "camelot": t.camelot,
                "energy": round(100 * t.energy / max_e)} for t in tracks]
        return {"analysis": out, "suggested_order": auto_order(tracks)}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/api/mix")
async def mix(files: list[UploadFile] = File(...), segment_seconds: str = Form("[]"),
              crossfade_bars: int = Form(8), tempo_match: bool = Form(True),
              bass_swap: bool = Form(True), auto_order_enabled: bool = Form(False),
              master_bpm: str = Form(""), transition_fx: str = Form("crossfade")):
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tracks = []
            for f in files:
                safe = "".join(c for c in (f.filename or "song") if c.isalnum() or c in " ._-").strip() or "song"
                p = Path(tmp) / f"{uuid.uuid4().hex[:8]}_{safe}"
                p.write_bytes(f.file.read())
                tracks.append(analyze(safe, load_stereo(str(p))))
            order = auto_order(tracks) if auto_order_enabled else list(range(len(tracks)))
            try:
                segs = json.loads(segment_seconds)
            except Exception:
                segs = []
            segs = [(float(s) if s not in (None, "", 0) else None) for s in segs]
            while len(segs) < max(0, len(tracks) - 1):
                segs.append(None)
            try:
                mbpm = float(master_bpm) if str(master_bpm).strip() else None
            except ValueError:
                mbpm = None
            final, recipe = build_mix(tracks, order, crossfade_bars=int(crossfade_bars),
                                      tempo_match=bool(tempo_match), bass_swap=bool(bass_swap),
                                      segment_seconds=segs, master_bpm=mbpm, transition_fx=transition_fx)

        version = uuid.uuid4().hex[:12]
        out_audio = OUTPUT_DIR / f"mix_{version}.mp3"
        out_recipe = OUTPUT_DIR / f"recipe_{version}.json"
        save_mp3(final, str(out_audio))
        out_recipe.write_text(json.dumps(recipe, indent=2), encoding="utf-8")
        try:
            mixes = sorted(OUTPUT_DIR.glob("mix_*.mp3"), key=lambda p: p.stat().st_mtime, reverse=True)
            for old in mixes[6:]:
                old.unlink(missing_ok=True)
                (OUTPUT_DIR / old.name.replace("mix_", "recipe_").replace(".mp3", ".json")).unlink(missing_ok=True)
        except OSError:
            pass

        max_e = max((t.energy for t in tracks), default=1.0) or 1.0
        analysis = [{"name": t.name, "bpm": t.bpm, "key": t.key, "camelot": t.camelot,
                     "energy": round(100 * t.energy / max_e)} for t in tracks]
        return {"audio_url": f"/api/download/{out_audio.name}",
                "recipe_url": f"/api/download/{out_recipe.name}",
                "recipe": recipe, "analysis": analysis, "order": order}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/api/download/{filename}")
def download(filename: str):
    path = OUTPUT_DIR / Path(filename).name
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "not found"})
    headers = {"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
    media = "audio/mpeg" if path.suffix == ".mp3" else "application/json"
    return FileResponse(path, media_type=media, filename=path.name, headers=headers)


# ===================================================================== WEB PAGE
INDEX_HTML = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/><title>AI DJ Studio</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--ink:#0E1116;--panel:#161B22;--panel2:#1B222B;--line:#262E38;--text:#E6EDF3;--muted:#8B97A5;--green:#3FD27E;--cyan:#4FD1E0}
*{box-sizing:border-box}body{margin:0;min-height:100vh;background:var(--ink);color:var(--text);font-family:'Space Grotesk',system-ui,sans-serif}
.shell{width:min(900px,calc(100% - 32px));margin:0 auto;padding:32px 0 64px}
.top{display:flex;align-items:baseline;gap:14px;border-bottom:1px solid var(--line);padding-bottom:18px}
.top h1{margin:0;font-size:24px;font-weight:700;letter-spacing:-.02em}
.top .tag{font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--muted);letter-spacing:.12em;text-transform:uppercase}
.card{margin-top:18px;padding:22px;border:1px solid var(--line);border-radius:16px;background:var(--panel)}
.head{display:flex;gap:14px;align-items:flex-start;margin-bottom:18px}
.head .num{display:grid;place-items:center;min-width:34px;height:34px;border-radius:10px;background:var(--panel2);font-family:'IBM Plex Mono',monospace;color:var(--cyan);font-weight:600}
.head h2{margin:0;font-size:18px}.head p{margin:3px 0 0;color:var(--muted);font-size:14px}
.row{display:flex;flex-wrap:wrap;gap:12px;align-items:center}
.btn{border:none;border-radius:12px;padding:13px 18px;font-weight:600;font-size:14px;cursor:pointer;text-decoration:none;color:#08101f;background:linear-gradient(135deg,var(--cyan),var(--green));transition:.15s}
.btn.ghost{color:var(--text);background:var(--panel2);border:1px solid var(--line)}.btn.big{font-size:16px;padding:15px 26px}
.btn:hover{filter:brightness(1.07)}.btn:disabled{opacity:.55;cursor:not-allowed}
.switch{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:14px;cursor:pointer}
.list{display:grid;gap:10px;margin-top:16px}.empty{color:var(--muted);padding:16px;border-radius:12px;background:var(--panel2)}
.song{display:grid;grid-template-columns:1fr auto;gap:12px;align-items:center;padding:14px;border-radius:12px;background:var(--panel2);border:1px solid var(--line)}
.song strong{display:block}.song .meta{font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--muted);margin-top:4px}
.song .key{color:var(--cyan)}.song .mixin{margin-top:10px;display:flex;align-items:center;gap:8px;color:var(--muted);font-size:13px}
.song .mixin input{width:80px;padding:7px 10px;border-radius:10px;border:1px solid var(--line);background:var(--ink);color:var(--text)}
.acts{display:flex;gap:6px}.icon{width:34px;height:34px;border-radius:9px;border:1px solid var(--line);background:var(--ink);color:var(--text);cursor:pointer}
.settings{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.set label{display:block;color:var(--muted);font-size:13px;margin-bottom:8px}
.set select,.set input{width:100%;padding:12px;border-radius:12px;border:1px solid var(--line);background:var(--ink);color:var(--text)}
.toggles{display:flex;flex-direction:column;gap:12px;justify-content:center}
.actions{display:flex;gap:16px;align-items:center;margin-top:22px}.status{color:var(--muted);font-size:14px}
.result{display:none}audio{width:100%;margin-bottom:14px}
pre{margin-top:16px;max-height:300px;overflow:auto;padding:14px;border-radius:12px;background:var(--ink);color:#cdebf2;font-size:12px;font-family:'IBM Plex Mono',monospace}
@media(max-width:680px){.settings,.song{grid-template-columns:1fr}}
</style></head><body>
<main class="shell">
<header class="top"><h1>AI DJ Studio</h1><span class="tag">harmonic · beat-matched · turntable FX</span></header>
<section class="card"><div class="head"><span class="num">01</span><div><h2>Queue</h2><p>Add songs anytime — nothing resets.</p></div></div>
<div class="row"><label class="btn" for="songFiles">+ Add songs</label>
<button id="analyzeBtn" class="btn ghost" type="button">Analyze</button>
<button id="clearQueue" class="btn ghost" type="button">Clear</button>
<label class="switch"><input type="checkbox" id="autoOrder"/> AI harmonic auto-order</label></div>
<input id="songFiles" type="file" accept=".mp3,.wav,.flac,.m4a,.aac,.ogg" multiple hidden/>
<div id="fileList" class="list"></div></section>
<section class="card"><div class="head"><span class="num">02</span><div><h2>Mix settings</h2><p>Tempo, transitions, and turntable effects.</p></div></div>
<div class="settings">
<div class="set"><label for="crossfadeBars">Crossfade length</label><select id="crossfadeBars">
<option value="4">4 bars (quick)</option><option value="8" selected>8 bars (classic)</option><option value="16">16 bars (long)</option></select></div>
<div class="set"><label for="transitionFx">Transition effect</label><select id="transitionFx">
<option value="crossfade" selected>Crossfade</option><option value="spinback">Spinback</option>
<option value="tapestop">Tape stop / brake</option><option value="echo">Echo out</option><option value="loop_roll">Beat-loop roll</option></select></div>
<div class="set"><label for="masterBpm">Master tempo (BPM)</label><input id="masterBpm" type="number" min="60" max="200" placeholder="auto (per-track)"/></div>
<div class="toggles"><label class="switch"><input type="checkbox" id="tempoMatch" checked/> Tempo-match (pitch-preserving)</label>
<label class="switch"><input type="checkbox" id="bassSwap" checked/> Bass-swap EQ transition</label></div>
</div></section>
<section class="actions"><button id="mixButton" class="btn big">Create Mix</button><div id="status" class="status"></div></section>
<section class="card result" id="resultCard"><div class="head"><span class="num">03</span><div><h2>Result</h2><p>Preview keeps playing across re-mixes.</p></div></div>
<audio id="audioPlayer" controls></audio><div class="row"><a id="downloadMix" class="btn" href="#" download>Download MP3</a>
<a id="downloadRecipe" class="btn ghost" href="#" download>Download recipe</a></div><pre id="recipeBox"></pre></section>
</main>
<script>
const fileInput=document.getElementById("songFiles"),fileList=document.getElementById("fileList"),mixButton=document.getElementById("mixButton"),clearQueue=document.getElementById("clearQueue"),statusBox=document.getElementById("status"),resultCard=document.getElementById("resultCard"),audioPlayer=document.getElementById("audioPlayer"),downloadMix=document.getElementById("downloadMix"),downloadRecipe=document.getElementById("downloadRecipe"),recipeBox=document.getElementById("recipeBox");
let queue=[],mixInSecs=[],analysisByName={};
function preload(u){return new Promise(r=>{const a=new Audio();a.preload="auto";a.src=u;const d=()=>r();a.addEventListener("canplaythrough",d,{once:true});a.addEventListener("error",d,{once:true});setTimeout(d,5000);});}
async function swap(u){const had=!!audioPlayer.src,at=audioPlayer.currentTime||0,pl=had&&!audioPlayer.paused&&!audioPlayer.ended;if(!had){audioPlayer.src=u;return;}await preload(u);await new Promise(r=>{const m=()=>{try{const t=Math.min(at,Math.max(0,(audioPlayer.duration||at)-0.1));if(!Number.isNaN(t))audioPlayer.currentTime=t;}catch(e){}if(pl)audioPlayer.play().catch(()=>{});r();};audioPlayer.addEventListener("loadedmetadata",m,{once:true});audioPlayer.src=u;audioPlayer.load();});}
const key=f=>`${f.name}_${f.size}_${f.lastModified}`;
function addFiles(fs){const h=new Set(queue.map(key));[...fs].forEach(f=>{if(!h.has(key(f))){queue.push(f);h.add(key(f));mixInSecs.push(null);}});render();}
function removeSong(i){queue.splice(i,1);mixInSecs.splice(i,1);render();}
function moveSong(i,d){const j=i+d;if(j<0||j>=queue.length)return;[queue[i],queue[j]]=[queue[j],queue[i]];[mixInSecs[i],mixInSecs[j]]=[mixInSecs[j],mixInSecs[i]];render();}
function render(){fileList.innerHTML="";if(!queue.length){fileList.innerHTML='<div class="empty">No songs yet. Click “Add songs”.</div>';return;}
queue.forEach((f,i)=>{const a=analysisByName[f.name];const meta=a?`<span class="key">${a.key} · ${a.camelot}</span> · ${a.bpm} BPM · energy ${a.energy}`:`${(f.size/1048576).toFixed(1)} MB`;
const mix=i===0?"":`<div class="mixin"><span>Mix in after</span><input type="number" min="1" max="3600" placeholder="auto" value="${mixInSecs[i]??""}" data-mixin="${i}"/><span>sec</span></div>`;
const c=document.createElement("div");c.className="song";c.innerHTML=`<div><strong>${i+1}. ${f.name}</strong><div class="meta">${meta}</div>${mix}</div><div class="acts"><button class="icon">↑</button><button class="icon">↓</button><button class="icon">×</button></div>`;
const mi=c.querySelector("input[data-mixin]");if(mi)mi.addEventListener("input",()=>{mixInSecs[i]=mi.value===""?null:Math.max(1,Number(mi.value)||0);});
const b=c.querySelectorAll(".acts button");b[0].onclick=()=>moveSong(i,-1);b[1].onclick=()=>moveSong(i,1);b[2].onclick=()=>removeSong(i);fileList.appendChild(c);});}
fileInput.addEventListener("change",()=>{addFiles(fileInput.files);fileInput.value="";});
clearQueue.addEventListener("click",()=>{queue=[];mixInSecs=[];render();});
document.getElementById("analyzeBtn").addEventListener("click",async()=>{if(!queue.length){statusBox.textContent="Add songs first.";return;}const b=document.getElementById("analyzeBtn");b.disabled=true;statusBox.textContent="Analyzing key, BPM and energy…";
try{const fd=new FormData();queue.forEach(f=>fd.append("files",f));const r=await fetch("/api/analyze",{method:"POST",body:fd});const d=await r.json();if(!r.ok)throw new Error(d.error||"Analyze failed.");
(d.analysis||[]).forEach(a=>{analysisByName[a.name]=a;});
if(document.getElementById("autoOrder").checked&&Array.isArray(d.suggested_order)){const o=d.suggested_order;queue=o.map(i=>queue[i]);mixInSecs=o.map(i=>mixInSecs[i]);statusBox.textContent="Analyzed and reordered harmonically.";}else statusBox.textContent="Analyzed.";render();}
catch(e){statusBox.textContent=e.message;}finally{b.disabled=false;}});
render();
mixButton.addEventListener("click",async()=>{if(!queue.length){statusBox.textContent="Add at least one song.";return;}
const fd=new FormData();queue.forEach(f=>fd.append("files",f));fd.append("segment_seconds",JSON.stringify(mixInSecs.slice(1)));
fd.append("crossfade_bars",document.getElementById("crossfadeBars").value);fd.append("tempo_match",document.getElementById("tempoMatch").checked);
fd.append("bass_swap",document.getElementById("bassSwap").checked);fd.append("auto_order_enabled",document.getElementById("autoOrder").checked);
fd.append("master_bpm",document.getElementById("masterBpm").value);fd.append("transition_fx",document.getElementById("transitionFx").value);
mixButton.disabled=true;statusBox.textContent="Analyzing & building your mix… preview keeps playing.";
try{const r=await fetch("/api/mix",{method:"POST",body:fd});const d=await r.json();if(!r.ok)throw new Error(d.error||"Mix failed.");
(d.analysis||[]).forEach(a=>{analysisByName[a.name]=a;});const wp=!!audioPlayer.src&&!audioPlayer.paused&&!audioPlayer.ended;await swap(d.audio_url);
downloadMix.href=d.audio_url;downloadRecipe.href=d.recipe_url;recipeBox.textContent=JSON.stringify(d.recipe,null,2);resultCard.style.display="block";render();
statusBox.textContent=wp?"New mix ready — playback continued.":"Mix created.";}
catch(e){statusBox.textContent=e.message;}finally{mixButton.disabled=false;}});
</script></body></html>"""


# ========================================================================== RUN
def _free_port(start: int = 8000, end: int = 8100) -> int:
    for p in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start


def _silence_proactor_reset() -> None:
    import asyncio
    try:
        base = asyncio.proactor_events._ProactorBasePipeTransport._call_connection_lost

        def patched(self, exc):
            try:
                base(self, exc)
            except (ConnectionResetError, ConnectionAbortedError):
                pass
        asyncio.proactor_events._ProactorBasePipeTransport._call_connection_lost = patched
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"\n  AI DJ Studio running at {url}\n  (press Ctrl+C to stop)\n")
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    _silence_proactor_reset()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
