#!/usr/bin/env python3
"""Builds the ACHERON intro soundtrack: placeholder VO, sound design, final mix.

Everything is synthesized here (no sample libraries), so the result is fully
owned. Outputs go to build/:
    vo_lines/NN.wav      processed VO takes, one per line
    stems/vo.wav         VO only, on the timeline
    stems/fx.wav         drone, heartbeat, plates, bell, breath, sub (no VO)
    audio_master.wav     site cut (headphones card + intro), -14 LUFS / -1 dBTP
    audio_master_reels.wav  Reels cut (stereo heartbeat open, no card)
    vo_timing.json       start and spoken length of every VO line (for captions)

To drop in a real human read: put one WAV per line in build/vo_human/NN.wav
(NN = line index in timeline.VO, 00..20) and run again with --human.
"""
import argparse, json, os, subprocess, sys
import numpy as np
import soundfile as sf
from scipy import signal

import timeline as TL

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(ROOT, "build")
SR = 48000
N = int(TL.DUR * SR)
rng = np.random.default_rng(7)


# ----------------------------------------------------------------- DSP helpers
def biquad(kind, f, sr=SR, q=0.707, gain_db=0.0):
    """RBJ cookbook biquad as an SOS row."""
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f / sr
    cw, sw = np.cos(w), np.sin(w)
    al = sw / (2 * q)
    if kind == "peak":
        b = [1 + al * A, -2 * cw, 1 - al * A]; a = [1 + al / A, -2 * cw, 1 - al / A]
    elif kind == "lowshelf":
        s = 2 * np.sqrt(A) * al
        b = [A * ((A + 1) - (A - 1) * cw + s), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - s)]
        a = [(A + 1) + (A - 1) * cw + s, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - s]
    elif kind == "highshelf":
        s = 2 * np.sqrt(A) * al
        b = [A * ((A + 1) + (A - 1) * cw + s), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - s)]
        a = [(A + 1) - (A - 1) * cw + s, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - s]
    else:
        raise ValueError(kind)
    b = np.array(b) / a[0]; a = np.array(a) / a[0]
    return np.concatenate([b, a])[None, :]


def filt(x, kind, f, order=2, sr=SR):
    sos = signal.butter(order, f, btype=kind, fs=sr, output="sos")
    return signal.sosfilt(sos, x, axis=0)


def eq(x, *rows):
    return signal.sosfilt(np.concatenate(rows), x, axis=0)


def env_follow(x, ms=10):
    k = max(1, int(SR * ms / 1000))
    return np.sqrt(np.convolve(x ** 2, np.ones(k) / k, mode="same"))


def compress(x, thresh_db=-24, ratio=3.0, attack=0.005, release=0.12):
    lvl = 20 * np.log10(env_follow(x, 8) + 1e-9)
    over = np.maximum(lvl - thresh_db, 0)
    gr = over * (1 - 1 / ratio)
    # one-pole smoothing with separate attack/release
    a_a, a_r = np.exp(-1 / (attack * SR)), np.exp(-1 / (release * SR))
    sm = np.empty_like(gr); g = 0.0
    for i, v in enumerate(gr):
        c = a_a if v > g else a_r
        g = c * g + (1 - c) * v
        sm[i] = g
    return x * 10 ** (-sm / 20)


def pink(n, ch=1):
    w = rng.standard_normal((n, ch))
    F = np.fft.rfft(w, axis=0)
    f = np.fft.rfftfreq(n); f[0] = f[1]
    F /= np.sqrt(f)[:, None]
    out = np.fft.irfft(F, n=n, axis=0)
    return out / np.abs(out).max()


def smooth_noise(n, rate_hz, seed):
    """Slow random curve in [-1, 1] at audio rate."""
    r = np.random.default_rng(seed)
    pts = int(TL.DUR * rate_hz) + 4
    v = r.uniform(-1, 1, pts)
    xs = np.linspace(0, pts - 1, n)
    return np.interp(xs, np.arange(pts), signal.savgol_filter(v, 5, 2) if pts > 5 else v)


def make_ir(rt_low=4.6, rt_mid=3.6, rt_high=1.8, predelay=0.035, seed=3, length=None):
    """Synthetic stereo cathedral impulse response, darker as it decays."""
    r = np.random.default_rng(seed)
    L = int(SR * (length or rt_low * 1.15))
    t = np.arange(L) / SR
    out = np.zeros((L, 2))
    for lo, hi, rt in [(None, 400, rt_low), (400, 3000, rt_mid), (3000, None, rt_high)]:
        n = r.standard_normal((L, 2))
        if lo is None:
            n = filt(n, "low", hi, 4)
        elif hi is None:
            n = filt(n, "high", lo, 4)
        else:
            n = filt(n, "band", [lo, hi], 2)
        out += n * (10 ** (-3 * t / rt))[:, None]
    # sparse early reflections
    for d, g in [(0.011, 0.5), (0.019, 0.42), (0.027, 0.36), (0.041, 0.3), (0.055, 0.22), (0.072, 0.18)]:
        i = int(d * SR)
        out[i, 0] += g * r.choice([-1, 1]); out[i + int(0.0013 * SR), 1] += g * r.choice([-1, 1])
    pd = int(predelay * SR)
    out = np.concatenate([np.zeros((pd, 2)), out])
    fade = np.ones(len(out)); fade[-int(0.3 * SR):] = np.linspace(1, 0, int(0.3 * SR))
    out *= fade[:, None]
    return out / np.sqrt((out ** 2).sum() / 2)


def reverb(x, ir, wet_db=-12, hp=180, lp=7000):
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    w = np.stack([signal.fftconvolve(x[:, c], ir[:, c])[: len(x)] for c in range(2)], 1)
    w = filt(filt(w, "high", hp, 2), "low", lp, 2)
    return w * 10 ** (wet_db / 20)


def place(buf, clip, t, gain=1.0):
    i = int(round(t * SR))
    if clip.ndim == 1 and buf.ndim == 2:
        clip = np.stack([clip, clip], 1)
    j = min(len(buf), i + len(clip))
    if j > i:
        buf[i:j] += clip[: j - i] * gain


def adsr(n, a, d_tau, sr=SR):
    t = np.arange(n) / sr
    e = np.exp(-np.maximum(t - a, 0) / d_tau)
    e[t < a] = t[t < a] / max(a, 1e-6)
    return e


def db(x):
    return 10 ** (x / 20)


# ----------------------------------------------------------------- voiceover
def build_vo_takes(voice="am_michael:0.7,am_onyx:0.3", speed=0.92, human=False):
    """Kokoro TTS (blended low voice) → warm, dark close-mic chain.

    No pitch shifting, saturation or added noise: each of those pushed the
    synthetic voice further toward sounding like a robot. Depth comes from the
    voice blend itself, and the top end is rolled off where TTS artifacts live.
    """
    out_dir = os.path.join(BUILD, "vo_lines"); os.makedirs(out_dir, exist_ok=True)
    raw_dir = os.path.join(BUILD, "vo_raw"); os.makedirs(raw_dir, exist_ok=True)
    takes = []
    if human:
        src_dir = os.path.join(BUILD, "vo_human")
        for i in range(len(TL.VO)):
            p = os.path.join(src_dir, f"{i:02d}.wav")
            a, sr = sf.read(p, always_2d=True)
            a = a.mean(1)
            if sr != SR:
                a = signal.resample_poly(a, SR, sr)
            takes.append(vo_chain(a, synthetic=False))
        return takes

    from kokoro_onnx import Kokoro
    mdir = os.environ.get("KOKORO_DIR", os.path.join(ROOT, "build", "models"))
    k = Kokoro(os.path.join(mdir, "kokoro-v1.0.onnx"), os.path.join(mdir, "voices-v1.0.bin"))
    if ":" in voice:                      # "name:weight,name:weight" blend
        voice = sum(float(w) * k.get_voice_style(n) for n, w in (p.split(":") for p in voice.split(",")))
    # single words on the beat get a brisker read so they fit the 0.8-0.9 s grid
    fast = {3: 1.0, 4: 1.0, 5: 0.97, 12: 0.97, 13: 0.97, 14: 0.97, 19: 0.97, 20: 0.97}
    for i, cue in enumerate(TL.VO):
        text = cue[1]
        sp = fast.get(i, speed)
        if len(cue) > 2:
            a, sr = k.create(cue[2], voice=voice, speed=sp, is_phonemes=True)
        else:
            a, sr = k.create(text, voice=voice, speed=sp, lang="en-us")
        rp = os.path.join(raw_dir, f"{i:02d}.wav")
        sf.write(rp, a, sr)
        pp = os.path.join(raw_dir, f"{i:02d}_48k.wav")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", rp, "-af",
                        f"aresample={SR}:resampler=soxr", "-ac", "1", pp], check=True)
        b, _ = sf.read(pp)
        takes.append(vo_chain(b))
        sf.write(os.path.join(out_dir, f"{i:02d}.wav"), takes[-1], SR, subtype="PCM_24")
    return takes


def vo_chain(x, synthetic=True):
    x = x / (np.abs(x).max() + 1e-9) * 0.7
    x = filt(x, "high", 70, 2)
    x = eq(x,
           biquad("lowshelf", 170, gain_db=3.0, q=0.7),     # chest
           biquad("peak", 320, gain_db=-2.5, q=1.0),         # clear the mud
           biquad("peak", 2600, gain_db=-2.0, q=1.2),        # take the edge off
           biquad("highshelf", 4200, gain_db=-7.0, q=0.7))   # treble down
    if synthetic:
        x = filt(x, "low", 8000, 4)                          # TTS fizz lives up here
    x = compress(x, thresh_db=-24, ratio=2.5)
    x = x / (np.abs(x).max() + 1e-9) * db(-3)
    # trim lead silence so the first syllable lands on the cue
    # (low threshold + 45 ms of lead so soft openers like the F in "Five" survive)
    e = env_follow(x, 5)
    on = np.argmax(e > e.max() * 0.012)
    start = max(0, on - int(0.045 * SR))
    x = x[start:]
    off = len(e) - np.argmax(e[::-1] > e.max() * 0.01)
    x = x[: max(1, off - start + int(0.08 * SR))]
    fade = min(len(x), int(0.008 * SR))
    x[:fade] *= np.linspace(0, 1, fade)
    return x


def fit_takes(takes):
    """Time-compress any take that would run into the next cue."""
    starts = [c[0] for c in TL.VO]
    report = []
    for i, x in enumerate(takes):
        nxt = starts[i + 1] if i + 1 < len(starts) else TL.CUT_TO_BLACK
        if i == 12:                       # "You didn't." must clear the silence
            nxt = TL.SILENCE[0] + 0.05
        room = nxt - starts[i] - 0.03
        dur = len(x) / SR
        if dur > room:
            factor = dur / room
            tmp_in = os.path.join(BUILD, "vo_raw", f"fit_in_{i:02d}.wav")
            tmp_out = os.path.join(BUILD, "vo_raw", f"fit_out_{i:02d}.wav")
            sf.write(tmp_in, x, SR, subtype="FLOAT")
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", tmp_in, "-af",
                            f"rubberband=tempo={factor:.4f}:pitchq=quality:transients=smooth", tmp_out], check=True)
            x, _ = sf.read(tmp_out)
            takes[i] = x
        report.append((i, starts[i], round(dur, 2), round(room, 2), round(len(takes[i]) / SR, 2)))
    return takes, report


# ----------------------------------------------------------------- sound design
def drone():
    t = np.arange(N) / SR
    root = 36.71                                   # D1
    # half-step rises with a short glide, back to the root on the bell
    (t1, s1), (t2, s2) = TL.DRONE_STEPS
    semis = np.interp(t, [0, t1, t1 + 0.45, t2, t2 + 0.45, TL.BELL - 1e-3, TL.BELL], [0, 0, s1, s1, s2, s2, 0])
    ratio = 2 ** (semis / 12)
    out = np.zeros((N, 2))
    voices = [(1, 1.0), (2, 0.55), (3, 0.42), (4, 0.22), (6, 0.12)]   # root, octave, fifth, 2 oct, fifth+2
    for ch in range(2):
        for mult, g in voices:
            for det in (-0.12, 0.0, 0.13):
                f = root * mult * ratio * (1 + det / 100 * (1 + ch * 0.3))
                ph = 2 * np.pi * np.cumsum(f) / SR + rng.uniform(0, 6.28)
                # organ-like: a few harmonics with soft roll-off
                s = np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.18 * np.sin(3 * ph) + 0.08 * np.sin(5 * ph)
                out[:, ch] += s * g / 3
    # slowly breathing low-pass
    lfo = 0.5 + 0.5 * smooth_noise(N, 0.35, 11)
    lo = filt(out, "low", 260, 2); hi = filt(out, "low", 900, 2)
    out = lo * (1 - lfo[:, None] * 0.6) + hi * (lfo[:, None] * 0.6)
    # arrangement
    lvl = np.interp(t, [0, 3.0, 6, 13, 17.6, 21.7, 24, 27.5, 28.05, 30, 33, 36.8, 41, 44.6],
                       [0, 0.55, 0.6, 0.7, 0.8, 0.95, 0.8, 0.85, 0.0, 0.55, 0.7, 1.0, 0.8, 0.75])
    return out * lvl[:, None] / np.abs(out).max() * db(-10)


def wind():
    n = pink(N, 2)
    b = filt(n, "band", [180, 1400], 2)
    sweep = 0.5 + 0.5 * smooth_noise(N, 0.5, 21)
    b2 = filt(n, "band", [400, 2600], 2)
    w = b * (1 - sweep[:, None]) + b2 * sweep[:, None]
    t = np.arange(N) / SR
    lvl = np.interp(t, [0, 1.5, 6, 9, 13, 21.7, 24, 27.5, 28.05, 33, 41, 44.6], [0, 1, 0.8, 0.35, 0.25, 0.4, 0.6, 0.5, 0, 0.3, 0.35, 0.3])
    return w / np.abs(w).max() * lvl[:, None] * db(-17)


def heartbeat(gain=1.0):
    def thump(f0, f1, dur, amp):
        n = int(dur * SR); t = np.arange(n) / SR
        f = f1 + (f0 - f1) * np.exp(-t / 0.04)
        s = np.sin(2 * np.pi * np.cumsum(f) / SR) * adsr(n, 0.004, 0.09)
        s += 0.35 * np.sin(4 * np.pi * np.cumsum(f) / SR) * adsr(n, 0.003, 0.05)   # audible on phones
        click = filt(rng.standard_normal(n), "low", 900, 2) * adsr(n, 0.001, 0.008) * 0.25
        return np.tanh(1.6 * (s + click)) * amp
    lub, dub = thump(70, 46, 0.45, 1.0), thump(82, 54, 0.35, 0.62)
    out = np.zeros(N)
    beats = TL.HEARTBEATS + [12.92, TL.FINAL_BEAT]
    for i, tb in enumerate(beats):
        a = 0.55 if tb == 12.92 else (0.8 if i < 3 else 1.0)
        place(out, lub, tb, a); place(out, dub, tb + TL.DUB_OFFSET, a)
    return out * gain * db(-6)


def clang(seed, base=190.0):
    r = np.random.default_rng(seed)
    n = int(3.2 * SR); t = np.arange(n) / SR
    ratios = np.array([1.0, 2.32, 2.78, 4.07, 5.41, 6.9, 8.93, 11.2])
    decays = np.array([1.6, 1.1, 0.9, 0.6, 0.45, 0.32, 0.22, 0.15])
    s = np.zeros(n)
    for rt, dk in zip(ratios, decays):
        for d in (-0.6, 0.6):                                  # beating pairs
            f = base * rt * r.uniform(0.985, 1.015) + d
            s += np.sin(2 * np.pi * f * t + r.uniform(0, 6)) * np.exp(-t / dk) / rt ** 0.55
    thud = np.sin(2 * np.pi * (52 + 40 * np.exp(-t / 0.03)) * t) * np.exp(-t / 0.12) * 1.6
    hit = filt(r.standard_normal(n), "band", [900, 7000], 2) * np.exp(-t / 0.012) * 1.2
    out = s / np.abs(s).max() * 0.8 + thud + hit
    return np.tanh(1.2 * out)


def breath(kind, seed):
    r = np.random.default_rng(seed)
    dur = 0.75 if kind == "in" else 1.0
    n = int(dur * SR); t = np.arange(n) / SR
    no = r.standard_normal(n)
    v = filt(no, "band", [350, 3200], 2)
    v = eq(v, biquad("peak", 1100, gain_db=6, q=2.5), biquad("peak", 2300, gain_db=4, q=3))
    if kind == "in":
        e = np.sin(np.pi * t / dur) ** 1.5 * np.clip(t / (dur * 0.7), 0, 1)
    else:
        e = np.clip(t / 0.06, 0, 1) * np.exp(-t / 0.35)
    return v / np.abs(v).max() * e


def bell():
    n = int(9.5 * SR); t = np.arange(n) / SR
    prime = 146.83                                           # D3 strike note
    parts = [(0.5, 1.0, 8.5), (1.0, 0.8, 6.5), (1.19, 0.55, 5.0), (1.5, 0.35, 3.4), (2.0, 0.7, 4.2),
             (2.51, 0.25, 2.2), (2.66, 0.22, 2.0), (3.01, 0.2, 1.7), (4.07, 0.14, 1.1), (5.2, 0.08, 0.7)]
    s = np.zeros(n)
    for ratio, amp, dk in parts:
        for d in (-0.45, 0.45):
            f = prime * ratio + d
            s += amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) * np.exp(-t / dk)
    strike = filt(rng.standard_normal(n), "band", [600, 6000], 2) * np.exp(-t / 0.006) * 2.0
    s = s / np.abs(s).max() + strike
    return np.tanh(s * 0.9)


def swell_and_hit():
    out = np.zeros((N, 2))
    t0 = TL.SWELL_HIT
    # 2.8 s riser: filtered noise opening up + rising sub
    rn = int(2.8 * SR); rt = np.arange(rn) / SR
    noise = pink(rn, 2)
    p = rt / 2.8
    fc = np.log2(200 + 3800 * p ** 2)
    riser = np.zeros_like(noise)
    for c in (250, 500, 1000, 2000, 4000):           # crossfaded bands, opening upward
        w = np.exp(-((fc - np.log2(c)) / 0.8) ** 2)
        riser += filt(noise, "band", [c / 1.5, c * 1.5], 2) * w[:, None]
    riser *= ((rt / 2.8) ** 2.2)[:, None] * 0.6
    sub = np.sin(2 * np.pi * np.cumsum(30 + 14 * (rt / 2.8)) / SR) * (rt / 2.8) ** 2 * 0.7
    riser += sub[:, None]
    place(out, riser, t0 - 2.8)
    # hit: sub-bass boom, pitch-dropping, ~3 s
    hn = int(3.6 * SR); ht = np.arange(hn) / SR
    f = 30 + 34 * np.exp(-ht / 0.09)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-ht / 1.1)
    boom += 0.3 * np.sin(4 * np.pi * np.cumsum(f) / SR) * np.exp(-ht / 0.5)
    crack = filt(rng.standard_normal(hn), "band", [80, 2500], 2) * np.exp(-ht / 0.05) * 0.9
    hit = np.tanh(1.8 * (boom + crack))
    place(out, hit, t0, 1.0)
    return out * db(-4)


def build_fx():
    ir = make_ir()
    ir_big = make_ir(rt_low=6.5, rt_mid=5.2, rt_high=2.4, predelay=0.05, seed=5)
    fx = np.zeros((N, 2))
    fx += drone()
    fx += wind()

    hb = heartbeat()
    fx += np.stack([hb, hb], 1) + reverb(hb, ir, wet_db=-16, hp=60, lp=2000)

    pl = np.zeros(N)
    for i, tc in enumerate(TL.CLANGS):
        place(pl, clang(100 + i, base=[176, 198, 186, 168, 158][i]), tc, [0.85, 0.8, 1.0, 0.95, 0.9][i])
    fx += np.stack([pl, pl], 1) * db(-5) + reverb(pl, ir_big, wet_db=-9)

    br = np.zeros(N)
    for i, (tb, kind) in enumerate(TL.BREATHS):
        place(br, breath(kind, 200 + i), tb, 1.0)
    fx += np.stack([br, br * 0.92], 1) * db(-19) + reverb(br, ir, wet_db=-24)

    bl = np.zeros(N)
    place(bl, bell(), TL.BELL)
    fx += np.stack([bl, bl], 1) * db(-6) + reverb(bl, ir_big, wet_db=-5, hp=120, lp=9000)

    fx += swell_and_hit()
    return fx


def preroll():
    """Headphones card: a low pulse in the left ear, then the right, over a soft air sweep."""
    n = int(TL.PRE * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))

    def panned(x, pan):
        a = (pan + 1) * np.pi / 4
        return np.stack([x * np.cos(a), x * np.sin(a)], 1)

    air = filt(pink(n, 1)[:, 0], "band", [350, 2600], 2)
    air /= np.abs(air).max()
    env = np.sin(np.pi * np.clip((t - 0.2) / 3.0, 0, 1)) ** 2
    pan = np.clip((t - 0.5) / 2.0, 0, 1) * 1.6 - 0.8          # drifts left to right
    a = (pan + 1) * np.pi / 4
    out += np.stack([air * np.cos(a), air * np.sin(a)], 1) * env[:, None] * db(-30)
    # quiet open fifth on the intro drone's root, so the hand-over feels like one piece
    pad = np.sin(2 * np.pi * 73.42 * t) + 0.6 * np.sin(2 * np.pi * 110.0 * t) + 0.25 * np.sin(2 * np.pi * 146.83 * t)
    pe = np.clip(t / 0.9, 0, 1) * np.clip((3.3 - t) / 0.8, 0, 1)
    out += np.stack([pad, pad], 1) * pe[:, None] * db(-36)
    for tp, side in TL.PRE_PULSES:
        m = int(1.6 * SR)
        tt = np.arange(m) / SR
        f = 58 + 30 * np.exp(-tt / 0.05)
        ph = 2 * np.pi * np.cumsum(f) / SR
        x = np.sin(ph) * np.exp(-tt / 0.32) + 0.4 * np.sin(2 * ph) * np.exp(-tt / 0.18)
        x += filt(rng.standard_normal(m), "band", [500, 3000], 2) * np.exp(-tt / 0.01) * 0.15
        i = int(tp * SR)
        j = min(n, i + m)
        out[i:j] += panned(np.tanh(1.4 * x), 0.85 * side)[: j - i] * db(-12)
    ir = make_ir(rt_low=2.6, rt_mid=2.2, rt_high=1.0, predelay=0.03, seed=17)
    wet = np.stack([signal.fftconvolve(out[:, c], ir[:, c])[:n] for c in range(2)], 1)
    out = out + filt(wet, "high", 150, 2) * db(-14)
    return out * np.clip((TL.PRE - 0.05 - t) / 0.2, 0, 1)[:, None]


def reels_open():
    """Reels opener: one heartbeat split across the ears, lub left, dub right."""
    n = int(1.6 * SR)
    out = np.zeros((n, 2))
    for (tb, side), amp in zip(TL.REELS_BEATS, (1.0, 0.7)):
        m = int(1.2 * SR)
        tt = np.arange(m) / SR
        f = 56 + 32 * np.exp(-tt / 0.04)
        ph = 2 * np.pi * np.cumsum(f) / SR
        x = np.sin(ph) * np.exp(-tt / 0.2) + 0.45 * np.sin(2 * ph) * np.exp(-tt / 0.1)
        x += filt(rng.standard_normal(m), "low", 900, 2) * np.exp(-tt / 0.008) * 0.25
        x = np.tanh(1.6 * x) * amp
        a = (0.9 * side + 1) * np.pi / 4
        i = int(tb * SR)
        out[i:i + m] += np.stack([x * np.cos(a), x * np.sin(a)], 1)[: n - i]
    ir = make_ir(rt_low=2.6, rt_mid=2.2, rt_high=1.0, predelay=0.03, seed=21)
    wet = np.stack([signal.fftconvolve(out[:, c], ir[:, c])[:n] for c in range(2)], 1)
    return out + filt(wet, "high", 120, 2) * db(-16)


def loudnorm(src, out):
    """Two-pass EBU R128 normalisation to -14 LUFS, -1 dBTP."""
    p = subprocess.run(["ffmpeg", "-hide_banner", "-i", src, "-af", "loudnorm=I=-14:TP=-1.0:LRA=11:print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True)
    js = json.loads(p.stderr[p.stderr.rfind("{"):])
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-af",
                    "loudnorm=I=-14:TP=-1.0:LRA=11:measured_I={input_i}:measured_TP={input_tp}:measured_LRA={input_lra}:"
                    "measured_thresh={input_thresh}:offset={target_offset}:linear=true".format(**js),
                    "-ar", str(SR), "-c:a", "pcm_s24le", out], check=True)
    print("measured", {k: js[k] for k in ("input_i", "input_tp", "input_lra")}, "->", out)


def gate(n=N):
    """Silence window before the bell and the cut to black."""
    t = np.arange(n) / SR
    s0, s1 = TL.SILENCE
    g = np.where(t < s1, np.clip((s0 - t) / 0.02, 0, 1), 1.0)
    after = t >= TL.CUT_TO_BLACK
    g[after] *= np.exp(-(t[after] - TL.CUT_TO_BLACK) / 0.09)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", action="store_true", help="use build/vo_human/NN.wav instead of TTS")
    ap.add_argument("--voice", default="am_michael:0.7,am_onyx:0.3")
    args = ap.parse_args()
    os.makedirs(os.path.join(BUILD, "stems"), exist_ok=True)

    takes = build_vo_takes(args.voice, human=args.human)
    takes, report = fit_takes(takes)
    for r in report:
        print("VO %02d  @%5.2fs  len %.2fs  room %.2fs  -> %.2fs" % r)
    # spoken length of each line, for the Reels captions
    with open(os.path.join(BUILD, "vo_timing.json"), "w") as fh:
        json.dump([[c[0], len(x) / SR] for c, x in zip(TL.VO, takes)], fh)
    vo = np.zeros((N, 2))
    for cue, x in zip(TL.VO, takes):
        place(vo, x, cue[0])
    ir_vo = make_ir(rt_low=3.4, rt_mid=3.0, rt_high=1.4, predelay=0.04, seed=9)
    vo_wet = reverb(vo.mean(1), ir_vo, wet_db=-15, hp=220, lp=4000)
    vo_bus = vo + vo_wet

    fx = build_fx()
    # duck the bed ~3 dB under the voice so every word reads on a phone speaker
    ve = env_follow(vo.mean(1), 60)
    duck = 1 - 0.3 * np.clip(ve / (ve.max() + 1e-9) * 4, 0, 1)
    fx *= duck[:, None]

    g = gate()
    vo_bus *= g[:, None]; fx *= g[:, None]
    # Reels cut: trim the lead-in, stereo heartbeat on frame one
    k0 = int(TL.REELS_OFFSET * SR)
    reels = vo_bus[k0:] * db(1.5) + fx[k0:]
    op = reels_open()
    reels[: len(op)] += op * db(-7)
    reels = reels / np.abs(reels).max() * db(-1)
    pre_r = os.path.join(BUILD, "mix_premaster_reels.wav")
    sf.write(pre_r, reels, SR, subtype="FLOAT")
    loudnorm(pre_r, os.path.join(BUILD, "audio_master_reels.wav"))
    # site cut: the headphones card goes in front of everything
    card = preroll()
    vo_bus = np.concatenate([np.zeros_like(card), vo_bus])
    fx = np.concatenate([card, fx])
    sf.write(os.path.join(BUILD, "stems", "vo.wav"), vo_bus * db(-1), SR, subtype="PCM_24")
    sf.write(os.path.join(BUILD, "stems", "fx.wav"), fx / max(1, np.abs(fx).max()) * db(-1), SR, subtype="PCM_24")

    mix = vo_bus * db(1.5) + fx
    mix = mix / np.abs(mix).max() * db(-1)
    pre = os.path.join(BUILD, "mix_premaster.wav")
    sf.write(pre, mix, SR, subtype="FLOAT")

    loudnorm(pre, os.path.join(BUILD, "audio_master.wav"))


if __name__ == "__main__":
    main()
