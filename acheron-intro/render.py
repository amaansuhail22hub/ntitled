#!/usr/bin/env python3
"""ACHERON — "Ahead of Time" intro. Frame renderer.

Every frame is a pure function of time, drawn in a 1080x1920 design space and
rasterised at any scale (2.0 = 2160x3840 UHD). Run `python3 render.py --help`.

The brand lockup comes from assets/acheron-logo.png (white on black, or a
transparent PNG): mark on the left, ACHERON, then the AHEAD OF TIME row. It is
split into those parts automatically. Without it, placeholders are drawn.
"""
import argparse, json, math, os, subprocess, sys, time
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import timeline as TL

cv2.setNumThreads(1)
ROOT = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(ROOT, "fonts")
ASSETS = os.path.join(ROOT, "assets")

U, W, H = 2.0, 2160, 3840          # set by setup()
CUT = "site"                       # "site": headphones card + intro, "reels": hook-first cut
FH, FW = 480, 270                  # low-res field grid (design / 4)
C = {}                             # per-process caches


def setup(scale, cut="site"):
    global U, W, H, CUT
    U = float(scale)
    CUT = cut
    W, H = int(round(TL.DESIGN_W * U)), int(round(TL.DESIGN_H * U))
    C.clear()


# ============================================================== easing / util
def cl(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lin(t, t0, t1):
    return cl((t - t0) / (t1 - t0))


def ss(x):
    x = cl(x)
    return x * x * (3 - 2 * x)


def eio(x):
    x = cl(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def eout(x):
    x = cl(x)
    return 1 - (1 - x) ** 3


def ssv(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def decay(t, t0, tau):
    return math.exp(-(t - t0) / tau) if t >= t0 else 0.0


def heart_env(t, beats=TL.HEARTBEATS):
    e = 0.0
    for b in beats:
        e += decay(t, b, 0.22) + 0.6 * decay(t, b + TL.DUB_OFFSET, 0.18)
    return min(e, 1.4)


# ============================================================== frame buffers
class Frame:
    def __init__(self, t):
        self.t = t
        self.L = np.zeros((H, W), np.float32)      # luminance
        self.G = np.zeros((H, W), np.float32)      # glow source
        self.lowL = np.zeros((FH, FW), np.float32)  # soft light fields
        self.lowG = np.zeros((FH, FW), np.float32)
        self.lowR = np.zeros((FH, FW), np.float32)  # red, only for heartbeat line and bell ring
        self.ash = np.zeros((H // 2, W // 2), np.float32)
        self.expo = 1.0
        self.fog_shock = None

    def add_layer(self, lay, gain=1.0, glow=0.0, colmask=None):
        f = lay.astype(np.float32)
        f *= gain / 255.0
        if colmask is not None:
            f *= colmask[None, :]
        self.L += f
        if glow:
            self.G += f * glow


def new_layer():
    return np.zeros((H, W), np.uint8)


def lowgrid():
    if "grid" not in C:
        gy, gx = np.mgrid[0:FH, 0:FW].astype(np.float32)
        C["grid"] = ((gx + 0.5) * TL.DESIGN_W / FW, (gy + 0.5) * TL.DESIGN_H / FH)
    return C["grid"]


def up(a):
    return cv2.resize(a, (W, H), interpolation=cv2.INTER_CUBIC)


# ============================================================== pen (line art)
class Pen:
    """Design-space drawing into a uint8 layer with a camera (zoom + shake)."""

    def __init__(self, img, cx=540, cy=960, s=1.0, dx=0.0, dy=0.0):
        self.img, self.cx, self.cy, self.s, self.dx, self.dy = img, cx, cy, s, dx, dy

    def xy(self, x, y):
        return ((x - self.cx) * self.s + self.cx + self.dx) * U, ((y - self.cy) * self.s + self.cy + self.dy) * U

    def P(self, x, y):
        X, Y = self.xy(x, y)
        return (int(round(X * 16)), int(round(Y * 16)))

    def pts(self, arr):
        a = np.asarray(arr, np.float64)
        X = ((a[:, 0] - self.cx) * self.s + self.cx + self.dx) * U
        Y = ((a[:, 1] - self.cy) * self.s + self.cy + self.dy) * U
        return np.round(np.stack([X, Y], 1) * 16).astype(np.int32)

    def th(self, w):
        return max(1, int(round(w * self.s * U)))

    @staticmethod
    def c(v):
        return int(cl(v) * 255)

    def line(self, p0, p1, v, w=1.5):
        if v <= 0.004:
            return
        cv2.line(self.img, self.P(*p0), self.P(*p1), self.c(v), self.th(w), cv2.LINE_AA, 4)

    def poly(self, pts, v, w=1.5, closed=False):
        if v <= 0.004:
            return
        cv2.polylines(self.img, [self.pts(pts)], closed, self.c(v), self.th(w), cv2.LINE_AA, 4)

    def fill(self, pts, v=0.0):
        cv2.fillPoly(self.img, [self.pts(pts)], self.c(v), cv2.LINE_AA, 4)

    def circle(self, c, r, v, w=1.5, fill=False):
        X, Y = self.P(*c)
        rr = int(round(r * self.s * U * 16))
        cv2.circle(self.img, (X, Y), rr, self.c(v), -1 if fill else self.th(w), cv2.LINE_AA, 4)


def ellipse_pts(cx, cy, rx, ry, a0=0.0, a1=360.0, n=120, rot=0.0):
    a = np.radians(np.linspace(a0, a1, n))
    x, y = rx * np.cos(a), ry * np.sin(a)
    if rot:
        cr, sr = math.cos(rot), math.sin(rot)
        x, y = x * cr - y * sr, x * sr + y * cr
    return np.stack([cx + x, cy + y], 1)


def rrect(x0, y0, x1, y1, r, n=6):
    pts = []
    for cx, cy, a0 in [(x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)]:
        for a in np.radians(np.linspace(a0, a0 + 90, n)):
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts)


def rotate_pts(pts, cx, cy, ang):
    c, s = math.cos(ang), math.sin(ang)
    p = np.asarray(pts, np.float64) - (cx, cy)
    return np.stack([p[:, 0] * c - p[:, 1] * s, p[:, 0] * s + p[:, 1] * c], 1) + (cx, cy)


# ============================================================== textures
def tileable(n, beta, seed):
    r = np.random.default_rng(seed)
    F = np.fft.fft2(r.standard_normal((n, n)))
    k = np.fft.fftfreq(n)
    kk = np.sqrt(k[None, :] ** 2 + k[:, None] ** 2)
    kk[0, 0] = 1
    F = F / kk ** beta
    F[0, 0] = 0
    o = np.real(np.fft.ifft2(F)).astype(np.float32)
    return (o - o.min()) / (o.max() - o.min())


def grain_bank():
    if "grain" not in C:
        r = np.random.default_rng(1)
        gh, gw = int(H / 1.7), int(W / 1.7)
        bank = []
        for _ in range(6):
            g = r.standard_normal((gh, gw)).astype(np.float32)
            g = cv2.GaussianBlur(g, (0, 0), 0.5)
            g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
            g /= g.std()
            bank.append(np.clip(g * 40, -127, 127).astype(np.int8))
        C["grain"] = bank
    return C["grain"]


def vignette():
    if "vig" not in C:
        yy, xx = np.mgrid[0:240, 0:135].astype(np.float32)
        nx, ny = (xx / 134 - 0.5) * 2, (yy / 239 - 0.5) * 2
        r = np.sqrt(nx ** 2 + 0.72 * ny ** 2)
        v = 1 - 0.66 * np.clip((r - 0.42) / 0.95, 0, 1) ** 1.5
        C["vig"] = cv2.resize(v, (W, H), interpolation=cv2.INTER_CUBIC)
    return C["vig"]


# ============================================================== text
def text_sprite(text, font, size, track):
    key = ("txt", text, font, size, track)
    if key in C:
        return C[key]
    px = size * U
    f = ImageFont.truetype(os.path.join(FONTS, font), int(round(px)))
    adv = [f.getlength(ch) for ch in text]
    tr = track * px
    total = sum(adv) + tr * (len(text) - 1)
    asc, desc = f.getmetrics()
    pad = int(px * 0.6)
    img = Image.new("L", (int(total) + 2 * pad, asc + desc + 2 * pad), 0)
    d = ImageDraw.Draw(img)
    x, cols = pad, []
    for ch, a in zip(text, adv):
        d.text((x, pad), ch, font=f, fill=255)
        cols.append((x, x + a))
        x += a + tr
    arr = np.asarray(img, np.float32) / 255.0
    return _finish_sprite(key, arr, cols, text)


def _finish_sprite(key, arr, cols, text):
    ys, xs = np.nonzero(arr > 0.02)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = int(6 * U) + 2
    Y0, Y1, X0, X1 = max(0, y0 - m), min(arr.shape[0], y1 + m), max(0, x0 - m), min(arr.shape[1], x1 + m)
    spr = np.ascontiguousarray(arr[Y0:Y1, X0:X1])
    col = np.zeros(spr.shape[1], np.int16)
    centers = np.array([(a + b) / 2 - X0 for a, b in cols])
    xs_ = np.arange(spr.shape[1])
    col[:] = np.abs(xs_[:, None] - centers[None, :]).argmin(1)
    # dissolve threshold: noise + left-to-right drift
    r = np.random.default_rng(abs(hash(key)) % 2 ** 32)
    cell = max(2, int(5 * U))
    nz = cv2.resize(r.random((spr.shape[0] // cell + 2, spr.shape[1] // cell + 2)).astype(np.float32),
                    (spr.shape[1], spr.shape[0]), interpolation=cv2.INTER_CUBIC)
    nz = cv2.GaussianBlur(nz, (0, 0), cell * 0.6)
    nz = (nz - nz.min()) / (nz.max() - nz.min() + 1e-6)
    xn = np.linspace(0, 1, spr.shape[1], dtype=np.float32)[None, :]
    thr = np.clip(0.42 * nz + 0.58 * xn, 0, 1).astype(np.float32)
    # ash emission points sampled from ink
    iy, ix = np.nonzero(spr > 0.35)
    k = min(len(iy), int(700 + spr.sum() / (U * U) / 16))
    sel = r.choice(len(iy), size=k, replace=False) if len(iy) > k else np.arange(len(iy))
    pts = dict(x=ix[sel].astype(np.float32), y=iy[sel].astype(np.float32), thr=thr[iy[sel], ix[sel]],
               v0=r.uniform(25, 90, len(sel)).astype(np.float32), acc=r.uniform(10, 60, len(sel)).astype(np.float32),
               sw=r.uniform(4, 22, len(sel)).astype(np.float32), ph=r.uniform(0, 6.28, len(sel)).astype(np.float32),
               fq=r.uniform(0.3, 1.1, len(sel)).astype(np.float32), b=r.uniform(0.35, 1.0, len(sel)).astype(np.float32),
               life=r.uniform(0.6, 1.4, len(sel)).astype(np.float32))
    cy = ((y0 + y1) / 2 - Y0)
    cx = ((x0 + x1) / 2 - X0)
    out = dict(a=spr, col=col, cx=cx, cy=cy, n=len(text), text=text, thr=thr, pts=pts)
    C[key] = out
    return out


def _per_letter(spr, arr):
    arr = np.asarray(arr, np.float32)
    if "lab" in spr:
        return arr[spr["lab"]]
    return arr[spr["col"]][None, :]


def blit(dst, spr, x0, y0, val, mode="over"):
    """Composite spr (premultiplied coverage * val) into dst at integer top-left."""
    h, w = spr.shape
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(dst.shape[1], x0 + w), min(dst.shape[0], y0 + h)
    if X1 <= X0 or Y1 <= Y0:
        return
    s = spr[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    reg = dst[Y0:Y1, X0:X1]
    if mode == "over":
        reg *= 1 - s
        reg += s * val
    else:
        reg += s * val


def draw_text(F, spr, cx, cy, val, alpha=1.0, glow=0.25, letter_alpha=None, letter_glow=None, dissolve=None):
    t = F.t
    a = spr["a"]
    m = a * alpha
    if letter_alpha is not None:
        m = m * _per_letter(spr, letter_alpha)
    x0 = int(round(cx * U - spr["cx"]))
    y0 = int(round(cy * U - spr["cy"]))
    edge = None
    if dissolve is not None:
        t0, D = dissolve
        q = -0.1 + 1.2 * (t - t0) / D
        if q > -0.1:
            thr = spr["thr"]
            vis = ssv((thr - q) / 0.05 + 0.5)
            edge = np.exp(-((thr - q) / 0.018) ** 2) * a * alpha
            m = m * vis
            emit_text_ash(F, spr, x0, y0, t0, D, val * alpha)
    if m.max() <= 0.002 and edge is None:
        return
    blit(F.L, m, x0, y0, val)
    g = m * glow
    if letter_glow is not None:
        g = g + m * _per_letter(spr, letter_glow)
    if edge is not None:
        g = g + edge * 0.8
        blit(F.L, edge * 0.35, x0, y0, 1.0, mode="add")
    blit(F.G, g, x0, y0, 1.0, mode="add")


def emit_text_ash(F, spr, x0, y0, t0, D, val):
    p = spr["pts"]
    te = t0 + (p["thr"] + 0.1) / 1.2 * D
    dt = F.t - te
    on = dt > 0
    if not on.any():
        return
    dt = dt[on]
    life = p["life"][on]
    b = p["b"][on] * np.exp(-dt / life) * np.clip(dt / 0.05, 0, 1) * ssv(1 - dt / 2.0) * val
    xs = x0 + p["x"][on] + (p["sw"][on] * np.sin(p["fq"][on] * 6.28 * dt + p["ph"][on]) + 10 * dt) * U
    ys = y0 + p["y"][on] - (p["v0"][on] * dt + 0.5 * p["acc"][on] * dt * dt) * U
    splat(F.ash, xs / 2, ys / 2, b * 1.9)


# ============================================================== particles
def splat(buf, xs, ys, vals):
    h, w = buf.shape
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    fx, fy = (xs - x0).astype(np.float32), (ys - y0).astype(np.float32)
    ok = (x0 >= 0) & (y0 >= 0) & (x0 < w - 1) & (y0 < h - 1)
    if not ok.any():
        return
    x0, y0, fx, fy, v = x0[ok], y0[ok], fx[ok], fy[ok], np.asarray(vals, np.float32)[ok]
    idx = y0 * w + x0
    flat = np.zeros(h * w, np.float32)
    flat += np.bincount(idx, v * (1 - fx) * (1 - fy), minlength=h * w).astype(np.float32)
    flat += np.bincount(idx + 1, v * fx * (1 - fy), minlength=h * w).astype(np.float32)
    flat += np.bincount(idx + w, v * (1 - fx) * fy, minlength=h * w).astype(np.float32)
    flat += np.bincount(idx + w + 1, v * fx * fy, minlength=h * w).astype(np.float32)
    buf += flat.reshape(h, w)


def ash_params():
    if "ashp" not in C:
        r = np.random.default_rng(5)
        n = 900
        C["ashp"] = dict(x=r.uniform(0, 1080, n), y=r.uniform(0, 2020, n), vy=r.uniform(16, 60, n),
                         sw=r.uniform(5, 32, n), fq=r.uniform(0.08, 0.35, n), ph=r.uniform(0, 6.28, n),
                         dr=r.uniform(-6, 6, n), b=r.uniform(0, 1, n) ** 2.2 * 0.5 + 0.06,
                         rank=r.uniform(0, 1, n), big=r.uniform(0, 1, n) < 0.12, rad=r.uniform(2.0, 5.5, n),
                         tw=r.uniform(0.3, 1.2, n))
    return C["ashp"]


def ash_density(t):
    return float(np.interp(t, [-3.5, -0.8, 0.2, 1, 6, 9, 13.4, 24, 28, 33, 36.8, 41, 44.6],
                           [0.3, 0.35, 0.7, 1.0, 0.85, 0.5, 0.35, 0.45, 0.6, 0.75, 0.9, 0.7, 0.6]))


def draw_ash(F):
    p = ash_params()
    t = F.t
    dens = ash_density(t)
    on = p["rank"] < dens
    a = np.clip((dens - p["rank"]) / 0.08, 0, 1)
    y = (p["y"] - p["vy"] * t) % 2020 - 50
    x = p["x"] + p["sw"] * np.sin(6.283 * p["fq"] * t + p["ph"]) + p["dr"] * t
    b = p["b"] * a * (0.7 + 0.3 * np.sin(6.283 * p["tw"] * t + p["ph"] * 3))
    small = on & ~p["big"]
    splat(F.ash, x[small] * U / 2, y[small] * U / 2, b[small] * 1.6)
    big = np.nonzero(on & p["big"])[0]
    if len(big):
        lay = np.zeros_like(F.ash)
        for i in big:
            cv2.circle(lay, (int(x[i] * U / 2 * 16), int(y[i] * U / 2 * 16)), int(p["rad"][i] * U / 2 * 16),
                       float(b[i] * 0.35), -1, cv2.LINE_AA, 4)
        F.ash += cv2.GaussianBlur(lay, (0, 0), 1.2 * U)


# ============================================================== fog
def fog_intensity(t):
    return float(np.interp(t, [-3.5, -0.5, 0, 2.0, 6, 9, 13.4, 24, 27, 33, 36, 41, 44.6],
                           [0.35, 0.22, 0.15, 1.0, 0.9, 0.55, 0.45, 0.55, 0.75, 0.8, 0.9, 0.85, 0.85]))


def fog_field(F):
    if "fogtex" not in C:
        C["fogtex"] = (tileable(256, 1.55, 11), tileable(256, 1.8, 12))
    T1, T2 = C["fogtex"]
    gx, gy = lowgrid()
    t = F.t

    def samp(T, sx, sy, ox, oy):
        mx = ((gx * sx + ox) % 256).astype(np.float32)
        my = ((gy * sy + oy) % 256).astype(np.float32)
        return cv2.remap(T, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    n1 = samp(T1, 0.21, 0.30, 7.0 * t, 1.2 * t)
    n2 = samp(T2, 0.43, 0.52, -13.0 * t, -2.2 * t)
    d = 0.62 * n1 + 0.38 * n2
    d = ssv((d - 0.36) / 0.44)
    band = ssv((gy - 1180) / 560)
    f = d * band * fog_intensity(t) + d * 0.10 * fog_intensity(t)
    if F.fog_shock is not None:
        cx, cy, r, s = F.fog_shock
        dist = np.sqrt((gx - cx) ** 2 + ((gy - cy) * 1.15) ** 2)
        w = 70 + 0.3 * r
        ring = np.exp(-((dist - r) / w) ** 2) * (0.45 + 0.9 * d)
        inside = ssv((r - dist) / 220)
        f = f * (1 - 0.6 * s * inside) + ring * s * 0.9
        F.lowG += ring * s * 0.25
    return f


# ============================================================== mark (dagger-cross)
def _mark_shapes():
    """Placeholder dagger-cross, point down. Unit height, centred on x=0."""
    S = []
    gL = [(-0.05, 0.229), (-0.17, 0.233), (-0.205, 0.213), (-0.232, 0.246), (-0.205, 0.279), (-0.17, 0.259), (-0.05, 0.263)]
    guard = np.array(gL + [(-x, y) for x, y in reversed(gL)])
    blade_l = np.array([(0, 0.262), (-0.05, 0.262), (-0.047, 0.45), (-0.041, 0.70), (-0.028, 0.86), (0, 1.0)])
    blade_r = blade_l * (-1, 1)
    S.append((blade_l, 0.68))
    S.append((blade_r, 0.96))
    S.append((np.array([(-0.0035, 0.29), (0.0035, 0.29), (0.0035, 0.88), (0, 0.93), (-0.0035, 0.88)]), 1.0))   # ridge
    S.append((guard, 0.94))
    S.append((np.array([(-0.2, 0.2465), (0.2, 0.2465), (0.2, 0.264), (-0.2, 0.264)]) * (1, 1), 0.74))        # guard underside
    S.append((np.array([(-0.019, 0.092), (0.019, 0.092), (0.019, 0.229), (-0.019, 0.229)]), 0.6))             # grip
    for y in (0.115, 0.145, 0.175, 0.205):
        S.append((np.array([(-0.019, y), (0.019, y - 0.012), (0.019, y - 0.006), (-0.019, y + 0.006)]), 0.34))
    S.append((np.array([(-0.03, 0.082), (0.03, 0.082), (0.03, 0.094), (-0.03, 0.094)]), 0.86))               # collar
    S.append((np.array([(0, 0.0), (0.043, 0.042), (0, 0.084), (-0.043, 0.042)]), 0.95))                        # pommel
    S.append((np.array([(0, 0.0), (0, 0.084), (-0.043, 0.042)]), 0.72))
    return S


def mark_sprite():
    if "mark" in C:
        return C["mark"]
    hs = int(800 * U)
    L_ = logo()
    icon = os.path.join(ASSETS, "acheron-icon.png")
    if os.path.exists(icon):
        a = _vec_file(icon, hs)
        v = a * 0.93
    elif L_ is not None:
        a = L_["vec"](L_["mark_ids"], L_["mark_box"], hs / L_["mark_box"][3])
        v = a * 0.93
    else:
        ss_ = 2
        Hh = hs * ss_
        Ww = int(Hh * 0.52)
        al = np.zeros((Hh, Ww), np.uint8)
        va = np.zeros((Hh, Ww), np.uint8)
        for poly, val in _mark_shapes():
            p = np.round((poly * Hh + (Ww / 2, 0)) * 16).astype(np.int32)
            cv2.fillPoly(al, [p], 255, cv2.LINE_AA, 4)
            cv2.fillPoly(va, [p], int(val * 255), cv2.LINE_AA, 4)
        a = cv2.resize(al, (Ww // ss_, Hh // ss_), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        v = cv2.resize(va, (Ww // ss_, Hh // ss_), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    C["mark"] = (a, v)    # v is premultiplied (value * coverage)
    return C["mark"]


def _vec_file(path, height_px, k=8):
    """Icon image (light on dark) -> tight, vector-sharp coverage mask of given height."""
    g = np.asarray(Image.open(path).convert("L"), np.float32) / 255
    bg = np.median(g)
    ink = np.percentile(g[g > bg + 0.4], 90)
    n_ = np.clip((g - bg) / (ink - bg), 0, 1)
    num, lab, st, _ = cv2.connectedComponentsWithStats((n_ > 0.45).astype(np.uint8), 8)
    keep = [i for i in range(1, num) if st[i][4] > 40]
    sel = cv2.dilate(np.isin(lab, keep).astype(np.uint8), np.ones((5, 5), np.uint8))
    ys, xs = np.nonzero(sel)
    n_ = (n_ * sel)[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    up = cv2.GaussianBlur(cv2.resize(n_, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC), (0, 0), k * 0.22)
    m = np.clip((up - 0.45) / 0.08 + 0.5, 0, 1)
    h, w = n_.shape
    return np.ascontiguousarray(cv2.resize(m, (round(w * height_px / h), height_px), interpolation=cv2.INTER_AREA).astype(np.float32))


# ============================================================== the real lockup
LOCK_W = 880              # design width of the full horizontal lockup
LOCK_C = (540, 900)       # design centre of the lockup when it first resolves
LOCK_RISE = 140           # how far it lifts in scene 6


def logo():
    """Split assets/acheron-logo.png into mark, letters and tagline row.

    Each part is cut by connected components, then upscaled through a smooth
    threshold so the low-res source edges come out vector-sharp at 4K.
    """
    if "logo" in C:
        return C["logo"]
    path = os.path.join(ASSETS, "acheron-logo.png")
    if not os.path.exists(path):
        C["logo"] = None
        return None
    arr = np.asarray(Image.open(path).convert("RGBA"), np.float32) / 255
    g = arr[..., :3].mean(-1) * arr[..., 3]
    if np.median(g) > 0.5:                  # dark logo on a light background
        g = 1 - g
    n, lab, st, _ = cv2.connectedComponentsWithStats((g > 0.35).astype(np.uint8), 8)
    comps = [i for i in range(1, n) if st[i][4] > 12]
    mark = max(comps, key=lambda i: st[i][3])
    mx, my, mw, mh = st[mark][:4]
    row2 = [i for i in comps if i != mark and st[i][1] > my + 0.72 * mh]
    rest = [i for i in comps if i != mark and i not in row2]
    med = np.median([st[i][4] for i in rest])
    big = sorted([i for i in rest if st[i][4] > 0.2 * med], key=lambda i: st[i][0])
    letters = [[i] for i in big]
    for i in rest:                           # specks join the nearest letter
        if i not in big:
            cx_ = st[i][0] + st[i][2] / 2
            k = int(np.argmin([abs(st[b][0] + st[b][2] / 2 - cx_) for b in big]))
            letters[k].append(i)

    def box(ids, pad=3):
        x0 = min(st[i][0] for i in ids) - pad
        y0 = min(st[i][1] for i in ids) - pad
        x1 = max(st[i][0] + st[i][2] for i in ids) + pad
        y1 = max(st[i][1] + st[i][3] for i in ids) + pad
        return (x0, y0, x1 - x0, y1 - y0)

    def vec(ids, b, scale, k=8):
        x, y, w, h = b
        sel = np.isin(lab[y:y + h, x:x + w], ids).astype(np.uint8)
        sel = cv2.dilate(sel, np.ones((5, 5), np.uint8))
        crop = g[y:y + h, x:x + w] * sel
        up = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        up = cv2.GaussianBlur(up, (0, 0), k * 0.3)
        m = np.clip((up - 0.42) / 0.08 + 0.5, 0, 1)
        return np.ascontiguousarray(cv2.resize(m, (max(1, round(w * scale)), max(1, round(h * scale))),
                                               interpolation=cv2.INTER_AREA).astype(np.float32))

    allids = comps
    lb = box(allids, 0)
    S = LOCK_W / lb[2]
    lcx, lcy = lb[0] + lb[2] / 2, lb[1] + lb[3] / 2

    def off(b):
        return ((b[0] + b[2] / 2 - lcx) * S, (b[1] + b[3] / 2 - lcy) * S)

    word_ids = [i for grp in letters for i in grp]
    out = dict(vec=vec, S=S, mark_ids=[mark], mark_box=box([mark]), word_ids=word_ids,
               word_box=box(word_ids), letters=letters, tag_ids=row2, tag_box=box(row2) if row2 else None)
    out["mark_off"], out["mark_h"] = off(out["mark_box"]), out["mark_box"][3] * S
    out["word_off"] = off(out["word_box"])
    out["tag_off"] = off(out["tag_box"]) if row2 else (0, 0)
    C["logo"] = out
    return out


def layout():
    """Where the mark, wordmark and tagline sit once the lockup resolves (design units)."""
    L_ = logo()
    if L_ is None:     # stacked placeholder lockup
        return dict(mark=(540, 640, 560), word=(540, 1135), tag=(540, 1268), rise=170, invite=1300)
    cx, cy = LOCK_C
    m = (cx + L_["mark_off"][0], cy + L_["mark_off"][1], L_["mark_h"])
    return dict(mark=m, word=(cx + L_["word_off"][0], cy + L_["word_off"][1]),
                tag=(cx + L_["tag_off"][0], cy + L_["tag_off"][1]), rise=LOCK_RISE,
                invite=cy + L_["tag_off"][1] - LOCK_RISE + 125)


def _logo_sprite(key, ids_groups, b):
    """Sprite dict compatible with draw_text, with a per-pixel letter index."""
    L_ = logo()
    sc = L_["S"] * U
    a = L_["vec"]([i for gp in ids_groups for i in gp], b, sc)
    lab = np.zeros(a.shape, np.int16)
    if len(ids_groups) > 1:
        stack = np.stack([L_["vec"](gp, b, sc) for gp in ids_groups])
        lab = stack.argmax(0).astype(np.int16)
    spr = dict(a=a, lab=lab, col=np.zeros(a.shape[1], np.int16), cx=a.shape[1] / 2, cy=a.shape[0] / 2,
               n=len(ids_groups), text=key)
    C[key] = spr
    return spr


def draw_mark(F, cx, cy, height, ang=0.0, val=1.0, glow=0.22, clip_below=None, reflect=None):
    """Composite the mark centred at design (cx, cy) with given height and angle (degrees).

    clip_below: design y; anything below is under water and hidden.
    reflect: (surface_y, strength) draws a wavy mirrored copy below the surface instead.
    """
    a, v = mark_sprite()
    hs, ws = a.shape
    sc = height * U / hs
    if reflect is not None:
        sy, k = reflect
        a, v = a[::-1], v[::-1]
        cy = 2 * sy - cy
        ang = -ang
    M = cv2.getRotationMatrix2D((ws / 2, hs / 2), ang, sc)
    M[0, 2] += cx * U - ws / 2
    M[1, 2] += cy * U - hs / 2
    corners = np.array([[0, 0, 1], [ws, 0, 1], [0, hs, 1], [ws, hs, 1]], np.float64) @ M.T
    bx0, by0 = int(max(0, corners[:, 0].min() - 2)), int(max(0, corners[:, 1].min() - 2))
    bx1, by1 = int(min(W, corners[:, 0].max() + 3)), int(min(H, corners[:, 1].max() + 3))
    if bx1 <= bx0 or by1 <= by0:
        return
    M2 = M.copy()
    M2[0, 2] -= bx0
    M2[1, 2] -= by0
    bw, bh = bx1 - bx0, by1 - by0
    A = cv2.warpAffine(a, M2, (bw, bh), flags=cv2.INTER_LINEAR, borderValue=0)
    V = cv2.warpAffine(v, M2, (bw, bh), flags=cv2.INTER_LINEAR, borderValue=0)
    ys = (np.arange(by0, by1, dtype=np.float32) / U)[:, None]
    if clip_below is not None:
        keep = ssv((clip_below - ys) / 1.5 + 0.5)
        A *= keep
        V *= keep
    if reflect is not None:
        sy, k = reflect
        t = F.t
        yy, xx = np.mgrid[0:bh, 0:bw].astype(np.float32)
        depth = (yy + by0) / U - sy
        off = (np.sin(depth * 0.09 + t * 2.3) * 3.0 + np.sin(depth * 0.23 - t * 3.1) * 1.6) * U * np.clip(depth / 60, 0.2, 1.6)
        A = cv2.remap(A, xx + off, yy, cv2.INTER_LINEAR, borderValue=0)
        V = cv2.remap(V, xx + off, yy, cv2.INTER_LINEAR, borderValue=0)
        fade = np.exp(-np.clip(depth, 0, None) / 520) * ssv(depth / 4)
        # broken into strips like light on moving water
        strips = 0.55 + 0.45 * np.sin(depth * 0.55 + np.sin(xx / U * 0.02 + t) * 2)
        A *= fade * k
        V *= fade * k * strips
        F.L[by0:by1, bx0:bx1] += V
        F.G[by0:by1, bx0:bx1] += V * 0.15
        return
    reg = F.L[by0:by1, bx0:bx1]
    reg *= 1 - A
    reg += V * val
    F.G[by0:by1, bx0:bx1] += V * glow


# ============================================================== wordmark
def wordmark():
    if "wm" in C:
        return C["wm"]
    L_ = logo()
    if L_ is not None:
        return _logo_sprite("wm", L_["letters"], L_["word_box"])
    C["wm"] = text_sprite("ACHERON", "Cinzel-SemiBold.ttf", 128, 0.22)
    return C["wm"]


def tagline():
    L_ = logo()
    if L_ is not None and L_["tag_ids"]:
        return C["tag"] if "tag" in C else _logo_sprite("tag", [L_["tag_ids"]], L_["tag_box"])
    return text_sprite("AHEAD OF TIME", "Inter-Medium.ttf", 33, 0.62)


# ============================================================== scene 1 — the void
def scene1(F):
    t = F.t
    lines = [("THE RIVER ASKS", 905), ("NOTHING OF YOU", 980)]
    k = 0
    for li, (txt, y) in enumerate(lines):
        spr = text_sprite(txt, "Inter-Regular.ttf", 34, 0.42)
        la = np.zeros(len(txt), np.float32)
        lg = np.zeros(len(txt), np.float32)
        for i, ch in enumerate(txt):
            if ch == " ":
                continue
            ts = 0.55 + k * 0.085
            p = lin(t, ts, ts + 0.6)
            la[i] = ss(p)
            lg[i] = 0.5 * math.sin(math.pi * p)
            k += 1
        draw_text(F, spr, 540, y, 0.68, letter_alpha=la, letter_glow=lg, glow=0.18,
                  dissolve=(4.25 + li * 0.18, 1.35))


# ============================================================== scene 2 — the heartbeat
def vline(F, x, y_end, inten, width=1.6, red=0.0):
    if inten <= 0.002:
        return
    xc = x * U
    hw = int(14 * U)
    x0, x1 = max(0, int(xc) - hw), min(W, int(xc) + hw + 1)
    if x1 <= x0:
        return
    xs = np.arange(x0, x1, dtype=np.float32)
    prof = np.exp(-((xs - xc) / (width * U * 0.5)) ** 2) + 0.18 * np.exp(-((xs - xc) / (width * U * 3)) ** 2)
    ys = np.arange(H, dtype=np.float32) / U
    vert = (0.45 + 0.55 * np.exp(-((ys - 960) / 760) ** 2)) * ssv((y_end - ys) / 20 + 0.5)
    band = vert[:, None] * prof[None, :] * inten
    F.L[:, x0:x1] += band
    F.G[:, x0:x1] += band * 1.6
    if red > 0:
        gx, gy = lowgrid()
        F.lowR += np.exp(-((gx - x) / 34) ** 2) * (0.5 + 0.5 * np.exp(-((gy - 960) / 700) ** 2)) * ssv((y_end - gy) / 20 + 0.5) * red


def draw_rack(p, rim, beat):
    p.line((30, 1500), (1050, 1500), 0.17 * rim, 1.6)
    for x0, x1 in [(250, 290), (790, 830)]:
        p.line((x0, 620), (x0, 1500), 0.20 * rim, 1.6)
        p.line((x1, 620), (x1, 1500), 0.78 * rim, 2.2)
        for y in range(690, 1470, 46):
            p.circle(((x0 + x1) / 2, y), 4.2, 0.20 * rim, 1.2)
    p.line((250, 620), (830, 620), 0.92 * rim, 2.4)
    p.line((290, 652), (790, 652), 0.32 * rim, 1.5)
    for xh, d in [(290, 1), (790, -1)]:
        p.poly([(xh, 1012), (xh + 26 * d, 1012), (xh + 26 * d, 994)], 0.55 * rim, 2.0)
    p.line((30, 993), (1050, 993), 0.96 * rim, 2.2)
    p.line((30, 1007), (1050, 1007), 0.28 * rim, 1.6)
    for side in (-1, 1):
        x = 540 + side * 325
        p.fill(rrect(min(x, x + side * 14), 968, max(x, x + side * 14), 1032, 3), 0)
        p.poly(rrect(min(x, x + side * 14), 968, max(x, x + side * 14), 1032, 3), 0.65 * rim, 1.6, True)
        x += side * 16
        for hh, wd in [(226, 34), (226, 34), (226, 34), (152, 26)]:
            xa, xb = sorted((x, x + side * wd))
            pts = rrect(xa, 1000 - hh, xb, 1000 + hh, 9)
            p.fill(pts, 0)
            p.poly(pts, 0.24 * rim, 1.5, True)
            p.line((xa + 6, 1000 - hh), (xb - 6, 1000 - hh), 0.98 * rim, 2.4)
            outer = xb if side > 0 else xa
            p.line((outer, 1000 - hh + 9), (outer, 1000 + hh - 9), (0.72 if side > 0 else 0.34) * rim, 1.8)
            x += side * (wd + 2)
        p.line((x, 993), (x + side * 70, 993), 0.9 * rim, 2.2)
        p.line((x, 1007), (x + side * 70, 1007), 0.3 * rim, 1.6)


def scene2(F):
    t = F.t
    beat = heart_env(t)
    red_env = sum(decay(t, b, 0.3) for b in TL.HEARTBEATS)
    op = eio(lin(t, TL.RACK_OPEN, 10.3))
    w = 620 * op
    # light line, then the opening doors of light
    if t < 10.9:
        grow = eout(lin(t, TL.LINE_CUT, 6.45))
        inten = (0.7 + 0.75 * beat) * (1 - ss(lin(t, 9.7, 10.9)))
        red = 0.2 * min(red_env, 1.0) * (1 - ss(lin(t, 9.7, 10.6)))
        if w < 2:
            vline(F, 540, 1920 * grow, inten, 1.7, red)
        else:
            for xo in (-w, w):
                vline(F, 540 + xo, 1920, inten * (1 - 0.5 * op), 1.7 + 1.2 * op, red * 0.5)
    if t >= TL.RACK_OPEN:
        words_on = ss(lin(t, 10.3, 10.45))
        rim = ss(lin(t, 9.0, 10.2)) * (1 - 0.5 * words_on) * (1 + 0.18 * min(beat, 1)) * (1 - ss(lin(t, 12.7, 13.3)))
        if rim > 0.003:
            lay = new_layer()
            pen = Pen(lay, 540, 1000, 1.0 + 0.07 * lin(t, 9.0, 13.3))
            draw_rack(pen, 1.0, beat)
            xs = (np.arange(W, dtype=np.float32) / U)
            mask = ssv((w - np.abs(xs - 540)) / 30 + 0.5) if op < 1 else None
            F.add_layer(lay, rim, glow=0.35, colmask=mask)
    for i, (tw, word) in enumerate(TL.WORDS):
        if t < tw - 0.03:
            continue
        nxt = TL.WORDS[i + 1][0] if i + 1 < len(TL.WORDS) else None
        a = eout(lin(t, tw - 0.03, tw + 0.15))
        if nxt is not None:
            a *= 1 - ss(lin(t, nxt - 0.08, nxt + 0.06))
        if a <= 0.003:
            continue
        spr = text_sprite(word, "Inter-SemiBold.ttf", 128, 0.10)
        g = 0.22 + 0.9 * decay(t, tw, 0.22)
        draw_text(F, spr, 540, 1330, 0.95, alpha=a, glow=g, dissolve=(12.78, 0.6) if nxt is None else None)


# ============================================================== scene 3 — the quiet work
def shot_cam(tau, cx, cy, s):
    sh = math.exp(-tau / 0.11)
    dx = 7 * sh * math.sin(6.283 * 23 * tau)
    dy = 5 * sh * math.cos(6.283 * 29 * tau)
    return cx, cy, s, dx, dy


def shot_window(F, tau):
    cx, cy, s, dx, dy = shot_cam(tau, 540, 900, 1.0 + 0.045 * tau / 2.1)
    gx, gy = lowgrid()
    X = (gx - cx - dx) / s + cx
    Y = (gy - cy - dy) / s + cy

    def win(x, y):
        return ssv((x - 262) / 6 + 0.5) * ssv((818 - x) / 6 + 0.5) * ssv((y - 362) / 6 + 0.5) * ssv((1158 - y) / 6 + 0.5)

    if "glass" not in C:
        C["glass"] = tileable(128, 1.4, 31)
    gl = C["glass"]
    tex = cv2.resize(gl, (FW, FH))
    pane = win(X, Y) * (0.27 - 0.15 * np.clip((Y - 360) / 800, 0, 1)) * (0.55 + 0.7 * tex)
    d = np.array([-0.40, 1.0]) / math.hypot(0.40, 1.0)
    shaft = np.zeros_like(X)
    for k in range(40):
        lam = (k + 0.5) * 24.0
        shaft += win(X - lam * d[0], Y - lam * d[1]) * math.exp(-lam / 650) * 0.31
    shaft = cv2.GaussianBlur(shaft, (0, 0), 2.2)
    floor = ssv((Y - 1480) / 6 + 0.5)
    shaft *= 0.05 * (1 + floor * (1.4 * np.exp(-np.clip(Y - 1480, 0, None) / 160) - 0.55))
    field = up((pane + shaft) * 1.0)

    lay = new_layer()
    p = Pen(lay, cx, cy, s, dx, dy)
    occ = new_layer()
    po = Pen(occ, cx, cy, s, dx, dy)
    # window frame
    p.poly(rrect(255, 355, 825, 1165, 2), 0.55, 7, True)
    for x in (445, 635):
        p.line((x, 358), (x, 1162), 0.45, 4)
    for y in (557, 760, 962):
        p.line((258, y), (822, y), 0.45, 4)
    p.line((232, 1176), (848, 1176), 0.5, 5)
    p.line((0, 1480), (1080, 1480), 0.16, 1.5)
    # silhouettes in front of the glass
    for x0, x1 in [(352, 374), (706, 728)]:
        po.fill([(x0, 820), (x1, 820), (x1, 1480), (x0, 1480)], 1.0)
        p.line((x1, 822), (x1, 1478), 0.26, 1.5)
    po.fill([(250, 1039), (830, 1039), (830, 1051), (250, 1051)], 1.0)
    p.line((250, 1038), (830, 1038), 0.42, 1.6)
    for xa, xb in [(286, 314), (316, 344), (736, 764), (766, 794)]:
        po.fill(rrect(xa, 1045 - 172, xb, 1045 + 172, 7), 1.0)
        p.line((xa + 4, 1045 - 172), (xb - 4, 1045 - 172), 0.5, 1.8)
    o = occ.astype(np.float32) * (1 / 255)
    F.L += field * (1 - o)
    F.G += field * (1 - o) * 0.5
    lay[occ > 128] = np.minimum(lay[occ > 128], 90)
    F.add_layer(lay, 1.0, glow=0.3)
    # dust in the light
    if "dust" not in C:
        r = np.random.default_rng(41)
        C["dust"] = (r.uniform(150, 900, 220), r.uniform(380, 1500, 220), r.uniform(0.2, 1, 220), r.uniform(0, 6.28, 220))
    x, y, b, ph = C["dust"]
    xx = x + 14 * np.sin(0.4 * (tau + 13) + ph) - 6 * tau
    yy = y + 10 * np.sin(0.3 * (tau + 13) + ph * 2) + 4 * tau
    li = np.clip(((xx - 262) > 0) * 0.3 + 0.7, 0, 1)
    X2, Y2 = (xx - cx) * s + cx + dx, (yy - cy) * s + cy + dy
    splat(F.ash, X2 * U / 2, Y2 * U / 2, b * li * 0.55)


def chalk_params():
    if "chalk" not in C:
        r = np.random.default_rng(77)
        n = 3200
        ang = r.normal(-math.pi / 2, 0.95, n)
        sp = r.lognormal(5.6, 0.55, n)
        C["chalk"] = dict(x=r.normal(0, 60, n), y=r.normal(0, 8, n), vx=np.cos(ang) * sp, vy=np.sin(ang) * sp,
                          k=r.uniform(1.6, 3.2, n), b=r.uniform(0.2, 1.0, n) ** 1.5, life=r.uniform(1.0, 2.6, n),
                          ph=r.uniform(0, 6.28, n))
    return C["chalk"]


def shot_chalk(F, tau):
    cx, cy, s, dx, dy = shot_cam(tau, 540, 900, 1.07 - 0.05 * eout(tau / 2.1))
    lay = new_layer()
    p = Pen(lay, cx, cy, s, dx, dy)
    a = math.radians(-6)
    u = np.array([math.cos(a), math.sin(a)])
    n = np.array([-u[1], u[0]])
    c = np.array([540, 900])
    p.line(c - u * 700 - n * 13, c + u * 700 - n * 13, 0.9, 2.4)
    p.line(c - u * 700 + n * 13, c + u * 700 + n * 13, 0.28, 1.6)
    for z0, z1 in [(-470, -70), (70, 470)]:
        for sgn in (-1, 1):
            for sv in np.arange(z0, z1, 10):
                p.line(c + u * sv - n * 11 * sgn, c + u * (sv + 10) + n * 11 * sgn, 0.3, 1.0)
    F.add_layer(lay, 1.0, glow=0.3)
    # chalk burst, slow-motion drag
    q = chalk_params()
    e = (1 - np.exp(-q["k"] * tau)) / q["k"]
    x = c[0] + q["x"] + q["vx"] * e + 10 * np.sin(1.3 * tau + q["ph"]) * tau
    y = c[1] + q["y"] + q["vy"] * e + 14 * tau * tau
    b = q["b"] * np.exp(-tau / q["life"]) * 0.9
    X, Y = (x - cx) * s + cx + dx, (y - cy) * s + cy + dy
    splat(F.ash, X * U / 2, Y * U / 2, b * 1.4)
    # the cloud itself
    gx, gy = lowgrid()
    cloud = np.zeros((FH, FW), np.float32)
    sel = slice(0, 400)
    splat(cloud, X[sel] * FW / 1080, Y[sel] * FH / 1920, b[sel] * 0.6)
    cloud = cv2.GaussianBlur(cloud, (0, 0), 7)
    F.lowL += cloud * 0.9
    F.lowG += cloud * 0.4


def plate_draw(p, xf, cy, ry, rim, ghost=1.0):
    rx = ry * 0.22
    th = 44
    outer_f = ellipse_pts(xf, cy, rx, ry)
    outer_b = ellipse_pts(xf - th, cy, rx, ry)
    body = np.concatenate([ellipse_pts(xf - th, cy, rx, ry, 90, 270, 60), ellipse_pts(xf, cy, rx, ry, 270, 450, 60)])
    p.fill(body, 0)
    p.fill(outer_f, 0)
    p.poly(ellipse_pts(xf - th, cy, rx, ry, 90, 270, 60), 0.22 * rim * ghost, 1.5)
    p.line((xf - th, cy - ry), (xf, cy - ry), 0.95 * rim * ghost, 2.2)
    p.line((xf - th, cy + ry), (xf, cy + ry), 0.18 * rim * ghost, 1.4)
    p.poly(ellipse_pts(xf, cy, rx, ry, 180, 360, 70), 0.9 * rim * ghost, 2.2)
    p.poly(ellipse_pts(xf, cy, rx, ry, 0, 180, 70), 0.36 * rim * ghost, 1.6)
    p.poly(ellipse_pts(xf, cy, rx * 0.88, ry * 0.9, 190, 350, 50), 0.4 * rim * ghost, 1.3)
    p.poly(ellipse_pts(xf, cy, rx * 0.36, ry * 0.36), 0.42 * rim * ghost, 1.4, True)
    p.poly(ellipse_pts(xf, cy, 8, 36), 0.7 * rim * ghost, 1.6, True)


def shot_plate(F, tau):
    cx, cy, s, dx, dy = shot_cam(tau, 540, 1000, 1.0 + 0.04 * tau / 2.1)
    lay = new_layer()
    p = Pen(lay, cx, cy, s, dx, dy)
    cyb = 1000
    p.line((-60, cyb - 32), (820, cyb - 32), 0.75, 2.0)
    p.line((-60, cyb + 32), (820, cyb + 32), 0.3, 1.6)
    p.poly(ellipse_pts(820, cyb, 7, 32), 0.6, 1.6, True)
    p.fill(ellipse_pts(120, cyb, 20, 70), 0)
    p.poly(ellipse_pts(120, cyb, 20, 70), 0.55, 1.8, True)
    for xf in (214, 262):
        plate_draw(p, xf, cyb, 330, 0.85)
    vib = 3.5 * math.exp(-tau / 0.15) * math.sin(6.283 * 38 * tau)
    xf = 310 + vib
    for gi, off in enumerate((260, 170, 95, 40)):
        g = math.exp(-tau / 0.05) * (0.12 + 0.1 * gi)
        if g > 0.01:
            plate_draw(p, xf + off, cyb, 330, 0.85, ghost=g)
    plate_draw(p, xf, cyb, 330, 1.0)
    F.add_layer(lay, 1.0, glow=0.32)
    # dust knocked off at the contact rim
    if "pdust" not in C:
        r = np.random.default_rng(91)
        n = 700
        a = r.uniform(0, 6.283, n)
        C["pdust"] = (a, r.lognormal(4.6, 0.5, n), r.uniform(0.2, 1, n), r.uniform(1.5, 3.5, n))
    a, sp, b, k = C["pdust"]
    e = (1 - np.exp(-k * tau)) / k
    x = 290 + 72 * np.cos(a) + np.cos(a) * sp * e * 0.35 + 30 * e
    y = cyb + 330 * np.sin(a) + np.sin(a) * sp * e + 20 * tau * tau
    X, Y = (x - cx) * s + cx + dx, (y - cy) * s + cy + dy
    splat(F.ash, X * U / 2, Y * U / 2, b * np.exp(-tau / 0.7) * 0.7)


def shot_bend(F, tau):
    cx, cy, s, dx, dy = shot_cam(tau, 540, 900, 1.0 + 0.035 * tau / 2.0)
    lay = new_layer()
    p = Pen(lay, cx, cy, s, dx, dy)
    yc = 960 - 70 * eio(lin(tau, 0.1, 2.0))
    A = 34 + 18 * math.sin(6.283 * 1.7 * tau) * math.exp(-1.3 * tau)
    trem = 1.2 * math.sin(6.283 * 11 * tau) + 0.8 * math.sin(6.283 * 17 * tau + 1)

    def dy_(x):
        return A * ((x - 540) / 500.0) ** 2 + trem * ((x - 540) / 500.0) ** 2

    def slope(x):
        return 2 * A * (x - 540) / 500.0 ** 2

    xs = np.linspace(20, 1060, 80)
    p.poly(np.stack([xs, yc - 9 + dy_(xs)], 1), 0.95, 2.2)
    p.poly(np.stack([xs, yc + 9 + dy_(xs)], 1), 0.3, 1.6)
    p.line((0, 1500), (1080, 1500), 0.16, 1.5)
    for side in (-1, 1):
        x = 540 + side * 322
        for hh, wd in [(0, 14), (220, 32), (220, 32), (220, 32), (220, 32), (150, 26)]:
            xm = x + side * wd / 2
            ang = math.atan(slope(xm))
            yy = yc + dy_(xm)
            h2 = hh if hh else 30
            xa, xb = xm - wd / 2, xm + wd / 2
            pts = rotate_pts(rrect(xa, yy - h2, xb, yy + h2, 8 if hh else 3), xm, yy, ang)
            p.fill(pts, 0)
            p.poly(pts, 0.24, 1.5, True)
            top = rotate_pts(np.array([(xa + 5, yy - h2), (xb - 5, yy - h2)]), xm, yy, ang)
            p.line(top[0], top[1], 0.98, 2.4)
            outer = xb if side > 0 else xa
            edge = rotate_pts(np.array([(outer, yy - h2 + 8), (outer, yy + h2 - 8)]), xm, yy, ang)
            p.line(edge[0], edge[1], 0.7 if side > 0 else 0.32, 1.8)
            x += side * (wd + 2)
    F.add_layer(lay, 1.0, glow=0.32)


def water_params(seed, n):
    key = ("water", seed, n)
    if key not in C:
        r = np.random.default_rng(seed)
        s = r.uniform(0, 1, n)
        glade = r.uniform(0, 1, n) < 0.62
        C[key] = dict(s=s, glade=glade, g=r.normal(0, 1, n), xa=r.uniform(0, 1080, n), len=r.uniform(0.5, 1.5, n),
                      w=r.uniform(0.6, 1.7, n), ph=r.uniform(0, 6.28, n), w2=r.uniform(0.8, 2.6, n), ph2=r.uniform(0, 6.28, n),
                      dy=r.uniform(-1, 1, n))
    return C[key]


def draw_water(p, t, horizon, bottom, glade_x=540, glade_w0=30, glade_w1=300, gain=1.0, amb=0.11, seed=3, n=1300):
    q = water_params(seed, n)
    s = q["s"]
    y = horizon + (bottom - horizon) * s ** 1.7 + q["dy"] * 3 * s
    gw = glade_w0 + glade_w1 * s
    x = np.where(q["glade"], glade_x + q["g"] * gw * 0.8, q["xa"])
    x = x + (3 + 14 * s) * np.sin(q["w"] * t + q["ph"])
    ln = (6 + 95 * s ** 1.3) * q["len"]
    tw = np.clip(0.35 + 0.65 * np.sin(q["w2"] * t + q["ph2"]), 0, 1) ** 1.5
    bright = np.where(q["glade"], 0.62 * np.exp(-((x - glade_x) / gw) ** 2) * (0.45 + 0.55 * (1 - s)), amb * (0.5 + 0.5 * s))
    bright = bright * (0.25 + 0.75 * tw) * gain
    for i in np.nonzero(bright > 0.012)[0]:
        p.line((x[i] - ln[i] / 2, y[i]), (x[i] + ln[i] / 2, y[i]), float(bright[i]), 0.7 + 1.8 * s[i])


def moon_sprite(r):
    key = ("moon", r)
    if key not in C:
        R = int(r * U)
        size = 2 * R + 8
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        d = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / R
        disc = ssv((1 - d) * R / 1.5 + 0.5)
        tex = cv2.resize(tileable(64, 1.7, 51), (size, size), interpolation=cv2.INTER_CUBIC)
        limb = 0.78 + 0.22 * np.sqrt(np.clip(1 - d ** 2, 0, 1))
        val = (0.46 + 0.26 * tex) * limb
        C[key] = (disc, disc * val)
    return C[key]


def figure_polys(phase):
    sp = math.sin(phase)
    bob = -0.008 * abs(sp)
    lift_l, lift_r = 0.045 * max(0, sp), 0.045 * max(0, -sp)
    P = []
    P.append(ellipse_pts(0, -0.915 + bob, 0.074, 0.086, n=40))
    P.append(np.array([(-0.05, -0.85), (-0.135, -0.80), (-0.152, -0.70), (-0.138, -0.47), (-0.128, -0.43),
                       (0.128, -0.43), (0.138, -0.47), (0.152, -0.70), (0.135, -0.80), (0.05, -0.85)]) + (0, bob))
    for sd, sw in ((-1, sp), (1, -sp)):
        P.append(np.array([(sd * 0.125, -0.79), (sd * 0.165, -0.76), (sd * 0.175 + 0.015 * sw, -0.46),
                           (sd * 0.14 + 0.015 * sw, -0.45), (sd * 0.12, -0.70)]) + (0, bob))
    for sd, lift in ((-1, lift_l), (1, lift_r)):
        P.append(np.array([(sd * 0.115, -0.45), (sd * 0.012, -0.45), (sd * 0.03, -lift - 0.02),
                           (sd * 0.035, -lift), (sd * 0.095, -lift), (sd * 0.098, -lift - 0.03)]) + (0, bob))
    return P


def shot_crossing(F, tau, alpha=1.0):
    cx, cy, s, dx, dy = shot_cam(tau, 540, 960, 1.0 + 0.03 * tau / 2.3)
    t = F.t
    lay = new_layer()
    p = Pen(lay, cx, cy, s, dx, dy)
    # sky glow and moon
    gx, gy = lowgrid()
    mx, my = 540, 520
    X = (gx - cx - dx) / s + cx
    Y = (gy - cy - dy) / s + cy
    F.lowL += 0.10 * np.exp(-np.sqrt((X - mx) ** 2 + (Y - my) ** 2) / 330) * ssv((1065 - Y) / 6 + 0.5) * alpha
    disc, val = moon_sprite(140)
    X0, Y0 = p.xy(mx, my)
    ms = disc.shape[0]
    blit(F.L, val * alpha, int(X0 - ms / 2), int(Y0 - ms / 2), 1.0, mode="add")
    blit(F.G, val * 0.55 * alpha, int(X0 - ms / 2), int(Y0 - ms / 2), 1.0, mode="add")
    p.line((0, 1062), (1080, 1062), 0.22, 1.4)
    draw_water(p, t, 1066, 1760, glade_x=540, glade_w0=26, glade_w1=200, gain=1.15, seed=3, n=1200)
    # shore, foam
    xs = np.linspace(-20, 1100, 60)
    shore = 1735 + 18 * np.sin(xs / 210 + 0.6) + 8 * np.sin(xs / 95)
    p.fill(np.concatenate([np.stack([xs, shore], 1), [(1100, 2100), (-20, 2100)]]), 0)
    for k in range(3):
        ph = (t * 0.45 + k / 3) % 1
        p.poly(np.stack([xs, shore - 5 - 22 * (1 - ph)], 1), 0.22 * math.sin(math.pi * ph), 1.2)
    p.poly(np.stack([xs, shore], 1), 0.3, 1.5)
    # lone figure walking to the edge, back to us
    walk = lin(tau, 0, 2.6)
    fh = 430 - 50 * walk
    fx, fy = 548, 1905 - 90 * walk
    for poly in figure_polys(6.283 * 0.95 * tau):
        p.fill(poly * fh + (fx, fy), 0)
    hood = ellipse_pts(fx, fy + (-0.915) * fh, 0.074 * fh, 0.086 * fh, 200, 340, 30)
    p.poly(hood, 0.75, 1.8)
    p.poly(np.array([(-0.05, -0.85), (-0.135, -0.80), (-0.152, -0.70), (-0.16, -0.6)]) * fh + (fx, fy), 0.5, 1.5)
    p.poly(np.array([(0.05, -0.85), (0.135, -0.80), (0.152, -0.70), (0.16, -0.6)]) * fh + (fx, fy), 0.5, 1.5)
    F.add_layer(lay, alpha, glow=0.3)


SHOTS = [shot_window, shot_chalk, shot_plate, shot_bend, shot_crossing]


def scene3(F):
    t = F.t
    k = max(i for i, c in enumerate(TL.CLANGS) if t >= c)
    tau = t - TL.CLANGS[k]
    if k == 4:
        shot_crossing(F, tau, alpha=1 - ss(lin(t, 23.9, 24.6)))
    else:
        SHOTS[k](F, tau)
    F.expo = 1 + 0.35 * math.exp(-tau / 0.07)
    cap = TL.CAPTIONS[k]
    if cap:
        nxt = TL.CLANGS[k + 1]
        a = ss(lin(tau, 0.12, 0.45)) * (1 - ss(lin(t, nxt - 0.25, nxt - 0.02)))
        if a > 0.003:
            draw_text(F, text_sprite(cap, "Inter-Medium.ttf", 42, 0.42), 540, 1600, 0.68, alpha=a, glow=0.15)


# ============================================================== scene 4 — the crossing
SURF = 1250
MARK_H = 760


def dagger_state(t):
    """Centre, angle and height of the mark from the water to the final lockup."""
    rise = eout(lin(t, 24.6, 27.3))
    cy = 1700 - (1700 - 780) * rise
    ang = 18 * (1 - eio(lin(t, 26.9, 27.95)))
    lock = TL.BELL
    cy += -2.5 * ss(lin(t, lock - 0.06, lock)) + 2.5 * ss(lin(t, lock, lock + 0.25))
    h = MARK_H
    cx = 540
    # scene 5: settle into its place in the lockup
    lay = layout()
    lx, ly, lh = lay["mark"]
    m = eio(lin(t, 33.2, 34.6))
    cx = cx + (lx - cx) * m
    cy = cy + (ly - cy) * m
    h = h + (lh - h) * m
    cy -= lay["rise"] * eio(lin(t, *TL.LOCKUP_RISE))
    return cx, cy, h, ang


def scene4(F, wa):
    """Water, dagger rising, bell, virtues. wa = water/scene alpha."""
    t = F.t
    lay = new_layer()
    p = Pen(lay, 540, 960, 1.0 + 0.02 * lin(t, 24.0, 33.0))
    F.lowL += 0.05 * np.exp(-((lowgrid()[0] - 540) / 300) ** 2) * ssv((lowgrid()[1] - 640) / 400) * wa
    draw_water(p, t, 640, 1980, glade_x=540, glade_w0=40, glade_w1=330, gain=0.9 * wa, amb=0.1 * wa, seed=8, n=1500)
    cx, cy, h, ang = dagger_state(t)
    # ripples where the blade breaks the surface
    for k, ts in enumerate(np.arange(24.75, 27.4, 0.42)):
        if t >= ts:
            age = t - ts
            r = 30 + 240 * age ** 0.8
            v = 0.5 * math.exp(-age / 1.3) * wa
            if v > 0.01:
                p.poly(ellipse_pts(540, SURF + 8, r, r * 0.16), v, 1.6, True)
    # drips off the point
    tip_y = cy + h / 2
    for k, td in enumerate((27.45, 27.85, 28.6, 29.9)):
        fall = t - td
        if 0 <= fall < 0.6:
            yy = SURF - 70 + 0.5 * 2400 * fall ** 2
            if yy < SURF:
                p.circle((540, yy), 2.6, 0.85 * wa, fill=True)
            else:
                age = fall - math.sqrt(2 * 70 / 2400)
                r = 8 + 140 * age
                p.poly(ellipse_pts(540, SURF + 6, r, r * 0.17), 0.55 * math.exp(-age / 0.5) * wa, 1.4, True)
    F.add_layer(lay, 1.0, glow=0.28)
    if t >= 24.5 and wa > 0.002:
        draw_mark(F, cx, cy, h, ang, reflect=(SURF, 0.30 * wa))
    if 24.5 <= t < 33.4:                      # scene 5 owns the mark from 33.4
        draw_mark(F, cx, cy, h, ang, val=1.0, glow=0.22 + 0.9 * decay(t, TL.BELL, 0.35), clip_below=SURF + 2)
        # flash on lock
        fl = 0.35 * decay(t, TL.BELL, 0.3)
        if fl > 0.005:
            draw_mark(F, cx, cy, h, ang, val=1.0 + fl, glow=fl * 2.0)
    # red shockwave ring on the bell
    if t >= TL.BELL:
        pr = lin(t, TL.BELL, TL.BELL + 1.8)
        if pr < 1:
            gx, gy = lowgrid()
            r = 40 + 1150 * eout(pr)
            d = np.sqrt((gx - cx) ** 2 + ((gy - cy) * 1.0) ** 2)
            wdt = 7 + 26 * pr
            ring = np.exp(-((d - r) / wdt) ** 2) * (1 - pr) ** 1.8
            F.lowR += ring * 0.42
            F.lowG += ring * 0.1
    # PERSISTENCE / CONSISTENCY / LOYALTY
    for i, (tv, word) in enumerate(TL.VIRTUES):
        if t < tv - 0.05:
            continue
        a = eout(lin(t, tv - 0.05, tv + 0.4))
        spr = text_sprite(word, "Inter-Medium.ttf", 46, 0.42)
        draw_text(F, spr, 540, 1440 + i * 84, 0.88, alpha=a, glow=0.18 + 0.6 * decay(t, tv, 0.3),
                  dissolve=(32.55 + i * 0.12, 0.8))


# ============================================================== scene 5/6 — the name, the invitation
def scene5(F):
    t = F.t
    cx, cy, h, ang = dagger_state(t)
    lay = layout()
    oy = -lay["rise"] * eio(lin(t, *TL.LOCKUP_RISE))
    real = logo() is not None
    beat = 0.22 * (decay(t, TL.FINAL_BEAT, 0.25) + 0.6 * decay(t, TL.FINAL_BEAT + TL.DUB_OFFSET, 0.2))
    hit = decay(t, TL.SWELL_HIT, 0.5)
    if t >= 33.4:
        draw_mark(F, cx, cy, h, ang, val=1.0 + beat, glow=0.22 + 0.5 * hit + beat)
    # fog shockwave
    if t >= TL.SHOCKWAVE:
        pr = lin(t, TL.SHOCKWAVE, TL.SHOCKWAVE + 2.8)
        if pr < 1:
            F.fog_shock = (cx, cy, 60 + 1300 * eout(pr), (1 - pr) ** 1.2)
    # wordmark, letter by letter
    wm = wordmark()
    n = wm["n"]
    la = np.zeros(n, np.float32)
    lg = np.zeros(n, np.float32)
    for i in range(n):
        ts = TL.LETTERS_START + i * (TL.SWELL_HIT - TL.LETTER_BURN - TL.LETTERS_START) / max(1, n - 1)
        pp = lin(t, ts, ts + TL.LETTER_BURN)
        la[i] = ss(pp)
        lg[i] = 0.7 * math.sin(math.pi * pp)
    if la.max() > 0:
        base = 0.55 + 0.4 * ss(lin(t, TL.SWELL_HIT - 0.02, TL.SWELL_HIT + 0.12))
        val = base * (1 + beat)
        wx, wy = lay["word"][0], lay["word"][1] + oy
        draw_text(F, wm, wx, wy, val, letter_alpha=la, letter_glow=lg + 1.1 * hit + beat * 2, glow=0.2)
        # light across wet iron
        if TL.SWEEP[0] <= t <= TL.SWEEP[1] + 0.2:
            sweep(F, wm, wx, wy, lin(t, *TL.SWEEP))
    # AHEAD OF TIME
    if t >= TL.TAGLINE_IN:
        a = ss(lin(t, TL.TAGLINE_IN, TL.TAGLINE_IN + 0.9))
        draw_text(F, tagline(), lay["tag"][0], lay["tag"][1] + oy, (0.86 if real else 0.68) * (1 + beat), alpha=a, glow=0.12)
    # the invitation
    for i, (ti, txt) in enumerate(TL.INVITE):
        if t >= ti - 0.05:
            a = ss(lin(t, ti - 0.05, ti + 0.55))
            lgx = 0.5 * math.sin(math.pi * lin(t, ti - 0.05, ti + 0.55))
            draw_text(F, text_sprite(txt, "Inter-Medium.ttf", 36, 0.42), 540, lay["invite"] + i * 76, 0.64,
                      alpha=a, glow=0.12 + lgx)


def sweep(F, spr, cx, cy, pr):
    a = spr["a"]
    h, w = a.shape
    x0 = int(round(cx * U - spr["cx"]))
    y0 = int(round(cy * U - spr["cy"]))
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    xb = -0.25 * w + 1.5 * w * eio(pr)
    band = np.exp(-(((xs - xb) + 0.45 * (ys - h / 2)) / (0.07 * w)) ** 2)
    band2 = np.exp(-(((xs - xb + 0.11 * w) + 0.45 * (ys - h / 2)) / (0.025 * w)) ** 2) * 0.6
    hl = a * (band + band2) * math.sin(math.pi * cl(pr))
    blit(F.L, hl * 0.45, x0, y0, 1.0, mode="add")
    blit(F.G, hl * 1.1, x0, y0, 1.0, mode="add")



# ============================================================== pre-roll — use headphones
def _hp_paths():
    """Line-art headphones as ordered polylines with their draw-on windows (card time)."""
    if "hp" in C:
        return C["hp"]
    P = []
    a = np.radians(np.linspace(180, 360, 90))
    P.append((np.stack([540 + 118 * np.cos(a), 905 + 118 * np.sin(a)], 1), 0.25, 1.05, 0.86, 3.0))
    a = np.radians(np.linspace(198, 342, 70))
    P.append((np.stack([540 + 106 * np.cos(a), 905 + 106 * np.sin(a)], 1), 0.45, 1.15, 0.38, 1.4))
    for cx, d in ((422, 1), (658, -1)):
        P.append((np.array([(cx, 905), (cx, 918)]), 0.9, 1.0, 0.86, 3.0))
        cup = rrect(cx - 24, 918, cx + 24, 1028, 22, n=10)
        k = int(np.argmin(np.abs(cup[:, 0] - cx) + np.abs(cup[:, 1] - 918)))
        cup = np.concatenate([cup[k:], cup[:k + 1]])
        P.append((cup, 0.6, 1.25, 0.86, 3.0))
        cx2 = cx + d * 30
        P.append((rrect(cx2 - 6, 932, cx2 + 6, 1014, 6, n=6), 0.9, 1.35, 0.42, 1.4))
    pts = np.concatenate([p[0] for p in P])
    r = np.random.default_rng(13)
    i = r.choice(len(pts), 900)
    jitter = r.normal(0, 1.2, (900, 2))
    ash = dict(x=pts[i, 0] + jitter[:, 0], y=pts[i, 1] + jitter[:, 1], v0=r.uniform(25, 80, 900),
               acc=r.uniform(10, 50, 900), sw=r.uniform(4, 18, 900), ph=r.uniform(0, 6.28, 900),
               fq=r.uniform(0.3, 1.0, 900), b=r.uniform(0.3, 1.0, 900), life=r.uniform(0.6, 1.3, 900))
    C["hp"] = (P, ash)
    return C["hp"]


def _partial(pts, frac):
    if frac >= 1:
        return pts
    seg = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(1))
    cum = np.concatenate([[0], np.cumsum(seg)])
    L = cum[-1] * frac
    k = int(np.searchsorted(cum, L))
    if k <= 0:
        return pts[:1]
    f = (L - cum[k - 1]) / max(seg[k - 1], 1e-6)
    return np.concatenate([pts[:k], [pts[k - 1] + (pts[k] - pts[k - 1]) * f]])


def headphones(F, tc):
    """3.5 s card: the headphones draw on, sound ripples left then right, then it turns to ash."""
    P, ash = _hp_paths()
    out_t0, out_d = TL.PRE_OUT, 0.6
    lay = new_layer()
    p = Pen(lay, 540, 960, 1.0 + 0.03 * min(tc, TL.PRE) / TL.PRE)
    drawn = tc < out_t0 + out_d + 0.05
    for pts, t0, t1, v, w in (P if drawn else []):
        fr = eio(lin(tc, t0, t1))
        if fr > 0:
            p.poly(_partial(pts, fr), v, w)
    # sound arcs, rippling outward from the ear that is playing
    for tp, side in TL.PRE_PULSES:
        cx = 422 if side < 0 else 658
        a0, a1 = (145, 215) if side < 0 else (-35, 35)
        for j, rad in enumerate((80, 106, 132)):
            b = 0.8 * math.exp(-((tc - tp - 0.06 - j * 0.11) / 0.12) ** 2) * (1 - 0.18 * j)
            if b > 0.01:
                p.poly(ellipse_pts(cx, 973, rad, rad, a0, a1, 40), b, 2.0)
    # dissolve left to right into ash
    q = -0.1 + 1.2 * lin(tc, out_t0, out_t0 + out_d)
    xs = np.arange(W, dtype=np.float32) / U
    xn = (xs - 380) / 320
    mask = ssv((xn - q) / 0.06 + 0.5) if q > -0.1 else None
    if drawn:
        F.add_layer(lay, 1.0, glow=0.55, colmask=mask)
    if q > -0.1:
        xn_p = (ash["x"] - 380) / 320
        te = out_t0 + (xn_p + 0.1) / 1.2 * out_d
        dt = tc - te
        on = dt > 0
        if on.any():
            dt = dt[on]
            b = ash["b"][on] * np.exp(-dt / ash["life"][on]) * np.clip(dt / 0.05, 0, 1) * ssv(1 - dt / 1.6) * 0.8
            x = ash["x"][on] + ash["sw"][on] * np.sin(ash["fq"][on] * 6.28 * dt + ash["ph"][on]) + 10 * dt
            y = ash["y"][on] - (ash["v0"][on] * dt + 0.5 * ash["acc"][on] * dt * dt)
            X, Y = p.xy(x, y)
            splat(F.ash, X / 2, Y / 2, b * 1.3)
    # USE HEADPHONES
    txt = "USE HEADPHONES"
    spr = text_sprite(txt, "Inter-Medium.ttf", 28, 0.6)
    la = np.zeros(len(txt), np.float32)
    lg = np.zeros(len(txt), np.float32)
    k = 0
    for i, ch in enumerate(txt):
        if ch == " ":
            continue
        ts = 0.95 + k * 0.055
        pp = lin(tc, ts, ts + 0.45)
        la[i], lg[i] = ss(pp), 0.45 * math.sin(math.pi * pp)
        k += 1
    draw_text(F, spr, 540, 1140, 0.74, letter_alpha=la, letter_glow=lg, glow=0.15,
              dissolve=(out_t0 + 0.08 - TL.PRE, out_d))



# ============================================================== Reels cut — badge and captions
def reels_badge(F, tf):
    """Small headphones badge up top for the first second, arcs on the stereo heartbeat."""
    # frame one: the heartbeat line flashes down the centre on lub and dub
    beat = sum(decay(tf, tb, 0.22) * a for (tb, _), a in zip(TL.REELS_BEATS, (1.0, 0.65)))
    if beat > 0.004:
        vline(F, 540, 1920, 1.15 * min(beat, 1.2), 1.7, 0.2 * min(beat, 1.0))
    out = 1 - ss(lin(tf, TL.REELS_BADGE_OUT, TL.REELS_BADGE_OUT + 0.4))
    if out <= 0.003:
        return
    P, _ = _hp_paths()
    lay = new_layer()
    p = Pen(lay, 540, 960, 0.42, 0, 345 - 960)
    fr = eio(lin(tf, -0.12, 0.3))
    for pts, _t0, _t1, v, w in P:
        p.poly(_partial(pts, fr), v, w * 2.0)
    for tp, side in TL.REELS_BEATS:
        cx = 422 if side < 0 else 658
        a0, a1 = (145, 215) if side < 0 else (-35, 35)
        for j, rad in enumerate((80, 106, 132)):
            b = 0.85 * math.exp(-((tf - tp - 0.04 - j * 0.08) / 0.09) ** 2) * (1 - 0.18 * j)
            if b > 0.01:
                p.poly(ellipse_pts(cx, 973, rad, rad, a0, a1, 40), b, 4.0)
    F.add_layer(lay, out, glow=0.5)
    draw_text(F, text_sprite("USE HEADPHONES", "Inter-Medium.ttf", 22, 0.5), 540, 410, 0.72,
              alpha=out * ss(lin(tf, 0.05, 0.4)), glow=0.1)


def vo_timing():
    if "vot" not in C:
        path = os.path.join(ROOT, "build", "vo_timing.json")
        if os.path.exists(path):
            with open(path) as fh:
                C["vot"] = json.load(fh)
        else:
            C["vot"] = [[c[0], 0.075 * len(c[1])] for c in TL.VO]
    return C["vot"]


def wrap_caption(text, size=50, track=0.12, max_w=940):
    key = ("wrap", text)
    if key in C:
        return C[key]
    f = ImageFont.truetype(os.path.join(FONTS, "Inter-SemiBold.ttf"), 100)

    def width(s_):
        return (sum(f.getlength(ch) for ch in s_) + track * 100 * (len(s_) - 1)) * size / 100

    words = text.split()
    if width(text) <= max_w:
        rows = [text]
    else:      # most balanced two-line split
        best = min(range(1, len(words)), key=lambda k: max(width(" ".join(words[:k])), width(" ".join(words[k:]))))
        rows = [" ".join(words[:best]), " ".join(words[best:])]
    C[key] = rows
    return rows


def draw_halo(F, spr, cx, cy, strength):
    """Soft dark bed behind a caption so it reads over water, light and fog."""
    m = int(26 * U)
    a = np.pad(spr["a"], m)
    a = cv2.GaussianBlur(cv2.dilate(a, np.ones((int(5 * U) | 1,) * 2, np.uint8)), (0, 0), 11 * U)
    x0 = int(round(cx * U - spr["cx"])) - m
    y0 = int(round(cy * U - spr["cy"])) - m
    h, w = a.shape
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if X1 > X0 and Y1 > Y0:
        k = np.clip(a[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0] * 2.2, 0, 1) * strength
        F.L[Y0:Y1, X0:X1] *= 1 - k
        F.G[Y0:Y1, X0:X1] *= 1 - k


def reels_captions(F, t):
    """Burned-in captions, typed on at the pace of the voice."""
    tim = vo_timing()
    for i in TL.REELS_CAPTIONS:
        st, dur = tim[i]
        on = st + 0.04
        nxt = TL.VO[i + 1][0] if i + 1 < len(TL.VO) else TL.CUT_TO_BLACK
        end = min(on + dur + 0.55, nxt - 0.08)
        if i == 12:                      # "You didn't." holds through the silence, gone on the bell
            end = TL.BELL
        if t < on - 0.05 or t > end + 0.25:
            continue
        out = 1 - ss(lin(t, end, end + 0.22))
        rows = wrap_caption(TL.VO[i][1].upper())
        total = sum(len(r) for r in rows)
        k = 0
        for ri, row in enumerate(rows):
            spr = text_sprite(row, "Inter-SemiBold.ttf", 50, 0.12)
            la = np.zeros(len(row), np.float32)
            for ci in range(len(row)):
                ts = on + 0.85 * dur * (k / total)
                la[ci] = ss(lin(t, ts - 0.03, ts + 0.1))
                k += 1
            y = TL.REELS_CAPTION_Y + (ri - (len(rows) - 1) / 2) * 74
            draw_halo(F, spr, 540, y, 0.62 * out * float(la.max()))
            draw_text(F, spr, 540, y, 0.95, alpha=out, letter_alpha=la, glow=0.12)


# ============================================================== frame assembly
def render(fi):
    tf = fi / TL.FPS                          # file time
    if CUT == "reels":
        t = tf + TL.REELS_OFFSET
    else:
        t = tf - TL.PRE                       # intro time; negative during the headphones card
    F = Frame(t)
    black = t >= TL.CUT_TO_BLACK
    if not black:
        if CUT == "site" and t < 1.6:         # the card's ash keeps rising into the intro
            headphones(F, t + TL.PRE)
        if CUT == "reels":
            reels_badge(F, tf)
            reels_captions(F, t)
        if 0 <= t < 7.9:                           # runs past 6.0 so its ash can finish rising
            scene1(F)
        if 6.0 <= t < 13.4:
            scene2(F)
        if 13.4 <= t < 24.6:
            scene3(F)
        if 23.9 <= t < 35.7:
            wa = ss(lin(t, 23.9, 24.6)) * (1 - ss(lin(t, 33.0, 34.2)))
            scene4(F, wa)
        if t >= 33.0:
            scene5(F)
        draw_ash(F)
    return finish(F, fi, black)


def finish(F, fi, black=False):
    L = F.L
    if not black:
        L += up(F.lowL)
        q = (W // 4, H // 4)
        gq = cv2.resize(F.G, q, interpolation=cv2.INTER_AREA) + cv2.resize(F.lowG, q, interpolation=cv2.INTER_LINEAR)
        glow = 0.6 * cv2.GaussianBlur(gq, (0, 0), 3 * U) + 0.5 * cv2.GaussianBlur(gq, (0, 0), 14 * U)
        L += cv2.resize(glow, (W, H), interpolation=cv2.INTER_LINEAR)
        a = cv2.GaussianBlur(F.ash, (0, 0), 0.55 * U)
        L += cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
        L += up(fog_field(F)) * 0.17
        if F.expo != 1.0:
            L *= F.expo
        L *= vignette()
        over = np.maximum(L - 0.8, 0)
        L -= over
        L += 0.2 * (1 - np.exp(-over / 0.2))
        red = None
        if F.lowR.max() > 0.002:
            red = up(cv2.GaussianBlur(F.lowR, (0, 0), 1.2))
    # grain, slightly heavier in the mids, never zero in the blacks
    bank = grain_bank()
    r = np.random.default_rng(1000 + fi)
    g = np.roll(bank[fi % len(bank)], (int(r.integers(0, H)), int(r.integers(0, W))), (0, 1)).astype(np.float32)
    if black:
        base = g * (0.006 / 40)
        out = np.clip(base * 255, 0, 255).astype(np.uint8)
        return np.repeat(out[:, :, None], 3, 2)
    amp = 0.013 + 0.05 * np.sqrt(np.clip(L, 0, 1)) * (1 - 0.6 * np.clip(L, 0, 1))
    base = L + g * amp * (1 / 40)
    out = np.empty((H, W, 3), np.uint8)
    for ch, (tint, rk) in enumerate([(0.929, 0.62), (0.929, 0.035), (0.902, 0.045)]):
        c = base * (tint * 255)
        if red is not None:
            c += red * (rk * 255)
        np.clip(c, 0, 255, out=c)
        out[:, :, ch] = c
    return out


# ============================================================== driver
_worker_scale = None


def _init(scale, cut):
    setup(scale, cut)


def _job(fi):
    return render(fi).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=2.0, help="2.0 = 2160x3840 (4K), 1.0 = 1080x1920")
    ap.add_argument("--stills", type=str, default="", help="comma list of file-time seconds to dump as PNG")
    ap.add_argument("--cut", choices=["site", "reels"], default="site")
    ap.add_argument("--out", default=None)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    setup(args.scale, args.cut)
    if args.end is None:
        args.end = TL.DUR - TL.REELS_OFFSET if args.cut == "reels" else TL.PRE + TL.DUR
    if args.out is None:
        args.out = os.path.join(ROOT, "build", f"video_4k_{args.cut}.mov")
    os.makedirs(os.path.join(ROOT, "build"), exist_ok=True)

    if args.stills:
        d = os.path.join(ROOT, "build", "stills")
        os.makedirs(d, exist_ok=True)
        for s in args.stills.split(","):
            fi = int(round(float(s) * TL.FPS))
            t0 = time.time()
            img = render(fi)
            cv2.imwrite(os.path.join(d, f"t{float(s):05.2f}.png"), img[:, :, ::-1])
            print(f"{s}s rendered in {time.time() - t0:.2f}s", flush=True)
        return

    f0, f1 = int(round(args.start * TL.FPS)), int(round(args.end * TL.FPS))
    # visually lossless intermediate; final delivery encodes come from this
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(TL.FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "6", "-pix_fmt", "yuv444p",
           "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", args.out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(args.jobs, initializer=_init, initargs=(args.scale, args.cut)) as pool:
        for i, buf in enumerate(pool.imap(_job, range(f0, f1), chunksize=2)):
            enc.stdin.write(buf)
            if i % 48 == 0:
                el = time.time() - t0
                print(f"frame {f0 + i}/{f1}  {el:.0f}s elapsed  {(i + 1) / el:.2f} fps", flush=True)
    enc.stdin.close()
    enc.wait()
    print(f"done in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
