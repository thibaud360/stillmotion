#!/usr/bin/env python3
"""stillmotion: turn one still photo into a seamless looping cinemagraph.

Only a masked part of the image moves; everything else stays pixel-identical.
No video model: masks + noise displacement fields + cv2.remap, looped by
blending two phases (or by using periodic cos/sin noise), so frame N == frame 0.

    python stillmotion.py photo.jpg                      # auto effect
    python stillmotion.py photo.jpg --effect drift       # clouds slide, loop
    python stillmotion.py photo.jpg --effect steam --region "35,5,60,38"
"""
import argparse
import math
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

EFFECTS = ("flow", "ripple", "drift", "shimmer", "steam", "snow")
DIRS = {"right": 0, "down": 90, "left": 180, "up": 270,
        "east": 0, "south": 90, "west": 180, "north": 270}


# ----------------------------------------------------------------- helpers
def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def fnoise(h, w, sx, sy=None, seed=0):
    """Smooth, unit-variance noise (gaussian-filtered white noise via FFT, generated on a
    reduced grid when the features are big, then upsampled: fast and smooth)."""
    sy = sx if sy is None else sy
    ds = int(np.clip(min(sx, sy) / 2.5, 1, 8))
    H, W = max(8, h // ds), max(8, w // ds)
    rng = np.random.default_rng(seed)
    f = np.fft.rfft2(rng.standard_normal((H, W)).astype(np.float32))
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.rfftfreq(W)[None, :]
    sxx, syy = sx * W / w, sy * H / h
    g = np.exp(-2 * np.pi ** 2 * ((sxx * fx) ** 2 + (syy * fy) ** 2))
    n = np.fft.irfft2(f * g, s=(H, W)).astype(np.float32)
    n = (n - n.mean()) / (n.std() + 1e-8)
    if (H, W) != (h, w):
        n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    return n.astype(np.float32)


def fbm(h, w, sx, sy=None, seed=0, octaves=3):
    """Perlin-like fractal noise: a few octaves of fnoise, each finer and weaker."""
    sy = sx if sy is None else sy
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        out += amp * fnoise(h, w, max(sx / 2 ** o, 0.6), max(sy / 2 ** o, 0.6), seed * 17 + o)
        tot += amp * amp
        amp *= 0.5
    return out / math.sqrt(tot)


def load_image(path, width):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = im.size
    if w > width:
        h = round(h * width / w)
        w = width
        im = im.resize((w, h), Image.LANCZOS)
    w -= w % 2
    h -= h % 2
    if (w, h) != im.size:
        im = im.crop((0, 0, w, h))
    return np.asarray(im, np.float32) / 255.0


def parse_dir(s):
    if s is None:
        return None
    s = str(s).strip().lower()
    if s in DIRS:
        return float(DIRS[s])
    return float(s)  # degrees: 0 = right, 90 = down, 180 = left, 270 = up


def lum(img):
    return img @ np.array([0.299, 0.587, 0.114], np.float32)


def feather(mask, px):
    if px > 0.5:
        mask = cv2.GaussianBlur(mask, (0, 0), px)
    return np.clip(mask, 0, 1).astype(np.float32)


# ------------------------------------------------------------------- masks
def parse_region(spec, w, h):
    """'x1,y1,x2,y2' (percent) -> rectangle; 'x,y;x,y;x,y...' (percent) -> polygon."""
    m = np.zeros((h, w), np.uint8)
    if ";" in spec:
        pts = np.array([[float(v) for v in p.split(",")] for p in spec.split(";") if p.strip()])
        pts = (pts / 100.0 * [w, h]).round().astype(np.int32)
        cv2.fillPoly(m, [pts], 255)
    else:
        x1, y1, x2, y2 = [float(v) for v in spec.split(",")]
        cv2.rectangle(m, (round(x1 / 100 * w), round(y1 / 100 * h)),
                      (round(x2 / 100 * w), round(y2 / 100 * h)), 255, -1)
    return m.astype(np.float32) / 255.0


def load_mask(path, w, h):
    m = Image.open(path)
    if m.mode in ("RGBA", "LA"):
        m = m.getchannel("A") if np.asarray(m.getchannel("A")).min() < 255 else m.convert("L")
    m = m.convert("L").resize((w, h), Image.BILINEAR)
    return np.asarray(m, np.float32) / 255.0


def _small(img, target=360):
    h, w = img.shape[:2]
    s = target / w
    return cv2.resize(img, (target, max(2, round(h * s))), interpolation=cv2.INTER_AREA)


def sky_mask(img):
    """Sky = smooth (low-gradient) region connected to the top edge, stopping at ridges/horizons."""
    h, w = img.shape[:2]
    sm = cv2.GaussianBlur(_small(img), (0, 0), 1.2)
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3) / 8
    grad = np.sqrt((gx ** 2 + gy ** 2).sum(-1))
    edge = cv2.dilate((grad > 0.028).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    smooth = (1 - edge).astype(np.uint8)
    sh, sw = smooth.shape
    smooth[int(sh * 0.92):] = 0                      # position prior: sky is not the bottom
    n, lab = cv2.connectedComponents(smooth, connectivity=4)
    keep = np.zeros(sh * sw, bool)
    top = np.unique(lab[: max(2, sh // 40)])
    for l in top:
        if l == 0:
            continue
        if (lab == l).sum() > 0.01 * sh * sw:
            keep |= (lab == l).ravel()
    m = keep.reshape(sh, sw).astype(np.uint8)
    # clouds: bright, desaturated (or blue) pixels touching the sky belong to it too
    mx, mn = sm.max(-1), sm.min(-1)
    sat = (mx - mn) / (mx + 1e-6)
    cloud = ((sat < 0.20) & (mx > 0.55)) | (sm[..., 2] > sm[..., 0] + 0.10)
    cloud[int(sh * 0.92):] = False
    seed = cv2.dilate(m, np.ones((5, 5), np.uint8))
    cand = (cloud | (m > 0)).astype(np.uint8)
    n2, lab2 = cv2.connectedComponents(cand, connectivity=8)
    ids = np.unique(lab2[seed > 0])
    m = np.isin(lab2, ids[ids > 0]).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.erode(m, np.ones((3, 3), np.uint8))       # stay off the ridge line
    m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return m


def water_mask(img):
    """Water = bluish / dark / low-texture region in the lower part, connected to the bottom edge."""
    h, w = img.shape[:2]
    sm = cv2.GaussianBlur(_small(img), (0, 0), 1.5)
    sh, sw = sm.shape[:2]
    r, g, b = sm[..., 0], sm[..., 1], sm[..., 2]
    bluish = (b > r * 1.02) & (b > 0.8 * g)
    l = lum(sm)
    tex = cv2.GaussianBlur(np.abs(l - cv2.GaussianBlur(l, (0, 0), 3)), (0, 0), 4)
    cand = (bluish | (tex < 0.012)).astype(np.uint8)
    cand[: int(sh * 0.30)] = 0
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(cand, connectivity=4)
    keep = np.zeros_like(cand, bool)
    for l_ in np.unique(lab[-3:]):
        if l_ and (lab == l_).sum() > 0.02 * sh * sw:
            keep |= lab == l_
    if keep.mean() < 0.04:
        print("  water not detected, using the bottom 40% (pass --region to choose)")
        keep[:] = False
        keep[int(sh * 0.6):] = True
    m = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.erode(m, np.ones((3, 3), np.uint8))
    return cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def light_mask(img):
    """Bright / glowing areas (for shimmer and light flicker)."""
    l = cv2.GaussianBlur(lum(img), (0, 0), 3)
    p = np.percentile(l, 99)
    m = smoothstep(0.18 * p, 0.55 * p, l)
    return cv2.GaussianBlur(m, (0, 0), max(img.shape[1] * 0.012, 2)).clip(0, 1)


def make_plate(img, mask):
    """Fill pixels outside the mask with colours extrapolated from inside, so warping never
    drags foreground (mountains, shore, cup) into the moving area."""
    known = (mask > 0.35).astype(np.float32)
    if known.min() > 0.5:
        return img
    h, w = mask.shape
    sc = 0.25
    sw, sh = max(8, int(w * sc)), max(8, int(h * sc))
    ks = cv2.resize(known, (sw, sh), interpolation=cv2.INTER_AREA)
    si = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA) * ks[..., None]
    est = np.zeros_like(si)
    filled = ks > 0.5
    est[filled] = si[filled] / ks[..., None][filled]
    for sigma in (1, 2, 5, 12, 30, 75):
        bw = cv2.GaussianBlur(ks, (0, 0), sigma)
        bf = cv2.GaussianBlur(si, (0, 0), sigma)
        e2 = bf / (bw[..., None] + 1e-6)
        ok = (~filled) & (bw > 0.02)
        est[ok] = e2[ok]
        filled |= ok
    est = cv2.resize(est, (w, h), interpolation=cv2.INTER_CUBIC)
    plate = np.where((known > 0.5)[..., None], img, est)
    return plate


# ----------------------------------------------------------------- effects
class Base:
    def __init__(self, img, mask, a):
        self.img, self.mask, self.a = img, mask, a
        self.h, self.w = img.shape[:2]
        self.X, self.Y = np.meshgrid(np.arange(self.w, dtype=np.float32),
                                     np.arange(self.h, dtype=np.float32))
        self.m3 = mask[..., None]

    def remap(self, src, mx, my):
        return cv2.remap(src, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    def composite(self, warped, plate):
        return np.clip(self.img + self.m3 * (warped - plate), 0, 1)


class Advect(Base):
    """flow / drift: content slides along a direction; two phases are crossfaded so the
    loop is seamless, and the phase differs from place to place so there is never one global
    'cross-dissolve moment'."""

    def __init__(self, img, mask, a, kind):
        super().__init__(img, mask, a)
        w, h = self.w, self.h
        ang = a.dir_deg if a.dir_deg is not None else (0.0 if kind == "drift" else 90.0)
        self.d = (math.cos(math.radians(ang)), math.sin(math.radians(ang)))
        base = 0.075 if kind == "drift" else 0.06
        self.L = w * base * a.speed
        self.plate = make_plate(img, mask)
        sc = w * (0.18 if kind == "drift" else 0.10)
        self.m = np.clip(1 + 0.35 * fbm(h, w, sc, sc * (1.8 if kind == "drift" else 1.0), a.seed + 1), 0.5, 1.7)
        self.phi = np.clip(0.5 + 0.30 * fbm(h, w, w * 0.14, w * 0.14, a.seed + 2, 2), 0.02, 0.98) * 0.95
        self.turb = fbm(h, w, w * 0.05, w * 0.05, a.seed + 3) * (0.25 if kind == "drift" else 0.4)
        self.kind = kind
        # small periodic ripple layered on top (sparkle for water, billowing for clouds)
        self.n1 = fbm(h, w, w * 0.02, w * 0.008 if kind == "flow" else w * 0.02, a.seed + 4, 2)
        self.n2 = fbm(h, w, w * 0.02, w * 0.008 if kind == "flow" else w * 0.02, a.seed + 5, 2)
        self.cycles = max(1, round(2 * a.speed))
        self.ramp = 1.0
        if kind == "flow":
            ys = np.where(mask.max(1) > 0.3)[0]
            if len(ys):
                t = np.clip((self.Y - ys[0]) / max(ys[-1] - ys[0], 1), 0, 1)
                self.ramp = 0.5 + 0.8 * t

    def frame(self, u, k=1.0):
        a = self.a
        up = (u + self.phi) % 1.0
        dx, dy = self.d
        px, py = -dy, dx
        L = self.L * self.m * k * a.strength
        th = 2 * math.pi * self.cycles * u
        rip = (0.0035 if self.kind == "flow" else 0.0015) * self.w * k * a.strength * self.ramp
        rx = rip * (self.n1 * math.cos(th) + self.n2 * math.sin(th))
        ry = 0.5 * rip * (self.n2 * math.cos(th) - self.n1 * math.sin(th))
        t = self.turb * self.L * k * a.strength
        out = np.zeros_like(self.img)
        for ph, wgt in ((up, 1 - up), (up - 1.0, up)):
            mx = self.X - dx * L * ph - px * t * ph + rx
            my = self.Y - dy * L * ph - py * t * ph + ry
            out += self.remap(self.plate, mx, my) * wgt[..., None]
        return self.composite(out, self.plate)


class Periodic(Base):
    """ripple / shimmer: displacement = noise1*cos(2*pi*c*u) + noise2*sin(2*pi*c*u): periodic by construction."""

    def __init__(self, img, mask, a, kind):
        super().__init__(img, mask, a)
        w, h = self.w, self.h
        self.kind = kind
        s = a.seed
        if kind == "ripple":
            sx, sy, amp, c = w * 0.035, w * 0.009, 0.0042, 2
        else:
            sx, sy, amp, c = w * 0.010, w * 0.030, 0.0016, 3
        self.amp = amp * w * a.strength
        self.cycles = max(1, round(c * a.speed))
        self.nx1, self.nx2 = (fbm(h, w, sx, sy, s + 10 + i, 3) for i in range(2))
        self.ny1, self.ny2 = (fbm(h, w, sx, sy, s + 20 + i, 3) for i in range(2))
        self.g1, self.g2 = (fbm(h, w, w * 0.12, w * 0.12, s + 30 + i, 2) for i in range(2))
        self.plate = make_plate(img, mask) if kind == "ripple" else img
        self.travel = kind == "shimmer" and a.dir_deg is not None
        if self.travel:   # light pulses that run along --dir (periodic: whole waves per loop)
            ang = math.radians(a.dir_deg)
            s_ = (self.X * math.cos(ang) + self.Y * math.sin(ang)) / w
            self.s = s_ - s_.min()
            self.wl = 0.55 / a.speed
            self.env1 = fbm(h, w, w * 0.18, w * 0.18, s + 70, 2)
            self.env2 = fbm(h, w, w * 0.18, w * 0.18, s + 71, 2)
        ys = np.where(mask.max(1) > 0.3)[0]
        self.ramp = 1.0
        if kind == "ripple" and len(ys):
            t = np.clip((self.Y - ys[0]) / max(ys[-1] - ys[0], 1), 0, 1)
            self.ramp = 0.45 + 0.9 * t

    def frame(self, u, k=1.0):
        th = 2 * math.pi * self.cycles * u
        c, s_ = math.cos(th), math.sin(th)
        A = self.amp * k * self.ramp
        dx = A * (self.nx1 * c + self.nx2 * s_)
        dy = 0.6 * A * (self.ny1 * c + self.ny2 * s_)
        if self.kind == "shimmer":
            dy = dy * 0.5
        warped = self.remap(self.plate, self.X + dx, self.Y + dy)
        out = self.composite(warped, self.plate)
        if self.kind == "shimmer":   # light flicker, spatially varying
            gain = 1 + 0.24 * k * self.a.strength * (self.g1 * c + self.g2 * s_)
            gain = 1 + self.m3 * (np.clip(gain, 0.6, 1.5)[..., None] - 1)
            out = np.clip(out * gain, 0, 1)
        if self.travel:
            ph = 2 * math.pi * (self.s / self.wl - u * max(1, round(self.a.speed)))
            wave = 0.5 + 0.5 * np.cos(ph + 0.8 * (self.env1 * math.cos(th) + self.env2 * math.sin(th)))
            wave = wave ** 1.5
            gain = 1 + self.m3 * (self.a.strength * k * 0.75 * (wave - 0.4))[..., None]
            out = np.clip(out * gain, 0, 1)
        return out


class Steam(Base):
    def __init__(self, img, mask, a, box):
        super().__init__(img, mask, a)
        w, h = self.w, self.h
        x0, y0, x1, y1 = box
        self.box = box
        bw, bh = x1 - x0, y1 - y0
        self.ww, self.hh = bw, bh
        self.Ht = max(16, int(bh * 0.85 / a.speed))
        Wt = bw * 2
        tex = fbm(self.Ht, Wt, bw * 0.045, bw * 0.16, a.seed + 40, 3)
        tex = np.ascontiguousarray(tex)
        self.tex2 = np.vstack([tex, tex])
        self.wob1, self.wob2 = (fbm(bh, bw, bw * 0.3, bh * 0.3, a.seed + 50 + i, 2) for i in range(2))
        yy, xx = np.mgrid[0:bh, 0:bw].astype(np.float32)
        self.xx, self.yy = xx, yy
        self.r = (bh - 1 - yy) / bh
        self.xr = xx - bw / 2
        self.mroi = mask[y0:y1, x0:x1]
        self.rise = fbm(bh, bw, bw * 0.12, bh * 0.1, a.seed + 60, 2)

    def frame(self, u, k=1.0):
        x0, y0, x1, y1 = self.box
        bw, bh, a = self.ww, self.hh, self.a
        r, xr = self.r, self.xr
        sway = 0.16 * bw * (0.3 + r) * np.sin(2 * math.pi * (1.1 * r - u))
        th = 2 * math.pi * u
        sway = sway + 0.08 * bw * (self.wob1 * math.cos(th) + self.wob2 * math.sin(th)) * (0.3 + r)
        mx = (bw + xr - sway).astype(np.float32)
        my = ((self.yy + u * self.Ht) % self.Ht).astype(np.float32)
        v = cv2.remap(self.tex2, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        wisp = smoothstep(0.05, 1.7, v)
        sig = bw * (0.10 + 0.20 * r)
        envx = np.exp(-0.5 * ((xr - sway * 0.8) / sig) ** 2)
        envy = smoothstep(0.0, 0.14, r) * (1 - smoothstep(0.6, 1.0, r))
        alpha = wisp * envx * envy * 0.85 * a.strength * k * self.mroi
        alpha = np.clip(alpha, 0, 0.9)[..., None]
        col = np.array([0.98, 0.97, 0.96], np.float32)
        out = self.img.copy()
        roi = out[y0:y1, x0:x1]
        out[y0:y1, x0:x1] = roi * (1 - alpha) + col * alpha
        return out


class Snow(Base):
    def __init__(self, img, mask, a):
        super().__init__(img, mask, a)
        w, h = self.w, self.h
        rng = np.random.default_rng(a.seed + 7)
        ang = a.dir_deg if a.dir_deg is not None else 90.0
        # layers: far, mid, near  -> (share, radius, travel as fraction of height, alpha, blur)
        spec = [(0.55, 2.0, 0.30, 0.65, 1.0), (0.30, 3.4, 0.50, 0.85, 1.6), (0.15, 6.0, 0.80, 0.90, 3.2)]
        total = int(w * h / 2600 * a.density)
        s = w / 1080.0
        vec = (math.cos(math.radians(ang)) , math.sin(math.radians(ang)))
        # slight wind slant so flakes don't fall perfectly vertically
        slant = 0.12 if a.dir_deg is None else 0.0
        self.layers = []
        for share, rad, trav, al, bl in spec:
            n = max(1, int(total * share))
            T = trav * max(h, w) * a.speed
            life_dx = (vec[0] + slant) * T
            life_dy = vec[1] * T
            x0 = rng.uniform(-0.2 * w - abs(life_dx), 1.2 * w, n)
            y0 = rng.uniform(-0.2 * h - abs(life_dy), 1.1 * h, n)
            self.layers.append(dict(
                x0=x0, y0=y0, dx=life_dx, dy=life_dy, ph=rng.uniform(0, 1, n),
                r=rad * s * rng.uniform(0.7, 1.3, n), al=al * rng.uniform(0.6, 1.0, n),
                sw=rng.uniform(8, 26, n) * s * trav / 0.5, sm=rng.integers(1, 3, n),
                sp=rng.uniform(0, 1, n), blur=bl * s, streak=0.0))

    def frame(self, u, k=1.0):
        h, w = self.h, self.w
        acc = np.zeros((h, w), np.float32)
        for L in self.layers:
            canvas = np.zeros((h, w), np.uint8)
            life = (u + L["ph"]) % 1.0
            fade = smoothstep(0, 0.10, life) * (1 - smoothstep(0.90, 1.0, life))
            x = L["x0"] + L["dx"] * life + L["sw"] * np.sin(2 * math.pi * (L["sm"] * u + L["sp"]))
            y = L["y0"] + L["dy"] * life
            val = np.clip(255 * L["al"] * fade * k * self.a.strength, 0, 255)
            for xi, yi, ri, vi in zip(x, y, L["r"], val):
                if vi < 2 or xi < -8 or yi < -8 or xi > w + 8 or yi > h + 8:
                    continue
                cv2.circle(canvas, (int(xi * 16), int(yi * 16)), max(int(ri * 16), 8),
                           int(vi), -1, cv2.LINE_AA, 4)
                if L["streak"] > 0.5:      # short motion trail: reads as falling, not as stars
                    cv2.line(canvas, (int(xi * 16), int(yi * 16)),
                             (int((xi - L["dx"] * 0.025) * 16), int((yi - L["dy"] * 0.025) * 16)),
                             int(vi * 0.7), max(1, int(ri * 1.1)), cv2.LINE_AA, 4)
            lay = cv2.GaussianBlur(canvas.astype(np.float32) / 255.0, (0, 0), L["blur"]) * (1.0 + 0.45 * L["blur"] / max(self.w / 1080, 0.3))
            acc = 1 - (1 - acc) * (1 - np.clip(lay, 0, 1))
        snow = (acc * self.mask)[..., None]
        tint = np.array([1.0, 0.99, 1.0], np.float32)
        return np.clip(1 - (1 - self.img) * (1 - snow * tint), 0, 1)


# -------------------------------------------------------------------- main
def build(effect, img, mask, a, box):
    if effect in ("flow", "drift"):
        return Advect(img, mask, a, effect)
    if effect in ("ripple", "shimmer"):
        return Periodic(img, mask, a, effect)
    if effect == "steam":
        return Steam(img, mask, a, box)
    return Snow(img, mask, a)


def mask_bbox(mask, w, h, thr=0.3):
    ys, xs = np.where(mask > thr)
    if len(xs) == 0:
        return (0, 0, w, h)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def prepare(path, a):
    """Load the photo, choose effect + mask. Returns (img, mask, effect, box)."""
    img = load_image(path, a.width)
    h, w = img.shape[:2]
    effect = a.effect
    region = None
    if a.mask:
        region = load_mask(a.mask, w, h)
        if a.invert:
            region = 1 - region
    elif a.region:
        region = parse_region(a.region, w, h)
    fpx = a.feather / 100.0 * min(w, h)

    if effect == "auto":
        if region is not None:
            effect = "drift" if sky_mask(img).mean() > 0.2 else "ripple"
        else:
            sm_, wm_ = sky_mask(img).mean(), water_mask(img).mean()
            effect = "drift" if sm_ >= 0.2 and sm_ >= wm_ else ("flow" if wm_ >= 0.1 else "shimmer")
        print(f"  auto effect -> {effect}")

    if region is not None:
        mask = feather(region, fpx)
    elif effect == "drift":
        mask = feather(sky_mask(img), fpx or w * 0.01)
    elif effect in ("flow", "ripple"):
        mask = feather(water_mask(img), fpx or w * 0.01)
    elif effect == "shimmer":
        mask = light_mask(img)
    elif effect == "steam":
        mask = np.ones((h, w), np.float32)
    else:
        mask = np.ones((h, w), np.float32)

    box = None
    if effect == "steam":
        if region is not None:
            box = mask_bbox(region, w, h, 0.5)
        else:
            box = (int(w * 0.35), 0, int(w * 0.65), int(h * 0.45))
            print("  no --region given: steam rises from the middle of the frame; "
                  "pass --region x1,y1,x2,y2 (percent) just above the cup")
        x0, y0, x1, y1 = box
        mask = np.ones((h, w), np.float32)
        mask[:] = 1.0
    return img, mask, effect, box


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        sys.exit("ffmpeg not found (install ffmpeg or `pip install imageio-ffmpeg`)")


def write_mp4(frames, path, fps, w, h, live=False):
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if live:
        cmd += ["-profile:v", "high", "-level", "4.0", "-tag:v", "avc1"]
    cmd += ["-an", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        sys.exit("ffmpeg failed")


def write_gif(mp4, path, gif_width, fps=15, max_mb=3.5):
    for gw, gf, colors in ((gif_width, fps, 128), (gif_width, 12, 96), (int(gif_width * .8), 12, 64),
                           (int(gif_width * .65), 10, 64), (int(gif_width * .5), 10, 48)):
        vf = (f"fps={gf},scale={gw}:-2:flags=lanczos,split[s0][s1];"
              f"[s0]palettegen=max_colors={colors}:stats_mode=full[p];"
              f"[s1][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
        subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", mp4, "-vf", vf, "-loop", "0", path], check=True)
        if os.path.getsize(path) <= max_mb * 1e6:
            return


def render(eff, n, k_fn=None):
    for i in range(n + 1):                 # n+1 frames: the last one is u = 1 = frame 0
        k = 1.0 if k_fn is None else k_fn(i / n)
        yield eff.frame(i / n, k)


def to8(f):
    return (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="One still photo -> seamless looping cinemagraph (MP4 + GIF).")
    ap.add_argument("input")
    ap.add_argument("--effect", default="auto", choices=("auto",) + EFFECTS)
    ap.add_argument("--region", help='moving area in percent: "x1,y1,x2,y2" or polygon "x,y;x,y;x,y"')
    ap.add_argument("--mask", help="mask PNG (white = moves)")
    ap.add_argument("--invert", action="store_true", help="invert --mask")
    ap.add_argument("--dir", dest="dir", help="direction: left|right|up|down or degrees (0=right, 90=down)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--density", type=float, default=1.0, help="snow amount")
    ap.add_argument("--feather", type=float, default=0.0, help="mask edge softness, %% of image size (default auto)")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--width", type=int, default=1080, help="max output width")
    ap.add_argument("--gif-width", type=int, default=480)
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--no-mp4", action="store_true")
    ap.add_argument("--live", action="store_true", help="also write a 3 s iPhone-friendly clip (30 fps)")
    ap.add_argument("--save-mask", action="store_true", help="write the mask as PNG to check it")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-o", "--out", default="stillmotion_out")
    a = ap.parse_args()
    a.dir_deg = parse_dir(a.dir)

    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.input))[0]
    print(f"{a.input}")
    img, mask, effect, box = prepare(a.input, a)
    h, w = img.shape[:2]
    print(f"  effect={effect}  size={w}x{h}  moving area={mask.mean() * 100:.0f}% of the frame")
    if a.save_mask:
        mp = os.path.join(a.out, f"{stem}_mask.png")
        Image.fromarray(to8(mask)).save(mp)
        ov = img * (0.35 + 0.65 * mask[..., None]) + np.array([0.0, 0.25, 0.0]) * mask[..., None] * 0.5
        Image.fromarray(to8(ov)).save(os.path.join(a.out, f"{stem}_maskview.jpg"), quality=88)
        print(f"  mask -> {mp}")

    eff = build(effect, img, mask, a, box)
    base = os.path.join(a.out, f"{stem}_{effect}")
    jobs = [("main", a.seconds, a.fps)]
    if a.live:
        jobs.append(("live", 3.0, 30))
    for tag, secs, fps in jobs:
        n = max(2, round(secs * fps))
        frames = [to8(f) for f in render(eff, n)]
        last, first = frames[-1].astype(np.int16), frames[0].astype(np.int16)
        wrap = np.abs(last - first)
        cons = np.mean([np.abs(frames[i + 1].astype(np.int16) - frames[i]).mean() for i in range(0, n, max(1, n // 24))])
        print(f"  [{tag}] loop check: frame {n} vs frame 0  max diff {int(wrap.max())}, mean {wrap.mean():.4f} "
              f"(typical consecutive frames: mean {cons:.3f})  -> {'seamless' if wrap.max() <= 1 else 'NOT seamless'}")
        loop = frames[:-1]                 # drop duplicate end frame: loops forever without a hitch
        suffix = "_live" if tag == "live" else ""
        mp4 = f"{base}{suffix}.mp4"
        if not a.no_mp4 or (tag == "main" and not a.no_gif):
            write_mp4(loop, mp4, fps, w, h, live=(tag == "live"))
            print(f"  -> {mp4}  ({os.path.getsize(mp4) / 1e6:.2f} MB)")
        if tag == "main" and not a.no_gif:
            gif = f"{base}.gif"
            write_gif(mp4, gif, a.gif_width)
            print(f"  -> {gif}  ({os.path.getsize(gif) / 1e6:.2f} MB)")
            if a.no_mp4:
                os.remove(mp4)


if __name__ == "__main__":
    main()
