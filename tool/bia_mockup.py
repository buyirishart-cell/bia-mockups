#!/usr/bin/env python3
"""Buy Irish Art mockup tool, v3: rendered frames in empty rooms.

The rooms are photographs of bare walls. The tool builds the frame itself, at the
painting's exact proportions and in the finish you choose (black / wood / white,
matching the Gelato range), places the REAL artwork file in it untouched, then
lays the room's own light and shadow (read from that patch of wall) over frame and
glass, and casts the frame's shadow onto the wall.

Nothing is stretched, nothing in the room is covered (each template says where the
frame may go), and the art is never cropped or redrawn.

Usage:
  python3 bia_mockup.py ARTWORK --templates templates/templates.json --out OUTDIR
                        [--name HANDLE] [--finish wood|black|white|auto] [--only lean,kitchen]
"""
import argparse, json, os
import cv2
import numpy as np

FINISH = {  # BGR
    "wood":  {"base": (142, 176, 200), "grain": 0.08},   # natural oak
    "black": {"base": (32, 32, 34),    "grain": 0.04},
    "white": {"base": (236, 238, 238), "grain": 0.02},
}

def srgb_to_lin(x):
    x = x / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def lin_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.clip(np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055) * 255, 0, 255)

def trim_border(img, tol=18):
    """Remove flat white/near-uniform padding around a flat artwork file."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
    corner = int(np.median([g[0, 0], g[0, -1], g[-1, 0], g[-1, -1]]))
    mask = np.abs(g - corner) > tol
    rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    if len(rows) == 0:
        return img
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    if (y1 - y0) * (x1 - x0) < 0.3 * img.shape[0] * img.shape[1]:
        return img
    return img[y0:y1, x0:x1]

def grain_texture(h, w, along, strength, seed):
    """Fine straight grain running along the bar (along = 'x' or 'y')."""
    rng = np.random.default_rng(seed)
    n = max(h, w)
    lines = rng.normal(0, 1, n * 2)
    lines = cv2.GaussianBlur(lines.reshape(1, -1).astype(np.float32), (0, 0), 1.2).ravel()
    if along == "x":   # grain lines are horizontal: vary with y
        t = np.tile(lines[:h].reshape(-1, 1), (1, w))
        t += 0.35 * cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=8, sigmaY=1)
    else:
        t = np.tile(lines[:w].reshape(1, -1), (h, 1))
        t += 0.35 * cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=1, sigmaY=8)
    t = (t - t.mean()) / (t.std() + 1e-6)
    return 1 + strength * t

def build_frame(art, moulding, finish, light_from_left=True):
    """Return (rgb, alpha) of the framed artwork, flat-lit. Art sits 1:1 inside."""
    ah, aw = art.shape[:2]
    m = moulding
    H, W = ah + 2 * m, aw + 2 * m
    f = FINISH[finish]
    base = np.array(f["base"], np.float32)
    img = np.zeros((H, W, 3), np.float32)
    # four mitred bars; top/left catch the light, bottom/right fall away
    yy, xx = np.mgrid[0:H, 0:W]
    top = (yy < m) & (xx >= yy) & (xx < W - yy)
    bot = (yy >= H - m) & (xx >= H - 1 - yy) & (xx < W - (H - 1 - yy))
    lef = (xx < m) & ~top & ~bot
    rig = (xx >= W - m) & ~top & ~bot
    lit = {"top": 1.10, "left": 1.04 if light_from_left else 0.90,
           "right": 0.90 if light_from_left else 1.04, "bottom": 0.82}
    gx = grain_texture(H, W, "x", f["grain"], 1)
    gy = grain_texture(H, W, "y", f["grain"], 2)
    for mask, k, g in ((top, "top", gx), (bot, "bottom", gx), (lef, "left", gy), (rig, "right", gy)):
        img[mask] = base * lit[k] * g[mask][:, None]
    # rounded profile: brighter crest, darker inner and outer edges
    d_out = np.minimum.reduce([yy, xx, H - 1 - yy, W - 1 - xx]).astype(np.float32)
    prof = np.clip(d_out / m, 0, 1)
    shade = 0.90 + 0.12 * np.sin(np.pi * prof)        # crest in the middle of the bar
    img *= np.where(d_out < m, shade, 1)[..., None]
    # thin dark rebate line where moulding meets the art
    img[(d_out >= m - 2) & (d_out < m)] *= 0.55
    # the art, untouched, plus a soft shadow from the moulding falling on it
    a = art.astype(np.float32)
    s = max(3, int(m * 0.45))
    sh = np.ones((ah, aw), np.float32)
    ramp = np.linspace(0.62, 1, s)
    sh[:s, :] *= ramp[:, None]                          # under the top bar
    if light_from_left:
        sh[:, :s] *= ramp[None, :]                      # under the left bar
    else:
        sh[:, -s:] *= ramp[::-1][None, :]
    a *= sh[..., None]
    img[m:m + ah, m:m + aw] = a
    return np.clip(img, 0, 255)

def light_map(room, x0, y0, x1, y1):
    """The wall's light pattern under where the frame will hang, texture removed."""
    patch = room[max(y0, 0):y1, max(x0, 0):x1]
    L = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    sig = max(4, 0.012 * max(patch.shape[:2]))
    L = cv2.GaussianBlur(L, (0, 0), sig)
    ref = np.percentile(L, 97)
    return np.clip(L / max(ref, 1), 0.35, 1.0)

def compose(room, tpl, art, finish, scale):
    H, W = room.shape[:2]
    ax0, ay0, ax1, ay1 = [int(round(v * scale)) for v in tpl["area"]]
    aw_, ah_ = ax1 - ax0, ay1 - ay0
    ar = art.shape[1] / art.shape[0]
    m_frac = tpl.get("moulding", 0.028)
    # largest framed size that fits the allowed area at this aspect
    fill = tpl.get("fill", 0.92)
    for _ in range(3):
        Wf = aw_ * fill
        Hf = Wf / ar
        if Hf > ah_ * fill:
            Hf = ah_ * fill
            Wf = Hf * ar
        m = max(10, int(round(max(Wf, Hf) * m_frac)))
        iw, ih = int(round(Wf - 2 * m)), int(round(Hf - 2 * m))
        if iw > 50 and ih > 50:
            break
    # keep the art proportions exact (inner window = art aspect)
    if iw / ih > ar:
        iw = int(round(ih * ar))
    else:
        ih = int(round(iw / ar))
    a = cv2.resize(art, (iw, ih), interpolation=cv2.INTER_AREA if iw < art.shape[1] else cv2.INTER_CUBIC)
    lfl = tpl.get("light", "left") == "left"
    framed = build_frame(a, m, finish, lfl)
    fh, fw = framed.shape[:2]
    cx = (ax0 + ax1) / 2
    x0 = int(round(cx - fw / 2))
    y0 = (ay1 - fh) if tpl.get("anchor") == "bottom" else int(round((ay0 + ay1) / 2 - fh / 2))
    x1, y1 = x0 + fw, y0 + fh
    out = room.astype(np.float32)
    # cast shadow on the wall, away from the light
    sdx = (1 if lfl else -1) * max(6, int(fw * 0.018))
    sdy = max(8, int(fh * 0.022))
    shadow = np.zeros((H, W), np.float32)
    cv2.rectangle(shadow, (x0 + sdx, y0 + sdy), (x1 + sdx, y1 + sdy), 1, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), max(6, fw * 0.02)) * tpl.get("shadow", 0.38)
    if tpl.get("anchor") == "bottom":   # contact shadow where it rests on floor / ledge
        cs = np.zeros((H, W), np.float32)
        cv2.rectangle(cs, (x0 - 4, y1 - 6), (x1 + 4, y1 + 6), 1, -1)
        shadow = np.maximum(shadow, cv2.GaussianBlur(cs, (0, 0), 6) * 0.5)
    out *= (1 - shadow[..., None])
    # room light over frame and glass, in linear light
    lm = light_map(room, x0, y0, x1, y1)
    lm = cv2.resize(lm, (fw, fh), interpolation=cv2.INTER_LINEAR)
    lin = srgb_to_lin(framed) * lm[..., None]
    # the wall's colour cast (warm sun / cool shade), gently
    wall = room[max(y0, 0):y1, max(x0, 0):x1].reshape(-1, 3).astype(np.float32)
    bright = wall[np.argsort(wall.sum(1))[-max(50, len(wall) // 20):]].mean(0)
    tint = 1 + (bright / bright.mean() - 1) * 0.35
    lin *= tint[None, None, :]
    res = lin_to_srgb(lin)
    # faint glass sheen across the art only
    gy, gx = np.mgrid[0:fh, 0:fw]
    sheen = np.clip(1 - np.abs((gx / fw + gy / fh) - 0.55) * 6, 0, 1) * 6
    res[m:fh - m, m:fw - m] = np.clip(res[m:fh - m, m:fw - m] + sheen[m:fh - m, m:fw - m, None], 0, 255)
    cy0, cx0, cy1, cx1 = max(y0, 0), max(x0, 0), min(y1, H), min(x1, W)
    out[cy0:cy1, cx0:cx1] = res[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    return np.clip(out, 0, 255).astype(np.uint8), (x0, y0, x1, y1), m

def detail_crop(art, size=2048, frac=0.42):
    """Brushwork close-up: a true crop of the master file, no AI."""
    h, w = art.shape[:2]
    side = int(min(h, w) * frac)
    g = cv2.cvtColor(art, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    best, bxy = -1, (0, 0)
    for fy in np.linspace(0.15, 0.85, 8):
        for fx in np.linspace(0.15, 0.85, 8):
            x = min(max(int(fx * w - side / 2), 0), w - side)
            y = min(max(int(fy * h - side / 2), 0), h - side)
            v = lap[y:y + side, x:x + side].mean()
            if v > best:
                best, bxy = v, (x, y)
    x, y = bxy
    c = art[y:y + side, x:x + side]
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_CUBIC if side < size else cv2.INTER_AREA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artwork"); ap.add_argument("--templates", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--only", default="")
    ap.add_argument("--name", default="", help="output file stem, e.g. the Shopify product handle")
    ap.add_argument("--finish", default="auto", help="wood | black | white | auto (each room's own default)")
    ap.add_argument("--size", type=int, default=2048)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    art = cv2.imread(args.artwork, cv2.IMREAD_COLOR)
    art = trim_border(art)
    cfg = json.load(open(args.templates))
    base = os.path.dirname(os.path.abspath(args.templates))
    only = set(filter(None, args.only.split(",")))
    stem = args.name or os.path.splitext(os.path.basename(args.artwork))[0]
    report = []
    for t in sorted(cfg["templates"], key=lambda t: t.get("order", 99)):
        if only and t["room"] not in only:
            continue
        room = cv2.imread(os.path.join(base, t["file"]))
        scale = args.size / room.shape[1]
        room = cv2.resize(room, (args.size, args.size), interpolation=cv2.INTER_LANCZOS4)
        finish = t.get("finish", "wood") if args.finish == "auto" else args.finish
        img, box, m = compose(room, t, art, finish, scale)
        fn = os.path.join(args.out, f"{stem}-{t.get('order', 9):02d}-{t['room']}.jpg")
        cv2.imwrite(fn, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        report.append(f"{t['room']}: {finish} frame {box[2]-box[0]}x{box[3]-box[1]}px -> {fn}")
    fn = os.path.join(args.out, f"{stem}-06-brushwork.jpg")
    cv2.imwrite(fn, detail_crop(art, args.size), [cv2.IMWRITE_JPEG_QUALITY, 92])
    report.append(f"brushwork -> {fn}")
    print("\n".join(report))

if __name__ == "__main__":
    main()
