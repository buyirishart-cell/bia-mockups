#!/usr/bin/env python3
"""Buy Irish Art template mockup compositor.

Places the REAL artwork file, pixel-true, into a blank-frame room template.
The room's own light and shadow (sampled from the template's blank grey
panel) is laid over the art, so window shadows cross the glass as in a photo.
The artwork is never cropped or redrawn: if its proportions differ from the
template's panel, the frame itself is resized (expanded) to fit the art.

Usage:
  python3 bia_mockup.py ARTWORK.jpg --templates templates.json --out OUTDIR [--only id1,id2]
"""
import argparse, json, os
import cv2
import numpy as np

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

def shading_map(panel_bgr):
    """Relative light on the blank panel: 1.0 = fully lit, <1 = shadow."""
    L = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    L = cv2.GaussianBlur(L, (0, 0), 1.2)
    ref = np.percentile(L, 97)
    s = L / max(ref, 1)
    return np.clip(s, 0.15, 1.08)

def place(room, tpl, art):
    H, W = room.shape[:2]
    ox0, oy0, ox1, oy1 = tpl["outer"]          # frame outer edge
    px0, py0, px1, py1 = tpl["panel"]          # blank panel (art window)
    bl, bt, br, bb = px0 - ox0, py0 - oy0, ox1 - px1, oy1 - py1   # frame border widths
    pw, ph = px1 - px0, py1 - py0
    ar = art.shape[1] / art.shape[0]
    # expand the panel (never shrink) so it matches the artwork's proportions
    if ar > pw / ph:
        npw, nph = int(round(ph * ar)), ph
    else:
        npw, nph = pw, int(round(pw / ar))
    grow = max(npw / pw, nph / ph)
    nW, nH = npw + bl + br, nph + bt + bb
    # anchor: leaning frames keep their bottom on the floor/ledge, hung frames stay centred
    cx = (ox0 + ox1) / 2
    nx0 = int(round(cx - nW / 2))
    if tpl.get("anchor") == "bottom":
        ny0 = oy1 - nH
    else:
        ny0 = int(round((oy0 + oy1) / 2 - nH / 2))
    out = room.copy()
    if grow > 1.001:
        # rebuild the frame at the new size from its own edges (9-slice), so the
        # moulding keeps its width and only gets longer
        frame = room[oy0:oy1, ox0:ox1]
        fh, fw = frame.shape[:2]
        new = np.zeros((nH, nW, 3), np.uint8)
        xs = [(0, bl, 0, bl), (bl, fw - br, bl, nW - br), (fw - br, fw, nW - br, nW)]
        ys = [(0, bt, 0, bt), (bt, fh - bb, bt, nH - bb), (fh - bb, fh, nH - bb, nH)]
        for (sy0, sy1, dy0, dy1) in ys:
            for (sx0, sx1, dx0, dx1) in xs:
                piece = frame[sy0:sy1, sx0:sx1]
                new[dy0:dy1, dx0:dx1] = cv2.resize(piece, (dx1 - dx0, dy1 - dy0), interpolation=cv2.INTER_CUBIC)
        # soft contact shadow so the bigger frame still sits on the wall
        sh = np.zeros((H, W), np.float32)
        cv2.rectangle(sh, (nx0 + 6, ny0 + 10), (nx0 + nW + 6, ny0 + nH + 10), 1, -1)
        sh = cv2.GaussianBlur(sh, (0, 0), 9) * 0.22
        out = (out.astype(np.float32) * (1 - sh[..., None])).astype(np.uint8)
        yy0, xx0 = max(ny0, 0), max(nx0, 0)
        yy1, xx1 = min(ny0 + nH, H), min(nx0 + nW, W)
        out[yy0:yy1, xx0:xx1] = new[yy0 - ny0:yy1 - ny0, xx0 - nx0:xx1 - nx0]
        panel_light = cv2.resize(room[py0:py1, px0:px1], (npw, nph), interpolation=cv2.INTER_CUBIC)
        qx0, qy0 = nx0 + bl, ny0 + bt
    else:
        panel_light = room[py0:py1, px0:px1]
        qx0, qy0 = px0, py0
    # artwork: scaled with high-quality resampling, no crop, no redraw
    a = cv2.resize(art, (npw, nph), interpolation=cv2.INTER_AREA)
    s = shading_map(panel_light)
    # warm/cool tint of the room light, taken from the panel itself
    pl = panel_light.reshape(-1, 3).astype(np.float32)
    bright = pl[np.argsort(pl.sum(1))[-max(50, len(pl) // 50):]].mean(0)
    tint = bright / bright.mean()
    tint = 1 + (tint - 1) * 0.5
    lit = srgb_to_lin(a.astype(np.float32)) * s[..., None] * tint[None, None, :]
    res = lin_to_srgb(lit).astype(np.uint8)
    # faint glass sheen: lift the lit regions very slightly
    sheen = np.clip((s - 0.9) * 0.25, 0, 0.03)[..., None] * 255
    res = np.clip(res.astype(np.float32) + sheen, 0, 255).astype(np.uint8)
    y0, x0 = qy0, qx0
    y1, x1 = y0 + nph, x0 + npw
    cy0, cx0, cy1, cx1 = max(y0, 0), max(x0, 0), min(y1, H), min(x1, W)
    out[cy0:cy1, cx0:cx1] = res[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    # inner-edge shadow from the moulding onto the print (1-3px), as in a real frame
    edge = np.ones((nph, npw), np.float32)
    d = max(2, int(min(npw, nph) * 0.006))
    edge[:d, :] *= np.linspace(0.75, 1, d)[:, None]
    edge[:, :d] *= np.linspace(0.8, 1, d)[None, :]
    reg = out[cy0:cy1, cx0:cx1].astype(np.float32)
    reg *= edge[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0][..., None]
    out[cy0:cy1, cx0:cx1] = reg.astype(np.uint8)
    return out, grow

def detail_crop(art, size=2048, frac=0.42):
    """Brushwork close-up: a true crop of the master file, no AI."""
    h, w = art.shape[:2]
    side = int(min(h, w) * frac)
    # pick the most textured window (highest local detail) away from the edges
    g = cv2.cvtColor(art, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    best, bxy = -1, (0, 0)
    for fy in np.linspace(0.15, 0.85, 8):
        for fx in np.linspace(0.15, 0.85, 8):
            x = int(fx * w - side / 2); y = int(fy * h - side / 2)
            x = min(max(x, 0), w - side); y = min(max(y, 0), h - side)
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
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--name", default="", help="output file stem, e.g. the Shopify product handle")
    ap.add_argument("--max-grow", type=float, default=1.2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    art = trim_border(cv2.imread(args.artwork))
    ar = art.shape[1] / art.shape[0]
    cfg = json.load(open(args.templates))
    base = os.path.dirname(os.path.abspath(args.templates))
    only = set(filter(None, args.only.split(",")))
    stem = args.name or os.path.splitext(os.path.basename(args.artwork))[0]
    # per room, choose the template whose panel needs the least frame resizing
    rooms = {}
    for t in cfg["templates"]:
        if only and t["room"] not in only:
            continue
        p = t["panel"]; pr = (p[2] - p[0]) / (p[3] - p[1])
        # the frame only ever grows; rank by growth relative to this template's own limit
        cost = max(ar / pr, pr / ar) / t.get("max_grow", args.max_grow)
        if t["room"] not in rooms or cost < rooms[t["room"]][0]:
            rooms[t["room"]] = (cost, t)
    report = []
    for room_id, (cost, t) in sorted(rooms.items(), key=lambda kv: kv[1][1].get("order", 99)):
        room = cv2.imread(os.path.join(base, t["file"]))
        # work at output resolution so the artwork keeps its full detail
        f = args.size / room.shape[1]
        room = cv2.resize(room, (args.size, args.size), interpolation=cv2.INTER_LANCZOS4)
        tt = dict(t)
        ov = 8  # overscan: art tucks just under the moulding so no grey sliver shows
        p = [int(round(v * f)) for v in t["panel"]]
        tt["panel"] = [p[0] - ov, p[1] - ov, p[2] + ov, p[3] + ov]
        o = [int(round(v * f)) for v in t["outer"]]
        pp = tt["panel"]  # keep at least a 2px moulding on every side
        tt["outer"] = [min(o[0], pp[0] - 2), min(o[1], pp[1] - 2), max(o[2], pp[2] + 2), max(o[3], pp[3] + 2)]
        img, grow = place(room, tt, art)
        if grow > t.get("max_grow", args.max_grow):
            report.append(f"{room_id}: SKIPPED, template {t['id']} would need the frame resized {grow:.2f}x (needs a closer-ratio template)")
            continue
        fn = os.path.join(args.out, f"{stem}-{t.get('order', 9):02d}-{room_id}.jpg")
        cv2.imwrite(fn, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        report.append(f"{room_id}: {t['id']} (frame resized {grow:.2f}x) -> {fn}")
    fn = os.path.join(args.out, f"{stem}-06-brushwork.jpg")
    cv2.imwrite(fn, detail_crop(art, args.size), [cv2.IMWRITE_JPEG_QUALITY, 92])
    report.append(f"brushwork-detail -> {fn}")
    print("\n".join(report))

if __name__ == "__main__":
    main()
