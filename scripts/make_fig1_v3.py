"""
Generate fig1_architecture_v3.html: compact 5-column taxonomy layout.

Layout: 5 columns side-by-side, each = 1 architecture (vertical flow).
Width: ~7.16 in (IEEE 2-column span), height: ~4-5 in.

Sample: idx=967 (surprised) — konsisten dengan pipeline figure.
Palette: muted pastel (same as v2).
"""
import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / 'docs' / 'fig1_architecture_v3.html'

SAMPLE_IDX = 967   # surprised (matches pipeline figure)
OUT_SIZE = 160


def gaussian_heatmap(lm_136, img_size=224, sigma=3.0):
    y_g, x_g = np.ogrid[:img_size, :img_size]
    coords = lm_136.reshape(-1, 2)
    h = np.zeros((img_size, img_size), dtype=np.float32)
    denom = 2.0 * sigma * sigma
    for x_n, y_n in coords:
        cx, cy = x_n * img_size, y_n * img_size
        g = np.exp(-((x_g - cx) ** 2 + (y_g - cy) ** 2) / denom)
        h = np.maximum(h, g.astype(np.float32))
    return h


def to_b64(img_arr):
    img = Image.fromarray(img_arr, mode='RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def build_icons():
    data_dir = PROJECT_ROOT / 'data' / 'dataset_frontonly_conf60'
    X = np.load(data_dir / 'X_train_images.npy')
    L = np.load(data_dir / 'X_train_landmarks.npy')
    idx = min(SAMPLE_IDX, len(X) - 1)

    face_224 = (np.clip(X[idx], 0, 1) * 255).astype(np.uint8)
    lm = L[idx]

    # Face
    face_img = Image.fromarray(face_224).resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
    face_b64 = to_b64(np.array(face_img))

    # Landmark dots only (sage green, off-white bg)
    lm_img = Image.new('RGB', (OUT_SIZE, OUT_SIZE), (245, 247, 248))
    draw = ImageDraw.Draw(lm_img)
    scale = OUT_SIZE / 224
    coords = lm.reshape(-1, 2) * 224 * scale
    sage = (90, 128, 85)
    sage_dark = (73, 107, 71)
    for x, y in coords:
        r = 2.2
        draw.ellipse((x - r, y - r, x + r, y + r), fill=sage, outline=sage_dark)
    lm_b64 = to_b64(np.array(lm_img))

    # Heatmap (coral gradient)
    hmap = gaussian_heatmap(lm, img_size=224, sigma=3.0)
    hmap_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    coral = (231, 111, 81)
    for y in range(224):
        for x in range(224):
            t = hmap[y, x]
            if t < 1e-3:
                hmap_rgb[y, x] = [255, 255, 255]
            else:
                hmap_rgb[y, x] = [
                    int(255 * (1 - t) + coral[0] * t),
                    int(255 * (1 - t) + coral[1] * t),
                    int(255 * (1 - t) + coral[2] * t),
                ]
    hmap_img = Image.fromarray(hmap_rgb).resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
    hmap_b64 = to_b64(np.array(hmap_img))

    return face_b64, lm_b64, hmap_b64


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Figure 1 — Five Architectures (Compact Taxonomy)</title>
<style>
  :root {{
    --img-bg:    #E2EAF2;   --img-border:  #4A6A8A;   --img-text:   #1F2E45;
    --lmk-bg:    #E5EDE4;   --lmk-border:  #5A8055;   --lmk-text:   #1F3322;
    --fuse-bg:   #F2E5D2;   --fuse-border: #A87143;   --fuse-text:  #4A2E14;
    --out-bg:    #ECEEF2;   --out-border:  #6A7280;   --out-text:   #1F2937;
    --arrow:     #6B7280;
    --ink:       #111827;
    --muted:     #6B7280;
    --panel:     #FFFFFF;
    --panel-border: #E3E7EE;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{
    margin: 0; padding: 0;
    font-family: "Inter", -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    color: var(--ink);
    background: #FFFFFF;
    line-height: 1.4;
  }}
  body {{ padding: 16px; }}

  .container {{ max-width: 1120px; margin: 0 auto; }}

  .figure-title {{
    text-align: center;
    margin-bottom: 18px;
    font-size: 12px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 600;
  }}
  .figure-title .big {{
    display: block;
    font-size: 20px;
    color: var(--ink);
    text-transform: none;
    letter-spacing: -0.01em;
    font-weight: 700;
    margin-top: 4px;
  }}
  .figure-title .sub {{
    font-size: 12px;
    text-transform: none;
    letter-spacing: 0;
    font-weight: 400;
    color: var(--muted);
    max-width: 780px;
    margin: 4px auto 0;
  }}

  .arch-grid {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
  }}

  .arch-col {{
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 14px 10px 12px;
    border-radius: 10px;
    border: 1px solid var(--panel-border);
    background: #FCFCFD;
    gap: 6px;
  }}

  .col-head {{
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 4px;
  }}
  .badge {{
    width: 22px; height: 22px;
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    background: var(--col-accent, #4A6A8A);
    color: #fff;
    font-weight: 700;
    font-size: 11px;
  }}
  .col-title {{
    font-size: 13px;
    font-weight: 700;
    color: var(--ink);
    letter-spacing: -0.005em;
  }}
  .col-sub {{
    font-size: 10px;
    color: var(--muted);
    text-align: center;
    margin-bottom: 4px;
  }}

  .box {{
    width: 100%;
    padding: 6px 8px;
    border-radius: 7px;
    border: 1.2px solid;
    text-align: center;
    font-size: 10px;
    line-height: 1.3;
  }}
  .box .title {{ display: block; font-weight: 700; font-size: 10px; }}
  .box .sub {{ display: block; font-size: 9px; opacity: 0.78; margin-top: 1px; }}
  .box img.icon {{
    display: block;
    width: 42px; height: 42px;
    margin: 0 auto 4px;
    border-radius: 4px;
  }}
  .box .mini-ico {{
    display: block;
    margin: 0 auto 3px;
  }}

  .box.img {{ background: var(--img-bg); border-color: var(--img-border); color: var(--img-text); }}
  .box.lmk {{ background: var(--lmk-bg); border-color: var(--lmk-border); color: var(--lmk-text); }}
  .box.fuse {{ background: var(--fuse-bg); border-color: var(--fuse-border); color: var(--fuse-text); }}
  .box.out {{ background: var(--out-bg); border-color: var(--out-border); color: var(--out-text); }}

  .arrow-v {{
    width: 2px; height: 14px;
    margin: 0 auto;
    position: relative;
    background: linear-gradient(180deg, #D1D5DB 0%, var(--arrow) 60%, var(--arrow) 100%);
  }}
  .arrow-v::after {{
    content: '';
    position: absolute;
    bottom: -1px; left: 50%;
    transform: translateX(-50%);
    width: 0; height: 0;
    border-top: 8px solid var(--arrow);
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
  }}

  /* Two-stream row (for intermediate & late) */
  .two-stream {{
    display: flex;
    gap: 6px;
    width: 100%;
    align-items: stretch;
  }}
  .two-stream .stream {{
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
  }}

  .merge-arrow {{
    display: block;
    width: 100%;
    margin: 2px auto;
    line-height: 0;
  }}
  .merge-arrow svg {{ display: block; margin: 0 auto; width: 92%; height: 28px; }}

  .col-a {{ --col-accent: #4A6A8A; }}
  .col-b {{ --col-accent: #5A8055; }}
  .col-c {{ --col-accent: #A87143; }}
  .col-d {{ --col-accent: #6F5A8F; }}
  .col-e {{ --col-accent: #9E5070; }}

  .fusion-caption {{
    font-size: 10px;
    color: var(--muted);
    margin-top: 10px;
    text-align: center;
    font-style: italic;
  }}

  @media print {{
    body {{ background: #fff; padding: 8px; }}
    .arch-grid {{ box-shadow: none; }}
  }}
</style>
</head>
<body>

<!-- Reusable SVG icons -->
<svg style="display:none">
  <symbol id="ico-cnn" viewBox="0 0 54 34">
    <rect x="2"  y="6"  width="10" height="22" rx="2" fill="#C7D6E5" stroke="#4A6A8A" stroke-width="1"/>
    <rect x="15" y="8"  width="10" height="18" rx="2" fill="#C7D6E5" stroke="#4A6A8A" stroke-width="1"/>
    <rect x="28" y="10" width="10" height="14" rx="2" fill="#C7D6E5" stroke="#4A6A8A" stroke-width="1"/>
    <rect x="41" y="12" width="10" height="10" rx="2" fill="#C7D6E5" stroke="#4A6A8A" stroke-width="1"/>
  </symbol>
  <symbol id="ico-fcnn" viewBox="0 0 54 34">
    <g fill="#5A8055">
      <circle cx="6"  cy="8" r="1.6"/><circle cx="6"  cy="17" r="1.6"/><circle cx="6"  cy="26" r="1.6"/>
      <circle cx="18" cy="5" r="1.6"/><circle cx="18" cy="14" r="1.6"/><circle cx="18" cy="23" r="1.6"/><circle cx="18" cy="31" r="1.6"/>
      <circle cx="30" cy="8" r="1.6"/><circle cx="30" cy="17" r="1.6"/><circle cx="30" cy="26" r="1.6"/>
      <circle cx="42" cy="14" r="1.6"/><circle cx="42" cy="22" r="1.6"/>
    </g>
    <g stroke="#5A8055" stroke-width="0.5" opacity="0.45" fill="none">
      <line x1="6" y1="8"  x2="18" y2="5"/>  <line x1="6" y1="8"  x2="18" y2="14"/>
      <line x1="6" y1="17" x2="18" y2="14"/> <line x1="6" y1="17" x2="18" y2="23"/>
      <line x1="6" y1="26" x2="18" y2="23"/> <line x1="6" y1="26" x2="18" y2="31"/>
      <line x1="18" y1="14" x2="30" y2="8"/> <line x1="18" y1="23" x2="30" y2="17"/>
      <line x1="30" y1="8"  x2="42" y2="14"/><line x1="30" y1="17" x2="42" y2="14"/>
      <line x1="30" y1="17" x2="42" y2="22"/><line x1="30" y1="26" x2="42" y2="22"/>
    </g>
  </symbol>
  <symbol id="ico-softmax" viewBox="0 0 44 34">
    <rect x="4"  y="22" width="5" height="9"  fill="#6A7280"/>
    <rect x="12" y="14" width="5" height="17" fill="#6A7280"/>
    <rect x="20" y="8"  width="5" height="23" fill="#374151"/>
    <rect x="28" y="18" width="5" height="13" fill="#6A7280"/>
    <rect x="36" y="25" width="5" height="6"  fill="#6A7280"/>
  </symbol>
</svg>

<div class="container">

  <div class="arch-grid">

    <!-- (a) CNN -->
    <div class="arch-col col-a">
      <div class="col-head"><div class="badge">a</div><div class="col-title">CNN</div></div>
      <div class="col-sub">Image only</div>
      <div class="box img">
        <img class="icon" src="{face_b64}" alt="face"/>
        <span class="title">Image</span><span class="sub">224 × 224 × 3</span>
      </div>
      <div class="arrow-v"></div>
      <div class="box img">
        <svg class="mini-ico" width="42" height="26"><use href="#ico-cnn"/></svg>
        <span class="title">Conv blocks</span><span class="sub">scratch or ResNet-18</span>
      </div>
      <div class="arrow-v"></div>
      <div class="box fuse">
        <span class="title">FC → K</span>
      </div>
      <div class="arrow-v"></div>
      <div class="box out">
        <svg class="mini-ico" width="36" height="26"><use href="#ico-softmax"/></svg>
        <span class="title">Softmax</span>
      </div>
    </div>

    <!-- (b) FCNN -->
    <div class="arch-col col-b">
      <div class="col-head"><div class="badge">b</div><div class="col-title">FCNN</div></div>
      <div class="col-sub">Landmark only</div>
      <div class="box lmk">
        <img class="icon" src="{lm_b64}" alt="landmark"/>
        <span class="title">Landmark</span><span class="sub">136-d (68 × 2)</span>
      </div>
      <div class="arrow-v"></div>
      <div class="box lmk">
        <svg class="mini-ico" width="42" height="26"><use href="#ico-fcnn"/></svg>
        <span class="title">MLP</span>
      </div>
      <div class="arrow-v"></div>
      <div class="box fuse">
        <span class="title">FC → K</span>
      </div>
      <div class="arrow-v"></div>
      <div class="box out">
        <svg class="mini-ico" width="36" height="26"><use href="#ico-softmax"/></svg>
        <span class="title">Softmax</span>
      </div>
    </div>

    <!-- (c) Early Fusion -->
    <div class="arch-col col-c">
      <div class="col-head"><div class="badge">c</div><div class="col-title">Early Fusion</div></div>
      <div class="col-sub">Input-level</div>
      <div class="two-stream">
        <div class="stream">
          <div class="box img" style="padding:4px 6px;">
            <img class="icon" src="{face_b64}" alt="img" style="width:34px;height:34px;"/>
            <span class="title" style="font-size:9px;">Image</span>
          </div>
        </div>
        <div class="stream">
          <div class="box fuse" style="padding:4px 6px;">
            <img class="icon" src="{hmap_b64}" alt="heatmap" style="width:34px;height:34px;"/>
            <span class="title" style="font-size:9px;">Heatmap</span>
          </div>
        </div>
      </div>
      <div class="merge-arrow">
        <svg viewBox="0 0 70 38" preserveAspectRatio="xMidYMid meet">
          <path d="M 18 0 C 18 18 35 18 35 32" stroke="#6B7280" stroke-width="1.8" fill="none"/>
          <path d="M 52 0 C 52 18 35 18 35 32" stroke="#6B7280" stroke-width="1.8" fill="none"/>
          <polygon points="35,38 31,30 39,30" fill="#6B7280"/>
        </svg>
      </div>
      <div class="box fuse">
        <span class="title">concat</span><span class="sub">224×224×4</span>
      </div>
      <div class="arrow-v"></div>
      <div class="box fuse">
        <svg class="mini-ico" width="42" height="26"><use href="#ico-cnn"/></svg>
        <span class="title">Conv (4-ch)</span><span class="sub">scratch / ResNet-18</span>
      </div>
      <div class="arrow-v"></div>
      <div class="box fuse">
        <span class="title">FC → K</span>
      </div>
      <div class="arrow-v"></div>
      <div class="box out">
        <svg class="mini-ico" width="36" height="26"><use href="#ico-softmax"/></svg>
        <span class="title">Softmax</span>
      </div>
    </div>

    <!-- (d) Intermediate Fusion -->
    <div class="arch-col col-d">
      <div class="col-head"><div class="badge">d</div><div class="col-title">Intermediate</div></div>
      <div class="col-sub">Feature-level</div>
      <div class="two-stream">
        <div class="stream">
          <div class="box img" style="padding:4px 6px;">
            <img class="icon" src="{face_b64}" alt="img" style="width:34px;height:34px;"/>
            <span class="title" style="font-size:9px;">Image</span>
          </div>
          <div class="arrow-v"></div>
          <div class="box img" style="padding:4px 6px;">
            <svg class="mini-ico" width="36" height="22"><use href="#ico-cnn"/></svg>
            <span class="title" style="font-size:9px;">CNN</span>
          </div>
          <div class="arrow-v"></div>
          <div class="box img" style="padding:4px 6px;">
            <span class="title" style="font-size:9px;">256-d</span>
          </div>
        </div>
        <div class="stream">
          <div class="box lmk" style="padding:4px 6px;">
            <img class="icon" src="{lm_b64}" alt="lm" style="width:34px;height:34px;"/>
            <span class="title" style="font-size:9px;">Landmark</span>
          </div>
          <div class="arrow-v"></div>
          <div class="box lmk" style="padding:4px 6px;">
            <svg class="mini-ico" width="36" height="22"><use href="#ico-fcnn"/></svg>
            <span class="title" style="font-size:9px;">FCNN</span>
          </div>
          <div class="arrow-v"></div>
          <div class="box lmk" style="padding:4px 6px;">
            <span class="title" style="font-size:9px;">128-d</span>
          </div>
        </div>
      </div>
      <div class="merge-arrow">
        <svg viewBox="0 0 70 38" preserveAspectRatio="xMidYMid meet">
          <path d="M 18 0 C 18 18 35 18 35 32" stroke="#6B7280" stroke-width="1.8" fill="none"/>
          <path d="M 52 0 C 52 18 35 18 35 32" stroke="#6B7280" stroke-width="1.8" fill="none"/>
          <polygon points="35,38 31,30 39,30" fill="#6B7280"/>
        </svg>
      </div>
      <div class="box fuse">
        <span class="title">concat</span><span class="sub">384-d</span>
      </div>
      <div class="arrow-v"></div>
      <div class="box fuse">
        <span class="title">FC head → K</span>
      </div>
      <div class="arrow-v"></div>
      <div class="box out">
        <svg class="mini-ico" width="36" height="26"><use href="#ico-softmax"/></svg>
        <span class="title">Softmax</span>
      </div>
    </div>

    <!-- (e) Late Fusion -->
    <div class="arch-col col-e">
      <div class="col-head"><div class="badge">e</div><div class="col-title">Late Fusion</div></div>
      <div class="col-sub">Decision-level</div>
      <div class="two-stream">
        <div class="stream">
          <div class="box img" style="padding:4px 6px;">
            <img class="icon" src="{face_b64}" alt="img" style="width:34px;height:34px;"/>
            <span class="title" style="font-size:9px;">Image</span>
          </div>
          <div class="arrow-v"></div>
          <div class="box img" style="padding:4px 6px;">
            <svg class="mini-ico" width="36" height="22"><use href="#ico-cnn"/></svg>
            <span class="title" style="font-size:9px;">CNN</span>
          </div>
          <div class="arrow-v"></div>
          <div class="box img" style="padding:4px 6px;">
            <svg class="mini-ico" width="30" height="22"><use href="#ico-softmax"/></svg>
            <span class="title" style="font-size:9px;">p<sub>img</sub></span>
          </div>
        </div>
        <div class="stream">
          <div class="box lmk" style="padding:4px 6px;">
            <img class="icon" src="{lm_b64}" alt="lm" style="width:34px;height:34px;"/>
            <span class="title" style="font-size:9px;">Landmark</span>
          </div>
          <div class="arrow-v"></div>
          <div class="box lmk" style="padding:4px 6px;">
            <svg class="mini-ico" width="36" height="22"><use href="#ico-fcnn"/></svg>
            <span class="title" style="font-size:9px;">FCNN</span>
          </div>
          <div class="arrow-v"></div>
          <div class="box lmk" style="padding:4px 6px;">
            <svg class="mini-ico" width="30" height="22"><use href="#ico-softmax"/></svg>
            <span class="title" style="font-size:9px;">p<sub>lmk</sub></span>
          </div>
        </div>
      </div>
      <div class="merge-arrow">
        <svg viewBox="0 0 70 38" preserveAspectRatio="xMidYMid meet">
          <path d="M 18 0 C 18 18 35 18 35 32" stroke="#6B7280" stroke-width="1.8" fill="none"/>
          <path d="M 52 0 C 52 18 35 18 35 32" stroke="#6B7280" stroke-width="1.8" fill="none"/>
          <polygon points="35,38 31,30 39,30" fill="#6B7280"/>
        </svg>
      </div>
      <div class="box fuse">
        <span class="title" style="font-family:'Cambria Math',serif;font-style:italic;">w · p<sub>img</sub> + (1 − w) · p<sub>lmk</sub></span>
      </div>
      <div class="arrow-v"></div>
      <div class="box out">
        <span class="title">argmax</span>
      </div>
    </div>

  </div>

</div>
</body>
</html>
"""


def main():
    face_b64, lm_b64, hmap_b64 = build_icons()
    html = HTML_TEMPLATE.format(
        face_b64=face_b64,
        lm_b64=lm_b64,
        hmap_b64=hmap_b64,
    )
    OUT_PATH.write_text(html, encoding='utf-8')
    print(f'Saved: {OUT_PATH}')
    print(f'Sample: idx={SAMPLE_IDX}')


if __name__ == '__main__':
    main()
