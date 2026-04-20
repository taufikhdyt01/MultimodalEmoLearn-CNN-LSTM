"""
Update fig1_architecture_v2.html:
  1. Embed real face + landmark overlay + heatmap images (base64 data URIs)
     for sample_idx=2523 (happy, konsisten dengan pipeline figure)
  2. Apply muted pastel palette (saturation ~20% lower than current pastel)
  3. Replace SVG stylized icons #ico-face, #ico-landmark, #ico-heatmap
     with real image symbols

Output: overwrites docs/fig1_architecture_v2.html
"""
import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HTML_PATH = PROJECT_ROOT / 'docs' / 'architecture_v2.html'

SAMPLE_IDX = 967   # surprised — konsisten dengan pipeline figure
OUT_SIZE = 160     # larger thumbnail for clear real image (ratio 2x SVG viewBox)


def gaussian_heatmap(landmarks_136, img_size=224, sigma=3.0):
    y_grid, x_grid = np.ogrid[:img_size, :img_size]
    coords = landmarks_136.reshape(-1, 2)
    heatmap = np.zeros((img_size, img_size), dtype=np.float32)
    denom = 2.0 * sigma * sigma
    for x_norm, y_norm in coords:
        cx, cy = x_norm * img_size, y_norm * img_size
        g = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / denom)
        heatmap = np.maximum(heatmap, g.astype(np.float32))
    return heatmap


def png_to_base64(img_arr_uint8):
    """Convert uint8 image array [H,W,3] or [H,W] to base64 PNG data URI."""
    if img_arr_uint8.ndim == 2:
        img = Image.fromarray(img_arr_uint8, mode='L')
    else:
        img = Image.fromarray(img_arr_uint8, mode='RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def main():
    # Load dataset
    data_dir = PROJECT_ROOT / 'data' / 'dataset_frontonly_conf60'
    X = np.load(data_dir / 'X_train_images.npy')
    L = np.load(data_dir / 'X_train_landmarks.npy')

    idx = min(SAMPLE_IDX, len(X) - 1)
    face_224 = (np.clip(X[idx], 0, 1) * 255).astype(np.uint8)
    lm = L[idx]

    # 1. Plain face (resized to OUT_SIZE)
    face_img = Image.fromarray(face_224)
    face_resized = face_img.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
    face_b64 = png_to_base64(np.array(face_resized))

    # 2. Landmark dots ONLY (no face image — FCNN input is 136-dim coord vector)
    #    Background off-white, dots in muted sage green to match landmark stream
    lm_only_img = Image.new('RGB', (OUT_SIZE, OUT_SIZE), (245, 247, 248))
    draw = ImageDraw.Draw(lm_only_img)
    scale = OUT_SIZE / 224
    coords = lm.reshape(-1, 2) * 224 * scale
    sage = (90, 128, 85)       # #5A8055 — match landmark stream palette
    sage_dark = (73, 107, 71)  # #496B47
    for x, y in coords:
        r = 2.2
        draw.ellipse((x - r, y - r, x + r, y + r), fill=sage, outline=sage_dark)
    face_lm_b64 = png_to_base64(np.array(lm_only_img))

    # Coral for heatmap (kept consistent with pipeline figure)
    coral = (231, 111, 81)
    coral_dark = (168, 64, 31)

    # 3. Heatmap with coral gradient
    hmap = gaussian_heatmap(lm, img_size=224, sigma=3.0)
    hmap_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    # Coral colormap: white → coral
    # Build using (1-t)*white + t*coral_dark with intermediate
    for y in range(224):
        for x in range(224):
            t = hmap[y, x]
            if t < 1e-3:
                hmap_rgb[y, x] = [255, 255, 255]
            else:
                # interpolate white → coral
                r = int(255 * (1 - t) + coral[0] * t)
                g = int(255 * (1 - t) + coral[1] * t)
                b = int(255 * (1 - t) + coral[2] * t)
                hmap_rgb[y, x] = [r, g, b]
    hmap_img = Image.fromarray(hmap_rgb).resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
    hmap_b64 = png_to_base64(np.array(hmap_img))

    print(f'face png base64 length: {len(face_b64)}')
    print(f'face+lm png base64 length: {len(face_lm_b64)}')
    print(f'heatmap png base64 length: {len(hmap_b64)}')

    # 4. Read HTML template
    html = HTML_PATH.read_text(encoding='utf-8')

    # 5. Replace SVG stylized icons with real images inside existing symbols
    # Pattern: replace the content of symbol definitions for ico-face, ico-landmark, ico-heatmap

    def replace_symbol_content(html, symbol_id, new_inner):
        """Replace the inner content of <symbol id="...">...</symbol>."""
        import re
        pattern = rf'(<symbol id="{symbol_id}"[^>]*>)(.*?)(</symbol>)'
        return re.sub(pattern, rf'\1{new_inner}\3', html, flags=re.DOTALL)

    # New symbol content: use <image> tag pointing to base64 PNG
    face_symbol_content = f'<image href="{face_b64}" width="44" height="34" preserveAspectRatio="xMidYMid slice"/>'
    lm_symbol_content = f'<image href="{face_lm_b64}" width="44" height="34" preserveAspectRatio="xMidYMid slice"/>'
    heatmap_symbol_content = f'<image href="{hmap_b64}" width="44" height="34" preserveAspectRatio="xMidYMid slice"/>'

    html = replace_symbol_content(html, 'ico-face', face_symbol_content)
    html = replace_symbol_content(html, 'ico-landmark', lm_symbol_content)
    html = replace_symbol_content(html, 'ico-heatmap', heatmap_symbol_content)

    # 6. Apply muted pastel palette (saturation ~20% lower)
    #    Original → Muted replacements
    color_replacements = {
        # Image stream — muted steel blue
        '--img-bg1:    #DCEBFB;':    '--img-bg1:    #E2EAF2;',   # slightly less saturated
        '--img-bg2:    #BFD9F2;':    '--img-bg2:    #C7D6E5;',
        '--img-border: #3B6BA5;':    '--img-border: #4A6A8A;',   # deeper, less saturated
        '--img-text:   #1A2F4E;':    '--img-text:   #1F2E45;',
        '--img-accent: #2E5E96;':    '--img-accent: #3D5A7A;',

        # Landmark stream — muted sage green
        '--lmk-bg1:    #E2F3E4;':    '--lmk-bg1:    #E5EDE4;',
        '--lmk-bg2:    #C7E6CC;':    '--lmk-bg2:    #CCDBCB;',
        '--lmk-border: #4A8F44;':    '--lmk-border: #5A8055;',
        '--lmk-text:   #16381C;':    '--lmk-text:   #1F3322;',
        '--lmk-accent: #357A33;':    '--lmk-accent: #496B47;',

        # Fusion — muted terracotta
        '--fuse-bg1:   #FCE8D1;':    '--fuse-bg1:   #F2E5D2;',
        '--fuse-bg2:   #F8D4A8;':    '--fuse-bg2:   #E5D1B1;',
        '--fuse-border:#D46A25;':    '--fuse-border:#A87143;',
        '--fuse-text:  #5C2D0D;':    '--fuse-text:  #4A2E14;',
        '--fuse-accent:#B85A1A;':    '--fuse-accent:#8E5C34;',
    }

    for old, new in color_replacements.items():
        html = html.replace(old, new)

    # 7. Save updated HTML
    HTML_PATH.write_text(html, encoding='utf-8')
    print(f'\nSaved: {HTML_PATH}')
    print(f'Sample: idx={idx}')


if __name__ == '__main__':
    main()
