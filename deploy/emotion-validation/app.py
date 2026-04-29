"""
Web Tool Validasi Emosi - Expert Review (Multi-Validator)
==========================================================
Tool interaktif untuk ahli psikologi memvalidasi label emosi otomatis.
Support 3 validator dengan hasil terpisah per validator.
Deployed on Streamlit Cloud.
"""

import streamlit as st
import json
import os
import base64
from pathlib import Path
from datetime import datetime
from collections import Counter

import requests

# ============== CONFIG ==============
BASE_DIR = Path("data")
RESULTS_DIR = BASE_DIR / "results"
ADMIN_CONFIG_PATH = BASE_DIR / "admin_config.json"
ADMIN_PASSWORD = "emoval2026"  # ganti sesuai kebutuhan

# GitHub auto-commit untuk persistent storage di Streamlit Cloud
# Default: repo deploy emotion-validation (terpisah dari repo riset utama).
# Bisa di-override via Streamlit secrets (GITHUB_REPO, GITHUB_BRANCH, GITHUB_PATH_PREFIX).
GITHUB_REPO_DEFAULT = "taufikhdyt01/emotion-validation"
GITHUB_BRANCH_DEFAULT = "main"
GITHUB_PATH_PREFIX_DEFAULT = "data/results"

DEFAULT_CONFIG = {
    "active_set": "1pct_frontonly",
    "active_label": "1% Front-Only (128 sample)",
}

VALIDATION_SETS = {
    "1% Front-Only (128 sample)": BASE_DIR / "sets" / "1pct_frontonly",
    "1% Front+Side (146 sample)": BASE_DIR / "sets" / "1pct",
    "5% Stratified (583 sample)": BASE_DIR / "sets" / "5pct",
    "10% Stratified (1,067 sample)": BASE_DIR / "sets" / "10pct",
}

EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
EMOTION_EMOJI = {
    "neutral": "😐", "happy": "😊", "sad": "😢", "angry": "😠",
    "fearful": "😨", "disgusted": "🤢", "surprised": "😲",
}
EMOTION_COLORS = {
    "neutral": "#6c757d", "happy": "#28a745", "sad": "#007bff",
    "angry": "#dc3545", "fearful": "#9b59b6", "disgusted": "#8e44ad",
    "surprised": "#ffc107",
}
# ====================================


@st.cache_data
def load_validation_data(set_path):
    """Load validation data from Excel."""
    import openpyxl
    wb = openpyxl.load_workbook(set_path / "validation_sheet.xlsx", read_only=True)
    ws = wb.active
    samples = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            break
        samples.append({
            "no": int(row[0]),
            "filename": row[1],
            "user_id": str(row[2]),
            "source": row[3],
            "angle": row[4],
            "auto_label": row[5],
            "confidence": float(row[6]) if row[6] else 0.0,
            "scores": {
                "neutral": float(row[7] or 0), "happy": float(row[8] or 0),
                "sad": float(row[9] or 0), "angry": float(row[10] or 0),
                "fearful": float(row[11] or 0), "disgusted": float(row[12] or 0),
                "surprised": float(row[13] or 0),
            },
        })
    wb.close()
    return samples


def get_results_path(validator_name):
    """Path hasil validasi per validator."""
    safe_name = validator_name.lower().replace(" ", "_")
    return RESULTS_DIR / f"results_{safe_name}.json"


def safe_validator_name(validator_name):
    return validator_name.lower().replace(" ", "_")


def _secret(key, default):
    """Read Streamlit secret with safe fallback."""
    try:
        val = st.secrets.get(key)
        return val if val else default
    except (FileNotFoundError, AttributeError, Exception):
        return default


def github_token():
    """Get GitHub PAT dari Streamlit secrets, return None kalau belum di-set."""
    return _secret("GITHUB_TOKEN", None)


def github_repo():
    return _secret("GITHUB_REPO", GITHUB_REPO_DEFAULT)


def github_branch():
    return _secret("GITHUB_BRANCH", GITHUB_BRANCH_DEFAULT)


def github_path_prefix():
    return _secret("GITHUB_PATH_PREFIX", GITHUB_PATH_PREFIX_DEFAULT)


def github_load(validator_name):
    """Fetch results dari GitHub via Contents API. Return dict atau None kalau gagal."""
    token = github_token()
    if not token:
        return None
    file_path = f"{github_path_prefix()}/results_{safe_validator_name(validator_name)}.json"
    api_url = f"https://api.github.com/repos/{github_repo()}/contents/{file_path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    try:
        r = requests.get(api_url, headers=headers,
                         params={"ref": github_branch()}, timeout=10)
        if r.status_code != 200:
            return None
        content_b64 = r.json().get("content", "")
        content = base64.b64decode(content_b64).decode("utf-8")
        return json.loads(content)
    except Exception:
        return None


def github_save(validator_name, results):
    """Save results ke GitHub via Contents API. Return (ok, message)."""
    token = github_token()
    if not token:
        return False, "GITHUB_TOKEN belum di-set di Streamlit secrets"
    file_path = f"{github_path_prefix()}/results_{safe_validator_name(validator_name)}.json"
    api_url = f"https://api.github.com/repos/{github_repo()}/contents/{file_path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    branch = github_branch()
    try:
        # Ambil SHA file existing kalau ada
        r = requests.get(api_url, headers=headers,
                         params={"ref": branch}, timeout=10)
        sha = r.json().get("sha") if r.status_code == 200 else None

        # Encode content + commit
        content_str = json.dumps(results, indent=2, ensure_ascii=False)
        content_b64 = base64.b64encode(content_str.encode("utf-8")).decode("ascii")
        n_validated = len([k for k, v in results.items() if v.get("expert_label")])
        payload = {
            "message": f"Update {validator_name} validation results ({n_validated} validated)",
            "content": content_b64,
            "branch": branch,
        }
        if sha:
            payload["sha"] = sha

        r = requests.put(api_url, headers=headers, json=payload, timeout=15)
        if r.status_code in (200, 201):
            return True, "Backup ke GitHub OK"
        return False, f"GitHub error {r.status_code}: {r.text[:120]}"
    except Exception as e:
        return False, f"Exception: {e}"


def load_results(validator_name):
    """Load hasil validasi: GitHub first (always-fresh), fallback lokal."""
    # Coba GitHub dulu (always-fresh, biar tidak stale dari container restart)
    remote = github_load(validator_name)
    if remote is not None:
        # Cache lokal untuk performa session
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = get_results_path(validator_name)
        try:
            with open(path, "w") as f:
                json.dump(remote, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
        return remote
    # Fallback lokal
    path = get_results_path(validator_name)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_results(validator_name, results):
    """Save lokal + auto-backup ke GitHub (persist saat Streamlit Cloud restart)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = get_results_path(validator_name)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Auto-commit ke GitHub (silent kalau token belum di-set)
    ok, msg = github_save(validator_name, results)
    if not ok:
        if "github_warned" not in st.session_state:
            st.warning(
                f"⚠️ Backup GitHub gagal: {msg}\n\n"
                "Data hanya tersimpan lokal di Streamlit container — bisa hilang saat app restart. "
                "Set GITHUB_TOKEN di Streamlit secrets supaya persistent."
            )
            st.session_state["github_warned"] = True


def find_image(images_dir, filename):
    """Cari gambar berdasarkan filename."""
    for f in images_dir.iterdir():
        if filename in f.name:
            return f
    return None


def cohens_kappa(y1, y2):
    """Hitung Cohen's Kappa (2 raters)."""
    import numpy as np
    n = len(y1)
    if n == 0:
        return 0.0
    labels = sorted(set(y1) | set(y2))
    n_labels = len(labels)
    idx = {l: i for i, l in enumerate(labels)}
    mat = np.zeros((n_labels, n_labels), dtype=int)
    for a, b in zip(y1, y2):
        mat[idx[a]][idx[b]] += 1
    po = np.trace(mat) / n
    pe = np.sum(mat.sum(axis=1) * mat.sum(axis=0)) / (n * n)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def fleiss_kappa(ratings_matrix):
    """Hitung Fleiss' Kappa (3+ raters).

    ratings_matrix: list of dicts, each row = {category: count_of_raters}
    e.g. [{"neutral": 2, "happy": 1}, {"neutral": 3}, ...]
    """
    import numpy as np
    all_cats = sorted(set(c for row in ratings_matrix for c in row))
    n_subjects = len(ratings_matrix)
    n_raters = sum(ratings_matrix[0].values()) if ratings_matrix else 0
    if n_subjects == 0 or n_raters <= 1:
        return 0.0

    mat = np.zeros((n_subjects, len(all_cats)))
    cat_idx = {c: i for i, c in enumerate(all_cats)}
    for i, row in enumerate(ratings_matrix):
        for cat, count in row.items():
            mat[i][cat_idx[cat]] = count

    # P_i per subject
    P_i = (np.sum(mat ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = np.mean(P_i)

    # p_j per category
    p_j = np.sum(mat, axis=0) / (n_subjects * n_raters)
    P_e_bar = np.sum(p_j ** 2)

    if P_e_bar == 1.0:
        return 1.0
    return (P_bar - P_e_bar) / (1 - P_e_bar)


def interpret_kappa(k):
    """Landis & Koch (1977) interpretation."""
    if k < 0: return "Poor"
    elif k < 0.20: return "Slight"
    elif k < 0.40: return "Fair"
    elif k < 0.60: return "Moderate"
    elif k < 0.80: return "Substantial"
    else: return "Almost Perfect"


def load_admin_config():
    """Load admin configuration."""
    if ADMIN_CONFIG_PATH.exists():
        with open(ADMIN_CONFIG_PATH, "r") as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()


def save_admin_config(config):
    """Save admin configuration."""
    os.makedirs(ADMIN_CONFIG_PATH.parent, exist_ok=True)
    with open(ADMIN_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def get_active_set():
    """Get the active validation set based on admin config."""
    config = load_admin_config()
    label = config.get("active_label", DEFAULT_CONFIG["active_label"])
    if label in VALIDATION_SETS:
        return label, VALIDATION_SETS[label]
    # Fallback to first available
    first_label = list(VALIDATION_SETS.keys())[0]
    return first_label, VALIDATION_SETS[first_label]


def show_admin():
    """Admin panel for managing validation settings."""
    st.title("⚙️ Admin Panel")

    # Password check
    if "admin_auth" not in st.session_state:
        st.session_state.admin_auth = False

    if not st.session_state.admin_auth:
        pwd = st.text_input("Password Admin", type="password")
        if st.button("Login Admin"):
            if pwd == ADMIN_PASSWORD:
                st.session_state.admin_auth = True
                st.rerun()
            else:
                st.error("Password salah!")
        return

    config = load_admin_config()
    st.success("✅ Login admin berhasil")

    st.divider()

    # ── 1. Set Active Validation Set ──
    st.subheader("1. Pilih Dataset Validasi Aktif")
    st.caption("Validator hanya bisa mengakses dataset yang aktif.")

    current_label = config.get("active_label", DEFAULT_CONFIG["active_label"])
    available = list(VALIDATION_SETS.keys())
    # Filter only sets that exist
    existing = [k for k in available if VALIDATION_SETS[k].exists()]
    not_existing = [k for k in available if not VALIDATION_SETS[k].exists()]

    if not_existing:
        st.warning(f"Dataset belum tersedia: {', '.join(not_existing)}")

    if existing:
        current_idx = existing.index(current_label) if current_label in existing else 0
        new_label = st.selectbox("Dataset aktif", existing, index=current_idx)

        if new_label != current_label:
            if st.button("💾 Simpan perubahan dataset"):
                config["active_label"] = new_label
                save_admin_config(config)
                st.success(f"Dataset aktif diubah ke: {new_label}")
                st.rerun()

        # Show dataset info
        set_path = VALIDATION_SETS[new_label]
        info_path = set_path / "validation_info.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            st.json(info)

    st.divider()

    # ── 2. Manage Validator Results ──
    st.subheader("2. Kelola Progress Validator")

    if RESULTS_DIR.exists():
        result_files = sorted(RESULTS_DIR.glob("results_*.json"))
        if result_files:
            for rf in result_files:
                vname = rf.stem.replace("results_", "")
                with open(rf) as fh:
                    data = json.load(fh)
                n_validated = len(data)

                col1, col2, col3 = st.columns([3, 1, 1])
                col1.write(f"**{vname}** — {n_validated} sampel divalidasi")
                with col2:
                    if st.button(f"📥 Download", key=f"dl_{vname}"):
                        st.download_button(
                            f"Download {vname}.json",
                            json.dumps(data, indent=2),
                            file_name=f"results_{vname}.json",
                            mime="application/json",
                            key=f"dl2_{vname}",
                        )
                with col3:
                    if st.button(f"🗑️ Hapus", key=f"del_{vname}", type="secondary"):
                        st.session_state[f"confirm_del_{vname}"] = True

                # Confirm delete
                if st.session_state.get(f"confirm_del_{vname}", False):
                    st.warning(f"⚠️ Yakin hapus semua progress **{vname}** ({n_validated} sampel)?")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button(f"Ya, hapus {vname}", key=f"yes_del_{vname}", type="primary"):
                            rf.unlink()
                            del st.session_state[f"confirm_del_{vname}"]
                            st.success(f"Progress {vname} dihapus!")
                            st.rerun()
                    with c2:
                        if st.button("Batal", key=f"no_del_{vname}"):
                            del st.session_state[f"confirm_del_{vname}"]
                            st.rerun()
        else:
            st.info("Belum ada hasil validasi.")
    else:
        st.info("Folder results belum dibuat.")

    st.divider()

    # ── 3. Reset All ──
    st.subheader("3. Reset Semua")
    st.caption("Hapus semua progress validasi dari semua validator.")

    if st.button("🔴 Reset Semua Progress", type="secondary"):
        st.session_state["confirm_reset_all"] = True

    if st.session_state.get("confirm_reset_all", False):
        st.error("⚠️ PERHATIAN: Ini akan menghapus SEMUA progress validasi!")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Ya, hapus semua", type="primary", key="yes_reset"):
                if RESULTS_DIR.exists():
                    for f in RESULTS_DIR.glob("results_*.json"):
                        f.unlink()
                del st.session_state["confirm_reset_all"]
                st.success("Semua progress dihapus!")
                st.rerun()
        with c2:
            if st.button("Batal", key="no_reset"):
                del st.session_state["confirm_reset_all"]
                st.rerun()

    st.divider()

    # ── Logout ──
    if st.button("🚪 Logout Admin"):
        st.session_state.admin_auth = False
        st.rerun()


def show_login():
    """Halaman login validator dengan pengantar penelitian."""
    st.title("🎭 Validasi Label Emosi - Expert Review")

    # ========== PENGANTAR PENELITIAN ==========
    st.divider()

    col_info, col_photo = st.columns([3, 1])
    with col_info:
        st.subheader("Tentang Penelitian")
        st.markdown("""
        **Judul Penelitian:**
        *Integrasi Multimodal Citra Wajah dan Facial Landmark untuk Pengenalan Emosi
        dalam Konteks Pembelajaran Pemrograman*

        **Peneliti:**
        1. Taufik Hidayat, S.Pd. (NIM: 256150117111007)
        2. Dr.Eng. Fitra Abdurrachman Bachtiar, S.T., M.Eng.
        3. Dr.Eng. Budi Darma Setiawan, S.Kom., M.Cs.

        Magister Ilmu Komputer, Fakultas Ilmu Komputer, Universitas Brawijaya

        **Deskripsi Singkat:**
        Penelitian ini mengembangkan sistem pengenalan ekspresi wajah multimodal
        untuk memantau emosi mahasiswa saat belajar pemrograman. Data dikumpulkan
        dari **37 mahasiswa** yang direkam melalui webcam selama sesi pembelajaran
        Block-Based Programming di platform e-block.
        """)

    st.divider()

    st.subheader("Tujuan Validasi")
    st.markdown("""
    Label emosi pada data saat ini diperoleh secara **otomatis** menggunakan model
    deep learning (Face API). Untuk memastikan **reliabilitas label**, diperlukan
    validasi dari ahli psikologi.

    **Tugas Anda:**
    1. Melihat gambar wajah mahasiswa (cropped, 224x224 piksel)
    2. Melihat label emosi yang diberikan sistem otomatis beserta confidence score
    3. Memutuskan apakah label tersebut **sudah benar** atau perlu **dikoreksi**

    **7 Kategori Emosi:**
    """)

    emo_cols = st.columns(7)
    for col, emo in zip(emo_cols, EMOTIONS):
        with col:
            st.markdown(f"<div style='text-align:center'>"
                        f"<span style='font-size:2em'>{EMOTION_EMOJI[emo]}</span><br>"
                        f"<small>{emo}</small></div>", unsafe_allow_html=True)

    st.markdown("""
    > **Catatan:** Data wajah yang ditampilkan telah mendapat *informed consent*
    > dari seluruh mahasiswa yang berpartisipasi. Data ini bersifat **rahasia**
    > dan hanya digunakan untuk keperluan penelitian.
    """)

    st.divider()

    # ========== MULAI VALIDASI ==========
    st.subheader("Mulai Validasi")

    # Get active set from admin config
    active_label, active_path = get_active_set()

    st.info(f"📁 Dataset aktif: **{active_label}**")

    # Show set info
    info_path = active_path / "validation_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        dist = info.get("distribution", {})
        with st.expander("Lihat distribusi emosi dalam dataset"):
            for emo in EMOTIONS:
                count = dist.get(emo, 0)
                st.write(f"  {EMOTION_EMOJI.get(emo, '')} {emo}: **{count}**")

    name = st.text_input("**Masukkan nama Anda:**", placeholder="Contoh: Dr. Siti Aminah")

    st.write("")
    if st.button("🚀 Mulai Validasi", type="primary", disabled=not name.strip(),
                 use_container_width=True):
        st.session_state.validator_name = name.strip()
        st.session_state.active_set = active_label
        st.session_state.set_path = str(active_path)
        st.rerun()

    # Show existing validators
    if RESULTS_DIR.exists():
        existing = [f.stem.replace("results_", "") for f in RESULTS_DIR.glob("results_*.json")]
        if existing:
            st.divider()
            st.caption(f"Validator yang sudah masuk: {', '.join(existing)}")

    # Footer
    st.divider()
    st.caption("Jika ada pertanyaan terkait validasi ini, silakan hubungi peneliti:")
    st.caption("Taufik Hidayat, S.Pd. — taufikhidayat@student.ub.ac.id")


def show_summary(samples):
    """Halaman ringkasan dengan perbandingan antar validator."""
    st.title("📊 Ringkasan Validasi")

    # Load all validator results
    validators = {}
    if RESULTS_DIR.exists():
        for f in sorted(RESULTS_DIR.glob("results_*.json")):
            vname = f.stem.replace("results_", "")
            with open(f) as fh:
                validators[vname] = json.load(fh)

    if not validators:
        st.info("Belum ada hasil validasi.")
        return

    total = len(samples)

    # Per-validator stats
    st.subheader("Progress per Validator")
    for vname, results in validators.items():
        validated = len(results)
        agreed = sum(1 for v in results.values() if v["expert_label"] == "agree")
        corrected = validated - agreed
        pct = validated / total * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"🧑‍⚕️ {vname}", f"{validated}/{total} ({pct:.0f}%)")
        col2.metric("Setuju", agreed)
        col3.metric("Dikoreksi", corrected)

        # Kappa vs auto-detection
        auto_labels = []
        expert_labels = []
        for k, v in results.items():
            auto_labels.append(v["auto_label"])
            expert_labels.append(v["auto_label"] if v["expert_label"] == "agree" else v["expert_label"])
        if auto_labels:
            k = cohens_kappa(auto_labels, expert_labels)
            col4.metric("Kappa vs Auto", f"{k:.3f} ({interpret_kappa(k)})")

    # Inter-rater agreement (if 2+ validators)
    vnames = list(validators.keys())
    if len(vnames) >= 2:
        st.divider()
        st.subheader("Inter-Rater Agreement")

        # Pairwise Cohen's Kappa
        st.markdown("**Pairwise Cohen's Kappa:**")
        for i in range(len(vnames)):
            for j in range(i + 1, len(vnames)):
                v1, v2 = vnames[i], vnames[j]
                r1, r2 = validators[v1], validators[v2]

                common_keys = set(r1.keys()) & set(r2.keys())
                if not common_keys:
                    st.warning(f"{v1} vs {v2}: Belum ada sample yang sama-sama divalidasi")
                    continue

                labels1 = []
                labels2 = []
                for k in common_keys:
                    l1 = r1[k]["auto_label"] if r1[k]["expert_label"] == "agree" else r1[k]["expert_label"]
                    l2 = r2[k]["auto_label"] if r2[k]["expert_label"] == "agree" else r2[k]["expert_label"]
                    labels1.append(l1)
                    labels2.append(l2)

                k = cohens_kappa(labels1, labels2)
                agree_count = sum(1 for a, b in zip(labels1, labels2) if a == b)

                st.metric(
                    f"🤝 {v1} vs {v2}",
                    f"Kappa: {k:.3f} ({interpret_kappa(k)})",
                    f"Agreement: {agree_count}/{len(common_keys)} ({agree_count/len(common_keys)*100:.0f}%) pada {len(common_keys)} sample"
                )

        # Fleiss' Kappa (3+ validators)
        if len(vnames) >= 3:
            st.divider()
            st.markdown("**Fleiss' Kappa (semua validator):**")

            # Find samples validated by all validators
            all_keys = set.intersection(*[set(validators[v].keys()) for v in vnames])
            if not all_keys:
                st.warning("Belum ada sample yang divalidasi oleh semua validator")
            else:
                ratings_matrix = []
                for key in sorted(all_keys):
                    row_labels = []
                    for v in vnames:
                        r = validators[v][key]
                        label = r["auto_label"] if r["expert_label"] == "agree" else r["expert_label"]
                        row_labels.append(label)
                    counts = Counter(row_labels)
                    ratings_matrix.append(dict(counts))

                fk = fleiss_kappa(ratings_matrix)
                target_met = fk >= 0.61

                col_fk1, col_fk2 = st.columns(2)
                col_fk1.metric(
                    f"Fleiss' Kappa ({len(vnames)} validator)",
                    f"{fk:.3f} ({interpret_kappa(fk)})",
                )
                col_fk2.metric(
                    "Target (Landis & Koch, 1977)",
                    "κ ≥ 0.61 (Substantial)",
                    "✅ Tercapai" if target_met else "❌ Belum tercapai",
                    delta_color="normal" if target_met else "inverse",
                )
                st.caption(f"Dihitung dari {len(all_keys)} sample yang divalidasi oleh semua {len(vnames)} validator")

    # Export
    st.divider()
    st.subheader("Export Hasil")

    for vname, results in validators.items():
        if not results:
            continue
        csv_lines = ["no,filename,user_id,auto_label,confidence,expert_label,notes,timestamp"]
        for s in samples:
            key = str(s["no"])
            if key in results:
                r = results[key]
                final = s["auto_label"] if r["expert_label"] == "agree" else r["expert_label"]
                notes = r.get("notes", "").replace(",", ";")
                csv_lines.append(
                    f'{s["no"]},{s["filename"]},{s["user_id"]},'
                    f'{s["auto_label"]},{s["confidence"]},{final},'
                    f'{notes},{r.get("timestamp", "")}'
                )
        csv_text = "\n".join(csv_lines)
        st.download_button(
            f"📥 Download hasil {vname} (CSV)",
            csv_text,
            file_name=f"validation_results_{vname}.csv",
            mime="text/csv",
            key=f"dl_{vname}",
        )


def show_validation(samples, set_path):
    """Halaman validasi utama."""
    validator_name = st.session_state.validator_name
    images_dir = set_path / "images"

    if "results" not in st.session_state:
        st.session_state.results = load_results(validator_name)
    results = st.session_state.results

    total = len(samples)
    validated = len(results)

    # ============ SIDEBAR ============
    with st.sidebar:
        st.title(f"🧑‍⚕️ {validator_name}")

        pct = validated / total * 100 if total > 0 else 0
        st.progress(pct / 100)
        st.write(f"**{validated} / {total}** ({pct:.1f}%)")

        if validated > 0:
            agreed = sum(1 for v in results.values() if v["expert_label"] == "agree")
            st.caption(f"Setuju: {agreed} | Dikoreksi: {validated - agreed}")

        st.divider()

        nav_mode = st.radio(
            "Tampilkan",
            ["Belum divalidasi", "Semua", "Yang dikoreksi"],
            index=0
        )
        filter_emo = st.selectbox("Filter emosi", ["Semua"] + EMOTIONS)

        st.divider()
        if st.button("🔄 Ganti Validator"):
            del st.session_state.validator_name
            if "results" in st.session_state:
                del st.session_state.results
            st.rerun()

        st.divider()
        st.caption("Petunjuk:")
        st.caption("1. Lihat gambar wajah")
        st.caption("2. Bandingkan dengan label otomatis")
        st.caption('3. Klik "Setuju" atau pilih emosi yang benar')

    # ============ FILTER ============
    if nav_mode == "Belum divalidasi":
        filtered = [s for s in samples if str(s["no"]) not in results]
    elif nav_mode == "Yang dikoreksi":
        filtered = [s for s in samples
                    if str(s["no"]) in results and results[str(s["no"])]["expert_label"] != "agree"]
    else:
        filtered = samples

    if filter_emo != "Semua":
        filtered = [s for s in filtered if s["auto_label"] == filter_emo]

    if not filtered:
        if nav_mode == "Belum divalidasi":
            st.balloons()
            st.success("🎉 Semua sample sudah divalidasi! Terima kasih!")
            st.info("Buka halaman **Ringkasan** di menu atas untuk melihat hasil.")
        else:
            st.info("Tidak ada data untuk filter ini.")
        return

    # ============ NAVIGATION ============
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0

    idx = st.session_state.current_idx
    if idx >= len(filtered):
        idx = 0
        st.session_state.current_idx = 0

    sample = filtered[idx]
    sample_key = str(sample["no"])

    col_info, col_nav = st.columns([3, 1])
    with col_info:
        st.markdown(f"### Sample #{sample['no']}  &nbsp; "
                    f"<small style='color:gray'>({idx + 1} / {len(filtered)})</small>",
                    unsafe_allow_html=True)
    with col_nav:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("⬅️", use_container_width=True, disabled=idx == 0):
                st.session_state.current_idx = idx - 1
                st.rerun()
        with c2:
            if st.button("➡️", use_container_width=True, disabled=idx >= len(filtered) - 1):
                st.session_state.current_idx = idx + 1
                st.rerun()

    st.divider()

    # ============ IMAGE + INFO ============
    col_img, col_detail = st.columns([1, 1])

    with col_img:
        img_path = find_image(images_dir, sample["filename"])
        if img_path:
            st.image(str(img_path), width=380)
        else:
            st.error(f"Gambar tidak ditemukan: {sample['filename']}")
        st.caption(f"User: {sample['user_id']} | Source: {sample['source']} | Angle: {sample['angle']}")

    with col_detail:
        auto_label = sample["auto_label"]
        confidence = sample["confidence"]
        emoji = EMOTION_EMOJI.get(auto_label, "")
        color = EMOTION_COLORS.get(auto_label, "#333")

        st.markdown(
            f"**Label Otomatis:** "
            f"<span style='background-color:{color}; color:white; padding:4px 12px; "
            f"border-radius:4px; font-size:1.2em'>{emoji} {auto_label.upper()}</span>"
            f"&nbsp;&nbsp; confidence: **{confidence:.4f}**",
            unsafe_allow_html=True
        )
        st.write("")
        st.write("**Skor per emosi:**")
        scores = sample["scores"]
        for emo in EMOTIONS:
            score = scores.get(emo, 0)
            is_max = (emo == auto_label)
            emoji_e = EMOTION_EMOJI.get(emo, "")
            label = f"**{emoji_e} {emo}**" if is_max else f"{emoji_e} {emo}"
            st.progress(min(float(score), 1.0), text=f"{label}: {score:.6f}")

    st.divider()

    # ============ EXPERT INPUT ============
    st.markdown("### Validasi")
    st.write("Apakah label otomatis sudah benar?")

    def handle_click(label):
        results[sample_key] = {
            "expert_label": label,
            "auto_label": auto_label,
            "notes": "",
            "timestamp": datetime.now().isoformat(),
            "validator": validator_name,
        }
        save_results(validator_name, results)
        st.session_state.results = results
        if idx < len(filtered) - 1:
            st.session_state.current_idx = idx + 1
        st.rerun()

    # Buttons
    other_emotions = [e for e in EMOTIONS if e != auto_label]
    cols = st.columns(1 + len(other_emotions))

    with cols[0]:
        if st.button("✅ Setuju", use_container_width=True, type="primary"):
            handle_click("agree")

    for i, emo in enumerate(other_emotions):
        with cols[i + 1]:
            emoji_e = EMOTION_EMOJI.get(emo, "")
            if st.button(f"{emoji_e} {emo}", use_container_width=True):
                handle_click(emo)

    # Notes
    existing = results.get(sample_key, {})
    notes = st.text_input("Catatan (opsional)", value=existing.get("notes", ""),
                          key=f"notes_{sample_key}")
    if notes and sample_key in results:
        results[sample_key]["notes"] = notes
        save_results(validator_name, results)

    # Status
    if sample_key in results:
        r = results[sample_key]
        if r["expert_label"] == "agree":
            st.success(f"✅ Divalidasi: Setuju dengan '{auto_label}'")
        else:
            st.warning(f"✏️ Dikoreksi: '{auto_label}' → '{r['expert_label']}'")

        if st.button("🗑️ Reset validasi ini", type="secondary"):
            del results[sample_key]
            save_results(validator_name, results)
            st.session_state.results = results
            st.rerun()


def main():
    st.set_page_config(
        page_title="Validasi Emosi - Expert Review",
        page_icon="🎭",
        layout="wide",
    )

    # Check for admin mode via query param or session
    query_params = st.query_params
    if query_params.get("admin") == "1" or st.session_state.get("show_admin", False):
        st.session_state.show_admin = True
        show_admin()
        return

    # Login check
    if "validator_name" not in st.session_state:
        show_login()
        return

    set_path = Path(st.session_state.set_path)
    samples = load_validation_data(set_path)

    # Top navigation
    tab_validate, tab_summary = st.tabs(["🎯 Validasi", "📊 Ringkasan"])

    with tab_validate:
        show_validation(samples, set_path)

    with tab_summary:
        show_summary(samples)


if __name__ == "__main__":
    main()
