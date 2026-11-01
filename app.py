import io
import json
import os
import sqlite3
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for, session,
    flash, send_file, abort
)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from reportlab.platypus import Table
from reportlab.platypus import TableStyle
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "app.db")

app = Flask(__name__)
app.secret_key = os.environ.get("FAHP_SECRET_KEY", "ganti-secret-key-ini-untuk-produksi")

# =========================
# DATABASE
# =========================
def db_conn():
    return sqlite3.connect(DB_PATH)


def db_init():
    con = db_conn()
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            ts TEXT NOT NULL,
            event TEXT NOT NULL,
            locked_criteria INTEGER NOT NULL,
            locked_alternatives INTEGER NOT NULL,
            kriteria_json TEXT NOT NULL,
            alternatif_json TEXT NOT NULL,
            pair_kriteria_json TEXT NOT NULL,
            pair_alternatif_json TEXT NOT NULL,
            results_json TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS experts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            nama TEXT NOT NULL,
            jabatan TEXT NOT NULL,
            instansi TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS work_state (
            user_id INTEGER PRIMARY KEY,
            state_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
        # =====================================
    # TABEL KRITERIA
    # =====================================
    cur.execute("""
        CREATE TABLE IF NOT EXISTS kriteria(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            history_id INTEGER,
            user_id INTEGER NOT NULL,
            kode TEXT NOT NULL,
            nama TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(history_id) REFERENCES history(id),
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    # =====================================
    # TABEL ALTERNATIF
    # =====================================
    cur.execute("""
        CREATE TABLE IF NOT EXISTS alternatif(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            history_id INTEGER,
            user_id INTEGER NOT NULL,
            kode TEXT NOT NULL,
            nama TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(history_id) REFERENCES history(id),
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    # =====================================
    # PAIRWISE KRITERIA
    # SATU RECORD = SATU PENILAIAN EXPERT
    # =====================================
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pairwise_kriteria(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            history_id INTEGER,
            expert_id INTEGER NOT NULL,
            kriteria1_id INTEGER NOT NULL,
            kriteria2_id INTEGER NOT NULL,
            winner_id INTEGER NOT NULL,
            nilai INTEGER NOT NULL,

            FOREIGN KEY(history_id) REFERENCES history(id),
            FOREIGN KEY(expert_id) REFERENCES experts(id),
            FOREIGN KEY(kriteria1_id) REFERENCES kriteria(id),
            FOREIGN KEY(kriteria2_id) REFERENCES kriteria(id),
            FOREIGN KEY(winner_id) REFERENCES kriteria(id)
        )
    """)

    # =====================================
    # PAIRWISE ALTERNATIF
    # =====================================
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pairwise_alternatif(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            history_id INTEGER,
            expert_id INTEGER NOT NULL,
            kriteria_id INTEGER NOT NULL,
            alternatif1_id INTEGER NOT NULL,
            alternatif2_id INTEGER NOT NULL,
            winner_id INTEGER NOT NULL,
            nilai INTEGER NOT NULL,

            FOREIGN KEY(history_id) REFERENCES history(id),
            FOREIGN KEY(expert_id) REFERENCES experts(id),
            FOREIGN KEY(kriteria_id) REFERENCES kriteria(id),
            FOREIGN KEY(alternatif1_id) REFERENCES alternatif(id),
            FOREIGN KEY(alternatif2_id) REFERENCES alternatif(id),
            FOREIGN KEY(winner_id) REFERENCES alternatif(id)
        )
    """)
    con.commit()
    con.close()


def _hash_password(password: str, username: str) -> str:
    salt = f"spk-fahp::{username}::v1"
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def db_create_user(username: str, password: str) -> Tuple[bool, str]:
    username = (username or "").strip()
    if len(username) < 3:
        return False, "Username minimal 3 karakter."
    if len(password or "") < 6:
        return False, "Password minimal 6 karakter."

    con = db_conn()
    cur = con.cursor()
    try:
        cur.execute(
            "INSERT INTO users(username,password_hash,created_at) VALUES(?,?,?)",
            (username, _hash_password(password, username), datetime.utcnow().isoformat()),
        )
        con.commit()
        return True, "Akun berhasil dibuat. Silakan login."
    except sqlite3.IntegrityError:
        return False, "Username sudah digunakan. Coba yang lain."
    finally:
        con.close()


def db_auth_user(username: str, password: str) -> Tuple[bool, Optional[int], str]:
    username = (username or "").strip()
    con = db_conn()
    cur = con.cursor()
    cur.execute("SELECT id, password_hash FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    con.close()

    if not row:
        return False, None, "Akun tidak ditemukan."
    uid, pw_hash = row
    if _hash_password(password or "", username) != pw_hash:
        return False, None, "Password salah."
    return True, int(uid), "Login berhasil."


def db_save_history(user_id: int, event: str, snapshot: dict):
    con = db_conn()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO history(
            user_id, ts, event, locked_criteria, locked_alternatives,
            kriteria_json, alternatif_json, pair_kriteria_json, pair_alternatif_json, results_json
        )
        VALUES(?,?,?,?,?,?,?,?,?,?)
        """,
        (
            user_id,
            datetime.utcnow().isoformat(),
            event,
            int(bool(snapshot.get("locked_criteria"))),
            int(bool(snapshot.get("locked_alternatives"))),
            json.dumps(snapshot.get("kriteria", []), ensure_ascii=False),
            json.dumps(snapshot.get("alternatif", []), ensure_ascii=False),
            json.dumps(snapshot.get("pair_kriteria", {}), ensure_ascii=False),
            json.dumps(snapshot.get("pair_alternatif", {}), ensure_ascii=False),
            json.dumps(snapshot.get("results", {}), ensure_ascii=False),
        ),
    )
    con.commit()
    con.close()


def db_list_history(user_id: int) -> pd.DataFrame:
    con = db_conn()

    df = pd.read_sql_query(
        """
        SELECT
            id,
            ts,
            event,
            locked_criteria,
            locked_alternatives
        FROM history
        WHERE user_id=?
          AND event='Buka Hasil Analisis'
        ORDER BY id DESC
        """,
        con,
        params=(user_id,),
    )

    con.close()
    return df


def db_get_history(user_id: int, hist_id: int) -> Optional[dict]:
    con = db_conn()
    cur = con.cursor()
    cur.execute(
        """
        SELECT id, ts, event, locked_criteria, locked_alternatives,
               kriteria_json, alternatif_json, pair_kriteria_json, pair_alternatif_json, results_json
        FROM history WHERE user_id=? AND id=?
        """,
        (user_id, hist_id),
    )
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    return {
        "id": row[0],
        "ts": row[1],
        "event": row[2],
        "locked_criteria": bool(row[3]),
        "locked_alternatives": bool(row[4]),
        "kriteria": json.loads(row[5] or "[]"),
        "alternatif": json.loads(row[6] or "[]"),
        "pair_kriteria": json.loads(row[7] or "{}"),
        "pair_alternatif": json.loads(row[8] or "{}"),
        "results": json.loads(row[9] or "{}"),
    }
def db_save_experts(user_id: int, experts: list):

    con = db_conn()
    cur = con.cursor()

    # Hapus expert lama
    cur.execute(
        "DELETE FROM experts WHERE user_id=?",
        (user_id,)
    )

    for e in experts:

        cur.execute(
            """
            INSERT INTO experts(
                user_id,
                nama,
                jabatan,
                instansi,
                created_at
            )
            VALUES(?,?,?,?,?)
            """,
            (
                user_id,
                e["nama"],
                e["jabatan"],
                e["instansi"],
                datetime.utcnow().isoformat()
            )
        )

    con.commit()
    con.close()


def db_get_experts(user_id:int):

    con=db_conn()
    cur=con.cursor()

    cur.execute("""
        SELECT
            id,
            nama,
            jabatan,
            instansi
        FROM experts
        WHERE user_id=?
        ORDER BY id
    """,(user_id,))

    rows=cur.fetchall()

    con.close()

    return [
        {
            "id":r[0],
            "nama":r[1],
            "jabatan":r[2],
            "instansi":r[3]
        }
        for r in rows
    ]
# =====================================
# KRITERIA
# =====================================

def db_save_kriteria(history_id: int, user_id: int, data: list):

    con = db_conn()
    cur = con.cursor()

    cur.execute(
        "DELETE FROM kriteria WHERE history_id=?",
        (history_id,)
    )

    for k in data:

        cur.execute("""
            INSERT INTO kriteria(
                history_id,
                user_id,
                kode,
                nama,
                created_at
            )
            VALUES(?,?,?,?,?)
        """,(
            history_id,
            user_id,
            k["kode"],
            k["nama"],
            datetime.utcnow().isoformat()
        ))

    con.commit()
    con.close()
def db_get_kriteria(history_id:int):

    con=db_conn()
    cur=con.cursor()

    cur.execute("""
        SELECT
            id,
            kode,
            nama
        FROM kriteria
        WHERE history_id=?
        ORDER BY id
    """,(history_id,))

    rows=cur.fetchall()

    con.close()

    return [
        {
            "id":r[0],
            "kode":r[1],
            "nama":r[2]
        }
        for r in rows
    ]
# =====================================
# ALTERNATIF
# =====================================

def db_save_alternatif(history_id,user_id,data):

    con=db_conn()
    cur=con.cursor()

    cur.execute(
        "DELETE FROM alternatif WHERE history_id=?",
        (history_id,)
    )

    for a in data:

        cur.execute("""
            INSERT INTO alternatif(
                history_id,
                user_id,
                kode,
                nama,
                created_at
            )
            VALUES(?,?,?,?,?)
        """,(
            history_id,
            user_id,
            a["kode"],
            a["nama"],
            datetime.utcnow().isoformat()
        ))

    con.commit()
    con.close()
def db_get_alternatif(history_id):

    con=db_conn()
    cur=con.cursor()

    cur.execute("""
        SELECT
            id,
            kode,
            nama
        FROM alternatif
        WHERE history_id=?
        ORDER BY id
    """,(history_id,))

    rows=cur.fetchall()

    con.close()

    return [
        {
            "id":r[0],
            "kode":r[1],
            "nama":r[2]
        }
        for r in rows
    ]

# =========================
# FAHP CORE
# =========================
RI_TABLE = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
    11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59
}

SAATY_TFN: Dict[int, Tuple[float, float, float]] = {
    1: (1.0, 1.0, 1.0),
    2: (1.0, 2.0, 3.0),
    3: (2.0, 3.0, 4.0),
    4: (3.0, 4.0, 5.0),
    5: (4.0, 5.0, 6.0),
    6: (5.0, 6.0, 7.0),
    7: (6.0, 7.0, 8.0),
    8: (7.0, 8.0, 9.0),
    9: (8.0, 9.0, 9.0),
}

SAATY_LABEL = {
    1: "1 - Sama Penting",
    2: "2 - Antara",
    3: "3 - Sedikit Lebih Penting",
    4: "4 - Antara",
    5: "5 - Lebih Penting",
    6: "6 - Antara",
    7: "7 - Sangat Penting",
    8: "8 - Antara",
    9: "9 - Mutlak Lebih Penting",
}

def max_allowed_cr(n):
    """
    Batas maksimum Consistency Ratio (CR)
    """
    if n <= 2:
        return 0.00
    elif n <= 5:
        return 0.10
    else:
        return 0.20
    
def pair_key(a: str, b: str) -> str:
    return f"{a}|{b}"


def split_key(key: str) -> Tuple[str, str]:
    a, b = key.split("|", 1)
    return a, b


def ahp_consistency(M: np.ndarray):

    n = M.shape[0]

    if n <= 2:
        return 0.0, 0.0, float(n)

    # 1. Normalisasi
    col_sum = M.sum(axis=0)

    norm = M / col_sum

    # 2. Priority Vector
    priority = norm.mean(axis=1)

    # 3. Weighted Sum Vector
    weighted_sum = np.dot(M, priority)

    # 4. Consistency Vector
    consistency_vector = weighted_sum / priority

    # 5. Lambda Max
    lambda_max = consistency_vector.mean()

    # 6. CI
    CI = (lambda_max - n) / (n - 1)

    # 7. CR
    RI = RI_TABLE.get(n, 1.59)

    CR = CI / RI if RI > 0 else 0.0

    return (
        float(CR),
        float(CI),
        float(lambda_max)
    )

def ahp_normalize_matrix(M: np.ndarray) -> np.ndarray:
    col_sum = M.sum(axis=0)
    return M / col_sum


# ==========================================
# GEOMETRIC MEAN UNTUK MULTI EXPERT
# ==========================================
import math

def geometric_mean_pair(a, b, values):
    """
    values:
    [
        ["K1",5],
        ["K2",3],
        ["K1",7]
    ]

    return:
        nilai geometric mean
    """

    ratios = []

    for winner, scale in values:

        scale = max(1, min(9, int(scale)))

        if winner == a:
            ratios.append(scale)
        else:
            ratios.append(1 / scale)

    gm = math.prod(ratios) ** (1 / len(ratios))

    return gm

def gm_to_tfn(gm: float) -> Tuple[float, float, float]:
    """
    Konversi nilai Geometric Mean ke TFN Saaty terdekat.
    """

    inverse = False

    if gm < 1:
        gm = 1 / gm
        inverse = True

    scale = int(round(gm))
    scale = max(1, min(9, scale))

    t = SAATY_TFN[scale]

    if inverse:
        t = tfn_inv(t)

    return t

def build_pcm(items: List[str], pair_state: Dict[str, List]) -> np.ndarray:
    n = len(items)
    idx = {items[i]: i for i in range(n)}
    M = np.ones((n, n), dtype=float)

    for key, value in (pair_state or {}).items():

        try:
            a, b = split_key(key)

            if a not in idx or b not in idx:
                continue

            # pindahkan ke sini
            i = idx[a]
            j = idx[b]

            # ==========================
            # MULTI EXPERT
            # ==========================
            if isinstance(value[0], list):

                gm = geometric_mean_pair(a, b, value)

                M[i, j] = gm
                M[j, i] = 1 / gm

                continue

            # ==========================
            # SINGLE EXPERT
            # ==========================
            winner = value[0]
            scale = int(value[1])

        except Exception:
            continue

        scale = max(1, min(9, scale))

        if winner == a:
            M[i, j] = scale
            M[j, i] = 1 / scale
        else:
            M[i, j] = 1 / scale
            M[j, i] = scale

    return M

def build_geomean_table(pair_state):

    rows = []

    for key, value in pair_state.items():

        a, b = split_key(key)

        if not isinstance(value[0], list):

            rows.append({
                "pair": f"{a} vs {b}",
                "experts": [{
                    "winner": value[0],
                    "scale": value[1]
                }],
                "gm": value[1]
            })

        else:

            gm = geometric_mean_pair(a, b, value)

            rows.append({
                "pair": f"{a} vs {b}",
                "experts": value,
                "gm": gm
            })

    return rows

def tfn_inv(t: Tuple[float, float, float]) -> Tuple[float, float, float]:
    l, m, u = t
    return (1.0 / u, 1.0 / m, 1.0 / l)


def tfn_add(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def tfn_mul(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (a[0] * b[0], a[1] * b[1], a[2] * b[2])


def tfn_to_str(t) -> str:
    return f"({float(t[0]):.4f}, {float(t[1]):.4f}, {float(t[2]):.4f})"


def build_fuzzy_pcm(items: List[str], pair_state: Dict[str, List]) -> np.ndarray:

    n = len(items)
    idx = {items[i]: i for i in range(n)}

    F = np.zeros((n, n, 3), dtype=float)

    for i in range(n):
        F[i, i, :] = (1.0, 1.0, 1.0)

    for key, value in (pair_state or {}).items():

        try:
            a, b = split_key(key)

            if a not in idx or b not in idx:
                continue

            i = idx[a]
            j = idx[b]

            # =====================================
            # MULTI EXPERT
            # =====================================
            if isinstance(value[0], list):

                # Hitung Geometric Mean CRISP
                gm = geometric_mean_pair(a, b, value)

                # Konversi GM -> TFN
                t = gm_to_tfn(gm)

                F[i, j, :] = t
                F[j, i, :] = tfn_inv(t)

                continue

            # =====================================
            # SINGLE EXPERT
            # =====================================
            winner = value[0]
            scale = max(1, min(9, int(value[1])))

            gm = float(scale)

            if winner != a:
                gm = 1 / gm

            t = gm_to_tfn(gm)

            F[i, j, :] = t
            F[j, i, :] = tfn_inv(t)

        except Exception:
            continue

    return F


def defuzz_centroid(t: Tuple[float, float, float]) -> float:
    l, m, u = t
    return float((l + m + u) / 3.0)


def chang_extent_weights_centroid(F: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    n = F.shape[0]
    row_sums = []
    total = (0.0, 0.0, 0.0)
    for i in range(n):
        rs = (0.0, 0.0, 0.0)
        for j in range(n):
            rs = tfn_add(rs, tuple(F[i, j, :]))
        row_sums.append(rs)
        total = tfn_add(total, rs)

    total_inv = tfn_inv(total)
    S = []
    for i in range(n):
        S_i = tfn_mul(row_sums[i], total_inv)
        S.append(S_i)

    c = np.array([defuzz_centroid(s) for s in S], dtype=float)
    if c.sum() <= 0:
        w = np.ones(n, dtype=float) / max(1, n)
    else:
        w = c / c.sum()

    detail = {
        "row_sums": [list(x) for x in row_sums],
        "total_sum": list(total),
        "total_inv": list(total_inv),
        "S": [list(x) for x in S],
        "centroid": c.tolist(),
    }
    return w.astype(float), detail


def json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)



# =========================
# DEFAULT STATE
# =========================
def default_state() -> dict:
    return {
        "experts_count": 1,

        "experts":[
            {
                "id":1,
                "nama": "Experts 1",
                "jabatan": "",
                "instansi": ""
            }
        ],

        "kriteria": [
            {"kode": "K1", "nama": "Kerentanan Website"},
            {"kode": "K2", "nama": "Keterbatasan SDM"},
            {"kode": "K3", "nama": "Tata Kelola Keamanan"},
            {"kode": "K4", "nama": "Ancaman Siber Modern"},
            {"kode": "K5", "nama": "Anomali Trafik & Risiko terhadap CIA"},
        ],
        "alternatif": [
            {"kode": "A1", "nama": "Proteksi terhadap Perangkat Perusak"},
            {"kode": "A2", "nama": "Pencadangan Informasi"},
            {"kode": "A3", "nama": "Kontrol Akses"},
            {"kode": "A4", "nama": "Pengkodean yang Aman"},
            {"kode": "A5", "nama": "Siklus Hidup Pengembangan Aman"},
            {"kode": "A6", "nama": "Kemanan Jaringan"},
        ],
        "pair_kriteria": {},

        "pair_alternatif": {},

        "alt_done": {},
        "locked_criteria": False,
        "locked_alternatives": False,
        "results": {},

    }



def compute_all(state: dict) -> dict:
    kc = k_codes(state)
    ac = a_codes(state)
    ensure_pair_kriteria(state)

    M_k = build_pcm(kc, state.get("pair_kriteria", {}))
    norm_k = ahp_normalize_matrix(M_k)
    CR_k, CI_k, lambda_k = ahp_consistency(M_k)

    F_k = build_fuzzy_pcm(kc, state.get("pair_kriteria", {}))
    w_k, detail_k = chang_extent_weights_centroid(F_k)

    w_alt_per_k: Dict[str, np.ndarray] = {}
    M_alt_per_k: Dict[str, np.ndarray] = {}
    norm_alt_per_k: Dict[str, np.ndarray] = {}
    CR_alt: Dict[str, dict] = {}
    F_alt_per_k: Dict[str, np.ndarray] = {}
    detail_alt: Dict[str, dict] = {}

    for k in kc:
        ensure_pair_alternatif(state, k)
        pair = state.get("pair_alternatif", {}).get(k, {})
        M_a = build_pcm(ac, pair)
        M_alt_per_k[k] = M_a
        norm_alt_per_k[k] = ahp_normalize_matrix(M_a)
        cr, ci, lmax = ahp_consistency(M_a)
        CR_alt[k] = {"CR": float(cr), "CI": float(ci), "lambda_max": float(lmax)}

        F_a = build_fuzzy_pcm(ac, pair)
        F_alt_per_k[k] = F_a
        w_a, det_a = chang_extent_weights_centroid(F_a)
        w_alt_per_k[k] = w_a
        detail_alt[k] = det_a

    W = np.zeros(len(ac), dtype=float)
    for i, k in enumerate(kc):
        W += float(w_k[i]) * np.array(w_alt_per_k[k], dtype=float)
    W = W / W.sum() if W.sum() > 0 else (np.ones(len(ac)) / max(1, len(ac)))
    order = np.argsort(-W)

    # Clustering prioritas. Jika semua bobot sama, cukup 1 cluster agar tidak error.
    if len(W) <= 1:
        priority_level = ["Tinggi"] * len(W)
    else:
        unique_count = len(np.unique(np.round(W, 12)))
        n_clusters = min(3, len(W), max(1, unique_count))
        if n_clusters <= 1:
            priority_level = ["Tinggi"] * len(W)
        else:
            X = W.reshape(-1, 1)
            Xs = StandardScaler().fit_transform(X)
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = km.fit_predict(Xs)
            centers = km.cluster_centers_.flatten()
            sorted_clusters = np.argsort(centers)
            level_map = {}
            if n_clusters == 3:
                level_map[sorted_clusters[0]] = "Rendah"
                level_map[sorted_clusters[1]] = "Sedang"
                level_map[sorted_clusters[2]] = "Tinggi"
            elif n_clusters == 2:
                level_map[sorted_clusters[0]] = "Sedang"
                level_map[sorted_clusters[1]] = "Tinggi"
            priority_level = [level_map[int(label)] for label in labels]

    top_idx = int(order[0]) if len(order) else 0
    top_alt = ac[top_idx] if ac else "-"

    results = {
        "k_codes": kc,
        "a_codes": ac,
        "M_k": M_k,
        "norm_k": norm_k,
        "CR_k": {"CR": float(CR_k), "CI": float(CI_k), "lambda_max": float(lambda_k)},
        "F_k": F_k,
        "w_k": w_k,
        "fuzzy_detail_k": detail_k,
        "M_alt_per_k": M_alt_per_k,
        "norm_alt_per_k": norm_alt_per_k,
        "CR_alt": CR_alt,
        "F_alt_per_k": F_alt_per_k,
        "w_alt_per_k": w_alt_per_k,
        "fuzzy_detail_alt": detail_alt,
        "W_global": W,
        "W": W,
        "order": order.tolist(),
        "priority_level": priority_level,
        "sistem_recommendation": f"Prioritas tertinggi: {top_alt} (berdasarkan bobot global).",
    }

    if state.get("experts_count", 1) > 1:

        results["gm_kriteria"] = build_geomean_table(
            state["pair_kriteria"]
        )

        results["gm_alternatif"] = {}

        for k in kc:
            results["gm_alternatif"][k] = build_geomean_table(
                state["pair_alternatif"][k]
            )

    else:

        results["gm_kriteria"] = None
        results["gm_alternatif"] = {}

    state["results"] = json_safe(results)
    return state["results"]



# =========================
# TEMPLATE HELPERS
# =========================
@app.context_processor
def inject_helpers():
    def logged_in():
        return bool(session.get("user_id"))

    def current_user():
        return session.get("username")

    def state_flags():
        state = get_state()
        return {

            "experts": state.get("experts", []),

            "locked_criteria": state.get("locked_criteria", False),

            "locked_alternatives": state.get("locked_alternatives", False)

        }

    return dict(
        logged_in=logged_in,
        current_user=current_user,
        state_flags=state_flags,
        saaty_label=SAATY_LABEL,
        tfn_to_str=tfn_to_str,
        zip=zip,
        enumerate=enumerate,
        len=len,
        float=float,
    )


def login_required():
    if not session.get("user_id"):
        flash("Silakan login terlebih dahulu.", "warning")
        return False
    return True


def format_matrix_number(x):
    x = float(x)

    if x.is_integer():
        return str(int(x))

    return f"{x:.4f}".rstrip("0").rstrip(".")


def matrix_table(M, rows, cols):
    arr = np.array(M, dtype=float)

    return [
        [rows[i]] + [format_matrix_number(arr[i, j]) for j in range(len(cols))]
        for i in range(len(rows))
    ]


# =========================
# DATABASE STATE
# =========================
def db_get_work_state(user_id: int) -> Optional[dict]:
    con = db_conn()
    cur = con.cursor()
    cur.execute(
        "SELECT state_json FROM work_state WHERE user_id=?",
        (int(user_id),)
    )
    row = cur.fetchone()
    con.close()

    if not row:
        return None

    try:
        return json.loads(row[0] or "{}")
    except (TypeError, json.JSONDecodeError):
        return None


def db_save_work_state(user_id: int, state: dict) -> None:
    con = db_conn()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO work_state(user_id, state_json, updated_at)
        VALUES(?,?,?)
        ON CONFLICT(user_id) DO UPDATE SET
            state_json=excluded.state_json,
            updated_at=excluded.updated_at
        """,
        (
            int(user_id),
            json.dumps(json_safe(state), ensure_ascii=False),
            datetime.utcnow().isoformat()
        )
    )
    con.commit()
    con.close()


def get_state() -> dict:
    user_id = session.get("user_id")
    if not user_id:
        return default_state()

    state = db_get_work_state(int(user_id))
    if not state:
        state = default_state()
        db_save_work_state(int(user_id), state)

    # Expert selalu diambil dari database agar tidak bergantung pada session.
    experts = db_get_experts(int(user_id))
    if experts:
        state["experts"] = experts
        state["experts_count"] = len(experts)

    return state


def set_state(state: dict) -> None:
    user_id = session.get("user_id")
    if not user_id:
        return
    db_save_work_state(int(user_id), json_safe(state))


def reset_work_state() -> None:
    user_id = session.get("user_id")
    if not user_id:
        return

    state = default_state()

    # Expert tidak dihapus saat reset; data expert tetap milik user di DB.
    experts = db_get_experts(int(user_id))
    if experts:
        state["experts"] = experts
        state["experts_count"] = len(experts)

    db_save_work_state(int(user_id), state)


def k_codes(state: dict) -> List[str]:
    return [k["kode"] for k in state.get("kriteria", [])]


def a_codes(state: dict) -> List[str]:
    return [a["kode"] for a in state.get("alternatif", [])]


def ensure_pair_kriteria(state: dict):
    codes = k_codes(state)
    valid_pairs = set()

    pair = state.setdefault("pair_kriteria", {})
    experts_count = state.get("experts_count", 1)

    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            key = pair_key(codes[i], codes[j])
            valid_pairs.add(key)

            if key not in pair:
                if experts_count == 1:
                    pair[key] = [codes[i], 1]
                else:
                    pair[key] = [[codes[i], 1] for _ in range(experts_count)]

    state["pair_kriteria"] = {
        k: v for k, v in pair.items() if k in valid_pairs
    }


def ensure_pair_alternatif(state: dict, kriterium: str):
    if not kriterium:
        return

    codes = a_codes(state)
    pair = state.setdefault("pair_alternatif", {}).setdefault(kriterium, {})
    experts_count = state.get("experts_count", 1)

    valid_pairs = set()

    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            key = pair_key(codes[i], codes[j])
            valid_pairs.add(key)

            if key not in pair:
                if experts_count == 1:
                    pair[key] = [codes[i], 1]
                else:
                    pair[key] = [[codes[i], 1] for _ in range(experts_count)]

    state["pair_alternatif"][kriterium] = {
        k: v for k, v in pair.items() if k in valid_pairs
    }


def snapshot_state(state):
    return json_safe({
        "experts_count": state.get("experts_count", 1),
        "experts": state.get("experts", []),
        "locked_criteria": state.get("locked_criteria", False),
        "locked_alternatives": state.get("locked_alternatives", False),
        "kriteria": state.get("kriteria", []),
        "alternatif": state.get("alternatif", []),
        "pair_kriteria": state.get("pair_kriteria", {}),
        "pair_alternatif": state.get("pair_alternatif", {}),
        "alt_done": state.get("alt_done", {}),
        "results": state.get("results", {})
    })


def save_history(event: str):
    user_id = session.get("user_id")
    if not user_id:
        return

    try:
        state = get_state()
        db_save_history(int(user_id), event, snapshot_state(state))
    except Exception as exc:
        flash(f"Gagal menyimpan riwayat: {exc}", "warning")


# =========================
# PDF REPORT
# =========================
def build_pdf_bytes_from(r: dict, state: dict, kriteria_list: List[dict], alternatif_list: List[dict]) -> bytes:
    if not r:
        return b""

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    Wp, Hp = A4

    def header(title):
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2 * cm, Hp - 2 * cm, title)
        c.setFont("Helvetica", 9)
        c.drawString(2 * cm, Hp - 2.6 * cm, "Sistem Pendukung Keputusan Penentuan Prioritas Kontrol Keamanan Menggunakan Metode Fuzzy AHP")
        c.line(2 * cm, Hp - 2.8 * cm, Wp - 2 * cm, Hp - 2.8 * cm)

    def new_page():
        c.showPage()
        header("LAPORAN HASIL ANALISIS PRIORITAS KONTROL KEAMANAN")

    def table_height(data, row_h=14):
        return len(data) * row_h

    def ensure_space(y, needed):

        if y - needed < 2.5 * cm:
            new_page()
            return Hp - 3.4 * cm

        return y

    def draw_table(x, y, data, col_widths, row_h=12, font_size=8):

        total_h = len(data) * row_h

        # jika tidak cukup ruang, langsung halaman baru
        if y - total_h < 2.5 * cm:
            new_page()
            y = Hp - 3.4 * cm

        table = Table(
            data,
            colWidths=col_widths,
            rowHeights=[row_h] * len(data)
        )

        table.setStyle(TableStyle([
            ("GRID",(0,0),(-1,-1),0.5,colors.black),
            ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTNAME",(0,1),(-1,-1),"Helvetica"),
            ("FONTSIZE",(0,0),(-1,-1),font_size),
            ("ALIGN",(0,0),(-1,-1),"CENTER"),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ]))

        table.wrapOn(c,0,0)
        table.drawOn(c,x,y-total_h)

        return y-total_h

    header("LAPORAN HASIL ANALISIS PRIORITAS KONTROL KEAMANAN")
    y = Hp - 3.8 * cm

    from datetime import datetime

    c.setFont("Helvetica-Bold",10)
    c.drawString(2*cm,y,"INFORMASI ANALISIS")
    y -= 15

    c.setFont("Helvetica",9)

    c.drawString(
        2*cm,
        y,
        f"Tanggal Analisis : {datetime.now().strftime('%d-%m-%Y %H:%M')}"
    )
    y -= 12

    jumlah_expert = len(state.get("experts", []))

    c.drawString(
        2*cm,
        y,
        f"Jumlah Experts : {jumlah_expert}"
    )
    y -= 20

    if state.get("experts_count",1)==1:
        judul="IDENTITAS PENGAMBIL KEPUTUSAN"
    else:
        judul="IDENTITAS PARA EXPERTS"

    c.setFont("Helvetica-Bold",10)
    c.drawString(2*cm,y,judul)
    y -= 15

    data_exp=[[
        "No",
        "Nama",
        "Jabatan",
        "Instansi"
    ]]

    for i,e in enumerate(state.get("experts",[]),1):

        data_exp.append([
            str(i),
            e["nama"],
            e["jabatan"],
            e["instansi"]
        ])

    y2 = draw_table(
        2*cm,
        y,
        data_exp,
        [1*cm,4*cm,6*cm,5*cm]
    )

    y = y2-20  

    c.setFont("Helvetica-Bold", 10)
    c.drawString(2 * cm, y, "1. RINGKASAN ANALISIS")
    y -= 14
    c.setFont("Helvetica", 9)
    c.drawString(2 * cm, y, f"Rekomendasi Sistem: {r.get('sistem_recommendation', '-')}")
    y -= 18

    c.setFont("Helvetica-Bold", 10)
    c.drawString(2 * cm, y, "2. PERHITUNGAN KRITERIA")
    y -= 14
    crk = r.get("CR_k", {})
    c.setFont("Helvetica", 9)
    c.drawString(2 * cm, y, f"lambda_max={float(crk.get('lambda_max', 0)):.4f} | CI={float(crk.get('CI', 0)):.4f} | CR={float(crk.get('CR', 0)):.4f}")
    y -= 14

    if state.get("experts_count", 1) > 1:

        c.setFont("Helvetica-Bold", 9)
        c.drawString(
            2 * cm,
            y,
            "Geometric Mean (GM) Perbandingan Kriteria"
        )

        y -= 15

        # ============================
        # Header Tabel
        # ============================

        data = [["Pair"]]

        for exp in state.get("experts", []):
            data[0].append(exp["nama"])

        data[0].append("GM")

        # ============================
        # Isi Tabel
        # ============================

        for row in r.get("gm_kriteria", []):

            baris = [row["pair"]]

            for nilai in row["experts"]:

                if isinstance(nilai, list):
                    winner, scale = nilai
                    baris.append(f"{winner} ({scale})")
                else:
                    baris.append(str(nilai))

            baris.append(f"{row['gm']:.6f}")

            data.append(baris)

        # ============================
        # Lebar Kolom
        # ============================

        col_width = [4 * cm]

        for _ in state.get("experts", []):
            col_width.append(2.5 * cm)

        col_width.append(3 * cm)

        y2 = draw_table(
            2 * cm,
            y,
            data,
            col_width,
            row_h=12,
            font_size=7
        )

        if y2 is None:

            new_page()
            y = Hp - 3.4 * cm

            y2 = draw_table(
                2 * cm,
                y,
                data,
                col_width,
                row_h=12,
                font_size=7
            )

        y = y2 - 20

    kcodes = r.get("k_codes", [])
    acodes = r.get("a_codes", [])
    mk = np.array(r.get("M_k", []), dtype=float)
    nk = np.array(r.get("norm_k", []), dtype=float)
    if len(kcodes) and mk.size:
        c.setFont("Helvetica-Bold", 9)
        c.drawString(2 * cm, y, "Matriks Perbandingan Kriteria ")
        y -= 14
        data_m = [["M"] + kcodes]
        for i, row in enumerate(mk):
            data_m.append([kcodes[i]] + [f"{v:.4f}" for v in row])
        y2 = draw_table(2 * cm, y, data_m, [2.2 * cm] + [2.0 * cm] * len(kcodes), row_h=12, font_size=7)
        if y2 is None:
            new_page(); y = Hp - 3.4 * cm
            y2 = draw_table(2 * cm, y, data_m, [2.2 * cm] + [2.0 * cm] * len(kcodes), row_h=12, font_size=7)
        y = y2 - 18
        # DETAIL FUZZY KRITERIA

        if y < 10 * cm:
            new_page()
            y = Hp - 3.4 * cm

        c.setFont("Helvetica-Bold", 9)
        c.drawString(2 * cm, y, "Detail perhitungan FuzzyAHP Kriteria")
        y -= 16

        fd = r.get("fuzzy_detail_k", {})

        # ROW SUM
        c.setFont("Helvetica-Bold", 8)
        c.drawString(2 * cm, y, "Row Sum")
        y -= 12

        data_rs = [["Kriteria", "L", "M", "U"]]

        for i,row in enumerate(fd.get("row_sums", [])):
            data_rs.append([
                kcodes[i],
                f"{row[0]:.6f}",
                f"{row[1]:.6f}",
                f"{row[2]:.6f}"
            ])

        y2 = draw_table(
            2 * cm,
            y,
            data_rs,
            [3*cm,4*cm,4*cm,4*cm]
        )

        y = y2 - 16

        c.setFont("Helvetica-Bold", 8)
        c.drawString(2 * cm, y, "Total Fuzzy")
        y -= 12

        tot = fd.get("total_sum",[0,0,0])

        c.setFont("Helvetica",8)
        c.drawString(
            2*cm,
            y,
            f"({tot[0]:.6f}, {tot[1]:.6f}, {tot[2]:.6f})"
        )
        y -= 18

        c.setFont("Helvetica-Bold", 8)
        c.drawString(2 * cm, y, "Inverse Total")
        y -= 12

        inv = fd.get("total_inv",[0,0,0])

        c.setFont("Helvetica",8)
        c.drawString(
            2*cm,
            y,
            f"({inv[0]:.6f}, {inv[1]:.6f}, {inv[2]:.6f})"
        )
        y -= 18

        c.setFont("Helvetica-Bold", 8)
        c.drawString(2 * cm, y, "Synthetic Extent (Si)")
        y -= 12

        data_s = [["Kriteria","L","M","U"]]

        for i,row in enumerate(fd.get("S",[])):
            data_s.append([
                kcodes[i],
                f"{row[0]:.6f}",
                f"{row[1]:.6f}",
                f"{row[2]:.6f}"
            ])

        y2 = draw_table(
            2*cm,
            y,
            data_s,
            [3*cm,4*cm,4*cm,4*cm]
        )

        y = y2 - 16

        c.setFont("Helvetica-Bold", 8)
        c.drawString(2 * cm, y, "Defuzzifikasi dan Bobot Prioritas")
        y -= 12

        centroid = fd.get("centroid", [])
        wk = np.array(r.get("w_k", []), dtype=float)

        data_def = [["Kriteria", "Centroid", "Bobot Prioritas"]]

        for i, val in enumerate(centroid):
            data_def.append([
                kcodes[i],
                f"{val:.6f}",
                f"{wk[i]:.6f}"
            ])

        y2 = draw_table(
            2 * cm,
            y,
            data_def,
            [4*cm,5*cm,5*cm]
        )

        if y2 is None:
            new_page()
            y = Hp - 3.4*cm

            y2 = draw_table(
                2 * cm,
                y,
                data_def,
                [4*cm,5*cm,5*cm]
            )

        y = y2 - 18

    c.setFont("Helvetica-Bold", 10)
    c.drawString(2 * cm, y, "3. PERHITUNGAN ALTERNATIF PER KRITERIA")
    y -= 14
    for k in kcodes:
        if y < 6 * cm:
            new_page(); y = Hp - 3.4 * cm
        c.setFont("Helvetica-Bold", 9)
        c.drawString(2 * cm, y, f"Kriteria {k} - Matriks Alternatif")
        y -= 12
        cra = (r.get("CR_alt", {}) or {}).get(k, {})
        c.setFont("Helvetica", 8)
        c.drawString(2 * cm, y, f"lambda_max={float(cra.get('lambda_max', 0)):.4f} | CI={float(cra.get('CI', 0)):.4f} | CR={float(cra.get('CR', 0)):.4f}")
        y -= 12

        # =====================================
        # GEOMETRIC MEAN ALTERNATIF (MULTI EXPERT)
        # =====================================

        if state.get("experts_count", 1) > 1:

            c.setFont("Helvetica-Bold", 9)
            c.drawString(2 * cm, y, "Geometric Mean Alternatif")
            y -= 12

            # Header tabel
            data_gm = [["Pair"]]

            # Tambahkan kolom nama expert
            for e in state.get("experts", []):
                data_gm[0].append(e["nama"])

            data_gm[0].append("GM")

            # Isi tabel
            for row in r.get("gm_alternatif", {}).get(k, []):

                baris = [row["pair"]]

                for nilai in row["experts"]:

                    if isinstance(nilai, list):
                        winner, scale = nilai
                        baris.append(f"{winner} ({scale})")
                    else:
                        baris.append(str(nilai))

                baris.append(f"{row['gm']:.6f}")

                data_gm.append(baris)

            # Lebar kolom
            col_width = [4 * cm]

            for _ in state.get("experts", []):
                col_width.append(2.5 * cm)

            col_width.append(3 * cm)

            y2 = draw_table(
                2 * cm,
                y,
                data_gm,
                col_width,
                row_h=12,
                font_size=7
            )

            if y2 is None:
                new_page()
                y = Hp - 3.4 * cm

                y2 = draw_table(
                    2 * cm,
                    y,
                    data_gm,
                    col_width,
                    row_h=12,
                    font_size=7
                )

            y = y2 - 18

        fd = r.get("fuzzy_detail_alt", {}).get(k, {})
        Ma = np.array((r.get("M_alt_per_k", {}) or {}).get(k, []), dtype=float)
        wa = np.array((r.get("w_alt_per_k", {}) or {}).get(k, []), dtype=float)
        if Ma.size:
            data_ma = [["M"] + acodes]
            for i, row in enumerate(Ma):
                data_ma.append([acodes[i]] + [f"{v:.4f}" for v in row])
            y2 = draw_table(2 * cm, y, data_ma, [2.2 * cm] + [2.0 * cm] * len(acodes), row_h=11, font_size=7)
            if y2 is None:
                new_page(); y = Hp - 3.4 * cm
                y2 = draw_table(2 * cm, y, data_ma, [2.2 * cm] + [2.0 * cm] * len(acodes), row_h=11, font_size=7)
            y = y2 - 14

        c.setFont("Helvetica-Bold", 9)
        jumlah_baris = len(fd.get("row_sums", [])) + 1
        tinggi_tabel = jumlah_baris * 14

        if y - tinggi_tabel < 3 * cm:
            new_page()
            y = Hp - 3.4 * cm
        c.drawString(2*cm, y, "Row Sum")
        y -= 12

        data_rs = [["Alternatif","L","M","U"]]

        for i,row in enumerate(fd.get("row_sums",[])):
            data_rs.append([
                acodes[i],
                f"{row[0]:.6f}",
                f"{row[1]:.6f}",
                f"{row[2]:.6f}"
            ])

        y2 = draw_table(
            2 * cm,
            y,
            data_rs,
            [3*cm,4*cm,4*cm,4*cm]
        )

        if y2 is None:
            new_page()
            y = Hp - 3.4 * cm

            y2 = draw_table(
                2 * cm,
                y,
                data_rs,
                [3*cm,4*cm,4*cm,4*cm]
            )

        y = y2 - 16

        tot = fd.get("total_sum",[0,0,0])

        c.setFont("Helvetica-Bold",9)
        c.drawString(2*cm,y,"Total Fuzzy")
        y -= 12

        c.setFont("Helvetica",8)

        c.drawString(
            2*cm,
            y,
            f"({tot[0]:.6f}, {tot[1]:.6f}, {tot[2]:.6f})"
        )

        y -= 18

        inv = fd.get("total_inv",[0,0,0])

        c.setFont("Helvetica-Bold",9)
        c.drawString(2*cm,y,"Inverse Total")
        y -= 12

        c.setFont("Helvetica",8)

        c.drawString(
            2*cm,
            y,
            f"({inv[0]:.6f}, {inv[1]:.6f}, {inv[2]:.6f})"
        )

        y -= 18

        if y < 8 * cm:
            new_page()
            y = Hp - 3.4 * cm
        c.setFont("Helvetica-Bold",9)
        c.drawString(2*cm,y,"Synthetic Extent (Si)")
        y -= 12

        data_s = [["Alternatif","L","M","U"]]

        for i,row in enumerate(fd.get("S",[])):
            data_s.append([
                acodes[i],
                f"{row[0]:.6f}",
                f"{row[1]:.6f}",
                f"{row[2]:.6f}"
            ])

        y2 = draw_table(
            2*cm,
            y,
            data_s,
            [3*cm,4*cm,4*cm,4*cm]
        )

        if y2 is None:
            new_page()
            y = Hp - 3.4 * cm

            y2 = draw_table(
                2*cm,
                y,
                data_s,
                [3*cm,4*cm,4*cm,4*cm]
            )

        y = y2 - 16

        if y < 6 * cm:
            new_page()
            y = Hp - 3.4 * cm
        c.setFont("Helvetica-Bold",9)
        c.drawString(2*cm,y,"Defuzzifikasi dan Bobot Prioritas")
        y -= 12

        centroid = fd.get("centroid",[])

        data_w = [["Alternatif","Centroid","Bobot Prioritas"]]

        for i, a in enumerate(alternatif_list):

            cen = centroid[i] if i < len(centroid) else 0
            bobot = wa[i] if i < len(wa) else 0

            data_w.append([
                a["nama"],
                f"{cen:.6f}",
                f"{bobot:.6f}"
            ])

        y2 = draw_table(
            2 * cm,
            y,
            data_w,
            [10*cm, 3*cm, 3*cm]
        )

        if y2 is None:
            new_page()
            y = Hp - 3.4 * cm

            y2 = draw_table(
                2 * cm,
                y,
                data_w,
                [10*cm, 3*cm, 3*cm]
            )

        y = y2 - 18

    new_page()
    y = Hp - 3.8*cm
    c.setFont("Helvetica-Bold", 10)
    c.drawString(2 * cm, y, "6. PRIORITAS GLOBAL + REKOMENDASI SISTEM")
    y -= 14
    Wg = np.array(r.get("W", []), dtype=float)
    order = [int(x) for x in r.get("order", [])]
    lvl = r.get("priority_level", [])
    data_final = [["Rank", "Nama Alternatif", "Skor Global", "Prioritas"]]
    for rank, idx in enumerate(order, start=1):
        if idx < len(alternatif_list):
            data_final.append([rank, alternatif_list[idx].get("nama", "-"), f"{float(Wg[idx]):.6f}", lvl[idx] if idx < len(lvl) else "-"])
    y2 = draw_table(2 * cm, y, data_final, [1.5 * cm, 10.5 * cm, 3 * cm, 3 * cm], row_h=12, font_size=8)
    if y2 is None:
        new_page(); y = Hp - 3.4 * cm
        draw_table(2 * cm, y, data_final, [1.5 * cm, 10.5 * cm, 3 * cm, 3 * cm], row_h=12, font_size=8)

    y = y2 - 25

    c.setFont("Helvetica-Bold", 10)
    c.drawString(2*cm, y, "Visualisasi Bobot Global")
    y -= 15

    max_width = 10 * cm

    for rank, idx in enumerate(order, start=1):

        nilai = float(Wg[idx])

        bar = nilai * max_width / max(Wg)

        c.setFont("Helvetica",8)

        c.drawString(
            2*cm,
            y+2,
            f"{rank}. {alternatif_list[idx]['nama']}"
        )

        c.rect(
            8*cm,
            y,
            bar,
            8,
            fill=1
        )

        c.drawString(
            18.3*cm,
            y+2,
            f"{nilai:.4f}"
        )

        y -= 14

        if y < 4*cm:
            new_page()
            y = Hp-3.4*cm
        
    y -= 20

    if y < 7*cm:
        new_page()
        y = Hp-3.4*cm

    idx = order[0]

    nama = alternatif_list[idx]["nama"]

    bobot = float(Wg[idx])

    prioritas = lvl[idx]

    c.setFont("Helvetica-Bold",10)
    c.drawString(
        2*cm,
        y,
        "Kesimpulan"
    )

    y -= 15

    text = c.beginText(
        2*cm,
        y
    )

    text.setFont(
        "Helvetica",
        9
    )

    paragraf = (
        f"Berdasarkan hasil sintesis bobot global menggunakan metode "
        f"Fuzzy AHP, alternatif '{nama}' memperoleh bobot tertinggi "
        f"sebesar {bobot:.4f}. Nilai tersebut menunjukkan bahwa "
        f"alternatif ini merupakan prioritas utama dalam implementasi "
        f"kontrol keamanan informasi. Berdasarkan hasil analisis, "
        f"alternatif ini termasuk kategori prioritas '{prioritas}', "
        f"sehingga direkomendasikan untuk diimplementasikan terlebih "
        f"dahulu sebelum alternatif lainnya."
    )

    styles=getSampleStyleSheet()

    style=styles["BodyText"]

    style.fontName="Helvetica"

    style.fontSize=10

    style.leading=16

    style.alignment=4   # justify
    p=Paragraph(paragraf,style)

    w,h=p.wrap(
        16*cm,
        20*cm
    )

    p.drawOn(
        c,
        2*cm,
        y-h
    )

    c.drawText(text)
    
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()


# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    if not session.get("user_id"):
        return redirect(url_for("login"))
    return redirect(url_for("beranda"))


@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        ok, uid, msg = db_auth_user(
            request.form.get("username"),
            request.form.get("password")
        )

        if not ok:

            flash(msg, "danger")

        else:

            session.clear()

            session["user_id"] = uid
            session["username"] = request.form.get("username", "").strip()

            # Seluruh state aplikasi disimpan di database.
            # Session hanya menyimpan identitas login.
            state = db_get_work_state(uid)
            if not state:
                state = default_state()

            experts = db_get_experts(uid)
            if experts:
                state["experts"] = experts
                state["experts_count"] = len(experts)

            db_save_work_state(uid, state)

            flash("Login berhasil.", "success")

            return redirect(url_for("beranda"))

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        p1 = request.form.get("password")
        p2 = request.form.get("password2")
        if p1 != p2:
            flash("Password tidak sama.", "danger")
        else:
            ok, msg = db_create_user(request.form.get("username"), p1)
            flash(msg, "success" if ok else "danger")
            if ok:
                return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logout berhasil.", "info")
    return redirect(url_for("login"))


@app.route("/beranda")
def beranda():
    if not login_required():
        return redirect(url_for("login"))

    state = get_state()

    return render_template(
        "beranda.html",
        state=state
    )

@app.route("/set_experts", methods=["POST"])
def set_experts():
    if not login_required():
        return redirect(url_for("login"))

    state = get_state()
    try:
        experts_count = int(request.form.get("experts_count", 1))
    except:
        experts_count = 1

    # batas maksimal 5 expert
    experts_count = max(1, min(5, experts_count))

    state["experts_count"] = experts_count
    experts = []
    for i in range(experts_count):

        nama = request.form.get(
            f"experts_name_{i}",
            ""
        ).strip()

        jabatan = request.form.get(
            f"experts_job_{i}",
            ""
        ).strip()

        instansi = request.form.get(
            f"experts_instansi_{i}",
            ""
        ).strip()

        if nama == "":
            nama = f"Experts {i+1}"

        experts.append({
            "nama": nama,
            "jabatan": jabatan,
            "instansi": instansi
        })

    # Simpan data expert
    state["experts"] = experts
    db_save_experts(
        int(session["user_id"]),
        experts
    )
    # ==========================
    # Reset seluruh matriks
    # ==========================
    state["pair_kriteria"] = {}
    state["pair_alternatif"] = {}
    state["alt_done"] = {}
    state["locked_criteria"] = False
    state["locked_alternatives"] = False
    state["results"] = {}

    # Bangun ulang struktur pair sesuai jumlah expert
    ensure_pair_kriteria(state)

    for k in k_codes(state):
        ensure_pair_alternatif(state, k)

    set_state(state)

    flash(f"Experts berhasil disimpan. Silahkan lanjut mengisi matriks kriteria.", "success")

    return redirect(url_for("kriteria"))

@app.route("/reset", methods=["POST"])
def reset():
    if not login_required():
        return redirect(url_for("login"))
    reset_work_state()
    flash("Semua data kerja berhasil direset.", "info")
    return redirect(url_for("beranda"))


@app.route("/kriteria", methods=["GET", "POST"])
def kriteria():
    if not login_required():
        return redirect(url_for("login"))

    state = get_state()

    if not state.get("experts"):
        flash("Silakan pilih jumlah expert terlebih dahulu.")
        return redirect(url_for("beranda"))

    if request.method == "POST":
        action = request.form.get("action")
        # ==================================================
        # Selalu update nilai pairwise (boleh diedit kapan saja)
        # ==================================================
        experts_count = state.get("experts_count", 1)

        for key in list(state.get("pair_kriteria", {}).keys()):

            a, b = split_key(key)

            if experts_count == 1:

                winner = request.form.get(
                    f"winner_{key}",
                    state["pair_kriteria"][key][0]
                )

                scale = int(
                    request.form.get(
                        f"scale_{key}",
                        state["pair_kriteria"][key][1]
                    )
                )

                if winner not in [a, b]:
                    winner = a

                state["pair_kriteria"][key] = [
                    winner,
                    max(1, min(9, scale))
                ]

            else:

                values = []

                for i in range(experts_count):

                    winner = request.form.get(
                        f"winner_{key}_{i}",
                        state["pair_kriteria"][key][i][0]
                    )

                    scale = int(
                        request.form.get(
                            f"scale_{key}_{i}",
                            state["pair_kriteria"][key][i][1]
                        )
                    )

                    if winner not in [a, b]:
                        winner = a

                    values.append([
                        winner,
                        max(1, min(9, scale))
                    ])

                state["pair_kriteria"][key] = values

        if action == "add" and not state.get("locked_criteria"):
            kode = request.form.get("kode_baru", "").strip()
            nama = request.form.get("nama_baru", "").strip()
            if not kode or not nama:
                flash("Kode dan nama kriteria wajib diisi.", "warning")
            elif kode in k_codes(state):
                flash("Kode kriteria sudah ada.", "warning")
            else:
                state["kriteria"].append({"kode": kode, "nama": nama})
                state["locked_criteria"] = False
                state["locked_alternatives"] = False
                state["alt_done"] = {}
                state["results"] = {}
                ensure_pair_kriteria(state)
                flash("Kriteria berhasil ditambahkan.", "success")

        elif action == "delete" and not state.get("locked_criteria"):
            kode = request.form.get("del_code")
            if kode and kode != "-":
                state["kriteria"] = [k for k in state["kriteria"] if k["kode"] != kode]
                state["locked_criteria"] = False
                state["locked_alternatives"] = False
                state["alt_done"] = {}
                state["results"] = {}
                ensure_pair_kriteria(state)
                flash("Kriteria berhasil dihapus.", "success")

        elif action == "reset_pairs" and not state.get("locked_criteria"):
            state["pair_kriteria"] = {}
            ensure_pair_kriteria(state)
            state["locked_alternatives"] = False
            state["alt_done"] = {}
            state["results"] = {}
            flash("Matriks kriteria direset.", "info")

        elif action in ("save_matrix", "save_next"):
            codes = k_codes(state)
            if len(codes) < 2:
                flash("Tambahkan minimal 2 kriteria.", "warning")
            
            else:
                M_k = build_pcm(codes, state.get("pair_kriteria", {}))
                cr, ci, lmax = ahp_consistency(M_k) 

                # Menentukan batas CR berdasarkan jumlah kriteria
                limit = max_allowed_cr(len(codes))

                if cr > limit:
                    flash(
                        f"CR = {cr:.4f} > {limit:.2f}. Perbaiki perbandingan kriteria terlebih dahulu.",
                        "danger"
                )
                else:
                    state["locked_criteria"] = True

                    for code in codes:
                        state.setdefault("alt_done", {}).setdefault(code, False)

                    save_history("Simpan Matriks Kriteria")
                    flash(
                        f"Matriks kriteria valid (CR ≤ {limit:.2f}). Silakan lanjut ke Input Alternatif.",
                        "success"
                    )

        set_state(state)

        if action == "save_next" and state.get("locked_criteria"):
            return redirect(url_for("alternatif"))

        return redirect(url_for("kriteria"))

    codes = k_codes(state)
    M_k = build_pcm(codes, state.get("pair_kriteria", {})) if len(codes) else np.array([])
    cr, ci, lmax = ahp_consistency(M_k) if len(codes) >= 2 else (0.0, 0.0, 0.0)
    return render_template(
        "kriteria.html",
        state=state,
        codes=codes,
        matrix=matrix_table(M_k, codes, codes) if len(codes) else [],
        cr=cr,
        ci=ci,
        lmax=lmax,
    )


@app.route("/alternatif", methods=["GET", "POST"])
def alternatif():
    if not login_required():
        return redirect(url_for("login"))
    state = get_state()
    if not state.get("locked_criteria"):
        flash("Simpan matriks kriteria terlebih dahulu.", "warning")
        return redirect(url_for("kriteria"))

    kc = k_codes(state)
    ac = a_codes(state)
    selected_k = request.values.get("selected_k") or request.args.get("k") or (kc[0] if kc else "")
    if selected_k not in kc and kc:
        selected_k = kc[0]
    ensure_pair_alternatif(state, selected_k)

    if request.method == "POST":
        action = request.form.get("action")
        selected_k = request.form.get("selected_k", selected_k)

        if action == "add_alt" and not state.get("locked_alternatives"):
            kode = request.form.get("kode_baru", "").strip()
            nama = request.form.get("nama_baru", "").strip()
            if not kode or not nama:
                flash("Kode dan nama alternatif wajib diisi.", "warning")
            elif kode in a_codes(state):
                flash("Kode alternatif sudah ada.", "warning")
            else:
                state["alternatif"].append({"kode": kode, "nama": nama})
                state["locked_alternatives"] = False
                state["alt_done"] = {k: False for k in kc}
                state["pair_alternatif"] = {k: {} for k in kc}
                state["results"] = {}
                flash("Alternatif berhasil ditambahkan. Perbandingan alternatif perlu diisi ulang.", "success")
            
        elif action == "delete_alt" and not state.get("locked_alternatives"):
            kode = request.form.get("del_code")
            if kode and kode != "-":
                state["alternatif"] = [a for a in state["alternatif"] if a["kode"] != kode]
                state["locked_criteria"] = False
                state["locked_alternatives"] = False
                state["alt_done"] = {}
                state["results"] = {}
                ensure_pair_alternatif(state)
                flash("Alternatif berhasil dihapus. Silakan isi ulang perbandingan alternatif.", "success") 

        elif action == "reset_selected":
            state.setdefault("pair_alternatif", {})[selected_k] = {}
            ensure_pair_alternatif(state, selected_k)
            state.setdefault("alt_done", {})[selected_k] = False
            state["locked_alternatives"] = False
            state["results"] = {}
            flash(f"Matriks alternatif untuk {selected_k} direset.", "info")

        elif action in ("save_alt", "save_next"):
            ensure_pair_alternatif(state, selected_k)
            experts_count = state.get("experts_count", 1)
            for key in list(state["pair_alternatif"].get(selected_k, {}).keys()):
                a, b = split_key(key)
                # ==========================
                # SINGLE EXPERT
                # ==========================
                if experts_count == 1:

                    winner = request.form.get(
                        f"winner_{key}",
                        state["pair_alternatif"][selected_k][key][0]
                    )

                    scale = int(
                        request.form.get(
                            f"scale_{key}",
                            state["pair_alternatif"][selected_k][key][1]
                        )
                    )

                    if winner not in [a, b]:
                        winner = a

                    state["pair_alternatif"][selected_k][key] = [
                        winner,
                        max(1, min(9, scale))
                    ]

                # ==========================
                # MULTI EXPERT
                # ==========================
                else:

                    values = []

                    for i in range(experts_count):

                        winner = request.form.get(
                            f"winner_{key}_{i}",
                            state["pair_alternatif"][selected_k][key][i][0]
                        )

                        scale = int(
                            request.form.get(
                                f"scale_{key}_{i}",
                                state["pair_alternatif"][selected_k][key][i][1]
                            )
                        )

                        if winner not in [a, b]:
                            winner = a

                        values.append([
                            winner,
                            max(1, min(9, scale))
                        ])

                    state["pair_alternatif"][selected_k][key] = values

            ac = a_codes(state)

            M_a = build_pcm(
                ac,
                state["pair_alternatif"].get(selected_k, {})
            )

            cr, ci, lmax = ahp_consistency(M_a)

            limit = max_allowed_cr(len(ac))

            if cr > limit:
                flash(
                    f"CR = {cr:.4f} > {limit:.2f}. Perbaiki perbandingan alternatif pada {selected_k}.",
                    "danger"
                )

            else:

                state.setdefault("alt_done", {})[selected_k] = True

                if all(state.get("alt_done", {}).get(k, False) for k in kc):

                    state["locked_alternatives"] = True

                    flash(
                        f"Semua matriks alternatif valid (CR ≤ {limit:.2f}). Silakan lanjut ke Hasil Analisis.",
                        "success"
                    )

                else:

                    flash(
                        f"Matriks alternatif untuk {selected_k} berhasil disimpan.",
                        "success"
                    )

                save_history(f"Simpan Matriks Alternatif ({selected_k})")

        else:
            # Hanya ganti pilihan kriteria
            pass

        set_state(state)

        # Jika klik Simpan & Berikutnya
        if action == "save_next":

            idx = kc.index(selected_k)

            if idx < len(kc) - 1:
                return redirect(
                    url_for(
                        "alternatif",
                        k=kc[idx + 1]
                    )
                )

        # Tetap di halaman yang sama jika hanya Simpan
        return redirect(
            url_for(
                "alternatif",
                k=selected_k
            )
        )

    ac = a_codes(state)
    M_a = build_pcm(
        ac,
        state.get("pair_alternatif", {}).get(selected_k, {})
    ) if len(ac) else np.array([])
        
    ac = a_codes(state)
    M_a = build_pcm(ac, state.get("pair_alternatif", {}).get(selected_k, {})) if len(ac) else np.array([])
    cr, ci, lmax = ahp_consistency(M_a) if len(ac) >= 2 else (0.0, 0.0, 0.0)
    done_count = sum(1 for k in kc if state.get("alt_done", {}).get(k, False))
    return render_template(
        "alternatif.html",
        state=state,
        kc=kc,
        ac=ac,
        selected_k=selected_k,
        done_count=done_count,
        matrix=matrix_table(M_a, ac, ac) if len(ac) else [],
        cr=cr,
        ci=ci,
        lmax=lmax,
    )

@app.route("/hasil")
def hasil():
    if not login_required():
        return redirect(url_for("login"))
    state = get_state()
    if not (state.get("locked_criteria") and state.get("locked_alternatives")):
        flash("Lengkapi dan simpan semua matriks terlebih dahulu.", "warning")
        return redirect(url_for("alternatif") if state.get("locked_criteria") else url_for("kriteria"))

    r = compute_all(state)
    set_state(state)
    save_history("Buka Hasil Analisis")
    return render_template("hasil.html", state=state, r=r, matrix_table=matrix_table)


@app.route("/laporan")
def laporan():
    if not login_required():
        return redirect(url_for("login"))
    state = get_state()
    if not state.get("results"):
        if not (state.get("locked_criteria") and state.get("locked_alternatives")):
            flash("Belum ada hasil analisis.", "warning")
            return redirect(url_for("beranda"))
        compute_all(state)
        set_state(state)
    save_history("Buka Laporan")
    return render_template("laporan.html", state=state, r=state.get("results", {}))


@app.route("/download/laporan.pdf")
def download_laporan():
    if not login_required():
        return redirect(url_for("login"))
    state = get_state()
    if not state.get("results"):
        compute_all(state)
        set_state(state)
    pdf = build_pdf_bytes_from(state.get("results", {}), state, state.get("kriteria", []), state.get("alternatif", []))
    return send_file(io.BytesIO(pdf), mimetype="application/pdf", as_attachment=True, download_name="laporan_hasil_analisis.pdf")


@app.route("/riwayat")
def riwayat():
    if not login_required():
        return redirect(url_for("login"))
    dfh = db_list_history(int(session["user_id"]))
    records = dfh.to_dict(orient="records") if not dfh.empty else []
    return render_template("riwayat.html", records=records)

@app.route("/riwayat/<int:hist_id>")
def riwayat_detail(hist_id):

    if not login_required():
        return redirect(url_for("login"))

    snap = db_get_history(
        int(session["user_id"]),
        hist_id
    )

    if not snap:
        abort(404)

    # Ambil data expert
    snap["experts"] = db_get_experts(
        int(session["user_id"])
    )

    snap["experts_count"] = len(snap["experts"])

    # ==============================
    # Bangun state dari snapshot
    # ==============================

    state = default_state()

    state["experts"] = snap["experts"]
    state["experts_count"] = snap["experts_count"]

    state["kriteria"] = snap["kriteria"]
    state["alternatif"] = snap["alternatif"]

    state["pair_kriteria"] = snap["pair_kriteria"]
    state["pair_alternatif"] = snap["pair_alternatif"]

    state["locked_criteria"] = snap["locked_criteria"]
    state["locked_alternatives"] = snap["locked_alternatives"]

    # Hitung ulang
    r = compute_all(state)

    return render_template(
        "riwayat_detail.html",
        snap=snap,
        r=r,
        matrix_table=matrix_table
    )

@app.route("/riwayat/<int:hist_id>/edit")
def riwayat_edit(hist_id):

    if not login_required():
        return redirect(url_for("login"))

    snap = db_get_history(
        int(session["user_id"]),
        hist_id
    )

    if not snap:
        abort(404)

    state = {
        "experts": db_get_experts(int(session["user_id"])),
        "experts_count": len(db_get_experts(int(session["user_id"]))),

        "kriteria": snap["kriteria"],
        "alternatif": snap["alternatif"],

        "pair_kriteria": snap["pair_kriteria"],
        "pair_alternatif": snap["pair_alternatif"],

        "alt_done": snap.get("alt_done", {}),
        "locked_criteria": snap["locked_criteria"],
        "locked_alternatives": snap["locked_alternatives"],

        "results": snap["results"],
    }
    set_state(state)

    flash("Riwayat berhasil dimuat. Silakan edit.", "success")

    return redirect(url_for("kriteria"))

@app.route("/riwayat/<int:hist_id>/pdf")
def riwayat_pdf(hist_id: int):

    if not login_required():
        return redirect(url_for("login"))

    snap = db_get_history(
        int(session["user_id"]),
        hist_id
    )

    if not snap or not snap.get("results"):
        abort(404)

    # Ambil data expert milik user
    snap["experts"] = db_get_experts(
        int(session["user_id"])
    )

    snap["experts_count"] = len(
        snap["experts"]
    )

    pdf = build_pdf_bytes_from(
        snap["results"],
        snap,
        snap["kriteria"],
        snap["alternatif"]
    )

    return send_file(
        io.BytesIO(pdf),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"laporan_snapshot_{hist_id}.pdf"
    )


if __name__ == "__main__":
    db_init()
    app.run(debug=True)
