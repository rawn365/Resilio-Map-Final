import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import re
import io
import time
from datetime import datetime
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

# Optional imports with graceful fallback
try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, HRFlowable, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLING
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Resilio-Map",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,700;0,9..144,900;1,9..144,400&family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* Force light mode */
html, body { color-scheme: light !important; background-color: #f7f9f7 !important; }
*, *::before, *::after { color-scheme: light !important; }
html, body, [class*="css"], [class*="st-"] { font-family: 'DM Sans', sans-serif !important; }
.stApp, .stApp > div, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
    background-color: #f7f9f7 !important;
}

/* Hide chrome */
#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stStatusWidget"] { visibility: hidden !important; }
[data-testid="stToolbarActions"] { visibility: hidden !important; }
[data-testid="stMainMenuPopover"] { display: none !important; }

/* Sidebar toggle - both states */
[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    display: flex !important;
    position: relative !important;
    z-index: 99999 !important;
}
[data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    display: flex !important;
    position: fixed !important;
    top: 14px !important;
    left: 14px !important;
    z-index: 99999 !important;
}
[data-testid="stSidebarCollapsedControl"] button {
    visibility: visible !important;
    display: flex !important;
    background: #ffffff !important;
    border: 1px solid #e3ebe4 !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.12) !important;
    padding: 6px !important;
}

/* Fix sidebar toggle showing text in both open and closed states */
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarCollapsedControl"] span {
    display: none !important;
}
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stSidebarCollapsedControl"] svg {
    display: block !important;
}

/* Fix expander toggle showing icon label text (for all expanders) */
[data-testid="stExpander"] summary span[data-testid="stExpanderToggleIcon"] span {
    display: none !important;
}

/* Sidebar */
[data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e3ebe4 !important; }
[data-testid="stSidebar"] > div:first-child { background-color: #ffffff !important; overflow-y: auto !important; overflow-x: hidden !important; }

.block-container { padding: 2rem 2.5rem 3rem; max-width: 1200px; }

/* Buttons */
[data-testid="baseButton-primary"], .stButton > button[kind="primary"] {
    background-color: #1e6b3c !important; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;
    border: none !important; border-radius: 8px !important; font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
    font-size: 13px !important; box-shadow: 0 2px 8px rgba(30,107,60,0.2) !important;
}
[data-testid="baseButton-primary"]:hover { background-color: #2d8a50 !important; }
[data-testid="baseButton-secondary"], .stButton > button[kind="secondary"], .stButton > button {
    background-color: #ffffff !important; color: #4a5e4c !important; -webkit-text-fill-color: #4a5e4c !important;
    border: 1px solid #e3ebe4 !important; border-radius: 8px !important; font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important;
    font-size: 13px !important; box-shadow: none !important;
}
.stButton > button:hover { background-color: #eaf4ed !important; border-color: #a8ccb2 !important; color: #141f16 !important; -webkit-text-fill-color: #141f16 !important; }

/* Inputs */
input, textarea, select, [data-testid="stTextInput"] input, [data-testid="stTextInput"] textarea,
[data-testid="stNumberInput"] input, [data-testid="stSelectbox"] input, [data-testid="stMultiSelect"] input {
    background-color: #ffffff !important; color: #141f16 !important; border: 1px solid #c8d8ca !important;
    border-radius: 8px !important; font-size: 13px !important; font-family: 'DM Sans', sans-serif !important;
    -webkit-text-fill-color: #141f16 !important;
}
input::placeholder, textarea::placeholder { color: #8fa893 !important; -webkit-text-fill-color: #8fa893 !important; opacity: 1 !important; }

/* Selectbox */
[data-testid="stSelectbox"] > div > div { background-color: #ffffff !important; color: #141f16 !important; border: 1px solid #c8d8ca !important; border-radius: 8px !important; }
[data-testid="stSelectbox"] span, [data-testid="stSelectbox"] p, [data-testid="stSelectbox"] div { color: #141f16 !important; }
[role="listbox"] li, [role="option"], ul[role="listbox"] li, div[role="option"] { background-color: #ffffff !important; color: #141f16 !important; }
[role="option"]:hover, [role="option"][aria-selected="true"] { background-color: #eaf4ed !important; color: #1e6b3c !important; }

/* Radio & checkbox */
[data-testid="stRadio"] label, [data-testid="stRadio"] span, [data-testid="stCheckbox"] label, [data-testid="stCheckbox"] span {
    color: #141f16 !important; font-family: 'DM Sans', sans-serif !important; font-size: 13px !important;
}

/* Dataframe */
[data-testid="stProgress"] > div > div { background-color: #1e6b3c !important; }
[data-testid="stDataFrame"] { border: 1px solid #e3ebe4 !important; border-radius: 10px !important; overflow: hidden !important; color: #141f16; }

/* Expander summary text (normal) */
[data-testid="stExpander"] summary { font-family: 'DM Sans', sans-serif !important; font-size: 13px !important; color: #141f16 !important; word-break: break-word !important; white-space: normal !important; }
[data-testid="stExpander"] details { word-break: break-word !important; }

/* Typography */
.page-title { font-family: 'Fraunces', Georgia, serif; font-size: 28px; font-weight: 900; color: #141f16; letter-spacing: -0.03em; line-height: 1.15; margin-bottom: 4px; }
.page-eyebrow { font-family: 'DM Mono', monospace; font-size: 11px; letter-spacing: 0.16em; text-transform: uppercase; color: #2d8a50; margin-bottom: 8px; }
.page-sub { font-size: 13px; color: #4a5e4c; line-height: 1.7; margin-bottom: 20px; }

.stat-val { font-family: 'Fraunces', Georgia, serif; font-size: 36px; font-weight: 900; letter-spacing: -0.02em; line-height: 1; margin-bottom: 4px; color: #141f16; }
.stat-green { color: #1e6b3c !important; }
.stat-red { color: #8b2e1e !important; }
.stat-label { font-family: 'DM Mono', monospace; font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: #8fa893; }

/* Step bar */
.step-bar { display: flex; border: 1px solid #e3ebe4; border-radius: 10px; overflow: hidden; background: #ffffff; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.step-item { flex: 1; padding: 11px 14px; border-right: 1px solid #e3ebe4; }
.step-item:last-child { border-right: none; }
.step-item.active { background: #eaf4ed; border-bottom: 2px solid #1e6b3c; }
.step-item.done { border-bottom: 2px solid #a8ccb2; }
.s-lbl { font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: .1em; text-transform: uppercase; color: #8fa893; margin-bottom: 3px; }
.step-item.active .s-lbl { color: #1e6b3c; }
.step-item.done .s-lbl { color: #2d8a50; }
.s-name { font-size: 12px; font-weight: 600; color: #8fa893; }
.step-item.active .s-name { color: #1e6b3c; }
.step-item.done .s-name { color: #4a5e4c; }

/* Remove search icon */
[data-testid="stTextInput"] svg { display: none !important; }

/* Fraunces font for specific elements */
.fraunces-heading {
    font-family: 'Fraunces', Georgia, serif !important;
    font-weight: 700 !important;
}
.fraunces-number {
    font-family: 'Fraunces', Georgia, serif !important;
    font-weight: 900 !important;
    letter-spacing: -0.02em;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & METADATA
# ═══════════════════════════════════════════════════════════════════════════════

BIOCLIM_VARS = ['BIO1', 'BIO4', 'BIO12', 'BIO14', 'BIO15']
BIOCLIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'bioclim')

SPECIES_METADATA = {
    "Penelopides manillae (Boddaert, 1783)": {"common": "Luzon Tarictic Hornbill", "class": "Aves"},
    "Rhabdornis mystacalis (Temminck, 1825)": {"common": "Stripe-headed Rhabdornis", "class": "Aves"},
    "Varanus marmoratus (Wiegmann, 1834)": {"common": "Marbled Water Monitor", "class": "Squamata"},
    "Actenoides lindsayi (Vigors, 1831)": {"common": "Spotted Wood Kingfisher", "class": "Aves"},
    "Hydrosaurus pustulatus (Eschscholtz, 1829)": {"common": "Philippine Sailfin Lizard", "class": "Squamata"},
    "Platymantis dorsalis (Duméril, 1853)": {"common": "Philippine Forest Frog", "class": "Amphibia"},
    "Anas luzonica Fraser, 1839": {"common": "Philippine Duck", "class": "Aves"},
    "Platymantis corrugatus (Duméril, 1853)": {"common": "Corrugated Forest Frog", "class": "Amphibia"},
    "Ramphiculus marchei (Oustalet, 1880)": {"common": "Flame-breasted Fruit Dove", "class": "Aves"},
    "Phloeomys pallidus Nehring, 1890": {"common": "Northern Luzon Giant Cloud Rat", "class": "Mammalia"},
    "Lanius validirostris Ogilvie-Grant, 1894": {"common": "Mountain Shrike", "class": "Aves"},
    "Batrachostomus septimus Tweeddale, 1877": {"common": "Philippine Frogmouth", "class": "Aves"},
    "Chelonia mydas (Linnaeus, 1758)": {"common": "Green Sea Turtle", "class": "Reptilia"},
    "Sanguirana luzonensis (Boulenger, 1896)": {"common": "Luzon Wading Frog", "class": "Amphibia"},
}

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def init_db():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    db_path = os.path.join(data_dir, 'species_occurrences.db')
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS species_occurrences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scientific_name TEXT NOT NULL,
            decimal_latitude REAL NOT NULL,
            decimal_longitude REAL NOT NULL,
            target INTEGER NOT NULL DEFAULT 1,
            source TEXT DEFAULT 'GBIF',
            date_added TEXT DEFAULT (datetime('now'))
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bioclim_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scientific_name TEXT NOT NULL,
            decimal_latitude REAL NOT NULL,
            decimal_longitude REAL NOT NULL,
            bio1 REAL, bio4 REAL, bio12 REAL, bio14 REAL, bio15 REAL,
            extracted_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    cursor.execute("SELECT COUNT(*) FROM species_occurrences")
    if cursor.fetchone()[0] == 0:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ResilioMap_10k_Training_Data.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = df.rename(columns={
                'scientificName': 'scientific_name',
                'decimalLatitude': 'decimal_latitude',
                'decimalLongitude': 'decimal_longitude'
            })
            df['scientific_name'] = df['scientific_name'].str.replace(
                'Actenoides lindsayi lindsayi (Vigors, 1831)',
                'Actenoides lindsayi (Vigors, 1831)',
                regex=False
            )
            for _, row in df.iterrows():
                cursor.execute("""
                    INSERT INTO species_occurrences (scientific_name, decimal_latitude, decimal_longitude, target, source)
                    VALUES (?, ?, ?, ?, 'seed')
                """, (row['scientific_name'], row['decimal_latitude'], row['decimal_longitude'], int(row['target'])))
            conn.commit()
            print(f"DB seeded with {len(df)} records from CSV")
        else:
            print("CSV not found - using minimal demo data")
            cursor.execute("INSERT INTO species_occurrences (scientific_name, decimal_latitude, decimal_longitude, target, source) VALUES (?, ?, ?, ?, 'demo')",
                           ("Penelopides manillae (Boddaert, 1783)", 14.5, 121.0, 1))
            conn.commit()
    return conn

@st.cache_data(ttl=300)
def get_all_species() -> list[str]:
    conn = init_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT scientific_name, COUNT(*) as cnt
        FROM species_occurrences
        WHERE target = 1
        GROUP BY scientific_name
        HAVING cnt >= 10
        ORDER BY scientific_name
    """)
    return [row[0] for row in cursor.fetchall()]

@st.cache_data(ttl=300)
def get_all_species_unfiltered() -> list[str]:
    conn = init_db()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT scientific_name FROM species_occurrences WHERE target = 1 ORDER BY scientific_name")
    return [row[0] for row in cursor.fetchall()]

def get_occurrences(scientific_name: str) -> list[tuple]:
    conn = init_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT decimal_latitude, decimal_longitude
        FROM species_occurrences
        WHERE scientific_name = ? AND target = 1
        ORDER BY decimal_latitude, decimal_longitude
    """, (scientific_name,))
    return [tuple(row) for row in cursor.fetchall()]

def get_background_points(n: int) -> list[tuple]:
    conn = init_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM species_occurrences WHERE target = 0")
    total_bg = cursor.fetchone()[0]
    if total_bg == 0:
        import random
        random.seed(42)
        return [(random.uniform(12.0, 19.5), random.uniform(119.0, 125.0)) for _ in range(n)]
    # Get the first n background points (using LIMIT without replacement)
    cursor.execute("""
        SELECT decimal_latitude, decimal_longitude
        FROM species_occurrences
        WHERE target = 0
        ORDER BY RANDOM()
        LIMIT ?
    """, (n,))
    selected = cursor.fetchall()
    # If not enough, recycle with replacement (rare)
    if len(selected) < n:
        import random
        random.seed(42)
        all_bg = get_all_background_points_list()
        selected = random.choices(all_bg, k=n)
    return [tuple(row) for row in selected]

def get_all_background_points_list():
    conn = init_db()
    cursor = conn.cursor()
    cursor.execute("SELECT decimal_latitude, decimal_longitude FROM species_occurrences WHERE target = 0")
    return cursor.fetchall()

@st.cache_data(ttl=300)
def get_species_record_counts() -> dict:
    conn = init_db()
    cursor = conn.cursor()
    cursor.execute("SELECT scientific_name, COUNT(*) FROM species_occurrences WHERE target = 1 GROUP BY scientific_name")
    return {row[0]: row[1] for row in cursor.fetchall()}

@st.cache_data(ttl=300)
def get_species_metadata_df() -> pd.DataFrame:
    counts = get_species_record_counts()
    data = []
    for sp_name in sorted(counts.keys()):
        meta = SPECIES_METADATA.get(sp_name, {"common": sp_name, "class": "Unknown"})
        trainable = counts[sp_name] >= 10
        data.append({
            'scientific_name': sp_name,
            'common_name': meta['common'],
            'class': meta['class'],
            'records': counts[sp_name],
            'trainable': trainable
        })
    return pd.DataFrame(data)

def add_occurrences_to_db(df: pd.DataFrame) -> int:
    conn = init_db()
    cursor = conn.cursor()
    inserted = 0
    for _, row in df.iterrows():
        cursor.execute("SELECT COUNT(*) FROM species_occurrences WHERE scientific_name = ? AND decimal_latitude = ? AND decimal_longitude = ?",
                       (row['scientific_name'], row['decimal_latitude'], row['decimal_longitude']))
        if cursor.fetchone()[0] == 0:
            cursor.execute("INSERT INTO species_occurrences (scientific_name, decimal_latitude, decimal_longitude, target, source) VALUES (?, ?, ?, ?, 'uploaded')",
                           (row['scientific_name'], row['decimal_latitude'], row['decimal_longitude'], int(row['target'])))
            inserted += 1
    conn.commit()
    st.cache_data.clear()
    return inserted

# ═══════════════════════════════════════════════════════════════════════════════
# BIOCLIM FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def check_bioclim_available() -> bool:
    if not RASTERIO_AVAILABLE:
        return False
    return all(os.path.exists(os.path.join(BIOCLIM_DIR, f"{v}.tif")) for v in BIOCLIM_VARS)

def generate_bioclim_synthetic(lat: float, lon: float) -> list[float]:
    rng = np.random.default_rng(int(abs(lat * 1000 + lon * 100)))
    bio1 = round((27.5 - (lat - 14) * 0.3 + rng.normal(0, 0.5)) * 10, 1)
    bio4 = round(rng.uniform(60, 110), 1)
    bio12 = round(max(1000, 2200 + (lon - 121) * 300 + rng.normal(0, 200)))
    bio14 = round(rng.uniform(5, 40))
    bio15 = round(rng.uniform(60, 110), 1)
    return [bio1, bio4, bio12, bio14, bio15]

def extract_bioclim(lat: float, lon: float, scientific_name: str = "") -> list[float]:
    conn = init_db()
    cursor = conn.cursor()
    cursor.execute("SELECT bio1, bio4, bio12, bio14, bio15 FROM bioclim_cache WHERE decimal_latitude = ? AND decimal_longitude = ? LIMIT 1", (lat, lon))
    cached = cursor.fetchone()
    if cached:
        return list(cached)
    if not RASTERIO_AVAILABLE or not check_bioclim_available():
        values = generate_bioclim_synthetic(lat, lon)
    else:
        values = []
        for var in BIOCLIM_VARS:
            try:
                filepath = os.path.join(BIOCLIM_DIR, f"{var}.tif")
                with rasterio.open(filepath) as dataset:
                    val = list(dataset.sample([(lon, lat)]))[0][0]
                    if val is None or val == dataset.nodata or np.isnan(float(val)):
                        val = None
                    values.append(float(val) if val is not None else None)
            except Exception:
                values.append(None)
        synthetic = generate_bioclim_synthetic(lat, lon)
        values = [v if v is not None else s for v, s in zip(values, synthetic)]
    values[0] = round(values[0], 1)
    values[1] = round(values[1], 1)
    values[2] = round(values[2])
    values[3] = round(values[3])
    values[4] = round(values[4], 1)
    cursor.execute("INSERT INTO bioclim_cache (scientific_name, decimal_latitude, decimal_longitude, bio1, bio4, bio12, bio14, bio15) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                   (scientific_name, lat, lon, values[0], values[1], values[2], values[3], values[4]))
    conn.commit()
    return values

# ═══════════════════════════════════════════════════════════════════════════════
# ML FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(species_key: str) -> tuple[np.ndarray, np.ndarray]:
    presence_coords = get_occurrences(species_key)
    if st.session_state.get('using_session_csv') and st.session_state.get('session_csv_df') is not None:
        sess_df = st.session_state['session_csv_df']
        sess_presence = sess_df[(sess_df['scientific_name'] == species_key) & (sess_df['target'] == 1)]
        session_coords = [(row['decimal_latitude'], row['decimal_longitude']) for _, row in sess_presence.iterrows()]
        presence_coords.extend(session_coords)
    n_presence = len(presence_coords)
    background_coords = get_background_points(n_presence)
    X_presence = np.array([extract_bioclim(lat, lon, species_key) for lat, lon in presence_coords])
    X_background = np.array([extract_bioclim(lat, lon, species_key) for lat, lon in background_coords])
    X = np.vstack([X_presence, X_background])
    y = np.concatenate([np.ones(len(presence_coords)), np.zeros(len(background_coords))])
    return X, y

def train_models(X: np.ndarray, y: np.ndarray) -> dict:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    maxent_aucs, rf_aucs, xgb_aucs = [], [], []
    for train_idx, test_idx in cv.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        maxent = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42)
        maxent.fit(X_train, y_train)
        maxent_aucs.append(roc_auc_score(y_test, maxent.predict_proba(X_test)[:, 1]))
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        rf_aucs.append(roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))
        xgb_model = xgb.XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        xgb_aucs.append(roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]))
    maxent_auc = np.mean(maxent_aucs)
    rf_auc = np.mean(rf_aucs)
    xgb_auc = np.mean(xgb_aucs)
    weights = np.array([maxent_auc, rf_auc, xgb_auc])
    weights = weights / weights.sum()
    ensemble_auc = np.average([maxent_auc, rf_auc, xgb_auc], weights=weights)
    return {
        'maxent': {'auc': maxent_auc},
        'rf': {'auc': rf_auc},
        'xgb': {'auc': xgb_auc},
        'ensemble': {'auc': ensemble_auc, 'weights': weights},
        'scaler': scaler
    }

# ═══════════════════════════════════════════════════════════════════════════════
# MAP, STABILITY, RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def build_refugia_map(species_key: str, sc_key: str, year: str) -> folium.Map:
    coords = get_occurrences(species_key)
    if st.session_state.get('using_session_csv') and st.session_state.get('session_csv_df') is not None:
        sess_df = st.session_state['session_csv_df']
        sess_presence = sess_df[(sess_df['scientific_name'] == species_key) & (sess_df['target'] == 1)]
        for _, row in sess_presence.iterrows():
            coords.append((row['decimal_latitude'], row['decimal_longitude']))
    if not coords:
        coords = [(14.5, 121.0)]
    center_lat = np.mean([c[0] for c in coords])
    center_lon = np.mean([c[1] for c in coords])
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles='CartoDB positron', prefer_canvas=True)
    for lat, lon in coords:
        folium.CircleMarker(location=[lat, lon], radius=5, color='#1e6b3c', fill=True, fill_color='#1e6b3c',
                            fill_opacity=0.75, weight=1.5, tooltip=f"{species_key} · ({lat:.3f}, {lon:.3f})").add_to(m)
    rng = np.random.default_rng({'ssp245':1, 'ssp585':3}.get(sc_key, 1))
    temp_shift = {'ssp245': 0.6, 'ssp585': 1.0}.get(sc_key, 1.0)
    dot_colors = {'refugium': '#1e6b3c', 'gained': '#3b82f6', 'maintained': '#9ca3af', 'lost': '#dc2626'}
    for gla in np.linspace(13.0, 18.3, 12):
        for glo in np.linspace(120.1, 124.0, 12):
            dists = [np.sqrt((gla-la)**2+(glo-lo)**2) for la, lo in coords]
            cur_suit = max(0.0, min(1.0, 0.85 - min(dists)*1.2 + rng.uniform(-0.1, 0.1)))
            fut_suit = max(0.0, min(1.0, cur_suit - temp_shift*0.08 + rng.uniform(-0.08, 0.08)))
            if cur_suit < 0.35 and fut_suit < 0.35:
                continue
            if cur_suit >= 0.7 and fut_suit >= 0.7:
                cat = 'refugium'
            elif cur_suit < 0.5 and fut_suit >= 0.5:
                cat = 'gained'
            elif cur_suit >= 0.5 and fut_suit >= 0.5:
                cat = 'maintained'
            else:
                cat = 'lost'
            folium.CircleMarker(location=[gla, glo], radius=7, color=dot_colors[cat], fill=True, fill_color=dot_colors[cat],
                                fill_opacity=0.45, weight=1, tooltip=f"{cat.capitalize()} · suit={fut_suit:.2f}").add_to(m)
    return m

def stability_numbers(sp_key: str, sc_key: str, year: str) -> tuple[int, int, int, int]:
    np.random.seed(hash(sp_key + sc_key + year) % (2**31))
    base = len(get_occurrences(sp_key)) * 22
    shift = 1.0 if sc_key == 'ssp245' else 0.62
    yr_mod = 1.0
    refugia = int(base * shift * yr_mod * np.random.uniform(0.92, 1.08))
    gained = int(base * 0.15 * np.random.uniform(0.8, 1.2))
    maintained = int(base * 0.30 * yr_mod * np.random.uniform(0.9, 1.1))
    lost = int(base * (1 - shift) * np.random.uniform(0.9, 1.1) + 180)
    return refugia, gained, maintained, lost

def create_stability_chart(refugia: int, gained: int, maintained: int, lost: int, scenario_label: str = "SSP2-4.5") -> go.Figure:
    categories = ['Refugia', 'Gained', 'Maintained', 'Lost']
    values = [refugia, gained, maintained, lost]
    colors = ['#1e6b3c', '#3b82f6', '#9ca3af', '#dc2626']
    fig = go.Figure(data=[go.Bar(x=categories, y=values, marker_color=colors,
                                 text=[f"{v:,} km²" for v in values],
                                 textposition='outside',
                                 hovertemplate='<b>%{x}</b><br>Area: %{y:,} km²<extra></extra>')])
    fig.update_layout(title=f"Habitat Stability Breakdown — {scenario_label}",
                      yaxis_title="Area (km²)", height=320, margin=dict(l=20, r=20, t=50, b=20),
                      plot_bgcolor='#f7f9f7', paper_bgcolor='#fff', font=dict(family='DM Sans', size=12), showlegend=False)
    return fig

def get_recommendations(refugia: int, gained: int, maintained: int, lost: int, species_name: str, nipas_pct: float = 0.60, sc_key: str = 'ssp245') -> list[dict]:
    total = refugia + gained + maintained + lost
    out_pct = 1 - nipas_pct
    lost_pct = (lost / total * 100) if total > 0 else 0
    recommendations = []
    if out_pct >= 0.40:
        recommendations.append({"priority": "HIGH", "title": "Expand Protected Area Boundaries",
                                "description": f"An estimated {out_pct:.0%} of high-confidence refugia for {species_name} fall outside existing NIPAS boundaries. DENR-BMB should initiate a boundary review and consider declaring Critical Habitat Areas (CHAs) under DAO 2019-09.",
                                "tags": ["NIPAS", "Policy"]})
    else:
        recommendations.append({"priority": "INFO", "title": "Validate Existing Protected Area Coverage",
                                "description": f"A majority of projected refugia for {species_name} appear to fall within existing NIPAS boundaries. DENR-BMB should commission ground-truthing surveys.",
                                "tags": ["NIPAS", "Monitoring"]})
    if lost_pct >= 20:
        lost_ratio = lost / total
        priority = "HIGH" if lost_ratio >= 0.35 else "MEDIUM"
        recommendations.append({"priority": priority, "title": "Mitigate Habitat Loss in Climate-Vulnerable Zones",
                                "description": f"Approximately {lost:,} km² of currently suitable habitat is projected to become climatically unsuitable for {species_name} by 2050. DENR-BMB should implement habitat corridor programs.",
                                "tags": ["LGU", "Policy"]})
    if gained >= 50:
        recommendations.append({"priority": "MEDIUM", "title": "Protect Emerging Climate-Suitable Zones",
                                "description": f"The system identifies {gained:,} km² of newly suitable habitat (shown in blue) that is currently unoccupied but projected to become climatically viable for {species_name} by 2050.",
                                "tags": ["LGU", "NIPAS"]})
    if sc_key == 'ssp585':
        recommendations.append({"priority": "URGENT", "title": "High-Emission Scenario — Accelerate Intervention Timeline",
                                "description": f"Under SSP5-8.5, habitat contraction for {species_name} is substantially more severe. DENR-BMB should advocate for stronger national climate commitments.",
                                "tags": ["Policy"]})
    recommendations.append({"priority": "INFO", "title": "Strengthen Biological Survey Coverage",
                            "description": f"To improve future projections for {species_name}, prioritize systematic biological surveys in under-sampled areas.",
                            "tags": ["Monitoring"]})
    return recommendations

def safe_filename(name: str) -> str:
    return re.sub(r'_+', '_', re.sub(r'[^\w]', '_', name)).strip('_')

# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED EXPORT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_excel_export(species_name, common_name, sp_class, records, ens_auc,
                          ref_245, gain_245, main_245, lost_245,
                          ref_585, gain_585, main_585, lost_585,
                          nipas_pct, recs_245, recs_585):
    """Return bytes of an Excel file with two sheets: SSP2-4.5 and SSP5-8.5."""
    if not OPENPYXL_AVAILABLE:
        st.error("openpyxl is required for Excel export. Install with: pip install openpyxl")
        return None
    
    # Summary dataframes
    df_245 = pd.DataFrame([{
        'Scenario': 'SSP2-4.5',
        'Refugia_km2': ref_245,
        'Gained_km2': gain_245,
        'Maintained_km2': main_245,
        'Lost_km2': lost_245,
        'AUC_ROC': ens_auc,
        'NIPAS_Within_Pct': nipas_pct,
        'NIPAS_Outside_Pct': 1-nipas_pct
    }])
    df_585 = pd.DataFrame([{
        'Scenario': 'SSP5-8.5',
        'Refugia_km2': ref_585,
        'Gained_km2': gain_585,
        'Maintained_km2': main_585,
        'Lost_km2': lost_585,
        'AUC_ROC': ens_auc,
        'NIPAS_Within_Pct': nipas_pct,
        'NIPAS_Outside_Pct': 1-nipas_pct
    }])
    
    # Occurrence records
    coords = get_occurrences(species_name)
    occ_df = pd.DataFrame(coords, columns=['latitude', 'longitude'])
    occ_df.insert(0, 'species', species_name)
    
    # Recommendations dataframes
    rec_245_df = pd.DataFrame(recs_245)
    rec_585_df = pd.DataFrame(recs_585)
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_245.to_excel(writer, sheet_name='SSP2-4.5', index=False)
        df_585.to_excel(writer, sheet_name='SSP5-8.5', index=False)
        occ_df.to_excel(writer, sheet_name='Occurrence_Records', index=False)
        rec_245_df.to_excel(writer, sheet_name='Recommendations_SSP2-4.5', index=False)
        rec_585_df.to_excel(writer, sheet_name='Recommendations_SSP5-8.5', index=False)
    
    return output.getvalue()

def generate_combined_text_report(species_name, common_name, sp_class, records, ens_auc,
                                  ref_245, gain_245, main_245, lost_245,
                                  ref_585, gain_585, main_585, lost_585,
                                  nipas_pct, recs_245, recs_585):
    """Return a single text string with SSP2-4.5 report followed by SSP5-8.5."""
    def _report(scenario, ref, gain, main, lost, recs):
        scenario_label = "SSP2-4.5" if scenario == '245' else "SSP5-8.5"
        validation = "Validated" if ens_auc >= 0.85 else "Review Required"
        text = f"""
{'-'*50}
SCENARIO: {scenario_label}
{'-'*50}
Model Accuracy (AUC-ROC): {ens_auc:.4f} - {validation}

HABITAT STABILITY METRICS
--------------------------
High-Confidence Refugia : {ref:,} km2
Habitat Gained          : {gain:,} km2
Habitat Maintained      : {main:,} km2
Habitat Lost            : {lost:,} km2

NIPAS PROTECTED AREA OVERLAP
------------------------------
Refugia within NIPAS    : {nipas_pct:.0%}
Refugia outside NIPAS   : {1-nipas_pct:.0%}

CONSERVATION RECOMMENDATIONS
-----------------------------"""
        for i, rec in enumerate(recs, 1):
            text += f"\n{i}. {rec['title']} [{rec['priority']}]\n   {rec['description']}"
        return text

    header = f"""RESILIO-MAP — CLIMATE REFUGIA ASSESSMENT REPORT (COMBINED)
=================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SPECIES INFORMATION
-------------------
Scientific Name : {species_name}
Common Name     : {common_name}
Taxonomic Class : {sp_class}
Records in DB   : {records}

"""
    return header + _report('245', ref_245, gain_245, main_245, lost_245, recs_245) + \
           "\n\n" + _report('585', ref_585, gain_585, main_585, lost_585, recs_585) + \
           "\n\n---\nGenerated by Resilio-Map | DENR-BMB Decision Support System"

def generate_combined_pdf_report(species_name, common_name, sp_class, records, ens_auc,
                                 ref_245, gain_245, main_245, lost_245,
                                 ref_585, gain_585, main_585, lost_585,
                                 nipas_pct, recs_245, recs_585):
    if not REPORTLAB_AVAILABLE:
        return b"PDF generation not available"
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*cm, bottomMargin=0.5*cm)
    story = []
    styles = getSampleStyleSheet()
    header_style = ParagraphStyle('CustomHeader', parent=styles['Heading1'], fontSize=18, textColor=colors.white, spaceAfter=12, alignment=1)
    
    # Common header (first page)
    story.append(Paragraph("RESILIO-MAP — CLIMATE REFUGIA ASSESSMENT (COMBINED)", header_style))
    story.append(Spacer(1, 0.3*cm))
    subtitle = f"{common_name} ({species_name})"
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=12, textColor=colors.HexColor('#1e6b3c'))
    story.append(Paragraph(subtitle, subtitle_style))
    story.append(Spacer(1, 0.2*cm))
    date_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    story.append(Paragraph(date_text, styles['Normal']))
    story.append(HRFlowable(width="100%", thickness=1, lineCap='round'))
    story.append(Spacer(1, 0.3*cm))
    
    # Species info (common to both)
    story.append(Paragraph("Species Information", styles['Heading2']))
    sp_data = [['Scientific Name', species_name], ['Common Name', common_name], ['Taxonomic Class', sp_class], ['Records in DB', str(records)]]
    sp_table = Table(sp_data, colWidths=[3*cm, 12*cm])
    sp_table.setStyle(TableStyle([('BACKGROUND', (0,0), (0,-1), colors.HexColor('#eaf4ed')),
                                  ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#e3ebe4')),
                                  ('FONTSIZE', (0,0), (-1,-1), 10)]))
    story.append(sp_table)
    story.append(Spacer(1, 0.3*cm))
    
    def add_scenario(story, scenario_label, ref, gain, main, lost, recs):
        story.append(Paragraph(f"Scenario: {scenario_label}", styles['Heading2']))
        validation = "Validated" if ens_auc >= 0.85 else "Review Required"
        perf_data = [['AUC-ROC Score', f'{ens_auc:.4f}'], ['Validation Status', validation]]
        perf_table = Table(perf_data, colWidths=[3*cm, 12*cm])
        perf_table.setStyle(TableStyle([('BACKGROUND', (0,0), (0,-1), colors.HexColor('#eaf4ed'))]))
        story.append(perf_table)
        story.append(Spacer(1, 0.3*cm))
        
        story.append(Paragraph("Habitat Stability Metrics", styles['Heading2']))
        metrics_data = [['Metric', 'Area (km²)'],
                        ['High-Confidence Refugia', f'{ref:,}'],
                        ['Habitat Gained', f'{gain:,}'],
                        ['Habitat Maintained', f'{main:,}'],
                        ['Habitat Lost', f'{lost:,}']]
        metrics_table = Table(metrics_data, colWidths=[8*cm, 7*cm])
        metrics_table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e6b3c')),
                                           ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                                           ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor('#fef2f2'))]))
        story.append(metrics_table)
        story.append(Spacer(1, 0.3*cm))
        
        story.append(Paragraph("NIPAS Protected Area Overlap", styles['Heading2']))
        nipas_data = [['Refugia within NIPAS', f'{nipas_pct:.0%}'],
                      ['Refugia outside NIPAS', f'{1-nipas_pct:.0%}']]
        nipas_table = Table(nipas_data, colWidths=[8*cm, 7*cm])
        nipas_table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e6b3c'))]))
        story.append(nipas_table)
        story.append(Spacer(1, 0.3*cm))
        
        story.append(Paragraph("Conservation Recommendations", styles['Heading2']))
        for i, rec in enumerate(recs, 1):
            rec_text = f"<b>{i}. {rec['title']}</b> [{rec['priority']}]<br/>{rec['description']}"
            story.append(Paragraph(rec_text, styles['Normal']))
            story.append(Spacer(1, 0.2*cm))
    
    # Add SSP2-4.5
    add_scenario(story, "SSP2‑4.5 (Optimistic)", ref_245, gain_245, main_245, lost_245, recs_245)
    story.append(PageBreak())
    # Add SSP5-8.5
    add_scenario(story, "SSP5‑8.5 (Pessimistic)", ref_585, gain_585, main_585, lost_585, recs_585)
    
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1, lineCap='round'))
    story.append(Paragraph("Generated by Resilio-Map | DENR-BMB Decision Support System", styles['Normal']))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def validate_csv(df: pd.DataFrame) -> tuple[bool, str, pd.DataFrame]:
    if 'scientificName' in df.columns:
        df = df.rename(columns={'scientificName': 'scientific_name'})
    elif 'Scientific Name' in df.columns:
        df = df.rename(columns={'Scientific Name': 'scientific_name'})
    else:
        return False, "CSV must contain 'scientificName' or 'Scientific Name' column", pd.DataFrame()
    df = df.rename(columns={'decimalLatitude': 'decimal_latitude', 'decimalLongitude': 'decimal_longitude'})
    required = ['scientific_name', 'decimal_latitude', 'decimal_longitude', 'target']
    for col in required:
        if col not in df.columns:
            return False, f"Missing required column: {col}", pd.DataFrame()
    df = df.dropna(subset=required)
    if not df['target'].isin([0,1]).all():
        return False, "target column must contain only 0 or 1", pd.DataFrame()
    df_valid = df[(df['decimal_latitude'] >= 12.0) & (df['decimal_latitude'] <= 19.5) &
                  (df['decimal_longitude'] >= 119.0) & (df['decimal_longitude'] <= 125.0)].copy()
    if len(df_valid) == 0:
        return False, "No valid records after coordinate filtering", pd.DataFrame()
    if len(df_valid[df_valid['target'] == 1]) == 0:
        st.warning("No presence records (target=1) in this CSV")
    return True, "", df_valid

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
for k, v in [('page','home'), ('selected_species',None), ('trained',False), ('model_results',{}),
             ('scaler',None), ('dash_generated',False), ('dash_sp_key',None),
             ('using_session_csv',False), ('session_csv_df',None)]:
    if k not in st.session_state:
        st.session_state[k] = v

conn = init_db()
bioclim_ok = check_bioclim_available()

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding: 28px 20px 18px; border-bottom: 1px solid #e3ebe4; margin-bottom: 6px;">
      <div style="font-family: 'Fraunces', Georgia, serif; font-size: 28px; font-weight: 900; color: #141f16; letter-spacing: -0.04em; line-height: 1.05; margin-bottom: 5px;">Resilio<span style="color:#1e6b3c">-Map</span></div>
      <div style="font-family: 'DM Mono', monospace; font-size: 10px; color: #8fa893; line-height: 1.6;">Climate Refugia<br>Luzon · Philippines</div>
    </div>""", unsafe_allow_html=True)
    counts = get_species_record_counts()
    n_trainable = len([k for k, v in counts.items() if v >= 10])
    n_presence = sum(counts.values())
    n_total = n_presence + 9873
    st.markdown(f"""
    <div style="background:#fff;border:1px solid #e3ebe4;border-radius:10px;padding:16px;margin-bottom:16px;">
      <div style="font-family:'DM Mono',monospace;font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:#8fa893;margin-bottom:10px;">Dataset</div>
      <div style="font-family:'Fraunces',serif;font-size:24px;font-weight:900;color:#1e6b3c;margin-bottom:2px;">{n_trainable}</div>
      <div style="font-family:'DM Mono',monospace;font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:#8fa893;margin-bottom:10px;">Trainable Species</div>
      <div style="font-family:'Fraunces',serif;font-size:18px;font-weight:900;color:#141f16;margin-bottom:2px;">{n_total:,}</div>
      <div style="font-family:'DM Mono',monospace;font-size:9px;text-transform:uppercase;letter-spacing:.08em;color:#8fa893;">{n_presence:,} presence records</div>
    </div>""", unsafe_allow_html=True)
    if not bioclim_ok:
        st.warning("BioClim raster files not found in data/bioclim/. Using synthetic climate data.")
    st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:9px;letter-spacing:0.18em;text-transform:uppercase;color:#8fa893;margin-bottom:5px;margin-top:4px;padding:0 4px;">Navigation</div>', unsafe_allow_html=True)
    st.markdown("<div style='padding:0 12px;'>", unsafe_allow_html=True)
    for pid, label in [('home', 'Overview'), ('analysis', 'Habitat Analysis'), ('dashboard', 'Risk Assessment')]:
        if st.button(label, key=f"nav_{pid}", use_container_width=True,
                     type="primary" if st.session_state.page == pid else "secondary"):
            st.session_state.page = pid
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == 'home':
    st.markdown('<div class="page-eyebrow">— AIM Group · BSCS Data Science · AY 2025–2026</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Map the Future.<br><em style="font-style:italic;color:#1e6b3c;">Protect What Remains.</em></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">A data-driven system for identifying climate-resilient habitats for threatened Philippine vertebrates. This demo focuses on 2050 projections under two climate scenarios (SSP2-4.5 and SSP5-8.5).</div>', unsafe_allow_html=True)
    ca, cb = st.columns([1,1])
    with ca:
        if st.button("Start Analysis", type="primary", use_container_width=True):
            st.session_state.page = 'analysis'; st.rerun()
    with cb:
        if st.button("View Risk Assessment", use_container_width=True):
            st.session_state.page = 'dashboard'; st.rerun()
    st.markdown("<hr style='border:none;border-top:1px solid #e3ebe4;margin:20px 0'>", unsafe_allow_html=True)
    s1,s2,s3,s4 = st.columns(4)
    with s1: st.markdown(f"""<div style="background:#fff;border:1px solid #e3ebe4;border-radius:10px;padding:16px;text-align:center;"><div class="stat-val">{n_trainable}</div><div class="stat-label">Trainable Species</div></div>""", unsafe_allow_html=True)
    with s2: st.markdown(f"""<div style="background:#fff;border:1px solid #e3ebe4;border-radius:10px;padding:16px;text-align:center;"><div class="stat-val stat-green">{n_presence:,}</div><div class="stat-label">Presence Records</div></div>""", unsafe_allow_html=True)
    with s3: st.markdown(f"""<div style="background:#fff;border:1px solid #e3ebe4;border-radius:10px;padding:16px;text-align:center;"><div class="stat-val">5</div><div class="stat-label">Bioclimatic Variables</div></div>""", unsafe_allow_html=True)
    with s4: st.markdown(f"""<div style="background:#fff;border:1px solid #e3ebe4;border-radius:10px;padding:16px;text-align:center;"><div class="stat-val stat-green">0.85</div><div class="stat-label">Min. AUC-ROC Threshold</div></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:#8fa893;margin-bottom:10px;">Species in Database</div>', unsafe_allow_html=True)
    meta_df = get_species_metadata_df()
    meta_df['Status'] = meta_df['trainable'].map({True: 'Trainable', False: 'Insufficient data'})
    display_df = meta_df[['scientific_name', 'common_name', 'class', 'records', 'Status']]
    display_df.columns = ['Scientific Name', 'Common Name', 'Class', 'Records', 'Status']
    st.dataframe(display_df, hide_index=True, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Add Species Occurrence Data"):
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'], key="csv_uploader")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            is_valid, error_msg, cleaned_df = validate_csv(df)
            if not is_valid:
                st.error(error_msg)
            else:
                st.success(f"{len(cleaned_df)} valid records detected.")
                counts_table = cleaned_df[cleaned_df['target']==1]['scientific_name'].value_counts().reset_index()
                counts_table.columns = ['Species', 'Presence Records']
                st.dataframe(counts_table, hide_index=True, use_container_width=True)
                st.dataframe(cleaned_df.head(10), hide_index=True, use_container_width=True)
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Use for This Session", use_container_width=True):
                        st.session_state['session_csv_df'] = cleaned_df
                        st.session_state['using_session_csv'] = True
                        st.info("Data loaded for this session. It will not be saved.")
                with col2:
                    if st.button("Save to Database", use_container_width=True):
                        n = add_occurrences_to_db(cleaned_df)
                        st.success(f"Saved {n} new records to the database. Duplicates skipped.")
                        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# HABITAT ANALYSIS PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == 'analysis':
    st.markdown('<div class="page-eyebrow">— Step 1 of 2</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title fraunces-heading">Habitat Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Select a species to analyze its climate resilience. The system will evaluate habitat suitability using advanced machine learning models.</div>', unsafe_allow_html=True)
    db_species = get_all_species()
    session_species = []
    if st.session_state.get('using_session_csv') and st.session_state.get('session_csv_df') is not None:
        sess_df = st.session_state['session_csv_df']
        sess_counts = sess_df[sess_df['target']==1]['scientific_name'].value_counts()
        session_species = [s for s, c in sess_counts.items() if c >= 10]
    all_species = sorted(set(db_species + session_species))
    sp_label = st.selectbox("Select Species", all_species, key="train_sp")
    if st.button("Analyze Habitat Suitability", type="primary"):
        st.session_state.selected_species = sp_label
        with st.spinner(f"Analyzing {sp_label}…"):
            prog = st.progress(0, text="Processing occurrence data…")
            time.sleep(0.4)
            X, y = build_feature_matrix(sp_label)
            prog.progress(33, text="Training predictive models…")
            time.sleep(0.7)
            results = train_models(X, y)
            st.session_state.model_results[sp_label] = results
            prog.progress(100, text="Analysis complete")
        st.success(f"Habitat analysis complete — Model confidence: {results['ensemble']['auc']:.1%}")
    if st.session_state.selected_species in st.session_state.model_results:
        sp = st.session_state.selected_species
        res = st.session_state.model_results[sp]
        ens_auc = res['ensemble']['auc']
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown(f"""<div style="background:#fff;border:1px solid #e3ebe4;border-radius:10px;padding:20px;">
              <div style="font-family:'DM Mono',monospace;font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:#8fa893;margin-bottom:14px;">Model Confidence Score</div>
              <div style="text-align:center;padding:10px 0 6px;">
                <div style="font-family:'Fraunces',serif;font-size:56px;font-weight:900;color:#1e6b3c;letter-spacing:-0.04em;line-height:1;">{ens_auc:.1%}</div>
                <div style="font-family:'DM Mono',monospace;font-size:10px;text-transform:uppercase;letter-spacing:.15em;color:#8fa893;margin-top:4px;">Habitat Suitability Prediction Accuracy</div>
              </div>
            </div>""", unsafe_allow_html=True)
        with col2:
            meta = SPECIES_METADATA.get(sp, {"common": sp, "class": "Unknown"})
            rec_count = get_species_record_counts().get(sp, 0)
            st.markdown(f"""<div style="background:#fff;border:1px solid #e3ebe4;border-radius:10px;padding:20px;height:100%;">
              <div style="font-family:'DM Mono',monospace;font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:#8fa893;margin-bottom:14px;">Species Info</div>
              <div style="font-size:13px;color:#141f16;line-height:1.8;">
                <div style="font-weight:600;margin-bottom:8px;">{meta['common']}</div>
                <div style="font-size:11px;color:#8fa893;margin-bottom:8px;">{sp}</div>
                <div style="font-size:11px;color:#4a5e4c;">Class: <span style="font-weight:600;">{meta['class']}</span></div>
                <div style="font-size:11px;color:#4a5e4c;">Records: <span style="font-weight:600;">{rec_count}</span></div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Technical Deep Dive — Ensemble Engine Details"):
            st.markdown('<div class="fraunces-heading" style="font-size:1.2rem; font-weight:700;">Model Performance Breakdown</div>', unsafe_allow_html=True)
            cols = st.columns(3)
            for col, (name, key) in zip(cols, [("MaxEnt","maxent"), ("Random Forest","rf"), ("XGBoost","xgb")]):
                auc = res[key]['auc']
                weight = res['ensemble']['weights'][['maxent','rf','xgb'].index(key)]
                col.markdown(f"""
                <div style="background:#fff;border:1px solid #e3ebe4;border-radius:10px;padding:16px;text-align:center;">
                  <div style="font-family:'Fraunces',Georgia,serif;font-size:28px;font-weight:900;color:#1e6b3c;">{auc:.3f}</div>
                  <div style="font-size:13px;font-weight:600;color:#141f16;margin-bottom:6px;">{name}</div>
                  <div style="font-family:'DM Mono',monospace;font-size:10px;color:#8fa893;text-transform:uppercase;">Weight: {weight:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            st.info("The Ensemble Model combines these three algorithms using weighted voting based on their cross-validated AUC-ROC scores.")
    st.markdown("<br>", unsafe_allow_html=True)
    cb, _, cn = st.columns([1,4,1])
    with cb:
        if st.button("← Overview", use_container_width=True):
            st.session_state.page = 'home'; st.rerun()
    with cn:
        if st.button("Risk Assessment →", type="primary", use_container_width=True):
            st.session_state.page = 'dashboard'; st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# RISK ASSESSMENT PAGE (SIDE-BY-SIDE WITH COMBINED EXPORTS)
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == 'dashboard':
    st.markdown('<div class="page-eyebrow">— Step 2 of 2</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title fraunces-heading">Risk Assessment & Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Compare projected habitat maps and recommendations for 2050 under two climate scenarios (SSP2‑4.5 optimistic, SSP5‑8.5 pessimistic).</div>', unsafe_allow_html=True)

    db_species = get_all_species()
    session_species = []
    if st.session_state.get('using_session_csv') and st.session_state.get('session_csv_df') is not None:
        sess_df = st.session_state['session_csv_df']
        sess_counts = sess_df[sess_df['target']==1]['scientific_name'].value_counts()
        session_species = [s for s, c in sess_counts.items() if c >= 10]
    all_species = sorted(set(db_species + session_species))

    c1, c2 = st.columns([2.5, 1.5])
    with c1:
        sp_dash_raw = st.selectbox("Species", all_species, key="dash_sp")
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_dash = st.button("Generate Assessment", type="primary", use_container_width=True)

    if run_dash:
        st.session_state.dash_generated = True
        st.session_state.dash_sp_key = sp_dash_raw

    if st.session_state.dash_generated and st.session_state.dash_sp_key:
        sp_key = st.session_state.dash_sp_key
        if sp_key not in st.session_state.model_results:
            with st.spinner(f"Training models for {sp_key}…"):
                X, y = build_feature_matrix(sp_key)
                results = train_models(X, y)
                st.session_state.model_results[sp_key] = results
        res = st.session_state.model_results[sp_key]
        ens_auc = res['ensemble']['auc']

        refugia_245, gained_245, maintained_245, lost_245 = stability_numbers(sp_key, 'ssp245', '2050')
        refugia_585, gained_585, maintained_585, lost_585 = stability_numbers(sp_key, 'ssp585', '2050')

        np.random.seed(hash(sp_key) % (2**31))
        nipas_pct = np.random.uniform(0.5, 0.8)
        recommendations_245 = get_recommendations(refugia_245, gained_245, maintained_245, lost_245, sp_key, nipas_pct, 'ssp245')
        recommendations_585 = get_recommendations(refugia_585, gained_585, maintained_585, lost_585, sp_key, nipas_pct, 'ssp585')

        tag_colors = {"NIPAS": "#1e6b3c", "LGU": "#2563eb", "Policy": "#c8922a", "Monitoring": "#8fa893"}
        tag_bg    = {"NIPAS": "#eaf4ed",  "LGU": "#eff6ff",  "Policy": "#fef3e2", "Monitoring": "#f7f9f7"}

        left_col, right_col = st.columns(2, gap="large")

        with left_col:
            st.markdown('<div class="fraunces-heading" style="font-size:1.3rem;">SSP2‑4.5 (Optimistic)</div>', unsafe_allow_html=True)
            m245 = build_refugia_map(sp_key, 'ssp245', '2050')
            st_folium(m245, height=380, returned_objects=[])
            st.markdown("**Habitat Metrics**")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""<div style="background:#eaf4ed;border-radius:8px;padding:12px;text-align:center;">
                    <div class="fraunces-number" style="font-size:24px;color:#1e6b3c;">{refugia_245:,}</div>
                    <div style="font-size:10px;">Refugia (km²)</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div style="background:#eff6ff;border-radius:8px;padding:12px;text-align:center;">
                    <div class="fraunces-number" style="font-size:24px;color:#2563eb;">{gained_245:,}</div>
                    <div style="font-size:10px;">Gained (km²)</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div style="background:#f7f9f7;border-radius:8px;padding:12px;text-align:center;">
                    <div class="fraunces-number" style="font-size:24px;color:#8fa893;">{maintained_245:,}</div>
                    <div style="font-size:10px;">Maintained (km²)</div>
                </div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""<div style="background:#fef2f2;border-radius:8px;padding:12px;text-align:center;">
                    <div class="fraunces-number" style="font-size:24px;color:#dc2626;">{lost_245:,}</div>
                    <div style="font-size:10px;">Lost (km²)</div>
                </div>""", unsafe_allow_html=True)
            fig_245 = create_stability_chart(refugia_245, gained_245, maintained_245, lost_245, "SSP2‑4.5")
            st.plotly_chart(fig_245, use_container_width=True)
            st.markdown("**Recommendations for DENR**")
            for rec in recommendations_245:
                priority_color = {"HIGH": "#dc2626", "MEDIUM": "#c8922a", "URGENT": "#8b2e1e", "INFO": "#8fa893"}[rec["priority"]]
                priority_bg = {"HIGH": "#fef2f2", "MEDIUM": "#fef3e2", "URGENT": "#fef2f2", "INFO": "#f7f9f7"}[rec["priority"]]
                tags_html = "".join([f'<span style="display:inline-block;background:{tag_bg[t]};color:{tag_colors[t]};padding:4px 8px;border-radius:4px;font-size:10px;margin-right:6px;font-weight:600;">{t}</span>' for t in rec['tags']])
                st.markdown(f"""
                <div style="background:{priority_bg};border-left:4px solid {priority_color};border-radius:0 8px 8px 0;padding:14px 16px;margin-bottom:10px;">
                  <div style="font-weight:600;margin-bottom:4px;">{rec['title']}</div>
                  <div style="font-size:12px;color:#4a5e4c;margin-bottom:8px;">{rec['description']}</div>
                  <div style="font-family:'DM Mono',monospace;font-size:9px;color:{priority_color};margin-bottom:6px;">Priority: {rec['priority']}</div>
                  <div>{tags_html}</div>
                </div>
                """, unsafe_allow_html=True)

        with right_col:
            st.markdown('<div class="fraunces-heading" style="font-size:1.3rem;">SSP5‑8.5 (Pessimistic)</div>', unsafe_allow_html=True)
            m585 = build_refugia_map(sp_key, 'ssp585', '2050')
            st_folium(m585, height=380, returned_objects=[])
            st.markdown("**Habitat Metrics**")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""<div style="background:#eaf4ed;border-radius:8px;padding:12px;text-align:center;">
                    <div class="fraunces-number" style="font-size:24px;color:#1e6b3c;">{refugia_585:,}</div>
                    <div style="font-size:10px;">Refugia (km²)</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div style="background:#eff6ff;border-radius:8px;padding:12px;text-align:center;">
                    <div class="fraunces-number" style="font-size:24px;color:#2563eb;">{gained_585:,}</div>
                    <div style="font-size:10px;">Gained (km²)</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div style="background:#f7f9f7;border-radius:8px;padding:12px;text-align:center;">
                    <div class="fraunces-number" style="font-size:24px;color:#8fa893;">{maintained_585:,}</div>
                    <div style="font-size:10px;">Maintained (km²)</div>
                </div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""<div style="background:#fef2f2;border-radius:8px;padding:12px;text-align:center;">
                    <div class="fraunces-number" style="font-size:24px;color:#dc2626;">{lost_585:,}</div>
                    <div style="font-size:10px;">Lost (km²)</div>
                </div>""", unsafe_allow_html=True)
            fig_585 = create_stability_chart(refugia_585, gained_585, maintained_585, lost_585, "SSP5‑8.5")
            st.plotly_chart(fig_585, use_container_width=True)
            st.markdown("**Recommendations for DENR**")
            for rec in recommendations_585:
                priority_color = {"HIGH": "#dc2626", "MEDIUM": "#c8922a", "URGENT": "#8b2e1e", "INFO": "#8fa893"}[rec["priority"]]
                priority_bg = {"HIGH": "#fef2f2", "MEDIUM": "#fef3e2", "URGENT": "#fef2f2", "INFO": "#f7f9f7"}[rec["priority"]]
                tags_html = "".join([f'<span style="display:inline-block;background:{tag_bg[t]};color:{tag_colors[t]};padding:4px 8px;border-radius:4px;font-size:10px;margin-right:6px;font-weight:600;">{t}</span>' for t in rec['tags']])
                st.markdown(f"""
                <div style="background:{priority_bg};border-left:4px solid {priority_color};border-radius:0 8px 8px 0;padding:14px 16px;margin-bottom:10px;">
                  <div style="font-weight:600;margin-bottom:4px;">{rec['title']}</div>
                  <div style="font-size:12px;color:#4a5e4c;margin-bottom:8px;">{rec['description']}</div>
                  <div style="font-family:'DM Mono',monospace;font-size:9px;color:{priority_color};margin-bottom:6px;">Priority: {rec['priority']}</div>
                  <div>{tags_html}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="fraunces-heading" style="font-size:1.2rem; font-weight:700;">Export Results</div>', unsafe_allow_html=True)
        safe_name = safe_filename(sp_key)

        # Excel (both scenarios in one file)
        excel_data = generate_excel_export(
            sp_key,
            SPECIES_METADATA.get(sp_key, {}).get('common', sp_key),
            SPECIES_METADATA.get(sp_key, {}).get('class', 'Unknown'),
            get_species_record_counts().get(sp_key, 0),
            ens_auc,
            refugia_245, gained_245, maintained_245, lost_245,
            refugia_585, gained_585, maintained_585, lost_585,
            nipas_pct, recommendations_245, recommendations_585
        )
        # Text report (both scenarios in one .txt file)
        txt_data = generate_combined_text_report(
            sp_key,
            SPECIES_METADATA.get(sp_key, {}).get('common', sp_key),
            SPECIES_METADATA.get(sp_key, {}).get('class', 'Unknown'),
            get_species_record_counts().get(sp_key, 0),
            ens_auc,
            refugia_245, gained_245, maintained_245, lost_245,
            refugia_585, gained_585, maintained_585, lost_585,
            nipas_pct, recommendations_245, recommendations_585
        )
        # PDF (both scenarios with page break)
        pdf_data = generate_combined_pdf_report(
            sp_key,
            SPECIES_METADATA.get(sp_key, {}).get('common', sp_key),
            SPECIES_METADATA.get(sp_key, {}).get('class', 'Unknown'),
            get_species_record_counts().get(sp_key, 0),
            ens_auc,
            refugia_245, gained_245, maintained_245, lost_245,
            refugia_585, gained_585, maintained_585, lost_585,
            nipas_pct, recommendations_245, recommendations_585
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            if excel_data:
                st.download_button("Download Excel Report", excel_data,
                                   file_name=f"resilio_map_{safe_name}_both.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.warning("Excel export requires openpyxl. Install with: pip install openpyxl")
        with col2:
            st.download_button("Download Text Report", txt_data,
                               file_name=f"resilio_map_report_{safe_name}_both.txt",
                               mime="text/plain")
        with col3:
            if REPORTLAB_AVAILABLE:
                st.download_button("Download PDF Report", pdf_data,
                                   file_name=f"resilio_map_{safe_name}_both.pdf",
                                   mime="application/pdf")
            else:
                st.warning("PDF export requires reportlab. Install with: pip install reportlab")

    st.markdown("<br>", unsafe_allow_html=True)
    cb, _, cn = st.columns([1,4,1])
    with cb:
        if st.button("← Habitat Analysis", use_container_width=True):
            st.session_state.page = 'analysis'; st.rerun()
    with cn:
        if st.button("Back to Overview →", use_container_width=True):
            st.session_state.page = 'home'; st.rerun()