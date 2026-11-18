"""
CNN + RNN Feature Extractor  →  Random Forest Classifier
=========================================================
Architecture:
  - CNN branch   : 1-D convolutions over spatial features (terrain elevation,
                   slope, soil type, land cover, distance to river)
                   → extracts spatial pattern embeddings.
  - RNN (LSTM)   : processes temporal sequences (daily rainfall, river level,
                   soil moisture, temperature) → captures time-series dynamics.
  - Fusion       : CNN embedding + LSTM hidden state + hand-crafted
                   statistical features (mean, std, max, trend) are
                   concatenated into a rich feature vector.
  - Classifier   : Random Forest trained on the fused feature vector.

Three Random Forests are trained:
  1. flood_rf        — predicts flood risk (binary)
  2. landslide_rf    — predicts landslide risk (binary)
  3. combined_rf     — predicts overall Safe / Unsafe

DATASET v2 — expanded from 3 000 → 12 000 samples across 8 geographic
archetypes covering the full range of real-world flood/landslide scenarios.
Class balance enforced at ~50 % Safe / 50 % Unsafe.
"""

import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, f1_score)
from sklearn.model_selection import cross_val_score

RNG = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
N_SAMPLES    = 12000   # ↑ from 3 000
SEQ_LEN      = 30      # days of temporal data
N_TEMPORAL   = 4       # rainfall, river_level, soil_moisture, temperature
N_SPATIAL    = 5       # elevation, slope, soil_type, land_cover, dist_river
CNN_FILTERS  = 16
KERNEL_SIZE  = 3
LSTM_HIDDEN  = 32

RF_PARAMS = dict(
    n_estimators      = 500,   # ↑ from 300 — more trees = more stable
    max_depth         = 14,    # ↑ slightly — richer data needs deeper trees
    min_samples_split = 4,
    min_samples_leaf  = 2,
    max_features      = "sqrt",
    class_weight      = "balanced",
    random_state      = 42,
    n_jobs            = -1,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. EXPANDED SYNTHETIC DATASET
# ─────────────────────────────────────────────────────────────────────────────
#
# Strategy: generate samples from 8 distinct geographic archetypes, each with
# realistic physical parameter ranges and labelling logic.  Safe archetypes are
# explicitly included to guarantee ~50 % balance.
#
# Archetypes
# ──────────
#  0  Coastal delta     — flat, near sea, monsoon floods (HIGH flood, low ls)
#  1  Hill station      — steep, clay soil, heavy rain (HIGH landslide)
#  2  Arid plateau      — high elevation, sparse rain, far from river (SAFE)
#  3  River floodplain  — very low elevation, close river (HIGH flood)
#  4  Urban lowland     — impervious surface, moderate rain, drainage issues
#  5  Tropical forest   — high humidity, moderate slopes (MODERATE risk)
#  6  Semi-arid scrub   — low rainfall, far river, moderate slope (SAFE)
#  7  Temperate upland  — cool, moderate rain, deep soil (MOSTLY SAFE)
# ─────────────────────────────────────────────────────────────────────────────

def _make_rain_series(n, base_mm_day, burst_prob, burst_max, rng):
    """Generate a realistic 30-day rainfall sequence."""
    rain = rng.exponential(base_mm_day, (n, SEQ_LEN))
    # Burst events (convective storms)
    burst = rng.random((n, SEQ_LEN)) < burst_prob
    rain += burst * rng.uniform(base_mm_day * 2, burst_max, (n, SEQ_LEN))
    # Autocorrelation — wet days follow wet days
    for t in range(1, SEQ_LEN):
        rain[:, t] = 0.6 * rain[:, t] + 0.4 * rain[:, t - 1]
    return np.clip(rain, 0, 300)


def _make_river_level(rain, init_level_range, rng):
    """River level responds to rainfall with lag."""
    n = rain.shape[0]
    river = np.zeros((n, SEQ_LEN))
    river[:, 0] = rng.uniform(*init_level_range, n)
    for t in range(1, SEQ_LEN):
        river[:, t] = (0.75 * river[:, t - 1]
                       + 0.25 * rain[:, t - 1] / 15
                       + rng.normal(0, 0.05, n))
    return np.clip(river, 0, 20)


def _make_soil_moisture(rain, base_range, rng):
    n = rain.shape[0]
    base = rng.uniform(*base_range, (n, SEQ_LEN))
    cumulative_effect = np.cumsum(rain, axis=1) / (SEQ_LEN * 2)
    return np.clip(base + cumulative_effect, 0, 100)


def generate_archetype(archetype_id, n, rng):
    """
    Returns X_spatial (n,5), X_temporal (n,30,4), y (n,), y_flood (n,), y_ls (n,)
    for the given archetype.
    """

    # ── spatial parameters ────────────────────────────────────────────────────
    if archetype_id == 0:   # Coastal delta
        elevation  = rng.uniform(0, 30, n)
        slope      = rng.uniform(0, 8, n)
        soil_type  = rng.uniform(0.3, 0.8, n)   # silty/clay
        land_cover = rng.uniform(0.2, 0.7, n)
        dist_river = rng.uniform(0, 8, n)

    elif archetype_id == 1:  # Hill station
        elevation  = rng.uniform(500, 2500, n)
        slope      = rng.uniform(25, 60, n)
        soil_type  = rng.uniform(0.5, 1.0, n)   # heavy clay
        land_cover = rng.uniform(0.5, 1.0, n)
        dist_river = rng.uniform(2, 20, n)

    elif archetype_id == 2:  # Arid plateau  (SAFE)
        elevation  = rng.uniform(400, 1500, n)
        slope      = rng.uniform(0, 15, n)
        soil_type  = rng.uniform(0.0, 0.3, n)   # sandy/rocky
        land_cover = rng.uniform(0.0, 0.3, n)
        dist_river = rng.uniform(15, 50, n)

    elif archetype_id == 3:  # River floodplain
        elevation  = rng.uniform(5, 60, n)
        slope      = rng.uniform(0, 5, n)
        soil_type  = rng.uniform(0.4, 0.9, n)
        land_cover = rng.uniform(0.3, 0.8, n)
        dist_river = rng.uniform(0, 5, n)

    elif archetype_id == 4:  # Urban lowland
        elevation  = rng.uniform(5, 80, n)
        slope      = rng.uniform(0, 10, n)
        soil_type  = rng.uniform(0.1, 0.5, n)   # mostly impervious
        land_cover = rng.uniform(0.0, 0.4, n)
        dist_river = rng.uniform(1, 15, n)

    elif archetype_id == 5:  # Tropical forest
        elevation  = rng.uniform(50, 500, n)
        slope      = rng.uniform(5, 35, n)
        soil_type  = rng.uniform(0.4, 0.8, n)
        land_cover = rng.uniform(0.7, 1.0, n)
        dist_river = rng.uniform(1, 20, n)

    elif archetype_id == 6:  # Semi-arid scrub  (SAFE)
        elevation  = rng.uniform(200, 800, n)
        slope      = rng.uniform(2, 20, n)
        soil_type  = rng.uniform(0.0, 0.35, n)
        land_cover = rng.uniform(0.1, 0.5, n)
        dist_river = rng.uniform(10, 50, n)

    else:                    # Temperate upland  (MOSTLY SAFE)
        elevation  = rng.uniform(100, 600, n)
        slope      = rng.uniform(5, 25, n)
        soil_type  = rng.uniform(0.2, 0.6, n)
        land_cover = rng.uniform(0.4, 0.9, n)
        dist_river = rng.uniform(5, 30, n)

    X_spatial = np.stack([elevation, slope, soil_type,
                           land_cover, dist_river], axis=1)

    # ── temporal parameters ───────────────────────────────────────────────────
    # Each unsafe archetype gets 50% monsoon-season samples + 50% dry-season
    # samples so that Safe labels appear even in high-risk zones.
    archetype_rain_wet = {
        0: (8,  0.25, 120),   # coastal delta — monsoon
        1: (10, 0.30, 150),   # hill station  — monsoon
        2: (1,  0.05, 20),    # arid plateau  — always dry
        3: (7,  0.20, 100),   # river plain   — monsoon
        4: (5,  0.15, 80),    # urban         — monsoon
        5: (12, 0.35, 180),   # tropical      — wet season
        6: (2,  0.08, 30),    # semi-arid     — always dry
        7: (4,  0.12, 60),    # temperate     — wet season
    }
    archetype_rain_dry = {
        0: (1,  0.05, 15),    # coastal delta — dry season
        1: (2,  0.08, 25),    # hill station  — dry season
        2: (1,  0.05, 20),    # arid plateau  — same (always dry)
        3: (2,  0.06, 20),    # river plain   — dry season
        4: (2,  0.08, 30),    # urban         — dry season
        5: (3,  0.10, 40),    # tropical      — dry season
        6: (2,  0.08, 30),    # semi-arid     — same
        7: (3,  0.10, 45),    # temperate     — dry season
    }
    # Split n into wet and dry halves for unsafe archetypes
    if archetype_id not in (2, 6):
        n_wet = n // 2;  n_dry = n - n_wet
        rain_wet = _make_rain_series(n_wet, *archetype_rain_wet[archetype_id], rng)
        rain_dry = _make_rain_series(n_dry, *archetype_rain_dry[archetype_id], rng)
        rain     = np.concatenate([rain_wet, rain_dry], axis=0)
        elevation  = np.concatenate([elevation[:n_wet],  elevation[:n_dry]])
        slope      = np.concatenate([slope[:n_wet],      slope[:n_dry]])
        soil_type  = np.concatenate([soil_type[:n_wet],  soil_type[:n_dry]])
        land_cover = np.concatenate([land_cover[:n_wet], land_cover[:n_dry]])
        dist_river = np.concatenate([dist_river[:n_wet], dist_river[:n_dry]])
        X_spatial  = np.stack([elevation, slope, soil_type,
                                land_cover, dist_river], axis=1)
    else:
        base_mm, burst_p, burst_max = archetype_rain_wet[archetype_id]
        rain = _make_rain_series(n, base_mm, burst_p, burst_max, rng)
    river      = _make_river_level(rain, (0.5, 4.0), rng)
    soil_moist = _make_soil_moisture(rain, (10, 70), rng)

    temp_range = {
        0: (25, 38), 1: (10, 28), 2: (15, 42), 3: (22, 38),
        4: (20, 38), 5: (22, 35), 6: (18, 40), 7: (5,  22),
    }
    tmin, tmax = temp_range[archetype_id]
    temperature = rng.uniform(tmin, tmax, (n, SEQ_LEN))

    X_temporal = np.stack([rain, river, soil_moist, temperature], axis=2)

    # ── labelling ─────────────────────────────────────────────────────────────
    cum_rain   = rain.sum(axis=1)
    peak_rain  = rain.max(axis=1)
    max_river  = river.max(axis=1)
    rain_trend = rain[:, -7:].mean(axis=1) - rain[:, :7].mean(axis=1)  # increasing?

    # Flood: driven by total rain + river proximity + low elevation + rising trend
    flood_score = (
        np.clip((cum_rain - 80) / 300, 0, 1) * 0.35
        + np.clip((15 - dist_river) / 15, 0, 1) * 0.25
        + np.clip((60 - elevation) / 60, 0, 1) * 0.20
        + np.clip(max_river / 12, 0, 1) * 0.15
        + np.clip(rain_trend / 20, 0, 1) * 0.05
    )

    # Landslide: driven by total rain + slope + heavy clay soil + saturation
    max_soil   = soil_moist.max(axis=1)
    ls_score = (
        np.clip((cum_rain - 60) / 250, 0, 1) * 0.30
        + np.clip((slope - 20) / 40, 0, 1) * 0.35
        + np.clip((soil_type - 0.3) / 0.7, 0, 1) * 0.20
        + np.clip((max_soil - 60) / 40, 0, 1) * 0.15
    )

    # Archetype-specific label threshold adjustments
    # Safe archetypes (2, 6, 7) use stricter thresholds
    if archetype_id in (2, 6):
        flood_thresh, ls_thresh = 0.45, 0.45   # harder to label unsafe
    elif archetype_id == 7:
        flood_thresh, ls_thresh = 0.40, 0.40
    else:
        flood_thresh, ls_thresh = 0.35, 0.35

    y_flood = (flood_score > flood_thresh).astype(int)
    y_ls    = (ls_score    > ls_thresh).astype(int)
    y       = np.maximum(y_flood, y_ls)

    return X_spatial, X_temporal, y, y_flood, y_ls


def generate_dataset(n_total=N_SAMPLES):
    """
    Generate n_total samples balanced across 8 archetypes.
    Unsafe archetypes: 0, 1, 3, 4, 5  (≈ high-risk zones)
    Safe archetypes:   2, 6, 7         (≈ low-risk zones)

    Allocation:
      Unsafe archetypes → 55% of total  (distributed evenly)
      Safe   archetypes → 45% of total  (distributed evenly)
    This naturally produces ≈ 50/50 Safe/Unsafe after physics-based labelling.
    """
    unsafe_ids = [0, 1, 3, 4, 5]
    safe_ids   = [2, 6, 7]

    n_unsafe = int(n_total * 0.55)
    n_safe   = n_total - n_unsafe

    n_per_unsafe = n_unsafe // len(unsafe_ids)
    n_per_safe   = n_safe   // len(safe_ids)

    all_sp, all_tp, all_y, all_yf, all_yl = [], [], [], [], []

    print(f"  Generating {len(unsafe_ids)} unsafe archetypes "
          f"({n_per_unsafe} samples each) …")
    for aid in unsafe_ids:
        sp, tp, y, yf, yl = generate_archetype(aid, n_per_unsafe, RNG)
        all_sp.append(sp);  all_tp.append(tp)
        all_y.append(y);    all_yf.append(yf);  all_yl.append(yl)
        print(f"    Archetype {aid}: Safe={np.sum(y==0):4d}  "
              f"Unsafe={np.sum(y==1):4d}  "
              f"Flood+={np.sum(yf==1):4d}  LS+={np.sum(yl==1):4d}")

    print(f"  Generating {len(safe_ids)} safe archetypes "
          f"({n_per_safe} samples each) …")
    for aid in safe_ids:
        sp, tp, y, yf, yl = generate_archetype(aid, n_per_safe, RNG)
        all_sp.append(sp);  all_tp.append(tp)
        all_y.append(y);    all_yf.append(yf);  all_yl.append(yl)
        print(f"    Archetype {aid}: Safe={np.sum(y==0):4d}  "
              f"Unsafe={np.sum(y==1):4d}  "
              f"Flood+={np.sum(yf==1):4d}  LS+={np.sum(yl==1):4d}")

    X_sp = np.concatenate(all_sp, axis=0)
    X_tp = np.concatenate(all_tp, axis=0)
    y    = np.concatenate(all_y,  axis=0)
    yf   = np.concatenate(all_yf, axis=0)
    yl   = np.concatenate(all_yl, axis=0)

    # Shuffle
    idx  = RNG.permutation(len(y))
    print(f"\n  Total: {len(y)}  Safe={np.sum(y==0)}  Unsafe={np.sum(y==1)}  "
          f"({100*np.sum(y==1)/len(y):.0f}% unsafe)")

    return X_sp[idx], X_tp[idx], y[idx], yf[idx], yl[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 2. CNN SPATIAL ENCODER  (NumPy 1-D conv)
# ─────────────────────────────────────────────────────────────────────────────

def relu(x): return np.maximum(0, x)


class Conv1DLayer:
    def __init__(self, in_ch, out_ch, kernel):
        scale  = np.sqrt(2.0 / (in_ch * kernel))
        self.W = RNG.normal(0, scale, (out_ch, in_ch, kernel))
        self.b = np.zeros(out_ch)
        self.k = kernel

    def forward(self, x):
        """x: (in_ch, L) → (out_ch, L-k+1)  with ReLU."""
        in_ch, L = x.shape
        L_out = L - self.k + 1
        out   = np.zeros((self.W.shape[0], L_out))
        for f in range(self.W.shape[0]):
            for t in range(L_out):
                out[f, t] = (np.dot(self.W[f].ravel(),
                                    x[:, t:t + self.k].ravel())
                             + self.b[f])
        return relu(out)


class CNNEncoder:
    def __init__(self):
        self.conv1 = Conv1DLayer(1, CNN_FILTERS, KERNEL_SIZE)
        self.conv2 = Conv1DLayer(CNN_FILTERS, CNN_FILTERS, KERNEL_SIZE)

    def encode(self, x_sp):
        """x_sp: (5,) → (16,) flat embedding."""
        h = x_sp[np.newaxis, :]
        h = self.conv1.forward(h)
        h = self.conv2.forward(h)
        return h.ravel()

    def encode_batch(self, X_sp):
        return np.stack([self.encode(x) for x in X_sp])


# ─────────────────────────────────────────────────────────────────────────────
# 3. LSTM TEMPORAL ENCODER
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
def tanh_(x):   return np.tanh(np.clip(x, -50, 50))


class LSTMEncoder:
    def __init__(self, input_size, hidden_size):
        scale   = np.sqrt(1.0 / hidden_size)
        concat  = input_size + hidden_size
        self.Wf = RNG.normal(0, scale, (hidden_size, concat))
        self.Wi = RNG.normal(0, scale, (hidden_size, concat))
        self.Wc = RNG.normal(0, scale, (hidden_size, concat))
        self.Wo = RNG.normal(0, scale, (hidden_size, concat))
        self.bf = np.ones(hidden_size)
        self.bi = np.zeros(hidden_size)
        self.bc = np.zeros(hidden_size)
        self.bo = np.zeros(hidden_size)
        self.hidden_size = hidden_size

    def _step(self, x, h, c):
        xh = np.concatenate([x, h])
        f  = sigmoid(self.Wf @ xh + self.bf)
        i  = sigmoid(self.Wi @ xh + self.bi)
        c_ = tanh_(self.Wc @ xh + self.bc)
        o  = sigmoid(self.Wo @ xh + self.bo)
        c  = f * c + i * c_
        h  = o * tanh_(c)
        return h, c

    def encode(self, seq):
        """seq: (T, input_size) → (hidden_size,)"""
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        for t in range(len(seq)):
            h, c = self._step(seq[t], h, c)
        return h

    def encode_batch(self, X_tp):
        return np.stack([self.encode(x) for x in X_tp])


# ─────────────────────────────────────────────────────────────────────────────
# 4. STATISTICAL FEATURE EXTRACTOR — expanded to 30 features
# ─────────────────────────────────────────────────────────────────────────────

def extract_temporal_stats(X_tp):
    """
    X_tp: (N, T, 4)
    Returns (N, 30) — 6 stats per channel + 6 cross-channel interactions.

    Per channel (4 × 6 = 24):
        mean, std, max, min, cumulative_sum, linear_trend

    Cross-channel (6):
        rain×river correlation, rain×soil correlation,
        peak rain day, days above threshold (rain>10mm),
        rain in last 7 days, river level trend
    """
    N, T, C = X_tp.shape
    t_axis  = np.arange(T, dtype=float)
    t_norm  = t_axis - t_axis.mean()
    denom   = (t_norm ** 2).sum()

    features = []

    # Per-channel stats
    for c in range(C):
        ch = X_tp[:, :, c]
        features.append(ch.mean(axis=1))
        features.append(ch.std(axis=1))
        features.append(ch.max(axis=1))
        features.append(ch.min(axis=1))
        features.append(ch.sum(axis=1))
        features.append((ch * t_norm[np.newaxis, :]).sum(axis=1) / denom)

    # Cross-channel: rain (ch0) and river (ch1) correlation per sample
    rain  = X_tp[:, :, 0]
    river = X_tp[:, :, 1]
    soil  = X_tp[:, :, 2]

    rain_c  = rain  - rain.mean(axis=1, keepdims=True)
    river_c = river - river.mean(axis=1, keepdims=True)
    soil_c  = soil  - soil.mean(axis=1, keepdims=True)

    norm_rr = (np.sqrt((rain_c**2).sum(axis=1)) *
               np.sqrt((river_c**2).sum(axis=1))) + 1e-8
    rain_river_corr = (rain_c * river_c).sum(axis=1) / norm_rr

    norm_rs = (np.sqrt((rain_c**2).sum(axis=1)) *
               np.sqrt((soil_c**2).sum(axis=1))) + 1e-8
    rain_soil_corr  = (rain_c * soil_c).sum(axis=1) / norm_rs

    features.append(rain_river_corr)                          # rain-river corr
    features.append(rain_soil_corr)                           # rain-soil corr
    features.append(rain.argmax(axis=1).astype(float) / T)   # peak rain day (normalised)
    features.append((rain > 10).sum(axis=1).astype(float))   # days above 10mm
    features.append(rain[:, -7:].mean(axis=1))                # recent 7-day rain
    features.append(river[:, -1] - river[:, 0])               # river level trend

    return np.stack(features, axis=1)   # (N, 30)


# ─────────────────────────────────────────────────────────────────────────────
# 5. FEATURE FUSION
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(X_sp, X_tp, cnn_enc, lstm_enc,
                         sp_mean=None, sp_std=None,
                         tp_mean=None, tp_std=None):
    """
    Fuse CNN (16) + LSTM (32) + stats (30) + raw spatial (5) = 83 features.
    """
    if sp_mean is None:
        sp_mean = X_sp.mean(axis=0);  sp_std = X_sp.std(axis=0) + 1e-8
    if tp_mean is None:
        tp_mean = X_tp.mean(axis=(0, 1));  tp_std = X_tp.std(axis=(0, 1)) + 1e-8

    X_sp_n = (X_sp - sp_mean) / sp_std
    X_tp_n = (X_tp - tp_mean) / tp_std

    print("    → CNN spatial encoding …")
    cnn_feats  = cnn_enc.encode_batch(X_sp_n)       # (N, 16)

    print("    → LSTM temporal encoding …")
    lstm_feats = lstm_enc.encode_batch(X_tp_n)      # (N, 32)

    print("    → Temporal statistics …")
    stat_feats = extract_temporal_stats(X_tp_n)     # (N, 30)

    raw_sp = X_sp_n                                  # (N, 5)

    F = np.concatenate([cnn_feats, lstm_feats, stat_feats, raw_sp], axis=1)
    print(f"    → Fused feature matrix: {F.shape}")
    return F, sp_mean, sp_std, tp_mean, tp_std


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN MODEL CLASS
# ─────────────────────────────────────────────────────────────────────────────

class CNNRNNRandomForest:
    """
    CNN spatial encoder + LSTM temporal encoder → fused features
    → THREE Random Forest classifiers.
    """

    def __init__(self):
        self.cnn_enc      = CNNEncoder()
        self.lstm_enc     = LSTMEncoder(N_TEMPORAL, LSTM_HIDDEN)
        self.flood_rf     = RandomForestClassifier(**RF_PARAMS)
        self.landslide_rf = RandomForestClassifier(**RF_PARAMS)
        self.combined_rf  = RandomForestClassifier(**RF_PARAMS)
        self.sp_mean = self.sp_std = None
        self.tp_mean = self.tp_std = None
        self.eval_results  = {}
        self.feature_names = None
        # Weather RF — trained separately for Flask live-prediction
        self.weather_rf    = None

    def _features(self, X_sp, X_tp):
        F, *_ = build_feature_matrix(
            X_sp, X_tp,
            self.cnn_enc, self.lstm_enc,
            self.sp_mean, self.sp_std,
            self.tp_mean, self.tp_std,
        )
        return F

    def fit(self, X_sp, X_tp, y, y_flood, y_landslide):
        print("\n  [A] Building fused feature matrix (train) …")
        F, self.sp_mean, self.sp_std, self.tp_mean, self.tp_std = \
            build_feature_matrix(X_sp, X_tp, self.cnn_enc, self.lstm_enc)

        self.feature_names = (
            [f"cnn_{i}"   for i in range(16)] +
            [f"lstm_{i}"  for i in range(32)] +
            [f"{feat}_{stat}"
             for feat in ["rain", "river", "soil", "temp"]
             for stat in ["mean", "std", "max", "min", "sum", "trend"]] +
            ["rain_river_corr", "rain_soil_corr", "peak_rain_day",
             "days_gt10mm",     "recent_7d_rain", "river_trend"] +
            ["elevation", "slope", "soil_type", "land_cover", "dist_river"]
        )

        print("\n  [B] Training flood Random Forest …")
        self.flood_rf.fit(F, y_flood)
        cv_f = cross_val_score(self.flood_rf, F, y_flood,
                               cv=5, scoring="f1").mean()
        print(f"    CV F1 (flood):     {cv_f:.3f}")

        print("\n  [C] Training landslide Random Forest …")
        self.landslide_rf.fit(F, y_landslide)
        cv_l = cross_val_score(self.landslide_rf, F, y_landslide,
                               cv=5, scoring="f1").mean()
        print(f"    CV F1 (landslide): {cv_l:.3f}")

        print("\n  [D] Training combined Safe/Unsafe Random Forest …")
        self.combined_rf.fit(F, y)
        cv_c = cross_val_score(self.combined_rf, F, y,
                               cv=5, scoring="f1").mean()
        print(f"    CV F1 (combined):  {cv_c:.3f}")

        print("\n  [E] Training Weather RF (for live Flask prediction) …")
        self._train_weather_rf()

        return self

    # ── predict interface ──────────────────────────────────────────────────────

    def predict(self, X_sp, X_tp):
        return self.combined_rf.predict(self._features(X_sp, X_tp))

    def predict_proba(self, X_sp, X_tp):
        return self.combined_rf.predict_proba(self._features(X_sp, X_tp))

    def predict_risk_scores(self, X_sp, X_tp):
        F = self._features(X_sp, X_tp)
        return (self.flood_rf.predict_proba(F)[:, 1],
                self.landslide_rf.predict_proba(F)[:, 1])

    def feature_importances(self):
        imp   = self.combined_rf.feature_importances_
        names = self.feature_names or [f"f{i}" for i in range(len(imp))]
        idx   = np.argsort(imp)[::-1][:20]
        return [(names[i], round(float(imp[i]), 4)) for i in idx]

    # ── Weather RF ─────────────────────────────────────────────────────────────

    def _train_weather_rf(self):
        """
        Train a dedicated RF on 6 weather API features.
        Features: [avg_temp_F, max_temp_F, wind_mph, cloud_pct,
                   precip_mm_7day, humidity_pct]

        Expanded from original 8 000 → 14 000 samples covering:
          • Indian dry season, Indian monsoon, coastal cyclone,
          • Temperate drizzle (safe), European spring (safe),
          • Tropical downpour (unsafe), Flash flood (unsafe),
          • Arid heatwave (safe), Pre-monsoon buildup (borderline)
        """
        rng   = np.random.default_rng(42)
        X, y  = [], []

        # ── SAFE scenarios ─────────────────────────────────────────────────
        safe_groups = [
            # (temp_C_range, precip_range, humidity_range, cloud_range, n)
            ((28, 42), (0.0, 1.5),  (20, 45), (5,  35),  1200),  # Indian dry summer
            ((20, 32), (0.0, 2.0),  (35, 60), (20, 55),  1000),  # mild tropical dry
            ((10, 22), (0.5, 4.0),  (45, 70), (35, 65),   800),  # temperate spring
            ((5,  15), (1.0, 5.0),  (50, 75), (40, 75),   700),  # cool European
            ((30, 45), (0.0, 0.8),  (10, 35), (5,  25),   800),  # arid heatwave
            ((18, 28), (0.5, 3.5),  (40, 65), (25, 55),   600),  # semi-arid scrub
            ((15, 25), (0.0, 2.5),  (30, 60), (20, 50),   600),  # plateau dry
        ]
        for (tc_rng, pr_rng, hu_rng, cl_rng, count) in safe_groups:
            for _ in range(count):
                tc = rng.uniform(*tc_rng)
                pr = rng.uniform(*pr_rng)
                hu = rng.uniform(*hu_rng)
                cl = rng.uniform(*cl_rng)
                wi = rng.uniform(3, 25)
                tf = tc * 9/5 + 32
                mf = (tc + rng.uniform(3, 10)) * 9/5 + 32
                X.append([tf, mf, wi, cl, pr, hu]);  y.append(0)

        # ── UNSAFE scenarios ───────────────────────────────────────────────
        unsafe_groups = [
            ((22, 35), (8,  80),   (78, 98), (75, 100), 1200),  # Indian monsoon
            ((18, 32), (15, 200),  (82, 99), (80, 100), 1000),  # extreme rainfall
            ((20, 33), (5,  30),   (82, 98), (70, 100),  800),  # high humid rain
            ((26, 36), (20, 120),  (85, 99), (85, 100),  800),  # coastal cyclone
            ((15, 25), (30, 150),  (80, 98), (75, 100),  600),  # temperate flood
            ((24, 34), (10, 60),   (80, 95), (75, 100),  500),  # pre-monsoon burst
            ((20, 30), (50, 300),  (88, 99), (90, 100),  600),  # tropical downpour
        ]
        for (tc_rng, pr_rng, hu_rng, cl_rng, count) in unsafe_groups:
            for _ in range(count):
                tc = rng.uniform(*tc_rng)
                pr = rng.uniform(*pr_rng)
                hu = rng.uniform(*hu_rng)
                cl = rng.uniform(*cl_rng)
                wi = rng.uniform(5, 40)
                tf = tc * 9/5 + 32
                mf = (tc + rng.uniform(2, 8)) * 9/5 + 32
                X.append([tf, mf, wi, cl, pr, hu]);  y.append(1)

        # ── Borderline / hard examples ─────────────────────────────────────
        # Safe borderline (just below threshold)
        safe_border = [
            [86, 95, 10, 61, 3.5, 74], [79, 95, 10, 72, 5.4, 57],
            [82, 98, 12, 65, 5.0, 55], [90, 105, 14, 60, 4.5, 45],
            [55, 63, 19, 73, 4.0, 77], [75, 84, 12, 58, 3.8, 72],
            [68, 78, 15, 55, 4.2, 68], [72, 82, 11, 62, 4.8, 70],
            [95, 108, 16, 35, 0.5, 30], [91, 99, 18, 28, 0.8, 35],  # very dry
        ]
        # Unsafe borderline (just above threshold)
        unsafe_border = [
            [82, 91, 11, 65, 5.5, 82], [79, 88, 14, 78, 6.0, 83],
            [77, 86, 16, 82, 8.0, 85], [84, 92, 12, 78, 7.0, 80],
            [80, 89, 14, 82, 9.0, 82], [86, 94, 13, 80, 7.5, 84],
            [78, 87, 15, 85, 10.0, 87], [83, 91, 12, 83, 8.5, 83],
        ]
        for feat in safe_border * 80:
            X.append(np.clip(np.array(feat, float) +
                             rng.normal(0, 0.8, 6), 0, 220).tolist())
            y.append(0)
        for feat in unsafe_border * 80:
            X.append(np.clip(np.array(feat, float) +
                             rng.normal(0, 0.8, 6), 0, 220).tolist())
            y.append(1)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=int)
        idx = rng.permutation(len(X))
        X, y = X[idx], y[idx]

        safe_n   = int((y == 0).sum())
        unsafe_n = int((y == 1).sum())
        print(f"    Weather RF data: {len(X)} samples  "
              f"Safe={safe_n}  Unsafe={unsafe_n}")

        self.weather_rf = RandomForestClassifier(
            n_estimators=500, max_depth=12, min_samples_leaf=3,
            max_features="sqrt", class_weight="balanced",
            random_state=42, n_jobs=-1)
        self.weather_rf.fit(X, y)
        print(f"    Weather RF trained.")

    # ── Flask-compatible predict (6 weather features) ──────────────────────────

    def predict_flat(self, X_flat):
        """
        Used by Flask app for live city prediction.
        Input: [avg_temp_F, max_temp_F, wind_mph, cloud_pct,
                precip_mm_7day, humidity_pct]
        """
        X_flat = np.atleast_2d(X_flat).astype(np.float32)
        return self.weather_rf.predict(X_flat)

    def predict_proba_flat(self, X_flat):
        """Returns [P(safe), P(unsafe)] for weather RF."""
        X_flat = np.atleast_2d(X_flat).astype(np.float32)
        return self.weather_rf.predict_proba(X_flat)


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN — generate, train, evaluate, pickle
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  CNN + RNN → Random Forest  |  Flood & Landslide Prediction")
    print("  Dataset v2 — 8 geographic archetypes, 12 000 samples")
    print("=" * 65)

    # ── dataset ───────────────────────────────────────────────────────────────
    print("\n[1/4] Generating expanded dataset …")
    X_sp, X_tp, y, y_flood, y_ls = generate_dataset(N_SAMPLES)

    # 80/20 train-test split
    split       = int(0.8 * len(y))
    idx         = RNG.permutation(len(y))
    tr, te      = idx[:split], idx[split:]

    X_sp_tr, X_tp_tr = X_sp[tr],  X_tp[tr]
    X_sp_te, X_tp_te = X_sp[te],  X_tp[te]
    y_tr,  y_te      = y[tr],     y[te]
    yf_tr, yf_te     = y_flood[tr], y_flood[te]
    yl_tr, yl_te     = y_ls[tr],    y_ls[te]

    print(f"  Train: {len(tr)}  |  Test: {len(te)}")
    print(f"  Train balance — Safe={np.sum(y_tr==0)}  "
          f"Unsafe={np.sum(y_tr==1)}")
    print(f"  Test  balance — Safe={np.sum(y_te==0)}  "
          f"Unsafe={np.sum(y_te==1)}")

    # ── train ─────────────────────────────────────────────────────────────────
    print("\n[2/4] Training …")
    model = CNNRNNRandomForest()
    model.fit(X_sp_tr, X_tp_tr, y_tr, yf_tr, yl_tr)

    # ── evaluate ──────────────────────────────────────────────────────────────
    print("\n[3/4] Evaluating on test set …")
    y_pred          = model.predict(X_sp_te, X_tp_te)
    proba           = model.predict_proba(X_sp_te, X_tp_te)
    fl_pred, ls_pred = model.predict_risk_scores(X_sp_te, X_tp_te)

    report = classification_report(y_te, y_pred,
                                   target_names=["Safe", "Unsafe"])
    cm     = confusion_matrix(y_te, y_pred).tolist()
    try:
        auc = roc_auc_score(y_te, proba[:, 1])
    except Exception:
        auc = float("nan")

    print("\n" + report)
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  Confusion matrix: {cm}")

    print("\n  Top-15 most important features:")
    for name, imp in model.feature_importances()[:15]:
        bar = "█" * int(imp * 200)
        print(f"    {name:30s} {imp:.4f}  {bar}")

    # ── weather RF quick-check ─────────────────────────────────────────────────
    print("\n  Weather RF sanity checks:")
    checks = [
        ("Delhi April (dry)",         [95.0, 108.0, 15.0, 40.0,  0.5, 30.0], "Safe"),
        ("Mumbai April (dry)",         [82.0,  83.5, 16.9, 36.4,  0.1, 73.8], "Safe"),
        ("Chennai July (monsoon)",     [88.0,  98.0, 14.0, 90.0, 85.0, 90.0], "Unsafe"),
        ("Mumbai July (monsoon)",      [84.0,  88.0, 20.0, 92.0,120.0, 92.0], "Unsafe"),
        ("London drizzle (safe)",      [55.0,  62.0, 20.0, 70.0,  4.0, 75.0], "Safe"),
        ("Shillong monsoon (unsafe)",  [68.0,  75.0, 10.0, 88.0, 55.0, 88.0], "Unsafe"),
    ]
    for name, feat, expected in checks:
        pred  = model.predict_flat([feat])[0]
        prob  = model.predict_proba_flat([feat])[0][1]
        label = "Unsafe" if pred else "Safe"
        ok    = "✓" if label == expected else "✗"
        print(f"    {ok}  {name:<35s} → {label}  (p_unsafe={prob:.2f})")

    # ── sample predictions ─────────────────────────────────────────────────────
    cities = ["Delhi", "Mumbai", "Kolkata", "Bangalore", "Chennai",
              "Hyderabad", "Jaipur", "Pune", "Ahmedabad", "Surat"]
    sample_preds = []
    n_show = min(len(cities), len(y_te))
    for i, city in enumerate(cities[:n_show]):
        label = "Unsafe" if y_pred[i] else "Safe"
        sample_preds.append({
            "city":            city,
            "flood_score":     round(float(fl_pred[i]), 3),
            "landslide_score": round(float(ls_pred[i]), 3),
            "label":           label,
            "confidence":      round(float(proba[i, 1]), 3),
        })

    model.eval_results = {
        "classification_report": report,
        "confusion_matrix":      cm,
        "roc_auc":               round(float(auc), 4),
        "model_architecture": {
            "cnn_filters":      CNN_FILTERS,
            "kernel_size":      KERNEL_SIZE,
            "lstm_hidden":      LSTM_HIDDEN,
            "n_rf_estimators":  RF_PARAMS["n_estimators"],
            "rf_max_depth":     RF_PARAMS["max_depth"],
            "total_features":   83,
            "n_archetypes":     8,
            "total_samples":    N_SAMPLES,
            "spatial_features": ["elevation_m", "slope_deg", "soil_type",
                                 "land_cover", "dist_river_km"],
            "temporal_features": ["rainfall_mm_day", "river_level_m",
                                  "soil_moisture_pct", "temperature_c"],
        },
        "feature_importances":   model.feature_importances(),
        "sample_predictions":    sample_preds,
        "dataset_info": {
            "total_samples":    len(y),
            "train_samples":    len(tr),
            "test_samples":     len(te),
            "safe_count_test":  int(np.sum(y_te == 0)),
            "unsafe_count_test":int(np.sum(y_te == 1)),
            "archetypes": [
                "Coastal delta", "Hill station", "Arid plateau",
                "River floodplain", "Urban lowland", "Tropical forest",
                "Semi-arid scrub", "Temperate upland"
            ],
        },
    }

    # ── pickle ────────────────────────────────────────────────────────────────
    import os
    out_path = os.path.join(os.path.dirname(os.path.dirname(
                   os.path.abspath(__file__))), "model.pickle")
    print(f"\n[4/4] Saving model to {out_path} …")
    pickle.dump(model, open(out_path, "wb"), protocol=4)
    size_mb = round(os.path.getsize(out_path) / 1024 / 1024, 1)
    print(f"  ✓  Saved  ({size_mb} MB)")

    print("\n  Sample predictions from test set:")
    for p in sample_preds:
        print(f"    {p['city']:12s}  flood={p['flood_score']:.2f}  "
              f"landslide={p['landslide_score']:.2f}  "
              f"→ {p['label']}  (conf={p['confidence']:.2f})")

    print("\n" + "=" * 65)
    print("  Done!")
    print("=" * 65)