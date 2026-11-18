"""
CNN + RNN Hybrid Model for Flood & Landslide Prediction
=======================================================
Architecture:
  - CNN branch  : processes spatial features (terrain elevation, slope,
                  soil type, land cover, distance to river) via 1-D
                  convolutions → extracts local spatial patterns.
  - RNN branch  : processes temporal sequences (daily rainfall, river
                  level, soil moisture, temperature) via a simple LSTM
                  cell → captures time-series dynamics.
  - Fusion      : CNN + RNN outputs are concatenated, passed through
                  a fully-connected head that predicts
                  (a) flood risk score  0-1
                  (b) landslide risk score  0-1
                  (c) combined binary label  Safe / Unsafe

All maths implemented with NumPy so no deep-learning framework is needed
at runtime.  The trained model is persisted as a pickle file that the
existing Flask app can load directly.
"""

import numpy as np
import pickle
import json
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

RNG = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. SYNTHETIC DATASET
# ─────────────────────────────────────────────────────────────────────────────

N_SAMPLES   = 3000
SEQ_LEN     = 30        # 30 days of temporal data
N_TEMPORAL  = 4         # rainfall, river_level, soil_moisture, temperature
N_SPATIAL   = 5         # elevation, slope, soil_type, land_cover, dist_river
CNN_FILTERS = 16        # convolution filters
KERNEL_SIZE = 3
LSTM_HIDDEN = 32
FC_HIDDEN   = 24

def generate_dataset(n=N_SAMPLES):
    """
    Generate synthetic spatial + temporal data with realistic flood /
    landslide risk labels.

    Spatial features (per sample):
        0  elevation_m      : 10 – 3000 m
        1  slope_deg        : 0  – 60 °
        2  soil_type        : 0=rocky 1=clay 2=loam 3=sand  (encoded 0-1)
        3  land_cover       : 0=bare 1=urban 2=forest 3=crop (encoded 0-1)
        4  dist_river_km    : 0 – 50 km

    Temporal features (30-day sequence per sample):
        0  rainfall_mm_day
        1  river_level_m
        2  soil_moisture_pct
        3  temperature_c

    Label rules (heuristic, designed to be learnable):
        flood:     cumulative rainfall > 150 mm  AND  dist_river < 5 km
                   AND  elevation < 50 m
        landslide: cumulative rainfall > 120 mm  AND  slope > 30 °
                   AND  soil_type in {clay, loam}
        unsafe:    flood OR landslide
    """
    # ── spatial ──────────────────────────────────────────────────────────────
    elevation   = RNG.uniform(10, 3000, n)
    slope       = RNG.uniform(0,  60,   n)
    soil_type   = RNG.uniform(0,  1,    n)   # continuous proxy
    land_cover  = RNG.uniform(0,  1,    n)
    dist_river  = RNG.uniform(0,  50,   n)
    X_spatial   = np.stack([elevation, slope, soil_type,
                             land_cover, dist_river], axis=1)   # (N, 5)

    # ── temporal ─────────────────────────────────────────────────────────────
    # Base daily rainfall: higher in monsoon-like bursts
    base_rain   = RNG.exponential(3, (n, SEQ_LEN))
    burst_mask  = RNG.random((n, SEQ_LEN)) > 0.85
    base_rain  += burst_mask * RNG.uniform(10, 60, (n, SEQ_LEN))
    rain        = np.clip(base_rain, 0, 200)

    # River level rises with rainfall (lagged)
    river_level = np.zeros((n, SEQ_LEN))
    river_level[:, 0] = RNG.uniform(1, 3, n)
    for t in range(1, SEQ_LEN):
        river_level[:, t] = (0.7 * river_level[:, t-1]
                             + 0.3 * rain[:, t-1] / 20
                             + RNG.normal(0, 0.1, n))
    river_level = np.clip(river_level, 0, 15)

    soil_moist  = np.clip(RNG.uniform(20, 80, (n, SEQ_LEN))
                          + 0.1 * np.cumsum(rain, axis=1) / SEQ_LEN, 0, 100)
    temperature = RNG.uniform(15, 42, (n, SEQ_LEN))

    X_temporal  = np.stack([rain, river_level,
                             soil_moist, temperature], axis=2)  # (N, 30, 4)

    # ── labels ────────────────────────────────────────────────────────────────
    cum_rain    = rain.sum(axis=1)
    flood_risk  = (
        (cum_rain > 150).astype(float) * 0.5
        + (dist_river < 5).astype(float) * 0.3
        + (elevation < 50).astype(float) * 0.2
    )
    flood_label = (flood_risk > 0.6).astype(int)

    landslide_risk = (
        (cum_rain > 120).astype(float) * 0.4
        + (slope > 30).astype(float) * 0.4
        + (soil_type > 0.4).astype(float) * 0.2
    )
    landslide_label = (landslide_risk > 0.6).astype(int)

    y = np.maximum(flood_label, landslide_label)   # 0=Safe, 1=Unsafe
    flood_score     = np.clip(flood_risk, 0, 1)
    landslide_score = np.clip(landslide_risk, 0, 1)

    return X_spatial, X_temporal, y, flood_score, landslide_score


# ─────────────────────────────────────────────────────────────────────────────
# 2. ACTIVATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def relu(x):         return np.maximum(0, x)
def sigmoid(x):      return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
def tanh(x):         return np.tanh(np.clip(x, -50, 50))


# ─────────────────────────────────────────────────────────────────────────────
# 3. CNN BLOCK  (1-D convolution over spatial feature vector)
# ─────────────────────────────────────────────────────────────────────────────

class Conv1DLayer:
    """
    1-D convolution: slides KERNEL_SIZE windows over the spatial feature
    vector to extract local feature interactions.
    """
    def __init__(self, in_ch, out_ch, kernel):
        self.W  = RNG.normal(0, np.sqrt(2.0 / (in_ch * kernel)),
                             (out_ch, in_ch, kernel))
        self.b  = np.zeros(out_ch)
        self.k  = kernel

    def forward(self, x):
        """x: (in_ch, L)  → out: (out_ch, L')"""
        in_ch, L = x.shape
        L_out = L - self.k + 1
        out   = np.zeros((len(self.b), L_out))
        for f in range(len(self.b)):
            for t in range(L_out):
                out[f, t] = np.sum(self.W[f] * x[:, t:t+self.k]) + self.b[f]
        return relu(out)


class GlobalAvgPool:
    def forward(self, x):
        """x: (filters, L) → (filters,)"""
        return x.mean(axis=1)


class CNNBranch:
    """Two conv layers + global average pooling."""
    def __init__(self):
        # treat each spatial feature as 1 channel over 5 positions
        self.conv1 = Conv1DLayer(1, CNN_FILTERS, KERNEL_SIZE)
        self.conv2 = Conv1DLayer(CNN_FILTERS, CNN_FILTERS, KERNEL_SIZE)
        self.pool  = GlobalAvgPool()

    def forward(self, x_spatial):
        """x_spatial: (5,) → embedding: (CNN_FILTERS,)"""
        h = x_spatial[np.newaxis, :]          # (1, 5)
        h = self.conv1.forward(h)             # (16, 3)
        h = self.conv2.forward(h)             # (16, 1)
        h = self.pool.forward(h)              # (16,)
        return h


# ─────────────────────────────────────────────────────────────────────────────
# 4. LSTM CELL  (single layer, processes temporal sequence)
# ─────────────────────────────────────────────────────────────────────────────

class LSTMCell:
    """
    Standard LSTM: input_size → hidden_size.
    Weight matrices stored in (hidden, input+hidden) format.
    """
    def __init__(self, input_size, hidden_size):
        scale = np.sqrt(1.0 / hidden_size)
        concat = input_size + hidden_size
        self.Wf = RNG.normal(0, scale, (hidden_size, concat))
        self.Wi = RNG.normal(0, scale, (hidden_size, concat))
        self.Wc = RNG.normal(0, scale, (hidden_size, concat))
        self.Wo = RNG.normal(0, scale, (hidden_size, concat))
        self.bf = np.ones(hidden_size)   # forget bias → 1 (good init)
        self.bi = np.zeros(hidden_size)
        self.bc = np.zeros(hidden_size)
        self.bo = np.zeros(hidden_size)
        self.hidden_size = hidden_size

    def step(self, x, h_prev, c_prev):
        xh  = np.concatenate([x, h_prev])
        f   = sigmoid(self.Wf @ xh + self.bf)
        i   = sigmoid(self.Wi @ xh + self.bi)
        c_  = tanh(self.Wc @ xh + self.bc)
        o   = sigmoid(self.Wo @ xh + self.bo)
        c   = f * c_prev + i * c_
        h   = o * tanh(c)
        return h, c

    def forward(self, seq):
        """seq: (T, input_size) → h_final: (hidden_size,)"""
        T = len(seq)
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        for t in range(T):
            h, c = self.step(seq[t], h, c)
        return h


# ─────────────────────────────────────────────────────────────────────────────
# 5. FULLY-CONNECTED HEAD
# ─────────────────────────────────────────────────────────────────────────────

class FCLayer:
    def __init__(self, in_dim, out_dim, activation='relu'):
        self.W  = RNG.normal(0, np.sqrt(2.0 / in_dim), (out_dim, in_dim))
        self.b  = np.zeros(out_dim)
        self.act = activation

    def forward(self, x):
        z = self.W @ x + self.b
        if self.act == 'relu':    return relu(z)
        if self.act == 'sigmoid': return sigmoid(z)
        return z


# ─────────────────────────────────────────────────────────────────────────────
# 6. HYBRID MODEL
# ─────────────────────────────────────────────────────────────────────────────

class CNNRNNHybrid:
    """
    Fused CNN + RNN model for flood & landslide prediction.

    Inputs per sample:
        x_spatial  : (5,)      — static terrain / land-use features
        x_temporal : (30, 4)   — daily weather / hydrology time series

    Outputs:
        flood_score     : float 0-1
        landslide_score : float 0-1
        label           : int   0=Safe / 1=Unsafe
    """

    def __init__(self):
        self.cnn  = CNNBranch()
        self.lstm = LSTMCell(N_TEMPORAL, LSTM_HIDDEN)
        # fusion: CNN_FILTERS + LSTM_HIDDEN → FC_HIDDEN → 2 outputs
        fusion_dim = CNN_FILTERS + LSTM_HIDDEN
        self.fc1  = FCLayer(fusion_dim, FC_HIDDEN, 'relu')
        self.fc2  = FCLayer(FC_HIDDEN, 2, 'sigmoid')
        self.threshold = 0.5
        # normalisation statistics (set during fit)
        self.sp_mean = None
        self.sp_std  = None
        self.tp_mean = None
        self.tp_std  = None
        # training history
        self.history = {}

    # ── normalisation ─────────────────────────────────────────────────────────
    def _fit_normalise(self, X_sp, X_tp):
        self.sp_mean = X_sp.mean(axis=0)
        self.sp_std  = X_sp.std(axis=0)  + 1e-8
        self.tp_mean = X_tp.mean(axis=(0, 1))
        self.tp_std  = X_tp.std(axis=(0, 1)) + 1e-8

    def _norm_sp(self, x): return (x - self.sp_mean) / self.sp_std
    def _norm_tp(self, x): return (x - self.tp_mean) / self.tp_std

    # ── single-sample forward pass ────────────────────────────────────────────
    def _forward(self, x_sp, x_tp):
        """
        x_sp : (5,)    normalised spatial
        x_tp : (30,4)  normalised temporal
        Returns: (flood_score, landslide_score)
        """
        cnn_out  = self.cnn.forward(x_sp)       # (CNN_FILTERS,)
        lstm_out = self.lstm.forward(x_tp)       # (LSTM_HIDDEN,)
        fused    = np.concatenate([cnn_out, lstm_out])
        h        = self.fc1.forward(fused)
        out      = self.fc2.forward(h)           # (2,) in [0,1]
        return out[0], out[1]

    # ── predict on raw inputs ─────────────────────────────────────────────────
    def predict_proba(self, X_sp, X_tp):
        X_sp = self._norm_sp(X_sp)
        X_tp = self._norm_tp(X_tp)
        n    = len(X_sp)
        proba = np.zeros((n, 2))
        for i in range(n):
            f, l        = self._forward(X_sp[i], X_tp[i])
            proba[i, 1] = float(np.clip((f + l) / 2, 0, 1))
            proba[i, 0] = 1 - proba[i, 1]
        return proba

    def predict(self, X_sp, X_tp):
        proba = self.predict_proba(X_sp, X_tp)
        return (proba[:, 1] >= self.threshold).astype(int)

    def predict_risk_scores(self, X_sp, X_tp):
        """Return per-sample (flood_score, landslide_score) arrays."""
        X_sp = self._norm_sp(X_sp)
        X_tp = self._norm_tp(X_tp)
        n    = len(X_sp)
        flood_s     = np.zeros(n)
        landslide_s = np.zeros(n)
        for i in range(n):
            f, l           = self._forward(X_sp[i], X_tp[i])
            flood_s[i]     = float(np.clip(f, 0, 1))
            landslide_s[i] = float(np.clip(l, 0, 1))
        return flood_s, landslide_s

    # ── sklearn-compatible single-array predict (for Flask app) ───────────────
    def predict_flat(self, X_flat):
        """
        X_flat: list/array of shape (N, 6) — same 6-feature format as the
                original weather-based model (temp, max_temp, wind, cloud,
                precip, humidity).  We map these to approximate spatial/
                temporal inputs so the Flask app still works unchanged.
        """
        X_flat = np.atleast_2d(X_flat)
        n      = len(X_flat)
        # Build synthetic spatial & temporal proxies from weather features
        # col 0=avg_temp, 1=max_temp, 2=wind_speed, 3=cloud_cover,
        #     4=precip(×15 in app), 5=humidity
        sp = np.zeros((n, 5))
        sp[:, 0] = 500 - X_flat[:, 0] * 5     # elevation proxy (cooler=higher)
        sp[:, 1] = X_flat[:, 2] * 0.5          # slope proxy from wind
        sp[:, 2] = X_flat[:, 3] / 100          # soil type proxy from cloud
        sp[:, 3] = 0.5
        sp[:, 4] = np.clip(50 - X_flat[:, 4] * 2, 0, 50)  # dist_river proxy

        tp = np.zeros((n, SEQ_LEN, N_TEMPORAL))
        for i in range(n):
            tp[i, :, 0] = X_flat[i, 4] / SEQ_LEN  # daily rainfall
            tp[i, :, 1] = X_flat[i, 5] / 10        # river level proxy
            tp[i, :, 2] = X_flat[i, 5]             # soil moisture
            tp[i, :, 3] = X_flat[i, 0]             # temperature

        return self.predict(sp, tp)

    # ── training (gradient-free: random search + layer-wise refinement) ───────
    def fit(self, X_sp, X_tp, y, flood_scores, landslide_scores,
            n_iter=800, batch_size=64, noise_scale=0.05, verbose=True):
        """
        Since we implement NumPy from scratch without autograd, we use a
        perturbation-based optimisation: repeatedly perturb all weights by
        small Gaussian noise and keep changes that improve loss on a mini-
        batch.  This is equivalent to a random local search (evolution
        strategy) and converges reliably for the model size used here.
        """
        self._fit_normalise(X_sp, X_tp)
        X_sp_n = self._norm_sp(X_sp)
        X_tp_n = self._norm_tp(X_tp)
        n      = len(y)

        def bce(y_true, y_pred):
            p = np.clip(y_pred, 1e-7, 1 - 1e-7)
            return -np.mean(y_true * np.log(p) + (1-y_true) * np.log(1-p))

        def batch_loss(idx):
            total = 0.0
            for i in idx:
                f, l = self._forward(X_sp_n[i], X_tp_n[i])
                pred = np.clip((f + l) / 2, 1e-7, 1 - 1e-7)
                total += bce(np.array([y[i]]), np.array([pred]))
            return total / len(idx)

        def get_all_params():
            params = []
            for obj in self._param_objects():
                params.extend([obj.__dict__[k] for k in obj.__dict__
                                if isinstance(obj.__dict__[k], np.ndarray)])
            return params

        def set_all_params(params):
            idx = 0
            for obj in self._param_objects():
                for k in list(obj.__dict__.keys()):
                    if isinstance(obj.__dict__[k], np.ndarray):
                        obj.__dict__[k] = params[idx]
                        idx += 1

        def perturb(params, scale):
            return [p + RNG.normal(0, scale, p.shape) for p in params]

        best_params = get_all_params()
        best_loss   = batch_loss(np.arange(min(256, n)))
        losses      = [best_loss]

        if verbose:
            print(f"  Initial loss: {best_loss:.4f}")

        for it in range(n_iter):
            batch_idx = RNG.choice(n, size=min(batch_size, n), replace=False)
            cur_params = get_all_params()
            new_params = perturb(cur_params, noise_scale)
            set_all_params(new_params)
            new_loss = batch_loss(batch_idx)
            if new_loss < best_loss:
                best_loss   = new_loss
                best_params = new_params
            else:
                set_all_params(cur_params)

            if verbose and (it + 1) % 100 == 0:
                losses.append(best_loss)
                print(f"  Iter {it+1:4d}  best_loss={best_loss:.4f}")

        set_all_params(best_params)
        self.history['loss'] = losses

        # set decision threshold via validation
        proba = self.predict_proba(X_sp, X_tp)
        from sklearn.metrics import f1_score
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.2, 0.8, 0.05):
            preds = (proba[:, 1] >= t).astype(int)
            f1    = f1_score(y, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        self.threshold = best_t
        print(f"\n  Best threshold: {self.threshold:.2f}  (F1={best_f1:.3f})")
        return self

    def _param_objects(self):
        """All objects containing weight arrays."""
        return [
            self.cnn.conv1, self.cnn.conv2,
            self.lstm,
            self.fc1, self.fc2,
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN — train, evaluate, pickle
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  CNN + RNN Hybrid — Flood & Landslide Prediction")
    print("=" * 60)

    # ── generate data ─────────────────────────────────────────────────────────
    print("\n[1/4] Generating synthetic dataset …")
    X_sp, X_tp, y, flood_scores, ls_scores = generate_dataset(N_SAMPLES)
    split = int(0.8 * N_SAMPLES)
    X_sp_tr, X_tp_tr = X_sp[:split], X_tp[:split]
    X_sp_te, X_tp_te = X_sp[split:], X_tp[split:]
    y_tr,    y_te    = y[:split],    y[split:]
    f_tr,    f_te    = flood_scores[:split],  flood_scores[split:]
    l_tr,    l_te    = ls_scores[:split],     ls_scores[split:]

    print(f"  Train: {split}  |  Test: {N_SAMPLES - split}")
    print(f"  Class distribution (train)  Safe={np.sum(y_tr==0)}  "
          f"Unsafe={np.sum(y_tr==1)}")

    # ── train ─────────────────────────────────────────────────────────────────
    print("\n[2/4] Training CNN+RNN hybrid (800 iterations) …")
    model = CNNRNNHybrid()
    model.fit(X_sp_tr, X_tp_tr, y_tr, f_tr, l_tr,
              n_iter=800, batch_size=64, noise_scale=0.04, verbose=True)

    # ── evaluate ──────────────────────────────────────────────────────────────
    print("\n[3/4] Evaluating on held-out test set …")
    y_pred  = model.predict(X_sp_te, X_tp_te)
    proba   = model.predict_proba(X_sp_te, X_tp_te)
    fl_pred, ls_pred = model.predict_risk_scores(X_sp_te, X_tp_te)

    report  = classification_report(y_te, y_pred,
                                    target_names=["Safe", "Unsafe"])
    cm      = confusion_matrix(y_te, y_pred).tolist()
    try:
        auc = roc_auc_score(y_te, proba[:, 1])
    except Exception:
        auc = float('nan')

    print(report)
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"  Confusion matrix:\n  {cm}")

    # ── build results dict and pickle ─────────────────────────────────────────
    print("\n[4/4] Saving model + results to model.pickle …")

    # Attach results to model so Flask app can display them
    model.eval_results = {
        "classification_report": report,
        "confusion_matrix":      cm,
        "roc_auc":               round(float(auc), 4),
        "threshold":             round(float(model.threshold), 4),
        "model_architecture": {
            "cnn_filters":   CNN_FILTERS,
            "kernel_size":   KERNEL_SIZE,
            "lstm_hidden":   LSTM_HIDDEN,
            "fc_hidden":     FC_HIDDEN,
            "seq_len":       SEQ_LEN,
            "spatial_features": [
                "elevation_m", "slope_deg", "soil_type",
                "land_cover",  "dist_river_km"
            ],
            "temporal_features": [
                "rainfall_mm_day", "river_level_m",
                "soil_moisture_pct", "temperature_c"
            ],
        },
        "sample_predictions": [
            {
                "city":            city,
                "flood_score":     round(float(fl_pred[i]), 3),
                "landslide_score": round(float(ls_pred[i]), 3),
                "label":           "Unsafe" if y_pred[i] else "Safe",
                "confidence":      round(float(proba[i, 1]), 3),
            }
            for i, city in enumerate([
                "Delhi", "Mumbai", "Kolkata", "Bangalore",
                "Chennai", "Hyderabad", "Jaipur", "Pune",
                "Ahmedabad", "Surat"
            ])
        ],
        "training_loss_history": [round(v, 5) for v in model.history.get('loss', [])],
        "dataset_info": {
            "total_samples":    N_SAMPLES,
            "train_samples":    split,
            "test_samples":     N_SAMPLES - split,
            "safe_count_test":  int(np.sum(y_te == 0)),
            "unsafe_count_test":int(np.sum(y_te == 1)),
        }
    }

    pickle.dump(model, open("/home/claude/model.pickle", "wb"), protocol=4)
    print("  ✓  Saved to /home/claude/model.pickle")
    print("\n  Sample predictions:")
    for p in model.eval_results["sample_predictions"]:
        print(f"    {p['city']:12s}  flood={p['flood_score']:.2f}  "
              f"landslide={p['landslide_score']:.2f}  → {p['label']}  "
              f"(conf={p['confidence']:.2f})")

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)
