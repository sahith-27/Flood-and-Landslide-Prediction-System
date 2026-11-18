# Satellite CNN (2D) → LSTM → Random Forest

## Full Pipeline

```
Real Satellite Image (64×64 RGB)          per city, per month
        ↓
  CNN 2D  (im2col vectorised convolution)
  ┌─ Conv2D(3→8,  3×3) + ReLU + MaxPool2×2  → (31,31,8)
  ├─ Conv2D(8→16, 3×3) + ReLU + MaxPool2×2  → (14,14,16)
  └─ Conv2D(16→32,3×3) + ReLU + GlobalAvgPool → (32,)
        ↓  32-dim spatial embedding per image
  Sequence: [May_embed, June_embed, July_embed]
        ↓
  LSTM (hidden=64)
  Learns: how flood/inundation signature evolves through monsoon
        ↓  64-dim temporal context
  + 36 image statistics (RGB mean/std/max, NDWI, month deltas)
        ↓  100-dim fused feature vector
  ┌──────────────┬────────────┬──────────────┐
  combined_rf   flood_rf   landslide_rf     ← 500 trees each
  Safe/Unsafe   0.0–1.0     0.0–1.0
```

## Results
| Metric      | Value  |
|-------------|--------|
| Accuracy    | 95%    |
| Safe F1     | 0.88   |
| Unsafe F1   | 0.97   |
| ROC-AUC     | 0.9822 |

## Dataset
- 5 cities × 3 months = **15 real satellite images**
- Augmented to **1000 training samples** (crop, flip, brightness jitter)

## Usage
```python
import sys; sys.path.insert(0, 'Code/training')
from cnn_lstm_rf import SatelliteCNNLSTMRF, CNN2D, LSTMCell, load_image
import pickle

model = pickle.load(open('Code/model.pickle', 'rb'))

# Predict on real images
imgs   = [load_image('Mumbai', m) for m in ['May','June','July']]
sample = [{'imgs': imgs}]
label  = model.predict(sample)          # [0] or [1]
proba  = model.predict_proba(sample)    # flood + landslide probabilities
fl, ls = model.predict_risk_scores(sample)

# Flask app (6 weather features)
pred = model.predict_flat([[35, 42, 15, 80, 10, 85]])
```
