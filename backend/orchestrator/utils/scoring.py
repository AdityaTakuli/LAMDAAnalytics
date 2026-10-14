import json
import math
import os
from typing import Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.settings import settings

DEFAULT_WEIGHTS = {
    "inventory_days": 0.20,
    "past_delay_days": 0.20,
    "news_vol_7d": 0.15,
    "neg_tone_frac_3d": 0.15,
    "strike_flag_7d": 0.15,
    "weather_anomaly_7d": 0.10,
    "global_risk": 0.05
}

class ScoringState:
    """
    Keeps rolling mean/var per feature to z-score then squashes to [0,1].
    Ensures run-to-run relative comparability.
    """
    def __init__(self, path: str):
        self.path = path
        self.state: Dict[str, Dict[str, float]] = {}  # {feature: {count, mean, M2}}

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                self.state = json.load(f)
        else:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self.state = {}
            self.save()

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.state, f)

    def update(self, features: Dict[str, float]):
        for k, v in features.items():
            if v is None:
                continue
            s = self.state.get(k, {"count": 0.0, "mean": 0.0, "M2": 0.0})
            c = s["count"] + 1.0
            delta = v - s["mean"]
            mean = s["mean"] + delta / c
            M2 = s["M2"] + delta * (v - mean)
            self.state[k] = {"count": c, "mean": mean, "M2": M2}
        self.save()

    def zscore(self, k: str, v: float) -> float:
        s = self.state.get(k)
        if not s or s["count"] < 2:
            return 0.5  # neutral until stats stabilize
        var = s["M2"] / (s["count"] - 1.0)
        std = math.sqrt(max(var, 1e-12))
        z = (v - s["mean"]) / std
        # squash to (0,1)
        return 1.0 / (1.0 + math.exp(-z))

scoring_state = ScoringState(settings.scoring_state_path)
scoring_state.load()

# More inventory cover lowers risk. The offline standardizer flips this feature
# (dataset/training/data.py PROTECTIVE_FEATURE); the live path now does the
# same. sigmoid(-z) == 1 - sigmoid(z), so orienting after the z-score is
# identical to flipping before it and keeps the stored Welford state valid.
PROTECTIVE_FEATURES = {"inventory_days"}


def normalize_features(raw: Dict[str, float]) -> Dict[str, float]:
    # Update rolling stats then produce normalized values in [0,1] via z-score sigmoid
    scoring_state.update(raw)
    out = {}
    for k, v in raw.items():
        if v is None:
            continue
        value = scoring_state.zscore(k, v)
        out[k] = 1.0 - value if k in PROTECTIVE_FEATURES else value
    return out

def weighted_risk(norm: Dict[str, float], weights: Dict[str, float] = None) -> tuple[float, Dict[str, float]]:
    if not weights:
        weights = DEFAULT_WEIGHTS
    total = sum(weights.values())
    score = 0.0
    contrib = {}
    for k, w in weights.items():
        val = norm.get(k, 0.5)
        score += w * val
        contrib[k] = w * val / total
    return score, contrib
