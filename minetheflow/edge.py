"""Edge layer: what runs on each logger / gateway, one sensor at a time.

Each EdgeNode sees ONLY its own signal. State is a handful of floats plus a
288-value daily profile (~1.2 kB as float32), so this fits on a
microcontroller. Parameters arrive from the cloud as an EdgeConfig.
"""
from dataclasses import dataclass, asdict
import json
import numpy as np

from .network import SAMPLES_PER_DAY

GOOD, MISSING, RANGE, SPIKE, FLATLINE = "good", "missing", "out_of_range", "spike", "flatline"


@dataclass
class EdgeConfig:
    sensor: str
    profile: list            # expected value per 5-min slot of the day (cloud-trained)
    lo: float                # physical/plausible range
    hi: float
    max_rate: float          # max plausible change per step, beyond the daily-pattern change
    sigma: float             # std of deviation from profile on clean data
    flat_n: int = 6          # 6 identical samples (30 min) = frozen
    alpha: float = 1 / 36    # EWMA for level offset (~3 h memory)
    cusum_k: float = 0.5     # CUSUM slack (in sigmas)
    cusum_h: float = 40.0    # CUSUM alarm threshold (in sigmas)

    def to_json(self):
        d = asdict(self)
        d["profile"] = [round(v, 3) for v in d["profile"]]
        return json.dumps(d)


def train_edge_config(sensor, series, train_mask, resolution=0.01):
    """Cloud-side: fit the tiny edge model from clean training data."""
    x = series[train_mask].to_numpy()
    tod = np.arange(len(series))[train_mask] % SAMPLES_PER_DAY
    prof = np.array([np.median(x[tod == k]) for k in range(SAMPLES_PER_DAY)])
    prof = np.convolve(np.r_[prof[-3:], prof, prof[:3]], np.ones(7) / 7, "valid")
    dev = x - prof[tod]
    diffs = np.diff(dev)          # change NOT explained by the daily pattern
    span = x.max() - x.min()
    return EdgeConfig(
        sensor=sensor, profile=prof.tolist(),
        lo=float(x.min() - 0.5 * span - 1), hi=float(x.max() + 0.5 * span + 1),
        max_rate=float(max(6 * np.std(diffs), 10 * resolution)),
        sigma=float(np.std(dev)),
    )


def tune_cusum(cfg, series, train_mask, margin=1.3):
    """Cloud-side: set the CUSUM alarm level from the clean training data, so the
    edge raises no alarms on normal behaviour (demand noise is autocorrelated)."""
    cfg.cusum_h = 1e9
    node = EdgeNode(cfg)
    peak = 0.0
    for k, v in enumerate(series[train_mask].to_numpy()):
        node.step(k, float(v))
        peak = max(peak, node.cpos, node.cneg)
    cfg.cusum_h = float(margin * peak)
    return cfg


class EdgeNode:
    """Streaming validator + reconstructor for one sensor."""

    def __init__(self, cfg: EdgeConfig):
        self.c = cfg
        self.prof = np.asarray(cfg.profile)
        self.offset = 0.0
        self.last_good = None
        self.last_raw = None
        self.run = 0
        self.cpos = self.cneg = 0.0
        self.alarm = False
        self.spike_buf = []      # consecutive "spike" samples (k, v)

    def step(self, k, v):
        """Process sample index k with raw value v -> (clean, flag, alert, retro_n)."""
        c = self.c
        exp = self.prof[k % SAMPLES_PER_DAY] + self.offset
        flag, retro, alert = GOOD, 0, None
        if v is None or np.isnan(v):
            flag = MISSING
        elif not (c.lo <= v <= c.hi):
            flag = RANGE
        else:
            self.run = self.run + 1 if (self.last_raw is not None and v == self.last_raw) else 1
            if self.run >= c.flat_n:
                flag = FLATLINE
                retro = c.flat_n - 1 if self.run == c.flat_n else 0
            elif (self.last_good is not None
                  and abs((v - self.last_good) - (self.prof[k % SAMPLES_PER_DAY] - self.prof[(k - 1) % SAMPLES_PER_DAY])) > c.max_rate
                  and abs(v - exp) > 4 * c.sigma):
                flag = SPIKE
        self.last_raw = v if flag != MISSING else self.last_raw
        if flag == SPIKE:
            self.spike_buf.append(v)
            # 3 "spikes" in a row at a consistent level is NOT a spike: it is a real
            # step (e.g. a burst or a valve). Never hide it; accept it and alert.
            b = self.spike_buf
            if len(b) >= 3 and max(b) - min(b) < c.max_rate:
                self.spike_buf = []
                self.last_good = v
                return v, GOOD, "step_change", -2   # -2: un-flag the 2 buffered samples
            return exp, flag, alert, retro
        self.spike_buf = []
        if flag != GOOD:
            return exp, flag, alert, retro
        # good sample: update level tracker and change detector
        d = v - self.prof[k % SAMPLES_PER_DAY]
        self.offset += c.alpha * (d - self.offset)
        z = d / c.sigma
        self.cpos = max(0.0, self.cpos + z - c.cusum_k)
        self.cneg = max(0.0, self.cneg - z - c.cusum_k)
        if not self.alarm and max(self.cpos, self.cneg) > c.cusum_h:
            self.alarm = True
            alert = "level_shift_up" if self.cpos > self.cneg else "level_shift_down"
        self.last_good = v
        return v, flag, alert, retro


def run_edge(cfg, raw):
    """Run one EdgeNode over a whole series (simulating real-time streaming)."""
    node = EdgeNode(cfg)
    n = len(raw)
    clean = np.empty(n)
    flags = np.empty(n, dtype=object)
    alerts = []
    for k, v in enumerate(raw):
        val, f, a, retro = node.step(k, float(v))
        clean[k], flags[k] = val, f
        if retro < 0:  # step accepted: restore buffered raw values
            for j in range(k + retro, k):
                flags[j], clean[j] = GOOD, raw[j]
        elif retro:  # frozen run discovered: re-flag buffered samples, reconstruct them
            for j in range(k - retro, k):
                flags[j] = FLATLINE
                clean[j] = node.prof[j % SAMPLES_PER_DAY] + node.offset
        if a:
            alerts.append((k, a))
    return clean, flags, alerts


def baseline_reconstructions(raw, bad):
    """Comparison methods for gap filling: last value hold, linear interpolation."""
    import pandas as pd
    s = pd.Series(np.where(bad, np.nan, raw))
    return {"hold_last": s.ffill().to_numpy(), "linear_interp": s.interpolate().to_numpy()}
