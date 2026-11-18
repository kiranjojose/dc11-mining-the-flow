"""Generate 'real' sensor data: stochastic demands, a leak, sensor faults.

Ground truth is kept for every injected event so detection and
reconstruction can be scored (we know the answer, unlike with field data).
"""
import numpy as np
import pandas as pd
import wntr

from .network import (build_dma, diurnal_pattern, STEP, DAYS, SAMPLES_PER_DAY,
                      PRESSURE_SENSORS, FLOW_SENSOR, SENSORS)

T0 = pd.Timestamp("2026-03-02 00:00")           # a Monday
TRAIN_DAYS = 6                                    # days 0-5 clean -> calibration
LEAK_NODE = "J52"
LEAK_LPS = 2.0                                    # ~10% of mean DMA inflow
LEAK_START = T0 + pd.Timedelta(days=8, hours=3)   # day 8, 03:00


def _idx(ts):
    return int((ts - T0) / pd.Timedelta(seconds=STEP))


def time_index():
    return pd.date_range(T0, periods=DAYS * SAMPLES_PER_DAY + 1, freq=f"{STEP}s")


def simulate_truth(seed=1, leak=True, leak_node=None, leak_lps=None):
    """Run EPANET with per-customer stochastic demand (+ optional leak)."""
    rng = np.random.default_rng(seed)
    wn = build_dma()
    nominal = diurnal_pattern()
    n = len(nominal)
    day_factor = np.repeat(1 + rng.normal(0, 0.03, DAYS), SAMPLES_PER_DAY)  # weather etc.
    for name, j in wn.junctions():
        # AR(1) multiplicative noise per node: customers are not the textbook curve
        e = np.zeros(n)
        for k in range(1, n):
            e[k] = 0.97 * e[k - 1] + rng.normal(0, 0.04)
        pat = np.clip(nominal * day_factor * (1 + e), 0.05, None)
        wn.add_pattern(f"pat_{name}", pat.tolist())
        j.demand_timeseries_list[0].pattern_name = f"pat_{name}"
    if leak:
        on = np.zeros(n)
        on[_idx(LEAK_START):] = 1.0
        wn.add_pattern("leak", on.tolist())
        wn.get_node(leak_node or LEAK_NODE).add_demand((leak_lps or LEAK_LPS) / 1000, "leak")
    res = wntr.sim.EpanetSimulator(wn).run_sim()
    df = pd.DataFrame(index=time_index())
    df["Q_in"] = res.link["flowrate"]["INLET"].values * 1000          # L/s
    for s, node in PRESSURE_SENSORS.items():
        df[s] = res.node["pressure"][node].values                     # m
    return df, wn


def add_measurement_noise(true, seed=2):
    rng = np.random.default_rng(seed)
    meas = true.copy()
    meas["Q_in"] = (meas["Q_in"] * (1 + rng.normal(0, 0.01, len(meas)))).round(2)
    for s in PRESSURE_SENSORS:
        meas[s] = (meas[s] + rng.normal(0, 0.03, len(meas))).round(2)  # 0.01 m resolution
    return meas


def inject_faults(meas, seed=3):
    """Inject sensor faults on the test days. Returns faulty data + labels."""
    rng = np.random.default_rng(seed)
    x = meas.copy()
    lab = pd.DataFrame("ok", index=x.index, columns=SENSORS)
    ti = x.index

    def win(d, h0, h1):
        return (ti >= T0 + pd.Timedelta(days=d, hours=h0)) & (ti < T0 + pd.Timedelta(days=d, hours=h1))

    # 1) P2: missing block, logger battery/sensor dropout (8 h)
    m = win(6, 6, 14)
    x.loc[m, "P2"] = np.nan
    lab.loc[m, "P2"] = "missing"
    # 2) P4: spikes / outliers (EMI, air in the line)
    test = np.where(ti >= T0 + pd.Timedelta(days=TRAIN_DAYS))[0]
    k = rng.choice(test, 12, replace=False)
    x.iloc[k, x.columns.get_loc("P4")] += rng.choice([-1, 1], 12) * rng.uniform(4, 12, 12)
    lab.iloc[k, lab.columns.get_loc("P4")] = "spike"
    # 3) P5: frozen / flatline sensor (18 h)
    m = win(7, 2, 20)
    x.loc[m, "P5"] = x.loc[m, "P5"].iloc[0]
    lab.loc[m, "P5"] = "flatline"
    # 4) P7: slow drift (calibration loss), -0.4 m/day from day 6 to end
    m = ti >= T0 + pd.Timedelta(days=6)
    days = (ti[m] - (T0 + pd.Timedelta(days=6))) / pd.Timedelta(days=1)
    x.loc[m, "P7"] = (x.loc[m, "P7"] - 0.4 * days).round(2)
    lab.loc[m, "P7"] = "drift"
    return x, lab


def comm_outage_mask(index):
    """Gateway backhaul down 6 h on day 8, 00:00-06:00, overlapping the leak onset (03:00). Loggers keep measuring."""
    return (index >= T0 + pd.Timedelta(days=8, hours=0)) & (index < T0 + pd.Timedelta(days=8, hours=6))


def leak_label(index):
    return index >= LEAK_START
