"""Mining the Flow - end-to-end demo.

    python run_demo.py            # ~10 s; writes figures/ and results/

Pipeline (mirrors the three DC11 objectives):
  simulate DMA (EPANET via WNTR) -> inject faults + leak -> EDGE validation &
  reconstruction per sensor -> MQTT/SensorThings transport -> CLOUD model-based
  diagnosis (sensor fault vs leak) + leak localisation -> compare edge-only,
  cloud-only and hybrid.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minetheflow.network import PRESSURE_SENSORS, FLOW_SENSOR, SENSORS, SAMPLES_PER_DAY, build_dma
from minetheflow.simulate import (T0, TRAIN_DAYS, LEAK_NODE, LEAK_LPS, LEAK_START, simulate_truth,
                                  add_measurement_noise, inject_faults, comm_outage_mask)
from minetheflow.edge import train_edge_config, tune_cusum, run_edge, baseline_reconstructions, GOOD
from minetheflow.cloud import twin_prediction, leak_sensitivity, Calibration, CloudAnalyser
from minetheflow.iot import Broker, topic, observation, sensorthings_entities
from minetheflow import plots

HERE = os.path.dirname(os.path.abspath(__file__))
FIG, RES = os.path.join(HERE, "figures"), os.path.join(HERE, "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)
H = pd.Timedelta(hours=1)


def hours(td):
    return round(td / H, 2)


# ---------------------------------------------------------------- 1. data
truth, wn = simulate_truth()
meas = add_measurement_noise(truth)
raw, labels = inject_faults(meas)
idx = raw.index
train = idx < T0 + pd.Timedelta(days=TRAIN_DAYS)
outage = comm_outage_mask(idx)
print(f"Simulated {len(idx)} samples x {len(SENSORS)} sensors; leak {LEAK_LPS} L/s at {LEAK_NODE} from {LEAK_START}")

# ---------------------------------------------------------------- 2. cloud trains edge models
configs = {s: tune_cusum(train_edge_config(s, raw[s], train), raw[s], train) for s in SENSORS}
cfg_bytes = {s: len(c.to_json()) for s, c in configs.items()}
with open(os.path.join(RES, "edge_config_P5.json"), "w") as f:
    f.write(configs["P5"].to_json())

# ---------------------------------------------------------------- 3. edge layer
edge_clean, edge_flags, edge_alerts = {}, {}, {}
for s in SENSORS:
    c, fl, al = run_edge(configs[s], raw[s].to_numpy())
    edge_clean[s], edge_flags[s], edge_alerts[s] = c, fl, al
edge_clean = pd.DataFrame(edge_clean, index=idx)
edge_flags = pd.DataFrame(edge_flags, index=idx)

# reconstruction quality on samples where data was missing / frozen
recon = []
for s, kind in [("P2", "missing"), ("P5", "flatline")]:
    m = (labels[s] == kind).to_numpy()
    base = baseline_reconstructions(raw[s].to_numpy(), m)
    for name, est in [("edge profile+level (causal)", edge_clean[s].to_numpy()),
                      ("hold last value (causal)", base["hold_last"]),
                      ("linear interpolation (needs future data)", base["linear_interp"])]:
        err = est[m] - truth[s].to_numpy()[m]
        recon.append(dict(sensor=s, fault=kind, method=name, MAE_m=round(float(np.mean(np.abs(err))), 3),
                          RMSE_m=round(float(np.sqrt(np.mean(err ** 2))), 3)))
recon = pd.DataFrame(recon)

# edge detection quality per fault type (sample level)
flagged = edge_flags != GOOD
edge_det = {}
for kind, s in [("missing", "P2"), ("spike", "P4"), ("flatline", "P5"), ("drift", "P7")]:
    m = labels[s] == kind
    edge_det[kind] = dict(sensor=s, n=int(m.sum()), flagged=int((flagged[s] & m).sum()))
false_flags = int((flagged & (labels == "ok")).to_numpy().sum())

# ---------------------------------------------------------------- 4. transport (MQTT + SensorThings)
ds_id = {s: i for i, s in enumerate(SENSORS, start=1)}

# 4a) CLOUD-ONLY: dumb loggers stream every raw 5-min sample; no local storage
b_cloud = Broker()
rx_cloud = pd.DataFrame(np.nan, index=idx, columns=SENSORS)
b_cloud.subscribe("v1.1/", lambda t, p: None)
for k, ts in enumerate(idx):
    for s in SENSORS:
        v = raw[s].iat[k]
        b_cloud.publish(topic(ds_id[s]), observation(ts, v), link_up=not outage[k])
        if not outage[k]:
            rx_cloud.iat[k, ds_id[s] - 1] = v

# 4b) HYBRID: edge validates, keeps raw locally, sends 15-min quality-tagged
#     aggregates + immediate alerts; store-and-forward when backhaul is down
b_hyb = Broker()
store = []                                # cloud-side received aggregates
buffers = {s: [] for s in SENSORS}
b_hyb.subscribe("v1.1/", lambda t, p: store.append((t, p)))
alert_log = []
per_block = 3                             # 3 x 5 min = 15 min
alert_at = {s: {k: a for k, a in edge_alerts[s]} for s in SENSORS}
for k0 in range(0, len(idx), per_block):
    blk = slice(k0, min(k0 + per_block, len(idx)))
    ts_end = idx[blk][-1]
    up = not outage[blk.stop - 1]
    for s in SENSORS:
        f = edge_flags[s].iloc[blk]
        good = (f == GOOD).to_numpy()
        vals = raw[s].iloc[blk].to_numpy()
        mean_good = float(np.mean(vals[good])) if good.any() else float("nan")
        q = "good" if good.all() else ("partial" if good.any() else "reconstructed")
        obs = observation(idx[k0], mean_good if good.any() else float(np.mean(edge_clean[s].iloc[blk])), q,
                          {"n_good": int(good.sum()), "flags": sorted(set(f) - {GOOD}),
                           "min": round(float(np.nanmin(edge_clean[s].iloc[blk])), 2),
                           "max": round(float(np.nanmax(edge_clean[s].iloc[blk])), 2)})
        obs["_sensor"], obs["_sent"] = s, ts_end
        if up:
            b_hyb.flush(buffers[s])
        b_hyb.publish(topic(ds_id[s]), obs, link_up=up, buffer=buffers[s])
        for k in range(blk.start, blk.stop):
            if k in alert_at[s]:
                alert_log.append(dict(sensor=s, time=idx[k], alert=alert_at[s][k]))
                b_hyb.publish(f"v1.1/Datastreams({ds_id[s]})/Alerts", {"t": str(idx[k]), "a": alert_at[s][k]},
                              link_up=up, buffer=buffers[s])
for s in SENSORS:
    b_hyb.flush(buffers[s])

# reconnection time for each sample (when the cloud actually gets it)
reconnect = idx[outage][-1] + pd.Timedelta(minutes=5) if outage.any() else None

# ---------------------------------------------------------------- 5. cloud layer
twin = twin_prediction(idx)
S, nodes = leak_sensitivity()
coords = {n: wn.get_node(n).coordinates for n in wn.node_name_list}

def hourly(df):
    return df.iloc[:-1].resample("1h").mean()

twin_h = hourly(twin)
# cloud-only: raw stream, hourly means of whatever arrived (spikes included)
res_cloud = hourly(rx_cloud) - twin_h
# hybrid: hourly means from good 15-min aggregates only
agg = pd.DataFrame([dict(t=pd.Timestamp(p["phenomenonTime"][:-1]), s=p["_sensor"], v=p["result"], n=p["parameters"]["n_good"])
                    for t, p in store if "_sensor" in p])
agg["vn"] = agg.v.fillna(0) * agg.n
g = agg.set_index("t").groupby([pd.Grouper(freq="1h"), "s"])[["vn", "n"]].sum()
hyb_h = (g.vn / g.n).where(g.n >= 6).unstack("s")[SENSORS]
res_hyb = hyb_h.reindex(twin_h.index) - twin_h

cal = Calibration((hourly(raw) - twin_h)[twin_h.index < T0 + pd.Timedelta(days=TRAIN_DAYS)])
an = CloudAnalyser(cal, S, nodes)

out_h = (twin_h.index >= idx[outage][0]) & (twin_h.index < reconnect) if reconnect is not None else np.zeros(len(twin_h), bool)
known_cloud = twin_h.index + H
known_hyb = pd.DatetimeIndex(np.where(out_h, reconnect, twin_h.index + H))

results = {}
for name, res, kn in [("cloud_only", res_cloud, known_cloud), ("hybrid", res_hyb, known_hyb)]:
    dec, qev, alarm, Z, quar = an.run(res, kn)
    loc = an.localise(Z, alarm, quar)
    results[name] = dict(dec=dec, qev=qev, alarm=alarm, Z=Z, loc=loc, quar=quar)
    dec.to_csv(os.path.join(RES, f"cloud_decisions_{name}.csv"))

# ---------------------------------------------------------------- 6. evaluation
def dist(a, b):
    (x1, y1), (x2, y2) = coords[a], coords[b]
    return float(np.hypot(x1 - x2, y1 - y2))

def fault_hours(dec, s, t0, t1):
    d = dec[(dec.index >= t0) & (dec.index < t1)]
    hit = d[d.fault_sensor == s]
    return hit

ev = {"P2_gap": (T0 + pd.Timedelta(days=6, hours=6), T0 + pd.Timedelta(days=6, hours=14)),
      "P5_flatline": (T0 + pd.Timedelta(days=7, hours=2), T0 + pd.Timedelta(days=7, hours=20)),
      "P7_drift": (T0 + pd.Timedelta(days=6), idx[-1])}
spike_hours = sorted(set(idx[labels.P4 == "spike"].floor("h")))

rows = []
def first_flag(s, kind):
    m = (labels[s] == kind) & flagged[s]
    return idx[m][0] if m.any() else None

# --- edge-only
leak_alerts = [a for a in alert_log if a["time"] >= LEAK_START]
leak_alert_sensors = sorted({a["sensor"] for a in leak_alerts if a["time"] < LEAK_START + pd.Timedelta(hours=24)})
drift_alert = [a for a in alert_log if a["sensor"] == "P7" and a["time"] < LEAK_START]
p2_mae = recon.query('sensor=="P2" and method.str.startswith("edge")').MAE_m.iat[0]
edge = {
    "P2 data gap (8 h)": f"detected at once; filled (MAE {p2_mae} m)",
    "P4 spikes (12)": f"{edge_det['spike']['flagged']}/12 removed",
    "P5 frozen sensor (18 h)": f"detected after {hours(first_flag('P5','flatline') - ev['P5_flatline'][0] + pd.Timedelta(minutes=5)) if first_flag('P5','flatline') else '-'} h (6 equal samples); filled",
    "P7 slow drift": (f"'level shift' alert after {hours(drift_alert[0]['time'] - ev['P7_drift'][0])} h, cause unknown" if drift_alert else "missed"),
    "Leak 2 L/s": f"'level shift' on {len(leak_alert_sensors)}/10 sensors within 24 h; cannot tell leak from drift, no location",
}
# --- cloud-only and hybrid
def cloud_summary(r, name):
    dec, alarm, loc = r["dec"], r["alarm"], r["loc"]
    out = {}
    gap = dec[(dec.index >= ev["P2_gap"][0]) & (dec.index < ev["P2_gap"][1])]
    out["P2 data gap (8 h)"] = ("not received; hours analysed without P2" if name == "cloud_only"
                                else "edge-filled values marked 'reconstructed'; cloud ignores them")
    sh = dec.loc[dec.index.isin(spike_hours)]
    out["P4 spikes (12)"] = (f"{int((sh.fault_sensor == 'P4').sum())}/{len(spike_hours)} spike-hours flagged as P4 fault"
                             if name == "cloud_only" else "already removed at edge; 0 spurious hours")
    fh = fault_hours(dec, "P5", *ev["P5_flatline"])
    if name == "cloud_only" and "P5" in r["quar"]:
        out["P5 frozen sensor (18 h)"] = (f"diagnosed P5 fault after {hours(fh.index[0] + H - ev['P5_flatline'][0])} h, "
                                          "then P5 stays quarantined (cloud cannot see it recovered)")
    else:
        out["P5 frozen sensor (18 h)"] = (f"diagnosed P5 fault after {hours(fh.index[0] + H - ev['P5_flatline'][0])} h" if len(fh)
                                          else ("edge flag used directly; P5 excluded" if name == "hybrid" else "missed"))
    q7 = [q for q in r["qev"] if q[0] == "P7"]
    out["P7 slow drift"] = (f"P7 quarantined after {hours(q7[0][2] - ev['P7_drift'][0])} h (single-sensor, physically inconsistent)" if q7 else "missed")
    if alarm:
        loc_s = (f"; localised at {loc['ranking'][0][0]} ({dist(loc['ranking'][0][0], LEAK_NODE):.0f} m from true {LEAK_NODE}, "
                 f"true node rank {[n for n, _ in loc['ranking']].index(LEAK_NODE) + 1}/{len(nodes)}) at {loc['known_at']:%a %H:%M}, est. {loc['size_lps']:.1f} L/s"
                 if loc else "; not localised")
        out["Leak 2 L/s"] = f"LEAK alarm known after {hours(alarm[1] - LEAK_START)} h" + loc_s
    else:
        out["Leak 2 L/s"] = "missed"
    pre = dec[dec.index < LEAK_START]
    fa_leak = int(pre.leak_node.notna().sum())
    wrong_fault = 0
    for t, row in dec.iterrows():
        s = row.fault_sensor
        if isinstance(s, str) and not ((s == "P2" and ev["P2_gap"][0] <= t < ev["P2_gap"][1]) or (s == "P4" and t in spike_hours)
                      or (s == "P5" and ev["P5_flatline"][0] <= t < ev["P5_flatline"][1]) or (s == "P7" and t >= ev["P7_drift"][0])):
            wrong_fault += 1
    n_alarm = 1 if (alarm and alarm[0] < LEAK_START) else 0
    out["False alarms (hours)"] = (f"{n_alarm} leak alarms before the leak ({fa_leak} isolated leak-hour, filtered by the 2-hour rule); "
                                   f"{wrong_fault} hours blaming a healthy sensor")
    return out

summ = {"Edge only": edge, "Cloud only": cloud_summary(results["cloud_only"], "cloud_only"),
        "Hybrid": cloud_summary(results["hybrid"], "hybrid")}
summ["Edge only"]["False alarms (hours)"] = f"{false_flags} wrongly flagged sample(s) (of {flagged.size})"
lost = int(outage.sum()) * len(SENSORS)
summ["Edge only"]["Data sent / lost in 6 h outage"] = "alerts only (no network view)"
summ["Cloud only"]["Data sent / lost in 6 h outage"] = f"{b_cloud.bytes/1e6:.2f} MB, {b_cloud.messages} msgs / {lost} samples lost"
summ["Hybrid"]["Data sent / lost in 6 h outage"] = f"{b_hyb.bytes/1e6:.2f} MB, {b_hyb.messages} msgs / 0 lost (store-and-forward)"
table = pd.DataFrame(summ)
table.to_csv(os.path.join(RES, "comparison.csv"))

with open(os.path.join(RES, "summary.md"), "w") as f:
    f.write("# Results (auto-generated by run_demo.py)\n\n## Edge vs cloud vs hybrid\n\n")
    f.write(table.to_markdown() + "\n\n## Reconstruction error (vs noise-free truth)\n\n")
    f.write(recon.to_markdown(index=False) + "\n\n")
    for name in ["cloud_only", "hybrid"]:
        loc = results[name]["loc"]
        if loc:
            f.write(f"## Leak localisation, {name}\n\nNight window {loc['window'][0]} to {loc['window'][1]}, sensors used: {', '.join(loc['used_sensors'])}\n\n")
            f.write("| rank | node | correlation | distance to true leak (m) |\n|---|---|---|---|\n")
            for i, (n, c) in enumerate(loc["ranking"][:5], 1):
                f.write(f"| {i} | {n} | {c:.3f} | {dist(n, LEAK_NODE):.0f} |\n")
            f.write("\n")
    f.write(f"## Edge model size\n\nEdge config (288-slot profile + thresholds) per sensor: {np.mean(list(cfg_bytes.values())):.0f} bytes as JSON.\n\n")
    f.write("## Edge alerts\n\n" + pd.DataFrame(alert_log).to_markdown(index=False) + "\n")

# SensorThings example payloads
st = {"Things": sensorthings_entities({s: coords[n] for s, n in {**PRESSURE_SENSORS, "Q_in": "J00"}.items()}),
      "example_hybrid_observations": [dict((k, v) for k, v in p.items() if not k.startswith("_"))
                                      for t, p in store if p.get("_sensor") == "P5"][679:684]}
with open(os.path.join(RES, "sensorthings_example.json"), "w") as f:
    json.dump(st, f, indent=2, default=str)

# ---------------------------------------------------------------- 7. figures
plots.network_map(wn, coords, results["hybrid"]["loc"], LEAK_NODE, os.path.join(FIG, "1_network_leak_localisation.png"))
plots.edge_signals(idx, raw, edge_clean, edge_flags, truth, labels, os.path.join(FIG, "2_edge_validation_reconstruction.png"))
plots.cloud_residuals(results["hybrid"], results["cloud_only"], LEAK_START, T0 + pd.Timedelta(days=TRAIN_DAYS),
                      os.path.join(FIG, "3_cloud_fault_vs_leak.png"))
plots.comparison_table(table, os.path.join(FIG, "4_edge_cloud_hybrid_comparison.png"))

print(table.to_string())
print(recon.to_string())
for name in results:
    l = results[name]["loc"]
    print(name, "alarm", results[name]["alarm"], "loc", l and l["ranking"][:3], l and l["known_at"])
