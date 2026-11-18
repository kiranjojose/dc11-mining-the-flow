"""Cloud layer: whole-network, model-based reasoning.

1. Digital twin: the EPANET model with nominal demands (no leak).
2. Residuals r = measured - twin, calibrated on clean training days
   (bias and noise level per sensor and hour of day).
3. Leak sensitivity matrix S (the Pérez et al. idea): the residual each
   sensor would see for a 1 L/s leak at each candidate node.
4. Hourly hypothesis test on the whitened residual vector:
     H0 nothing | F_i sensor i faulty | L_j leak at node j | L_j + F_i
   Score = squared residual left unexplained + penalty per extra cause.
   A sensor fault explains ONE sensor; a leak must explain all of them
   at once with one physically consistent signature.
"""
import numpy as np
import pandas as pd
import wntr

from .network import build_dma, PRESSURE_SENSORS, SENSORS, STEP


def twin_prediction(index):
    wn = build_dma()
    res = wntr.sim.EpanetSimulator(wn).run_sim()
    df = pd.DataFrame(index=index)
    df["Q_in"] = res.link["flowrate"]["INLET"].values * 1000
    for s, node in PRESSURE_SENSORS.items():
        df[s] = res.node["pressure"][node].values
    return df


def leak_sensitivity(leak_lps=1.0):
    """S[hour, sensor, node]: hourly mean residual for a 1 L/s leak at each node."""
    wn = build_dma()
    wn.options.time.duration = 86400
    base = wntr.sim.EpanetSimulator(wn).run_sim()
    nodes = wn.junction_name_list
    S = np.zeros((24, len(SENSORS), len(nodes)))
    wn.add_pattern("unit", [1.0])
    for j, n in enumerate(nodes):
        wn2 = build_dma()
        wn2.options.time.duration = 86400
        wn2.add_pattern("unit", [1.0])
        wn2.get_node(n).add_demand(leak_lps / 1000, "unit")
        r = wntr.sim.EpanetSimulator(wn2).run_sim()
        d = pd.DataFrame(index=range(len(r.node["pressure"])))
        d["Q_in"] = (r.link["flowrate"]["INLET"].values - base.link["flowrate"]["INLET"].values) * 1000
        for s, node in PRESSURE_SENSORS.items():
            d[s] = r.node["pressure"][node].values - base.node["pressure"][node].values
        d = d.iloc[:-1]
        d["h"] = (d.index * STEP // 3600) % 24
        S[:, :, j] = d.groupby("h")[SENSORS].mean().to_numpy()
    return S, nodes


class Calibration:
    """Per sensor and hour of day: residual bias and spread on clean data."""

    def __init__(self, hourly_resid_train):
        h = hourly_resid_train.index.hour
        g = hourly_resid_train.groupby(h)
        self.bias = g.mean()
        sd = g.std()
        # pool neighbouring hours (few training days) and floor the spread
        v = sd.reindex(range(24)).to_numpy()
        pooled = (np.roll(v, 1, 0) + v + np.roll(v, -1, 0)) / 3
        pooled = np.maximum(pooled, 0.5 * np.median(pooled, axis=0))
        self.sigma = pd.DataFrame(pooled, index=range(24), columns=sd.columns)

    def whiten(self, hourly_resid):
        h = hourly_resid.index.hour
        return (hourly_resid - self.bias.loc[h].to_numpy()) / self.sigma.loc[h].to_numpy()


def diagnose_hour(z, Sw, penalty=30.0, min_leak_lps=0.3):
    """z: whitened residuals (NaN = no data); Sw: whitened signatures [sensor, node].
    Returns (label, target, leak_size, score_table)."""
    ok = ~np.isnan(z)
    z, Sw = z[ok], Sw[ok]
    names = np.array(SENSORS)[ok]
    if len(z) < 4:
        return "no_data", None, 0.0
    best = ("none", None, 0.0, float(z @ z))

    def leak_fit(zz, SS):
        a = (SS.T @ zz) / np.einsum("ij,ij->j", SS, SS)
        a = np.clip(a, 0, None)
        sse = ((zz[:, None] - SS * a) ** 2).sum(0)
        j = int(np.argmin(sse))
        return j, a[j], sse[j]

    cands = [best]
    for i in range(len(z)):                       # single sensor fault
        cands.append(("sensor_fault", names[i], 0.0, float(z @ z - z[i] ** 2) + penalty))
    j, a, sse = leak_fit(z, Sw)                   # leak alone
    cands.append(("leak", j, a, float(sse) + penalty))
    for i in range(len(z)):                       # leak + one faulty sensor
        m = np.arange(len(z)) != i
        j2, a2, sse2 = leak_fit(z[m], Sw[m])
        cands.append(("leak+sensor_fault", (j2, names[i]), a2, float(sse2) + 2 * penalty))
    lab, tgt, size, _ = min(cands, key=lambda c: c[3])
    if lab.startswith("leak") and size < min_leak_lps:
        lab, tgt = ("sensor_fault", tgt[1]) if lab == "leak+sensor_fault" else ("none", None)
    return lab, tgt, size


def leak_ranking(z_mean, Sw):
    """Correlation between the mean residual pattern and each node's signature."""
    ok = ~np.isnan(z_mean)
    zz, SS = z_mean[ok], Sw[ok]
    zc = zz - zz.mean()
    Sc = SS - SS.mean(0)
    return (Sc.T @ zc) / (np.linalg.norm(Sc, axis=0) * np.linalg.norm(zc) + 1e-12)


class CloudAnalyser:
    """Hour-by-hour supervision of the DMA with simple persistence logic.

    * a sensor diagnosed faulty 3 hours in a row is QUARANTINED (excluded from
      leak analysis and a maintenance ticket is sent back to the edge);
    * a leak diagnosed 2 hours in a row raises a LEAK ALARM;
    * after the alarm, night hours (00-06, minimum night flow: lowest demand
      uncertainty) are averaged to LOCALISE the leak with the sensitivity matrix.
    """

    def __init__(self, cal, S, nodes, penalty=30.0, quarantine_after=3, leak_after=2):
        self.cal, self.S, self.nodes = cal, S, nodes
        self.penalty, self.qa, self.la = penalty, quarantine_after, leak_after

    def run(self, hourly_resid, known_at=None):
        Z = self.cal.whiten(hourly_resid)
        known_at = known_at if known_at is not None else Z.index + pd.Timedelta(hours=1)
        quarantined, q_events = {}, []
        streak = {}
        leak_streak, leak_alarm = 0, None
        rows = []
        for (ts, row), kn in zip(Z.iterrows(), known_at):
            z = row.copy()
            for s in quarantined:
                z[s] = np.nan
            h = ts.hour
            Sw = self.S[h] / self.cal.sigma.loc[h].to_numpy()[:, None]
            lab, tgt, a = diagnose_hour(z.to_numpy(), Sw, self.penalty)
            fault = tgt if lab == "sensor_fault" else (tgt[1] if lab == "leak+sensor_fault" else None)
            node = self.nodes[tgt] if lab == "leak" else (self.nodes[tgt[0]] if lab == "leak+sensor_fault" else None)
            for s in list(streak):
                if s != fault:
                    streak[s] = 0
            if fault:
                streak[fault] = streak.get(fault, 0) + 1
                if streak[fault] >= self.qa and fault not in quarantined:
                    quarantined[fault] = ts
                    q_events.append((fault, ts, kn))
            leak_streak = leak_streak + 1 if node else 0
            if leak_streak >= self.la and leak_alarm is None:
                leak_alarm = (ts - pd.Timedelta(hours=self.la - 1), kn)
            rows.append(dict(time=ts, known_at=kn, label=lab, fault_sensor=fault,
                             leak_node=node, leak_lps=round(float(a), 2),
                             quarantined=",".join(sorted(quarantined))))
        dec = pd.DataFrame(rows).set_index("time")
        return dec, q_events, leak_alarm, Z, quarantined

    def localise(self, Z, leak_alarm, quarantined, min_hours=3):
        """Average night-time residuals after the alarm and rank candidate nodes."""
        if leak_alarm is None:
            return None
        start = leak_alarm[0]
        Zq = Z.copy()
        for s, t in quarantined.items():
            Zq.loc[Zq.index >= t - pd.Timedelta(hours=self.qa), s] = np.nan
        night = Zq[(Zq.index >= start) & (Zq.index.hour < 6)].dropna(how="all")
        night = night[night.notna().sum(1) >= 4]
        if night.empty:
            return None
        for day, grp in night.groupby(night.index.date):
            if len(grp) >= min_hours:
                break
        else:
            return None
        hrs = grp.index.hour
        sig = self.cal.sigma.loc[hrs].to_numpy().mean(0)
        Sw = self.S[hrs].mean(0) / sig[:, None]
        zm = grp.mean().to_numpy()
        corr = leak_ranking(zm, Sw)
        ok = ~np.isnan(zm)
        a = (Sw[ok].T @ zm[ok]) / np.einsum("ij,ij->j", Sw[ok], Sw[ok])
        order = np.argsort(-corr)
        return dict(window=(grp.index[0], grp.index[-1] + pd.Timedelta(hours=1)),
                    known_at=grp.index[-1] + pd.Timedelta(hours=1),
                    ranking=[(self.nodes[j], float(corr[j])) for j in order],
                    corr=dict(zip(self.nodes, corr.tolist())),
                    size_lps=float(a[order[0]]),
                    used_sensors=[s for s, o in zip(SENSORS, ok) if o])
