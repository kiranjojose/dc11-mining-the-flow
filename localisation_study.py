"""How good is leak localisation in general (not just one lucky case)?

For leaks at many nodes and two sizes, with fresh random demand noise each
time and clean sensors, run the cloud localisation on the first night after
the leak and record how far the top candidate is from the true leak.

    python localisation_study.py      # ~1 min
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minetheflow.simulate import simulate_truth, add_measurement_noise, T0, TRAIN_DAYS
from minetheflow.cloud import twin_prediction, leak_sensitivity, Calibration, CloudAnalyser

S, nodes = leak_sensitivity()
truth0, wn = simulate_truth(seed=100, leak=False)
idx = truth0.index
twin_h = twin_prediction(idx).iloc[:-1].resample("1h").mean()
coords = {n: wn.get_node(n).coordinates for n in nodes}
rng = np.random.default_rng(0)
cases = rng.choice(nodes, 14, replace=False)
LEAK_T = T0 + pd.Timedelta(days=8, hours=3)
rows = []
for size in [1.0, 2.0]:
    for k, node in enumerate(cases):
        tr, _ = simulate_truth(seed=200 + k, leak_node=node, leak_lps=size)
        meas = add_measurement_noise(tr, seed=300 + k)
        res = meas.iloc[:-1].resample("1h").mean() - twin_h
        cal = Calibration(res[res.index < T0 + pd.Timedelta(days=TRAIN_DAYS)])
        an = CloudAnalyser(cal, S, nodes)
        loc = an.localise(cal.whiten(res), (LEAK_T, LEAK_T), {})
        best = loc["ranking"][0][0]
        rank = [n for n, _ in loc["ranking"]].index(node) + 1
        d = float(np.hypot(*(np.subtract(coords[best], coords[node]))))
        rows.append(dict(leak_lps=size, true=node, estimated=best, rank=rank, error_m=d))
df = pd.DataFrame(rows)
summary = df.groupby("leak_lps").agg(cases=("rank", "size"), top1=("rank", lambda r: (r == 1).mean()),
                                    top5=("rank", lambda r: (r <= 5).mean()),
                                    median_error_m=("error_m", "median"), max_error_m=("error_m", "max"))
print(df.to_string())
print(summary.round(2).to_string())
os.makedirs("results", exist_ok=True)
with open("results/localisation_study.md", "w") as f:
    f.write("# Leak localisation over many leak positions (clean sensors, first night after onset)\n\n")
    f.write("Grid spacing is 150 m, so an error of 150 m means 'the neighbouring junction'.\n\n")
    f.write(summary.round(2).to_markdown() + "\n\n" + df.to_markdown(index=False) + "\n")
