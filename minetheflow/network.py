"""Build a synthetic District Metered Area (DMA) in WNTR/EPANET.

One reservoir feeds the DMA through a single inlet main (where the DMA flow
meter sits), like the DMAs used in UPC's leak-localisation work. The grid is
irregular (some pipes removed), with mixed diameters and a terrain slope.
"""
import numpy as np
import networkx as nx
import wntr

STEP = 300            # 5-minute hydraulic / sampling step [s]
DAYS = 10
SAMPLES_PER_DAY = 86400 // STEP

# Typical residential diurnal curve (hourly multipliers, mean ~1)
_HOURLY = np.array([0.45, 0.38, 0.34, 0.33, 0.36, 0.50, 0.95, 1.45, 1.55, 1.35,
                    1.20, 1.15, 1.20, 1.15, 1.05, 1.00, 1.05, 1.20, 1.40, 1.50,
                    1.35, 1.10, 0.80, 0.58])


def diurnal_pattern(days=DAYS):
    """Nominal pattern at 5-min resolution (smooth interpolation of hourly values)."""
    h = np.arange(0, 24, STEP / 3600)
    one_day = np.interp(h, np.arange(25), np.r_[_HOURLY, _HOURLY[0]])
    return np.tile(one_day / one_day.mean(), days)


def build_dma(seed=7, nx_=7, ny_=6, spacing=150.0):
    rng = np.random.default_rng(seed)
    wn = wntr.network.WaterNetworkModel()
    t = wn.options.time
    t.duration = DAYS * 86400
    t.hydraulic_timestep = t.pattern_timestep = t.report_timestep = STEP
    t.quality_timestep = STEP
    wn.options.hydraulic.headloss = "H-W"

    wn.add_pattern("nominal", diurnal_pattern().tolist())
    for i in range(nx_):
        for j in range(ny_):
            x, y = i * spacing, j * spacing
            elev = 20 + 0.025 * x + 0.02 * y + rng.normal(0, 1.5)
            base = rng.uniform(0.25, 0.75) / 1000.0  # m3/s
            wn.add_junction(f"J{i}{j}", base_demand=base, demand_pattern="nominal",
                            elevation=float(elev), coordinates=(x, y))

    edges = []
    for i in range(nx_):
        for j in range(ny_):
            if i + 1 < nx_:
                edges.append((f"J{i}{j}", f"J{i+1}{j}", j == 0))
            if j + 1 < ny_:
                edges.append((f"J{i}{j}", f"J{i}{j+1}", i == 0))
    # remove ~15% of distribution pipes while keeping the graph connected
    g = nx.Graph([(a, b) for a, b, _ in edges])
    order = rng.permutation(len(edges))
    removed = set()
    for k in order[: int(0.3 * len(edges))]:
        a, b, main = edges[k]
        if main or len(removed) >= int(0.15 * len(edges)):
            continue
        g.remove_edge(a, b)
        if nx.is_connected(g):
            removed.add(k)
        else:
            g.add_edge(a, b)
    for k, (a, b, main) in enumerate(edges):
        if k in removed:
            continue
        d = 0.15 if main else float(rng.choice([0.06, 0.08, 0.08, 0.10]))
        wn.add_pipe(f"P_{a}_{b}", a, b, length=spacing, diameter=d,
                    roughness=float(rng.uniform(95, 125)))

    wn.add_reservoir("R", base_head=92.0, coordinates=(-400, 0))
    wn.add_pipe("INLET", "R", "J00", length=800, diameter=0.25, roughness=110)
    return wn


# Sensor layout: inlet flow meter + 9 pressure loggers spread over the DMA
PRESSURE_SENSORS = {
    "P1": "J00", "P2": "J30", "P3": "J60", "P4": "J02", "P5": "J33",
    "P6": "J63", "P7": "J05", "P8": "J35", "P9": "J65",
}
FLOW_SENSOR = {"Q_in": "INLET"}
SENSORS = list(FLOW_SENSOR) + list(PRESSURE_SENSORS)
