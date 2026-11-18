"""IoT flavour: OGC SensorThings data model + an MQTT-like broker.

The broker is in-process (no Mosquitto needed for the demo) but uses the
SensorThings MQTT topic convention `v1.1/Datastreams(<id>)/Observations` and
counts bytes, so bandwidth of the edge-vs-cloud designs can be compared.
"""
import json
from collections import defaultdict

from .network import PRESSURE_SENSORS, FLOW_SENSOR, SENSORS

MQTT_OVERHEAD = 2 + 2 + 40   # fixed header + topic length + ~topic bytes


def sensorthings_entities(coords):
    """Thing / Location / Sensor / ObservedProperty / Datastream for each logger."""
    props = {
        "pressure": {"@iot.id": 1, "name": "Pressure head", "definition": "http://vocab.nerc.ac.uk/collection/P07/current/CFSN0330/", "description": "Gauge pressure head"},
        "flow": {"@iot.id": 2, "name": "Volumetric flow rate", "definition": "https://qudt.org/vocab/quantitykind/VolumeFlowRate", "description": "DMA inlet flow"},
    }
    things = []
    for i, s in enumerate(SENSORS, start=1):
        is_flow = s in FLOW_SENSOR
        where = "INLET main" if is_flow else f"junction {PRESSURE_SENSORS[s]}"
        x, y = coords[s]
        things.append({
            "@iot.id": i,
            "name": f"Logger {s}",
            "description": f"Battery IoT logger at {where} (synthetic DMA)",
            "properties": {"dma": "DMA-01", "edge_firmware": "minetheflow-edge 0.1"},
            "Locations": [{"name": where, "encodingType": "application/geo+json",
                           "location": {"type": "Point", "coordinates": [x, y]}}],
            "Datastreams": [{
                "@iot.id": i,
                "name": f"{s} {'flow' if is_flow else 'pressure'}",
                "unitOfMeasurement": ({"name": "litre per second", "symbol": "L/s", "definition": "http://qudt.org/vocab/unit/L-PER-SEC"} if is_flow
                                      else {"name": "metre (water column)", "symbol": "m", "definition": "http://qudt.org/vocab/unit/M"}),
                "observationType": "http://www.opengis.net/def/observationType/OGC-OM/2.0/OM_Measurement",
                "Sensor": {"name": "Electromagnetic flow meter" if is_flow else "Piezoresistive pressure transducer",
                           "encodingType": "application/pdf", "metadata": "datasheet.pdf"},
                "ObservedProperty": props["flow" if is_flow else "pressure"],
            }],
        })
    return things


def observation(ts, value, quality=None, params=None):
    o = {"phenomenonTime": ts.isoformat() + "Z", "result": None if value != value else round(float(value), 3)}
    if quality is not None:
        o["resultQuality"] = quality
    if params:
        o["parameters"] = params
    return o


class Broker:
    """Minimal pub/sub with per-topic byte accounting and offline buffering."""

    def __init__(self):
        self.bytes = 0
        self.messages = 0
        self.subs = defaultdict(list)
        self.dropped = 0

    def subscribe(self, prefix, fn):
        self.subs[prefix].append(fn)

    def publish(self, topic, payload, link_up=True, buffer=None):
        """If the link is down: buffer (store-and-forward) or drop."""
        if not link_up:
            if buffer is not None:
                buffer.append((topic, payload))
            else:
                self.dropped += 1
            return
        wire = {k: v for k, v in payload.items() if not str(k).startswith("_")}  # "_" = demo bookkeeping, not sent
        body = json.dumps(wire, separators=(",", ":"))
        self.bytes += len(body) + MQTT_OVERHEAD
        self.messages += 1
        for prefix, fns in self.subs.items():
            if topic.startswith(prefix):
                for fn in fns:
                    fn(topic, payload)

    def flush(self, buffer):
        while buffer:
            self.publish(*buffer.pop(0))


def topic(datastream_id):
    return f"v1.1/Datastreams({datastream_id})/Observations"
