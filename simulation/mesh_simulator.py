"""
Virtual ESP32 mesh node simulator for local development and testing.
Simulates N nodes publishing MQTT sensor data without real hardware.
SDKs: paho-mqtt, NumPy, threading
"""
import time
import json
import threading
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import paho.mqtt.client as mqtt


@dataclass
class VirtualNode:
    node_id: str
    anomaly_rate: float = 0.1    # Probability of sending anomaly reading
    sample_rate_hz: int = 10     # Simulated publish rate
    topic_prefix: str = "sensor"
    rms_normal: float = 0.3
    rms_anomaly: float = 2.5
    noise_std: float = 0.05
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    def generate_reading(self) -> Dict[str, Any]:
        is_anomaly = self.rng.random() < self.anomaly_rate
        base_rms = self.rms_anomaly if is_anomaly else self.rms_normal
        rms = float(base_rms + self.rng.normal(0, self.noise_std))
        confidence = float(self.rng.uniform(0.75, 0.99))
        return {
            "device": self.node_id,
            "ts": time.time(),
            "label": "anomaly" if is_anomaly else "normal",
            "confidence": confidence,
            "rms": max(0.0, rms),
            "temp_c": float(self.rng.uniform(20, 45)),
            "battery_pct": float(self.rng.uniform(60, 100)),
        }

    @property
    def topic(self) -> str:
        return f"{self.topic_prefix}/{self.node_id}/vibration"


class MeshSimulator:
    """
    Simulate a fleet of ESP32 nodes publishing to MQTT.
    Use for integration testing without physical hardware.
    """

    def __init__(
        self,
        n_nodes: int = 10,
        broker: str = "localhost",
        port: int = 1883,
        topic_prefix: str = "sensor",
        anomaly_rate: float = 0.1,
        seed: int = 42,
    ):
        self.n_nodes = n_nodes
        self.broker = broker
        self.port = port
        self.rng = np.random.default_rng(seed)

        self.nodes = [
            VirtualNode(
                node_id=f"esp32_node_{i:03d}",
                anomaly_rate=anomaly_rate,
                topic_prefix=topic_prefix,
                rng=np.random.default_rng(seed + i),
            )
            for i in range(n_nodes)
        ]

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = lambda c, u, f, rc, p: print(f"[Sim] MQTT connected (rc={rc})")
        self._running = False
        self._threads: List[threading.Thread] = []
        self.stats = {"published": 0, "anomalies": 0, "errors": 0}

    def _node_loop(self, node: VirtualNode, steps: Optional[int] = None):
        """Publish loop for a single virtual node."""
        interval = 1.0 / node.sample_rate_hz
        count = 0
        while self._running:
            if steps and count >= steps:
                break
            try:
                reading = node.generate_reading()
                payload = json.dumps(reading).encode()
                self.client.publish(node.topic, payload, qos=0)
                self.stats["published"] += 1
                if reading["label"] == "anomaly":
                    self.stats["anomalies"] += 1
            except Exception as e:
                self.stats["errors"] += 1
            time.sleep(interval)
            count += 1

    def run(self, steps: Optional[int] = None, duration_sec: Optional[float] = None):
        """
        Start all virtual nodes publishing to MQTT.
        steps: publish N readings per node then stop (None = run forever)
        duration_sec: stop after N seconds (None = run until stop() called)
        """
        try:
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()
        except Exception as e:
            print(f"[Sim] MQTT connection failed: {e}. Running in offline mode.")

        self._running = True
        self._threads = []
        print(f"[Sim] Starting {self.n_nodes} virtual nodes on {self.broker}:{self.port}")

        for node in self.nodes:
            t = threading.Thread(target=self._node_loop, args=(node, steps), daemon=True)
            t.start()
            self._threads.append(t)

        try:
            if duration_sec:
                time.sleep(duration_sec)
                self.stop()
            elif steps:
                for t in self._threads:
                    t.join()
                self.stop()
            else:
                while self._running:
                    time.sleep(1)
                    total = self.stats["published"]
                    anomalies = self.stats["anomalies"]
                    rate = total / max(1, time.time() - self._start_time)
                    print(f"[Sim] {total} msgs ({anomalies} anomalies) | {rate:.1f} msg/s")
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self._running = False
        self.client.loop_stop()
        self.client.disconnect()
        print(f"[Sim] Stopped. Stats: {self.stats}")
        return self.stats
